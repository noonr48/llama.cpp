from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger

from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin


_PLE_Q8_CHUNK_ROWS = 16_384
_PLE_WRITE_CHUNK_BYTES = 8 * 1024 * 1024


def _require_range_eviction() -> None:
    if not hasattr(mmap, "MADV_DONTNEED") or not hasattr(os, "posix_fadvise") or \
            not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("Qwen4Exp PLE streaming requires MADV_DONTNEED and "
                           "POSIX_FADV_DONTNEED on this platform")


def _aligned_file_range(total: int, offset: int, length: int) -> tuple[int, int]:
    start = offset - (offset % mmap.PAGESIZE)
    end = min(total, ((offset + length + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE)
    return start, end - start


def _flush_drop_range(mm: np.memmap, fd: int, offset: int, length: int) -> None:
    _require_range_eviction()
    start, size = _aligned_file_range(mm.nbytes, offset, length)
    mm._mmap.flush(start, size)
    mm._mmap.madvise(mmap.MADV_DONTNEED, start, size)
    os.posix_fadvise(fd, start, size, os.POSIX_FADV_DONTNEED)


def _drop_read_range(mm: np.memmap, fd: int, offset: int, length: int) -> None:
    _require_range_eviction()
    start, size = _aligned_file_range(mm.nbytes, offset, length)
    mm._mmap.madvise(mmap.MADV_DONTNEED, start, size)
    os.posix_fadvise(fd, start, size, os.POSIX_FADV_DONTNEED)


class _EvictingMemmap(np.memmap):
    """Binary ``tofile`` that releases each source range after the write copies it."""

    def __new__(cls, filename, *, dtype, mode, shape,
                write_chunk: int = _PLE_WRITE_CHUNK_BYTES):
        obj = super().__new__(cls, filename, dtype=dtype, mode=mode, shape=shape)
        obj._evict_filename = os.fspath(filename)
        obj._evict_write_chunk = write_chunk
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._evict_filename = getattr(obj, "_evict_filename", None)
        self._evict_write_chunk = getattr(obj, "_evict_write_chunk", _PLE_WRITE_CHUNK_BYTES)

    def tofile(self, fid, sep="", format="%s"):
        if sep:
            return super().tofile(fid, sep=sep, format=format)

        own_target = isinstance(fid, (str, bytes, os.PathLike))
        target = open(fid, "wb", buffering=0) if own_target else fid
        source_fd = os.open(self._evict_filename, os.O_RDONLY)
        flat = self.view(np.ndarray).view(np.uint8).reshape(-1)
        try:
            for offset in range(0, flat.nbytes, self._evict_write_chunk):
                end = min(flat.nbytes, offset + self._evict_write_chunk)
                view = memoryview(flat[offset:end])
                written = 0
                while written < len(view):
                    count = target.write(view[written:])
                    if count is None:
                        count = len(view) - written
                    if count <= 0:
                        raise OSError(f"binary tofile made no progress at byte {offset + written}")
                    written += count
                _drop_read_range(self, source_fd, offset, end - offset)
        finally:
            os.close(source_fd)
            if own_target:
                target.close()


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen4-Exp")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # n-gram embedding tables arrive as many row shards, each placed as it streams in straight
    # into a per-layer np.memmap at its final row offset, so the joined table never has to exist
    # in RAM. Q8 output is then quantized into a second memmap in bounded row chunks. State is
    # per-bid so two PLE layers cannot mix; all named temp files are unlinked in write().
    _ngram_streams: dict[int, dict] | None = None

    # write() is one-shot per instance: a second call, even after a failed first call, must
    # fail loudly instead of re-entering the writer or re-cleaning consumed streams
    _ngram_write_done: bool = False

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_hyper_connection_count(self.hparams["hc_count"])

        # the MTP block consumes the target's hyper-connection streams, not the collapsed hidden
        # state, so both files declare the wider row: the target for the nextn read-back and the
        # draft for common/speculative.cpp. Same as DeepSeek-V4, see conversion/deepseek.py.
        self.gguf_writer.add_embedding_length_out(
            self.hparams["hc_count"]*self.hparams["hidden_size"])
        self.gguf_writer.add_hyper_connection_lowrank(self.hparams["hc_lowrank"])

        # the sigmoid output gate of the linear attention layers is hardcoded in the graph
        gate = self.hparams.get("output_gate_type") or self.hparams.get("hidden_act")
        if gate != "sigmoid":
            raise ValueError(f"unsupported output_gate_type {gate!r} (only 'sigmoid' is supported)")

        # the MTP block has no PLE layer (the reference clears ple_layer_ids for it)
        if not self.mtp_only and self.hparams.get("ple_layer_ids"):
            if len(self.hparams["ple_layer_ids"]) != 1:
                raise ValueError("only a single PLE layer is supported")
            self.gguf_writer.add_ple_embedding_length(self.hparams.get("ple_embed_dim") or self.hparams["hidden_size"])
            self.gguf_writer.add_ple_conv_kernel(self.hparams["ple_conv_kernel_size"])

            # the n-gram hash constants are needed on the host, keep them as metadata. They are read
            # here and not in modify_tensors because that already casts int64 buffers to float32.
            self.gguf_writer.add_ple_ngram_multipliers(self._read_u64(".ple_embedding.layer_multipliers"))
            self.gguf_writer.add_ple_ngram_vocab_sizes(self._read_u64(".ple_embedding.ngram_heads_vocab_sizes"))
            self.gguf_writer.add_ple_ngram_offsets(self._read_u64(".ple_embedding.ngram_heads_offsets"))

        # export the layer types instead of letting llama.cpp infer them from
        # full_attention_interval, so an irregular layout cannot misalign silently
        if layer_types := self.hparams.get("layer_types"):
            n_layer = self.hparams["num_hidden_layers"]
            if len(layer_types) != n_layer:
                raise ValueError(f"layer_types has {len(layer_types)} entries, "
                                 f"expected num_hidden_layers ({n_layer})")

            recurrent = []
            for t in layer_types:
                if t == "linear_attention":
                    recurrent.append(True)
                elif t == "full_attention" or t.endswith("sparse_attention"):
                    recurrent.append(False)
                else:
                    raise ValueError(f"unsupported qwen4exp layer type {t!r}")

            # llama.cpp reads this with get_key_or_arr(.., n_layer_all), which wants exactly
            # block_count entries. The MTP block is sparse attention, so it is not recurrent
            recurrent += [False] * (self.block_count - n_layer)
            self.gguf_writer.add_recurrent_layers(recurrent)

        if self.hparams.get("indexer_n_heads") is not None:
            self.gguf_writer.add_indexer_head_count(self.hparams["indexer_n_heads"])
            self.gguf_writer.add_indexer_key_length(self.hparams["indexer_head_dim"])
            self.gguf_writer.add_indexer_top_k     (self.hparams["indexer_budget"])
            self.gguf_writer.add_indexer_block_size(self.hparams["indexer_compress_ratio"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        # the reference adds the two MTP input projections, and A*e + B*h == [A|B]*concat(e, h),
        # so they join into the single eh_proj the tensor map already knows.
        # ref: conversion/deepseek.py, which joins the DeepSeek-V4 pair the same way
        e_name = "mtp.fc_embedding.weight"
        h_name = "mtp.fc_hidden.weight"

        have_e = e_name in self.model_tensors
        have_h = h_name in self.model_tensors
        if not have_e and not have_h:
            return
        if not have_e or not have_h:
            raise KeyError(f"unpaired MTP input projection: need both {e_name} and {h_name}")

        e = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
        h = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                                       self.hparams["num_hidden_layers"]),
               torch.cat([e, h], dim=1).contiguous())

        del self.model_tensors[e_name]
        del self.model_tensors[h_name]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the graph slices the PLE conv weight and multiplies it, which needs F32
        if ".ple_conv1d.weight" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]

        # the vision tower goes to the mmproj file
        if name.startswith("model.visual."):
            return None

        # the MTP block brings its own hyper-connection mixer, which takes the model-level slot of
        # an MTP-only file. Rename it here, before _QwenMtpMixin drops it as a non-MTP tensor.
        if name.startswith("mtp.hyper_connection_mixer."):
            return None if cls.no_mtp else (name.replace("mtp.", "model.", 1), item[1])

        return super().filter_tensors(item)

    def _read_u64(self, suffix: str) -> list[int]:
        name = next((n for n in self.model_tensors if n.endswith(suffix)), None)
        if name is None:
            raise ValueError(f"missing tensor *{suffix}, needed for the PLE n-gram hash")
        return [int(v) for v in LazyTorchTensor.to_eager(self.model_tensors[name]()).tolist()]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # hash constants, already exported as metadata by set_gguf_parameters
        if ".ple_embedding." in name and not name.endswith(".weight"):
            return

        if ".ple_embedding.ngram_embedding.shard_" in name:
            shard = int(name.rpartition(".ple_embedding.ngram_embedding.shard_")[2].partition(".")[0])
            n_shard = self.hparams["split_ngram_parts"]

            assert bid is not None

            if self._ngram_streams is None:
                self._ngram_streams = {}
            stream = self._ngram_streams.setdefault(bid, {
                "path": None, "mm": None, "qpath": None, "qmm": None,
                "uniform": None, "dtype": None, "width": None,
                "placed": {}, "deferred": None,
            })

            joined = self._ngram_stream_join(bid, stream, shard, n_shard, data_torch)
            if joined is None:
                return

            tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.PLE_NGRAM_EMBD, bid)
            if getattr(self, "ftype", None) == gguf.LlamaFileType.MOSTLY_Q8_0:
                self._ngram_add_q8_tensor(tensor_name, stream, joined.shape[0], bid)
                return

            yield (tensor_name, joined)
            return

        # the QSA indexer packs its query and key projections together
        if name.endswith(".indexer.index_qk_proj.weight"):
            n_q = self.hparams["indexer_n_heads"]*self.hparams["indexer_head_dim"]
            assert bid is not None
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid), data_torch[:n_q])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid), data_torch[n_q:])
            return

        # PLE norms are also 1-centered, the base class only catches names ending in "norm.weight"
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)

    def _ngram_tmp_path(self, bid: int) -> Path:
        # the joined table can be far larger than RAM, so the memmap lives next to the output
        # file (same filesystem as the GGUF, never a tmpfs-backed /tmp) under a deterministic
        # per-output/per-bid name: a hard reboot cannot leave uniquely named giant orphans, and
        # a retry reopens the same stale path, which np.memmap mode "w+" truncates on creation.
        # Concurrent conversions writing the same output are unsupported: the final GGUF path
        # itself is not concurrency-safe, so the deterministic name intentionally supports
        # crash retry, never concurrent writers.
        fname_out = Path(self.fname_out)
        return fname_out.with_name(f".{fname_out.stem}.ple-{bid}.tmp")

    def _ngram_q8_tmp_path(self, bid: int) -> Path:
        fname_out = Path(self.fname_out)
        return fname_out.with_name(f".{fname_out.stem}.ple-{bid}.q8.tmp")

    def _ngram_add_q8_tensor(self, tensor_name: str, stream: dict, rows: int, bid: int) -> None:
        source = stream["mm"][:rows]
        qtype = gguf.GGMLQuantizationType.Q8_0
        qshape = gguf.quant_shape_to_byte_shape(source.shape, qtype)
        stream["qpath"] = self._ngram_q8_tmp_path(bid)
        stream["qmm"] = _EvictingMemmap(stream["qpath"], mode="w+", dtype=np.uint8,
                                           shape=qshape)

        source_fd = os.open(stream["path"], os.O_RDONLY)
        output_fd = os.open(stream["qpath"], os.O_RDWR)
        try:
            n_chunks = (rows + _PLE_Q8_CHUNK_ROWS - 1) // _PLE_Q8_CHUNK_ROWS
            for chunk, start in enumerate(range(0, rows, _PLE_Q8_CHUNK_ROWS), start=1):
                end = min(rows, start + _PLE_Q8_CHUNK_ROWS)
                quantized = gguf.quants.quantize(source[start:end], qtype)
                if quantized.shape != (end - start, qshape[1]):
                    raise ValueError(f"chunked PLE Q8 shape {quantized.shape}, "
                                     f"expected {(end - start, qshape[1])}")
                stream["qmm"][start:end] = quantized

                input_offset = start * source.shape[1] * source.dtype.itemsize
                input_length = (end - start) * source.shape[1] * source.dtype.itemsize
                output_offset = start * qshape[1]
                output_length = (end - start) * qshape[1]
                _flush_drop_range(stream["qmm"], output_fd, output_offset, output_length)
                _drop_read_range(stream["mm"], source_fd, input_offset, input_length)
                del quantized
                if chunk == 1 or chunk % 1024 == 0 or chunk == n_chunks:
                    logger.info(f"prequantizing {tensor_name} Q8_0 chunk "
                                f"{chunk}/{n_chunks} ({end}/{rows} rows)")
        finally:
            os.close(source_fd)
            os.close(output_fd)

        logger.info(f"prequantized {tensor_name} to Q8_0 in bounded row chunks "
                    f"({rows} rows x {source.shape[1]})")
        self.gguf_writer.add_tensor(tensor_name, stream["qmm"], raw_dtype=qtype)

    def _ngram_stream_place(self, bid: int, stream: dict, shard: int, n_shard: int,
                            offset: int, np_data) -> None:
        # every shard is written exactly once, at its final row offset, into the memmap
        if stream["mm"] is None:
            stream["path"] = self._ngram_tmp_path(bid)
            stream["mm"] = np.memmap(stream["path"], mode="w+", dtype=stream["dtype"],
                                     shape=(n_shard * stream["uniform"], stream["width"]))
        stream["mm"][offset:offset + np_data.shape[0]] = np_data
        byte_offset = offset * stream["width"] * stream["dtype"].itemsize
        byte_length = np_data.shape[0] * stream["width"] * stream["dtype"].itemsize
        fd = os.open(stream["path"], os.O_RDWR)
        try:
            _flush_drop_range(stream["mm"], fd, byte_offset, byte_length)
        finally:
            os.close(fd)
        stream["placed"][shard] = (offset, np_data.shape[0])

    def _ngram_stream_join(self, bid: int, stream: dict, shard: int, n_shard: int,
                           data_torch: Tensor) -> Tensor | None:
        # the shards are joined in shard order, so the numbering has to be 0-based, in range,
        # unique and dense
        if not 0 <= shard < n_shard:
            raise ValueError(f"layer {bid}: n-gram shard index {shard} out of range 0..{n_shard - 1}")
        if shard in stream["placed"] or \
                (stream["deferred"] is not None and stream["deferred"][0] == shard):
            raise ValueError(f"layer {bid}: duplicate n-gram shard {shard}")

        data = LazyTorchTensor.to_eager(data_torch)
        if data.dtype == torch.bfloat16:
            # numpy has no bfloat16; prepare_tensors normally casts BF16 to F32 before this
            data = data.to(torch.float32)
        rows, width = data.shape
        if rows <= 0 or width <= 0:
            raise ValueError(f"layer {bid}: n-gram shard {shard} has {rows} rows x {width} cols; "
                             f"rows and width must be positive")
        np_data = data.numpy()
        if stream["dtype"] is None:
            stream["dtype"] = np_data.dtype
        elif np_data.dtype != stream["dtype"]:
            raise ValueError(f"layer {bid}: n-gram shard {shard} has dtype {np_data.dtype}, "
                             f"expected {stream['dtype']}")
        if stream["width"] is None:
            stream["width"] = width
        elif width != stream["width"]:
            raise ValueError(f"layer {bid}: n-gram shard {shard} has width {width}, "
                             f"expected {stream['width']}")

        uniform = stream["uniform"]
        if shard < n_shard - 1:
            # every non-final shard fixes the uniform shard height (or has to match it)
            if uniform is None:
                uniform = stream["uniform"] = rows
            elif rows != uniform:
                raise ValueError(f"layer {bid}: non-final n-gram shard {shard} has {rows} rows, "
                                 f"expected the uniform {uniform}")
            self._ngram_stream_place(bid, stream, shard, n_shard, shard * uniform, np_data)
        elif uniform is not None:
            if rows > uniform:
                raise ValueError(f"layer {bid}: final n-gram shard {shard} has {rows} rows, "
                                 f"exceeds the uniform {uniform}")
            self._ngram_stream_place(bid, stream, shard, n_shard, (n_shard - 1) * uniform, np_data)
        elif n_shard == 1:
            stream["uniform"] = uniform = rows
            self._ngram_stream_place(bid, stream, shard, n_shard, 0, np_data)
        else:
            # a short final shard that arrives before the uniform height is known cannot be
            # placed yet; it is the smallest shard, so holding just this one stays bounded
            stream["deferred"] = (shard, np_data)
            return None

        deferred = stream["deferred"]
        if uniform is not None and deferred is not None:
            d_shard, d_data = deferred
            if d_data.shape[0] > uniform:
                raise ValueError(f"layer {bid}: final n-gram shard {d_shard} has "
                                 f"{d_data.shape[0]} rows, exceeds the uniform {uniform}")
            stream["deferred"] = None
            self._ngram_stream_place(bid, stream, d_shard, n_shard,
                                     (n_shard - 1) * uniform, d_data)

        placed = stream["placed"]
        if len(placed) < n_shard:
            return None

        # the shards are joined in shard order, so the numbering has to be 0-based and dense
        if sorted(placed) != list(range(n_shard)):
            raise ValueError(f"layer {bid}: expected n-gram shards 0..{n_shard - 1}, "
                             f"got {sorted(placed)}")

        rows_joined = (n_shard - 1) * uniform + placed[n_shard - 1][1]

        # llama.cpp indexes the joined table as offs[h] + 0..vocab[h]-1, so a table that is
        # too short has to fail here and not hours later as a shape mismatch
        offs = self._read_u64(".ple_embedding.ngram_heads_offsets")
        vocab = self._read_u64(".ple_embedding.ngram_heads_vocab_sizes")
        # zip would silently pair only the shorter prefix, and max over an empty pairing would
        # raise its own error, so both conditions are rejected here with a name
        if len(offs) != len(vocab):
            raise ValueError(f"layer {bid}: n-gram hash metadata mismatch: "
                             f"{len(offs)} offsets vs {len(vocab)} vocab sizes")
        if not offs or not vocab:
            raise ValueError(f"layer {bid}: empty n-gram hash metadata: "
                             f"offsets and vocab sizes must be non-empty")
        need = max(o + v for o, v in zip(offs, vocab))
        if rows_joined < need:
            raise ValueError(f"joined n-gram table has {rows_joined} rows, "
                             f"the hash offsets need {need}")

        logger.info(f"joining {n_shard} n-gram embedding shards of layer {bid} "
                    f"({rows_joined} rows)")
        # For non-Q8 output this view follows the normal converter path. The Q8 caller consumes
        # the same view in bounded chunks and registers a prequantized memmap directly.
        return torch.from_numpy(stream["mm"][:rows_joined])

    def write(self):
        if self._ngram_write_done:
            raise RuntimeError("write() has already been called on this model instance; "
                               "writing twice is not supported")
        self._ngram_write_done = True
        try:
            super().write()
        finally:
            # Both joined-F32 and prequantized-Q8 tables are memmap-backed; whether preparation,
            # writing, or cleanup itself follows a successful or failed writer path, close and
            # unlink every deterministic temp artifact.
            if self._ngram_streams:
                for stream in self._ngram_streams.values():
                    for key in ("qmm", "mm"):
                        mmap_handle = getattr(stream.get(key), "_mmap", None)
                        if mmap_handle is not None:
                            mmap_handle.close()
                    for key in ("qpath", "path"):
                        path = stream.get(key)
                        if path is not None:
                            path.unlink(missing_ok=True)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._ngram_streams:
            incomplete = {}
            for bid, stream in self._ngram_streams.items():
                missing = sorted(set(range(self.hparams["split_ngram_parts"])) - set(stream["placed"]))
                if missing:
                    incomplete[bid] = missing
            if incomplete:
                raise ValueError(f"unprocessed n-gram embedding shards: {incomplete}")
