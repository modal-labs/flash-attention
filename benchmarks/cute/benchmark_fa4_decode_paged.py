#!/usr/bin/env python3
"""
FA4 (Cute SM100) paged-attention decode benchmark sweep.

Goal: approximate vLLM / sglang-style decode attention:
  - seqlen_q = 1 (one token per sequence)
  - paged KV cache (K/V stored in pages; page_table maps seq -> physical pages)
  - per-sequence cache lengths (seqused_k / cache_seqlens)
  - Example “find best settings” sweep (writes CSV):
    ./venv/bin/python benchmarks/cute/benchmark_fa4_decode_paged.py --batch-sizes 1,8,32,128 --seqlen-ks 1k,2k,4k,8k,16k,32k --page-sizes 128,256,512
    --m-block-sizes 128 --n-block-sizes 128 --num-splits 1,2,4,8,16 --bench-mode compiled+cudagraph --csv /tmp/fa4_decode.csv
  - Note: on SM100, the current TMA paged-KV fast path uses `n_block_size=128` and supports `page_size` multiples of 128.
  - If you want to test “decode as noncausal” (often valid for q_len=1 and can enable different scheduling):
    ./venv/bin/python benchmarks/cute/benchmark_fa4_decode_paged.py --no-causal ...
This script is intentionally SM100-only (B200).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
import re
import sys
from typing import Iterable, Literal, Optional

import torch

from triton.testing import do_bench, do_bench_cudagraph

from flash_attn.cute import interface as fa4


def _parse_int_list(s: str, *, name: str) -> list[int]:
    if not s:
        raise ValueError(f"{name} must be non-empty")
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


_SUFFIX_RE = re.compile(r"^([0-9]+)([kKmMgG]?)$")


def _parse_len(x: str) -> int:
    """Parse lengths like '8192', '16k', '64K'."""
    x = x.strip()
    m = _SUFFIX_RE.match(x)
    if m is None:
        raise ValueError(f"Invalid length: {x!r} (expected e.g. 8192 or 16k)")
    base = int(m.group(1))
    suf = m.group(2).lower()
    if suf == "k":
        return base * 1024
    if suf == "m":
        return base * 1024 * 1024
    if suf == "g":
        return base * 1024 * 1024 * 1024
    return base


def _parse_len_list(s: str, *, name: str) -> list[int]:
    if not s:
        raise ValueError(f"{name} must be non-empty")
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_parse_len(part))
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp8_e4m3fn", "fp8e4m3fn", "fp8_e4m3", "fp8e4m3", "e4m3"):
        return torch.float8_e4m3fn
    if s in ("fp8_e5m2", "fp8e5m2", "e5m2"):
        return torch.float8_e5m2
    raise ValueError(
        f"Unsupported dtype {s!r} (expected bf16, fp16, fp8_e4m3fn, or fp8_e5m2)"
    )


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    nheads_q: int
    nheads_kv: int
    headdim: int
    headdim_v: int
    dtype: torch.dtype
    causal: bool

    batch_size: int
    seqlen_k: int
    page_size: int
    page_table_layout: Literal["semi_contig_shuffle", "contig", "random"]

    m_block_size: int
    n_block_size: int
    num_splits: int
    pack_gqa: Optional[bool]


def _make_page_table(
    *,
    batch_size: int,
    pages_per_seq: int,
    device: torch.device,
    layout: Literal["semi_contig_shuffle", "contig", "random"],
    seed: int,
) -> torch.Tensor:
    total_pages = batch_size * pages_per_seq
    if layout == "contig":
        page_ids = torch.arange(total_pages, device=device, dtype=torch.int32)
        return page_ids.view(batch_size, pages_per_seq).contiguous()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if layout == "semi_contig_shuffle":
        perm = torch.randperm(batch_size, generator=gen, device=device, dtype=torch.int32)
        base = perm[:, None] * pages_per_seq
        offsets = torch.arange(pages_per_seq, device=device, dtype=torch.int32)[None, :]
        return (base + offsets).contiguous()

    if layout == "random":
        perm_pages = torch.randperm(total_pages, generator=gen, device=device, dtype=torch.int32)
        return perm_pages.view(batch_size, pages_per_seq).contiguous()

    raise ValueError(f"Unknown page_table_layout: {layout}")


def _bytes_per_step(
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k_total: int,
    nheads_q: int,
    nheads_kv: int,
    headdim: int,
    headdim_v: int,
    dtype_qkv: torch.dtype,
    dtype_out: torch.dtype,
) -> int:
    bpe_qkv = torch.tensor([], dtype=dtype_qkv).element_size()
    bpe_out = torch.tensor([], dtype=dtype_out).element_size()
    bytes_k = seqlen_k_total * nheads_kv * headdim * bpe_qkv
    bytes_v = seqlen_k_total * nheads_kv * headdim_v * bpe_qkv
    bytes_q = batch_size * seqlen_q * nheads_q * headdim * bpe_qkv
    bytes_o = batch_size * seqlen_q * nheads_q * headdim_v * bpe_out
    return bytes_k + bytes_v + bytes_q + bytes_o


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"SM100 (major=10) required; got SM{major}{minor}.")


def _bench_one(
    cfg: SweepConfig,
    *,
    warmup: int,
    rep: int,
    use_cudagraph: bool,
    call_mode: Literal["compiled", "wrapper"],
    seed: int,
) -> dict:
    device = torch.device("cuda")
    seqlen_q = 1

    assert cfg.nheads_q % cfg.nheads_kv == 0, "nheads_q must be divisible by nheads_kv"
    qhead_per_kvhead = cfg.nheads_q // cfg.nheads_kv
    is_fp8 = _is_fp8_dtype(cfg.dtype)

    compute_capability = fa4._get_device_capability()
    pack_gqa_eff = cfg.pack_gqa
    if pack_gqa_eff is None:
        pack_gqa_eff = qhead_per_kvhead > 1

    # Mirror interface q_stage heuristic (SM100+): use 2-stage Q pipelining only when it helps.
    # Note: interface uses max_seqlen_q * qhead_per_kvhead (independent of pack_gqa).
    seqlen_q_packgqa = seqlen_q * qhead_per_kvhead
    if compute_capability == 10:
        q_stage = 2 if seqlen_q_packgqa > cfg.m_block_size else 1
    else:
        q_stage = 1

    # SM100 TMA paged path supports page_size multiples of 128 with n_block_size=128.
    if cfg.page_size >= 128 and cfg.page_size % 128 == 0 and cfg.n_block_size != 128:
        raise ValueError(
            f"For page_size={cfg.page_size} (TMA paged path), require n_block_size=128."
        )

    # Allocate tensors.
    pages_per_seq = (cfg.seqlen_k + cfg.page_size - 1) // cfg.page_size
    total_pages = cfg.batch_size * pages_per_seq

    torch.manual_seed(seed)
    if is_fp8:
        q = torch.randn(
            (cfg.batch_size, seqlen_q, cfg.nheads_q, cfg.headdim),
            device=device,
            dtype=torch.bfloat16,
        ).to(cfg.dtype)
        k = torch.randn(
            (total_pages, cfg.page_size, cfg.nheads_kv, cfg.headdim),
            device=device,
            dtype=torch.bfloat16,
        ).to(cfg.dtype)
        v = torch.randn(
            (total_pages, cfg.page_size, cfg.nheads_kv, cfg.headdim_v),
            device=device,
            dtype=torch.bfloat16,
        ).to(cfg.dtype)
        out_dtype = torch.bfloat16
    else:
        q = torch.randn(
            (cfg.batch_size, seqlen_q, cfg.nheads_q, cfg.headdim),
            device=device,
            dtype=cfg.dtype,
        )
        k = torch.randn(
            (total_pages, cfg.page_size, cfg.nheads_kv, cfg.headdim),
            device=device,
            dtype=cfg.dtype,
        )
        v = torch.randn(
            (total_pages, cfg.page_size, cfg.nheads_kv, cfg.headdim_v),
            device=device,
            dtype=cfg.dtype,
        )
        out_dtype = cfg.dtype
    page_table = _make_page_table(
        batch_size=cfg.batch_size,
        pages_per_seq=pages_per_seq,
        device=device,
        layout=cfg.page_table_layout,
        seed=seed,
    )
    cache_seqlens = torch.full((cfg.batch_size,), cfg.seqlen_k, device=device, dtype=torch.int32)
    out = torch.empty((cfg.batch_size, seqlen_q, cfg.nheads_q, cfg.headdim_v), device=device, dtype=out_dtype)

    # Compile & warm up using the public wrapper once.
    # For split-KV we benchmark via compiled kernels to avoid per-iter allocations in _flash_attn_fwd.
    fa4._flash_attn_fwd(
        q,
        k,
        v,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=cfg.causal,
        m_block_size=cfg.m_block_size,
        n_block_size=cfg.n_block_size,
        num_splits=cfg.num_splits,
        pack_gqa=pack_gqa_eff,
        return_lse=False,
        out=out,
        lse=None,
    )

    # The "compiled" call mode: call the compiled kernel directly (plus combine if split),
    # with preallocated partial buffers. This isolates GPU work and avoids allocations.
    if call_mode == "compiled":
        softmax_scale = 1.0 / math.sqrt(cfg.headdim)
        # Find the compile key used by the wrapper by recomputing it (mirrors interface._flash_attn_fwd).
        cute_dtype = fa4.torch2cute_dtype_map[cfg.dtype]

        is_split_kv = cfg.num_splits > 1
        compile_key = (
            cute_dtype,
            cfg.headdim,
            cfg.headdim_v,
            qhead_per_kvhead,
            cfg.causal,
            False,  # score_mod_hash
            False,  # mask_mod_hash
            False,  # use_block_sparsity
            0,  # aux_tensors count
            True,  # lse is None
            True,  # cu_seqlens_q is None
            True,  # cu_seqlens_k is None
            True,  # seqused_q is None
            False,  # seqused_k is None
            True,  # page_table is not None
            cfg.page_size,
            False,  # window_size_left is not None
            False,  # window_size_right is not None
            False,  # learnable_sink is not None
            False,  # q_descale is not None
            False,  # k_descale is not None
            False,  # v_descale is not None
            cfg.m_block_size,
            cfg.n_block_size,
            q_stage,
            384,  # num_threads (ignored on sm100 but in compile key)
            is_split_kv,
            pack_gqa_eff,
            compute_capability,
            not (cfg.page_size >= 128 and cfg.page_size % 128 == 0 and cfg.n_block_size == 128),  # paged KV non-TMA flag
        )
        compiled_fwd = fa4._flash_attn_fwd.compile_cache[compile_key]

        if is_split_kv:
            out_partial = torch.empty(
                (cfg.num_splits, cfg.batch_size, seqlen_q, cfg.nheads_q, cfg.headdim_v),
                device=device,
                dtype=torch.float32,
            )
            # lse_partial: shape (num_splits, batch, head, seqlen_q)
            lse_partial = torch.empty(
                (cfg.num_splits, cfg.batch_size, cfg.nheads_q, seqlen_q),
                device=device,
                dtype=torch.float32,
            )

            def fn():
                current_stream = fa4.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                q_call, k_call, v_call = q, k, v
                if is_fp8:
                    q_call = q.view(torch.uint8)
                    k_call = k.view(torch.uint8)
                    v_call = v.view(torch.uint8)
                compiled_fwd(
                    q_call,
                    k_call,
                    v_call,
                    out_partial,
                    lse_partial,
                    softmax_scale,
                    current_stream,
                    None,  # cu_seqlens_q
                    None,  # cu_seqlens_k
                    None,  # seqused_q
                    cache_seqlens,
                    page_table,
                    None,  # window_size_left
                    None,  # window_size_right
                    None,  # learnable_sink
                    None,  # q_descale
                    None,  # k_descale
                    None,  # v_descale
                    None,  # block_sparse_tensors
                    None,  # aux_tensors
                )
                fa4._flash_attn_fwd_combine(
                    out_partial,
                    lse_partial.transpose(-1, -2),
                    out,
                    None,
                    None,
                    None,
                )

        else:

            def fn():
                current_stream = fa4.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                q_call, k_call, v_call = q, k, v
                if is_fp8:
                    q_call = q.view(torch.uint8)
                    k_call = k.view(torch.uint8)
                    v_call = v.view(torch.uint8)
                compiled_fwd(
                    q_call,
                    k_call,
                    v_call,
                    out,
                    None,  # lse
                    softmax_scale,
                    current_stream,
                    None,  # cu_seqlens_q
                    None,  # cu_seqlens_k
                    None,  # seqused_q
                    cache_seqlens,
                    page_table,
                    None,  # window_size_left
                    None,  # window_size_right
                    None,  # learnable_sink
                    None,  # q_descale
                    None,  # k_descale
                    None,  # v_descale
                    None,  # block_sparse_tensors
                    None,  # aux_tensors
                )

    else:

            def fn():
                fa4._flash_attn_fwd(
                    q,
                    k,
                    v,
                    seqused_k=cache_seqlens,
                    page_table=page_table,
                    causal=cfg.causal,
                    m_block_size=cfg.m_block_size,
                    n_block_size=cfg.n_block_size,
                    num_splits=cfg.num_splits,
                    pack_gqa=pack_gqa_eff,
                    return_lse=False,
                    out=out,
                    lse=None,
                )

    # Benchmark
    if use_cudagraph:
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            t_ms = do_bench_cudagraph(fn, rep=rep)
    else:
        t_ms = do_bench(fn, warmup=warmup, rep=rep)

    total_seqlen_k = int(cache_seqlens.sum().item())
    mem_bytes = _bytes_per_step(
        batch_size=cfg.batch_size,
        seqlen_q=seqlen_q,
        seqlen_k_total=total_seqlen_k,
        nheads_q=cfg.nheads_q,
        nheads_kv=cfg.nheads_kv,
        headdim=cfg.headdim,
        headdim_v=cfg.headdim_v,
        dtype_qkv=cfg.dtype,
        dtype_out=out_dtype,
    )

    t_s = t_ms * 1e-3
    tokens = cfg.batch_size * seqlen_q
    tok_s = tokens / t_s
    gb_s = (mem_bytes / 1e9) / t_s
    return {
        "dtype": str(cfg.dtype).replace("torch.", ""),
        "batch": cfg.batch_size,
        "seqlen_k": cfg.seqlen_k,
        "page_size": cfg.page_size,
        "page_table_layout": cfg.page_table_layout,
        "m_block": cfg.m_block_size,
        "n_block": cfg.n_block_size,
        "q_stage": q_stage,
        "num_splits": cfg.num_splits,
        "pack_gqa_req": cfg.pack_gqa,
        "pack_gqa_eff": pack_gqa_eff,
        "causal": cfg.causal,
        "call_mode": call_mode,
        "use_cudagraph": use_cudagraph,
        "time_us": t_ms * 1e3,
        "tok_s": tok_s,
        "gb_s_est": gb_s,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16, fp16, fp8_e4m3fn, or fp8_e5m2")
    parser.add_argument("--nheads-q", type=int, default=128)
    parser.add_argument("--nheads-kv", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--headdim-v", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch-sizes", type=str, default="1,8,32,128")
    parser.add_argument("--seqlen-ks", type=str, default="1k,2k,4k,8k,16k,32k")
    # Default to page_size=128 since that's the current SM100 TMA paged fast path.
    parser.add_argument("--page-sizes", type=str, default="128")
    parser.add_argument(
        "--page-table-layout",
        type=str,
        default="semi_contig_shuffle",
        choices=["semi_contig_shuffle", "contig", "random"],
    )
    # SM100 decode defaults (safe/fast path).
    parser.add_argument("--m-block-sizes", type=str, default="128")
    parser.add_argument("--n-block-sizes", type=str, default="128")
    parser.add_argument("--num-splits", type=str, default="1,2,4,8,16")
    parser.add_argument(
        "--pack-gqa",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="auto uses FA4 default (on when Hq/Hkv > 1, subject to kernel constraints).",
    )

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--bench-mode",
        type=str,
        default="compiled+cudagraph",
        choices=["compiled", "wrapper", "compiled+cudagraph", "compiled+events"],
        help="compiled: call compiled kernel directly; wrapper: time _flash_attn_fwd wrapper.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")

    args = parser.parse_args(argv)

    _require_sm100()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = _dtype_from_str(args.dtype)

    batch_sizes = _parse_int_list(args.batch_sizes, name="--batch-sizes")
    seqlen_ks = _parse_len_list(args.seqlen_ks, name="--seqlen-ks")
    page_sizes = _parse_int_list(args.page_sizes, name="--page-sizes")
    m_block_sizes = _parse_int_list(args.m_block_sizes, name="--m-block-sizes")
    n_block_sizes = _parse_int_list(args.n_block_sizes, name="--n-block-sizes")
    num_splits_list = _parse_int_list(args.num_splits, name="--num-splits")

    pack_gqa: Optional[bool]
    if args.pack_gqa == "auto":
        pack_gqa = None
    elif args.pack_gqa == "true":
        pack_gqa = True
    else:
        pack_gqa = False

    if args.bench_mode == "compiled":
        call_mode = "compiled"
        use_cudagraph = False
    elif args.bench_mode == "wrapper":
        call_mode = "wrapper"
        use_cudagraph = False
    elif args.bench_mode == "compiled+cudagraph":
        call_mode = "compiled"
        use_cudagraph = True
    elif args.bench_mode == "compiled+events":
        call_mode = "compiled"
        use_cudagraph = False
    else:
        raise ValueError(f"Unhandled --bench-mode: {args.bench_mode}")

    rows: list[dict] = []

    sweep_iter = itertools.product(
        batch_sizes,
        seqlen_ks,
        page_sizes,
        m_block_sizes,
        n_block_sizes,
        num_splits_list,
    )
    for batch_size, seqlen_k, page_size, m_block, n_block, num_splits in sweep_iter:
        # Skip unsupported combos:
        # - SM100 TMA paged path uses n_block_size=128 (page_size must be a multiple of 128).
        if page_size >= 128 and page_size % 128 == 0 and n_block != 128:
            continue
        if page_size != 128 and page_size <= 0:
            continue

        cfg = SweepConfig(
            nheads_q=args.nheads_q,
            nheads_kv=args.nheads_kv,
            headdim=args.headdim,
            headdim_v=args.headdim_v,
            dtype=dtype,
            causal=bool(args.causal),
            batch_size=batch_size,
            seqlen_k=seqlen_k,
            page_size=page_size,
            page_table_layout=args.page_table_layout,
            m_block_size=m_block,
            n_block_size=n_block,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
        )

        try:
            row = _bench_one(
                cfg,
                warmup=args.warmup,
                rep=args.rep,
                use_cudagraph=use_cudagraph,
                call_mode=call_mode,
                seed=args.seed,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(
                f"SKIP (error): B={batch_size} K={seqlen_k} page={page_size} m={m_block} "
                f"n={n_block} splits={num_splits} dtype={dtype}: {e}",
                file=sys.stderr,
            )
            continue
        rows.append(row)

        print(
            "dtype={dtype:<14s} B={batch:<4d} K={seqlen_k:<6d} page={page_size:<3d} "
            "m={m_block:<3d} n={n_block:<3d} splits={num_splits:<3d} "
            "qstage={q_stage:<1d} pack_gqa={pack_gqa_eff!s:<5} causal={causal!s:<5} "
            "t={time_us:8.1f} us  tok/s={tok_s:10.1f}  est={gb_s_est:7.1f} GB/s".format(**row)
        )

    if args.csv is not None:
        import csv

        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
