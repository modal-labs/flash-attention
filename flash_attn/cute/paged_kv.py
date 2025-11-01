from typing import Type
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Int32, const_expr

from flash_attn.cute import utils
from flash_attn.cute.fast_math import FastDivmod
from flash_attn.cute.cute_dsl_utils import ParamsBase


@dataclass
class PagedKVManager(ParamsBase):
    mPageTable: cute.Tensor
    mK_paged: cute.Tensor
    mV_paged: cute.Tensor
    thread_idx: Int32

    page_size_divmod: FastDivmod
    seqlen_k: Int32
    leftpad_k: Int32
    n_block_size: Int32
    num_threads: cutlass.Constexpr[Int32]
    head_dim_padded: cutlass.Constexpr[Int32]
    head_dim_v_padded: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    page_entry_per_thread: Int32
    async_copy_elems: Int32

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    tPrPage: cute.Tensor
    tPrPageOffset: cute.Tensor
    tKpK: cute.Tensor
    tVpV: cute.Tensor

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        page_size_divmod: FastDivmod,
        bidb: Int32,
        bidh: Int32,
        thread_idx: Int32,
        seqlen_k: Int32,
        leftpad_k: Int32,
        n_block_size: cutlass.Constexpr[Int32],
        head_dim_padded: cutlass.Constexpr[Int32],
        head_dim_v_padded: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
    ):
        universal_copy_bits = 128
        gmem_threads_per_row = 1  # 8 threads loading 128 bits = 128 bytes = 1 cache line
        async_copy_elems = universal_copy_bits // dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)
        page_entry_per_thread = n_block_size * gmem_threads_per_row // num_threads

        tPrPage = cute.make_rmem_tensor((page_entry_per_thread,), Int32)
        tPrPageOffset = cute.make_rmem_tensor((page_entry_per_thread,), Int32)

        mPageTable = mPageTable[bidb, None]
        mK_paged = mK_paged[None, None, bidh, None]
        mV_paged = mV_paged[None, None, bidh, None]

        cK = cute.make_identity_tensor((n_block_size, head_dim_padded))
        tKcK = gmem_thr_copy_KV.partition_S(cK)
        tKpK = utils.predicate_k(tKcK, limit=mK_paged.shape[1])

        if const_expr(head_dim_padded == head_dim_v_padded):
            tVpV = tKpK
        else:
            cV = cute.make_identity_tensor((n_block_size, head_dim_v_padded))
            tVcV = gmem_thr_copy_KV.partition_S(cV)
            tVpV = utils.predicate_k(tVcV, limit=mV_paged.shape[1])

        return PagedKVManager(
            mPageTable,
            mK_paged,
            mV_paged,
            thread_idx,
            page_size_divmod,
            seqlen_k,
            leftpad_k,
            n_block_size,
            num_threads,
            head_dim_padded,
            head_dim_v_padded,
            gmem_threads_per_row,
            page_entry_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            tPrPage,
            tPrPageOffset,
            tKpK,
            tVpV,
        )

    @cute.jit
    def load_page_table(self, n_block: Int32):
        for i in cutlass.range(self.page_entry_per_thread, unroll=1):
            row = (i * self.num_threads + self.thread_idx) // self.gmem_threads_per_row
            row_idx = n_block * self.n_block_size + row

            page_idx, page_offset = self.page_size_divmod.divmod(row_idx + self.leftpad_k)

            is_valid = (
                (i + 1) * self.num_threads <= self.n_block_size or row < self.n_block_size
            ) and row_idx < self.seqlen_k
            page = self.mPageTable[page_idx] if is_valid else 0

            self.tPrPage[i] = page
            self.tPrPageOffset[i] = page_offset

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, K_or_V: str):
        assert K_or_V in ("K", "V")

        # Finesse sX layout to be (M, N).
        sX_pi = cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
                stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2])),
            ),
        )

        if const_expr(K_or_V == "V"):
            # Need to transpose V
            sX_pi = cute.make_tensor(sX_pi.iterator, cute.select(sX_pi.layout, mode=[1, 0]))

        head_dim = self.head_dim_v_padded if const_expr(K_or_V == "V") else self.head_dim_padded
        cX = cute.make_identity_tensor((self.n_block_size, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)

        print("gmem_thr_copy_KV: ", self.gmem_thr_copy_KV)
        print("tXsX: ", tXsX)

        seqlenk_row_limit = self.seqlen_k - n_block * self.n_block_size
        for m in cutlass.range(cute.size(tXsX, mode=[1]), unroll=1):
            should_load = tXcX[0, m, 0][0] < seqlenk_row_limit

            page = self.tPrPage[m]
            page_offset = self.tPrPageOffset[m]
            mX_paged_cur = (
                self.mK_paged[page_offset, None, page]
                if const_expr(K_or_V == "K")
                else self.mV_paged[None, page_offset, page]
            )
            mX_paged_cur_copy = cute.tiled_divide(mX_paged_cur, (self.async_copy_elems,))

            if should_load:
                for k in cutlass.range(cute.size(tXsX, mode=[2]), unroll=1):
                    ki = tXcX[0, 0, k][1] // self.async_copy_elems
                    cute.copy(
                        self.gmem_tiled_copy_KV,
                        mX_paged_cur_copy[None, ki],
                        tXsX[None, m, k],
                        # pred=self.tKpK[0, 0, k],
                    )
