"""
Based on python/tutorials/06-fused-attention.py
"""

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, Decay, First,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = q.to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    decay = tl.load(Decay + off_hz % H)
    first = tl.load(First + off_hz % H)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.dot(q, k)
        powers = (offs_m[:, None] - (offs_n + start_n)[None, :])
        att_mask = tl.where(powers > 0, tl.math.fast_powf(decay, powers - 1), 0).to(tl.float16)
        att_mask += tl.where(powers == 0, first, 0)
        qk *= att_mask
        acc += tl.dot(qk.to(tl.float16), v)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


@triton.jit
def _bwd_kernel(
    Q, K, V, Decay, First, Out, DO,
    DQ, DK, DV, DDecay, DFirst,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    assert BLOCK_M == BLOCK_N
    num_block = (N_CTX + BLOCK_M - 1) // BLOCK_M
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    decay = tl.load(Decay + off_h)
    first = tl.load(First + off_h)
    ddecay = 0.
    dfirst = 0.
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k))
            powers = (offs_m_curr[:, None] - offs_n[None, :])
            att_mask = tl.where(powers > 0, tl.math.fast_powf(decay, powers - 1), 0).to(tl.float16)
            att_mask += tl.where(powers == 0, first, 0)
            qkm = qk * att_mask
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(qkm.to(Q.dtype.element_ty)), do)
            # compute dqkm = dot(v, do)
            dqkm = tl.dot(do, tl.trans(v))
            dqk = dqkm * att_mask
            datt_mask = dqkm * qk
            ddecay += tl.sum(tl.where(powers > 0, datt_mask * att_mask / decay * (powers - 1), 0))
            dfirst += tl.sum(tl.where(powers == 0, datt_mask, 0))
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(dqk.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(dqk.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    tl.atomic_add(DDecay + off_h, ddecay)
    tl.atomic_add(DFirst + off_h, dfirst)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, decay, first):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q, k, v, decay, first,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=4)

        ctx.save_for_backward(q, k, v, o, decay, first)
        ctx.grid = grid
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        # BLOCK, num_warps and num_stages are got by autotuning on A100
        BLOCK = 64
        q, k, v, o, decay, first = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        ddecay = torch.zeros_like(decay)
        dfirst = torch.zeros_like(first)
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, decay, first,
            o, do,
            dq, dk, dv, ddecay, dfirst,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=4, num_stages=1,
        )
        return dq, dk, dv, ddecay, dfirst


attention = _attention.apply


def pytorch_mha(r, k, v, att_mask):
    qk = (r @ k.transpose(-2, -1))
    qkm = qk * att_mask
    ref_out = qkm @ v
    return ref_out


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(6, 9, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    r = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    dout = torch.randn_like(r)
    decay = torch.arange(0.9, 0.9 + H * 0.01, 0.01, device='cuda').requires_grad_()
    first = torch.arange(0.95, 0.95 + H * 0.001, 0.001, device='cuda').requires_grad_()
    # reference implementation
    att_mask = torch.zeros(H, N_CTX, N_CTX, dtype=dtype, device="cuda")
    rows, cols = torch.tril_indices(N_CTX, N_CTX, device='cuda')
    powers = rows - cols - 1
    for h in range(H):
        att_mask[h][rows, cols] = (decay[h] ** powers).half()
        att_mask[h][range(N_CTX), range(N_CTX)] = first[h].half()
    ref_out = pytorch_mha(r, k, v, att_mask)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, r.grad = r.grad.clone(), None
    ref_ddecay, decay.grad = decay.grad.clone(), None
    ref_dfirst, first.grad = first.grad.clone(), None

    # triton implementation
    tri_out = attention(r, k, v, decay, first)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, r.grad = r.grad.clone(), None
    tri_ddecay, decay.grad = decay.grad.clone(), None
    tri_dfirst, first.grad = first.grad.clone(), None

    # compare
    assert torch.allclose(ref_out, tri_out, atol=5e-2, rtol=2e-3)
    assert torch.allclose(ref_dv, tri_dv, atol=5e-2, rtol=2e-3)
    assert torch.allclose(ref_dk, tri_dk, atol=5e-2, rtol=2e-3)
    assert torch.allclose(ref_dq, tri_dq, atol=5e-2, rtol=2e-3)
    assert torch.allclose(ref_ddecay, tri_ddecay, atol=5e-2, rtol=1e-2)
    assert torch.allclose(ref_dfirst, tri_dfirst, atol=5e-2, rtol=2e-2)


BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(7, 12)],
    line_arg='provider',
    line_vals=['triton', 'pytorch'],
    line_names=['triton', 'pytorch'],
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, 'mode': mode}
) for mode in ['fwd', 'bwd']]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)

    att_mask = torch.zeros(H, N_CTX, N_CTX, dtype=dtype, device="cuda")
    decay = torch.arange(0.9, 0.9 + H * 0.01, 0.01, device='cuda')
    first = torch.arange(0.95, 0.95 + H * 0.001, 0.001, device='cuda')
    rows, cols = torch.tril_indices(N_CTX, N_CTX, device='cuda')
    powers = rows - cols - 1
    for h in range(H):
        att_mask[h][rows, cols] = (decay[h] ** powers).half()
        att_mask[h].fill_diagonal_(first[h].item())

    if provider == "triton":
        fn = lambda: attention(q, k, v, decay, first)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "pytorch":
        fn = lambda: pytorch_mha(q, k, v, att_mask)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
if __name__ == '__main__':
    bench_flash_attention.run(save_path='.', print_data=True)


