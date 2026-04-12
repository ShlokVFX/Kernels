# Supported:
# - BF16 & FP16 dtype
# - causal & non-causal attention
# - MHA
# - hdim 64, 128
#
# Unsupported (not implemented):
# - GQA, MQA
# - varlen
# - sliding window
# - split-KV
# - hdim 96, 192, 256

import torch
import triton
import triton.language
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
    tcgen05_commit,
)

LOG2E = triton.language.constexpr(1.44269504)

@gluon.constexpr_function
def _nvmma_layout(shape, dtype):
    return gl.NVMMASharedLayout.get_default_for(list(shape), dtype)

@gluon.jit
def flash_attn_kernel(
    q_desc, k_desc, v_desc, o_desc,
    sm_scale, N_CTX,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    CAUSAL: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Each program (start_m, off_hz) handles BLOCK_M rows of Q for one batch*head.
    Grid: (cdiv(N_CTX, BLOCK_M), Z*H).
    """
    dtype: gl.constexpr = q_desc.dtype

    start_m = gl.program_id(0)
    off_hz  = gl.program_id(1)

    q_off = off_hz * N_CTX + start_m * BLOCK_M

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], q_desc.layout)
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, HEAD_DIM], v_desc.layout)

    p_layout: gl.constexpr = _nvmma_layout([BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_layout)

    qk_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N],  col_stride=1)
    o_layout:  gl.constexpr = TensorMemoryLayout([BLOCK_M, HEAD_DIM], col_stride=1)
    qk_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N],  qk_layout)
    o_tmem  = allocate_tensor_memory(gl.float32, [BLOCK_M, HEAD_DIM], o_layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    mbarrier.expect(tma_bar, q_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [q_off, 0], tma_bar, q_smem)
    mbarrier.wait(tma_bar, phase=0)
    tma_phase = 1

    o_reg_layout: gl.constexpr = o_tmem.get_reg_layout()
    o_tmem.store(gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, o_reg_layout))

    qk_reg_layout:   gl.constexpr = qk_tmem.get_reg_layout()
    qk_slice_layout: gl.constexpr = gl.SliceLayout(1, qk_reg_layout)
    qk_col_layout:   gl.constexpr = gl.SliceLayout(0, qk_reg_layout)
    o_slice_layout:  gl.constexpr = gl.SliceLayout(1, o_reg_layout)

    m_i = gl.full([BLOCK_M], float('-inf'), gl.float32, qk_slice_layout)
    l_i = gl.full([BLOCK_M], 0.0,          gl.float32, qk_slice_layout)

    mma_phase = 0

    loop_end = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for j in range(0, loop_end, BLOCK_N):
        kv_off = off_hz * N_CTX + j

        mbarrier.expect(tma_bar, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [kv_off, 0], tma_bar, k_smem)
        tma.async_copy_global_to_shared(v_desc, [kv_off, 0], tma_bar, v_smem)
        mbarrier.wait(tma_bar, phase=tma_phase)
        tma_phase ^= 1

        tcgen05_mma(q_smem, k_smem.permute((1, 0)), qk_tmem, use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1

        qk = qk_tmem.load() * sm_scale

        if CAUSAL:
            offs_m = gl.arange(0, BLOCK_M, qk_slice_layout) + start_m * BLOCK_M
            offs_n = gl.arange(0, BLOCK_N, qk_col_layout)   + j
            causal_mask = offs_n[None, :] > offs_m[:, None]
            qk = gl.where(causal_mask, float('-inf'), qk)

        m_j   = gl.max(qk, 1)
        m_new = gl.maximum(m_i, m_j)
        corr  = gl.exp2((m_i - m_new) * LOG2E)
        p     = gl.exp2((qk - m_new[:, None]) * LOG2E)
        l_i   = l_i * corr + gl.sum(p, 1)

        corr_o = gl.convert_layout(corr, o_slice_layout)
        o_old  = o_tmem.load()
        o_tmem.store(o_old * corr_o[:, None])

        p_smem.store(p.to(dtype))
        fence_async_shared()
        tcgen05_mma(p_smem, v_smem, o_tmem, use_acc=True)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1

        m_i = m_new

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    l_i_o = gl.convert_layout(l_i, o_slice_layout)
    o_reg = o_tmem.load() * (1.0 / l_i_o[:, None])

    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], o_desc.layout)
    o_smem.store(o_reg.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(o_desc, [q_off, 0], o_smem)
    tma.store_wait(pendings=0)

def flash_attn_fwd(q, k, v, sm_scale, causal=True, BLOCK_M=128, BLOCK_N=64):
    """
    q, k, v: [Z, H, N_CTX, HEAD_DIM] fp16 or bf16, contiguous.
    Returns o with the same shape and dtype.
    """
    Z, H, N_CTX, HEAD_DIM = q.shape
    gl_dtype = gl.float16 if q.dtype == torch.float16 else gl.bfloat16

    q_2d = q.reshape(Z * H * N_CTX, HEAD_DIM)
    k_2d = k.reshape(Z * H * N_CTX, HEAD_DIM)
    v_2d = v.reshape(Z * H * N_CTX, HEAD_DIM)
    o    = torch.empty_like(q)
    o_2d = o.reshape(Z * H * N_CTX, HEAD_DIM)

    q_smem_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl_dtype)
    k_smem_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl_dtype)
    v_smem_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl_dtype)
    o_smem_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl_dtype)

    q_desc = TensorDescriptor.from_tensor(q_2d, [BLOCK_M, HEAD_DIM], q_smem_layout)
    k_desc = TensorDescriptor.from_tensor(k_2d, [BLOCK_N, HEAD_DIM], k_smem_layout)
    v_desc = TensorDescriptor.from_tensor(v_2d, [BLOCK_N, HEAD_DIM], v_smem_layout)
    o_desc = TensorDescriptor.from_tensor(o_2d, [BLOCK_M, HEAD_DIM], o_smem_layout)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    flash_attn_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        sm_scale, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        CAUSAL=causal,
        num_warps=4,
    )
    return o

def run_triton_impl():
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def ref_cudnn(q, k, v, scale):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

    torch.manual_seed(42)

    def tflops(ms, Z, H, N, D):
        return 4 * Z * H * N * N * D * 0.5 * 1e-12 / (ms * 1e-3)

    def bench(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench_cudagraph(fn)

    print("Correctness (Gluon causal vs cuDNN causal, bf16)")
    print("=" * 60)
    for Z, H, N, D in [(1, 16, 2048, 128), (2, 16, 4096, 128)]:
        sm = D ** -0.5
        q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
        ref = ref_cudnn(q, k, v, sm)
        out = flash_attn_fwd(q, k, v, sm, causal=True)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"  Z={Z} H={H} N={N} D={D}  max_diff={diff:.5f}  "
              f"{'OK' if diff < 0.02 else 'FAIL'}")
        torch.cuda.empty_cache()

    bench_cfgs = [
        (1, 16,  2048, 128),
        (1, 16,  4096, 128),
        (1, 16,  8192, 128),
        (1, 16, 16384, 128),
        (1, 16, 32768, 128),
    ]

    W = 10
    hdr  = f"{'SeqLen':<10}"
    hdr += f" {'Gluon':>{W}} {'TF/s':>{W}}"
    hdr += f" {'cuDNN':>{W}} {'TF/s':>{W}}"
    hdr += f" {'vsCuDNN':>{W}}"
    sep = "─" * len(hdr)

    print("\nBenchmark — causal=True, bf16, H=16, D=128, B200  [Gluon vs cuDNN]")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for Z, H, N, D in bench_cfgs:
        sm = D ** -0.5
        q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)

        ms_g = bench(lambda: flash_attn_fwd(q, k, v, sm, causal=True))
        tf_g = tflops(ms_g, Z, H, N, D)

        ms_c = bench(lambda: ref_cudnn(q, k, v, sm))
        tf_c = tflops(ms_c, Z, H, N, D)

        row = f"{N:<10}"
        row += f" {ms_g:{W}.3f} {tf_g:{W}.1f}"
        row += f" {ms_c:{W}.3f} {tf_c:{W}.1f}"
        row += f" {ms_c / ms_g:{W}.2f}x"

        print(row)
        torch.cuda.empty_cache()

    print(sep)