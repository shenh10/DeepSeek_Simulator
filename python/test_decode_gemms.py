import argparse
import csv
import math
import os
import random
import torch
from typing import Tuple, Callable
from enum import Enum

# import deep_gemm
from deepgemm_utils import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor, set_num_sms

from common import TestConfig


class Mode(Enum):
    GROUP = 0
    BATCH = 1
    BASE = 2


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def construct_bmm(b: int, m: int, k: int, n: int) -> \
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
              Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((b, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((b, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((b, m, n), device='cuda', dtype=torch.bfloat16)

    y = y.transpose(-2, -1)
    ref_out = torch.bmm(x, y)

    x_fp8 = to_float8(x)
    y_fp8 = to_float8(y)
    return x, y, x_fp8, y_fp8, out, ref_out


def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty(
        (num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty(
        (num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


block_shape = (128, 128)

from aiter import gemm_a8w8_blockscale

def test_aiter_gemm_asm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda")/10).to(torch.float8_e4m3fnuz)
    weight = (torch.rand( (n, k), dtype=torch.float16, device="cuda")/10).to(torch.float8_e4m3fnuz)
    x_scale = torch.rand([m, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")
    output = torch.zeros(
            [x.shape[0], weight.shape[0]],
            dtype=dtype,
            device=x.device,
        )

    gemm_a8w8_blockscale(x, weight, x_scale, w_scale, output)


def test_aiter_batch_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([b, m, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([b, 1, n], dtype=torch.float32).cuda() + 1e-6

    b, avg_b = aiter.batched_gemm_a8w8_CK(x, weight, x_scale, w_scale, None, dtype)



######################################## fused moe ########################################

BLOCK_SIZE_M = 32
quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
    "wint4afp8smoothquant", # g1u1 support
]

from aiter import ActivationType
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck
from aiter.fused_moe_gelu import fused_topk, moe_align_block_size, fused_experts
from aiter import pertoken_quant, ck_moe
from aiter.ops.shuffle import shuffle_weight


def test_aiter_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=True, shared_E=1, activation = ActivationType.Silu):
        quant_dtype = torch.float8_e4m3fnuz

        input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
        
        w13 = torch.randn((E+shared_E, inter_dim*2, model_dim),
                            dtype=dtype, device="cuda") / 10.0
        
        w2 = torch.randn((E+shared_E, model_dim, inter_dim),
                        dtype=dtype, device="cuda")
        score = torch.randn((token, E), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)

        if shared_E > 0:
            shared_E_score = 0.5
            s_topk_weights = torch.tensor([[shared_E_score, shared_E_score],] * token,
                                        dtype=torch.float32,
                                        device=input.device)
            topk_weights = torch.cat((topk_weights, s_topk_weights), dim=1)
            s_topk_ids = torch.tensor([[E, E+1],] * token,
                                    dtype=torch.int32,
                                    device=input.device)
            topk_ids = torch.cat((topk_ids, s_topk_ids), dim=1)

        w13, fc1_scale = pertoken_quant(
            w13, torch.float, quant_dtype=quant_dtype, dtypeMax=None)
        w2, fc2_scale = pertoken_quant(
            w2, torch.float, quant_dtype=quant_dtype, dtypeMax=None)

        sp1 = (E+shared_E, inter_dim)
        sp2 = (E+shared_E, model_dim)

    
        fc1_smooth_scale = None
        fc2_smooth_scale = None

        # b implement
        w13b = shuffle_weight(w13)
        w2b = shuffle_weight(w2)
        

        asm_moe(input, w13b, w2b, topk_weights, topk_ids,
                                    fc1_scale, fc2_scale,
                                    fc1_smooth_scale, fc2_smooth_scale,
                                    a16=False, activation=activation)



class PerformanceLogger:
    def __init__(self, base_csv_file: str = 'performance_metrics.csv'):
        # 为两种不同的日志创建不同的文件
        self.base_file, self.base_file_exists = self.create_file(base_csv_file)
        self.batch_file, self.batch_file_exists = self.create_file(
            base_csv_file, replace=True, tag='batch')
        self.group_file, self.group_file_exists = self.create_file(
            base_csv_file, replace=True, tag='group')

    def create_file(self, base_csv_file, replace=False, tag=''):
        if replace:
            new_file = base_csv_file.replace('dense', tag)
        else:
            new_file = base_csv_file
        if os.path.isfile(new_file):
            os.remove(new_file)
        return new_file, False

    def log_group(self,
                  matrix_idx: int,
                  num_groups: int,
                  m: int,
                  n: int,
                  k: int,
                  t: float,
                  tp: int,
                  d: int,
                  b_mla: int,
                  m_per_group: int,
                  print_console: bool = True) -> None:
        """完整版本的logger，包含所有字段"""
        # 计算性能指标
        time_us = t * 1e6
        throughput_TFLOPS = 2 * num_groups * m * n * k / t / 1e12
        bandwidth_GBps = (num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t

        # 打印到控制台，包含所有字段
        if print_console:
            print(f' > Performance d={d:2}, num_groups={num_groups:2}, b_mla={b_mla:4}, '
                  f'm_per_group={m_per_group:4}, matrix_idx={matrix_idx:2}, tp={tp:2}, '
                  f'm={m:4}, n={n:4}, k={k:4}): {time_us:4.0f} us | '
                  f'throughput: {throughput_TFLOPS:4.0f} TFLOPS, '
                  f'{bandwidth_GBps:4.0f} GB/s')

        # 写入CSV文件
        with open(self.group_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.group_file_exists:
                headers = [
                    'd',
                    'num_groups',
                    'b_mla',
                    'm_per_group',
                    'matrix_idx',
                    'tp',
                    'm',
                    'n',
                    'k',
                    'time_us',
                    'throughput_TFLOPS',
                    'bandwidth_GBps'
                ]
                writer.writerow(headers)
                self.group_file_exists = True

            row = [
                d,
                num_groups,
                b_mla,
                m_per_group,
                matrix_idx,
                tp,
                m,
                n,
                k,
                f'{time_us:.0f}',
                f'{throughput_TFLOPS:.0f}',
                f'{bandwidth_GBps:.0f}'
            ]
            writer.writerow(row)

    def log_batch(self,
                  matrix_idx: int,
                  batch: int,
                  m: int,
                  n: int,
                  k: int,
                  t: float,
                  tp: int,
                  is_bf16: bool = True,
                  print_console: bool = True) -> None:
        """基础版本的logger"""
        # 计算性能指标
        if is_bf16:
            t = t / 1.7  # discount factor to mimic fp8 performance
            time_us = t * 1e6
            throughput_TFLOPS = 2 * batch * m * n * k / t / 1e12
            bandwidth_GBps = (batch * (m * k + k * n + m * n)) / 1e9 / t
        else:
            time_us = t * 1e6
            throughput_TFLOPS = 2 * batch * m * n * k / t / 1e12
            bandwidth_GBps = (batch * (m * k + k * n + m * n * 2)) / 1e9 / t

        # 打印到控制台，只包含基础字段
        if print_console:
            print(f' > Performance matrix_idx={matrix_idx:2}, tp={tp:2}, batch={batch:2}, '
                  f'm={m:4}, n={n:4}, k={k:4}): {time_us:4.0f} us | '
                  f'throughput: {throughput_TFLOPS:4.0f} TFLOPS, '
                  f'{bandwidth_GBps:4.0f} GB/s')

        # 写入CSV文件
        with open(self.batch_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.batch_file_exists:
                headers = [
                    'matrix_idx',
                    'tp',
                    'batch',
                    'm',
                    'n',
                    'k',
                    'time_us',
                    'throughput_TFLOPS',
                    'bandwidth_GBps'
                ]
                writer.writerow(headers)
                self.batch_file_exists = True

            row = [
                matrix_idx,
                tp,
                batch,
                m,
                n,
                k,
                f'{time_us:.0f}',
                f'{throughput_TFLOPS:.0f}',
                f'{bandwidth_GBps:.0f}'
            ]
            writer.writerow(row)

    def log_base(self,
                 matrix_idx: int,
                 m: int,
                 n: int,
                 k: int,
                 t: float,
                 tp: int,
                 print_console: bool = True) -> None:
        """基础版本的logger"""
        # 计算性能指标
        time_us = t * 1e6
        throughput_TFLOPS = 2 * m * n * k / t / 1e12
        bandwidth_GBps = (m * k + k * n + m * n * 2) / 1e9 / t

        # 打印到控制台，只包含基础字段
        if print_console:
            print(f' > Performance matrix_idx={matrix_idx:2}, tp={tp:2},'
                  f'm={m:4}, n={n:4}, k={k:4}): {time_us:4.0f} us | '
                  f'throughput: {throughput_TFLOPS:4.0f} TFLOPS, '
                  f'{bandwidth_GBps:4.0f} GB/s')

        # 写入CSV文件
        with open(self.base_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.base_file_exists:
                headers = [
                    'matrix_idx',
                    'tp',
                    'm',
                    'n',
                    'k',
                    'time_us',
                    'throughput_TFLOPS',
                    'bandwidth_GBps'
                ]
                writer.writerow(headers)
                self.base_file_exists = True

            row = [
                matrix_idx,
                tp,
                m,
                n,
                k,
                f'{time_us:.0f}',
                f'{throughput_TFLOPS:.0f}',
                f'{bandwidth_GBps:.0f}'
            ]
            writer.writerow(row)


class GEMMTester:
    def __init__(self, logger: PerformanceLogger):
        self.logger = logger

    def run_benchmark(self,
                      test_func: Callable,
                      matrix_idx: int,
                      m: int,
                      n: int,
                      k: int,
                      *,
                      compute_mode: Mode = Mode.BASE,
                      tag: str = 'fp8_gemm',
                      tp: int = 1,
                      batch: int = None,
                      d: int = None,
                      b_mla: int = None,
                      num_groups: int = 1,
                      m_per_group: int = None,
                      ) -> None:
        t = bench_kineto(test_func, tag, suppress_kineto_output=True)

        if compute_mode == Mode.GROUP:
            self.logger.log_group(
                matrix_idx=matrix_idx,
                num_groups=num_groups,
                m=m,
                n=n,
                k=k,
                t=t,
                tp=tp,
                d=d,
                b_mla=b_mla,
                m_per_group=m_per_group,
                print_console=True
            )
        elif compute_mode == Mode.BATCH:
            self.logger.log_batch(
                matrix_idx=matrix_idx,
                batch=batch,
                m=m,
                n=n,
                k=k,
                t=t,
                tp=tp,
                is_bf16=True if tag in ["gemm_bf16", "nvjet_tst"] else False,
                print_console=True
            )
        else:
            self.logger.log_base(
                matrix_idx=matrix_idx,
                m=m,
                n=n,
                k=k,
                t=t,
                tp=tp,
                print_console=True
            )

    def test_gemm(self, config: TestConfig) -> None:
        print('Testing GEMM:')
        num_groups = 1
        b_and_m_per_groups = config.generate_b_and_m_per_groups()
        tp_vars = config.get_tp_configs()
        m_set = sorted(
            set([b_mla for d, tp, num_groups, b_mla, m_per_group in b_and_m_per_groups]))

        test_set = [(1, 1, 7168, 2112), (2, 1, 1536, 24576),
                    (4, 1, 16384, 7168), (5, 1, 7168, 4096),
                    (6, 1, 2048, 7168)]

        def add_tp_shapes(tp_vars, test_set):
            update_set = []
            for matrix_idx, tp, k, n in test_set:
                update_set.append((matrix_idx, tp, k, n))
                for t in tp_vars:
                    if matrix_idx in [2]:
                        # column slice
                        update_set.append((matrix_idx, t, k, math.ceil(n / t)))
                    elif matrix_idx in [4]:
                        # row slice
                        update_set.append((matrix_idx, t, math.ceil(k / t), n))
            return update_set

        updated_set = sorted(set(add_tp_shapes(tp_vars, test_set)))
        print(updated_set)
        for m in m_set:
            for matrix_idx, tp, k, n in updated_set:
                # x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
                # diff = calc_diff(out, ref_out)
                # assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

                def test_func():
                    test_aiter_gemm_asm(torch.bfloat16, m, n, k)
                    # x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                    # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

                self.run_benchmark(test_func, matrix_idx, m, n, k,
                                   tp=tp, compute_mode=Mode.BASE, tag='_ZN2ck27kernel_gemm_xdl_cshuffle_v3INS_42GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3INS_13tensor_l')
        print()

    def test_bmm(self, config: TestConfig, *, use_flashinfer_bmm: bool = False) -> None:
        import torch.cuda.nvtx as nvtx
        print('Testing batch GEMM:')
        # if use_flashinfer_bmm:
        #     from flashinfer import bmm_fp8
        b_and_m_per_groups = config.generate_b_and_m_per_groups()
        tp_vars = config.get_tp_configs()
        m_set = sorted(
            set([b_mla for d, tp, num_groups, b_mla, m_per_group in b_and_m_per_groups]))
        test_set = [(3, 1, 128, 128, 512), (9, 1, 128, 512, 128)]
        # test_set = [(3, 1, 128, 128, 512)]
        # tp_vars = [1]
        # m_set = [64]

        def add_tp_shapes(tp_vars, test_set):
            update_set = []
            for matrix_idx, tp, b, k, n in test_set:
                update_set.append((matrix_idx, tp, b, k, n))
                for t in tp_vars:
                    # column slice
                    update_set.append((matrix_idx, t, math.ceil(b / t), k, n))

            return update_set
        updated_set = sorted(set(add_tp_shapes(tp_vars, test_set)))
        print(updated_set)

        for m in m_set:
            for matrix_idx, tp, b, k, n in updated_set:

                
                def test_func():
                        test_aiter_batch_gemm(torch.bfloat16, b, m, n, k)

                self.run_benchmark(test_func, matrix_idx,
                                    m, n, k, Mode.BATCH, 'cutlass') ## TODO: tag cutlass is wrong!!

                # x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                #     b, m, k, n)
                # if use_flashinfer_bmm:
                #     bmm_fp8(x_fp8[0], y_fp8[0], x_fp8[1],
                #             y_fp8[1], "torch.bfloat16", out)
                #     diff = calc_diff(out, ref_out)
                #     assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

                #     def test_func():
                #         x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                #             b, m, k, n)
                #         bmm_fp8(x_fp8[0], y_fp8[0], x_fp8[1],
                #                 y_fp8[1], "torch.bfloat16", out)
                #     self.run_benchmark(test_func, matrix_idx,
                #                        m, n, k, Mode.BATCH, 'cutlass')
                # else:
                #     def test_func():
                #         x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                #             b, m, k, n)
                #         # with nvtx.range("matmul"):
                #         out = torch.bmm(x_bf16, y_bf16)
                #         # torch.cuda.synchronize()
                #         # nvtx.range_pop()
                #     # test_func()
                #     try:
                #         self.run_benchmark(
                #             test_func, matrix_idx, m, n, k, tp=tp, compute_mode=Mode.BATCH, tag='gemm_bf16', batch=b)
                #     except:
                #         # H20 use a strange kernel named "nvjet_tst_176x64_64x7_1x1_v_bz_TNN" to perform
                #         self.run_benchmark(
                #             test_func, matrix_idx, m, n, k, tp=tp, compute_mode=Mode.BATCH, tag='nvjet_tst', batch=b)


    def test_m_grouped_gemm_masked(self, config: TestConfig) -> None:
        print('Testing grouped masked GEMM:')
        b_and_m_per_groups = config.generate_b_and_m_per_groups()
        for d, _, num_groups, b_mla, m_per_group in b_and_m_per_groups:
            for matrix_idx, k, n in ((7, 2048, 7168),):
                # masked_m_candidates = list(filter(
                #     lambda candidate: candidate <= m_per_group,
                #     (4, 8, 16, 32, 64, 128, 192, 256, 320, 384)
                # ))

                # # Correctness testing
                # for i in range(10):
                #     x_fp8, y_fp8, out, ref_out = construct_grouped(
                #         num_groups, m_per_group, k, n, is_masked=True
                #     )
                #     masked_m = torch.empty(
                #         (num_groups,), device='cuda', dtype=torch.int)
                #     for j in range(num_groups):
                #         masked_m[j] = random.choice(masked_m_candidates)
                #     expected_m = min(
                #         int(masked_m.float().mean()) + 1, m_per_group)
                #     deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                #         x_fp8, y_fp8, out, masked_m, expected_m
                #     )

                #     for j in range(num_groups):
                #         diff = calc_diff(
                #             out[j, :masked_m[j].item()],
                #             ref_out[j, :masked_m[j].item()]
                #         )
                #         assert diff < 0.001, (
                #             f'{m_per_group=}, {k=}, {n=}, {j=}, '
                #             f'masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
                #         )

                def test_func():
                    test_aiter_fmoe(dtype=torch.bfloat16,
                                    token=m_per_group,
                                    model_dim=n,
                                    inter_dim=k,
                                    E=num_groups,
                                    topk=8,
                                    quant='fp8quant',
                                    use_g1u1=True,
                                    shared_E=1,
                                    activation=ActivationType.Silu)

                self.run_benchmark(
                    test_func, matrix_idx, m_per_group, n, k, tp=1, compute_mode=Mode.GROUP, tag='fmoe_fp8_g1u1_subGU_', ## TODO: fp8_gemm needs to change
                    d=d, b_mla=b_mla, num_groups=num_groups, m_per_group=m_per_group
                )
        print()






def parse_args():
    parser = argparse.ArgumentParser(description='GEMM Performance Testing')

    # 添加输出目录参数
    parser.add_argument('--output-dir',
                        type=str,
                        default='.',
                        help='Directory to save output files (default: current directory)')

    # 添加文件名前缀参数
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Prefix for output files (default: no prefix)')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    # print('Library path:')
    # print(f' > {deep_gemm.__path__}\n')

    output_filename = f"{args.prefix}dense_gemm.csv" if args.prefix else "dense_gemm.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    # 初始化配置和测试器
    config = TestConfig()
    logger = PerformanceLogger(output_path)
    tester = GEMMTester(logger)

    # 运行测试
    tester.test_gemm(config)
    tester.test_m_grouped_gemm_masked(config)
    # tester.test_bmm(config)


if __name__ == '__main__':
    main()
