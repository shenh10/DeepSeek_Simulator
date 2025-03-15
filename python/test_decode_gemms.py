import argparse
import csv
import math
import os
import random
import torch
from typing import Tuple, Callable  

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor, set_num_sms

from common import TestConfig

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
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

def construct_bmm(b_lhs: int, b_rhs: int, m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out



def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
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

class PerformanceLogger:  
    def __init__(self, base_csv_file: str = 'performance_metrics.csv'):  
        # 为两种不同的日志创建不同的文件  
        self.base_file = base_csv_file  
        self.full_file = base_csv_file.replace('dense', 'group')
        self.base_file_exists = os.path.isfile(self.base_file)  
        self.full_file_exists = os.path.isfile(self.full_file)  
        if self.base_file_exists:  
            os.remove(self.base_file)  
            self.base_file_exists = False  
            
        if self.full_file_exists:  
            os.remove(self.full_file)  
            self.full_file_exists = False 

    def log_full(self,  
                 matrix_idx: int,  
                 num_groups: int,  
                 m: int,  
                 n: int,  
                 k: int,  
                 t: float,  
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
                  f'm_per_group={m_per_group:4}, matrix_idx={matrix_idx:2}, '  
                  f'm={m:4}, n={n:4}, k={k:4}): {time_us:4.0f} us | '  
                  f'throughput: {throughput_TFLOPS:4.0f} TFLOPS, '  
                  f'{bandwidth_GBps:4.0f} GB/s')  

        # 写入CSV文件  
        with open(self.full_file, 'a', newline='') as f:  
            writer = csv.writer(f)  
            if not self.full_file_exists:  
                headers = [  
                    'd',  
                    'num_groups',  
                    'b_mla',  
                    'm_per_group',  
                    'matrix_idx',  
                    'm',  
                    'n',  
                    'k',  
                    'time_us',  
                    'throughput_TFLOPS',  
                    'bandwidth_GBps'  
                ]  
                writer.writerow(headers)  
                self.full_file_exists = True  

            row = [  
                d,  
                num_groups,  
                b_mla,  
                m_per_group,  
                matrix_idx,  
                m,  
                n,  
                k,  
                f'{time_us:.0f}',  
                f'{throughput_TFLOPS:.0f}',  
                f'{bandwidth_GBps:.0f}'  
            ]  
            writer.writerow(row)  

    def log(self,  
            matrix_idx: int,  
            num_groups: int,  
            m: int,  
            n: int,  
            k: int,  
            t: float,  
            print_console: bool = True) -> None:  
        """基础版本的logger"""  
        # 计算性能指标  
        time_us = t * 1e6  
        throughput_TFLOPS = 2 * num_groups * m * n * k / t / 1e12  
        bandwidth_GBps = (num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t  

        # 打印到控制台，只包含基础字段  
        if print_console:  
            print(f' > Performance matrix_idx={matrix_idx:2}, num_groups={num_groups:2}, '  
                  f'm={m:4}, n={n:4}, k={k:4}): {time_us:4.0f} us | '  
                  f'throughput: {throughput_TFLOPS:4.0f} TFLOPS, '  
                  f'{bandwidth_GBps:4.0f} GB/s')  

        # 写入CSV文件  
        with open(self.base_file, 'a', newline='') as f:  
            writer = csv.writer(f)  
            if not self.base_file_exists:  
                headers = [  
                    'matrix_idx',  
                    'num_groups',  
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
                num_groups,  
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
                     num_groups: int,   
                     m: int,   
                     n: int,   
                     k: int,  
                     d: int = None,  
                     b_mla: int = None,  
                     m_per_group: int = None) -> None:  
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)  
        
        if all(x is not None for x in [d, b_mla, m_per_group]):  
            self.logger.log_full(  
                matrix_idx=matrix_idx,  
                num_groups=num_groups,  
                m=m,  
                n=n,  
                k=k,  
                t=t,  
                d=d,  
                b_mla=b_mla,  
                m_per_group=m_per_group,  
                print_console=True  
            )  
        else:  
            self.logger.log(  
                matrix_idx=matrix_idx,  
                num_groups=num_groups,  
                m=m if matrix_idx != 3 else int(m / 128),  
                n=n if matrix_idx != 3 else n * 128,
                k=k,  
                t=t,  
                print_console=True  
            )  

    def test_gemm(self, config: TestConfig) -> None:  
        print('Testing GEMM:')  
        num_groups = 1  
        b_and_m_per_groups = config.generate_b_and_m_per_groups()
        m_set = sorted(set([b_mla for d, num_groups, b_mla, m_per_group in b_and_m_per_groups]))
        for m in m_set:  
            for matrix_idx, k, n in [(1, 7168, 2112), (2, 1536, 24576),   
                                   (4, 16384, 7168), (5, 7168, 4096),   
                                   (6, 2048, 7168)]:  
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)  
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)  
                diff = calc_diff(out, ref_out)  
                assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'  

                def test_func():  
                    x_fp8, y_fp8, out, ref_out = construct(m, k, n)  
                    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)  

                self.run_benchmark(test_func, matrix_idx, num_groups, m, n, k)  
        print()  

    def test_m_grouped_gemm_contiguous(self, config: TestConfig) -> None:  
        print('Testing grouped contiguous GEMM:')  
        num_groups = 1  

        b_and_m_per_groups = config.generate_b_and_m_per_groups()
        m_set = sorted(set([(3, b_mla * 128, 512, 128) for d, num_groups, b_mla, m_per_group in b_and_m_per_groups]))
        for matrix_idx, m, k, n in m_set:  
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)  
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)  
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)  
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)  
            diff = calc_diff(out, ref_out)  
            assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'  

            def test_func():
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)  
                m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)  
                m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)  
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)  

            self.run_benchmark(test_func, matrix_idx, num_groups, m, n, k)  
        print()  

    def test_bmm(self) -> None:
        print('Testing batch GEMM:')  
        import torch.cuda.nvtx as nvtx  

        a = torch.randn((64, 1, 512), device='cuda', dtype=torch.bfloat16)  
        b = torch.randn((128, 512, 128), device='cuda', dtype=torch.bfloat16)
        nvtx.range_push("matrix_multiplication")  # 开始标记  
        result = torch.einsum('bij,kjl->bikl', a, b)
        torch.cuda.synchronize()  # 确保 CUDA 操作完成  
        nvtx.range_pop()  # 结束标记 
        print(result[0, 0, 1, :])  


    def test_m_grouped_gemm_masked(self, config: TestConfig) -> None:  
        print('Testing grouped masked GEMM:')  
        b_and_m_per_groups = config.generate_b_and_m_per_groups()  
        for d, num_groups, b_mla, m_per_group in b_and_m_per_groups:  
            for matrix_idx, k, n in ((7, 7168, 4096), (8, 2048, 7168)):  
                masked_m_candidates = list(filter(  
                    lambda candidate: candidate <= m_per_group,   
                    (4, 8, 16, 32, 64, 128, 192, 256, 320, 384)  
                ))  
                
                # Correctness testing  
                for i in range(10):  
                    x_fp8, y_fp8, out, ref_out = construct_grouped(  
                        num_groups, m_per_group, k, n, is_masked=True  
                    )  
                    masked_m = torch.empty((num_groups,), device='cuda', dtype=torch.int)  
                    for j in range(num_groups):
                        try:
                            masked_m[j] = random.choice(masked_m_candidates)  
                        except:
                            import pdb
                            pdb.set_trace()
                    expected_m = min(int(masked_m.float().mean()) + 1, m_per_group)  
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(  
                        x_fp8, y_fp8, out, masked_m, expected_m  
                    )  
                    
                    for j in range(num_groups):  
                        diff = calc_diff(  
                            out[j, :masked_m[j].item()],   
                            ref_out[j, :masked_m[j].item()]  
                        )  
                        assert diff < 0.001, (  
                            f'{m_per_group=}, {k=}, {n=}, {j=}, '  
                            f'masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'  
                        )  

                def test_func():  
                    x_fp8, y_fp8, out, ref_out = construct_grouped(  
                        num_groups, m_per_group, k, n, is_masked=True  
                    )  
                    masked_m = torch.ones((num_groups,), device='cuda', dtype=torch.int) * m_per_group  
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(  
                        x_fp8, y_fp8, out, masked_m, m_per_group  
                    )  

                self.run_benchmark(  
                    test_func, matrix_idx, num_groups, m_per_group, n, k, d, b_mla, m_per_group  
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

    print('Library path:')  
    print(f' > {deep_gemm.__path__}\n')  

    output_filename = f"{args.prefix}dense_gemm.csv" if args.prefix else "dense_gemm.csv"  
    output_path = os.path.join(args.output_dir, output_filename)  

    # 初始化配置和测试器 
    config = TestConfig()  
    logger = PerformanceLogger(output_path)  
    tester = GEMMTester(logger)  

    # 运行测试  
    tester.test_gemm(config)
    tester.test_m_grouped_gemm_contiguous(config)  
    tester.test_m_grouped_gemm_masked(config)
    # tester.test_bmm()

if __name__ == '__main__':  
    main()  