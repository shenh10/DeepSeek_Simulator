import argparse
import pandas as pd
import math
import os
from dataclasses import dataclass
from typing import List, Tuple
from itertools import chain
from common import TestConfig


def process_data(dense_gemm_file: str, group_gemm_file: str, batch_gemm_file: str, mla_file: str, output_path: str, output_prefix: str):
    # 读取输入文件
    dense_df = pd.read_csv(dense_gemm_file)
    group_df = pd.read_csv(group_gemm_file)
    batch_df = pd.read_csv(batch_gemm_file)
    mla_df = pd.read_csv(mla_file)

    config = TestConfig()
    configs = config.generate_b_and_m_per_groups()

    results = []
    for d, tp, num_groups, b_mla, m_per_group in configs:
        # 获取各个组件的时间
        qkv_time = int(dense_df[(dense_df['m'] == b_mla) &
                                (dense_df['tp'] == 1) &
                                (dense_df['matrix_idx'] == 1)
                                ]['time_us'].iloc[0]) + \
            int(dense_df[(dense_df['m'] == b_mla) &
                         (dense_df['tp'] == tp) &
                         (dense_df['matrix_idx'] == 2)]['time_us'].iloc[0]) + \
            int(batch_df[(batch_df['m'] == b_mla) &
                         (batch_df['tp'] == tp) &
                         (batch_df['matrix_idx'] == 3)]['time_us'].iloc[0])
        attn_time = int(mla_df[(mla_df['mean_sk'] == config.s) &
                        (mla_df['s_q'] == 1) &
                        (mla_df['b'] == b_mla) &
                        (mla_df['varlen'] == True) &
                        (mla_df['h_q'] == config.model_config.q_head / tp)]['latency'].iloc[0] * 1000)
        o_time = int(dense_df[
            (dense_df['m'] == b_mla) &
            (dense_df['tp'] == tp) &
            (dense_df['matrix_idx'] == 4)]['time_us'].iloc[0]) + \
            int(batch_df[(batch_df['m'] == b_mla) &
                         (batch_df['tp'] == tp) &
                         (batch_df['matrix_idx'] == 9)]['time_us'].iloc[0])
        shared_time = int(dense_df[(dense_df['m'] == b_mla) &
                                   (dense_df['matrix_idx'].isin([5, 6]))]['time_us'].sum())
        up_gemm = int(group_df[(group_df['d'] == d) &
                               (group_df['b_mla'] == b_mla) &
                               (group_df['m_per_group'] == m_per_group) &
                               (group_df['matrix_idx'] == 7)]['time_us'].iloc[0])
        down_gemm = int(group_df[(group_df['d'] == d) &
                                 (group_df['m_per_group'] == m_per_group) &
                                 (group_df['b_mla'] == b_mla) &
                                 (group_df['matrix_idx'] == 8)]['time_us'].iloc[0])

        dispatch_alltoall = int(
            config.calculate_alltoall_time(d, tp, b_mla, True))
        combine_alltoall = int(
            config.calculate_alltoall_time(d, tp, b_mla, False))

        allreduce = int(config.calculate_allreduce_time(
            tp, b_mla)) if tp > 1 else 0

        # 计算两种模式下的层时间
        # Two microbatch overlapping
        t_moe_layer_two = int(2 * (max(dispatch_alltoall, shared_time + qkv_time) +
                              up_gemm + down_gemm +
                              max(attn_time + o_time + allreduce, combine_alltoall)))
        t_dense_layer_two = int(
            2 * (shared_time + qkv_time + up_gemm + down_gemm + attn_time + o_time + allreduce))
        # Single batch comp-compute overlapping
        t_moe_layer_single = int(max(dispatch_alltoall, shared_time) + qkv_time + up_gemm +
                                 max(down_gemm, combine_alltoall) + attn_time + o_time + allreduce)
        t_dense_layer_single = int(
            shared_time + qkv_time + up_gemm + down_gemm + attn_time + o_time + allreduce)

        # 计算TPOT和吞吐量
        tpot_two = int((t_moe_layer_two * 58 + t_dense_layer_two * 3) / 1000)
        tpot_single = int(
            (t_moe_layer_single * 58 + t_dense_layer_single * 3) / 1000)

        throughput_two = int(b_mla * 2 * 1000 / tp / tpot_two)
        throughput_single = int(b_mla * 1000 / tp / tpot_single)

        # 存储结果
        base_result = {
            'd': d,
            'tp': tp,
            'b_mla': b_mla,
            'QKV(us)': qkv_time,
            'ATTN(us)': attn_time,
            'O(us)': o_time,
            'Shared(us)': shared_time,
            'Up_Gemm(us)': up_gemm,
            'Down_Gemm(us)': down_gemm,
            'Dispatch_AlltoAll(us)': dispatch_alltoall,
            'Combine_AlltoAll(us)': combine_alltoall,
            'AllReduce(us)': allreduce,

        }

        results.append({
            **base_result,
            't_{dense_layer}(us)': t_dense_layer_two,
            't_{moe_layer}(us)': t_moe_layer_two,
            'TPOT(ms)': tpot_two,
            'Single-Device Throughput(Tokens/s)': throughput_two,
            'mode': 'two-microbatch'
        })

        results.append({
            **base_result,
            't_{dense_layer}(us)': t_dense_layer_single,
            't_{moe_layer}(us)': t_moe_layer_single,
            'TPOT(ms)': tpot_single,
            'Single-Device Throughput(Tokens/s)': throughput_single,
            'mode': 'single-batch'
        })

    # 创建DataFrame并保存结果
    results_df = pd.DataFrame(results)

    # 分别保存两种模式的结果
    two_microbatch_df = results_df[results_df['mode']
                                   == 'two-microbatch'].drop('mode', axis=1)
    single_batch_df = results_df[results_df['mode']
                                 == 'single-batch'].drop('mode', axis=1)

    def float_format(x): return '{:.2f}'.format(
        x) if isinstance(x, float) else x

    two_microbatch_outfile = os.path.join(
        output_path, output_prefix + 'two-microbatch-overlapping.csv')
    single_batch_outfile = os.path.join(
        output_path, output_prefix + 'single-batch-comp-comm-overlapping.csv')
    two_microbatch_df.to_csv(two_microbatch_outfile,
                             index=False, float_format=float_format)
    single_batch_df.to_csv(single_batch_outfile,
                           index=False, float_format=float_format)


def main():
    parser = argparse.ArgumentParser(
        description='Process performance data files')
    parser.add_argument('--dense_gemm', required=True,
                        help='Path to dense_gemm.csv')
    parser.add_argument('--group_gemm', required=True,
                        help='Path to group_gemm.csv')
    parser.add_argument('--batch_gemm', required=True,
                        help='Path to batch_gemm.csv')
    parser.add_argument('--mla', required=True, help='Path to mla.csv')
    parser.add_argument('--output_path',
                        default='.',
                        help='Output directory path (default: current directory)')
    parser.add_argument('--output_prefix',
                        default='',
                        help='Prefix for output files (default: no prefix)')

    args = parser.parse_args()

    process_data(args.dense_gemm, args.group_gemm, args.batch_gemm,
                 args.mla, args.output_path, args.output_prefix)


if __name__ == '__main__':
    main()
