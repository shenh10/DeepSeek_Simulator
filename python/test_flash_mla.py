import argparse
import math
import random

import torch
import triton

# from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from aiter.ops.triton import decode_mla

from aiter.test_mha_common import attention_ref

import aiter 

from common import TestConfig


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(
            s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / \
        max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-5


@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen):

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(
                random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    num_kv_splits = 16  # don't why but sglang force 16.... for triton

    kv_max_sz = 65536  # calculated by rest of mem after weight loaded in frameworks
    page_size = 1
    num_page = (kv_max_sz + page_size - 1) // page_size


    # d = qk_head_dim, h_kv = nhead_kv = 1, qk_head_dim = d, nhead = h_q, v_head_dim = dv
    q = torch.randn(b, h_q, d)

    kv_buffer = torch.randn(
        (num_page * page_size, h_kv, d),  # decode kv head
    )

    sm_scale = 1.0 / (d**0.5)

    # seq_lens = torch.tensor([ctx_lens for _ in range(b)], dtype=torch.int)
    kv_indptr = torch.zeros((b + 1,), dtype=torch.int)
    kv_indptr[1 : b + 1] = torch.cumsum(cache_seqlens, dim=0)
    kv_indices = torch.randint(
        0, num_page, (kv_indptr[-1].item() + 1,), dtype=torch.int
    )
    
    # block_size = 64
    # block_table = torch.arange(
    #     b * max_seqlen_pad // block_size, dtype=torch.int32
    # ).view(b, max_seqlen_pad // block_size)
    # blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    # for i in range(b):
    #     blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
    #         float("nan")
    #     )
    # blocked_v = blocked_k[..., :dv]

    # tile_scheduler_metadata, num_splits = get_mla_metadata(
    #     cache_seqlens, s_q * h_q // h_kv, h_kv
    # )

    def aiter_flash_mla():
        attn_logits = torch.empty(
            (b, h_q, num_kv_splits, dv + 1),
            dtype=torch.float32,
        )

        kv_last_page_lens = torch.ones(b, dtype=torch.int)
        out_asm = torch.empty((b, h_q, dv), ).fill_(-1)
        attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
            q,
            kv_buffer.view(num_page, page_size, h_kv, d),
            out_asm,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
        )


    # def flash_mla():
    #     return flash_mla_with_kvcache(
    #         q,
    #         blocked_k,
    #         block_table,
    #         cache_seqlens,
    #         dv,
    #         tile_scheduler_metadata,
    #         num_splits,
    #         causal=causal,
    #     )

    # def ref_mla():
    #     out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    #     lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
    #     for i in range(b):
    #         begin = i * max_seqlen_pad
    #         end = begin + cache_seqlens[i]
    #         O, LSE = scaled_dot_product_attention(
    #             q[i].transpose(0, 1),
    #             blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
    #             blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
    #             h_q=h_q,
    #             h_kv=h_kv,
    #             is_causal=causal,
    #         )
    #         out[i] = O.transpose(0, 1)
    #         lse[i] = LSE
    #     return out, lse

    # out_flash, lse_flash = flash_mla()
    # out_torch, lse_torch = ref_mla()
    # cal_diff(out_flash, out_torch, "out")
    # cal_diff(lse_flash, lse_torch, "lse")

    t = triton.testing.do_bench(aiter_flash_mla)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
        torch.finfo(q.dtype).bits // 8
    )

    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}, {t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
    )


def main(torch_dtype):
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    config = TestConfig()
    d, dv = config.model_config.d_kv_compression + \
        config.model_config.d_r, config.model_config.d_kv_compression
    causal = True

    b_and_m_per_groups = config.generate_b_and_m_per_groups()
    m_set = sorted(set([b_mla for d, tp, num_groups, b_mla,
                   m_per_group in b_and_m_per_groups]))
    for b in m_set:
        for s in [config.s]:
            for h_q in [math.ceil(config.model_config.q_head / tp) for tp in config.get_tp_configs()]:
                # for s_q in [1, 2]:  # MTP = 1, 2
                for s_q in [1, ]:  # only need to calculate MTP=1 cause MTP=2 is got by calculation from MTP=1
                    for varlen in [False, True]:
                        test_flash_mla(b, s_q, s, h_q, h_kv,
                                       d, dv, causal, varlen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype)
