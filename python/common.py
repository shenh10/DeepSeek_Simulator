import math
from dataclasses import dataclass
from typing import List, Tuple
from itertools import chain
import logging

logger = logging.getLogger("DeepSeek-Simulator")
logger.setLevel(logging.INFO)  # 默认级别

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(console_handler)


def ceil_to_power_of_2(x):
    return 2 ** math.ceil(math.log2(x))


def floor_to_power_of_2(x):
    return 2 ** math.floor(math.log2(x))


def find_factors_multiple_of_8(x):
    # 存储所有符合条件的因子
    valid_factors = []

    # 从8开始，因为我们只需要8的倍数
    # 步长设为8，确保每个数都是8的倍数
    for i in range(8, x + 1, 8):
        # 检查是否是因子
        if x % i == 0:
            valid_factors.append(i)

    return valid_factors


def find_power_of_2_lower_than(x):
    return [2**i for i in range(3, x.bit_length()) if 2**i < x]


def param_num_to_GB(param_num, ele_size=1):
    return param_num * ele_size / 1024 / 1024 / 1024


GPUSpec = {
    "H20-96": {
        "volume": 96,               # GB
        "intra_node_bw": 180,       # 机内卡间实际可达带宽 GB/s
        "inter_node_bw": 39,        # 机间卡间实际可达带宽 GB/s
    },
    "H800-80": {
        "volume": 80,
        "intra_node_bw": 180,
        "inter_node_bw": 39,
    }
}


@dataclass
class ModelConfig:
    # MLA param
    d_kv_compression: int = 512
    d_q_compression: int = 1536
    d_r: int = 64
    q_head: int = 128
    d_q: int = 128      # q head dim
    d_kv: int = 128     # kv head dim
    d_h: int = 7168

    # expert param
    # it is hard-coded and should be careful to change. device_nums is bonded to this value
    expert_duplication_num: int = 32
    router_expert_num: int = 256
    moe_layer_num: int = 58
    dense_layer_num: int = 3
    shared_expert_num: int = 1
    expert_hidden_dim: int = 2048
    per_expert_param_num: int = 7168 * 2048 * 3

    topk = 8

    dense_param_num: int = 14 * 1000 * 1000 * 1000

    @property
    def total_duped_router_expert(self):
        return self.expert_duplication_num + self.router_expert_num

    @property
    def total_nodup_expert_params(self):
        return ((self.router_expert_num + self.shared_expert_num) * self.moe_layer_num + 9 * self.dense_layer_num) * self.per_expert_param_num


@dataclass
class TestConfig:
    device_nums: List[int] = None
    s: int = 5000
    gpu: str = "H800-80"
    model_config: ModelConfig = None
    tp_nums:  List[int] = None
    debug: bool = False

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()

        if self.device_nums is None:
            self.device_nums = find_factors_multiple_of_8(
                self.model_config.total_duped_router_expert)

        if self.tp_nums is None:
            self.tp_nums = [1, 2, 4, 8]

        if self.debug:
            logger.setLevel(logging.DEBUG)
        # 计算派生参数
        self.b_mla_and_device_pair = self.get_b_mla_device_pair()

        logger.debug(f"Search space: ")
        logger.debug(f"GPU Type: {self.gpu}")
        for tp, b_mla_and_device_pair in zip(self.tp_nums, self.b_mla_and_device_pair):
            logger.debug(f"TP={tp:2}")
            logger.debug(
                f"\t\t Device Number: {b_mla_and_device_pair['device_nums']}")
            logger.debug(
                f"\t\t Max batch size per GPU: {b_mla_and_device_pair['b_mla_peak']}")

    def get_tp_configs(self):
        return self.tp_nums

    def get_b_mla_device_pair(self):
        b_mla_and_device_pair = []
        for tp in self.tp_nums:
            device_nums = []
            b_mla_peak = []
            for d in self.device_nums:
                b_mla = self.calculate_b_mla_peak(
                    d, tp, GPUSpec[self.gpu]['volume'])
                if b_mla > 0:
                    device_nums.append(d)
                    b_mla_peak.append(b_mla)
            b_mla_and_device_pair.append(
                {'device_nums': device_nums, 'b_mla_peak': b_mla_peak})
        return b_mla_and_device_pair

    def calculate_b_mla_peak(self, d, tp, volume):
        """
            volume * d - model_config.attention_param_num
        """
        model_config = self.model_config
        total_weights_fp8 = param_num_to_GB(
            model_config.total_nodup_expert_params +
            # duplicated expert weights
            model_config.per_expert_param_num * model_config.moe_layer_num * model_config.expert_duplication_num +
            # shared expert duplication
            model_config.per_expert_param_num *
            model_config.moe_layer_num * (d - 1)
        )

        mla_weights = param_num_to_GB(
            # W_lora_a_Q + W_lora_a_KV + W_kv_Rope
            model_config.d_h * (model_config.d_q_compression + model_config.d_kv_compression + model_config.d_r) +
            # W_UQ + W_QR
            model_config.d_q_compression * model_config.q_head / tp * (model_config.d_q + model_config.d_r) +
            # W_UK + W_UV
            model_config.d_kv_compression * model_config.q_head / tp * model_config.d_kv * 2 +
            # W_O
            model_config.q_head * model_config.d_kv * model_config.d_h) * (model_config.moe_layer_num + model_config.dense_layer_num)
        extra_dense_weights = 2.42

        kvcache_max = volume * d - \
            (extra_dense_weights + mla_weights) * d - total_weights_fp8
        if kvcache_max < 0:
            return -1
        logger.debug(
            f"d={d:2}, TP={tp:2}, mla_weights={mla_weights:.4}, moe_weights={total_weights_fp8/d:.4}, total_kv={kvcache_max/d:.4}")
        per_token_kv_weight_fp8 = param_num_to_GB(
            (model_config.d_kv_compression + model_config.d_r) * (model_config.moe_layer_num + model_config.dense_layer_num))
        b_mla = math.ceil(
            kvcache_max / per_token_kv_weight_fp8 / d / self.s
        )
        return b_mla

    def generate_b_and_m_per_groups(self) -> List[List[Tuple[int, int, int, int]]]:
        configs = []
        print("Generate All Configs: ")
        for tp, b_mla_and_device_pair in zip(self.tp_nums, self.b_mla_and_device_pair):
            for d, b_mla_peak in zip(b_mla_and_device_pair['device_nums'], b_mla_and_device_pair['b_mla_peak']):
                num_groups = int(
                    self.model_config.total_duped_router_expert / d)
                dp = d / tp         # it is always an integer since d is force to be multiple of 8
                config_group = []
                for b_mla in find_power_of_2_lower_than(b_mla_peak):
                    m_per_group = ceil_to_power_of_2(
                        b_mla * dp * self.model_config.topk / self.model_config.total_duped_router_expert)
                    if m_per_group < 4:         # DeepGemm TMA need m % 4 == 0
                        continue
                    config = (d, tp, num_groups, b_mla, m_per_group)
                    config_group.append(config)
                configs.extend(sorted(config_group))
                logger.debug(f"TP={tp}, d={d}:\t\t {config_group}")
        logger.debug(f"In total {len(configs)} configs.")
        return configs

    def calculate_alltoall_time(self, d: int, tp: int, b_mla: int, is_dispatch: bool = True) -> float:
        """
        Per layer dispatch/combine duration. By default, dispatch in fp8 type and combine in bf16 type.
        """
        inter_node_bw = GPUSpec[self.gpu]['inter_node_bw']
        intra_node_bw = GPUSpec[self.gpu]['intra_node_bw']

        model_config = self.model_config
        inter_node_token = 8
        if d <= (model_config.topk + 1) * 8:
            inter_node_token = math.ceil(d / 8) - 1
        else:
            inter_node_token = model_config.topk

        ele_type = 1 if is_dispatch else 2
        inter_node_comm_duration = param_num_to_GB(
            model_config.d_h * inter_node_token * b_mla / tp * ele_type) / inter_node_bw * 10 ** 6  # in us

        intra_node_comm_duration = param_num_to_GB(
            model_config.d_h * min(model_config.topk - 1, 8) * b_mla * ele_type) / intra_node_bw * 10 ** 6
        return max(inter_node_comm_duration, 5)
        # return max(inter_node_comm_duration, intra_node_comm_duration)

    def calculate_allreduce_time(self, tp: int, b_mla: int) -> float:
        """
        Per layer allreduce duration. By default, allreduce only performs inside nodes and in bf16 type.
        """
        intra_node_bw = GPUSpec[self.gpu]['intra_node_bw']

        model_config = self.model_config

        ele_type = 2  # bf16
        intra_node_comm_duration = param_num_to_GB(
            2 * (tp - 1) / tp * model_config.d_h * b_mla * ele_type) / intra_node_bw * 10 ** 6
        # lower bound for latency bound communication
        return max(intra_node_comm_duration, 5)


if __name__ == '__main__':
    config = TestConfig(gpu="H20-96", debug=True)
    config.generate_b_and_m_per_groups()
