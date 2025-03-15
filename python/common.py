import math
from dataclasses import dataclass  
from typing import List, Tuple
from itertools import chain  


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



def param_num_to_GB(param_num, ele_size = 1):
    return param_num * ele_size / 1024 / 1024 / 1024


GPUSpec = {
    "H20-96": {
        "volume": 96,               # GB
        "intra_node_bw": 360,       # 机内卡间实际可达带宽 GB/s
        "inter_node_bw": 39,        # 机间卡间实际可达带宽 GB/s
    },
    "H800-80": {
        "volume": 80,
        "intra_node_bw": 360,
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
    q_h: int = 128
    d_h: int = 7168 

    # expert param
    expert_duplication_num: int = 32   # it is hard-coded and should be careful to change. device_nums is bonded to this value
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
    b_mlas: List[int] = None
    gpu: str = "H20-96"
    model_config: ModelConfig = None 


    def __post_init__(self):  
        if self.model_config is None:
            self.model_config = ModelConfig()

        if self.device_nums is None:  
            self.device_nums = find_factors_multiple_of_8(self.model_config.total_duped_router_expert)
        if self.b_mlas is None:  
            self.b_mlas = [8, 16, 32, 64, 128, 256]  
    
        # 计算派生参数  
        self.b_mla_peak = [self.calculate_b_mla_peak(d, GPUSpec[self.gpu]['volume']) for d in self.device_nums]
        self.filter_invalid_config()
        self.num_groups = [int(self.model_config.total_duped_router_expert / d) for d in self.device_nums]  

        print("Search space: ")
        print("GPU Type: ", self.gpu)
        print("Device number: ", self.device_nums)
        print("Max batch size per GPU: ", self.b_mla_peak)
        print("Number of router experts per GPU:", self.num_groups)

    def filter_invalid_config(self):
        self.b_mla_peak, self.device_nums = zip(*filter(lambda x: x[0] != -1, zip(self.b_mla_peak, self.device_nums)))  


    def calculate_b_mla_peak(self, d, volume):
        """
            volume * d - model_config.attention_param_num
        """
        model_config = self.model_config
        total_weights_fp8 = (
            param_num_to_GB(model_config.total_nodup_expert_params) +
            # duplicated expert weights
            param_num_to_GB(model_config.per_expert_param_num) * model_config.moe_layer_num * model_config.expert_duplication_num +
            # shared expert duplication
            param_num_to_GB(model_config.per_expert_param_num) * model_config.moe_layer_num * (d - 1)
        )
        kvcache_max = volume * d - param_num_to_GB(model_config.dense_param_num) * d - total_weights_fp8
        if kvcache_max < 0:
            return -1
        per_token_kv_weight_fp8 = param_num_to_GB((model_config.d_kv_compression + model_config.d_r) * (model_config.moe_layer_num + model_config.dense_layer_num))
        b_mla = math.ceil(
             kvcache_max / per_token_kv_weight_fp8 / d / self.s
        )
        return b_mla


    def generate_b_and_m_per_groups(self) -> List[List[Tuple[int, int, int, int]]]:
        b_and_m_per_groups = list(chain(*[  
            [(self.device_nums[i],   
              int(self.model_config.total_duped_router_expert / self.device_nums[i]),   
              b_mla,   
              ceil_to_power_of_2(b_mla * self.device_nums[i] * self.model_config.topk / self.model_config.total_duped_router_expert))   
             for b_mla in find_power_of_2_lower_than(self.b_mla_peak[i])]  
            for i in range(len(self.device_nums))  
        ]))
        return b_and_m_per_groups

    def calculate_alltoall_time(self, d: int, b_mla: int, is_dispatch: bool = True) -> float:
        """
        Per layer dispatch/combine duration. By default, dispatch in fp8 type and combine in bf16 type.
        """
        inter_node_bw = GPUSpec[self.gpu]['inter_node_bw']
        intra_node_bw = GPUSpec[self.gpu]['intra_node_bw']

        model_config = self.model_config
        per_token_in_GB = param_num_to_GB(model_config.d_h)
        inter_node_token = 8  
        if d <= (model_config.topk + 1) * 8:
            inter_node_token = math.ceil(d / 8) - 1
        else:
            inter_node_token = model_config.topk

        ele_type = 1 if is_dispatch else 2
        inter_node_comm_duration = param_num_to_GB(
            model_config.d_h * inter_node_token * b_mla * ele_type) / inter_node_bw * 10 ** 6 # in us

        intra_node_comm_duration = param_num_to_GB(
            model_config.d_h * min(model_config.topk - 1, 8) * b_mla * ele_type) / intra_node_bw * 10 ** 6
        return max(inter_node_comm_duration, intra_node_comm_duration)  

if __name__ == '__main__':
    config = TestConfig(gpu="H800-80")
    print(config.generate_b_and_m_per_groups())