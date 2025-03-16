# DeepSeek Simulator

This is a simulator to mimic the possible performance of DeepSeek V3/R1 at different Hopper hardwares. 


测试硬件：
H800 80G
H20 96G
并行方式：
Attention DP ,  MoE EP
Attention TP+DP, MoE EP
Overlap 方式：
two-mircobatch overlapping （DeepSeek 官方）
single-batch compute-communication overlapping