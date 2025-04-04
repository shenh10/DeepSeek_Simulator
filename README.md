# DeepSeek Simulator

This simulator emulates the potential performance of DeepSeek V3/R1 across various NVIDIA Hopper architectures. For compatibility with this tool, other hardware platforms need to implement their respective GEMM/MLA kernels.


More info about this tool is show in my Chinese blog:
- [DeepSeek V3/R1 推理效率分析（1）：关于DeepSeek V3/R1 Decoding吞吐极限的一些不负责任估计](https://zhuanlan.zhihu.com/p/27292649125)
- [DeepSeek V3/R1 推理效率分析（2）: DeepSeek 满血版逆向工程分析](https://zhuanlan.zhihu.com/p/29841050824)
- [DeepSeek V3/R1 推理效率分析（3）：Decode 配置泛化讨论](https://zhuanlan.zhihu.com/p/29540042383)

## Installation

### Requirements
Follow DeepGemm, it requires:
- Hopper architecture GPUs, sm_90a must be supported
- Python 3.8 or above
- CUDA 12.3 or above
- PyTorch 2.1 or above
- CUTLASS 3.6 or above (could be cloned by Git submodule)

```bash
# omit install torch

# install FlashMLA
git clone  --recursive https://github.com/deepseek-ai/FlashMLA.git
python setup.py install

# install DeepGemm
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
python setup.py install

```
## Features
### Hardware Supported
- H800 80G(tested)
- H20 96G(tested)
- Other Hopper architectures should be working

### Parallel Method：
- Attention DP ,  MoE EP
- Attention TP+DP, MoE EP
  
### Overlap Method：
- two-mircobatch overlapping （DeepSeek Official）
- single-batch compute-communication overlapping

## Results
### H800
- H800 80G with two-mircobatch overlapping
  ![H800_two_microbatch_overlapping_results](./figures/H800_two_microbatch_overlapping_results.png)

- H800 80G with single-batch compute-communication overlapping
    ![H800_single_batch_comp_comm_overlapping_results](./figures/H800_single_batch_comp_comm_overlapping_results.png)

### H20
- H20 96G with two-mircobatch overlapping
  ![H20_two_microbatch_overlapping_results](./figures/H20_two_microbatch_overlapping_results.png)
- H20 96G with single-batch compute-communication overlapping
  ![H20_single_batch_comp_comm_overlapping_results](./figures/H20_single_batch_comp_comm_overlapping_results.png)

## License
This code repository is released under the [MIT License](./LICENCE).

## Citation
```bibtex
@misc{deepseek_simulator,
      title={DeepSeek-Simulator: A test-based Performance Simulator for DeepSeek V3/R1}, 
      author={Han Shen},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/shenh10/DeepSeek_Simulator.git}},
}
```