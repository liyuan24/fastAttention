# Flash Attention

Flash Attention paper: https://arxiv.org/abs/2205.14135. FLash attention is building on top of the online normalizer for softmax paper below. 

Online normalizer calculation for softmax paper: https://arxiv.org/abs/1805.02867.


## Why Flash Attention is faster
Attention matrix calculation needs softmax and numerical stable softmax also needs the max value for each row. So softmax needs both max and sum operations. Those operations are not element-wise operations. So standard attention needs multiple passes of reading from GPU High Bandwidth Memory (HBM) to GPU Chip SRAM. The contribution of Flash Attention is to get the max and sum in an online fashion to reduce the memory access to HBM.

## About this repo

This is contains the CPU implementation of Flash Attention and Online Normalizer for Softmax with Numpy. I added details comments to the code to explain the process.

## Future work

Of course, CUDA implementation.

