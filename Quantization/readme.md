

| Paper | Article | Note |
|-------|---------|------|
|LLM.int8()|[link](https://huggingface.co/blog/hf-bitsandbytes-integration)|   |



---
## Floating Point Formats for Model Quantization in Deep Learning

This table summarizes various floating-point formats used in deep learning, particularly in the context of model quantization, precision, software support, and hardware compatibility.

| Format       | Size                          | Structure                                     | Range                         | Usage in DL                                                        | Software Support                                                                 | Hardware Support                                                                      |
|--------------|-------------------------------|------------------------------------------------|-------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **FP80**     | 10 bytes                      | 1 bit sign, 15 bits exponent, 64 bits fraction | ~\(3.65 \times 10^{-4951}\) to ~\(1.18 \times 10^{4932}\)   | Not used in DL                                                     | Not typically supported in DL frameworks                                          | Supported by x86 CPUs (x87 instruction subset)                                          |
| **FP64**     | 8 bytes                       | 1 bit sign, 11 bits exponent, 52 bits fraction | ~\(2.23 \times 10^{-308}\) to ~\(1.80 \times 10^{308}\)     | Rarely used due to high precision and slow computation             | TensorFlow (tf.float64), PyTorch (torch.float64)                                      | Limited support in most GPUs; better support in high-end GPUs like NVIDIA Tesla V100, A100 |
| **FP32**     | 4 bytes                       | 1 bit sign, 8 bits exponent, 23 bits fraction  | ~\(1.18 \times 10^{-38}\) to ~\(3.40 \times 10^{38}\)       | Standard for DL training                                           | TensorFlow (tf.float32), PyTorch (torch.float32)                                   | Widely supported in CPUs and GPUs (NVIDIA, AMD)                                          |
| **FP16**     | 2 bytes                       | 1 bit sign, 5 bits exponent, 10 bits fraction  | ~\(5.96 \times 10^{-8}\) to ~\(6.55 \times 10^4\)           | Used in mixed-precision training and post-training quantization    | TensorFlow (tf.float16), PyTorch (torch.float16)                                    | Supported in modern GPUs (NVIDIA RTX series)                                             |
| **BFLOAT16** | 2 bytes                       | 1 bit sign, 8 bits exponent, 7 bits fraction   | ~\(1.18 \times 10^{-38}\) to ~\(3.40 \times 10^{38}\)       | Replacing FP16 for better range without loss scaling               | TensorFlow (tf.bfloat16), PyTorch (torch.bfloat16)                                  | Supported in NVIDIA A100, Google TPU v2/v3, Intel Xeon (AVX-512 BF16)                    |
| **TF32**     | 2.375 bytes (~2.4 bytes)      | 1 bit sign, 8 bits exponent, 10 bits fraction  | ~\(1.18 \times 10^{-38}\) to ~\(3.40 \times 10^{38}\)       | Faster computation in A100 GPUs; easy switch from FP32             | Supported in CUDA 11                                                               | Supported in NVIDIA A100                                                                |
| **INT8**     | 1 byte                        | Integer                                        | -128 to 127                   | Post-training quantization                                         | TensorFlow Lite, PyTorch                                                          | Supported on most modern GPUs and specialized accelerators                              |
| **INT4**     | 0.5 bytes                     | Integer                                        | -8 to 7                       | Extreme quantization (experimental)                                | TensorFlow Lite (experimental), custom libraries                                   | Limited support, mainly in research hardware setups                                      |
| **INT1**     | 0.125 bytes (1/8 byte)        | Binary                                         | 0, 1                          | Binary Neural Networks (BNNs)                                      | Custom libraries                                                                  | Specialized hardware (e.g., FPGA, custom ASICs)                                         |

### Key Takeaways:
- **FP16 and BFLOAT16** are gaining traction for mixed-precision training and quantization due to their balance of range and precision.
- **TF32** is a new format introduced by NVIDIA that allows easy transition from FP32 to achieve faster computation with minimal changes in DL frameworks.
- **INT8, INT4, and INT1** are used for more aggressive quantization, mainly in inference scenarios to achieve higher speeds and lower memory usage.
