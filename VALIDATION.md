# 验证记录

## 2026-11-03：用户在远程 910B3 上执行

- 环境：aarch64，CANN 9.0.0，Python 3.11.17，PyTorch 2.8.0+cpu，
  torch_npu 2.8.0.post4，Transformers 5.12.1，NumPy 1.26.4。
- 运行 `bash build.sh`，成功生成 `libarm_kernels.so` 和 `libarm_torch.so`。
- 启动封装中六处 `reinterpret_cast<GM_ADDR>` 修正为 `static_cast<uint8_t*>` 后编译通过。
- 真实测试结果：

```text
PASS: 51 NPU cases; three outputs, both rounding modes, tails, non-default stream
```

已验证：FP32/FP16/BF16、两种计算语义、三路输出、部分非对齐形状、
全零行、输入不变、输出不共享存储、非默认 stream。

尚待执行：完整219个单算子用例、性能测量、随机微型Qwen3集成测试、
预训练千问模型精度和性能对比。不得把快速用例通过等同于这些验证已经完成。

构建日志中的 Kineto 静态库 warning 未阻止编译。
`torch.zeros_like` 的内部格式 warning 未阻止上述测试通过。

## 本地检查

本机无 CANN/NPU，完成270组 NumPy 数学模拟、3项适配结构测试和 Python 静态语法检查。
这些检查不替代远端编译、NPU测试或模型验证。
