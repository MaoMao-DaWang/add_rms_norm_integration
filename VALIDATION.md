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

随后用户运行 `python3 test_operator.py`，完整测试通过：

```text
PASS: 219 NPU cases; three outputs, both rounding modes, tails, non-default stream
```

已验证：FP32/FP16/BF16、两种计算语义、三路输出、部分非对齐形状、
全零行、epsilon 变化（1e-8 / 1e-6 / 0.5）、输入不变、输出不共享存储、非默认 stream。

尚待执行：性能测量、随机微型Qwen3集成测试、预训练千问模型精度和性能对比。
不得把单算子正确性通过等同于模型精度和性能验证已经完成。

### 首次微型模型测试失败，修复待复测

用户在内联微型Qwen3测试中遇到 `AssertionError: 生成 token 不一致`。
该脚本在检查 logits 前先检查 token，尚无 logits 差异数据，不能断言只是舍入误差。
检查发现启动封装使用 `getCurrentNPUStream().stream(false)`，跳过 TorchNPU 主机任务队列下发检查。
已改为 `.stream(true)`，以保证直接启动前的框架生产者任务已下发；不等待设备执行完成。
修复依据为 TorchNPU NPUStream 实现与 op-plugin 的 Ascend C 直调示例。
新增 `test_pipeline.py`，验证无中间CPU读取的连续生产者/消费者调用，共96个用例。
修复后编译、回归测试与微型模型结果尚待用户在NPU执行，未标记通过。

构建日志中的 Kineto 静态库 warning 未阻止编译。
`torch.zeros_like` 的内部格式 warning 未阻止上述测试通过。

## 本地检查

本机无 CANN/NPU，完成270组 NumPy 数学模拟、3项适配结构测试和 Python 静态语法检查。
这些检查不替代远端编译、NPU测试或模型验证。
