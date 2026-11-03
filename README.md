# AddRmsNorm 开发与千问推理集成

目标环境：Ascend 910B3 / CANN 9.0.0 / PyTorch 2.8.0 / torch_npu 2.8.0.post4。
这是独立工程，支持前向推理；尚未实现 backward、TorchAir converter 或图模式适配。
本机没有 CANN/NPU；本地完成 CPU 数学模拟、适配代码结构测试和静态检查。
2026-11-03 用户已在远端完成编译及51个 NPU 快速用例，详见 `VALIDATION.md`。
完整用例和模型验证的结果仍需单独记录。

## 1. 上传、检查和构建

把整个 `add_rms_norm_integration` 目录上传到 `/mnt/workspace`，在远程终端执行：

```bash
cd /mnt/workspace/add_rms_norm_integration
python3 inspect_env.py
bash build.sh
python3 test_operator.py
```

若尚未加载 CANN 环境，先执行环境中的 `set_env.sh`。当前服务器路径为
`/home/developer/Ascend/cann-9.0.0`；`ASCEND_HOME_PATH` 应指向这个目录。
构建只使用本机已安装的 CANN、PyTorch 和 C++ 工具，不执行 pip 安装或模型下载。
`build.sh` 默认使用2个编译进程，可通过 `BUILD_JOBS` 调整。
如果构建报错，提供从第一个 error 开始的日志，以及 `inspect_env.py` 的输出。

构建产物为 `build/libarm_kernels.so` 和 `build/libarm_torch.so`。
接口层使用 PyTorch 分配 NPU 张量，并在 PyTorch 当前 stream 上调用 Ascend C 核函数。
不会创建私有 stream，不在每次调用时同步设备或进行 CPU 中转。

## 2. 算子接口和精度语义

```python
import torch
from arm_npu import add_rms_norm

with torch.inference_mode():
    y, rstd, residual_out = add_rms_norm(x, residual, weight, 1e-6)
```

底层注册名：`torch.ops.training_camp.add_rms_norm`。
输入为连续 NPU 张量，x/residual 同 shape、dtype，weight 的长度等于最后一维。
支持 rank 2～4、FP32/FP16/BF16、hidden 1～16384、最多33554432个元素。
y/residual_out 与输入 shape 和 dtype 一致；rstd 为 FP32，shape 为输入除最后一维。
输入不会被原地修改。启用梯度且输入或权重需要梯度时明确拒绝调用。

提供两种明确的语义：

| 参数 | 残差参与归一化 | 乘 weight 前的处理 | 用途 |
|---|---|---|---|
| `model_compatible=False` | FP32 相加结果，保留未回转的值 | FP32 乘 rstd 后直接乘 FP32 weight | 题目2的三路输出参考 |
| `model_compatible=True` | 残差相加后先回转输入 dtype，再提升到 FP32 | 归一化值先回转输入 dtype | 对齐普通 Qwen2/Qwen3 eager RMSNorm |

两种模式都计算并写出真实的三路输出。模型模式并非简单把 rstd 或残差省略。
归约树和 rsqrt/除法实现可能带来浮点差异，不能承诺端到端逐 bit 相同。
测试分别比较两个参考，报告明确的误差；模型验证还要求 greedy token 序列相同。

## 3. 单算子测试与计时

`test_operator.py` 验证三种 dtype、两种语义、rank 2～4、奇数宽度、2048以上尾块、
16384隐藏维、全零行、epsilon 变化、输入不变，以及非默认 stream。
残差输出要求精确匹配，FP32/rstd 使用1e-4级阈值，FP16/BF16使用相应低精度阈值。
这些是初始工程阈值，若课程另有标准，应按课程标准收紧并重新测试。

```bash
python3 bench_operator.py --rows 32 --hidden 1024 --dtype float16
python3 bench_operator.py --rows 1 --hidden 1024 --dtype float16 --model-compatible
python3 bench_operator.py --rows 128 --hidden 4096 --dtype bfloat16 --model-compatible
```

基线是同样计算语义的 PyTorch 组合操作，不代表已超过厂商融合算子。
输出同时报告每次调用的平均 wall time 和 NPU event 流间隔；Python提交、输出分配等
开销会影响结果，不能把流间隔当作单条硬件 kernel 的纯执行时间。纯 kernel 性能应另用 msprof。

## 4. 接入本地千问模型

没有预训练权重时，可以先执行随机初始化的两层 Qwen3 集成测试：

```bash
python3 test_qwen_tiny.py
python3 test_qwen_tiny.py --all-dtypes
```

这不访问网络，不需要 tokenizer 或 checkpoint。它比较融合前后的 prefill logits、
KV Cache 单步 decode、4个 greedy token，检查每层自定义算子调用，以及恢复原始 forward。
测试只证明这套集成路径能工作，不代表验证了预训练模型质量或真实大模型性能。
结果写入 `results/tiny_qwen_report.json`。

首版适配普通 `model_type=qwen2/qwen3` 的 causal LM；不默认适配 MoE、VL、Next 或 Qwen3.5。
用 `inspect_env.py` 查找已下载模型的 config.json，或手工提供模型目录。
如果老师没有指定模型，先使用课程已有的小模型完成验证；本工程不会自动下载权重。

```bash
python3 run_qwen.py --model /实际模型目录 --dtype float16 --tokens 32 --profile
```

该命令依次运行原始模型和替换后的模型，对比 logits、greedy token 序列和生成耗时。
模型以 eager attention 运行，且显式使用 KV Cache；固定生成 token 数以保证计时可比。
两种配置使用同一份权重，不重新训练模型。

适配器只替换每层的“attention输出与残差相加 → post_attention_layernorm”组合，
不把 Q/K norm 或单独的首个 RMSNorm 当作 AddRmsNorm。它通过 AST 精确匹配三个语句，
保留安装版本原有 attention/KV Cache 逻辑、forward 参数和返回形式。
找不到唯一匹配模式时会拒绝替换，且在修改任何层之前完成全部结构检查。
测试结束后恢复原始 forward。适配器只改当前模型实例，不修改 transformers 安装文件。

结果写入 `results/qwen_report.json`。报告包含匹配阈值、最大误差、greedy一致性、
自定义算子调用数、生成 token 数和速度；失败也会写报告并返回非零退出码。
默认 logits atol 为 FP32 0.001 / FP16 0.05 / BF16 0.2，rtol 为0.01。
这些阈值仅用于初始集成排查，不能替代课程或模型精度验收要求，可通过命令行显式修改。
若输出不一致，应先解决精度问题，再解释性能数据。

`--profile` 导出 CPU 调度 trace，便于确认 `training_camp::add_rms_norm` 被真实调用。
该 trace 不包含硬件流水分析；设备 kernel 时间需要 NPU profiler/msprof。
集成后不保证整体模型提速，收益需要在真实 prompt、prefill/decode 和 batch 下实测。

## 5. 本机可做的检查

```bash
python3 -B test_adapter_ast.py
```

该测试只使用 Python/NumPy，验证结构替换保留输出、参数、cache，以及对不匹配结构拒绝处理。
它不能替代 Transformers 5.12.1 的实际运行测试。

## 参考

- CANN 9.0 PyTorch kernel launch 与 torch.library 接入：
  https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0057.html
- 工程封装参考本仓库 `cann-learning-hub` 的 softmax/torch_library 示例。
- 模型语义与待替换位置参考 Hugging Face Transformers v5.12.1 Qwen3 实现：
  https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3/modeling_qwen3.py
