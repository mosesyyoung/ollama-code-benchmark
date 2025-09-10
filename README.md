# Ollama Code Benchmark

一个简单的 **代码能力基准测试工具**，用于在本地 GPU 上评估不同大模型（如 Qwen、CodeLlama、DeepSeek 等）的 **代码生成正确率、速度和显存占用**。

## ✨ 功能
- 使用 [HumanEval](https://github.com/openai/human-eval) 数据集测试代码正确性  
- 自动统计 **通过率、平均推理速度、显存占用**  
- 结果保存为 `benchmark_results.csv`，便于对比分析  

## 🚀 安装
```bash
git clone https://github.com/mosesyyoung/ollama-code-benchmark.git
cd ollama-code-benchmark

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
ollama
human-eval
psutil
nvidia-ml-py
```

⚠️ 需要本地已安装 [Ollama](https://ollama.ai) 并能正常调用模型。

## 📊 使用方法
运行基准测试：
```bash
python benchmark_main.py
```

默认会测试 `benchmark_main.py` 中配置的模型列表，每个模型运行 3 个 HumanEval 测试用例。  
你可以修改：
```python
MODELS = [
    "codellama:7b",
    "codellama:13b",
    "deepseek-coder-v2:16b",
    "deepseek-coder:6.7b",
    "deepseek-r1:14b",
    "llama3.2:latest",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:latest",
    "qwen3:14b",
    "qwen3-coder:30b",
    "yi:9b",
    "yi-coder:9b",
]                                   # 只需要保留自己·ollama list·中的且想测试的模型
```
```python
benchmark(MODELS, num_samples=10)  # 调整任务数量，设为0则是跑整个测试集，当前是163个测试测试用例
```

## 📈 输出结果示例
运行后会在终端输出表格：

```
模型                 通过率     平均速度(tok/s)     平均显存使用(MB)
qwen2.5-coder:14b     35.0%           22.5               820
codellama:7b          28.0%           18.2               750
```

同时生成 `benchmark_results.csv`，包含每个任务的详细结果。

## 📌 注意事项
- 建议在显存 ≥12GB 的 GPU 上运行  
- 如果没有 `nvidia-ml-py`，脚本会自动使用 `nvidia-smi` 获取显存  
- HumanEval 默认只抽样部分任务，你可根据需要修改  

---

💡 欢迎贡献新的评测方法和模型配置！

