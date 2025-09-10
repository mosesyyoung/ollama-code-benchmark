import time
import subprocess
from human_eval.data import read_problems
import psutil
import os
import re
import pynvml

# 要对比的模型列表（你可以改成自己本地有的）
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
]

# 添加全局客户端实例
import ollama
client = ollama.Client()

# 初始化pynvml
try:
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
    print("pynvml初始化成功，开始监控GPU显存")
except Exception as e:
    PYNVML_AVAILABLE = False
    print(f"pynvml初始化失败，将使用备用方法: {e}")

def get_gpu_memory_detail():
    """
    使用pynvml获取第一块GPU的显存使用详情（MB）
    返回: 已用显存(MB)
    """
    if not PYNVML_AVAILABLE:
        return get_gpu_memory()  #  fall back to nvidia-smi
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 第一块GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used // (1024 * 1024)  # 转换为MB
    except Exception as e:
        print(f"pynvml获取显存信息失败: {e}")
        return get_gpu_memory()  # 降级到nvidia-smi

def get_gpu_memory():
    """
    备用方法：调用nvidia-smi获取显存占用（MB）
    注意：nvidia-smi显示的是整个GPU的显存使用总量。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5  # 添加超时，防止命令挂起
        )
        # 获取第一行并转换为整数
        mem_str = result.stdout.strip().split('\n')[0]
        return int(mem_str)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"nvidia-smi获取显存失败: {e}")
        return -1  # 返回错误值

def extract_code_from_response(response_text):
    """从模型响应中提取Python代码块"""
    # 匹配 ```python ... ``` 或 ``` ... ``` 格式的代码块
    code_pattern = r'```(?:python)?\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果没有代码块格式，尝试提取函数定义
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith('def ') and not in_code:
            in_code = True
        if in_code:
            code_lines.append(line)
        if in_code and line.strip() == '' and len(code_lines) > 1:
            break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return response_text


def run_test(problem, model, max_tokens=512):
    """调用 ollama 生成代码并统计时间 & 显存"""
    #prompt = f"Write a Python function {problem['prompt']}"
    prompt = f"""Please generate complete Python code for this function. Only output the code, no explanations.
{problem['prompt']}
Remember to generate the complete function implementation."""

    # 获取推理前的显存使用量
    start_mem = get_gpu_memory_detail()
    start = time.time()

    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])

    duration = time.time() - start
    # 获取推理后的显存使用量
    end_mem = get_gpu_memory_detail()

    output = response["message"]["content"]
    #print(f"DEBUG - Raw output: \n{output}")
    code = extract_code_from_response(output)
    #print(f"DEBUG - Extracted code: \n{code}")

    tokens = len(output.split())
    speed = tokens / duration if duration > 0 else 0
    # 计算本次推理产生的显存变化
    mem_used = end_mem - start_mem if (start_mem >= 0 and end_mem >= 0) else -1

    return code, duration, speed, mem_used

def run_unit_test(code, test_code):
    """执行单元测试，判断是否正确"""
    try:
        full_code = code + "\n" + test_code
        #print(f"DEBUG - Full code: \n{full_code}")
        proc = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def benchmark(models, num_samples=0):
    problems = list(read_problems().items())
    if num_samples > 0:
        problems = problems[:num_samples]
    summary = []
    detailed_results = []

    for model in models:
        print(f"\n=== 开始测试模型: {model} ===")
        results = []
        model_results = []

        for task_id, problem in problems:
            code, duration, speed, mem_used = run_test(problem, model)
            ok = run_unit_test(code, problem["test"])
            results.append((ok, duration, speed, mem_used))
            model_results.append((task_id, ok, duration, speed, mem_used))
            print(f"{task_id}: {'✅ 通过' if ok else '❌ 失败'} | {duration:.2f}s | {speed:.2f} tok/s | 显存变化 {mem_used} MB")
            time.sleep(3)

        pass_rate = sum(1 for r in results if r[0]) / len(results)
        avg_speed = sum(r[2] for r in results) / len(results)
        avg_mem = sum(r[3] for r in results if r[3] >= 0) / max(1, sum(1 for r in results if r[3] >= 0))
        summary.append((model, pass_rate, avg_speed, avg_mem))
        detailed_results.append({"model": model, "results": model_results})

    # 输出对比表
    print("\n=== 对比结果 ===")
    print(f"{'模型':<20} {'通过率':<10} {'平均速度(tok/s)':<20} {'平均显存使用(MB)':<20}")
    for model, pr, sp, mem in summary:
        mem_str = f"{mem:.0f}" if mem >= 0 else "N/A"
        print(f"{model:<20} {pr*100:>6.1f}% {sp:>15.2f} {mem_str:>20}")

    # 新增：保存详细结果到CSV
    save_detailed_results(detailed_results)

def save_detailed_results(detailed_results, filename="benchmark_results.csv"):
    """保存详细结果到CSV文件"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '任务ID', '通过', '耗时(s)', '速度(tok/s)', '显存变化(MB)'])
        
        for model_data in detailed_results:
            model = model_data['model']
            for task_id, passed, duration, speed, mem in model_data['results']:
                writer.writerow([model, task_id, passed, f"{duration:.2f}", f"{speed:.2f}", mem])
    
    print(f"\n详细结果已保存到 {filename}")

if __name__ == "__main__":
    benchmark(MODELS, num_samples=3)

