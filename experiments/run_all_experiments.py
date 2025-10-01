# import yaml
# from commons import *
# import subprocess
#
# def run_command(command):
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()
#
#     if process.returncode != 0:
#         print(f"Command failed with error: {stderr.decode()}")
#         return False
#     else:
#         return True
#
# def main():
#     commands1 = [
#         "python3 experiment_datasets.py",
#         "python3 experiment_datasets_xcmp.py",
#         "python3 experiment_datasets_sortinghats.py",
#         "python3 experiment_synthetic.py",
#         f"mkdir results-{version_tag}/cmp/ && " \
#          "../src/build/cmp_bench 10 > results-{version_tag}/cmp/cmp_bench.csv",
#     ]
#
#     commands2 = [
#         "python3 visualization.py",
#         "python3 visualization_synthetic.py",
#         "python3 visualization_cmp.py",
#     ]
#
#     for command in commands1:
#         if not run_command(command):
#             return
#     print("All experiments done!")
#
#     for command in commands2:
#         if not run_command(command):
#             return
#     print("All visualizations done!")
#
# if __name__ == "__main__":
#     main()


import yaml
import os
from commons import *  # 假设 version_tag 和 WORKSPACE_DIR 在 commons.py 中定义
import subprocess
import shlex  # 推荐用于更安全的命令分割

# 确保 version_tag 是可用的。假设它在 commons.py 或全局定义
# 如果它不是全局可用的，您可能需要从 experiment.yaml 中加载它
try:
    with open('experiment.yaml', 'r') as f:
        config = yaml.safe_load(f)
        #version_tag = config.get('version_tag', 'default')  # 使用默认值以防万一
        #version_tag = config.get('version_tag')
except FileNotFoundError:
    print("Warning: experiment.yaml not found. Using default version_tag='default'.")
    version_tag = 'default'


def run_command(command):
    # 使用 shlex.split() 分割命令，增强安全性，并避免使用 shell=True
    # 但如果命令中包含管道或 I/O 重定向，可能仍然需要 shell=True
    # 对于本例中带 I/O 重定向的命令，保留 shell=True 更简单。

    # 对于简单的 Python 脚本调用，推荐：
    # process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 由于您的命令中有复杂的串联和重定向，我们保留 shell=True
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Command failed: {command}")
        print(f"Error output:\n{stderr.decode()}")
        return False
    else:
        print(f"Command succeeded: {command}")
        return True


def main():
    # --- 关键修复点：在运行命令前，确保目录结构存在 ---
    results_dir = f"results-{version_tag}/cmp"

    # 使用 os.makedirs 及其 exist_ok=True 参数，安全地创建目录
    # 这样就不会因为 "Directory nonexistent" 或 "File exists" 而失败
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Ensured results directory exists: {results_dir}")
    except PermissionError:
        print(f"Error: Permission denied when creating directory {results_dir}")
        return  # 权限问题，终止执行

    # 移除原命令中不安全的 `mkdir` 部分
    commands1 = [
        "python3 experiment_datasets.py",
        "python3 experiment_datasets_xcmp.py",
        "python3 experiment_datasets_sortinghats.py",
        "python3 experiment_synthetic.py",
        # 目录已在 Python 中创建，这里只运行基准测试并重定向输出
        f"../src/build/cmp_bench 10 > {results_dir}/cmp_bench.csv",
    ]

    commands2 = [
        "python3 visualization.py",
        "python3 visualization_synthetic.py",
        "python3 visualization_cmp.py",
    ]

    print("\n--- Running Experiments (Commands 1) ---")
    for command in commands1:
        if not run_command(command):
            return  # 任何一个命令失败，则终止
    print("\nAll experiments done!")

    print("\n--- Running Visualizations (Commands 2) ---")
    for command in commands2:
        if not run_command(command):
            return
    print("All visualizations done!")


if __name__ == "__main__":
    main()