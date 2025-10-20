#!/bin/bash
echo "正在运行第一个文件..."
python3 visualization.py

echo "正在运行第二个文件..."
python3 visualization_cmp.py

echo "正在运行第三个文件..."
python3 visualization_synthetic.py

echo "所有文件运行完毕。"