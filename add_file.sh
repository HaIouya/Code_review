#!/bin/bash

# 设置最大文件大小为 100MB
MAX_SIZE=100000000

# 使用 find 命令查找所有大小小于或等于 100MB 的文件
# -type f 表示只查找文件，不包括目录
# -size -${MAX_SIZE}c 表示文件大小小于或等于 MAX_SIZE 字节
# -exec git add {} \; 表示对每个找到的文件执行 git add 命令
find . -type f -size -${MAX_SIZE}c -exec git add {} \;

