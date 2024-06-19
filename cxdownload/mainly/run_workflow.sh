#!/bin/bash

# 定义输入和输出文件参数
INPUT_FILE="Boss直聘.csv"
WORK_CONTENT_OUTPUT="Boss_work_content.csv"
WORK_REQUIRES_OUTPUT="Boss_work_requires.csv"
WORK_SKILLS_OUTPUT="Boss直聘_skills_qwen.csv"

# 第一个.py文件
python work_content.py --input "$INPUT_FILE" --output "$WORK_CONTENT_OUTPUT"

# 第二个.py文件
python work_requires.py --input "$WORK_CONTENT_OUTPUT" --output "$WORK_REQUIRES_OUTPUT"

# 第三个.py文件
python work_skills.py --input "$WORK_REQUIRES_OUTPUT" --output "$WORK_SKILLS_OUTPUT"

rm "$WORK_REQUIRES_OUTPUT"
rm "$WORK_CONTENT_OUTPUT"

echo "All scripts have been executed."
    