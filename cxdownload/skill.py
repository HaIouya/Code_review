import requests
import re
import pandas as pd
from transformers import BertTokenizer

import concurrent.futures
import csv
import requests
from tqdm import tqdm
import argparse
import json


def call_large_model_api(user_prompt, user_input):
    url = 'https://u21829-a763-f63ca723.westc.gpuhub.com:8443/v1/chat/completions'  # 请替换为实际的API URL
    headers = {
        'Content-Type': 'application/json',
        # 请替换为实际的API密钥
    }
    data = {
        "model": "glm4",
        "stream": False,
        "temperature": 0.01,
        "max_tokens": 1024,
        "repetition_penalty": 10,
        "top_p": 0.01,
        "do_sample": False,
        "messages": [
            {
                "role": "system",
                "content": user_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    return response
# 处理每一行的函数
def process_rows(rows):
    processed_rows = []
    for row in rows:
        try:
            # 调用大模型API提取技术栈
            prompt = "你的任务是帮助用户从给定的岗位描述中提取技术栈。您将得到一个类似于[岗位描述1,...]的列表，然后你需要提取列表中的每个元素中的技能栈。我不需要你提供代码，你只需要提取，提取完成后，返回一个对应的list。规则：\n1. 必须精准，严肃 \n2.只生成json，其他一点都不要。\n3.返回格式必须是[[skill1,skill2...],[skill1,skill2...]]！！！"
            data = row[12]  # 假定这是包含工作描述的列
            response = call_large_model_api(prompt, data)
            response = response.json()
            skills = response["choices"][0]["message"]["content"]
            
            cleaned_skills_str = skills.replace("```json\n", "").replace("\n```", "").replace("\n", "")

            # 解析JSON字符串
            skill = json.loads(cleaned_skills_str)
            
            
        except Exception as e:
            print(f"Error processing row: {e}")
            skill = []

        # 将技能列表添加到原始行的末尾
        row.append(skill)
        processed_rows.append(row)

    return processed_rows

# 主函数
def main():
    # args = parse_args()

    # 打开输入和输出文件
    with open('Boss直聘/Boss直聘.csv', 'r', newline='', encoding='utf-8') as csvfile, \
         open('args.output.csv', 'w', newline='', encoding='utf-8') as output_file:

        reader = csv.reader(csvfile)
        writer = csv.writer(output_file)
        
        # 写入标题行
        writer.writerow(['职位名称', '工作地址','学历要求', '工作年限要求','招聘人数','薪资待遇','公司行业','公司性质','公司规模','融资阶段','招聘状态','职位类型','岗位描述','公司介绍','公司工商信息','简历详情页地址','更新日期','技术栈'])
        next(reader)  # 跳过标题行
        
        # 初始化一个列表来存储所有的行，以便并行处理
        rows = [row for row in reader]

        # 批量处理行数据
        batch_size = 5  # 假设每批处理50行
        processed_batches = []
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            processed_batches.append(batch)
        
        # 使用线程池并行处理数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            # 提交所有任务并获取future对象列表
            futures = [executor.submit(process_rows, batch) for batch in processed_batches]
            
            # 初始化计数器和批处理列表
            count = 0
            batch = []
            
            # 遍历future对象，获取结果并写入文件
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing rows", unit="batch"):
                # 结果处理
                processed_rows = future.result()
                batch.extend(processed_rows)
                count += len(processed_rows)
                
                # 每100行，写入文件并清空批处理列表
                if count % 100 == 0:
                    writer.writerows(batch)
                    batch.clear()

            # 写入剩余的数据
            if batch:
                writer.writerows(batch)

    print("数据处理完成！")

if __name__ == "__main__":
    main()