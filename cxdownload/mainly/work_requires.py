import concurrent.futures
import csv
import requests
from tqdm import tqdm
import argparse



def call_large_model_api(user_prompt, user_input):
    url = 'https://u21829-892d-10b0deb3.westc.gpuhub.com:8443/v1/chat/completions'  # 请替换为实际的API URL
    headers = {
        'Content-Type': 'application/json',
        # 请替换为实际的API密钥
    }
    data = {
        "model": "llama3",
        "stream": False,
        "temperature": 0.01,
        "top_p": 0.8,
        "do_sample": True,
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

def parse_args():
    parser = argparse.ArgumentParser(description='Process CSV files.')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    return parser.parse_args()


# 处理每一行的函数
def process_row(row):
    # 处理行的逻辑，例如提取入职要求和工作内容
    data = row[12]  # 假定这是包含工作描述的列
    try:
        # 调用大模型API提取技术栈
        prompt = "从我给你们的岗位描述中摘录出来任职资格或者任职要求，注意不是岗位职责。返回格式：\n1. 任职资格1\n2. 任职资格2\n3. ..."
        response = call_large_model_api(prompt, data)
        skill = response.json()
        skills = skill["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error processing row: {e}")
        skills = ""
    
    row.append(skills)
    return row

# 主函数
def main():
    args = parse_args()

    # 打开输入和输出文件
    with open(args.input, 'r', newline='', encoding='utf-8') as csvfile, \
         open(args.output, 'w', newline='', encoding='utf-8') as output_file:

        reader = csv.reader(csvfile)
        writer = csv.writer(output_file)
        
        # 写入标题行
        writer.writerow(['职位名称', '工作地址','学历要求', '工作年限要求','招聘人数','薪资待遇','公司行业','公司性质','公司规模','融资阶段','招聘状态','职位类型','岗位描述','公司介绍','公司工商信息','简历详情页地址','更新日期','工作内容','任职资格'])
        next(reader)  # 跳过标题行
        
        # 初始化一个列表来存储所有的行，以便并行处理
        rows = [row for row in reader]
        
        # 使用线程池并行处理数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            # 提交所有任务并获取future对象列表
            futures = [executor.submit(process_row, row) for row in rows]
            
            # 初始化计数器和批处理列表
            count = 0
            batch = []
            
            # 遍历future对象，获取结果并写入文件
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing rows", unit="row"):
                # 结果处理
                processed_row = future.result()
                batch.append(processed_row)
                count += 1
                
                # 每100行，写入文件并清空批处理列表
                if count % 100 == 0:
                    writer.writerows(batch)
                    batch.clear()

# 运行主函数
if __name__ == "__main__":
    main()