import argparse
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN

# 设置命令行参数
parser = argparse.ArgumentParser(description='Process job skills data.')
parser.add_argument('--job_threshold', type=float, default=0.9, help='Threshold for job similarity merging.')
args = parser.parse_args()

# 创建一个空的字典来存储每个职位的技术栈
job_skills_dict = defaultdict(list)
csv_path = 'update_Boss_skills.csv'  # 假定文件已上传到这个路径
df = pd.read_csv(csv_path)

# 加载预训练的 SentenceTransformer 模型
model = SentenceTransformer("./paraphrase-multilingual-MiniLM-L12-v2")

# 提取技术栈列中的每个技术，并转换为 JSON 格式
for index, row in df.iterrows():
    skills_str = row['技术栈']
    skills_list = [skill.split('. ', 1)[1].strip() if '. ' in skill else skill.strip() for skill in skills_str.split('\n') if skill.strip()]
    job_skills_dict[row['职位名称']].append(skills_list)

# 将技术列表转换为 JSON 格式，并统计每个技术的出现频率
job_tec_json = {}
for job_title, skills_list in job_skills_dict.items():
    skill_counts = defaultdict(int)
    for skills in skills_list:
        for skill in skills:
            skill_counts[skill] += 1
    job_tec_json[job_title] = dict(skill_counts)

# 计算每个 job_name 的嵌入向量
tec = job_tec_json
job_names = list(tec.keys())
job_embeddings = model.encode(job_names)

# 计算job_name之间的相似度矩阵
job_similarities = model.similarity(job_embeddings, job_embeddings)

# 将相似度矩阵裁剪到 [0, 1] 范围内
job_similarities_CLAMPED = np.clip(job_similarities, 0, 1)

# 将相似度矩阵转换为距离矩阵
distance_matrix = 1 - job_similarities_CLAMPED

# 使用 DBSCAN 进行聚类
db = DBSCAN(eps=1-args.job_threshold, min_samples=1, metric='precomputed', n_jobs=-1)
db.fit(distance_matrix)

# 获取聚类结果
labels = db.labels_

# 创建一个字典来存储每个聚类的职位名称
clustered_jobs = defaultdict(list)

# 将职位名称分配到对应的聚类中
for i, label in enumerate(labels):
    clustered_jobs[label].append(job_names[i])

# 创建一个字典来存储每个聚类的代表职位名称
representative_jobs = {label: jobs[0] for label, jobs in clustered_jobs.items()}

# 创建一个列表来存储所有原始行的 DataFrame
all_jobs_df = []

# 遍历原始 DataFrame，合并相似的 job_name
for index, row in df.iterrows():
    main_job = row['职位名称']
    # 如果职位名称已经在聚类中，则使用聚类中的代表职位名称
    if main_job in job_skills_dict:  # 假设 job_skills_dict 包含了所有职位名称和它们的技能
        cluster_label = labels[job_names.index(main_job)]
        main_job = representative_jobs[cluster_label]
    # 将原始行添加到结果 DataFrame，确保更新了职位名称
    row['职位名称'] = main_job
    all_jobs_df.append(row)

# 将所有原始行的 DataFrame 转换为 DataFrame
df_all_jobs = pd.DataFrame(all_jobs_df)

# 将结果写入 CSV 文件
csv_file_name = f"clustered_jobs_with_{args.job_threshold}.csv"
df_all_jobs.to_csv(csv_file_name, index=False)

print(f"结果已写入文件 {csv_file_name}")
