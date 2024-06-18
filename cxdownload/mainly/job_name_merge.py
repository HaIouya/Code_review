import argparse
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


###############
#待添加相似度写入
###############

# 设置命令行参数
parser = argparse.ArgumentParser(description='Process job skills data.')
parser.add_argument('--job_threshold', type=float, default=0.9, help='Threshold for job similarity merging.')
args = parser.parse_args()

print(f"正在使用相似度阈值 {args.job_threshold} 合并相同项：\n")

# 创建一个空的字典来存储每个职位的技术栈
job_skills_dict = defaultdict(list)
csv_path = 'Boss直聘_skills.csv'  # 假定文件已上传到这个路径
df = pd.read_csv(csv_path)

# 提取技术栈列中的每个技术，并转换为 JSON 格式
for index, row in df.iterrows():
    skills_str = row['技术栈']
    skills_list = [skill.split('. ', 1)[1].strip() if '. ' in skill else skill.strip() for skill in skills_str.split('\n') if skill.strip()]  # 清除空白字符并分割技术
    job_skills_dict[row['职位名称']].append(skills_list)

# 将技术列表转换为 JSON 格式，并统计每个技术的出现频率
job_tec_json = {}
for job_title, skills_list in job_skills_dict.items():
    skill_counts = defaultdict(int)
    for skills in skills_list:
        for skill in skills:
            skill_counts[skill] += 1
    job_tec_json[job_title] = dict(skill_counts)

print("相同项已合并!\n")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# 计算每个job_name的嵌入向量
tec = job_tec_json
job_names = list(tec.keys())
job_embeddings = model.encode(job_names)

# 计算job_name之间的相似度矩阵
job_similarities = model.similarity(job_embeddings, job_embeddings)

job_similarities_CLAMPED = np.clip(job_similarities, 0, 1)

# 将相似度矩阵转换为距离矩阵
distance_matrix = 1 - job_similarities_CLAMPED

# 使用DBSCAN进行聚类
db = DBSCAN(eps=1-args.job_threshold, min_samples=1, metric='precomputed', n_jobs=-1)
db.fit(distance_matrix)
# 获取聚类结果
labels = db.labels_

# 创建一个字典来存储每个聚类的职位名称
clustered_jobs = defaultdict(list)

# 将职位名称分配到对应的聚类中
for i, label in enumerate(labels):
    if label != -1:  # -1 表示噪声点，不分配到任何聚类
        clustered_jobs[label].append(job_names[i])

# 移除只包含一个职位的聚类，因为这些可能是噪声点
for label, jobs in list(clustered_jobs.items()):
    if len(jobs) == 1:
        del clustered_jobs[label]

# 创建一个字典来存储每个聚类的代表职位名称
representative_jobs = {label: jobs[0] for label, jobs in clustered_jobs.items()}

# 创建一个列表来存储所有原始行的DataFrame
all_jobs_df = []

# 遍历原始DataFrame，合并相似的job_name
for index, row in df.iterrows():
    main_job = row['职位名称']
    # 如果职位名称已经在聚类中，则使用聚类中的代表职位名称
    if main_job in clustered_jobs:
        cluster_label = next(label for label, jobs in clustered_jobs.items() if main_job in jobs)
        main_job = representative_jobs[cluster_label]
    # 将原始行添加到结果DataFrame
    all_jobs_df.append(main_job)

# 将所有原始行的DataFrame转换为DataFrame
df_all_jobs = pd.DataFrame(all_jobs_df)

# 将结果写入CSV文件
csv_file_name = f"clustered_jobs_with_{args.job_threshold}.csv"
df_all_jobs.to_csv(csv_file_name, index=False)

print(f"结果已写入文件 {csv_file_name}")