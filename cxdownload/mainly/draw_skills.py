# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.font_manager import findSystemFonts, FontProperties
from sklearn.cluster import DBSCAN

# 创建一个函数来加载模型
def load_model():
    return SentenceTransformer("./paraphrase-multilingual-MiniLM-L12-v2")

class JobSkillsAnalyzer:
    def __init__(self, csv_path, model):
        self.csv_path = csv_path
        self.model = model
        self.job_skills_dict = self.load_job_skills()
        self.job_tec_json = self.process_job_skills()

    def load_job_skills(self):
        job_skills_dict = defaultdict(list)
        df = pd.read_csv(self.csv_path)
        for index, row in df.iterrows():
            skills_str = row['技术栈']
            skills_list = [skill.split('. ', 1)[1].strip() if '. ' in skill else skill.strip() for skill in skills_str.split('\n') if skill.strip()]
            job_skills_dict[row['职位名称']].append(skills_list)
        return job_skills_dict

    def process_job_skills(self):
        job_tec_json = {}
        for job_title, skills_list in self.job_skills_dict.items():
            skill_counts = defaultdict(int)
            for skills in skills_list:
                for skill in skills:
                    skill_counts[skill] += 1
            job_tec_json[job_title] = dict(skill_counts)
        return job_tec_json

    def get_top_skills(self, job_title, N):
        # 获取前N个技能
        skills = self.job_tec_json[job_title]
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_skills[:N])

    def merge_similar_skills(self, threshold=0.6):
        for job_name, skills in self.job_tec_json.items():
            skill_names = list(skills.keys())
            skill_embeddings = self.model.encode(skill_names)
            
            # 计算相似度矩阵并裁剪到 [0, 1] 范围内
            similarities = np.clip(self.model.similarity(skill_embeddings, skill_embeddings), 0, 1)

            # 将相似度矩阵转换为距离矩阵
            distance_matrix = 1 - similarities
            
            # 使用 DBSCAN 进行聚类
            db = DBSCAN(eps=1-threshold, min_samples=1, metric='precomputed', n_jobs=-1)
            db.fit(distance_matrix)
            
            # 合并聚类结果中的相似技能
            labels = db.labels_
            merged_skills = {}
            
            for label in set(labels):
                indices = [index for index, lab in enumerate(labels) if lab == label]
                if len(indices) > 1:  # Ensure there is more than one skill in the cluster
                    merged_skill_name = ' & '.join(np.array(skill_names)[indices])
                    merged_skill_value = sum([self.job_tec_json[job_name][skill_names[idx]] for idx in indices])
                    merged_skills[merged_skill_name] = merged_skill_value
                else:  # If only one skill, keep it as is
                    idx = indices[0]
                    merged_skills[skill_names[idx]] = self.job_tec_json[job_name][skill_names[idx]]
            
            # 更新 job_tec_json
            self.job_tec_json[job_name] = merged_skills

    def find_most_similar_job(self, user_input):
        job_titles = list(self.job_tec_json.keys())
        job_embeddings = self.model.encode(job_titles)
        user_embedding = self.model.encode([user_input])

        # 计算用户输入与每个岗位名称的相似度
        similarities = self.model.similarity(user_embedding, job_embeddings)

        # 找到最相似的岗位名称
        most_similar_index = np.argmax(similarities)
        most_similar_job = job_titles[most_similar_index]

        return most_similar_job

class RadarChartDrawer:
    def __init__(self, normalized_data):
        self.normalized_data = normalized_data

    def plot_radar_chart(self, selected_job_contents, N=5):
        
        # 查找 SimHei 字体路径
        simhei_font_path = 'C:\Windows\Fonts\SIMHEI.TTF'
        for font in findSystemFonts(fontpaths=None, fontext='ttf'):
            if 'SimHei' in font:
                simhei_font_path = font
                break

        if not simhei_font_path:
            raise Exception("未找到 SimHei 字体，请确保系统中安装了该字体。")

        font = FontProperties(fname=simhei_font_path)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.80, top=0.85, wspace=0.2, hspace=0.2)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        for i, (job, skills) in enumerate(selected_job_contents.items()):
            top_skills = dict(list(skills.items())[:N])
            values = list(top_skills.values())
            values += values[:1]
            angles_job = angles[:len(top_skills)] + angles[:1]
            ax.fill(angles_job, values, color=np.random.rand(3).tolist(), alpha=0.25)
            ax.plot(angles_job, values, color=np.random.rand(3).tolist(), linewidth=2, linestyle='solid', label=job)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:len(top_skills)])
        ax.set_xticklabels(list(top_skills.keys()), fontproperties=font)

        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), prop=font)
        
        sorted_top_skills = dict(sorted(top_skills.items(), key=lambda item: item[1], reverse=True))

        # 绘制技能排行榜标题
        ax.text(1.1, 1.0, '技能排行榜:', transform=ax.transAxes, ha='left', va='center', fontsize=12, fontproperties=font)

        # 根据排序后的结果绘制技能排行榜
        for rank, (skill, count) in enumerate(sorted_top_skills.items(), start=1):
            ax.text(1.1, 1.0 - rank * 0.05, f'{rank}. {skill}: {count}', transform=ax.transAxes, ha='left', va='center', fontsize=10, fontproperties=font)

        plt.show()

def main():
    # 加载模型
    model = load_model()
    analyzer = JobSkillsAnalyzer('Boss/Boss直聘_skills.csv', model)
    analyzer.merge_similar_skills()
    radar_chart_drawer = RadarChartDrawer(analyzer.job_tec_json)

    root = tk.Tk()
    root.title("职业星图绘制")
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # 在 plot_most_similar_radar_chart 函数中调用 display_skills_to_draw
    def plot_most_similar_radar_chart():
        user_input = job_dropdown.get()
        most_similar_job = analyzer.find_most_similar_job(user_input)
        
        radar_chart_drawer.plot_radar_chart({most_similar_job: analyzer.job_tec_json[most_similar_job]}, N=10)

    job_dropdown = ttk.Combobox(root, values=list(analyzer.job_tec_json.keys()))
    job_dropdown.pack()

    plot_button = tk.Button(root, text="绘制星图", command=plot_most_similar_radar_chart)
    plot_button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()