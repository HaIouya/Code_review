# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# 创建一个函数来加载模型
def load_model():
    return SentenceTransformer("./paraphrase-multilingual-MiniLM-L12-v2")

class JobSkillsAnalyzer:
    def __init__(self, csv_path, model,skill_column):
        self.csv_path = csv_path
        self.model = model
        self.skill_column = skill_column  # 新增技能列名称参数
        self.job_skills_dict = self.load_job_skills()
        self.job_tec_json = self.process_job_skills()
        self.similarity_cache = {}

    def set_similarity_threshold(self, threshold):
        self.threshold = threshold
        
    def load_job_skills(self):
        job_skills_dict = defaultdict(list)
        df = pd.read_csv(self.csv_path)
        for index, row in df.iterrows():
            # 使用self.skill_column替换硬编码的'技术栈'
            skills_str = row[self.skill_column]
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
    def merge_similar_skills(self, threshold=0.85):
        for job_name, skills in self.job_tec_json.items():
            skill_names = list(skills.keys())
            skill_embeddings = self.model.encode(skill_names)
            similarities = self.model.similarity(skill_embeddings, skill_embeddings)

            merged_skills = set()
            for i, skill1 in enumerate(skill_names):
                for j, skill2 in enumerate(skill_names):
                    if j > i:
                        score = similarities[i][j]
                        if score > threshold:
                            if skill1 not in merged_skills:
                                self.job_tec_json[job_name][skill1] += self.job_tec_json[job_name][skill2]
                                merged_skills.add(skill2)

            for skill in merged_skills:
                del self.job_tec_json[job_name][skill]
    
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
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        plt.subplots_adjust(left=0.26, bottom=0.20, right=0.60, top=1, wspace=0.2, hspace=0.2)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        for i, (job, skills) in enumerate(selected_job_contents.items()):
            top_skills = dict(list(skills.items())[:N])
            values = list(top_skills.values())
            values += values[:1]
            angles_job = angles[:len(top_skills)] + angles[:1]
            ax.fill(angles_job, values, color=np.random.rand(3).tolist(), alpha=0.25)
            ax.plot(angles_job, values, color=np.random.rand(3).tolist(), linewidth=2, linestyle='solid', label=job)

        
        # 设置中文字体为'SimHei'
        ax.set_xticklabels(list(top_skills.keys()), fontproperties='SimHei')

        ax.set_yticklabels([])
        ax.set_xticks(angles[:len(top_skills)])
        ax.set_xticklabels(list(top_skills.keys()))
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        ax.text(0, -0.1, '技能排行榜:', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontproperties='SimHei')
        for rank, (skill, count) in enumerate(top_skills.items(), start=1):
            ax.text(0, -0.1 - rank * 0.05, f'{rank}. {skill}: {count}', transform=ax.transAxes, ha='center', va='center', fontsize=10, fontproperties='SimHei')
        
        plt.show()

def main():
    # 加载模型
    model = load_model()

    # 创建窗口
    root = tk.Tk()
    root.title("职业星图绘制")

    # 读取CSV文件并获取列名
    df = pd.read_csv('Boss直聘_skills.csv')
    column_names = df.columns.tolist()

    # 初始化分析器
    analyzer = JobSkillsAnalyzer('Boss直聘_skills.csv', model, '技术栈')
    analyzer.merge_similar_skills()

    # 创建下拉菜单的函数
    def create_column_combobox():
        column_label = tk.Label(root, text="关注列名称:")
        column_label.pack()
        column_combobox = ttk.Combobox(root, values=column_names)
        column_combobox.pack()
        column_combobox.set('技术栈')  # 设置默认值
        return column_combobox

    # 创建阈值输入框的函数
    def create_threshold_entry():
        threshold_label = tk.Label(root, text="相似度阈值:")
        threshold_label.pack()
        threshold_entry = tk.Entry(root)
        threshold_entry.pack()
        threshold_entry.insert(tk.END, '0.85')  # 设置默认值
        return threshold_entry

    # 更新分析器的函数
    def update_analyzer(column_combobox, threshold_entry):
        skill_column = column_combobox.get()
        threshold = float(threshold_entry.get())
        analyzer.skill_column = skill_column
        analyzer.threshold = threshold
        analyzer.job_skills_dict = analyzer.load_job_skills()
        analyzer.job_tec_json = analyzer.process_job_skills()
        analyzer.merge_similar_skills()

    # 绘制星图的函数
    def plot_most_similar_radar_chart(column_combobox, threshold_entry):
        update_analyzer(column_combobox, threshold_entry)  # 更新分析器

        user_input = job_dropdown.get()
        most_similar_job = analyzer.find_most_similar_job(user_input)
        
        radar_chart_drawer = RadarChartDrawer(analyzer.job_tec_json)
        radar_chart_drawer.plot_radar_chart({most_similar_job: analyzer.job_tec_json[most_similar_job]}, N=10)

    # 创建下拉菜单让用户选择职位
    job_dropdown = ttk.Combobox(root, values=list(analyzer.job_tec_json.keys()))
    job_dropdown.pack()

    # 创建绘制星图的按钮
    plot_button = tk.Button(root, text="绘制星图", command=lambda: plot_most_similar_radar_chart(column_combobox, threshold_entry))
    plot_button.pack()

    # 懒惰初始化GUI元素
    column_combobox = create_column_combobox()
    threshold_entry = create_threshold_entry()

    # 启动GUI
    root.mainloop()

if __name__ == "__main__":
    main()