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
        ax.set_xticklabels(list(top_skills.keys()))
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()

def main():
    # 加载模型
    model = load_model()
    analyzer = JobSkillsAnalyzer('Boss直聘_skills.csv', model)
    analyzer.merge_similar_skills()
    radar_chart_drawer = RadarChartDrawer(analyzer.job_tec_json)

    root = tk.Tk()
    root.title("职业星图绘制")
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    def display_skills_to_draw(selected_job, N):
        # 销毁之前的排行榜窗口
        if hasattr(display_skills_to_draw, 'frame'):
            display_skills_to_draw.frame.destroy()
        
        # 创建一个新的框架用于显示排行榜
        display_skills_to_draw.frame = tk.Frame(root)
        display_skills_to_draw.frame.pack()

        # 获取前N个技能
        skills_to_display = analyzer.get_top_skills(selected_job, N)
        skills_text = f"Displaying the top {N} skills for job '{selected_job}':\n"
        for skill, count in skills_to_display.items():
            skills_text += f"{skill}: {count}\n"
        skills_label = tk.Label(display_skills_to_draw.frame, text=skills_text)
        skills_label.pack()

    # 在 plot_most_similar_radar_chart 函数中调用 display_skills_to_draw
    def plot_most_similar_radar_chart():
        user_input = job_dropdown.get()
        most_similar_job = analyzer.find_most_similar_job(user_input)
        #display_skills_to_draw(most_similar_job, N=10)
        radar_chart_drawer.plot_radar_chart({most_similar_job: analyzer.job_tec_json[most_similar_job]}, N=10)


    job_dropdown = ttk.Combobox(root, values=list(analyzer.job_tec_json.keys()))
    job_dropdown.pack()

    plot_button = tk.Button(root, text="Draw a star map", command=plot_most_similar_radar_chart)
    plot_button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()
