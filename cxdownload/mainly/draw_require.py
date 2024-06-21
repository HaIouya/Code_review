# -*- coding: utf-8 -*-


import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from matplotlib.font_manager import FontProperties
from tkinter import ttk
class JobSkillsAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.job_skills_dict = self.load_job_skills()
        self.job_tec_json = self.process_job_skills()

    def load_job_skills(self):
        job_skills_dict = defaultdict(list)
        df = pd.read_csv(self.csv_path)
        for index, row in df.iterrows():
            skills_str = row['任职资格']
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

    def merge_similar_skills(self, threshold=0.6):
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        for job_name, skills in self.job_tec_json.items():
            skill_names = list(skills.keys())
            skill_embeddings = model.encode(skill_names)
            similarities = model.similarity(skill_embeddings, skill_embeddings)

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
        # 在雷达图下方添加文本
        ax.text(0, -0.1, '技能排行榜:', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontproperties='SimHei')
        for rank, (skill, count) in enumerate(top_skills.items(), start=1):
            ax.text(0, -0.1 - rank * 0.05, f'{rank}. {skill}: {count}', transform=ax.transAxes, ha='center', va='center', fontsize=10, fontproperties='SimHei')
        plt.show()

        
def main():
    analyzer = JobSkillsAnalyzer('Boss直聘_skills.csv')
    analyzer.merge_similar_skills()
    radar_chart_drawer = RadarChartDrawer(analyzer.job_tec_json)
    
    # 设置字体
    font_family = 'SimSun'  # 宋体
    font_size = 12
    font = (font_family, font_size)

    root = tk.Tk()
    root.title("职业星图绘制") 
        
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    job_dropdown = ttk.Combobox(root, values=list(analyzer.job_tec_json.keys()))
    job_dropdown.pack()

    plot_button = tk.Button(root, text="绘制星图", command=lambda: radar_chart_drawer.plot_radar_chart({job_dropdown.get(): analyzer.job_tec_json[job_dropdown.get()]}, N=5))
    plot_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
