import pandas as pd
import tkinter as tk
from tkinter import font
from tkinter import filedialog
from tkinter import ttk

# 设置Tkinter支持中文
font.family = 'SimHei'  # 使用黑体字体，这是一种常见的中文字体
font.size = 12  # 设置字体大小

def filter_dataframe(df, job_name=None, province=None, start_time=None, end_time=None, top_n=None, sort_by=None):
    # 应用过滤条件
    filter_conditions = pd.Series(True, index=df.index)
    
    # 如果提供了job_name，添加到过滤条件
    if job_name:
        filter_conditions &= df['职位名称'] == job_name
    
    # 如果提供了province，添加到过滤条件
    if province:
        filter_conditions &= df['省'] == province
    
    # 如果提供了start_time，添加到过滤条件
    if start_time:
        filter_conditions &= df['日期'] >= start_time
    
    # 如果提供了end_time，添加到过滤条件
    if end_time:
        filter_conditions &= df['日期'] <= end_time
    
    # 应用过滤条件
    filtered_df = df[filter_conditions]
    
    # 如果提供了sort_by，进行排序
    if sort_by and sort_by in df.columns:
        sorted_df = filtered_df.groupby(sort_by).size().reset_index(name='count').sort_values(by='count', ascending=False)
        
        # 如果提供了top_n，则返回前top_n个
        if top_n:
            return sorted_df.head(top_n)
        else:
            return sorted_df
    else:
        # 如果没有提供sort_by，直接返回过滤后的数据
        return filtered_df

def filter_and_export():
    # 获取用户输入
    job_name = combo_job_name.get()
    province = combo_province.get()
    start_time = entry_start_time.get()
    end_time = entry_end_time.get()
    sort_by = entry_sort_by.get()
    top_n = int(entry_top_n.get()) if entry_top_n.get().isdigit() else None
    
    # 过滤数据
    filtered_df = filter_dataframe(df, job_name, province, start_time, end_time, top_n, sort_by)
    
    # 显示结果
    if hasattr(display_skills_to_draw, 'frame'):
        display_skills_to_draw.frame.destroy()
    
    # 创建一个新的框架用于显示排行榜
    display_skills_to_draw.frame = tk.Frame(root)
    display_skills_to_draw.frame.pack()
    
    text_result = tk.Text(display_skills_to_draw.frame, height=10, width=50)
    text_result.pack()
    text_result.insert(tk.END, filtered_df.to_string(index=False))
    
    # 导出文件
    new_file_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
    if new_file_name:
        if new_file_name.endswith('.csv'):
            filtered_df.to_csv(new_file_name, index=False)
        elif new_file_name.endswith('.xlsx'):
            filtered_df.to_excel(new_file_name, index=False)

df = pd.read_csv('clustered_jobs_with_0.9.csv')
# 创建窗口
root = tk.Tk()
root.title("职位信息过滤")

# 设置窗口大小
root.geometry("600x400")

# 创建标签和输入框
label_job_name = tk.Label(root, text="职位名称:")
label_job_name.grid(row=0, column=0, sticky="e")
combo_job_name = ttk.Combobox(root)
combo_job_name.grid(row=0, column=1)

label_province = tk.Label(root, text="省:")
label_province.grid(row=1, column=0, sticky="e")
combo_province = ttk.Combobox(root)
combo_province.grid(row=1, column=1)

label_start_time = tk.Label(root, text="开始时间:")
label_start_time.grid(row=2, column=0, sticky="e")
entry_start_time = tk.Entry(root)
entry_start_time.grid(row=2, column=1)

label_end_time = tk.Label(root, text="结束时间:")
label_end_time.grid(row=3, column=0, sticky="e")
entry_end_time = tk.Entry(root)
entry_end_time.grid(row=3, column=1)

label_sort_by = tk.Label(root, text="排序依据:")
label_sort_by.grid(row=4, column=0, sticky="e")
entry_sort_by = tk.Entry(root)
entry_sort_by.grid(row=4, column=1)

label_top_n = tk.Label(root, text="前N名:")
label_top_n.grid(row=5, column=0, sticky="e")
entry_top_n = tk.Entry(root)
entry_top_n.grid(row=5, column=1)

# 更新Combobox的选项
combo_job_name['values'] = df['职位名称'].unique()
combo_province['values'] = df['省'].unique()

# 创建按钮
button_filter = tk.Button(root, text="过滤并排序", command=filter_and_export)
button_filter.grid(row=6, column=0, columnspan=2, pady=10)

# 创建文本框显示结果
text_result = tk.Text(root, height=10, width=50)
text_result.grid(row=7, column=0, columnspan=2)

# 运行主循环
root.mainloop()
