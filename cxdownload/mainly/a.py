# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk

# 设置支持中文的字体
font = ('Noto Sans CJK', 12)  # 假设使用的是Windows系统的宋体字体

root = tk.Tk()
root.title("职业星图绘制")

# 创建一个输入框，并设置字体
job_entry = tk.Entry(root, font=font)
job_entry.pack()

# 创建一个按钮，并设置字体
plot_button = tk.Button(root, text="绘制星图", font=font)
plot_button.pack()

root.mainloop()
