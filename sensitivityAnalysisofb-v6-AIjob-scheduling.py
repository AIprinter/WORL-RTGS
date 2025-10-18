import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r"")

# 设置 seaborn 样式
sns.set(style="whitegrid")

# 使用 catplot 分面绘图
g = sns.catplot(
    data=df,
    x='b',
    y='Makespan',
    col='JobSize',          
    kind='box',
    col_order=['Small', 'Medium', 'Large'],  
    height=5,               
    aspect=1,               
    sharey=False            
)

# 添加标签
g.set_axis_labels("Parameter b", "Makespan")

# 去掉默认标题
g.set_titles("")

# 给每个子图加底部标题
titles = ['(a) Small Job', '(b) Medium Job', '(c) Large Job']
for ax, title in zip(g.axes.flat, titles):
    ax.set_xlabel(f"Parameter b\n {title}")   # 把标题挪到 x 轴标签下面

plt.tight_layout()
plt.show()
