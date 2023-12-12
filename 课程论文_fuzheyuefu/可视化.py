import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
data = pd.read_csv('wealth_ranking_data.csv')
# 拟合幂函数
def power_law(x, a, b):
    return a * (x**b)

# 获取排名和财富数据
ranks = data['Rank']
wealth_values = data['Wealth']

# 按照选择次数从高到低排名
sorted_selection_counts = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)

# 获取排名和财富数据
ranks = [i + 1 for i in range(len(sorted_selection_counts))]
selection_counts_values = [selection_count for _, selection_count in sorted_selection_counts]
print('selection_counts_values',selection_counts_values)
calculate_sel_ratio(selection_counts_values, num_nodes)
# 绘制函数图
plt.plot(ranks, selection_counts_values, marker='o', linestyle='-', color='b')
plt.xlabel('Rank')
plt.ylabel('selection_counts')
plt.title('Ranking of selection counts')
plt.savefig('sel_ranking_plot.png')
plt.show()