import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
wealth = {}  # 保存每个节点的财富
selection_counts = {}  # 保存每个节点被选中的次数
# 拟合幂函数
def power_law(x, a, b):
    return a * (x**b)


def calculate_wealth_ratio(wealth, selection_counts, percentage=0.2):
    # 前20%的人的数量
    number_of_high_wealth = int(percentage * selection_counts)
    # 获取字典中values最大的两个值
    # 获取字典中所有的财富值
    wealth_values = list(wealth.values())
    sum_wealthy = sum(wealth_values)
    # 获取最大的两个财富值
    top_two_wealth = sorted(wealth_values, reverse=True)[:number_of_high_wealth]
    # 求最大的两个财富值的和
    sum_of_top_two_wealth = sum(top_two_wealth)
    print(f'Top twenty percent sum_wealthy: {sum_of_top_two_wealth}')
    print(f'Others sum_wealthy: {sum_wealthy - sum_of_top_two_wealth}')
    print(f'前20%的人占据财富占总财富的比例为{sum_of_top_two_wealth*100 / sum_wealthy}%')

def calculate_sel_ratio(selection_counts_values, selection_counts,percentage=0.2):
    # 前20%的人的数量
    number_of_high_sel = int(percentage * selection_counts)
    # 获取字典中values最大的两个值
    # 获取字典中所有的财富值
    sel_values = list(wealth.values())
    sum_sel = sum(selection_counts_values)
    # 获取最大的两个财富值
    top_two_sel = sorted(selection_counts_values, reverse=True)[:number_of_high_sel]
    # 求最大的两个财富值的和
    sum_of_top_two_sel = sum(top_two_sel)
    print(f'Top twenty percent sum_wealthy: {sum_of_top_two_sel}')
    print(f'Others sum_wealthy: {sum_sel - sum_of_top_two_sel}')
    print(f'前20%的人被选择的次数占总次数的比例为{sum_of_top_two_sel * 100 / sum_sel}%')




def rich_get_richer_simulation_with_wealth(num_nodes, num_steps):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
        wealth[i] = 1.0  # 初始财富为1.0
        selection_counts[i] = 0  # 初始化被选中次数为0

    #print(G.nodes())
    for step in range(num_steps):
        # 选择一个节点，概率与财富成正比
        probabilities = np.array([wealth[node] for node in G.nodes()],dtype='float64')
        probabilities /= probabilities.sum()  # 归一化
        selected_node = np.random.choice(G.nodes(), p=probabilities)

        # 更新节点的财富和被选中次数
        # wealth[selected_node] *= 1.2-(0.2 / (1 + np.exp(-0.5 * (wealth[selected_node] - 100))))
        # wealth[selected_node] *=1.0025
        wealth[selected_node] += 1
        # 引入饱和度
        selection_counts[selected_node] += 1

        # 添加链接到选中节点
        G.add_edge(np.random.choice(G.nodes()), selected_node)

        # 绘制当前网络图
        '''plt.plot()
        pos = nx.spring_layout(G)
        node_sizes = [wealth[node] * 1000 for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', font_size=8)
        plt.show()'''
        # plt.title(f'Step {step + 1}')
        # plt.pause(0.1)
        # plt.clf()

    # print(G.edges())
    return selection_counts


num_nodes = 100
num_steps = 100000
# 模拟包含10个节点的网络，模拟100次链接添加过程
selection_counts = rich_get_richer_simulation_with_wealth(num_nodes, num_steps)

# 打印每个节点被选中的次数
for node, count in selection_counts.items():
    print(f'Node {node}: Selected {count} times.')

for i in range(num_nodes):
    print(f'Node {i}:wealthy :{wealth[i]} ')
calculate_wealth_ratio(wealth, num_nodes)

# 按照财富从高到低排名
sorted_wealth = sorted(wealth.items(), key=lambda x: x[1], reverse=True)

# 获取排名和财富数据
ranks = [i + 1 for i in range(len(sorted_wealth))]
wealth_values = [wealth for _, wealth in sorted_wealth]
print('wealth_values',wealth_values)
# 绘制函数图
plt.plot(ranks, wealth_values, marker='o', linestyle='-', color='b')
plt.xlabel('Rank')
plt.ylabel('Wealth')
plt.title('Ranking of Wealth')
# 保存图形到本地
plt.savefig('wealth_ranking_plot2.png')
plt.show()

# 导出数据到本地
np.savetxt('wealth_ranking_data2.csv', np.column_stack((ranks, wealth_values)), delimiter=',', header='Rank,Wealth', comments='')

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
plt.savefig('sel_ranking_plot2.png')
plt.show()

# 导出数据到本地
np.savetxt('sel_ranking_data2.csv', np.column_stack((ranks, selection_counts_values)), delimiter=',', header='Rank,sel_count', comments='')


# 使用 curve_fit 进行拟合
params, covariance = curve_fit(power_law, ranks, wealth_values)

# 获取拟合参数
a, b = params
std_a, std_b = np.sqrt(np.diag(covariance))

# 计算拟合值
fit_curve = power_law(np.array(ranks), a, b)

# 计算拟合误差
residuals = wealth_values - fit_curve

# 计算决定系数
mean_wealth = np.mean(wealth_values)
total_variance = np.sum((wealth_values - mean_wealth)**2)
explained_variance = np.sum((fit_curve - mean_wealth)**2)
r_squared = explained_variance / total_variance

# 计算均方根误差
rmse = np.sqrt(np.mean(residuals**2))

# 计算相对误差（Relative Error）
relative_errors = np.abs((wealth_values - fit_curve) / wealth_values)

# 计算平均相对误差（Mean Absolute Percentage Error，MAPE）
mape = np.mean(relative_errors) * 100

# 计算平均绝对误差（Mean Absolute Error，MAE）
mae = np.mean(np.abs(wealth_values - fit_curve))

# 输出拟合参数及相关指标
print(f'Fit parameters: a = {a:.2f} ± {std_a:.2f}, b = {b:.2f} ± {std_b:.2f}')
print(f'Residuals: {residuals}')
print(f'R-squared: {r_squared:.4f}')
print(f'RMSE: {rmse:.2f}')

# 输出指标
print(f'Relative Errors: {relative_errors}')
print(f'MAPE: {mape:.2f}%')
print(f'MAE: {mae:.2f}')

# 绘制函数图
plt.plot(ranks, wealth_values, marker='o', linestyle='-', color='b', label='Actual Wealth')
plt.plot(ranks, fit_curve, linestyle='--', color='r', label=f'Fit: $a x^b$, $a={a:.2f}$, $b={b:.2f}$')
plt.xlabel('Rank')
plt.ylabel('selection')
plt.title('Ranking of selection')
plt.legend()

# 保存包含拟合曲线的图形到本地
plt.savefig('selection_ranking_plot_with_fit2.png')

# 显示图形
plt.show()

x_log=np.log10(ranks)
y_log=np.log10(selection_counts_values)
plt.plot(x_log, y_log, marker='o', linestyle='-', color='b')
plt.xlabel('Rank_ln')
plt.ylabel('selection_counts_ln')
plt.title('log Ranking of selection counts')
plt.savefig('sel_ranking_plot_log.png')
plt.show()