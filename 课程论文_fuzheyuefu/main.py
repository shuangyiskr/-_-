import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def rich_get_richer_simulation_with_wealth(num_nodes, num_steps):
    G = nx.DiGraph()
    wealth = {}  # 保存每个节点的财富
    selection_counts = {}  # 保存每个节点被选中的次数

    for i in range(num_nodes):
        G.add_node(i)
        wealth[i] = i + 1  # 财富等于节点的编号+1
        selection_counts[i] = 0  # 初始化被选中次数为0

    for step in range(num_steps):
        # 选择一个节点，概率与财富成正比
        probabilities = np.array([wealth[node] for node in G.nodes()],dtype='float64')
        probabilities /= probabilities.sum()  # 归一化
        selected_node = np.random.choice(G.nodes(), p=probabilities)

        # 更新节点的财富和被选中次数
        wealth[selected_node] *= 1.05
        selection_counts[selected_node] += 1

        # 添加链接到选中节点
        G.add_edge(np.random.choice(G.nodes()), selected_node)

        # 绘制当前网络图
        pos = nx.spring_layout(G)
        node_sizes = [wealth[node] * 100 for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', font_size=8)
        plt.title(f'Step {step + 1}')
        plt.pause(0.1)
        plt.clf()

    return selection_counts


# 模拟包含10个节点的网络，模拟100次链接添加过程
selection_counts = rich_get_richer_simulation_with_wealth(10, 100)

# 打印每个节点被选中的次数
for node, count in selection_counts.items():
    print(f'Node {node}: Selected {count} times.')
