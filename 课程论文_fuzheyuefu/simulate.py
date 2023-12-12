import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def rich_get_richer_simulation(num_nodes, num_steps):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)

    for step in range(num_steps):
        # 选择一个节点
        selected_node = np.random.choice(G.nodes())

        # 添加链接到选中节点
        G.add_edge(np.random.choice(G.nodes()), selected_node)

        # 绘制当前网络图
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8)
        plt.title(f'Step {step + 1}')
        plt.pause(0.1)
        plt.clf()


# 模拟包含10个节点的网络，模拟100次链接添加过程
rich_get_richer_simulation(10, 100)
