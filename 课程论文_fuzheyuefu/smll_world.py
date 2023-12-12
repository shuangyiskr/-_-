import networkx as nx
import random


def construct_small_world_model(N, K, p):
    G = nx.watts_strogatz_graph(N, K, p)
    return G


def random_walk_search(graph, start_node, steps):
    visited_nodes = [start_node]
    current_node = start_node
    for _ in range(steps):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = random.choice(neighbors)
            visited_nodes.append(current_node)
        else:
            break  # 如果当前节点没有邻居，终止搜索
    return visited_nodes


def evaluate_search_efficiency(graph, start_node, search_steps):
    # 计算平均路径长度
    average_path_length = nx.average_shortest_path_length(graph)

    # 计算聚类系数
    average_clustering = nx.average_clustering(graph)

    # 进行随机游走搜索
    visited_nodes = random_walk_search(graph, start_node, search_steps)

    # 计算搜索路径长度
    search_path_length = len(visited_nodes) - 1

    # 输出评价结果
    print("Evaluation Results:")
    print("Average Path Length:", average_path_length)
    print("Average Clustering Coefficient:", average_clustering)
    print("Search Path Length:", search_path_length)


# 参数设置
N = 100  # 节点数量
K = 6  # 每个节点的邻居数
p = 0.3  # 重连概率
start_node = 1  # 搜索起始节点
search_steps = 20  # 搜索步数

# 构建小世界模型
small_world_model = construct_small_world_model(N, K, p)

# 评估搜索效率
evaluate_search_efficiency(small_world_model, start_node, search_steps)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 假设X和y是你的数据
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差和R²分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('均方误差: ', mse)
print('R²分数: ', r2)
