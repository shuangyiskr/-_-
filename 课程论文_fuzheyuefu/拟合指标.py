import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv('sel_ranking_data.csv')
# 拟合幂函数
def power_law(x, a, b):
    return a * (x**b)

def cal_r2(ranks,sel_values):
    params, covariance = curve_fit(power_law, ranks, sel_values)
    a, b = params
    std_a, std_b = np.sqrt(np.diag(covariance))
    # 计算拟合值
    fit_curve = power_law(np.array(ranks), a, b)
    # 计算决定系数
    mean_sel = np.mean(sel_values)
    total_variance = np.sum((sel_values - mean_sel) ** 2)
    explained_variance = np.sum((fit_curve - mean_sel) ** 2)
    r_squared = explained_variance / total_variance
    # 输出指标
    print(f'R2: {r_squared:.4f}')
    return r_squared

def cal_mape(ranks,sel_values):
    # 使用 curve_fit 进行拟合
    params, covariance = curve_fit(power_law, ranks, sel_values)
    a, b = params
    std_a, std_b = np.sqrt(np.diag(covariance))
    # 计算拟合值
    fit_curve = power_law(np.array(ranks), a, b)
    # 计算相对误差（Relative Error）
    relative_errors = np.abs((sel_values - fit_curve) / sel_values)

    # 计算平均相对误差（Mean Absolute Percentage Error，MAPE）
    mape = np.mean(relative_errors) * 100
    # 输出指标
    # print(f'Relative Errors: {relative_errors}')
    print(f'MAPE: {mape:.2f}%')
    return mape

def cal_rmse(ranks,sel_values):
    # 使用 curve_fit 进行拟合
    params, covariance = curve_fit(power_law, ranks, sel_values)
    a, b = params

    std_a, std_b = np.sqrt(np.diag(covariance))

    # 计算拟合值
    fit_curve = power_law(np.array(ranks), a, b)
    # 计算拟合误差
    residuals = sel_values - fit_curve
    # print(residuals)
    # 计算均方根误差
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f'MSE:{rmse:.4f}')
    return rmse


# 获取排名和财富数据
ranks = data['Rank']
sel_values = data['sel_count']
cal_r2(ranks,sel_values)
cal_mape(ranks,sel_values)
cal_rmse(ranks,sel_values)
params, covariance = curve_fit(power_law, ranks,sel_values)

# 获取拟合参数
a, b = params
std_a, std_b = np.sqrt(np.diag(covariance))
# 计算拟合值
fit_curve = power_law(np.array(ranks), a, b)
# 输出拟合参数及相关指标
print(f'Fit parameters: a = {a:.2f} ± {std_a:.2f}, b = {b:.2f} ± {std_b:.2f}')


#可视化
plt.plot(ranks, fit_curve, linestyle='--', color='r', label=f'Fit: $a x^b$, $a={a:.2f}$, $b={b:.2f}$')
plt.plot(ranks, sel_values, marker='o', linestyle='-', color='b')
plt.xlabel('Rank')
plt.ylabel('selection_counts')
plt.title('Ranking of selection counts')
plt.savefig('sel_ranking_plot_log.png')
plt.show()
print(sum(sel_values))

z1 = np.polyfit(ranks, sel_values, 2)
p1 = np.poly1d(z1)

print('----------------')
print('多项式拟合')
print(p1)

yvals = p1(ranks)
mean_sel = np.mean(sel_values)
total_variance = np.sum((sel_values - mean_sel) ** 2)
explained_variance = np.sum((yvals - mean_sel) ** 2)
r_squared = explained_variance / total_variance
# 输出指标
print(f'R2: {r_squared:.4f}')

# 计算相对误差（Relative Error）
relative_errors = np.abs((sel_values - yvals) / sel_values)

# 计算平均相对误差（Mean Absolute Percentage Error，MAPE）
mape = np.mean(relative_errors) * 100
# 输出指标
# print(f'Relative Errors: {relative_errors}')
print(f'MAPE: {mape:.2f}%')

# 计算拟合误差
residuals = sel_values - yvals
# print(residuals)
# 计算均方根误差
rmse = np.sqrt(np.mean(residuals ** 2))
print(f'RMSE:{rmse:.4f}')


plt.plot(ranks, sel_values, '*',label='original values',linestyle='-')
plt.plot(ranks, yvals, 'r',label='polyfit values')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend在图中的位置，类似象限的位置
plt.title('polyfitting')
plt.show()

x_log=np.log10(ranks)
y_log=np.log10(sel_values)
plt.plot(x_log, y_log, marker='o', linestyle='-', color='b')
plt.xlabel('Rank_ln')
plt.ylabel('selection_counts_ln')
plt.title('log Ranking of selection counts')
plt.savefig('sel_ranking_plot_log.png')
plt.show()

# 假设X和y是你的数据
X = x_log
y = y_log

X = np.reshape(X,(-1,1))
y = np.reshape(y,(-1,1))


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train)
#print(y_train)
# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差和R²分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('linear regression')
print('mse:',mse)
print('r2:',r2)

# 绘制散点图
plt.scatter(X_test, y_test, color='black', label='Actual data')

# 绘制线性回归线
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression')

# 添加标签和标题
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Linear Regression Model Visualization')

# 显示图例
plt.legend()

# 显示图形
plt.show()