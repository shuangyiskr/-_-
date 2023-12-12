import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x, k=0.5, x0=50):
    return 1.1-(0.1 / (1 + np.exp(-k * (x - x0))))
# 生成一些 x 值
x_values = np.linspace(30, 70, 300)

# 计算对应的 Logistic 函数值
y_values = logistic_function(x_values)

# 绘制 Logistic 函数图像
plt.plot(x_values, y_values, label='Logistic Function')
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
