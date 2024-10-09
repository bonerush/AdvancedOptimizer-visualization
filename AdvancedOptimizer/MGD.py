import numpy as np
import matplotlib.pyplot as plt
import time

# 定义鞍点代价函数
def cost_function(x, y):
    return x**2 - y**2

# 计算梯度
def compute_gradient(x, y):
    return np.array([2*x, -2*y])

# 梯度下降法
def gradient_descent(start, learning_rate, iterations):
    path = [start]
    x, y = start
    for _ in range(iterations):
        grad = compute_gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path.append((x, y))
    return np.array(path)

# 动量梯度下降法
def momentum_gradient_descent(start, learning_rate, beta, iterations):
    path = [start]
    x, y = start
    v_x, v_y = 0, 0
    for _ in range(iterations):
        grad = compute_gradient(x, y)
        v_x = beta * v_x + grad[0]
        v_y = beta * v_y + grad[1]
        x -= learning_rate * v_x
        y -= learning_rate * v_y
        path.append((x, y))
    return np.array(path)

# 设置参数
start_point = (-25, 2)
learning_rate = 0.02
beta = 0.9
iterations = 50

# 测量梯度下降法的时间
start_time = time.time()
gd_path = gradient_descent(start_point, learning_rate, iterations)
gd_time = time.time() - start_time

# 测量动量梯度下降法的时间
start_time = time.time()
momentum_path = momentum_gradient_descent(start_point, learning_rate, beta, iterations)
momentum_time = time.time() - start_time

# 打印运行时间
print(f"Gradient Descent Time: {gd_time:.6f} seconds")
print(f"Momentum Gradient Descent Time: {momentum_time:.6f} seconds")

# 绘制等高线
range = 25 #范围
pointnum = 1000 #精度
x = np.linspace(-range, range, pointnum)
y = np.linspace(-range, range, pointnum)
X, Y = np.meshgrid(x, y)
Z = cost_function(X, Y)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, Z, levels=50, cmap='magma')
plt.colorbar(contour)
plt.plot(gd_path[:, 0], gd_path[:, 1], 'o-', label='Gradient Descent', color='red')
plt.legend()
plt.title('Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-range, range])
plt.ylim([-range, range])
plt.grid(True)

plt.subplot(1, 2, 2)
contour = plt.contourf(X, Y, Z, levels=50, linewidths=2,cmap='magma')
plt.colorbar(contour)
plt.plot(momentum_path[:, 0], momentum_path[:, 1], 'o-', label='Momentum Gradient Descent', color='blue')
plt.legend()
plt.title('Momentum Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-range, range])
plt.ylim([-range, range])
plt.grid(True)

plt.show()
