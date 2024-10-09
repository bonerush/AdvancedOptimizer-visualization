import numpy as np
import matplotlib.pyplot as plt
import time

def function(x,y):
    fun = x**2-y**2
    return fun

def Gradfunction(x,y):
    return np.array([2*x, -2*y])

# RMSProp法
def RMSProp(start,learning_rate,decay_rate,small_number,iteration):
    path = [start]
    x, y = start
    loss_fun = function(x,y)
    g_x = 0
    g_y = 0
    for i in range(iteration):
        grad_fun = Gradfunction(x,y)
        g_x = decay_rate *g_x + (1-decay_rate)*(grad_fun[0])**2
        g_y = decay_rate *g_y + (1-decay_rate)*(grad_fun[1])**2
        x = x - learning_rate*grad_fun[0]/(np.sqrt(g_x + small_number))
        y = y - learning_rate*grad_fun[1]/(np.sqrt(g_y + small_number))
        path.append((x, y))
    return np.array(path)

# 动量梯度下降法
def momentum_gradient_descent(start, learning_rate, beta, iterations):
    path = [start]
    x, y = start
    v_x, v_y = 0, 0
    for _ in range(iterations):
        grad = Gradfunction(x,y)
        v_x = beta * v_x + grad[0]
        v_y = beta * v_y + grad[1]
        x -= learning_rate * v_x
        y -= learning_rate * v_y
        path.append((x, y))
    return np.array(path)
    
# 设置参数
start_point = (-25, 2)
learning_rate_MGD = 0.01
learning_rate_RMSProp = 0.15
decay_rate_MGD= 0.9
decay_rate_RMSProp= 0.99
small_number = 1e-9
iterations = 35

# 测量RMSProp的时间
start_time = time.time()
RMSProp_path = RMSProp(start_point, learning_rate_RMSProp, decay_rate_RMSProp,small_number,iterations)
RMSProp_time = time.time() - start_time

# 测量动量梯度下降法的时间
start_time = time.time()
momentum_path = momentum_gradient_descent(start_point, learning_rate_MGD , decay_rate_MGD, iterations)
momentum_time = time.time() - start_time

# 打印运行时间
print(f"RMSProp Time: {RMSProp_time:.6f} seconds")
print(f"Momentum Gradient Descent Time: {momentum_time:.6f} seconds")

# 绘制等高线
range = 25 #范围
pointnum = 1000 #精度
x = np.linspace(-range, range, pointnum)
y = np.linspace(-range, range, pointnum)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, Z, levels=50, cmap='magma')
plt.colorbar(contour)
plt.plot(RMSProp_path[:, 0], RMSProp_path[:, 1], 'o-', label='RMSProp', color='red')
plt.legend()
plt.title('RMSProp')
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
