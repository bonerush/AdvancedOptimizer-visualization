import numpy as np
import matplotlib.pyplot as plt
import time
def function(x,y):
    fun = x**2-y**2
    return fun

def Gradfunction(x,y):
    return np.array([2*x, -2*y])

class AdancedOptimizer:
    def __init__(self,start,learning_rate,decay_rate,small_number,iteration):
        """
        start:起始点
        learning_rate:学习率
        decay_rate[]=[decay_rate_MGD,decay_rate_RMSProp,decay_rate_1,decay_rate_2]
        decay_rate_1:一阶矩估计的衰减率 
        decay_rate_2:二阶矩估计的衰减率
        decay_rate_RMSProp:RMSProp衰减率
        decay_rate_MGD:MGD衰减率
        learning_rate[]=[learning_rate_MGD,learning_rate_RMSProp,learning_rate_Adam]
        learning_rate_Adam:Adam学习率
        learning_rate_RMSProp:RMSProp学习率
        learning_rate_MGD:MGD学习率
        small_number:防止除零的小常数
        iteration:迭代次数
        """
        self.start = start
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.small_number= small_number
        self.iteration = iteration

    def Adam(self):
        m_x = 0
        m_y = 0 #初始化一阶矩向量
        v_x = 0
        v_y = 0 #初始化二阶矩向量
        g_x = 0
        g_y = 0  #初始化梯度
        path = [self.start]
        x, y = self.start
        for i in range(1,self.iteration+1):
            g_t = Gradfunction(x,y)
            #更新一阶矩估计
            m_x= self.decay_rate[2]*m_x + (1-self.decay_rate[2])*g_t[0]
            m_y= self.decay_rate[2]*m_y + (1-self.decay_rate[2])*g_t[1]
            #更新二阶矩估计
            v_x= self.decay_rate[3]*v_x + (1-self.decay_rate[3])*(g_t[0])**2
            v_y= self.decay_rate[3]*v_y + (1-self.decay_rate[3])*(g_t[1])**2
            #偏差矫正
            m_x_hat = m_x/(1-(self.decay_rate[2])**i)
            m_y_hat = m_y/(1-(self.decay_rate[2])**i)
            v_x_hat = v_x/(1-(self.decay_rate[3])**i)
            v_y_hat = v_y/(1-(self.decay_rate[3])**i)
            #更新参数
            x = x - self.learning_rate[2]*(m_x_hat/(np.sqrt(v_x_hat+self.small_number)))
            y = y - self.learning_rate[2]*(m_y_hat/(np.sqrt(v_y_hat+self.small_number)))         
        path.append((x, y))
        return np.array(path)
    
    def RMSProp(self):
        path = [self.start]
        x, y = self.start
        g_x = 0
        g_y = 0
        for i in range(self.iteration):
            grad_fun = Gradfunction(x,y)
            g_x = self.decay_rate[1]*g_x + (1-self.decay_rate[1])*(grad_fun[0])**2
            g_y = self.decay_rate[1] *g_y + (1-self.decay_rate[1])*(grad_fun[1])**2
            x = x - self.learning_rate[1]*grad_fun[0]/(np.sqrt(g_x + self.small_number))
            y = y - self.learning_rate[1]*grad_fun[1]/(np.sqrt(g_y + self.small_number))
            path.append((x, y))
        return np.array(path)
    
    def momentum_gradient_descent(self):
        path = [self.start]
        x, y = self.start
        v_x, v_y = 0, 0
        for _ in range(self.iteration):
            grad = Gradfunction(x,y)
            v_x = self.decay_rate[0] * v_x + grad[0]
            v_y = self.decay_rate[0] * v_y + grad[1]
            x -= self.learning_rate[0] * v_x
            y -= self.learning_rate[0] * v_y
            path.append((x, y))
        return np.array(path)
    
start_point = (-50, 2)
learning_rate = [0.15,0.15,0.15]
#learning_rate_MGD,learning_rate_RMSProp,learning_rate_Adam
decay_rate= [0.9,0.999,0.9,0.999]
#decay_rate_MGD,decay_rate_RMSProp,decay_rate_1,decay_rate_2
small_number = 1e-9
iterations = 35
#建立类
MyOptimizer = AdancedOptimizer(start_point,learning_rate,decay_rate,small_number,iterations)

# 测量RMSProp的时间
start_time_RMSP= time.time()
RMSProp_path = MyOptimizer.RMSProp()
RMSProp_time = time.time() - start_time_RMSP

# 测量动量梯度下降法的时间
start_time_MGD= time.time()
momentum_path = MyOptimizer.momentum_gradient_descent()
momentum_time = time.time() - start_time_MGD

# 测量Adam的时间
start_time_Adam = time.time()
Adam_path = MyOptimizer.Adam()
Adam_time = time.time() - start_time_Adam

# 打印运行时间
print(f"RMSProp Time: {RMSProp_time:.6f} seconds")
print(f"Momentum Gradient Descent Time: {momentum_time:.6f} seconds")
print(f"Adam Time: {Adam_time:.6f} seconds")

# 绘制等高线
range = 50 #范围
pointnum = 1000 #精度
x = np.linspace(-range, range, pointnum)
y = np.linspace(-range, range, pointnum)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 3)
contour = plt.contourf(X, Y, Z, levels=50, cmap='magma')
plt.colorbar(contour)
plt.plot(RMSProp_path[:, 0], RMSProp_path[:, 1], 'o-', label='Adam', color='red')
plt.legend()
plt.title('Adam')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-range, range])
plt.ylim([-range, range])
plt.grid(True)

plt.subplot(1, 3, 2)
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

plt.subplot(1, 3, 1)
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