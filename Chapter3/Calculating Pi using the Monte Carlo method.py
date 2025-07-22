import torch
import math
import matplotlib.pyplot as plt

def estimate_pi_mc(n_iteration):
    n_point_circle = 0
    pi_iteration = []
    for iteration in range(1,1 + n_iteration):
        point = torch.rand(2) * 2 - 1
        if torch.sqrt(point[0] ** 2 + point[1] ** 2) <= 1:
            n_point_circle += 1
        pi_iteration.append(4 * n_point_circle / iteration)
    plt.plot(pi_iteration)
    plt.plot([math.pi] * n_iteration,'--')
    plt.xlabel('Iteration')
    plt.ylabel('Estimate pi')
    plt.title('Estimation history')
    plt.show()
    print('Estimated value of pi is:', pi_iteration[-1])

n_point = 1000
points = torch.rand(n_point,2) * 2 - 1

n_point_circle = 0
points_circle = []

for point in points:
    r = torch.sqrt(point[0] ** 2 + point[1] ** 2)
    if r <= 1:
        #在里面
        n_point_circle += 1
        points_circle.append(point)

points_circle = torch.stack(points_circle)


#画图
plt.axes().set_aspect('equal')
plt.plot(points[:,0].numpy(),points[:,1].numpy(),'y.')
plt.plot(points_circle[:,0].numpy(),points_circle[:,1].numpy(),'c*')

i = torch.linspace(0,2 * math.pi,steps=10000)
plt.plot(torch.cos(i).numpy(),torch.sin(i).numpy())
#计算pi
pi_estimated = 4 * n_point_circle / n_point
print('Estimated value of pi is:', pi_estimated)
plt.show()

estimate_pi_mc(10000)

