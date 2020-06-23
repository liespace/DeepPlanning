from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def g(yy ,aa):
    return np.abs(np.log(yy/(1-yy)) - np.log(aa / (1-aa)))

def f(yy, aa):
    return 4 * np.abs(yy - aa)

#y = np.linspace(0, 1, 30)[1:-1]
#a = np.linspace(0, 1, 30)[1:-1]
#Y, A = np.meshgrid(y, a)
#G = g(Y, A)
#F = f(Y, A)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_wireframe(Y, A, G)
#ax.plot_wireframe(Y, A, F)
#ax.scatter3D(ydata, adata, f(ydata, adata))
#ax.scatter3D(ydata, adata, g(ydata, adata), color='r', s=50)

mid = 0.7
gap = 0.2
a = np.linspace(mid-gap, mid+gap, 20)
y = np.array([mid]*len(a))
print(g(y,a))
plt.plot(a-mid, g(y, a), 'r')
plt.plot(a-mid, f(y, a), 'g')

mid = 0.5
gap = 0.2
a = np.linspace(mid-gap, mid+gap, 20)
y = np.array([mid]*len(a))
print(g(y,a))
plt.plot(a-mid, g(y, a), 'r')
plt.plot(a-mid, f(y, a), 'g')


plt.show()
