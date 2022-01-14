import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("out.csv")

plt.figure(figsize=(8,8))
for i in range(10):
    plt.plot(data[:,1+(3*i)], data[:, 2+(3*i)])

plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.gca().set_aspect("equal")
plt.show()