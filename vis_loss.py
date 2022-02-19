import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('Loss-1000.txt')

data_len=len(data)
epp=np.linspace(0,data_len,data_len)


plt.plot(epp,data)
plt.show()