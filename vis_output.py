import pickle 
import numpy as np
import matplotlib.pyplot as plt


print("Loading datas ...")
with open('dataset.pkl', 'rb') as f:
    map_dataset = pickle.load(f) 

data_size=len(map_dataset)
train_data=np.zeros((data_size,7))
label=np.zeros((data_size,10))
for i in range(data_size):
    train_data[i,:]=np.array([map_dataset[i][0][1][0],map_dataset[i][0][1][1],map_dataset[i][0][1][2],map_dataset[i][0][1][3],map_dataset[i][0][3][0],map_dataset[i][0][3][1],map_dataset[i][0][3][2]])
    label[i,:]=np.array([map_dataset[i][1][0][1],map_dataset[i][1][0][2],map_dataset[i][1][0][3],map_dataset[i][1][1][1],map_dataset[i][1][1][2],map_dataset[i][1][1][3],map_dataset[i][1][2][1],map_dataset[i][1][2][2],map_dataset[i][1][2][3],map_dataset[i][1][3][1]])

indexs=np.linspace(0,data_size,data_size)

for i in range(10):
    plt.plot(indexs,label[:,i])
    plt.title("c-{}".format(i))

    plt.show()