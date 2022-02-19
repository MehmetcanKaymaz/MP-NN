import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pickle


torch.manual_seed(1)    # reproducible


print("Loading datas ...")
with open('dataset_s.pkl', 'rb') as f:
    map_dataset = pickle.load(f) 

data_size=len(map_dataset)
train_data=np.zeros((data_size,7))
label=np.zeros((data_size,10))
for i in range(data_size):
    train_data[i,:]=np.array([map_dataset[i][0][1][0],map_dataset[i][0][1][1],map_dataset[i][0][1][2],map_dataset[i][0][1][3],map_dataset[i][0][3][0],map_dataset[i][0][3][1],map_dataset[i][0][3][2]])
    label[i,:]=np.array([map_dataset[i][1][0][1],map_dataset[i][1][0][2],map_dataset[i][1][0][3],map_dataset[i][1][1][1],map_dataset[i][1][1][2],map_dataset[i][1][1][3],map_dataset[i][1][2][1],map_dataset[i][1][2][2],map_dataset[i][1][2][3],map_dataset[i][1][3][1]])



train = torch.from_numpy(train_data.astype(np.float32))
label = torch.from_numpy(label.astype(np.float32))

print("Datas loaded")

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(7, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
	    torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 256
EPOCH = 10001

torch_dataset = Data.TensorDataset(train, label)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)


epp_loss=[]

print("Training ....")

for epoch in range(EPOCH):
    if epoch%500==0 :
        torch.save(net.state_dict(), "Traj-Models/checkpoint-{}.pth".format(epoch))
        print("checkpoint-{} saved!".format(epoch))
        np.savetxt("Loss-{}.txt".format(epoch),np.array(epp_loss))

    loss_arr=[]
    print("Epoch {} started...".format(epoch))
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x.float())     # input x and predict based on x
        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        loss_arr.append(loss.item())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    epp_loss.append(np.mean(np.array(loss_arr)))

#np.savetxt("Loss.txt",np.array(epp_loss))