import os
import math
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

DIR_NAME = "WTD_upload"
bad_files = set(["201105240921_board_setPoint_600V_fan_setPoint_100_mfc_setPoint_CO_4000ppm_p1"])
chemicals_list = ["Acetaldehyde_500", "Acetone_2500", "Ammonia_10000", "Benzene_200", 
                  "Butanol_100", "CO_1000", "CO_4000", "Ethylene_500", "Methane_1000", 
                  "Methanol_200", "Toluene_200"]
sequence = []
chemicals_label = []
chemicals_count = 0

g = torch.Generator()
g.manual_seed(0)

def data_preprocessing(data):
    drop_list = [9,17,25,33,41,49,57,65,73]

    data1 = data.drop(data.columns[1:12], axis=1)

    for a in drop_list:
        data1 = data1.drop(data1.columns[a], axis=1)

    data1.columns = ['T','A1','A2','A3','A4','A5','A6','A7','A8',
                     'B1','B2','B3','B4','B5','B6','B7','B8',
                     'C1','C2','C3','C4','C5','C6','C7','C8',
                     'D1','D2','D3','D4','D5','D6','D7','D8',
                     'E1','E2','E3','E4','E5','E6','E7','E8',
                     'F1','F2','F3','F4','F5','F6','F7','F8',
                     'G1','G2','G3','G4','G5','G6','G7','G8',
                     'H1','H2','H3','H4','H5','H6','H7','H8',
                     'I1','I2','I3','I4','I5','I6','I7','I8']

    data1 = data1.drop(data1[data1['T'] <= 2000].index)
    data1 = data1.drop(data1[data1['T'] >= 200000].index)

    x_list = list(data1['T'])

    series = pd.to_timedelta(x_list, unit='ms')
    series = series.to_series().reset_index(drop=True)
    data1.iloc[:,0] = series

    data2 = data1.resample('500ms', on='T', origin = 'start').max()
    data2 = data2.fillna(method="bfill")
    data2 = data2[0:300].transpose().to_numpy(dtype=float)

    #scaling from 0 to 1
    data2 = (data2-0)/(4096-0)

    return data2

class custom_dataset:
    def __init__(self, dataset, label_ds):
        self.dataset = dataset
        self.label_ds = label_ds

    def __len__(self):
        return len(self.label_ds)

    def __getitem__(self, index):
        return dict(
            dataset = self.dataset[index],
            label = self.label_ds[index]
        )
    def create_datasets(self, dataset, label_ds):

        self.dataset = dataset
        self.label_ds = label_ds
        train_x, test_x, train_y, test_y = train_test_split(dataset, label_ds, test_size = 0.2)

        train_ds = TensorDataset(
            torch.tensor(train_x).float(),
            torch.tensor(train_y).long()
        )
        test_ds = TensorDataset(
            torch.tensor(test_x).float(),
            torch.tensor(test_y).long()
        )

        return train_ds, test_ds

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_loaders(data, bs=5, jobs=0):
    
    train_ds, test_ds = data
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=jobs, worker_init_fn=seed_worker, generator=g)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=jobs, worker_init_fn=seed_worker, generator=g)

    return train_dl, test_dl

for c in os.listdir(DIR_NAME):
    p1 = os.path.join(DIR_NAME,c)
    for line in os.listdir(p1):
        p2 = os.path.join(p1,line)
        for file in os.listdir(p2):
            if file not in bad_files:
                if (int(file[0:12]) >= 201105301622) and (int(file[0:12]) <= 201105311620): 
                    continue
                else:
                    df = pd.read_table(os.path.join(p2,file))
                    
                    value = data_preprocessing(df)
                    sequence.append(value)

                    chemicals_label.append(chemicals_count)

    chemicals_count += 1

sequence = np.asarray(sequence)

#save the dataset into pytorch file
torch.save(sequence, 'act_sequence.pt')
torch.save(chemicals_label, 'act_c_label.pt')

#load the pytorch file dataset
sequence = torch.load('act_sequence.pt')
c_label = torch.load('act_c_label.pt')

#create dataset and create loaders
ds = custom_dataset(sequence,c_label)
train_dl, test_dl = create_loaders(ds.create_datasets(sequence,c_label))

class CNN(nn.Module):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining first 1D convolution layer
            nn.Conv1d(in_channels=72, out_channels=72, kernel_size=25, stride=1, bias=True), # output = 72 * 276
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=25, stride=1),                                          # output = 72 * 252

            # Defining second 1D convolution layer
            nn.Conv1d(in_channels=72, out_channels=72, kernel_size=25, stride=1, bias=True), # output = 72 * 228
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=25, stride=1),                                          # output = 72 * 204

            # Defining third 1D convolution layer
            nn.Conv1d(in_channels=72, out_channels=72, kernel_size=25, stride=1, bias=True), # output = 72 * 180
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=25, stride=1),                                          # output = 72 * 156
        
            nn.Flatten(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(72 * 156, 1024),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 11),
            nn.Softmax(dim=1)
        )

    # Defining the forward pass    
    def forward(self, x):
        return self.cnn_layers(x)

#Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
# print(model)

#Defining the model hyper-parameters
learning_rate = 0.002 #0.001
weight_decay = 0.01 #0.01
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.7)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_accu = []
train_losses = []
eval_accu = []
eval_losses = []

def train(epoch):
    print('\nEpoch : %d'%epoch)
   
    model.train()
    
    running_loss=0
    correct=0
    total=0
    
    for i, (train_x, train_y) in enumerate(train_dl):
        
        inputs, labels = train_x.to(device), train_y.to(device)
        
        #Calculating the model output and the cross entropy loss
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)

        # print(outputs)

        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # print('input size =', labels.size(0))
        # print('running_loss =', running_loss)

        #Calculating prediction and comparing predicted and true labels
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    # print('train_loss =', running_loss)
    # print('correct =', correct)
    # print('total =', total)

    # train_loss=running_loss/len(train_dl)
    train_loss = (running_loss/total)
    accu = (100.) * (correct/total)
    
    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

def test(epoch):
    model.eval()
    
    running_loss=0
    correct=0
    total=0
    
    with torch.no_grad():

        for i, (test_x, test_y) in enumerate(test_dl):
        
            inputs, labels = test_x.to(device), test_y.to(device)

            #Calculating outputs and the cross entropy loss
            outputs = model(inputs)

            print(outputs)

            loss = loss_fn(outputs,labels)

            running_loss += loss.item() * inputs.size(0)
            
            #Calculating predictions and comparing predicted and true labels
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    # print('test_loss =', running_loss)
    # print('correct =', correct)
    # print('total =', total)

    test_loss = (running_loss/total)
    accu = (100.) * (correct/total)
    
    eval_losses.append(test_loss)
    eval_accu.append(accu)
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 

epochs = 20

for epoch in range(1,epochs+1): 
  train(epoch)
  test(epoch)

#Plotting the model for every epoch
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('1DCNN Model')

ax1.set_title('Train vs Validation Losses')
ax1.plot(train_losses, 'o-', linewidth=0.75)
ax1.plot(eval_losses, 'o-', linewidth=0.75)
ax1.set(xlabel='epoch', ylabel='losses')
ax1.legend(['Train','Valid'])

ax2.set_title('Train vs Validation Accuracy')
ax2.plot(train_accu, 'o-', linewidth=0.75)
ax2.plot(eval_accu, 'o-', linewidth=0.75)
ax2.set(xlabel='epoch', ylabel='accuracy')
ax2.legend(['Train','Valid'])

ax1.grid(color='grey', linestyle='-', linewidth=0.5)
ax1.set_xlim(0,epochs-1)
ax1.set_ylim(0, math.ceil(2 * max(max(eval_losses), max(train_losses))))

ax2.grid(color='grey', linestyle='-', linewidth=0.5)
ax2.set_xlim(0,epochs-1)
ax2.set_ylim(0, 100)

fig.tight_layout() 
plt.show()

print('Done!')