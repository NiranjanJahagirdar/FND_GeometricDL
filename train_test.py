import model
from helper import plotROC, print_pretty, test_and_score, plotTrainingLoss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric.nn as gn
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
#parameters

seed = 100 
device = 'cpu'   #setting default device 
dataset_name = 'PROTEINS' #proteins database to label as enzymatic(0)/non enzymatic(1)
batch = 100
lr = 0.01   #parameters for ADAM 
w_decay = 0.01  #parameters for ADAM 
hidden_layers = 128  #default 
epochs = 100
dataset = TUDataset('data/', name=dataset_name, use_node_attr=True)
classes = dataset.num_classes
features = dataset.num_features
data_size = int(len(dataset))
one = 1
zero = 0
torch.manual_seed(seed)

# test and train dataset 80-20 split 

num_training = int(data_size*one*0.8)
num_test= int(data_size - num_training)

training_set,test_set = random_split(dataset,[num_training,num_test])


#loading into standard data_loader

train_loader = DataLoader(training_set, batch_size = batch, shuffle = True)
test_loader = DataLoader(test_set,batch_size = batch,shuffle = False)

model1 = model.NeuralNetwork(features,hidden_layers,classes).to(device)

#optimizer 
optimizer = torch.optim.Adam(model1.parameters(), lr= lr, weight_decay= w_decay)

def train(model:nn.Module):
  i=0
  model.train()   #training mode 
  e = []
  tloss = []
  while i<epochs:
    i+=1
    print("Epoch: {}".format(i))
    e.append(i)
    train_loss  = 0.0 * one
    for data in train_loader:
      optimizer.zero_grad()   #zeros gradients, since pytorch accumulates them 
      data = data.to(device)
      out = model(data)
      y = data.y
      loss = nn.functional.nll_loss(out,y)  #calculate NLL loss
      loss.backward()  
      optimizer.step()  #update weights basis gradient values
      #print(loss)
      train_loss += loss
    tloss.append(loss.mean())
  plotTrainingLoss(e,tloss,dataset_name)
    
  print("Training Loss:{:.4f}".format(train_loss))
  print("Saving Weights.....")
  torch.save(model1.state_dict(),dataset_name+'.pth')   #saves model state so that weights might be used later for testing or continuing training 

def test():
	model1.load_state_dict(torch.load(dataset_name+'.pth'))  #loads saved weights from training 
	a,b,c,tloss = test_and_score(model1,device,test_loader,dataset_name)  #calculates score, and prints it
	print_pretty(a,b,c,tloss)


if __name__ == '__main__':
	train(model1)
	test()

