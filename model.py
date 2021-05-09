import torch
import torch.nn as nn
import torch_geometric.nn as gn
#implementing the model architecture

class NeuralNetwork(nn.Module):
	def __init__(self,in_features,hidden_features,out_features):#initialize class  with # of features, inputs and outputs, and the layers of the model  
		
		super(NeuralNetwork,self).__init__()
		
		self.features = in_features
		self.hidden = hidden_features
		self.output = out_features

		#layers
		self.layer1 = gn.GATConv(self.features, 2*self.hidden)   #first Graph Attention Network Transformer Conv layer
		self.layer2 = gn.GATConv(2*self.hidden, self.hidden * 2) 

		self.linear1 = nn.Linear(self.hidden*2, self.hidden)     #Linear layers to process extracted features and make prediction 
		self.linear2 = nn.Linear(self.hidden,self.output)

	def forward(self,data): #defining flow of data through model architecture, defining activation functions 

		x = data.x
		edge = data.edge_index
		batch_size = data.batch

		x = self.layer1(x,edge)
		x = nn.functional.selu(x)  #Use SELU(Simple RELUs have a vanishing gradient problem)

		x = self.layer2(x,edge)
		x = nn.functional.selu(x)

		x = gn.global_mean_pool(x,batch_size) #Global Mean Pooling to replace fully connected layers in CNN
		x = nn.functional.selu(x)

		x = self.linear1(x)
		x = nn.functional.selu(x)

		x = self.linear2(x)						#Linear Layers to predict outcome based on extracted features
		x = nn.functional.log_softmax(x,dim = -1)

		return x
    


