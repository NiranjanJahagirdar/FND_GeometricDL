#import dependencies to plot curves, metrics, and load data
#torch-geometric required for graph processing 

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric.nn as gn
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve

#function to plot ROC-AUC curve to measure performance of the model, receives the fpr and tpr, and plots it 
def plotROC(fpr,tpr,dataset):
	plt.style.use('seaborn')
	plt.plot(fpr,tpr,lw =2, linestyle = '--', color = 'blue', label = 'ROC_Curve')
	plt.title("ROC CURVE")
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	
	plt.savefig('ROC-'+dataset, dpi = 100)

	plt.show()

#plots training loss with respect to # of epochs completed. Helps to keep track that proper training is taking place.
def plotTrainingLoss(epoch,loss,dataset):
	plt.style.use('seaborn')
	plt.plot(epoch,loss,lw =2, linestyle = '--', color = 'red', label = 'Training Loss')
	plt.title("Training Loss")
	plt.xlabel("EPOCH")
	plt.ylabel("LOSS")
	
	plt.savefig('LOSS-'+dataset, dpi = 100)

	plt.show()


#calculate accuracy, precision,recall & loss for each batch after loading data & making predictions
def test_and_score(model:nn.Module,device,data_loader,dataset):
	
	model.eval() #entering evaluation mode
	t_loss = 0
	log = []

	with torch.no_grad():#since we are only predicting, we use no grad
		for data in data_loader:
			data = data.to(device) #sends data to device (CUDA or CPU)

			result = model(data)
			y = data.y
			log.append([nn.functional.softmax(result,dim = -1), y ]) #softmax in last layer to predict final probabilities 

			t_loss += nn.functional.nll_loss(result,y).item() #calculates loss basis Negative Log-Likelihood loss


	#Computing Scores

	size = int(len(log))
	accuracy,precision,recall = 0,0,0

	y_list = []
	pred_list = []  #accumlating scores and then dividing by size, since we using mini batch gradient descent 
	for batch in log:
		y_pred = batch[0].data.cpu().numpy().argmax(axis=1)
		y = batch[1].data.cpu().numpy().tolist()
		
		y_list.extend(y)
		pred_list.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		#pred_list.extend(y_pred)
		accuracy += accuracy_score(y,y_pred)
		precision +=precision_score(y,y_pred)
		recall +=recall_score(y,y_pred)

	'''
	final_pred = []
	for i in range(len(pred_list)):
		if(pred_list[i][0] >= pred_list[i][1]):
			final_pred.append(0)
		else:
			final_pred.append(1)

	'''

	#print(y_list,pred_list)
	fpr,tpr,thresh = roc_curve(y_list,pred_list,pos_label = 1) #gets fpr,tpr, values to plot and passes it to plotROC function
	plotROC(fpr,tpr,dataset)

	return accuracy/size, precision/size , recall/size, t_loss


def print_pretty(acc,pre,recall,loss):
	print('Accuracy:{:.4f}, Precision:{:.4f}, Recall: {:.4f}, Loss: {:.4f}'.format(acc,pre,recall,loss))
	return








