# Reviewing Fake News Detections using Geometric Deep Learning

We had to review and implement the paper, Fake News Detection on Social Media using
Geometric Deep Learning (2019) by Frederico Monti et al as part of the course Information Retrieval taught by Dr. Vinti Aggarwal in Second Semester 2020-21 at BITS Pilani.

Group Members
- Aditya Saxena
- Manjot Singh
- Niranjan Ashok Jahagirdar
- Pratik Borikar
- Prakhar Agnihotri

# Code

# Prerequisites
The following libraries must be installed before running the code:
- torch-sparse
- torch-geometric
- torch_scatter
- ipykernel
- pytorch
- sklearn

# Dataset Description
Used the PROTEINS database for training the model, since one for Tweets was not available readily. It
is a dataset of proteins classified as enzymatic or non-enzymatic, with proteins represented by graphs.
Enzymatic - 0
Non-Enzymatic - 1
Dataset has multiple csv files to describe the graph of the protein.
1. PROTEINS_A
adjacency matrix for all graphs (i.e. proteins in this regard)
2. PROTEINS_graph_indicator
column vector of graph identifiers for all nodes of all graphs
3. PROTEINS_graph_labels
labels for all graphs (1-Enzymatic, 0-Non-Enzymatic)
4. PROTEINS_node_labels
column vector of all node labels
5. PROTEINS_node_attributes
matrix of node attributes, actual biological features, numerically represented
# File Description
# helper.py

**Automatically runs on execution of train_test.py**

**def plotROC(fpr,tpr):** function to plot the ROC curve to quantify performance of the classification
model. AUC (Area under the Curve) is calculated from this graph, and serves as an indicator of how
capable the model is in distinguishing between classes. Higher the AUC, the better the model is at
making the correct predictions.

**def test_and_score(model:nn.Module,device,data_loader):** to score and evaluate the model. Recall,
Precision, Accuracy and Loss of the model is returned.

**def plotTrainingLoss(epoch,loss,dataset):** Function to plot the loss for each epoch, serving as an
indicator to make sure training is taking place properly, and that continued training is helping
decrease loss.

**def print_pretty(acc,pre,recall,loss):** Prints metrics in a way that it is easy to read.

# model.py

**class NeuralNetwork(nn.Module):** Class to define the model architecture, with two Graph Attention
Convolution Layers to extract relevant information from input propagation (or Protein connections, in
our case) followed by a Global Mean Pool layer, which is followed by two linear layers to process and
generate the binary output (Fake News or Not, Enzymatic or not)

**def init(self,in_features,hidden_features,out_features) :** Constructor to intialise the model layers
def forward(self,data): Defines input for the model, the forward pass through layers, the activation
functions to use (SELU*, in our case), and finally returns the output after passing through a Softmax
layer.
*SELUs because RELUs have a vanishing gradient problem

# train_test.py

Intialises the model, the optimizer, and contains two functions, one for training and one for testing.

Imports helper.py, model.py, and other torch and torch-geometric libraries.

**Default values:-**
- Device= CPU
- Dataset=Proteins
- Batch Size=100
- Learning Rate=0.01
- Weight Decay=0.01
- Hidden Layers =128
- Epochs=100
- Train:Test Split=80:20

We use the adam optimizer with the aforementioned parameters to carry out training since it is
computationally efficient and effective.
