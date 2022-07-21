# Microbe-bridged Disease-metabolite Associations prediction by Heterogeneous Graph Fusion

### Introduction
Dimime model uses microbial information as a bridge to fuse disease and metabolite information through bipartite graph attention network, and finally realizes the association prediction of the disease-metabolite.

### Usage
#### 'Dataset' directory
All node features and edge data of bipartite graph. Positive samples and negative samples of all kinds.
#### 'Analysis_results' directory
Result data used in experimental case analysis.
#### 'Sampling' directory
K-means method used in negative sampling.
#### 'model' and 'utiles'
Model code, initialization function and evaluation metrics function involved.

### Requirements
The model is tested to work under python3.6. The required dependencies versions are as follows:
```
torch==1.4.0+cu100
torch-cluster==1.5.2
torch-geometric==1.4.1
torch-scatter==2.0.3
torch-sparse==0.5.1
torch-spline-conv==1.2.0
torchvision==0.5.0+cu100
scikit-learn==0.21.3
pandas==1.1.5
numpy==1.16.0
```
