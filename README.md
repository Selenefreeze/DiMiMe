# Microbe-bridged Disease-metabolite Associations prediction by Heterogeneous Graph Fusion

### Introduction
In this study, we integrate some databases and extracted a variety of associations data among microbes, metabolites, and diseases. After obtaining the trilateral association data (microbe-metabolite, metabolite-disease, and disease-microbe), we consider building a heterogeneous graph to describe the association data. In our model, microbes are used as a bridge between diseases and metabolites. In order to fuse the information of disease-microbe-metabolite graph, we use the bipartite graph attention network on the disease-microbe and metabolite-microbe bigraph. 

### Usage
#### 'Dataset' directory
All node features and edge data of bipartite graph. Positive samples and negative samples of all kinds.
#### 'Analysis_results' directory
Result data used in experimental case analysis.
#### 'Sampling' directory
K-means method used in negative sampling.


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
