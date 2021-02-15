# GAT_PPI_Legacy

This repository presents an implementation of an improved Graph Attention Network for binary classification on the PPI Legacy dataset. The PPI Legacy dataset is provided by the Deep Graph Library (dgl).

## The architecture

The GAT architecture we designed is close to the one presented in the paper “Graph Attention Network” ( Velickovic, 2018) but not identical. Our network has 3 layers, each of which is composed of one Multi-Head Attention Block and a dense layer, whose outputs we sum, and pass through a Batch Normalization - LeakyReLU - Dropout block. The hidden dimension of our network is 256. We use 8 heads in the Multi-Head Attention Block. The architecture is summarized below. To each block legend we associated a pair (x, y) where x is the dimension of the input of the block and y the dimension of its output.

![alt text] (https://github.com/CarlaMartin092/GAT_PPI_Legacy/blob/master/pictures/attention_block.png?raw=true)
![alt text] (https://github.com/CarlaMartin092/GAT_PPI_Legacy/blob/master/pictures/multi_head_attention_block.png?raw=true)
![alt text] (https://github.com/CarlaMartin092/GAT_PPI_Legacy/blob/master/pictures/improved_gat_architecture.png?raw=true)

## Performance
