# Code-for-AAM-master

An Autoencoder Framework with Attention Mechanism for Cross-Domain Recommendation
This is our implementation for the paper:

In this paper, we propose a novel Autoencoder framework with Attention Mechanism (AAM) for cross-domain recommendation, which can transfer and fuse information between different domains and make a more accurate rating prediction. In addition, to learn the affinity of the user latent
factors between different domains in a multi-aspect level, we also strengthen the self-attention mechanism by using multi-head selfattention and propose AAM++.

Please cite our TCYB'20 paper if you use our codes. Thanks! 

-Environment Settings
python 3.7.3
pytorch 1.3.0

-Guidelines to run the codes.
python run.py

-Dataset
We provide the Amazon datasets obtained from http://jmcauley.ucsd.edu/data/amazon. 
There are 4 domains, namely Book, CD, Music and Movie.
Each domain has training set and testing set.

Reference:

Shi-Ting Zhong, Ling Huang, Chang-Dong Wang, Jian-Huang Lai and Philip S. Yu. "An Autoencoder Framework with Attention Mechanism for Cross-Domain Recommendation", TCYB2022


