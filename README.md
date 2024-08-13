# Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source 
Here is the official implementation of the model BC-MDA.

## Abstract
Unsupervised Multi-Source Domain Adaptation (UMDA) aims to utilize multiple source domains with ample labeled data to find the optimal training and combination strategies, thereby training a robust predictive model for a target domain that lacks labeled data. Unfortunately, previous UMDA research has overlooked practical concerns such as privacy, data storage, and computational overhead. A new challenge arises when data from different domains are segregated; only isolated source-domain data and unlabeled target-domain data are available for training the model and completing the entire domain adaptation process. We refer to this new paradigm as Multi-Source Domain Adaptation Without Access to Source (MSFDA). To address this challenge, we propose a generic domain adaptation framework based on contrastive learning, named Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source (BC-MDA). This approach facilitates the learning of more discriminative representations while ensuring data localization. Moreover, to harness the full potential of backbone networks, we substitute the original batch normalization layers with matching normalization layers, further enhancing the model's adaptability to the target domain and reducing the distributional differences between source and target domains in the feature space. Extensive experimental evaluations on multiple datasets validate the efficacy of our proposed method.

## Method
![image](https://github.com/user-attachments/assets/41b80403-8bbf-4f86-ad01-bce056537871)




