# Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source 
Here is the official implementation of the model BC-MDA.

## Abstract
Unsupervised Multi-Source Domain Adaptation (UMDA) aims to utilize multiple source domains with ample labeled data to find the optimal training and combination strategies, thereby training a robust predictive model for a target domain that lacks labeled data. Unfortunately, previous UMDA research has overlooked practical concerns such as privacy, data storage, and computational overhead. A new challenge arises when data from different domains are segregated; only isolated source-domain data and unlabeled target-domain data are available for training the model and completing the entire domain adaptation process. We refer to this new paradigm as Multi-Source Domain Adaptation Without Access to Source (MSFDA). To address this challenge, we propose a generic domain adaptation framework based on contrastive learning, named Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source (BC-MDA). This approach facilitates the learning of more discriminative representations while ensuring data localization. Moreover, to harness the full potential of backbone networks, we substitute the original batch normalization layers with matching normalization layers, further enhancing the model's adaptability to the target domain and reducing the distributional differences between source and target domains in the feature space. Extensive experimental evaluations on multiple datasets validate the efficacy of our proposed method.

## Method
![F1](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/F1.png)

* First, to achieve unsupervised domain adaptation, we separately train models on the source and target domains and then align them through contrastive learning loss. We retrain the models iteratively until they converge, achieving bidirectional training between the source and target domains.

* Second, we train the model using both cross-entropy loss and contrastive learning loss. In the target domain, we generate pseudo-labels and apply a confidence thresholding strategy to implement cross-entropy loss with these pseudo-labels.

* Finally, we make improvements to the batch normalization layers in the feature extractor network by utilizing matched normalization layers that share affine parameters between the source and target domains. This achieves domain alignment at the feature extraction level, yielding very good results.

