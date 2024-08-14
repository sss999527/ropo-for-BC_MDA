# Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source 
Here is the official implementation of the model BC-MDA.

## Abstract
Unsupervised Multi-Source Domain Adaptation (UMDA) aims to utilize multiple source domains with ample labeled data to find the optimal training and combination strategies, thereby training a robust predictive model for a target domain that lacks labeled data. Unfortunately, previous UMDA research has overlooked practical concerns such as privacy, data storage, and computational overhead. A new challenge arises when data from different domains are segregated; only isolated source-domain data and unlabeled target-domain data are available for training the model and completing the entire domain adaptation process. We refer to this new paradigm as Multi-Source Domain Adaptation Without Access to Source (MSFDA). To address this challenge, we propose a generic domain adaptation framework based on contrastive learning, named Bidirectional Contrastive Learning for Multi-Source Domain Alignment Without Access to Source (BC-MDA). This approach facilitates the learning of more discriminative representations while ensuring data localization. Moreover, to harness the full potential of backbone networks, we substitute the original batch normalization layers with matching normalization layers, further enhancing the model's adaptability to the target domain and reducing the distributional differences between source and target domains in the feature space. Extensive experimental evaluations on multiple datasets validate the efficacy of our proposed method.

## Method
![F1](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/F1.png)

* First, to achieve unsupervised domain adaptation, we separately train models on the source and target domains and then align them through contrastive learning loss. We retrain the models iteratively until they converge, achieving bidirectional training between the source and target domains.

* Second, we train the model using both cross-entropy loss and contrastive learning loss. In the target domain, we generate pseudo-labels and apply a confidence thresholding strategy to implement cross-entropy loss with these pseudo-labels.

* Finally, we make improvements to the batch normalization layers in the feature extractor network by utilizing matched normalization layers that share affine parameters between the source and target domains. This achieves domain alignment at the feature extraction level, yielding very good results.

## Setup
### Install Package Dependencies

```
Python Environment: >= 3.6
torch >= 1.2.0
torchvision >= 0.4.0
tensorbard >= 2.0.0
numpy
yaml
```

### Install Datasets
We need users to download the DigitFive, Office-Caltech, Office31, DomainNet, or MiniDomainNet, and Office-Home datasets for the MSFDA experiments. They should declare a base path to store the datasets with the following directory structure:

```
base_path/
├── DigitFive/
│   │   mnist_data.mat
│   │   mnistm_with_label.mat
|   |   svhn_test_32x32.mat
|   |   svhn_train_32x32.mat  
│   │   ...
├── Office-Caltech/
│   │   ...
├── Office31/
│   │   ...
├── MiniDomainNet/
│   │   ...
└── Office-Home/
│   │   ...
├── DomainNet/
│   │   Clipart
│   │   Infograph
│   │   ...
└───trained_model
│   │	parmater
│   │	runs
...
```
### Unsupervised Multi-source-free Domain Adaptation
The configuration files are located in the `./config` directory, where you will find four `.yaml`-formatted config files. To execute unsupervised multi-source decentralized domain adaptation on a particular dataset, such as painting in DomainNet, utilize the following commands.

```
python main.py --config MiniDomainNet.yaml --target-domain painting -bp "$(pwd)"
```
You can download the dataset from [DomainNet](https://ai.bu.edu/M3SDA/).

The training parameters of some datasets can be downloaded from Baidu Netdisk:
Link:( https://pan.baidu.com/s/1wgzt6fGnqlkjJIv-Jn5MAg?pwd=3why)


The training results for four main datasets are as follows:
  * Results on DigitFive and Office31 dataset.
![T1](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/T1.png)
  * Results on Office-Home and Office-Caltech dataset.
![T2,3](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/F2_3.png)
  * Results on DomainNet dataset.
![T4](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/T4.png)

## Acknowledgments

## Miscellaneous
Due to the adaptation of text classification and and multimodal adaptation, some programs are inconvenient to display, and will be released once they are organized.

