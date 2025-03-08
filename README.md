# BiCoDA: Bidirectional Contrastive Learning with Cross-Source Consistency for Domain Adaptation 
Here is the official implementation of the model BiCoDA.

## Abstract
Unsupervised Multi-Source Domain Adaptation (UMDA) aims to utilize multiple source domains with abundant labeled data to identify the optimal training and combination strategies, thus training robust predictive models for target domains that lack labeled data. Unfortunately, previous UMDA research has overlooked practical issues such as privacy, data storage, and computational overhead. When data from different domains are isolated, new challenges emerge; only isolated source domain data and unlabeled target domain data are available for model training and the entire domain adaptation process. To address this challenge, we propose a contrastive learning-based general domain adaptation framework called \textbf{Bi}directional \textbf{Co}ntrastive Learning based Multi-Source \textbf{D}omain \textbf{A}lignment (BiCoDA). This method achieves the construction of a target domain model through iterative training, realizing the domain adaptation process at the model level while ensuring data localization. Furthermore, to fully exploit the potential of the backbone network, we replaced the original batch normalization layers with matching normalization layers, further enhancing the model's adaptability to the target domain and reducing the distributional discrepancies between the source and target domains in feature space. Extensive experimental evaluations on multiple datasets have verified the effectiveness of our proposed method.
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
base_path
│       
└───dataset
│   │   DigitFive
│       │   mnist_data.mat
│       │   mnistm_with_label.mat
|       |   svhn_test_32x32.mat
|       |   svhn_train_32x32.mat  
│       │   ...
│   │   DomainNet
│       │   Clipart
│       │   Infograph
│       │   ...
│   │   MiniDomainNet
│       │   ...
│   │   OfficeCaltech10
│       │   ...
|   |   Office31
|       |   ...
└───trained_model
│   │	parmater
│   │	runs
...
```
Note that the `dataset` folder is different from the `datasets` folder, one for the dataset and the other for the network model.

### Unsupervised Multi-source-free Domain Adaptation
The configuration files are located in the `./config` directory, where you will find four `.yaml`-formatted config files. To execute unsupervised multi-source decentralized domain adaptation on a particular dataset, such as painting in DomainNet, utilize the following commands.

```
python main.py --config MiniDomainNet.yaml --target-domain painting -bp "$(pwd)"
```
You can download the dataset from [DomainNet](https://ai.bu.edu/M3SDA/).

The training parameters of some datasets can be downloaded from Baidu Netdisk:
https://pan.baidu.com/s/1wgzt6fGnqlkjJIv-Jn5MAg?pwd=3why


The training results for four main datasets are as follows:
  * Results on DigitFive and Office31 dataset.
![T1](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/T1.png)
  * Results on Office-Home and Office-Caltech dataset.
![T2,3](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/F2_3.png)
  * Results on DomainNet dataset.
![T4](https://github.com/sss999527/ropo-for-BC_MDA/blob/main/images/T4.png)

## Acknowledgments

## Miscellaneous
Due to the further completion of text classification and multimodal adaptation, some unimportant programs are not displayed, and once sorted out and annotated, the complete code will be released.

