# Awesome Weakly-supervised Semantic Segmentation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![GitHub stars](https://img.shields.io/github/stars/gyguo/awesome-weakly-supervised-semantic-segmentation?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/gyguo/awesome-weakly-supervised-semantic-segmentation?color=green&label=Fork)

------

**<u>Contact gyguo95@gmail.com if any paper is missed!</u>**

------

# Table of Contents

- [1. Paper List](#1-paper-list)
    + [1.1. supervised by image tags (I)](#11-supervised-by-image-tags)
    + [1.2. Supervised by bounding box (B)](#12-supervised-by-bounding-box)
    + [1.3. Supervised by scribble (S)](#13-supervised-by-scribble)
    + [1.4. Supervised by point (P)](#14-supervised-by-point)
- [2. Performance list](#2-performance-list)
    + [2.1. Results on PASCAL VOC dataset](#21-results-on-pascal-voc-dataset)
    + [2.2. Results on MS-COCO dataset](#22-results-on-ms-coco-dataset)
- [3. Dataset](#3-dataset)
- [4. Awesome-list of Weakly-supervised Learning from Our Team](#4-awesome-list-of-weakly-supervised-learning-from-our-team)

# 1. Paper List

### 1.1. supervised by image tags

#### 2025
- Frozen CLIP-DINO: a Strong Backbone for Weakly Supervised Semantic Segmentation *TPAMI2025*
- Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation *CVPR2025*
- Weakly Supervised Semantic Segmentation via Progressive Confidence Region Expansion *CVPR2025*
- WISH: Weakly Supervised Instance Segmentation using Heterogeneous Labels *CVPR2025*
- FFR: Frequency Feature Rectification for Weakly Supervised Semantic Segmentation *CVPR2025*
- POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation *CVPR2025*
- Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation *CVPR2025*
- Complementary branch fusing class and semantic knowledge for robust weakly supervised semantic segmentation *PR2025*
- WeakCLIP: Adapting CLIP for Weakly-Supervised Semantic Segmentation *IJCV2025*
- Toward Modality Gap: Vision Prototype Learning for Weakly-supervised Semantic Segmentation with CLIP *AAAI2025*

#### 2024
- DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation  *ECCV2024*
- DHR: Dual Features-Driven Hierarchical Rebalancing in Inter- and Intra-Class Regions for Weakly-Supervised Semantic Segmentation *ECCV2024*
- Diffusion-Guided Weakly Supervised Semantic Segmentation *ECCV2024*
- Knowledge Transfer with Simulated Inter-Image Erasing for Weakly Supervised Semantic Segmentation *ECCV2024*
- Phase Concentration and Shortcut Suppression for Weakly Supervised Semantic Segmentation *ECCV2024*
- WeakCLIP: Adapting CLIP for Weakly-Supervised Semantic Segmentation *IJCV2024*
- Curriculum Point Prompting for Weakly-Supervised Referring Image Segmentation *CVPR2024*
- DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation *CVPR2024*
- Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation *CVPR2024*
- PSDPM: Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation *CVPR2024*
- Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation *CVPR2024*
- Class Tokens Infusion for Weakly Supervised Semantic Segmentation *CVPR2024*
- From SAM to CAMs: Exploring Segment Anything Model for Weakly Supervised Semantic Segmentation *CVPR2024*
- Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation *CVPR2024*
- SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation *AAAI2024*
- Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation *AAAI2024*
- Mctformer+: Multi-class token transformer for weakly supervised semantic segmentation *TPAMI2024*
- Foundation Model Assisted Weakly Supervised Semantic Segmentation  *WACV2024*


#### 2023
- CLIP Is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation *CVPR2023*
- Token Contrast for Weakly-Supervised Semantic Segmentation *CVPR2023*
- Out-of-Candidate Rectification for Weakly Supervised Semantic Segmentation *CVPR2023*
- Uncertainty Estimation via Response Scaling for Pseudo-Mask Noise Mitigation in Weakly-Supervised Semantic Segmentation *AAAI2023*
- Salvage of Supervision in Weakly Supervised Object Detection and Segmentation *TPAMI2023*

#### 2022
- Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation  *CVPR2022*
- **MCTformer:** Multi-class Token Transformer for Weakly Supervised Semantic Segmentation *CVPR2022*
- **AFA:** Learning Affinity from Attention End-to-End Weakly-Supervised Semantic Segmentation with Transformers *CVPR2022*
- **WegFormer:** WegFormer Transformers for Weakly Supervised Semantic Segmentation *CVPR2022*
- **L2G:** L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation *CVPR2022*
- **ReCAM:** Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation. *CVPR2022*
- **GETAM:** GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic segmentation *arxiv2022*

#### 2021

- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **Li *et al.*:** "Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation" *AAAI2021*
- **DRS:** "Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation" *AAAI2021*
- **AdvCAM:** " Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation" *CVPR2021*
- **Yao et al.  **: "Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation" *CVPR2021*
- **EDAM:** "Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation" *CVPR2021*
- **EPS:** Railroad is not a Train Saliency as Pseudo-pixel Supervision for Weakly Supervised Semantic Segmentation *CVPR2021*
- **WSGCN:** "Weakly-Supervised Image Semantic Segmentation Using Graph Convolutional Networks" *ICME2021*
- **PuzzleCAM:** "Puzzle-CAM Improved localization via matching partial and full features" *2021arXiv*
- **CDA:** "Context Decoupling Augmentation for Weakly Supervised Semantic Segmentation" *ICCV2021*
- **ECS-Net:** ECS-Net: Improving Weakly Supervised Semantic Segmentation by Using Connections Between Class Activation Maps.* ICCV2021*
- **Ru *et al.*:** "Learning Visual Words for Weakly-Supervised Semantic Segmentation" *IJCAI2021*
- **AuxSegNet:** "Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation" *ICCV2021*
- **CPN:** "Complementary Patch for Weakly Supervised Semantic Segmentation" *ICCV2021*
- **PMM:** "Pseudo-mask Matters in Weakly-supervised Semantic Segmentation" *ICCV2021*
- **RPNet:** "Cross-Image Region Mining with Region Prototypical Network for Weakly Supervised Segmentation" *TMM2021*
- Weakly-supervised semantic segmentation with superpixel guided local and global consistency *PR2021*

#### 2020

- **RRM:** "Reliability Does Matter An End-to-End Weakly Supervised Semantic Segmentation Approach" *AAAI2020*
- **IAL:** "Weakly-Supervised Semantic Segmentation by Iterative Affinity Learning" *IJCV2020*
- **SEAM:** "Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation" *CVPR2020*
- **Chang *et al.*:** "Weakly-Supervised Semantic Segmentation via Sub-category Exploration" *CVPR2020*
- **ICD:** "Learning Integral Objects with Intra-Class Discriminator for Weakly-Supervised Semantic Segmentation" *CVPR2020*
- **Fan *et al.*:** "Employing multi-estimations for weakly-supervised semantic segmentation" *ECCV2020*
- **MCIS:** "Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation" *2020*
- **BES:** "Weakly Supervised Semantic Segmentation with Boundary Exploration" *ECCV2020*
- **CONTA:** "Causal intervention for weakly-supervised semantic segmentation" *NeurIPS2020*
- **Method:** "Find it if You Can: End-to-End Adversarial Erasing for Weakly-Supervised Semantic Segmentation" *2020arXiv*
- **Zhang *et al.*:** "Splitting vs. Merging: Mining Object Regions with Discrepancy and Intersection Loss for Weakly Supervised Semantic Segmentation" *ECCV2020*
- **LIID** "Leveraging Instance-, Image- and Dataset-Level Information for Weakly Supervised Instance Segmentation" *TPAMI2020*

#### 2019

- **IRN:** "Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations" *CVPR2019*
- **Ficklenet:** " Ficklenet: Weakly and semi-supervised semantic image segmentation using stochastic inference" *CVPR2019*
- **Lee *et al.*:** "Frame-to-Frame Aggregation of Active Regions in Web Videos for Weakly Supervised Semantic Segmentation" *ICCV2019*
- **OAA:** "Integral Object Mining via Online Attention Accumulation" *ICCV2019*
- **SSDD:** "Self-supervised difference detection for weakly-supervised semantic segmentation" *ICCV2019*

#### 2018

- **DSRG:** "Weakly-supervised semantic segmentation network with deep seeded region growing" *CVPR2018*
- **AffinityNet:** "Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation" *CVPR2018*
- **GAIN:** " Tell me where to look: Guided attention inference network" *CVPR2018*
- **AISI:** "Associating inter-image salient instances for weakly supervised semantic segmentation" *ECCV2018*
- **SeeNet:** "Self-Erasing Network for Integral Object Attention" *NeurIPS2018*
- **Method:** "" *2018*

#### 2017

- **CrawlSeg:** "Weakly Supervised Semantic Segmentation using Web-Crawled Videos" *CVPR2017*
- **WebS-i2:** "Webly supervised semantic segmentation" *CVPR2017*
- **Oh *et al*.:** "Exploiting saliency for object segmentation from image level labels" *CVPR2017*
- **TPL:** "Two-phase learning for weakly supervised object localization" *ICCV2017*

#### 2016

- **SEC:** "Seed, expand and constrain: Three principles for weakly-supervised image segmentation" *ECCV2016*
- **AF-SS:** "Augmented Feedback in Semantic Segmentation under Image Level Supervision" *2016*
- **DCSM:** Distinct class-specific saliency maps for weakly supervised semantic segmentation ECCV2016

### 1.2. Supervised by bounding box

- **WSSL:** "Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation" *ICCV2015*
- **Boxsup:** "Boxsup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation" *ICCV2015*
- **Song *et al.*:** "Box-driven class-wise region masking and filling rate guided loss for weakly supervised semantic segmentation" *CVPR2019*
- **BBAM:** "BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation" *CVPR2021*
- **Oh *et al.*:** "Ba ckground-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation" CVPR2021
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 1.3. Supervised by scribble

- **Scribblesup:** "Scribblesup: Scribble-supervised convolutional networks for semantic segmentation" *CVPR2016*
- **NormalCut :** "Normalized cut loss for weakly-supervised cnn segmentation" *CVPR2018*
- **KernelCut :** "On regularized losses for weakly-supervised cnn segmentation" *ECCV2018*
- **BPG:**  "Boundary Perception Guidance: A Scribble-Supervised Semantic Segmentation Approach" *IJCAI2019*
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **DFR:** "Dynamic Feature Regularized Loss for Weakly Supervised Semantic Segmentation" *arxiv2021*
- **A2GNN:** "Affinity attention graph neural network for weakly supervised semantic segmentation" *TPAMI2021*
- Scribble Hides Class: Promoting Scribble-Based Weakly-Supervised Semantic Segmentation with Its Class Label *AAAI2024*
- Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation *CVPR2025*


### 1.4. Supervised by point
- **WhatsPoint:** "What’s the Point: Semantic Segmentation with Point Supervision" *ECCV2016*
- **PCAM:** "PCAMs: Weakly Supervised Semantic Segmentation Using Point Supervision" *arxiv2020*
- P2Object: Single Point Supervised Object Detection and Instance Segmentation *IJCV2025*


# 2. Performance list
2016-2022

### 2.1. Results on PASCAL VOC dataset


#### Image-level supervision without extra data

| Method         | Pub.      | Bac. C     | Arc. S              | Sup.  | Extra data | Pre.S     | val  | test |
|:--------------:|:---------:|:----------:|:-------------------:|:-----:|:----------:|:---------:| ---- | ---- |
| AffinityNet    | CVPR18    | ResNet38   | ResNet38            | **I** | -          | **?**     | 61.7 | 63.7 |
| ICD            | CVPR20    | VGG16      | ResNet101 DeepLabv1 | **I** | -          | **?**     | 64.1 | 64.3 |
| IRN            | CVPR19    | ResNet50   | ResNet50 DeepLabv2  | **I** | -          | ***I***   | 63.5 | 64.8 |
| IAL            | IJCV20    | ResNet?    | ResNet?             | **I** | -          | ***I***   | 64.3 | 65.4 |
| SSDD (PSA)     | ICCV19    | ResNet38   | ResNet38            | **I** | -          | ***I***   | 64.9 | 65.5 |
| SEAM           | CVPR20    | ResNet38   | ResNet38 DeepLabv2  | **I** | -          | ***I***   | 64.5 | 65.7 |
| Chang *et al.* | CVPR20    | ResNet38   | ResNet101 DeepLabv2 | **I** | -          | **?**     | 66.1 | 65.9 |
| RRM            | AAAI20    | ResNet38   | ResNet101 DeepLabv2 | **I** | -          | **?**     | 66.3 | 66.5 |
| BES            | ECCV20    | ResNet50   | ResNet101 DeepLabv2 | **I** | -          | **?**     | 65.7 | 66.6 |
| AFA            | CVPR22    | MiT-B1     | -                   | **I** | -          | **?**     | 66.0 | 66.3 |
| CONTA (+SEAM)  | NeurIPS20 | ResNet38   | ResNet101 DeepLabv2 | **I** | -          | **?**     | 66.1 | 66.7 |
| ESC-Net        | ICCV21    | ResNet38   | ResNet38 DeepLabv2  | **I** | -          | ***I***   | 66.6 | 67.6 |
| Ru *et al.*    | IJCAI21   | ResNet101  | ResNet101 DeepLabv2 | **I** | -          | **?**     | 67.2 | 67.3 |
| WSGCN (IRN)    | ICME21    | ResNet50   | ResNet101 DeepLabv2 | **I** | -          | ***I***   | 66.7 | 68.8 |
| CPN            | ICCV21    | ResNet38   | ResNet38 DeepLabv1  | **I** | -          | **?**     | 67.8 | 68.5 |
| RPNet          | TMM21     | ResNet101  | ResNet50 DeepLabv2  | **I** | -          | ***I***   | 68.0 | 68.2 |
| AdvCAM         | CVPR21    | ResNet50   | ResNet101 DeepLabv2 | **I** | -          | ***I***   | 68.1 | 68.0 |
| ReCAM          | CVPR22    | ResNet50   | ResNet101 DeepLabv2 | **I** | -          | ***I***   | 68.5 | 68.4 |
| PMM            | ICCV21    | ResNet38   | ResNet38 PSPnet     | **I** | -          | **?**     | 68.5 | 69.0 |
| WSGCN (IRN)    | ICME21    | ResNet50   | ResNet101 DeepLabv2 | **I** | -          | ***I+C*** | 68.7 | 69.3 |
| PMM            | ICCV21    | Res2Net101 | Res2Net101 PSPnet   | **I** | -          | **?**     | 70.0 | 70.5 |
| MCTformer      | CVPR22    | DeiT-S     | ResNet38 DeeplabV1  | **I** | -          | **?**     | 71.9 | 71.6 |

#### Box-level supervision

| Method               | Pub.   | Bac. C    | Arc. S               | Sup.  | Extra data | Pre.S     | val  | test |
|:--------------------:|:------:|:---------:|:--------------------:|:-----:|:----------:|:---------:| ---- | ---- |
| BBAM                 | CVPR21 | ?         | ResNet101 DeepLabv2  | **B** | MCG        | ***I***   | 73.7 | 73.7 |
| WSSL                 | ICCV15 | -         | VGG16 DeepLabv1      | **B** | -          | ***I***   | 60.6 | 62.2 |
| Song *et al.*        | CVPR19 | -         | ResNet101  DeepLabv1 | **B** | -          | ***I***   | 70.2 | -    |
| SPML (Song *et al.*) | ICLR21 | -         | ResNet101  DeepLabv2 | **B** | -          | ***I***   | 73.5 | 74.7 |
| Oh *et al.*          | CVPR21 | ResNet101 | ResNet101 DeepLabv2  | **B** | -          | ***I+C*** | 74.6 | 76.1 |

#### Scribble-level supervision

| Method           | Pub.    | Bac. C | Arc. S                    | Sup.  | Extra data  | Pre.S   | val  | test |
|:----------------:|:-------:|:------:|:-------------------------:|:-----:|:-----------:|:-------:| ---- | ---- |
| Scribblesup      | CVPR16  | -      | VGG16 DeepLabv1           | **S** | -           | **?**   | 63.1 | -    |
| NormalCut        | CVPR18  | -      | ResNet101 DeepLabv1       | **S** | Saliency    | **?**   | 74.5 | -    |
| KernelCut        | ECCV18  | -      | ResNet101 DeepLabv1       | **S** | -           | **?**   | 75.0 | -    |
| BPG              | IJCAI19 | -      | ResNet101 DeepLabv2       | **S** | -           | **?**   | 76.0 | -    |
| SPML (KernelCut) | ICLR21  | -      | ResNet101 DeepLabv2       | **S** | -           | ***I*** | 76.1 | -    |
| A2GNN            | TPAMI21 | -      | ?                         | **S** | -           | **?**   | 76.2 | 76.1 |
| DFR              | arxiv21 | -      | UperNet+Swin  Transformer | **S** | 22KImageNet | -       | 82.8 | 82.9 |

#### Point-level supervision

| Method     | Pub.    | Bac. C   | Arc. S     | Sup.  | Extra data | Pre.S   | val  | test |
|:----------:|:-------:|:--------:|:----------:|:-----:|:----------:|:-------:| ---- | ---- |
| WhatsPoint | ECCV16  | -        | VGG16 FCN  | **P** | Objectness | ***I*** | 46.1 | -    |
| PCAM       | arxiv20 | ResNet50 | DeepLabv3+ | **P** | -          | **?**   | 70.5 | -    |

### 2.2. Results on MS-COCO dataset

#### Image-level supervision with extra data

| Method    | Pub.   | Bac. C   | Arc. S              | Sup.  | Extra data | val  | test |
|:---------:|:------:|:--------:|:-------------------:|:-----:|:----------:| ---- | ---- |
| AuxSegNet | ICCV21 | ResNet38 | -                   | **I** | Saliency   | 33.9 | -    |
| EPS       | CVPR21 | ResNet38 | ResNet101 DeepLabv2 | **I** | Saliency   | 35.7 | -    |
| L2G       | CVPR22 | L2G      | VGG16 DeepLabv2     | **I** | Saliency   | 42.7 | -    |
| L2G       | CVPR22 | L2G      | ResNet101 DeepLabv2 | **I** | Saliency   | 44.2 | -    |

#### Image-level supervision without extra data

| Method               | Pub.   | Bac. C   | Arc. S              | Sup.  | Extra data | val  | test |
|:--------------------:|:------:|:--------:|:-------------------:|:-----:|:----------:| ---- | ---- |
| MCTformer            | CVPR22 | DeiT-S   | ResNet38 DeeplabV1  | **I** | -          | 42.0 | -    |
| ReCAM (AdvCAM + IRN) | CVPR22 | ResNet50 | ResNet101 DeepLabv2 | **I** | -          | 45.0 | -    |




# 3. Dataset

### PASCAL VOC 2012

```
@article{everingham2010pascal,
  title={The pascal visual object classes (voc) challenge},
  author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
  journal={International journal of computer vision},
  volume={88},
  number={2},
  pages={303--338},
  year={2010},
  publisher={Springer}
}
```

### MS COCO 2014

```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}

```
---

# 4. Awesome-list of Weakly-supervised Learning from Our Team
- [Awesome Weakly-supervised Semantic Segmentation](https://github.com/gyguo/awesome-weakly-supervised-semantic-segmentation)
- [Awesome Weakly-supervised Action Localization](https://github.com/VividLe/awesome-weakly-supervised-action-localization)
- [Awesome Weakly-supervised Object Localization](https://github.com/gyguo/awesome-weakly-supervised-object-localization)

## Stargazers over time
[![Stargazers over time](https://starchart.cc/gyguo/awesome-weakly-supervised-semantic-segmentation.svg?variant=adaptive)](https://starchart.cc/gyguo/awesome-weakly-supervised-semantic-segmentation)
