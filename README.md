# Awesome Weakly-supervised Image Segmentation

------

**<u>Contact me if any paper is missed!</u>**

------

[TOC]

## 1. Semantic Segmentation

### 1.1. Weakly Supervised Semantic Segmentation performance on PASCAL VOC 2012 dataset

**Sup.:** I-image-level class label, B-bounding box label, S-scribble label, P-point label.

|   Method   |   Pub.    |  Backbone  | Sup.  | Extra data | val  | test |
| :--------: | :-------: | :--------: | :---: | :--------: | :--: | :--: |
| Oh et al.  | CVPR2021  | ResNet-101 | **B** |  MS-COCO   | 74.6 | 76.1 |
|    WSSL    | ICCV2015  |   VGG-16   | **B** |     -      | 60.6 | 62.2 |
|    BBAM    | CVPR2021  | ResNet101  | **B** |     -      | 73.7 | 73.7 |
|    SPML    | ICLR2021  | ResNet101  | **B** |     -      | 73.5 | 74.7 |
|            |           |            |       |            |      |      |
| NormalCut  | CVPR2018  | ResNet101  | **S** |     -      | 74.5 |  -   |
| KernelCut  | ECCV2018  | ResNet101  | **S** |     -      | 75.0 |  -   |
|    BPG     | IJCAI2019 | ResNet101  | **S** |     -      | 76.0 |  -   |
|    SPML    | ICLR2021  | ResNet101  | **S** |     -      | 76.1 |  -   |
|            |           |            |       |            |      |      |
| WhatsPoint | ECCV2016  |   VGG-16   | **P** |     -      | 46.1 |  -   |
|    PCAM    | arxiv2020 |  ResNet50  | **P** |     -      | 70.5 |  -   |
|            |           |            |       |            |      |      |
|    DSRG    | CVPR2018  | ResNet-101 | **I** |   MSRA-B   | 61.4 | 63.2 |
| Ficklenet  | CVPR2019  | ResNet-101 | **I** |   MSRA-B   | 64.9 | 65.3 |
|            |           |            |       |            |      |      |
|   CONTA    | NeurIPS20 | ResNet101  | **I** |     -      | 66.1 | 66.7 |
|            |           |            | **I** |            |      |      |
|            |           |            | **I** |            |      |      |
|            |           |            | **I** |            |      |      |
|    SPML    | ICLR2021  | ResNet101  | **I** |     -      | 69.5 | 71.6 |
|            |           |            | **I** |            |      |      |
|            |           |            | **I** |            |      |      |
|            |           |            | **I** |            |      |      |
|            |           |            | **I** |            |      |      |

### 1.2. Semantic Segmentation supervised by image-level class Only (I)

- **Method:** "" *2020*
- **DSRG:** "Weakly-supervised semantic segmentation network with deep seeded region growing" *CVPR2018*
- **Ficklenet:** " Ficklenet: Weakly and semi-supervised semantic image segmentation using stochastic inference" *CVPR2019*
- **CONTA:** "Causal intervention for weakly-supervised semantic segmentation" *NeurIPS20*
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **Method:** "" *2020*

### 1.3. Semantic Segmentation supervised by bounding box (B)

- **WSSL:** "Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation" *ICCV2015*
- **Method:** "" *2020*
- **BBAM:** "BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation" *CVPR2021*
- **Oh et al.:** "Ba ckground-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation" CVPR2021
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 1.4. Semantic Segmentation supervised by scribble (S)

- **NormalCut :** "Normalized cut loss for weakly-supervised cnn segmentation" *CVPR2018*
- **KernelCut :** "On regularized losses for weakly-supervised cnn segmentation" *ECCV2018*
- **BPG:**  "Boundary Perception Guidance: A Scribble-Supervised Semantic Segmentation Approach" *IJCAI2019*
- **Method:** "" *2020*
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 1.5. Semantic Segmentation supervised by point (P)

- **WhatsPoint:** "What’s the Point: Semantic Segmentation with Point Supervision" *ECCV2016*
- **PCAM:** "PCAMs: Weakly Supervised Semantic Segmentation Using Point Supervision" *arxiv2020*

## 2. Instance Segmentation

##### Todo

## 3. Panoptic segmentation

##### Todo

## 4. Dataset

##### PASCAL VOC 2012

##### MS COCO