# Awesome Weakly-supervised Image Segmentation

------

**<u>Contact me if any paper is missed!</u>**

------

[TOC]

## 1. Semantic Segmentation

### 1.1. Weakly Supervised Semantic Segmentation performance on PASCAL VOC 2012 dataset

- For each method, I will provide the name of baseline in brackets if it has. 
- **Sup.:** I-image-level class label, B-bounding box label, S-scribble label, P-point label.
- **Bac. P:** backbone or method for generating pseudo label
- **Arc. F:** backbone and method of the fully-supervised branch. "-" indicates no fully-supervised model is utilized, "?" indicates the corresponding item is unknown
- For methods that use multiple backbones, I only reports the results of **ResNet-101**

|       Method       |    Pub.     |    Bac. P     |        Arc. F         | Sup.  |    Extra data    | val  | test |
| :----------------: | :---------: | :-----------: | :-------------------: | :---: | :--------------: | :--: | :--: |
|        BBAM        |  CVPR2021   |       ?       | ResNet-101 DeepLabv2  | **B** |       MCG        | 73.7 | 73.7 |
|     Oh et al.      |  CVPR2021   |  ResNet-101   | ResNet-101 DeepLabv2  | **B** |     MS-COCO      | 74.6 | 76.1 |
|        WSSL        |  ICCV2015   |      CRF      |   VGG-16 DeepLabv1    | **B** |        -         | 60.6 | 62.2 |
|    Song et al.     |  CVPR2019   |      CRF      | ResNet-101  DeepLabv1 | **B** |        -         | 70.2 |  -   |
| SPML (Song et al.) |  ICLR2021   |      CRF      | ResNet-101  DeepLabv1 | **B** |        -         | 73.5 | 74.7 |
|                    |             |               |                       |       |                  |      |      |
|     NormalCut      |  CVPR2018   |       -       | ResNet-101 DeepLabv1  | **S** |        -         | 74.5 |  -   |
|     KernelCut      |  ECCV2018   |       -       | ResNet-101 DeepLabv1  | **S** |        -         | 75.0 |  -   |
|        BPG         |  IJCAI2019  |       -       | ResNet-101 DeepLabv2  | **S** |        -         | 76.0 |  -   |
|  SPML (KernelCut)  |  ICLR2021   |       -       | ResNet-101 DeepLabv1  | **S** |        -         | 76.1 |  -   |
|                    |             |               |                       |       |                  |      |      |
|     WhatsPoint     |  ECCV2016   |  VGG-16 FCN   |           -           | **P** | Objectness Prior | 46.1 |  -   |
|        PCAM        |  arxiv2020  |   ResNet-50   |     ? DeepLabv3+      | **P** |        -         | 70.5 |  -   |
|                    |             |               |                       |       |                  |      |      |
|     DSRG (SEC)     |  CVPR2018   |               | ResNet-101 DeepLabv2  | **I** |      MSRA-B      | 61.4 | 63.2 |
|  Ficklenet (DSRG)  |  CVPR2019   |               | ResNet-101 DeepLabv2  | **I** |      MSRA-B      | 64.9 | 65.3 |
|  SPML (Ficklenet)  |  ICLR2021   |               | ResNet-101 DeepLabv2  | **I** |      MSRA-B      | 69.5 | 71.6 |
|        DRS         |  AAAI2021   |    VGG-16     | ResNet-101 DeepLabv2  | **I** |      MSRA-B      | 71.2 | 71.4 |
|                    |             |               |                       |       |                  |      |      |
|   CONTA (+SEAM)    | NeurIPS2020 | WideResNet-38 | ResNet-101 DeepLabv2  | **I** |        -         | 66.1 | 66.7 |
|        RRM         |  AAAI2020   | WideResNet-38 | ResNet-101 DeepLabv2  | **I** |        -         | 66.3 | 66.5 |
|        LIID        |             |               |                       | **I** |                  | 66.5 | 67.5 |
|                    |             |               |                       | **I** |                  |      |      |
|                    |             |               |                       |       |                  |      |      |
|       AdvCAM       |  CVPR2021   |   ResNet-50   | ResNet-101 DeepLabv2  | **I** |        -         | 68.1 | 68.0 |
|     Li et al.      |  AAAI2021   |  ResNet-101   | ResNet-101 DeepLabv2  | **I** |        -         | 68.2 | 68.5 |

### 1.2. Semantic Segmentation supervised by image tags (I)

- **Method:** "" *2020*
- **DSRG:** "Weakly-supervised semantic segmentation network with deep seeded region growing" *CVPR2018*
- **Ficklenet:** " Ficklenet: Weakly and semi-supervised semantic image segmentation using stochastic inference" *CVPR2019*
- **Method:** "" *2020*
- **Method:** "Reliability Does Matter An End-to-End Weakly Supervised Semantic Segmentation Approach" *AAAI2020*
- **CONTA:** "Causal intervention for weakly-supervised semantic segmentation" *NeurIPS2020*
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **AdvCAM:** " Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation" *CVPR2021*
- **Li et al.:** "Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation" *AAAI2021*
- **DRS:** "Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation" *AAAI2021*
- **Method:** "" *2020*

### 1.3. Semantic Segmentation supervised by bounding box (B)

- **WSSL:** "Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation" *ICCV2015*
- **Song et al.:** "Box-driven class-wise region masking and filling rate guided loss for weakly supervised semantic segmentation" *CVPR2019*
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