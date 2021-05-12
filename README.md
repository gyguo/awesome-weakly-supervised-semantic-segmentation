# Awesome Weakly-supervised Semantic Segmentation （Image）

------

**<u>Contact me if any paper is missed!</u>**

------



# 1. Performance

### 1.1. Results on PASCAL VOC 2012 dataset

- For each method, I will provide the name of baseline in brackets if it has. 
- **Sup.:** I-image-level class label, B-bounding box label, S-scribble label, P-point label.
- **Bac. C:** Method for generating pseudo label, or backbone of the classification network.
- **Arc. S:** backbone and method of the segmentation network. 
- **Pre.s :**  The dataset used to pre-train the **segmentation** network, **"*I*"** denotes ImageNet, "***C***" denotes COCO. **Note that many works use COCO pre-trained DeepLab model but not mentioned in the paper.** 
- For methods that use multiple backbones, I only reports the results of **ResNet101**.
- **"-"** indicates no fully-supervised model is utilized, **"?"** indicates the corresponding item is not mentioned in the paper.

|        Method        |    Pub.     |    Bac. C     |         Arc. S          | Sup.  |       Extra data        |       Pre.S       | val  | test |
| :------------------: | :---------: | :-----------: | :---------------------: | :---: | :---------------------: | :--: | -------------------- | -------------------- |
|         BBAM         |  CVPR2021   |       ?       |  ResNet101 DeepLabv2   | **B** |           MCG           |           ***I***           | 73.7 | 73.7 |
|     Oh *et al.*      |  CVPR2021   |  ResNet101   |  ResNet101 DeepLabv2   | **B** |            -             |    ***I+C***    | 74.6 | 76.1 |
|         WSSL         |  ICCV2015   |      -      |    VGG16 DeepLabv1     | **B** |            -            |            ***I***            | 60.6 | 62.2 |
|    Song *et al.*     |  CVPR2019   |      -      |  ResNet101  DeepLabv1  | **B** |            -            |            ***I***            | 70.2 |  -   |
| SPML (Song *et al.*) |  ICLR2021   |      -      |  ResNet101  DeepLabv2  | **B** |            -            |            ***I***            | 73.5 | 74.7 |
|                      |             |               |                         |       |                         |                         |      |      |
|      NormalCut       |  CVPR2018   |       -       |  ResNet101 DeepLabv1   | **S** |            Saliancy            |            **?**            | 74.5 |  -   |
|      KernelCut       |  ECCV2018   |       -       |  ResNet101 DeepLabv1   | **S** |            -            |            **?**            | 75.0 |  -   |
|         BPG          |  IJCAI2019  |       -       |  ResNet101 DeepLabv2   | **S** |            -            |            **?**            | 76.0 |  -   |
|   SPML (KernelCut)   |  ICLR2021   |       -       |  ResNet101 DeepLabv2  | **S** |            -            |  ***I***  | 76.1 |  -   |
|                      |             |               |                         |       |                         |                         |      |      |
|      WhatsPoint      |  ECCV2016   |  -  |            VGG16 FCN            | **P** |    Objectness     | ***I*** | 46.1 |  -   |
|         PCAM         |  arxiv2020  |   ResNet50   |      ? DeepLabv3+       | **P** |            -            |            **?**            | 70.5 |  -   |
|                      |             |               |                         |       |                         |                         |      |      |
| SEC | ECCV2016 | VGG16 | VGG16 DeepLabv1 | **I** | Saliancy | ***I*** | 50.7 | 51.7 |
|      DSRG (SEC)      |  CVPR2018   | VGG16 |  ResNet101 DeepLabv2   | **I** |     Saliancy      |     ***I***     | 61.4 | 63.2 |
| Fan *et al.* | ECCV2018 | ResNet101 | ResNet101 DeepLabv2 | **I** | Saliancy | **?** | 63.6 | 64.5 |
|   Ficklenet (DSRG)   |  CVPR2019   | VGG16 |  ResNet101 DeepLabv2   | **I** |     Saliancy      | ***I*** | 64.9 | 65.3 |
| Fan *et al.* | ECCV2018 | ResNet101 | ResNet101 DeepLabv2 | **I** | Saliancy<br/>24KImageNet | **?** | 64.5 | 65.6 |
|         OAA         |   ICCV2019   | VGG16 | ResNet101 DeepLabv1 | **I** | Saliancy | ***I*** | 65.2 | 66.4 |
|     Fan *et al.*     |  ECCV2020   | ResNet38 |  ResNet101 DeepLabv1   | **I** |     Saliancy      | **?** | 67.2 | 66.7 |
|     MCIS     |  ECCV2020   |    VGG16     |  ResNet101 DeepLabv1   | **I** |     Saliancy      |   **?** | 66.2 | 66.9 |
| Lee *et al.* | ICCV2019 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliancy Web | ***I*** | 66.5 | 67.4 |
|         LIID         |  PAMI2020   |   ResNet50   |  ResNet101 DeepLabv2   | **I** |     Saliancy      | **?** | 66.5 | 67.5 |
|     MCIS     |  ECCV2020   |    VGG16     |  ResNet101 DeepLabv1   | **I** | Saliancy Web | **?** | 67.7 | 67.5 |
| ICD | CVPR2020 | VGG16 | ResNet101 DeepLabv1 | **I** | Saliancy | **?** | 67.8 | 68.0 |
| LIID | PAMI2020 | ResNet50 | ResNet101 DeepLabv2 | **I** | Saliancy<br>24KImageNet | **?** | 67.8 | 68.3 |
| Li *et al.* | AAAI2021 | ResNet101 | ResNet101 DeepLabv2 | **I** | Saliancy | **?** | 68.2 | 68.5 |
| Yao et al. | CVPR2021 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliancy | ***I*** | 68.3 | 68.5 |
| Yao et al. | CVPR2021 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliancy | ***I+C*** | 70.4 | 70.2 |
|         DRS          |  AAAI2021   |    VGG16     |  ResNet101 DeepLabv2   | **I** | Saliancy | **?** | 71.2 | 71.4 |
|   SPML (Ficklenet)   |  ICLR2021   |    VGG16     |  ResNet101 DeepLabv2  | **I** |     Saliancy      | ***I*** | 69.5 | 71.6 |
|                      |             |               |                         |       |                         |                         |      |      |
|                      |             |               |                         |       |                         |                         |      |      |
| ICD | CVPR2020 | VGG16 | ResNet101 DeepLabv1 | **I** | - | **?** | 64.1 | 64.3 |
|         IRN          |  CVPR2019   | ResNet50 | ResNet50 DeepLabv2 | **I** | - | ***I*** | 63.5 | 64.8 |
| IAL | IJCV20 | ResNet? | ResNet? | **I** | - | ***I*** | 64.3 | 65.4 |
|         SSDD (PSA)         |  ICCV2019   | ResNet38 | ResNet38 | **I** |            -            |            ***I***            | 64.9 | 65.5 |
|         SEAM         |  CVPR2020   | ResNet38 | ResNet38 DeepLabv2 | **I** |            -            |            ***I***            | 64.5 | 65.7 |
|    Chang *et al.*    |  CVPR2020   | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 66.1 | 65.9 |
|         RRM          |  AAAI2020   | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            | **?** | 66.3 | 66.5 |
|    BES  |  ECCV2020   |   ResNet50    |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 65.7 | 66.6 |
|    CONTA (+SEAM)     | NeurIPS2020 | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 66.1 | 66.7 |
|        AdvCAM        |  CVPR2021   |   ResNet-50   |  ResNet101 DeepLabv2   | **I** |            -            |            ***I***            | 68.1 | 68.0 |
|                      |             |           |                      |       |                          |           |      |      |

### 1.2. Results on MS-COCO dataset

**TODO**

# 2. Paper List

## 2.1. supervised by image tags (I)

**2016**

- **SEC:** "Seed, expand and constrain: Three principles for weakly-supervised image segmentation" *ECCV2016*

**2018**

- **DSRG:** "Weakly-supervised semantic segmentation network with deep seeded region growing" *CVPR2018*
- **Fan *et al.*:** "Associating inter-image salient instances for weakly supervised semantic segmentation" *ECCV2018*

**2019**

- **IRN:** "Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations" *CVPR2019*

- **Ficklenet:** " Ficklenet: Weakly and semi-supervised semantic image segmentation using stochastic inference" *CVPR2019*
- **Lee *et al.*:** "Frame-to-Frame Aggregation of Active Regions in Web Videos for Weakly Supervised Semantic Segmentation" *ICCV2019*
- **OAA:** "Integral Object Mining via Online Attention Accumulation" *ICCV2019*
- **SSDD:** "Self-supervised difference detection for weakly-supervised semantic segmentation" *ICCV2019*
- **Method:** "" *2019*

**2020**

- **RRM:** "Reliability Does Matter An End-to-End Weakly Supervised Semantic Segmentation Approach" *AAAI2020*
- **IAL:** "Weakly-Supervised Semantic Segmentation by Iterative Affinity Learning" *IJCV2020*
- **SEAM:** "Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation" *CVPR2020*
- **Chang *et al.*:** "Weakly-Supervised Semantic Segmentation via Sub-category Exploration" *CVPR2020*
- **ICD:** "Learning Integral Objects with Intra-Class Discriminator for Weakly-Supervised Semantic Segmentation" *CVPR2020*
- **Fan *et al.*:** "Employing multi-estimations for weakly-supervised semantic segmentation" *ECCV2020*
- **MCIS:** "Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation" *2020*
- **BES:** "Weakly Supervised Semantic Segmentation with Boundary Exploration" *ECCV2020*
- **CONTA:** "Causal intervention for weakly-supervised semantic segmentation" *NeurIPS2020*

**2021**

- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **Li *et al.*:** "Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation" *AAAI2021*
- **DRS:** "Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation" *AAAI2021*
- **AdvCAM:** " Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation" *CVPR2021*
- **Yao et al.  **: "Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation" *CVPR2021*
- **Method:** "" *2021*

### 2.2. Supervised by bounding box (B)

**2015**

- **WSSL:** "Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation" *ICCV2015*

**2019**

- **Song *et al.*:** "Box-driven class-wise region masking and filling rate guided loss for weakly supervised semantic segmentation" *CVPR2019*

**2021**

- **BBAM:** "BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation" *CVPR2021*
- **Oh *et al.*:** "Ba ckground-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation" CVPR2021
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 2.3. Supervised by scribble (S)

**2018**

- **NormalCut :** "Normalized cut loss for weakly-supervised cnn segmentation" *CVPR2018*
- **KernelCut :** "On regularized losses for weakly-supervised cnn segmentation" *ECCV2018*

**2019**

- **BPG:**  "Boundary Perception Guidance: A Scribble-Supervised Semantic Segmentation Approach" *IJCAI2019*

**2020**

- **Method:** "" *2020*

**2021**

- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 2.4. Supervised by point (P)

- **WhatsPoint:** "What’s the Point: Semantic Segmentation with Point Supervision" *ECCV2016*
- **PCAM:** "PCAMs: Weakly Supervised Semantic Segmentation Using Point Supervision" *arxiv2020*

# 3. Dataset

##### PASCAL VOC 2012

##### MS COCO