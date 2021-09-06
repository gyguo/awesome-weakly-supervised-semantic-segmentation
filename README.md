# Awesome Weakly-supervised Semantic Segmentation （Image）

------

**<u>Contact me if any paper is missed!</u>**

------



# 1. Performance

### 1.1. Results on PASCAL VOC 2012 dataset

- For each method, I will provide the name of baseline in brackets if it has. 
- **Sup.:** **I**-image-level class label, **B**-bounding box label, **S**-scribble label, **P**-point label.
- **Bac. C:** Method for generating pseudo label, or backbone of the classification network.
- **Arc. S:** backbone and method of the segmentation network. 
- **Pre.s :**  The dataset used to pre-train the **segmentation** network, **"*I*"** denotes ImageNet, "***C***" denotes COCO. **Note that many works use COCO pre-trained DeepLab model but not mentioned in the paper.** 
- For methods that use multiple backbones, I only reports the results of **ResNet101**.
- **"-"** indicates no fully-supervised model is utilized, **"?"** indicates the corresponding item is not mentioned in the paper.

|        Method        |    Pub.     |    Bac. C     |         Arc. S          | Sup.  |       Extra data        |       Pre.S       | val  | test |
| :------------------: | :---------: | :-----------: | :---------------------: | :---: | :---------------------: | :--: | -------------------- | -------------------- |
| AF-SS | ECCV16 |  |  | **I** | Saliency |  | 52.6 | 52.7 |
| CrawlSeg | CVPR17 |               |                         | **I** | YouTube Videos |                         | 58.1 | 58.7 |
| SeeNet | NeurIPS18 | | | **I** | Saliency | | 63.1 | 62.8 |
| WebS-i2 | CVPR17 | | | **I** | Web | |  |  |
| TPL | ICCV17 | | | **I** | Saliency | |  |  |
| GAIN | CVPR18 | | | **I** | Saliency | |  |  |
| Oh *et al*. | CVPR17 | | | **I** | Saliency | |  |  |
| SEC | ECCV16 | VGG16 | VGG16 DeepLabv1 | **I** | Saliency | ***I*** | 50.7 | 51.7 |
|      DSRG (SEC)      |  CVPR18   | VGG16 |  ResNet101 DeepLabv2   | **I** |     Saliency      |     ***I***     | 61.4 | 63.2 |
| AISI | ECCV18 | ResNet101 | ResNet101 DeepLabv2 | **I** | Saliency | **?** | 63.6 | 64.5 |
|   Ficklenet (DSRG)   |  CVPR19   | VGG16 |  ResNet101 DeepLabv2   | **I** |     Instance-level Saliency     | ***I*** | 64.9 | 65.3 |
| AISI | ECCV18 | ResNet101 | ResNet101 DeepLabv2 | **I** | Instance-level Saliency<br/>24KImageNet | **?** | 64.5 | 65.6 |
|         OAA         |   ICCV19   | VGG16 | ResNet101 DeepLabv1 | **I** | Saliency | ***I*** | 65.2 | 66.4 |
| Zhang *et al.* | ECCV20 | ResNet50 | ResNet50 DeepLabv2 | **I** | Saliency |   **?**   | 66.6 | 66.7 |
|     Fan *et al.*     |  ECCV20   | ResNet38 |  ResNet101 DeepLabv1   | **I** |     Saliency      | **?** | 67.2 | 66.7 |
|     MCIS     |  ECCV20   |    VGG16     |  ResNet101 DeepLabv1   | **I** |     Saliency      |   **?** | 66.2 | 66.9 |
| Lee *et al.* | ICCV19 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliency Web | ***I*** | 66.5 | 67.4 |
|         LIID         |  PAMI20   |   ResNet50   |  ResNet101 DeepLabv2   | **I** |     Saliency      | **?** | 66.5 | 67.5 |
|     MCIS     |  ECCV20   |    VGG16     |  ResNet101 DeepLabv1   | **I** | Saliency Web | **?** | 67.7 | 67.5 |
| ICD | CVPR20 | VGG16 | ResNet101 DeepLabv1 | **I** | Saliency | **?** | 67.8 | 68.0 |
| LIID | PAMI20 | ResNet50 | ResNet101 DeepLabv2 | **I** | Saliency<br />24KImageNet | **?** | 67.8 | 68.3 |
| Li *et al.* | AAAI21 | ResNet101 | ResNet101 DeepLabv2 | **I** | Saliency | **?** | 68.2 | 68.5 |
| Yao et al. | CVPR21 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliency | ***I*** | 68.3 | 68.5 |
| AuxSegNet | ICCV21 | ResNet38 | - | **I** | Saliency | **?** | 69.0 | 68.6 |
| SPML (Ficklenet) | ICLR21 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliency | ***I*** | 69.5 | 71.6 |
| Yao et al. | CVPR21 | VGG16 | ResNet101 DeepLabv2 | **I** | Saliency | ***I+C*** | 70.4 | 70.2 |
| EDAM | CVPR21 | ResNet38 | ResNet101 DeepLabv2 | **I** | Saliency | **?** | 70.9 | 70.6 |
|         DRS          |  AAAI21   |    VGG16     |  ResNet101 DeepLabv2   | **I** | Saliency | **?** | 71.2 | 71.4 |
|                      |             |               |                         |       |                         |                         |      |      |
| AffinityNet | CVPR18 | ResNet38 | ResNet38 | **I** | - | **?** | 61.7 | 63.7 |
| ICD | CVPR20 | VGG16 | ResNet101 DeepLabv1 | **I** | - | **?** | 64.1 | 64.3 |
|         IRN          |  CVPR19   | ResNet50 | ResNet50 DeepLabv2 | **I** | - | ***I*** | 63.5 | 64.8 |
| IAL | IJCV20 | ResNet? | ResNet? | **I** | - | ***I*** | 64.3 | 65.4 |
|         SSDD (PSA)         |  ICCV19   | ResNet38 | ResNet38 | **I** |            -            |            ***I***            | 64.9 | 65.5 |
|         SEAM         |  CVPR20   | ResNet38 | ResNet38 DeepLabv2 | **I** |            -            |            ***I***            | 64.5 | 65.7 |
|    Chang *et al.*    |  CVPR20   | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 66.1 | 65.9 |
|         RRM          |  AAAI20   | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            | **?** | 66.3 | 66.5 |
|    BES  |  ECCV20   |   ResNet50    |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 65.7 | 66.6 |
|    CONTA (+SEAM)     | NeurIPS20 | ResNet38 |  ResNet101 DeepLabv2   | **I** |            -            |            **?**            | 66.1 | 66.7 |
| Ru *et al.* | IJCAI21 | ResNet101 | ResNet101 DeepLabv2 | **I** | - | **?** | 67.2 | 67.3 |
| WSGCN (IRN) | ICME21 | ResNet50 | ResNet101 DeepLabv2 | **I** | - | **I** | 66.7 | 68.8 |
|         CPN          |  ICCV21   | ResNet38  |  ResNet38 DeepLabv1  | **I** |                    -                    |   **?**   | 67.8 | 68.5 |
| RPNet | arxiv21 | ResNet101 | ResNet50 DeepLabv2 | **I** | - | ***I*** | 68.0 | 68.2 |
|        AdvCAM        |  CVPR21   |   ResNet50   |  ResNet101 DeepLabv2   | **I** |            -            |            ***I***            | 68.1 | 68.0 |
| WSGCN (IRN) | ICME21 | ResNet50 | ResNet101 DeepLabv2 | **I** | - | **I+C** | 68.7 | 69.3 |
|  |  |  |  |  |  |  |  |  |
| | | | | | | | | |
|        Boxsup        |  ICCV15   |           |                      |       |                                         |           |      |      |
|         BBAM         |  CVPR21   |     ?     | ResNet101 DeepLabv2  | **B** |                   MCG                   |  ***I***  | 73.7 | 73.7 |
|         WSSL         |  ICCV15   |      -      |    VGG16 DeepLabv1     | **B** |            -            |            ***I***            | 60.6 | 62.2 |
|    Song *et al.*     |  CVPR19   |      -      |  ResNet101  DeepLabv1  | **B** |            -            |            ***I***            | 70.2 |  -   |
| SPML (Song *et al.*) |  ICLR21   |      -      |  ResNet101  DeepLabv2  | **B** |            -            |            ***I***            | 73.5 | 74.7 |
| Oh *et al.* | CVPR21 | ResNet101 | ResNet101 DeepLabv2 | **B** | - | ***I+C*** | 74.6 | 76.1 |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
| Scribblesup | | | | **S** | | | | |
|      NormalCut       |  CVPR18   |       -       |  ResNet101 DeepLabv1   | **S** |            Saliency            |            **?**            | 74.5 |  -   |
|      KernelCut       |  ECCV18   |       -       |  ResNet101 DeepLabv1   | **S** |            -            |            **?**            | 75.0 |  -   |
|         BPG          |  IJCAI19  |       -       |  ResNet101 DeepLabv2   | **S** |            -            |            **?**            | 76.0 |  -   |
|   SPML (KernelCut)   |  ICLR21   |       -       |  ResNet101 DeepLabv2  | **S** |            -            |  ***I***  | 76.1 |  -   |
| A2GNN | TPAMI21 | - | ? | **S** | - | **?** | 76.2 | 76.1 |
| DFR | arxiv21 | - | UperNet+Swin  Transformer | **S** | 22KImageNet | - | 82.8 | 82.9 |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|      WhatsPoint      |  ECCV16   |  -  |            VGG16 FCN            | **P** |    Objectness     | ***I*** | 46.1 |  -   |
|         PCAM         |  arxiv20  |   ResNet50   |      DeepLabv3+       | **P** |            -            |            **?**            | 70.5 |  -   |
|  |  |  |  |  |  |  |  |  |

### 1.2. Results on MS-COCO dataset

**TODO**

# 2. Paper List

## 2.1. supervised by image tags (I)

**2021**

- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **Li *et al.*:** "Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation" *AAAI2021*
- **DRS:** "Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation" *AAAI2021*
- **AdvCAM:** " Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation" *CVPR2021*
- **Yao et al.  **: "Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation" *CVPR2021*
- **EDAM:** "Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation" *CVPR2021*
- **WSGCN:** "Weakly-Supervised Image Semantic Segmentation Using Graph Convolutional Networks" *ICME2021*
- **PuzzleCAM:** "Puzzle-CAM Improved localization via matching partial and full features" *2021arXiv*
- **CDA:** "Context Decoupling Augmentation for Weakly Supervised Semantic Segmentation" *2021arXiv*
- **Ru *et al.*:** "Learning Visual Words for Weakly-Supervised Semantic Segmentation" *IJCAI2021*
- **AuxSegNet:** "Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation" *ICCV2021*
- **CPN:** "Complementary Patch for Weakly Supervised Semantic Segmentation" *ICCV2021*
- **RPNet:** "Cross-Image Region Mining with Region Prototypical Network for Weakly Supervised Segmentation" *arxiv2021*
- **Method:** "" *2021*

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
- **Method:** "Find it if You Can: End-to-End Adversarial Erasing for Weakly-Supervised Semantic Segmentation" *2020arXiv*
- **Zhang *et al.*:** "Splitting vs. Merging: Mining Object Regions with Discrepancy and Intersection Loss for Weakly Supervised Semantic Segmentation" *ECCV2020*

**2019**

- **IRN:** "Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations" *CVPR2019*
- **Ficklenet:** " Ficklenet: Weakly and semi-supervised semantic image segmentation using stochastic inference" *CVPR2019*
- **Lee *et al.*:** "Frame-to-Frame Aggregation of Active Regions in Web Videos for Weakly Supervised Semantic Segmentation" *ICCV2019*
- **OAA:** "Integral Object Mining via Online Attention Accumulation" *ICCV2019*
- **SSDD:** "Self-supervised difference detection for weakly-supervised semantic segmentation" *ICCV2019*

**2018**

- **DSRG:** "Weakly-supervised semantic segmentation network with deep seeded region growing" *CVPR2018*
- **AffinityNet:** "Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation" *CVPR2018*
- **GAIN:** " Tell me where to look: Guided attention inference network" *CVPR2018*
- **AISI:** "Associating inter-image salient instances for weakly supervised semantic segmentation" *ECCV2018*
- **SeeNet:** "Self-Erasing Network for Integral Object Attention" *NeurIPS2018*
- **Method:** "" *2018*

**2017**

- **CrawlSeg:** "Weakly Supervised Semantic Segmentation using Web-Crawled Videos" *CVPR2017*
- **WebS-i2:** "Webly supervised semantic segmentation" *CVPR2017*
- **Oh *et al*.:** "Exploiting saliency for object segmentation from image level labels" *CVPR2017*
- **TPL:** "Two-phase learning for weakly supervised object localization" *ICCV2017*

**2016**

- **SEC:** "Seed, expand and constrain: Three principles for weakly-supervised image segmentation" *ECCV2016*
- **AF-SS:** "Augmented Feedback in Semantic Segmentation under Image Level Supervision" *2016*

### 2.2. Supervised by bounding box (B)

- **WSSL:** "Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation" *ICCV2015*
- **Boxsup:** "Boxsup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation" *ICCV2015*
- **Scribblesup:** "Scribblesup: Scribble-supervised convolutional networks for semantic segmentation" *CVPR2016*
- **Song *et al.*:** "Box-driven class-wise region masking and filling rate guided loss for weakly supervised semantic segmentation" *CVPR2019*
- **BBAM:** "BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation" *CVPR2021*
- **Oh *et al.*:** "Ba ckground-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation" CVPR2021
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*

### 2.3. Supervised by scribble (S)

- **NormalCut :** "Normalized cut loss for weakly-supervised cnn segmentation" *CVPR2018*
- **KernelCut :** "On regularized losses for weakly-supervised cnn segmentation" *ECCV2018*
- **BPG:**  "Boundary Perception Guidance: A Scribble-Supervised Semantic Segmentation Approach" *IJCAI2019*
- **SPML:** "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning" *ICLR2021*
- **DFR:** "Dynamic Feature Regularized Loss for Weakly Supervised Semantic Segmentation" *arxiv2021*
- **A2GNN:** "Affinity attention graph neural network for weakly supervised semantic segmentation" *TPAMI2021*
- **Method:** "" *2021*

### 2.4. Supervised by point (P)

- **WhatsPoint:** "What’s the Point: Semantic Segmentation with Point Supervision" *ECCV2016*
- **PCAM:** "PCAMs: Weakly Supervised Semantic Segmentation Using Point Supervision" *arxiv2020*

# 3. Dataset

##### PASCAL VOC 2012

##### MS COCO