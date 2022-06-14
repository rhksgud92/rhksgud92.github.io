---
layout: post
title:  "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers - 리뷰"

categories:
  - Image Segmentation
  - Semantic Segmentation
  - Transformer

tags:
  - Image Segmentation
  - Semantic Segmentation
  - Transformer

---

# 2022_06_14

# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers - 리뷰

최근 몇년동안 NLP 쪽에서는 Transformer 모델이 크게 좋은 성능들을 내며 각광받게 되었습니다. 이를 기반으로 BERT, GPT등 유명한 Large-scale 자연어처리 모델들이 나왔고, 음성인식에서도 Wav2Vec2.0, 그리고 이미지 쪽에서도 DETR, SETR, UNETR, ViT, NLP와 Image를 multi-modal로 처리하는 Pixel-BERT, ViLT 등 다양한 모델들이 나왔습니다. 이번엔 흔히들 사용하는 FCNN대신 Transformer를 사용하여 2D Semantic image segmentation을 한 Segformer 논문을 보겠습니다. Segformer는 Transformer encoder blocks들의 각 output들을 multi-scale features로 사용하고, 간단한 MLP decoder 로 각각의 다른 layers들의 latent vectors를 aggregate하여 local, global한 두가지의 이미지의 특성을 기반으로 powerful 한 representations를 만들었다고 주장합니다.

논문링크: https://arxiv.org/abs/2105.15203

### 1. Introduction

본 논문은 세가지 Novelties를 강조합니다.

1. 보통 Transformer Architecture가 쓰이면 같이 항상 쓰이는 Positional encoding이 없다는 점
2. Light weight하면서도 높은 성능을 보인 MLP 디코더의 사용
3. Efficiency, accuracy, robustness를 공개된 3개의 semantic segmentation datasets에 대하여 SOTA달성입니다.

위 1번처럼 ViT는 Pretraining때와 fine-tuning때 다른 resolution의 데이터를 학습하고 resolution이 다르기때문에 pretraining때 2d-interpolation을 사용하여 학습했었는데요. 여기 Segformer에선 다른 resolution을 사용하긴하지만 positional encoding을 위한 interpolation을 안하기 때문에 arbitrary test resolution에도 performance degrade가 없다고 합니다. 또한 single fixed low-resolution feature map만 뽑아내는 ViT와 다르게 hierarchical 한 구조로 여러 단계에서 high(coarse features) ~ low(fine features) resolution feature maps를 뽑아내기때문에 다양한 representation 을 보게하는 이점도 있다고 합니다. 또한 본 논문에서의 MLP decoder로 lower layers의 local한 특성과 higher layers의 global 한 attentions들을 합하여 segmentation을 한다고 합니다.

본 논문의 성능은 ADE20K, Cityscapes, COCO-Stuff 세가지의 datasets을 가지고 좋은 성능을 달성했다고 합니다.

### 2. Methods

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220609110324072.png" alt="">

***ENCODER***

1. **Hierarchical Feature Representation:** 먼저 본 논문은 H x W x 3 사이즈의 이미지를 4x4의 패치들로 나누어줍니다. 논문에선 ViT와 다르게 더 작은 패치를 사용함으로써 더 좋은 dense prediction task 를 수행할 수 있었다고 합니다. 먼저 구조를 크게 보자면 Segformer에선 Hierarchical Feature Representation 구조로 transformer encoder blocks들을 사용했습니다. 항상 block output 별로 resolution이 같던 ViT와 다르게 각 block에선 H/(2^(i+1)) x W/(2^(i+1)) x C_i where i = 1, 2, 3, 4에의 representation의 resolution의 output들이 나오게 됩니다 (오른쪽으로 갈수록 low-resolution fine-grained). 끝으로 decoder를 통해 H/4 x W/4 x N_cls의 output이 나오는데 여기서 N_cls는 categories의 개수입니다. 
2. **Overlapped Patch Merging:** ViT에서 input 쪽에서 overlapping없이 자른 patches들을 1 x 1 x C 처럼 linear projection을 통해서 만들었듯이 여기서도 2 x 2 x C들을 1 x 1 x C_i로 만들어줍니다. 이렇게 해서 i=1 -> i =2로 넘어가는거죠 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220613023126823.png" alt="">. 또한 non-overlapping대신 overlapping patches를 사용하면서 local continuity를 preserve합니다 마치 CNN에서 sharing filter가 조금씩 움직이는것처럼. 논문에서 나올 K는 patch size, S는 stride, P는 padding size입니다. 
3. **Efficient Self-Attention:** 본논문에선 self-attention에서의 계산량을 줄이기위해 length의 일부를 channgel (dimension)으로 reshape을 합니다. Ratio (R)만큼 길이를 나누고 그만큼 Channel을 늘려줍니다. 이러면 inner-product가 길어질 뿐이게 되죠.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220613022213437.png" alt="">
4. **Mix-FFN:** ViT에선 pretraining 때와 fine-tuning때 resolution이 달라서, positional encoding 이 의미없어지는걸 방지하기 위하여 interpolation을 사용합니다. 본 논문에선 semantic segmentation에선 positional encoding이 의미가 없다고 주장합니다. 때문에 흔히들 사용하는 PE대신 Mix-FFN을 사용하여 zero-padding을 이용하여 location information leaking을 막습니다. Mix-FFN의 식은 다음과 같습니다. 논문에서 CNN을 쓸때 depth-wise CNN을 썼습니다.????
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220613022821026.png" alt="">



***DECODER***

1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220613023602321.png" alt="">

Method 끝으로 본 논문은 

- pretraining시에 ViT나 SETR과 다르게 ImageNet-22K대신 ImageNet-1K만 사용하여 pretraining했습니다.
- 하나의 single low resolution feature map만 사용하는 ViT나 SETR대신 Segformer는 다양한 high to low resolution을 사용했습니다.
- Semantic Segmentation에 안맞는 PE를 안사용하고 CNNDMF
- MLP decoder또한 간단하여 SETR보다 빠르다고 합니다.



### 3. Real Coding

하지만 실제 코드를 보면 다른점들이 좀 있는데, 다음 링크에선 그 문제들을 아래와 같이 지적합니다. https://github.com/FrancescoSaverioZuppichini/SegFormer/blob/main/README.ipynb 

아래의 내용들은 전부 위 링크에서 나온 말 들입니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220613143925815.png" alt="image-20220613143925815" style="zoom:50%;" />

1. 먼저 위와 같이 official implentation (https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py) 쪽에선 Overlap patch merging이 각 transformer block 초반에 위치한다고 합니다. 이 "overlap patch merging"은 CNN + 2D_Layernorm으로 이뤄져있습니다.
2. Efficient Self-Attn에선 보통 self-attention의 computation cost를 줄이기위해 length 부분을 줄이는데요. 이걸 spatial dimension을 먼저 flat시키고 linear layer를 통해 줄여 줍니다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220614135722841.png" alt="image-20220614135722841" style="zoom:50%;" />
   2. 위 1번은 아래와 같이 CNN을 통해 구현할수도 있습니다.
      1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220614135952572.png" alt="image-20220614135952572" style="zoom:67%;" />
   3. 논문에선 PE를 안쓰고 대신 3x3 depthwise conv를 Mix-FFN을 통해 사용한다고 했는데요. Mix-FFN은 "dense layer --> 3x3 depthwise conv --> GELU --> dense layer"로 이루어져 있습니다.

위 내용들을 전부 합하면 아래와 같은 segformer encoder block이 만들어 집니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220614140346080.png" alt="image-20220614140346080" style="zoom:67%;" />

디코더 부분은 본논문이나 official implementation 코드와 차이가 없다고 합니다.

디코더 순서는 먼저 각 feature_i 별로 output_size H/4 x W/4 x C가 나올수 있게 아래와 같이 upsampling + conv2d를 합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220614143045155.png" alt="">

그리고 나온 각 new-features들을 concat하고 H/4 x W/4 x 4C 를 H/4 x W/4 x N_cls로 바꿔서 output을 만들면 끝!!
