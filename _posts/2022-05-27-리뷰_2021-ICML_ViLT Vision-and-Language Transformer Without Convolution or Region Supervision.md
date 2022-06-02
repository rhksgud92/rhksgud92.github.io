---
layout: post
title:  "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision - 리뷰"

categories:
  - Multimodal
  - Transformer

tags:
  - Multimodal
  - ViT
  - Transformer

---
# 2022_05_27

"ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision" - 리뷰

멀티모달 공부중에 2021년 ICML에 억셉트된 ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision를 오늘 살펴보도록 하겠습니다. 많이 부족한 리뷰지만, 멀티모달쪽을 관심있게 보던 중에 또 Pretraining for Multi-modal model쪽을 공부하던 중에 읽어볼만한 논문이라고 생각해서 읽게 되었습니다.

본 논문 링크: https://proceedings.mlr.press/v139/kim21k.html

### <u>1. Introduction</u>

본 논문은 멀티모달 학습시에 많은 computation cost가 소요되는 CNN등의 encoder input process를 지적합니다. 아래 그림처럼 1) Object detector pretrained cnn based models (ViLBERT) (CNN 후 R-CNN등이 사용되는 방식), 2) ResNet based pretrained models (Pixel-BERT), 3) Only-linear embedding 방식의 encoder의 실험과 그 속도를 비교합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220527112942519.png" alt="image-20220527112942519" style="zoom:50%;" />

논문의 세가지 contributions는 아래와 같습니다.

1) ViLT는 Vision and language model 중 가장 간단하고 빠른 모델
2) CNN 없이 높은 성능의 VL (Vision and Language)모델 달성
3) Word masking 및 Image augmentations를 통한 downstram performance increase

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220527131435842.png" alt="">

위의 그림에서 네모의 크기는 computational size (similar to performance)를 의미합니다. 본논문에선 a와 b처럼 각 modality 별로 embedder를 잘 학습해도 multi-modal representation의 성능이 낮을수 있다고 말합니다. 본논문은 위의 Figure 2.d 처럼 modality 별 extractor보단 modality interaction에 집중하여 c보다 빠르면서 비슷하거나 높은 성능을 내는 모델에 초점을 맞추고 있습니다.

본논문은 UNITER나 VisualBERT처럼 single-stream approahces (late fusion)와 VilBERT나 LXMERT 처럼 dual-stream approaches (early fusion)중 dual은 추가로 필요한 parameter가 있어서 좀더 간단한 single-stream방식을 따릅니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220527133817570.png" alt="">

먼저 본논문의 모델은 이미지를 ViT처럼 patch로 나뉘어서 간단한 linear layer기반 embedding vectors로 나타냅니다. 그리고 아래와같이 embedding vector를 만들고 text또한 image의 height dimension에 맞춰서 linear layer통해 vector를 줄이거나 늘려줍니다. ViT와 마찬가지로 Transformer는 inductive bias가 CNN비해서 적어서 사용가능한 데이터가 적을 경우 CNN을 사용하는게 좋을 수 있지만, 준비된 데이터가 매우 많을 경우, Transformer 만으로도 높은 성능의 모델을 훈련할 수 있습니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220527134716996.png" alt="image-20220527134716996" style="zoom:50%;" />

### 1. Pretraining Objectives

이미지쪽은 ViT-B/32 로 imageNet으로 이미 학습된 모델을 사용했습니다.

Pre-training시에 본 논문은 보통 VLP 논문들처럼 image text matching (ITM)와 masked language modeling (MLM)을 학습했습니다.

1. ITM: randomly replace the image with different image with the probability of 0.5.
2. Alignment Score between [textual subset] and [visual subset] using inexact proximal point method (IPOT)
3. MLM: masked t with the probability of 0.15 (Whole word masking: masking "giraffee" instead of ["gi", "[mask]", "##fe"])
4. Image Augmentation: RandAugment



### 2. Implementation Details

1. AdamW with lr of 10^-4 and weight decay of 10^-2

2. warmup time for 10% of the total training step and linear decay toward zero

3. Image resize: shorter edge max: 384 and longer edge max: 640 with preserved aspect ratio

4. Note that downstream performance may be further improved if we customize the hyperparameters to each task. 

   
