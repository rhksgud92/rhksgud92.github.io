# "Tensor Fusion Network for Multimodal Sentiment Analysis" - 리뷰

---
title:  "Tensor Fusion Network for Multimodal Sentiment Analysis - 리뷰"
excerpt: "논문리뷰"

categories:
  - Multimodal
  - Self-supervised
tags:
  - Multimodal
  - Self-supervised
  - Transformer

---

멀티모달 Fusion쪽 공부중에 2017년 EMNLP에 억셉트된 Tensor Fusion Network for Multimodal Sentiment Analysis를 오늘 살펴보도록 하겠습니다. 많이 부족한 리뷰지만, 의료AI부분에 멀티모달을 적용하려면 또 Fusion쪽을 관심있게 보던 중에읽어볼만한 논문이라고 생각해서 읽고 이해하는 바를 쓰게 되었습니다.

본 논문 링크: https://arxiv.org/abs/1707.07250

### <u>1. Introduction</u>

본논문은 Multi-Modality: Text, Image, Voice 세가지를 이용하여 Sentiment Classification을 수행한 논문입니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214131050571.png" alt="">

위 이미지와 같이 unimodal 데이터만으론 특정 Task를 잘 수행하기 어려운 점들이 있습니다. 본 논문의 Figure 1의 예시처럼 "This movie is fair"이면 보편적으로 Positive한 text이지만 "This movie is sick"의 경우 좋다는건지 안좋다는건지 알기가 애매합니다. 하지만 아래 Bimodal, Trimodal의 예시처럼 Text이외에 Acoustic data, Visual data등이 추가되어 Inter-modality의 정보가 더해져 "This movie is sick"이란 문장이 Positive한 sentiment란걸 의미했다는걸 알수있게 됩니다.

본 논문에서 Multi-modal방식의 학습시에 중요시 생각하는 포인트는: 1) Inter-modality 와 2) Intra-modality인데, Inter-modality의 경우는 위에 예시가 설명되었고 Intra-modality의 경우는 각 개별의 modal안에서의 좋은 representative feature vectors를 의미합니다. Early fusion (feature-level fusion)의 경우 모달끼리 discriminative한 representative feature vector를 학습하기도전에 너무 빨리 modal끼리 합치기에 Intra-modality가 efficient하게 model되지 못했다고 설명합니다. 반대로 Late fusion (Decision-level fusion)의 경우 보통 independent하게 modal끼리 학습을 하고 decision making때만 voting 방법등으로 inference하기에 inter-modality가 efficient하게 model되지 못한다고 설명합니다. 

약간 옛날 논문이라 그런지 metric-learning등이 많이 쓰이기전의 Inter, Intra modality내용에 옛날 SOTA 논문들과 비교가 된점들이 있긴하지만 계속 살펴보겠습니다.

### <u>2. Dataset</u>

데이터는 CMU-MOSI Dataset을 사용했습니다. Spoken Language Text (대화체라서 NLP에서 common하게 쓰이는 Text랑은 다르게 noisy함), Video, Voice로 구분되어 있는 데이터입니다. 93명의 서로다른 화자(speakers)들이각 비디오마다 평균 23.2개의 opinion utterance를 기록했고 이 utterance는 7가지의 감정으로 분류되어 기록된 데이터 입니다. 본논문에서 이 똑같은 데이터를 1) Binary Class, 2) Five Class sentiment, 3) Sentiment Regression in range [-3, 3]으로 분류했습니다.

### <u>3. Methods</u>

1. **Intra-Modality Parts** 에선 각 Spoken Language (Text), Voice (sound), Video frame (Image)를 개별적으로 rich한 representation feature vector를 얻고자 모델이 설계됩니다.
   1. **Spoken Language:** 아래 Figure3과 같이 Word2Vec과 한께 예전에 많이 쓰였던 GloVe가 사용되었고, Temporal한 Video의 Text의 특성을 고려하여 LSTM-layer가 following되고 후에 LSTM의 모든 output 값들이 concatenation되어 Fully-Connected layers에 input으로 사용되어 나온 output 값이 text embedding feature값으로 사용되었습니다.
      1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214152245455.png" alt="">
   2. **Video Images**: FACET facial expression analysis framework을 사용하여 9개의 image-based 정보값과, 20개의 muscle movement based facial action units 값 그리고 OpenFace를 사용하여 68개의 facial landmark locations등이 feature로 extracted되어 user가 정해준 길이의 frames를 시간축으로 mean-pooling한 후 그 값들이 input으로 three hidden layers of 32 ReLU를 지나서 나온 embedding값이 본 논문에 사용됩니다.
   3. **Acoustic Voice**: 12 MFCCs, pitch tracking and Voiced/UnVoiced segmenting features, glottal source parameters, peak slope parameters, 등 다양한 acoustic signal processing을 통한 features를  extracting하여 user가 정해준 길이의 frames를 시간축으로 mean-pooling하여 위 Video Image와 똑같이 그 mean-pooling된 값들이 input으로 three hidden layers of 32 ReLU를 지나서 나온 embedding값이 본 논문에 사용됩니다.
2. ***<u>Tensor Fusion Layer</u>*** 자 이제 본 논문에서 제일 중요한 Tensor Fusion Layer부분입니다.
   1. 위 "Spoken Language Embedding Vector", "Video Images Embedding Vector", "Acoustic Voice Embedding Vector"를 본 논문에선 아래와같이 표시를 합니다.
      1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214161716346.png" alt="">
   2. 그리고 다음과 같이 Unimodal, Bimodal, Trimodal로 계산하여 features를 구한다. 여기서 X는 외적 (Outer-product)를 의미합니다.
      1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214161909810.png" alt="">
   3. Unimodal은 각 modal별로 구해진 Intra-modality가 잘 discriminative하게 구현된 feature 값
   4. Bimodal은 두개의 Modal 별로 외적을 통해 얻은 값 (위 figure 4에서 있는 하늘색 부분). 두 데이터들간의 관계도 정보가 포함이 되게 됩니다.
   5. Trimodal은 세개의 Modal 전부를 외적하여 얻은 값 (위 figure 4에서 있는 보라색 부분). 세가지 종류의 데이터들간의 관계도 정보가 포함이 됩니다.
   6. 본 논문에서 이 방식이 매우 high한 dimension의 feature vector를 나오게 하긴 하지만 이 high dimensional 한 방식이 empirically하게  over-fitting을 낮춘다는 것을 관찰하였다고 합니다.
3. **Sentiment Inference Subnetwork**: 위 Unimodal, Bimodal, Trimodal중 사용할 features들을 모두 concatenation하여 두개의 128 size hidden layer followed by ReLU 를 사용하여 classification 학습을 진행하였습니다.

### <u>4. Results</u>

1. 2011년부터 2016년까지 나왔던 multi-modal sentiment 모델들과 비교 했을때 다음과 같은 성능을 냈는데요. 아래 세가지 tasks들은 1) Binary Class, 2) Five Class sentiment, 3) Sentiment Regression in range [-3, 3]으로 분류하는 tasks입니다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214164118493.png" alt="">
2. Unimodal, Bimodal, Trimodal 별로 조합하였을땐 아래와 같은 성능을 얻었습니다. 여기서 모든 Uni, Bi, Trimodal들이 사용되었을때 가장 좋은 성능을 낸걸 볼수 있습니다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214164403593.png" alt="">

### <u>5. 생각</u>

1. 이 Tensor Fusion Layer 방식은 Unimodal의 종류가 더 많아지거나 크기가 커지면 커질수록 Bimodal, Trimodal의 크기가 Computation-Inefficient하게 빠르게 커질수 있어 모든 Multi-modal데이터에는 적합하진 않을 것 같습니다. 하지만 매우 Explicit하게 inter-modality를 보기에 좋은 multi-modal based feature representation이 되어 학습성능을 높인것 같습니다.
2. 또한 Modal간의 관계도 혹은 Multi-modal model의 outcome을 explain하는데 있어서 유용해보이기도 하는것 같습니다. 위 논문에서 사용된 데이터들에서야 모두 서로 관계가 어느정도있어서 모든 Uni, Bi, Trimodal matrix를 사용했을때 좋은 성능을 냈다지만 다른 데이터 예를 들어 서로 inference output에는 영향이 있지만 서로는 관계가 없는 데이터에서는 다른 modal들의 조합이 더 좋은 성능을 낼수 있을텐데 거기서 modal간의 관계도를 볼때 empirically하게 사용해볼수도 있는 방법인것 같습니다.

