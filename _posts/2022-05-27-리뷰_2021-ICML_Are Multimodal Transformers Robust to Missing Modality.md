---
layout: post
title:  "Are Multimodal Transformers Robust to Missing Modality? - 리뷰"

categories:
  - Multimodal
  - Fusion

tags:
  - Multimodal
  - Transformer
  
---

# 2022_05_27

Are Multimodal Transformers Robust to Missing Modality?" - 리뷰

멀티모달 공부중에 2021년 ICML에 억셉트된 Are Multimodal Transformers Robust to Missing Modality?를 오늘 살펴보도록 하겠습니다. 많이 부족한 리뷰지만, 멀티모달쪽을 관심있게 보던 중에 또 Pretraining for Multi-modal model쪽을 공부하던 중에 읽어볼만한 논문이라고 생각해서 읽게 되었습니다.

본 논문 링크: https://arxiv.org/abs/2204.05454

### <u>1. Introduction</u>

본 논문은 Multi-modal Transformer의 missing modality의 영향에 대하여 다루고 있습니다. 먼저 논문은 performance of model according to its fusion methods는 dataset-dependent하다고 설명합니다. 즉, 최상의 fusion 방식은 존재하지 않고 각 dataset별로 알맞는 fusion방식이 따로 있다고 주장합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220527145420171.png" alt="image-20220527145420171" style="zoom:50%;" />

본 논문은 robustness of Transformer를 multi-task optimization을 통하여 improve합니다. 먼저 이 논문은 오로지 Transformer 만 사용하여 missing modality 의 robustness 본 논문으로 SMIL같이 다른 모델/방식의 missing modality를 상대하는 방식과는 비교가 없습니다.

Transformer는 최근 다양한 방면에서 multi-modal 학습을 위해 사용되고 높은 성능을 내고 있습니다. (UNIT, UNITER ,ViLT, Pixel-BERT, VATT, etc.) 또한, missing modality가 있을 경우에 다른 multi-modal model들은 missing modality를 처리하기 위해 여러가지 방법들을 써야하지만 transformer는 단순히 missing 된 modality를 mask하면 되는 장점도 있습니다 (여러가지의 modalities가 하나의 transformer로 들어갈 경우). 

먼저 논문은 논문 제목인 "Are Transformer models robust against modal-incomplete data?"에 답변하기 위해서 modal-incomplete data로 Transformer 모델의 성능을 측정해봅니다. (MM-IMDb, UPMC Food-101, Hateful Memes 데이터 사용). Full-modality 로 학습된 모델들이 modality-incomplete로 테스트 했을때 급격히 성능이 떨어지는걸 볼수 있었고 일부 테스트 셋은 unimodal보다도 성능이 낮게 나왔습니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220528124518064.png" alt="">

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220528124942742.png" alt="">

먼저 위처럼 Full-modal 학습과 unimodal 학습을 통하여 어떤 modal이 학습에 있어서 영향을 더 끼치는지 분석합니다. 분석 결과 본논문의 task에서는 Text데이터가 target 결과에 더 영향을 많이 끼치는 걸 볼 수 있습니다.



### <u>2. Methods</u>

1. Multi-Task Learning:
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220528125058029.png" alt="">
   2. 위와 같이 세가지 Task를 Multi-task learning 을 함으로써 아래 3번과같은 식으로 loss를 계산 합니다. 세가지 task는 full-modal을 input으로 했을때, text-input-only로써, image-input-only로써 output을 구할때로 세가지 loss를 구하여 학습합니다.
   3. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220528125130102.png" alt="">
2. Optimal Fusion Strategy:
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220528125328086.png" alt="">
   2. 본 논문에선 multi-modal 학습에서 optimal한 fusion 방식은 dataset-dependent하다고 합니다. 때문에 이 논문에선 위와같이 search policy를 구현하여 데이터셋마다 fusion을 다르게하여 학습합니다.






