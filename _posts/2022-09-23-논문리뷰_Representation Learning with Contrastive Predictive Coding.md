---
layout: post
title:  "Representation Learning with Contrastive Predictive Coding - 리뷰"

categories:
  - Self-Supervised Learning
  - Unsupervised Learning
  - Mutual Information

tags:
  - Self-Supervised Learning
  - Unsupervised Learning
  - Mutual Information

---

# 2022_09_23

# Representation Learning with Contrastive Predictive Coding - 리뷰

최근 각 modality (image, audio, vision, medical-data) 별로 Self-supervised learning 방식으로 미리 label 없이 많은 데이터로 학습을 pre-trained model을 만들어서 후에 fine-tuning, linear learning 등으로 downstream task에서 높은 성능을 내는 다양한 모델들이 계속하여 나왔습니다. 오늘은 후에 Wav2vec2.0 이나 CPC2의 기반이 된 현재와 과거의 값의 context vector와 미래값의 encoder output 사이에 mutual information을 높여줘서 각 encoder 및 auto-regressive part를 학습하는 2019년에 뉴립스에서 나온 논문 "Representation Learning with Contrastive Predictive Coding" 논문을 리뷰해보겠습니다.

논문링크: https://arxiv.org/pdf/1807.03748.pdf

### 1. Introduction

본 논문은 세가지 Contributions를 강조합니다.

1. 먼저 image나 audio의 spectrogram 같은 high-dimensional data를 compact한 latent embedding space로 projection하여 conditional prediction을 보다 쉽게 학습하게 하는 모델을 제시합니다.
1. 두번째로, 미래 값들의 latent space를 예측하는 autoregressive predictive model을 제시합니다.
1. 마지막으로 본 논문은 Noise-Contrastive Estimation (NCE)방식을 사용하여 loss function을 구하여 다양한 modalities (image, speech, NLP, RL)등에 적용 가능성을 보여줍니다.



먼저 본 논문은 "underlying shared information"을 학습하는데 초점을 두고 있다고 말합니다. Predictive coding을 수행할때 보통 가까운 미래값의 경우 local한 information만으로도 어느정도 예측이 가능하지만 예측지점이 멀어지면 멀어질수록 time-line상 local한 information보다 전체적인 global information을 알고 있는것이 학습에 도움이 될거라고 가정합니다. 논문에서 이 global information을 'slow features'라고 부릅니다. 

본 논문에서 미래값의 latent representation을 현재와 과거값만을 가지고 학습을 하는데, 아래 식처럼 여기서 현재까지의 값을 벡터화한 context c vector가 있고 미래값의 encoder output인 x가 있습니다. 여기서 이 두 벡터들간의 mutual information을 maximally preserve하기 위한 학습 즉 두 벡터들간의 mutual information 을 높히는 학습을 self-supervised learning 방식으로 진행합니다. 

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923155119396.png" alt="">

### 2. Methods

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923154023717.png" alt="">

***ENCODER and Autoregessive Parts***

1. 먼저 x_t로 표현된 raw값 (이미지의 경우 patch 혹은 CNN's strided pixels, 오디오의 경우 signal 혹은 spectrogram, filterbank, MFCC 등의 각각의 time-steps)등이 g_enc 에 들어갑니다. 여기서 t는 time-step입니다. 이 non-linear encoder g_enc는 x_t 를 latent representation 인 z_t로 projection해줍니다. 여기서 z_t는 x_t보다 lower한 time-resolution을 가질수도 있고 아닐수도 있습니다 (8~16Hz 시그널처럼 너무 컷으면 여기서 줄이고 심전도나 다른 애초에 작은 resolution을 가지고 있던 데이터면 안줄여도 되겠지요).  
2. 그리고 autoregressive 모델인 g_ar 은 과거부터 현재까지의 z_t들을 종합하여 c_t, context vector를 뽑아냅니다. 이 context vector는 만약 학습이 잘 되었을 경우, 미래를 예측하기 위한 과거부터 현재까지의 종합적인 정보를compact하게 잘 vectorized 된  vector값일것 입니다. 

본 논문에선 c_t를 가지고 직접적으로 x_t를 예측하는게 아닌 density ratio를 사용하여, 아래 식과 같은 두 변수사이의 의존도 mutual information을 높히는 방식으로 학습을 합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923160058282.png" alt="">

실제 코드에선 아래와 같이 log-bilinear model을 사용합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923160145072.png" alt="">

위의 W_k 는 fully-connected layer의 weight인데 이 weight는 각 time-step별로 만듭니다. 후에 나오지만, 2~20개의 time-steps를 예측하게 실험을 했고, 본 논문에서 사용한 데이터셋에선 12개의 times-steps 예측이 성능이 제일 좋았다고 합니다. 

Negative samples는 그냥 다른 randomly sampled values를 기준으로 진행했습니다. 

***InfoNCE Loss and Mutual Information Estimation***

본 논문에선 NCE를 기반으로 한 loss를 사용합니다. 아래 식과 같이 Positive sample 은 한개 그리고 아주 여러개의 negative samples가 있는데 여기서 softmax 를 사용하여 positive 한개를 찾는 방식입니다. 

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923162044361.png" alt="">

여기서 f_k는 positive sample의 density ratio입니다. 

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923162250552.png" alt="">

위의 식에서 d = i는 x_i가 positive 인지 아닌지를 나타냅니다. 여기서 결국 위 L_N과 p(d=i | X, c_t)가 위의 식 f_x와 density ratio 식처럼 proportional하다는걸 보여줍니다. 

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923162539078.png" alt="">

마지막으로 위의 식처럼 c_t와 x_t를 사용하여 mutual information을 evaluate할 수 있습니다. N이 커지면 커질 수록 (batch size가 커지면 커질 수록 혹은 negative samples개수가 커지면 커질수록) tight 해지는걸 알 수 있습니다. 즉 큰 negative samples를 수행할수 있는 gpus가 있을수록 성능이 좋다는 얘기죠. 또한 당연히 학습되어지는 L_N도 줄으면 줄을수록 mutual information이 커지는것도 알 수 있습니다.



### 3. Real code

본 코드는 아래에서 가져왔습니다.

https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923162911100.png" alt="">

먼저 위의 self.encoder는 g_enc입니다. 오디오 데이터 기반이라 1D-CNN을 사용한걸 볼 수 있습니다. 그리고 self.gru는 g_ar (autoregressive) 파트입니다. 마지막으로 self.Wk는 fully-connected layer인데 predict하고 싶은 time-step 개수만큼 만듭니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220923163605081.png" alt="">

먼저 전체 데이터를 encoder로 넣어서 z latent representaion vector를 뽑아냅니다. 그리고 랜덤하게 t_samples (predict-time-point)를 뽑습니다. 그리고 과거부터 t_sample까지의 z를 g_ar에 넣고 g_ar의 output 중 last vector of gru output 을 C_t context vector로 사용합니다. 여기서 C_t를 각각의 모든 W_k에 넣어서 WK x C_t값들을 준비합니다. 여기서 단순히 matrix multiplication을 통하여 matrix를 뽑고 soft-max로 probability를 구합니다. 학습이 잘되었다면 같은 positive sample 들이 있는 diagonal에는 높은 숫자가 다른 배치에 있어서 negative인 값들과는 낮은숫자가 있어야겠죠. 

마지막으로 diagonal의 숫자가 클수록 loss가 낮아지는 방식입니다.
