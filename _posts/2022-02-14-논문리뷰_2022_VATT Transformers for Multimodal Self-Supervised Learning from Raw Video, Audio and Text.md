---
layout: post
title:  ""VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text" - 리뷰"
---
# 독학 AI연구원의 논문리뷰

멀티모달 Fusion쪽 공부중에 2021년 NeurIPS에 억셉트된 VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text를 오늘 살펴보도록 하겠습니다. 많이 부족한 리뷰지만, 멀티모달 Fusion쪽을 관심있게 보던 중에 또 Pretraining for Multi-modal model쪽을 공부하던 중에 읽어볼만한 논문이라고 생각해서 읽게 되었습니다.

본 논문 링크: https://proceedings.neurips.cc/paper/2021/hash/cb3213ada48302953cb0f166464ab356-Abstract.html

### <u>1. Introduction</u>

CNN의 경우 translation invariance와 locality등의 강한 강점으로 인하여 visual data 에서 강한 모습을 계속 보여왔습니다. 하지만 최근 몇년동안 NLP쪽뿐만 아니라 Vision 쪽에서도 RNN, CNN등의 models with stronge inductive biases만을 사용한 모델에서 self-attention을 추가/조합하거나 혹은 ViT처럼 Self-attention만 이용한 Transformer계열의 모델들이 좋은 성능을 보이고 있습니다. 

본 논문은 이처럼 많이 사용되어지기 시작한 Transformer 모델을 Large-scale, unlabeled visual data에서도 활용하고 또한 ViT의 아이디어 처럼 CNN등을 사용하지 않고 오로지 raw-data그대로 transformer에 input으로 사용하여 학습하는데 의의를 두고 있습니다.

본 논문을 짧게 요약하자면 세 종류의 data (RGB frames의 Videos, audio waveforms, text transcripts of the speech audio)를 사용하여 contrastive learning을 활용한 self-supervised multimodal pre-training method for Transformer model을 제시합니다 (한 문장으로 요약하려니 좀 길어졌네요 ㅎㅎ). VATT은 Video, Audiom Text Transformer의 약자입니다. 이렇게 학습된 모델들은 논문에서 여러가지의 Downstream tasks (image classification, video action recognition, audio event classification, and zero-shot text-to-video retrieval)로 성능이 증명됩니다. 또한 서로 다른 modals끼리 같은 sharing weight model로써 학습되어 modality-agnostic한 모델 "a single, general-purpose model for all the modalities"의 가능성또한 추가로 실험을 했고 사이드로 DropToken이라는 training complexity등을 낮춰주는 method를 제시합니다.

### <u>2. Methods</u>

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214171007473.png" alt="">

전체적 모델 및 학습 방식은 위에 figure 1과 같습니다. 이제 하나하나 찬찬히 살펴보겠습니다.

1. **Tokenization**:
   1. 먼저 data들은 각 modal별로 서로 다른 tokenization이 진행 됩니다. 당연히 Multi-modal training이니깐 Input data는 종류별로 size, shape등이 다를건데 같은 모델 (심지어 sharing weight model)에 들어가려면 같은 모양의 matrix형태로 바꿔줘야 합니다.
      1. **Visual Data**: Visual data의 경우 R, G, B까지 포함하여 T x H x W x 3 (Time, Height, Width)크기의 raw input이 들어옵니다. ViT와 비슷하게 저 raw-input을 t x h x w로 나눠서 group of patches로 만듭니다. 그리고 나서 아래 1번 처럼 learnable weight를 사용하여 학습 가능하게 d-dimensional vector 로 변환합니다. 그 후 transformer 모델 특성상 추가로 필요한 position 정보값을 아래 2번을 통하여 더하여 넣습니다.
         1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214190409018.png" alt="">
         2. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214190745025.png" alt="">
      2. **Audio Data**: Visual data와 거의 똑같이 T길이의 1D signal data를 t길이의 patch의 group으로 slicing 을 한 후 아래 1번의 learnable weight를 통해 d-dimensional vector로 변환하고 똑같이 1D Positional encoding 값을 구하여 더합니다.
         1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214191453754.png" alt="">
      3. **Text Data**: 보통 NLP에서 하는 방법과 똑같이 먼저 vocab size를 정하고 input text data를 mapping 하여 v-dimensional one-hot vector로 만들고 아래의 linear proejection with a learnable weight를 곱하여 d-dimensional vector로 변환합니다. Visual 이나 audio data와 다르게 text에선 positional encoding을 사용안하고 learnable relative bias (relative positional encoding)을 사용합니다. 이방식은 각 token간의 거리를 attention score에 더하는 방식으로 절대적인 위치값이 아니라 상대적 거리 정보값을 넣어줍니다. 논문에선 이 relative positional encoding 방식을 사용하여 SOTA text model T5에 weight transferrable하게 만들었다고 합니다. (같은 형태)
         1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214191649568.png" alt="">
2. **Drop Token**: 본 논문에서 간략하게 제안하는 drop token은 매우 간단합니다. 본 논문에서 쓰는 데이터 특성상 상대적으로 abundant한 (raw-data 그대로 쓰기에 redundant함) visual, audio data based tokens를 무작위로 dropout 처럼 훈련때 제거하여 transformer 모델 특성상 sequence가 길면 길수록 커지는 computational cost를 줄여줍니다. 본 논문에선 data의 resolution을 줄이거나 dimension을 줄이지말고 이 방법 을 사용하여 information loss 없이 빠르게 학습을 하게 한다고 합니다.
3. **Aggregation**: 본 논문의 Transformer의 input은 결국 아래 1번과 같이 되게 됩니다. 여기서 X_AGG란 보통 NLP 모델에 있는 CLS token (Classification token) 으로 사용되었습니다. 이 Z_in 값은 위 figure 1의 두번째 그림인 Transformer Encoder에 들어갑니다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220214230223550.png" alt="">
4. **Common Space Projection**: 자 이제 위에서 뽑은 Z_in 값이 Transformer 모델을 지나 output 값 (Z_out)을 얻었습니다. 본 논문에선 이 각기 다른 세가지의 Z_out (Visual, audio, text)들이 이제 같은 dimension 값을 가진 common space에 있다고 말합니다 (자주 쓰는 term인지는 모르겠습니다). 이제 얘네들을 서로 metric learning을 하면서 positive (augmentation 되어 서로 비슷하거나 같은 data에서 나온 샘플들)끼리는 common space에서 가깝게 혹은 다른 negative (positive의 반대; 서로 의미상 거리가 있다고 해주고 싶은 데이터들에서 나온 샘플들)끼리는 common space에서 멀게 해주는걸 해볼텐데요 (거리 혹은 비슷한 정도는 본논문에서 consine similarity를 사용). 본 논문에선 아래 1번 식과 같이 이 세가지의 Z_out들을 의미적으로 다른 순차적인 common space에서 따로따로 (Visual-Audio -> Visual-Text 순서로) pairing을 하여  metric-learning 방식중 contrastive learning을 사용하여 comparing을 진행합니다. 이렇게 순차적으로 한 이유는 데이터들간에 "different levels of semantic granulariy for these modalities"라고 하는데요. 제 생각엔 아마 Audio-Text의 comparing의 경우 데이터의 전체적인 의미보다 단순히 Speech Recognition에서 필요한 phones-words 혹은 phones-context_dependent phones의 관계 정보가 얻어질거기에 필요가 없어서 안한게 아닌가 싶습니다 (또한 metric learning상 negative, positive만 있는데 세종류를 한번에 비교하기도 어렵고요). 아래 1번 식들중 comparing 을 위해 Z_out이 들어가는 식들은 하나의 linear projection으로 구성되어 있고, Visual-Text pairing을 하기전의 Visual이 이전 하나의 linear proejction을 지난 벡터는 두겹의 linear projections with ReLU를 통과하여 Visual-Text를 비교하게 됩니다. (Batchnorm은 linear layer 앞마다 있음)
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215105313061.png" alt="">
5. **Multimodal Contrastive Learning**: 자 이제 위의 Common Space Projection으로 나온 벡터들로 이제 어떻게 contrastive learning을 하는지 살펴봅시다. Positive pairing은 각 modalities에서 동시간대그리고 같은 위치의 Video에서 나온 데이터끼리 pairing하고 Negative pairing은 서로 다른 위치의 Video에서 나온 데이터끼리 pairing합니다. 아래 1번 식은 Visual-Audio pair는 Noise Contrastive Estimation으로 cosince similarity가 positive끼리는 높게 negative끼리는 낮게 학습되도록 유도하는 방식이고, 2번은 Visual-Text pair를 위한 Multiple Instance Learning NCE (MIL-NCE 방식)입니다. MIL_NCE가 다른 점은, Text의 경우 Visual이나 Audio에 비하여 continuous하게 데이터가 있지않고 띄엄띄엄 있을텐데, 이 P가 Text데이터가 있는 시기의 앞뒤로 몇개의 visual frames정도는 같은 positive pair로 두고 하는 방식입니다. 비디오에서 Frames는 빠르기 때문에 보통 몇 frame 지났다고 Text값이 바뀌진 않을테니까요. 타우 값은 클수록 distribution이 평평해져서 비교하기 어렵고 작으면 sharp해져서 비교하기 쉽게 됩니다. 본 논문에선 타우 값으로 0.07을 pretraining시에 사용합니다. 이 1번 2번 식 계산이 끝나면 3번 처럼 둘을 합하여 Loss를 구하는데, λ값은 아래 1, 2번의 두 losses들간의 balancing을 할때 사용합니다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215115947362.png" alt="">
   2. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215115955414.png" alt="">
   3. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215120157839.png" alt="">

### <u>3. Results</u>

학습 후 다양하게 Video action recognition, audio event classification, zero-shot text-to-video retrieval, image classification등의 downstram task로 fine-tuning을 하였고 아래는 다른 멀티모달 모델들과의 성능 비교 테이블들입니다. 여기서 VATT-MA의 경우 위 introduction 쪽에서 설명한 modality-agnostic 학습 방식 모델입니다. 세가지 다른 종류의 input data가 들어가는 sharing weight를 하는 같은 모델에서도 놀랍게도 modality-specific방식보다는 약간 성능이 낮지만, 괜찮은 성능이 나온걸 볼 수 있습니다. 아래 3번 예시는 Modality-agnostic 모델도 modality-specific 방식에 비해 나쁘지 않게 positive와 negative pairing distribution이 구분된것을 볼 수 있습니다.

1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215120504656.png" alt="">
2. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215120515452.png" alt="">
3. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220215121125981.png" alt="">

