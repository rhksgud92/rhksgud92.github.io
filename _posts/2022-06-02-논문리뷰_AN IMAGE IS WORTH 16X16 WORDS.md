# AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE - 리뷰

최근 몇년동안 NLP 쪽에서는 Transformer 모델이 크게 좋은 성능들을 내며 각광받게 되었습니다. 이를 기반으로 BERT, GPT등 유명한 Large-scale 자연어처리 모델들이 나왔고, 음성인식에서도 Wav2Vec2.0, 그리고 이미지 쪽에서도 DETR, Segformer등 다양한 모델들이 나왔습니다. 하지만 Wav2Vec2.0등의 모델들은 Transformer encoder를 쓰지만 여전히 CNN을 첫 레이어로 사용하여 특징을 추출하는 무거운 형태의 모델들이였습니다. ViT는 CNN을 쓰지 않고도 오로지 patch projection만을 사용하여 빠르고 CNN에 못지 않은 정확도를 갖춘 영상인식 모델입니다.

### 1. Introduction

ViT 모델은 기본 Transformer 모델을 최대한 적게 modification하여 , image를 patches들로 나눠서 단순히 linear embedding 후 각 embedding 들은 nlp의 tokens처럼 다루어져 transformer 에 들어가게 됩니다. Transformer는 CNN처럼 locality에 대한 inductive bias가 없어서 ImageNet 처럼 mid-sized의 dataset을 사용할 경우 성능이 ResNet에 못미친다고 합니다. 하지만 더 큰 데이터에 학습이 될 경우 large-scale training trumps inductive bias한 결과를 이 ViT모델에서 얻었다고 합니다.



### 2. Method

본 논문에선 input data가 이미지이지만 원 "Attention is all you need"의 NLP 방식을 최대한 그대로 사용했습니다. 먼저 2D image의 H x W x C dimension을 (P, P)의 resolution의 작은 patch로 나눕니다. 이 patch 개수는 
$$
N=HW/P^2
$$
 과 같습니다. Transformer의 D-dimension을 맞추기 위해서 그 후, 이 패치들을 D-dimension으로 flatten 합니다. 이러면 아래의 식 1번과 같게 됩니다. BERT 모델과 똑같이 [CLS] token을 만들어 후에 classification등을 할때 사용합니다. 이 CLS토큰은 learnable parameter로, nn.parameter등으로 만들면 됩니다. 논문에선 positional embedding vector도 learnable한 1D position embedding을 사용하여 patch embedding에 더하여 transformer에 들어가게 합니다. 2D-aware positional embedding이란걸 사용했지만 성능의 효과는 못봤다고 합니다.![image-20220603162730430](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220603162730430.png)

![image-20220603155915726](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220603155915726.png)

본 논문은 영상인식에서 Transformer만을 사용하는데 중점을 두긴 했으나 CNN과 hybrid로 학습하는것도 염두에 두고 있습니다. CNN과 같이 쓸 경우, CNN의 feature map에 patch projection을 하여 진행합니다.

성능의 향상을 위하여 논문에선 pretraining때보다 fine-tuning 때 더 높은 resolution의 input 응 사용했습니다. (Fine-tuning 할때는 pretrain때 썼던 prediction head를 떼낸후 zero-initialized 된 DxK feedforawrd만 단순히 사용했습니다. 여기서 K는 당연히 class개수입니다.) higher-resolution이나 low-resolution이나 둘다 같은 patch size를 사용합니다. 이렇게 resolution을 바꿀경우 position embedding의 의미가 없어지는데요. 그래서 그걸 방지하기 위하여 fine-tuning시에 pretrain때 사용했던 positional embeddings의 original image의 2D-interpolation을 사용합니다. (이 부분이 아직 확실히는 이해가 안가네요. 그냥 fine-tuning때의 positional embedding에서 pretrain때의 positional embedding의 사이사이를 2D interpolation으로 채워서 사용한다는 말인것 같습니다.) 이 resolution adjustment와 patch extraction이 이 모델에서 유일한 2d input 에 대한 inductive bias라고 합니다.



### 3. Experiment

본 논문의 이미지 전처리 형식은 다음 논문의 방식을 사용하였습니다. Big transfer (BiT): General visual representation learning.  학습 할때, Adam으로 linear learning rate warmup and cosine learning rate decay를 사용했습니다. (batch size = 4096) (cosine decay는 appendix에 나옴.) 그리고 후에 downstream-task에서 fine-tuning할때 SGD with momentum을 사용합니다. (batch size = 512)

Metrics는  few-shot이나 fine-tuning accuracy를 사용하여 report 했습니다. 보통은 fine-tuning accuracy 만 보긴하지만 few-shot은 빠르게 on-the-fly로 evaluation할 수 있다는 장점이 있습니다.







### 4. Results

![image-20220603171414998](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220603171414998.png)
