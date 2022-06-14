# UNETR: Transformers for 3D Medical Image Segmentation - 리뷰

최근 몇년동안 NLP 쪽에서는 Transformer 모델이 크게 좋은 성능들을 내며 각광받게 되었습니다. 이를 기반으로 BERT, GPT등 유명한 Large-scale 자연어처리 모델들이 나왔고, 음성인식에서도 Wav2Vec2.0, 그리고 이미지 쪽에서도 DETR, Segformer, ViT등 다양한 모델들이 나왔습니다. 이번엔 흔히들 사용하는 FCNN대신 Transformer를 사용하여 3D Medical image segmentation을 한 UNETR 논문을 보겠습니다.

논문링크: https://arxiv.org/abs/2103.10504

### 1. Introduction

영상쪽에서 U-Net 기반의 encoder의 downsampling 그리고 decoder의 upsampling과 spatial representation의 정보전달을 위한 skip conenction 을 사용하는 semantic segmentation은 많이 사용되어왔습니다. 하지만 CNN특유의 inductive bias를 만드는 localized receptive field의 특성은 long range dependecies특징에는 약한점들이 본 논문에선 지적됩니다. 본 논문의 세가지 contributions들은 다음과 같습니다. 1) 처음으로 transformer encoder 기반 medical image segmentation 모델을 보여줌 2) Transformer 기반 3D volume input model + skip connection decoder combined model. 3) BTCV 와 MSD 데이터셋에서 SOTA 달성입니다. 

### 2. Related Works

1. SETR: pretrained transformer encoder with CNN-based decoder for semantic segmentation
2. Multi-organ 2d segmentation by adding transformer bottleneck to U-Net architecture
   1. "Transunet: Transformers make strong encoders for medical image segmentation"
3. transformer-based axial attention mechanism for 2D medical image segmentation
   1. Medical transformer: Gated axialattention for medical image segmentation.
4. Fusion of CNN and Transformer architectures at the end
   1. Transfuse: Fusing transformers and cnns for medical image segmentation.

위 논문들과 UNETR이 다른 점은 3D segmentation을 한다는 점, volumetric 데이터를 한번에 reduction없이 사용하는 것, Transformer가 메인 encoder로 사용되는 점들이 있습니다.

### 3. Methods

먼저 UNETR은 다음 형태의 3D image 데이터를 
$$
H * W * D * C
$$
다음과 같이 N개의 P,P,P 사이즈의 3D 패치들로 바꿉니다.
$$
N * (P^3 * C)
$$
N 개수는 다음과 같고요.
$$
N = (H * W * D) / P^3
$$
그리고 이 Patches의 sequence를 linear layer 하나를 통하여 k dimension의 embedding vector로 바꿉니다. 디멘션크기가 안 바뀌는 Transformer에선 이 k가 constant transforemr dimension 크기가 됩니다. 그리고 왠만한 Transformer 기반 모델들에서 대부분 하듯 1D learnable positional embedding vector도 처음에 만들어서 더해줍니다. BERT나 ViT처럼 learnable한 class token은 여기선 없습니다.

3D데이터를 P^3 형태의 patches로 나눠서 sequencetial 하게 만들어서 Transformer encoder block들의 형태 까지는 Attention is All you need와 별 다른점이 없습니다. 하지만 U-Net의 architecture를 활용하기 위하여 아래 사진과 같이 3, 6, 9, 12번째의 encoder output을 빼서 아래와 같이 deconvolution을 써서 값을 얻은 후 U-Net과 비슷하게 skip-connnection 으로 정보값을 더해줍니다.



![image-20220604182037877](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220604182037877.png)

Loss function 은 다음과 같이 soft dice loss 와 cross-entropy loss를 합하여 voxel-wise하게도 loss를 구할수 있게 한다. 여기서 J는 class 개수이고, I는 voxel (3d) 개수, Y_i,j는 output probability, G_i,j는 one-hot encoded ground truth for class j at voxel i이다.

<img src="C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220607170549998.png" alt="image-20220607170549998" style="zoom:50%;" />

데이터셋은 CT 데이터인 BTCV와 MRI/CT 데이터인 MSD를 사용했습니다.



### 4. Results

![image-20220607170709705](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220607170709705.png)

성능 결과 비교는 Dice score와 HD metrics을 아래와 같이 사용하였습니다.

<img src="C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220607170803690.png" alt="image-20220607170803690" style="zoom:50%;" />

아래는 다른 모델들과 비교한 segmentation 결과 입니다. 아래 결과만 보자면 UNETR이 다른 모델들에 비하여 좀더 세밀하게 그리고 더 디테일하게 segmentation해주는걸 볼 수 있습니다.

![image-20220607170832864](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220607170832864.png)

![image-20220607170848848](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220607170848848.png)
