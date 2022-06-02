---
layout: post
title:  "Bayesian Structure Learning with Generative Flow Networks - 리뷰"

categories:
  - Bayesian Structure Learning
  - Structure Learning

tags:
  - Bayesian Structure Learning
  - GFlowNet
  - FlowNet
  - Reinforcement Learning
  - Transformer

---

# 2022_05_02
Bayesian Structure Learning with Generative Flow Networks - 리뷰

원래 Bayesian Structure Learning은 Probabilistic Graphical Modeling 쪽 베이시안 학문쪽에서 constraint-based 나 score-based로 전통적이게 통계적으로 쓰이던 방식입니다. 머신러닝쪽에서는 최근 머신러닝을 이용하여 MCMC, Variational Inference쪽방삭을 사용하여 좀더 유연하게 사용성있는 bayesian structure learning을 제시하는 트렌드입니다. 본 논문에서는 제시한 GFlowNet을 기반으로 만든 small한 DAG-GFlowNet을 제시합니다.

본 논문은 다음과 같은 이 모델의 장점들을 먼저 소개합니다.

1. Graph의 sample space가 오직 DAG (Directed Acyclic Graph)만 포함함.
2. MCMC나 Score-based처럼 search node order순서에 따른 변화 혹은 slow mixing문제가 없다.
   1. <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503152004239.png" alt="image-20220503152004239" style="zoom:50%;" />
   2. 위와 같이 MCMC의 경우 Mixing time can be exponential for well separated modes in high-dimensional distributions. 또한 sampling large number of data points can be slow as computation happens during sampling
3. 또한 현실세계 데이터에서도 좋은 성능을 나타낸다고한다.
4. 좋은 posterior distribution을 small한 graph을 generate할때도 만들수있다고 제시한다.

### 1. GFlowNets

Generateive Flow Network (2021)은 본래 DAG에 의한 discovery of diverse modes of an unnormalized  distribution을 위한 generative model입니다. Initial state인 s_0 에서 시작하여 samples are constructed sequentially by following the edges of the DAG. s_f 는 'terminal state'  혹은 'complete states'라고 불립니다. 



### 2. DAG-GFlowNets

DAG-GFlowNets은 markov decision process와 비슷하게 sequential하게 DAG를 아래와 같이 construct합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503125907125.png" alt="image-20220503125907125" style="zoom:67%;" />

GFlowNets의 complete state는 reward R(s) >= 0 와 관계가 있으며 아래 식과 같은 **flow-matching condition**을 따라서 이 reward를 계산한다. 그리고 이 flow-matching condition은 regression 문제로 학습되어집니다. 여기서 R은 residual이고 left의 first term은 부모 노드들로부터 들어오는 flow이며 left의 second term은 자식 노드들로 나가는 flow양입니다. 만약 complete state일 경우, 첫번째 term과 R(s')는 거의 같아집니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503130159542.png" alt="image-20220503130159542" style="zoom: 67%;" />

여기서 s --> s' 의 transitions는 다음과 같은 detailed-balance loss를 통해 loss가 계산되고 학습됩니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503155413603.png" alt="image-20220503155413603" style="zoom:67%;" />

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503132953848.png" alt="image-20220503132953848" style="zoom:67%;" />

이제부터 아래 term들은 s나 s'대신 G나 G'가 쓰입니다. 이유는 GFlowNet 에서 DAG-GFlowNet으로의 application을 보기때문입니다. 

이 위의 parameter값들을 이제 설명하겠습니다. 먼저 Edge를 하나하나 추가하며 확률값을 보는데 이때 다음과같은 2개의 rule이 있습니다. 이 내용들은 전부 mask로 control되어집니다.

1. Is the edge that will be added already present in G?
2. Does addition of the edge introduce a cycle?

GFlowNets에서 Parametrizing 은 오직 forward transition probability을 위해서만 있다. 이때도 두가지의 condition만이 아래와 같이 존재한다.

1. a network modeling the probability of terminating <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503134932294.png" alt="image-20220503134932294" style="zoom:50%;" />
2. another giving the probability to a new graph G'': <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503135330886.png" alt="image-20220503135330886" style="zoom:50%;" />(Knowing it does not terminate.)

**마지막으로 조합하여 the transition G --> G' 는 다음과 같습니다.** <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503135437557.png" alt="image-20220503135437557" style="zoom:50%;" />

DAG-GFlowNets에선 transformer 기반 neural network를 generation하는데 사용한다. 위의 두가지의 condition을 계산할때 하나의 neural network with a common backbone 을 사용하고 이 모델은 아래와 같이 두개의 다른 head를 가지고 위의 term들이 계산되어집니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503142213710.png" alt="">

이모델의 motivations들은 다음과 같습니다.

1. It has to be invariant to the order of the inputs since G represents a set of edges.
2. That transforms a set of input edges into a set of output probabilities for each edge to be added.



### 3. Application to bayesian structure learning

MCMC나 Variational inference대신 GFlowNet을 사용하여 DAGs의 posterior distribution을 가늠합니다. DAG 인 G를 위해 여기서 RL방식의 다음과 같은 reward를 계산합니다. P(G)는 DAGs의 priority이고 P(D|G)는 marginal likelihood입니다. 논문에선 Bayes theorem에 의해서 R(G)는 P(G|D)와 proportional하다고 합니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503143511807.png" alt="image-20220503143511807" style="zoom:67%;" />

위 식을 현실적으로 구할때, 아래 log(reward)는 sum of local scores로 individual variables와 그들의 parents in G에만 관계되어 있습니다. 이 로그값은 bayesian score와 같다고 볼 수 있다. 그리고 G'와 G의 로그값의 차이 (Addition of edge가 영향을 준 정도)는 그 아래 식과 같습니다 (delta score 혹은 incremental value 라고도 불리운다).

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503144236223.png" alt="image-20220503144236223" style="zoom:67%;" />

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503144516257.png" alt="image-20220503144516257" style="zoom:67%;" />

논문에서는 위 식들을 사용하여 detailed-balance loss를 구합니다. 위의 log difference는 결국 위에 설명했던 이 detailed-balance loss를 의미합니다. 이방법을 사용하여 아래의 loss를 off-policy learning인 deep q-learning 방식으로 구하여 학습합니다.

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/image-20220503183848545.png" alt="image-20220503183848545" style="zoom:50%;" />



