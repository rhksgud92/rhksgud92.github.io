---
layout: post
title:  "Bayesian Structure Learning from Probabilistic Graphical Model - 리뷰"

categories:
  - Bayesian
  - Structure Learning
  - Probabilistic Graphical Model

tags:
  - Probabilistic Graphical Model
  - Bayesian Structure Learning

---

# 독학 연구원의 공부 - 1
Bayesian Structure Learning에는 여러가지 방법들이 오랬동안 제시되어 왔다.

하지만 그전에 먼저 기초적으로 알아야할 Bayesian Networks의 기본 이론들을 공부해보겠다.

아래 내용은 모두 ***"Probabilistic Graphical Models" by Koller and Friedman*** 책의 내용입니다.

### A. 기초 Bayesian Network의  Definitions

1. **For each variable X_i:** ![image-20220426161956522](C:\Users\kwanl\OneDrive\바탕 화면\Kwan_Study\storage\pgm1)
   
   1. Node 변수 X_i에 대하여 parents nodes가 condition으로 주어질때 nondescendants (child가 아닌 nodes)와 전부 independent하다.
   
2. **I_Maps:** 
   1. ![image-20220426162719838](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426162719838.png)
   2. Minimal I_Map: If a a single edge removal can make it not an I-Map, it is called Minimal I_Map.
   
3. **Bayesian Network의 유용한점:** 전체 그래프(joint distribution)를 다음과 같이 decompose하여 local하게 identify할 수 있음. (as a data structure that provides the skeleton for representing a joint distribution compactly in a factorized way)
   
   1. 예시: ![image-20220426164623913](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426164623913.png)
   2. ![image-20220426164749446](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426164749446.png)
   
4. **Factorization Property:**
   1. ![image-20220426165046316](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426165046316.png)
   2. ![image-20220426165500636](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426165500636.png)
   
5. **Zero probabilities:** One can never condition away a zero probability, no matter how much evidence we get in the condition of an event that is extremely unlikely but not impossible.

6. **D-separation (Directed separation)**
   
   1. ![image-20220426181245808](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220426181245808.png)
      1. (a), (b), (c): if z is observed, x and y are independent. (x, y are somewhat dependent --> independent when z is observed.)
      2. (d): X, Y are independent if Z is not observed. When Z is observed, information flows in X->Z<-Y ways (X and Y become correlated). We call this structure "V-structure".
      3. We say X and Y are d-separated given Z, denoted ![image-20220427112118195](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427112118195.png)
         1. If there is no active trail between any node X and Y given Z.
         2. (a): Causal Trail: X --> Z --> Y: active if and only if Z is not observed.
         3. (b): Evidential Trail: X <-- Z <-- Y: active if and only if Z is not observed.
         4. (c): Common cause:   X <-- Z --> Y: active if and only if Z is not observed.
         5. (d): Common effect: X --> Z <-- Y: active if and only if either Z or one of Z's descendants is observed.
            1. If no edge exists between X and Y for X --> Z <-- Y case, we call the structure immorality. If there exist such edge, it is called a covering edge for the v-structure.
   2. Active / Inactive: When information flows like above example 6.D-separation-2, we call it active.
      1. ![image-20220427111605781](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427111605781.png)
      2. If there is no active trail between any node X and Y given Z --> D-separated.
   3. **When two nodes X and Y are d-separated given some Z, we guaranteed that they are in fact conditionally independent give Z.** (Soundness of d-separation)
   4. **D-separation detects all possible independencies.** (Completeness of d-separation)
   5. If X and Y are not d-separated give Z in G, then X and Y are dependent in all distributions P that factorize over G.
   
7. **I-Equivalence:** I(G) indicates a set of conditional independence assertions that are associated with a graph. The X --> Z --> Y, X <-- Z <-- Y, X <-- Z --> Y are I-equivalent.

   1. Two graphs structures K_1 and K_2 are I-equivalent if I(K_1) = I(K_2). The set of all graphs over X is partitioned into a set of mutually exclusive and exhaustive I-equivalence classes, which are the set of equivalence classes induced by the I-equivalence relation.
   2. If two graphs over X and if the two graphs have same skeleton (undirected graphs) and the same set of v-structures(immoralities), then they are I-equivalent.

   

### B. Bayesian Network Building Method

1. **Steps for identifying the undirected PMAP-skeleton:** If X and Y are not adjacent, we would be able to find a set of variables that makes these two variables conditionally independent. The conditioned variables are called witness of their independence.
   1. If we find the witnesses of X_i and X_j, we gather them first.
   2. If we know the witnesses are not from neighbors of either X_i and X_j, then remove the witnesses from the witness list.
2. **Steps for identifying the immorality directions of skeleton:** The main cue for learning about edge directions in the graph are immoralities. 
   1. If a triplet of variables X, Z, Y is a potential immorality if the graph contains X-Z-Y edges without an edge between X and Y.
   2. If the triplet is really an immorality in the graph, then the X and Y cannot be independent given Z.
   3. ![image-20220427143926854](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427143926854.png)
   4. Immorality based orientation.
3. **Step 1 for the rest of the undirected edges:**
   1. ![image-20220427145728465](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427145728465.png)
   2. 위에서 B의 경우 C와는 I-equivalent하지 않음 (I-equivalent: same skeleton and same immoralities.) 그래서 무조건 A만 가능함.
4. **Step 2 for the rest of the undirected edges:**
   1. If X --> Y -- Z and if they are not in immorality, X --> Y --> Z.
   2. No cyclic graph for DAG. (Acyclicity)
   3. If X --> Y and Y -- Z then X --> Z if no immorality.
   4. Contradiction method. If Z --> X, Y1 --> X and Y2 --> X for acyclic and then it makes it immorality so impossible.
      1. ![image-20220427151755483](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427151755483.png)
   5. **Process Examples): B --> C (A is the answer.)**
      1. ![image-20220427160414129](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427160414129.png)
      2. Complete the directions of I-equivalence class.



### C. Bayesian Structure Learning 783~843

For a answer graph G, there can be multiple number of perfect maps that satisfies the I-equivalence class as G with limited data. All of these I-equicalent graphs are equally good structures and there we cannot distinguish between them based only on the data D.

Considering "data fragmentation", the data used to estimated CPD fragment into more bins, leaving fewer instances in each bin to estimate the parameter. (More parents or edges --> more bins required.)

1. **Constraint based bayesian structure learning:** These approaches view a bayesian network as a representation of independencies. The goal is to find an (minimal) I-map that satisfies a set of independencies. The individual failure can be sensitive to building the model.
   1. Null-hypothesis is the data sampled from a distributions of two variables that are independent. We make a decision rule that accepts or not accepts this hypothesis (independence). If the decision rule R(D) = Accept, then the variables are independent, vice versa.
   2. Probably, compute false rejection to make and check a good decision rule above.
      1. By deviance (Larget value of deviance implies that D is far away from the null hypothesis)
         1. Mutual Information based deviance measure:
            1. ![image-20220427192137983](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220427192137983.png)
            2. The rule accepts the hypothesis if the deviance is small by a threshold, vice versa.
            3. The choice of threshold t determines the false rejection probability of the decision rule.
2. **Score based bayesian structure learning:** We define a hypothesis space of potential models and a scoring function that measures how well the model fits the observed data. Score-based methods the whole structure at once, so they are less sensitive to individual failtues and better at making compromises between which variables are dependent in the data. But this score based structure learning has a search problems since there are numerous possible structures to test when the number of nodes increases.
   1. Score function: ![image-20220429005056124](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220429005056124.png)
      1. Where G0 if X and Y are independent and G1 if X --> Y.
   2. Since we have uncertainty both over structure and parameter, we define a prior structure that gives a prior probability on different graph structures.
   3. ![image-20220502103139180](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220502103139180.png)
   4. ![image-20220502103154312](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220502103154312.png)
   5. Above number 4 describes "marginal likelihood" which differs from maximum likelihood. Marginal likelihood measures the expected likelihood averaged over different choices of theta_graphs, so it is less optimistic and is more conservative in our estimate of the goodness of the model.
   6. **The bayesian score seems to be biased toward simpler structures, but as it gets more data, it is willing to recognize that a more complex strucutre is necessary.**
   7. ![image-20220502104644013](C:\Users\kwanl\AppData\Roaming\Typora\typora-user-images\image-20220502104644013.png)
   8. From above number 7, second term can be ignored.
3. **Bayesian Model Averaging:** This method generates an ensemble of possible structures.


