---
layout : post
title : Multi-view Algorithms
---
해당 글은 고려대학교 강필성 교수님의 2018학년도 2학기 Business Analytics 수업을 참조로 작성되었습니다.

### 1. Semi-Supervised Learning
> Multi-view Algorithms에 대한 설명을 하기에 앞서, 우리는 Semi-Supervised Learning에 대해 간략하게 알아야 한다. Semi-Supervised Learning이란 Label이 되어 있는 Data와 Label이 되어 있지 않은 Data가 섞여 있는 경우를 의미한다. 따라서 Semi-Supervised Learning에서 해결해야 할 핵심 문제는 **'Label 되어 있는 data를 활용하여 Label 되어 있지 않은 Data를 어떻게 처리할 것인가?'**가 이다. 이와 관련해서 Self-Training, Generative Model 활용, Graph-based SSL, Multi-view Algoritms 등이 있다. 이번 글에서는 Multi-view Algorithms(Co-Training)에 대해 자세히 알아보려 한다.
 
>> [Semi-Supervised Learning(강필성 교수님 Lecture Note 5강 p15참고)]![semi-supervised](https://user-images.githubusercontent.com/46133856/50424712-436b1880-08ac-11e9-8550-72c4dc4d9887.jpg)
  
### 2. Multi-view Algorithms(Co-Training)
> Multi-view Algorithm의 핵심은 같은 문제에 대해서도 서로 다른 관점에서 문제를 해결 할 수 있다면 각각의 다른 관점에서 해결한 결과물을 활용해서 문제를 해결해 보자는 것이다. 이는 앙상블 기법과 유사한 접근 방법이라고 할 수 있다. 앙상블 기법 또한 하나의 문제에 대해 한 가지 알고리즘을 적용하는 것보다는 여러 가지 알고리즘을 활용하여 문제를 해결한다면 더 좋은 결과를 얻을 수 있다는 아이디어에서 시작한다. 

### 3. 실제 적용 과정
> Text Classfication 문제에서 TF-IDF, LDA, Doc2vec 이 3가지 알고리즘을 확용하여 Unlabeled Data에 Label을 다는 기법에 대해 설명해보고자 한다.
> 1. 먼저 Text Data를 TF-IDF, LDA, Doc2vec 3가지 알고리즘에 대해 적용
>>[Text Classfication(강필성 교수님 Lecture Note 5강 pp59~61 참고)]![1](https://user-images.githubusercontent.com/46133856/50424915-4a942580-08b0-11e9-881a-bb5d4edc015e.JPG)

> 2. 각각의 Classfication Model을 구현해서 Labeled Data를 활용해 Unlabeled Data의 Label을 예측
>>[Predict Unlabeled Data(강필성 교수님 Lecture Note 5강 pp59~16 참고)![2](https://user-images.githubusercontent.com/46133856/50424943-b5ddf780-08b0-11e9-83c4-8bee62af0e8e.JPG)

> 3. 높은 confidence를 가지는 Data는 Label을 그대로 남기고 나머지 Data 다시 1~3 과정 반복하며 Label을 달아줌
>>[Repeat Multi-view Algorithms] ![3](https://user-images.githubusercontent.com/46133856/50424963-56341c00-08b1-11e9-91e9-e9202e3a9f21.JPG)

### 4. Python Code
> 아래부터 나오는 Python Code는 2017년 Business Analytics 강의를 수강 하신 이준헌 석사 과정님의 Code를 활용했습니다.
