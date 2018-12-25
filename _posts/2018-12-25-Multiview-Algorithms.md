---
layout : post
title : Multi-view Algorithms
---
해당 글은 고려대학교 강필성 교수님의 2018학년도 2학기 Business Analytics 수업을 참조로 작성되었습니다.

### 1. Semi-Supervised Learning
> Multi-view Algorithms에 대한 설명을 하기에 앞서, 우리는 Semi-Supervised Learning에 대해 간략하게 알아야 한다. Semi-Supervised Learning이란 Label이 되어 있는 Data와 Label이 되어 있지 않은 Data가 섞여 있는 경우를 의미한다. 따라서 Semi-Supervised Learning에서 해결해야 할 핵심 문제는'Label 되어 있는 data를 활용하여 Label 되어 있지 않은 Data를 어떻게 처리할 것인가?'가 이다. 이와 관련해서 Self-Training, Generative Model 활용, Graph-based SSL, Multi-view Algoritms 등이 있다. 이번 글에서는 Multi-view Algorithms(Co-Training)에 대해 자세히 알아보려 한다.
 
>> [Semi-Supervised Learning(강필성 교수님 Lecture Note 5강 p15참고)]![semi-supervised](https://user-images.githubusercontent.com/46133856/50424712-436b1880-08ac-11e9-8550-72c4dc4d9887.jpg)
 
### 2. Multi-view Algorithms(Co-Training)
> Multi-view Algorithm은 서로 다른 기법의 관점에서 문제를 해결 할 수 있다면 이를 통해 서로 도움을 줘서 문제를 해결해 보자. 라는 아이디어에서 시작한다. 이는 앙상블 기법하고 비슷하다고 볼 수 있다. 
 
### 3. 활용 사례



### 4. Python Code
 아래부터 나오는 Python Code는 2017년 Business Analytics 강의를 수강 하신 
