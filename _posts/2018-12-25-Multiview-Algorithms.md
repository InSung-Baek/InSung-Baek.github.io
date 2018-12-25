---
layout : post
title : Multi-view Algorithms
---
해당 글은 고려대학교 강필성 교수님의 2018학년도 2학기 Business Analytics 수업을 참조로 작성되었습니다.

### 1. Semi-Supervised Learning
>Multi-view Algorithms에 대한 설명을 하기에 앞서, 우리는 Semi-Supervised Learning에 대해 간략하게 알아야 한다. Semi-Supervised Learning이란 Label이 되어 있는 Data와 Label이 되어 있지 않은 Data가 섞여 있는 경우를 의미한다. 따라서 Semi-Supervised Learning에서 해결해야 할 핵심 문제는 __'Label 되어 있는 data를 활용하여 Label 되어 있지 않은 Data를 어떻게 처리할 것인가?'__ 가 이다. 이와 관련해서 Self-Training, Generative Model 활용, Graph-based SSL, Multi-view Algoritms 등이 있다. 이번 글에서는 Multi-view Algorithms(Co-Training)에 대해 자세히 알아보려 한다.
 
>>###### [Semi-Supervised Learning(강필성 교수님 Lecture Note 5강 p3참고)]![semi](https://user-images.githubusercontent.com/46133856/50425536-335b3500-08bc-11e9-9e43-70ff51bc9068.jpg)

****
  
### 2. Multi-view Algorithms(Co-Training)
>Multi-view Algorithm의 핵심은 __같은 문제(여기서는 Unlabeled Data에 Label을 부여하는 것)에 대해서도 서로 다른 관점에서 문제를 해결 할 수 있다면 각각의 다른 관점에서 해결한 결과물을 활용해 문제를 더 정확하게 해결해 보자는 것이다.__ 이는 앙상블 기법과 유사한 접근 방법이라고 할 수 있다. 앙상블 기법 또한 하나의 문제에 대해 한 가지 알고리즘을 적용하는 것보다는 여러 가지 알고리즘을 활용하여 문제를 해결한다면 더 좋은 결과를 얻을 수 있다는 아이디어에서 시작하는 것이기 때문이다. 이처럼 Multi-view Algorithms에 경우 앙상블 기법과 유사한 아이디어에서 시작하기 때문에 앙상블 기법에서 핵심인 __Diversity를 어떻게 확보할 것인가?__ 에 대한 전략적인 고민이 가장 중요한 고민 중 하나이다. 

>>###### [Co-Training(CoDeL: A Human Co-detection and Labeling Framework,2013,shi)![cotraining](https://user-images.githubusercontent.com/46133856/50425367-f5f4a880-08b7-11e9-993d-488c5dfed7df.JPG)

### 3. 실제 적용 과정
>Text Classfication 문제에서 TF-IDF, LDA, Doc2vec 이 3가지 알고리즘을 확용하여 Unlabeled Data에 Label을 부여하는 기법에 대해 설명해보고자 한다.
>1. 먼저 Text Data를 TF-IDF, LDA, Doc2vec 3가지 알고리즘에 대해 적용한다.
>>###### [Text Classfication(강필성 교수님 Lecture Note 5강 pp59~61 참고)]![1](https://user-images.githubusercontent.com/46133856/50424915-4a942580-08b0-11e9-881a-bb5d4edc015e.JPG)

>2. 각각의 Classfication Model을 구현한 뒤, Labeled Data를 활용해 Unlabeled Data의 Label을 예측한다.
>>###### [Predict Unlabeled Data(강필성 교수님 Lecture Note 5강 pp59~16 참고)![2](https://user-images.githubusercontent.com/46133856/50424943-b5ddf780-08b0-11e9-83c4-8bee62af0e8e.JPG)

>3. 높은 confidence를 가지는 Data는 Label을 그대로 남기고 나머지 Data는 다시 1~3 과정 반복하며 Label을 달아준다.
>>###### [Repeat Multi-view Algorithms(강필성 교수님 Lecture Note 5강 pp59~61 참고)] ![3](https://user-images.githubusercontent.com/46133856/50424963-56341c00-08b1-11e9-91e9-e9202e3a9f21.JPG)

> 위 방법론의 경우 결국 __각각 학습한 Model에서 가장 자신 있는 상위의 결과물만 공유한 이후, 다시 학습해서 또 자신 있는 결과물만 공유하는__ 방향으로 반복해서 진행한다면 성능이 좋아질 것으로 예상 된다는 아이디어가 적용 된 것이다. 이처럼 위의 1~3번 반복 과정을 통해 Confidence가 높은 Data에 대해서만 Label을 달아주는 과정을 진행한다면 좀 더 정확한 Labeled Data를 얻을 수 있을 것으로 예상된다.

### 4. Python Code
>아래에 나오는 Python Code는 2017년 Business Analytics 강의를 수강 하신 이준헌 석사 과정님의 Code를 활용했습니다.
>먼저 Cotraining(Multi-view) Algorithms을 적용하는 데 기본적으로 필요한 패키지 Code입니다.

<pre><code>
#Cotraining을 구성하는데 필요한 패키지
import random
import numpy as np

#Cotraining에 적용 시킬 알고리즘
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#부가 기능을 위한 패키지
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
</code></pre>

>Cotraing Algorithms을 구현한 Python Code입니다.

<pre><code>
class CoTraining:

    def __init__(self, clf1, clf2, p, n, k, u):
        self.clf1 = clf1    # 첫 번째 Classifier
        self.clf2 = clf2    # 두 번째 Classifier
        self.p = p   # U'에서 1회 동안 Class 1로 정의할 instance 개수
        self.n = n   # U'에서 1회 동안 Class 0으로 정의할 instance 개수
        self.k = k   # U'를 업데이트 하면서 Label을 다는 과정의 반복 수
        self.u = u   # U'의 크기
        
        # hyper parameter가 0보다 작은 경우 Assert error
        assert(self.p > 0 and self.n > 0 and self.k > 0 and self.u > 0)


    def fit(self, X1, X2, y):  #Co-training을 수행 함수

        # Class의 정보가 -1인 경우 Unlabeled Data로 정의
        U = [i for i, y_i in enumerate(y) if y_i == -1]

        # U'를 랜덤하게 추출하기 위해서 Unlabeled data를 Shuffle
        random.shuffle(U)

        # parameter u의 크기로 U'를 정의
        U_ = U[-min(len(U), self.u):]

        # Unlabeled data에서 U'크기 만큼을 제외
        U = U[:-len(U_)]

        # Class의 정보가 -1이 아닌 경우 labeled Data로 정의
        L = [i for i, y_i in enumerate(y) if y_i != -1]

        it = 0 

        # k번 만큼 Unlabeled data를 Labeled data로 변화 시키는 과정
        while it != self.k and U:
            it += 1
            
            # Labeled data를 통해 두 개의 Classifier를 학습
            self.clf1.fit(X1[L], y[L])
            self.clf2.fit(X2[L], y[L])
            
            # Unlabeled data를 학습된 Classifier로 Class 예측
            y1 = self.clf1.predict(X1[U_])
            y2 = self.clf2.predict(X2[U_])
            # Unlabeled data를 학습된 Classifier로 probability 예측
            y1_proba = self.clf1.predict_proba(X1[U_])
            y2_proba = self.clf2.predict_proba(X2[U_]) 
            
            # 두 Classifier의 예측 Class가 다른 instance는 확률을 -1로 변경
            # 이 경우는 각 Classifier의 예측확률이 높아도 배제하기 위함
            for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
                if y1_i != y2_i:
                    y1_proba[i,:] = 0
                    y2_proba[i,:] = 0
            
            # 각 Classifier에서 class0, 1이 될 확률이 높은 상위 n, p개 추출
            idx_clf1_n = np.argsort(y1_proba[:,0])[-self.n:]
            idx_clf2_n = np.argsort(y2_proba[:,0])[-self.n:]
            idx_clf1_p = np.argsort(y1_proba[:,1])[-self.p:]
            idx_clf2_p = np.argsort(y2_proba[:,1])[-self.p:]
            
            # n, p의 index를 리스트화 하고 순서대로 정렬
            n = list(idx_clf1_n) + list(idx_clf2_n)
            p = list(idx_clf1_p) + list(idx_clf2_p)
            n.sort()
            p.sort()
            
            # 중복 제거
            dup_n = []
            for i in range(0, 2*self.n -1 , 1):
                if n[i] == n[i+1]:
                    dup_n.append(n[i])  
            for i in dup_n:
                n.remove(i)
            
            dup_p = []
            for i in range(0, 2*self.p -1 , 1):
                if p[i] == p[i+1]:
                    dup_p.append(p[i])  
            for i in dup_p:
                p.remove(i)
            
            # 추출된 instance의 index에 Labeling        
            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0
            
            # Labeling된 instance를 Label data로 추가
            L.extend([U_[x] for x in p])
            L.extend([U_[x] for x in n])
            
            # 추출된 instance(2p+2n-중복)의 index를 U'에서 제거
            remove = p + n
            remove.sort()
            remove.reverse()

            for i in remove: 
                U_.pop(i)
            
            
            # U'에서 빠진만큼 Unlabeled data에서 채워 넣기
            add_cnt = 0 
            num_to_add = len(remove)
            while add_cnt != num_to_add and U:
                add_cnt += 1
                U_.append(U.pop())

        # 최종적으로 추가완료 된 Labeled data로 각 Classifier를 학습
        self.clf1.fit(X1[L], y[L])
        self.clf2.fit(X2[L], y[L])

    # 두개의 Classifier의 Class예측 확률의 합쳐서 최종 Class 예측
    def predict(self, X1, X2):

        y1 = self.clf1.predict(X1)
        y2 = self.clf2.predict(X2)

        # 두 Classifier가 동일한 Class시 예측하면 해당 Class로 판정
        # 그렇지 않은 경우, 확률의 합의 비교하여 높은 Class로 판정
        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
            else:
                y1_probs = self.clf1.predict_proba([X1[i]])
                y2_probs = self.clf2.predict_proba([X2[i]])
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) 
                               in zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = sum_y_probs.index(max_sum_prob)
        return y_pred
    
    # 판정을 첫번째 Classifier를 통해서만 예측
    def predict_clf1(self, X1, X2):

        y_pred = self.clf1.predict(X1)

        return y_pred        
</code></pre>

>평가를 위한 Data Set입니다. Labeled Data와 Unlabeled Data를 구분한 Code입니다.

<pre><code>
if __name__ == '__main__':   
    accuracy = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for iteration in range(1,11,1):
        #평가하기 위한 데이터 생성
        N_SAMPLES = 5000    # 총 instance 개수
        N_FEATURES = 10    # 총 feature개수
        N_REDUNDANT = 4    # feature간 Correlation를 가지는 feature 개수
        Lable_Percent = 5   # 총 instance중에 Labeled data를 백분율을 설정
        
        # Data 생성
        X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                   n_redundant=N_REDUNDANT, n_informative=N_REDUNDANT)
        
        #Labeled, Unlabeled data를 설정

        y[:N_SAMPLES*(100-2*Lable_Percent)//100] = -1
        X_test = X[-N_SAMPLES*Lable_Percent//100:]
        y_test = y[-N_SAMPLES*Lable_Percent//100:]
        X_labeled = X[N_SAMPLES*(100-2*Lable_Percent)//100:-N_SAMPLES*Lable_Percent//100]
        y_labeled = y[N_SAMPLES*(100-2*Lable_Percent)//100:-N_SAMPLES*Lable_Percent//100]
        y = y[:-N_SAMPLES*Lable_Percent//100]
        X = X[:-N_SAMPLES*Lable_Percent//100]
        
        # Feature를 반반으로 나눔
        X1 = X[:,:N_FEATURES // 2]
        X2 = X[:, N_FEATURES // 2:]        
</code></pre>

>Logistic, Naive Bayes, Random Forests 단일 Algorithms을 적용한 Accuracy를 확인하는 Code입니다.

<pre><code>
print ('Logistic')
        
base_lr = LogisticRegression()
base_lr.fit(X_labeled, y_labeled)
y_pred = base_lr.predict(X_test)

print (classification_report(y_test, y_pred, digits=3))

accuracy[0].append('%0.3f'% accuracy_score(y_test, y_pred))
        
print ('Naive Bayes Classifier')
        
base_nb = GaussianNB()
base_nb.fit(X_labeled, y_labeled)
y_pred = base_nb.predict(X_test)
    
print (classification_report(y_test, y_pred, digits=3))
        
accuracy[1].append('%0.3f'% accuracy_score(y_test, y_pred))
        
print ('Random Forest')
        
base_rf = RandomForestClassifier(n_estimators=100)      
base_rf.fit(X_labeled, y_labeled)
y_pred = base_rf.predict(X_test)
        
print (classification_report(y_test, y_pred, digits=3))
        
accuracy[2].append('%0.3f'% accuracy_score(y_test, y_pred))
</code></pre>

>Co-Training 방법을 적용한 Code입니다. (Random Forest + Random Forest/Logistic/Naive Bayes Algorithms)  

<pre><code>
#Cotrining을 이용하여 Classifier 조합별 성능 평가  
        
#Random Forest & Random Forest Cotraining        
print ('Random Forest-Random Forest CoTraining')
        
RR_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), 
                       RandomForestClassifier(n_estimators=100), 
                       p=2, n=2, k=20, u=100)
RR_co_clf.fit(X1, X2, y)
y_pred = RR_co_clf.predict(X_test[:, :N_FEATURES // 2], 
                           X_test[:, N_FEATURES // 2:])

print (classification_report(y_test, y_pred, digits=3))
accuracy[3].append('%0.3f'% accuracy_score(y_test, y_pred))
y_pred = RR_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], 
                                X_test[:, N_FEATURES // 2:])
        
print (classification_report(y_test, y_pred, digits=3))
accuracy[4].append('%0.3f'% accuracy_score(y_test, y_pred))
        
#Random Forest & Logistic Regression Cotraining
        
print ('Random Forest-Logistic Regression CoTraining')
RL_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), 
                       LogisticRegression(), p=2, n=2, k=20, u=100)
RL_co_clf.fit(X1, X2, y)
y_pred = RL_co_clf.predict(X_test[:, :N_FEATURES // 2], 
                           X_test[:, N_FEATURES // 2:])
        
print (classification_report(y_test, y_pred, digits=3))
accuracy[5].append('%0.3f'% accuracy_score(y_test, y_pred))
y_pred = RL_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], 
                                X_test[:, N_FEATURES // 2:])
        
print (classification_report(y_test, y_pred, digits=3))
accuracy[6].append('%0.3f'% accuracy_score(y_test, y_pred))
        
        
#Random Forest & Naive Bayes Classifier Cotraining
print ('Random Forest-Naive Bayes Classifier CoTraining')
RN_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), 
                       LogisticRegression(), p=2, n=2, k=20, u=100)
RN_co_clf.fit(X1, X2, y)
y_pred = RN_co_clf.predict(X_test[:, :N_FEATURES // 2], 
                           X_test[:, N_FEATURES // 2:])
        
print (classification_report(y_test, y_pred, digits=3))
accuracy[7].append('%0.3f'% accuracy_score(y_test, y_pred))
y_pred = RN_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], 
                                X_test[:, N_FEATURES // 2:])
        
print (classification_report(y_test, y_pred, digits=3))
accuracy[8].append('%0.3f'% accuracy_score(y_test, y_pred))
</code></pre>

### 5.Conclusion
>위의 Code를 통해 Multi-view Algorithms(Co-Training)을 구현해 본 결과, 가장 좋은 성능을 보인 것은 단일 Ensemble Model이었다. Co-Training Model은 단일 Rogistic, Naive Bayes Model보다는 좋은 성능을 보였지만, 단일 Ensemble에는 조금 미치지 못한 것으로 보였다. 이는 Co-Training에 사용한 Classfication Model끼리의 성능 차이가 있고, 더 다양한 Model을 활용한 Diversity를 확보하지 못했기 때문이라고 생각 된다. 따라서 __좋은 성능을 얻을 수 있는 Co-Training 방법론을 사용하기 위해서는 다양한 Model의 장,단점을 활용해 Diversity를 확보하는 것이 중요__ 하다고 생각 된다. 긴 글 읽어주셔서 감사합니다. 수정사항이 있을 시에는(babogato33@gamil.com)으로 언제든지 연락주세요.

>>###### [Test 결과(Youtube-Tutorial 17 Co Training,이준헌 참고)]![accuracy](https://user-images.githubusercontent.com/46133856/50425692-2429b680-08bf-11e9-8aab-5540e1e301cb.JPG)
