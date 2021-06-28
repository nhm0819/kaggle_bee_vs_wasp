# bee_vs_wasp
Data 출처 : https://www.kaggle.com/jerzydziewierz/bee-vs-wasp <br>

### CNN의 기본적인 구조 학습 및 실습


##### bee vs insect
- model1 : train[loss: 0.4287 - accuracy: 0.8103], test[loss: 0.4017 - accuracy: 0.8125] : ResNet34 <br>
- model2 : train[loss: 0.2518 - accuracy: 0.9877], test[loss: 0.4425 - accuracy: 0.9307] : Custom CNN <br>
- model3 : train[loss: 0.1827 - accuracy: 0.9997], test[loss: 0.4485 - accuracy: 0.9284] : model2 간소화 --> overfitting<br>
- model4 : train[loss: 0.2195 - accuracy: 0.9964], test[loss: 0.4397 - accuracy: 0.9364] : model3 간소화 --> overfitting<br>


##### bee vs wasp vs insect
- model_DenseNet2_fold_15.h5 : test[loss=0.2544 -accuracy=0.9124] : DenseNet121 - Cross Validation<br><br><br>

---
###### 결론
##### 모델의 Layer를 간소화하는 것으론 한계가 있음.
##### Data Augmentation 시도
##### 추후 더 깊은 ResNet의 적용 및 Data Augmentation, Object Detection 등과 같은 기술들을 추가적으로 적용해 나갈 것<br><br>

---
###### 06/05
##### [bee, insect] 만 훈련한 뒤 [bee, wasp, insect] 로 확장
##### DenseNet을 Cross-Validation 으로 적용하여 벌, 말벌, 곤충을 분류할 때 정확도 91.2%
