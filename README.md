# 환경
+ python : 3.7.13
+ biopython : 1.79
+ numpy : 1.91.1
+ scikit learn : 0.23.1
+ pandas : 1.0.5
+ tqdm : 4.56.0
+ optuna : 2.10.1
+ xgboost : 1.1.1


# 코드 실행 방법
+ Data_processing.py
  + 전처리 코드
  + 만들고 있음
  
+ XGBoost_train.py
  + $ python XGBoost_train.py
  + 학습 코드 (약 6시간 소요) 
  + model 폴더에 'XGBoost.pkl'이 생성됨

+ XGBoost_test.py
  + $ python XGBoost_test.py
  + 추론 코드 (25분 소요)
  + data 폴더에 'XGBoost.csv'파일이 생성됨
