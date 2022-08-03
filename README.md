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

+ XGBoost_train.py
  + $ python XGBoost_optimization.py
  + Optimization 코드가 포함이 되어 있습니다. (약 6시간 걸림)
  + 최적화된 parameter 기반 점수 복원 가능한 코드는 'XGBoost_test.py'에 있습니다.

+ XGBoost_test.py
  + $ python XGBoost_test.py
  + data 폴더에 'XGBoost.csv'파일이 생성됩니다.
