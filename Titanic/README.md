# Kaggle Titanic - Machine Learning from Disaster

| **Variable** | **Definition**                            | **Key**                                        |
| ------------ | ------------------------------------------ | ---------------------------------------------- |
| survival     | Survival                                   | 0 = No, 1 = Yes                                |
| pclass       | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex          | Sex                                        |                                                |
| Age          | Age in years                               |                                                |
| sibsp        | # of siblings / spouses aboard the Titanic |                                                |
| parch        | # of parents / children aboard the Titanic |                                                |
| ticket       | Ticket number                              |                                                |
| fare         | Passenger fare                             |                                                |
| cabin        | Cabin number                               |                                                |
| embarked     | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |


### 1. obtain the response logs from learners to samples by running

```python 
python gen_model_result.py
```

the hyperparameters of each learner are listed in corresponding learner file in "model/...".

### 2. the response logs of learners on the test set of Titanic are saved in titanic.csv file.
