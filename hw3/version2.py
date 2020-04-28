import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('processedDataNorm.csv')
dff = pd.read_csv('train.csv')
train_end_idx = len(dff) 
df_test = pd.read_csv('test.csv')
df_test['RainToday'] = np.zeros((len(df_test),))

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing

#pca = PCA(n_components=17, random_state=420) # (24609, 17)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns = ['RainToday']).values[:train_end_idx, :],
    df['RainToday'].values[:train_end_idx], test_size=0.3)
X_ans = df.drop(columns = ['RainToday']).values[train_end_idx:, :]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#final_dt = DecisionTreeClassifier()                   
#final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=10, random_state=0, oob_score=True) 

#final_bc.fit(X_train, y_train)
#y_pred_decision = final_bc.predict(X_test) # 0.244
model8 = SVC(kernel="rbf", C=0.025,random_state=101, probability=True)
model2 = LogisticRegression(class_weight='balanced', random_state=101) #0.344418
model1 = DecisionTreeClassifier(class_weight='balanced', random_state=101, criterion="entropy", min_samples_leaf=4, min_samples_split=9) # 0.279
model3 = RandomForestClassifier(class_weight='balanced', random_state=101) #0.236
model4 = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101, alpha=0.00011) #0.019
model6 = GaussianNB() # 0.31
model7 = AdaBoostClassifier(n_estimators=90, learning_rate= 1.4, random_state=101) # 0.27
#model.fit(X_train, y_train)
#weights=[0.1, 0.35, 0.25,0.3],
model = VotingClassifier(estimators=[('d2', model2),('d6', model6)], voting='soft')
model.fit(X_train, y_train) #  0.409821
#model = GaussianNB()
#model1.fit(X_train,y_train)

y_pred_decision = model.predict(X_test)
print('Accuracy: %f' % accuracy_score(y_test, y_pred_decision))
print('f1-score: %f' % f1_score(y_test, y_pred_decision))


#ans_pred = model8.predict(X_ans)
#df_sap = pd.DataFrame(ans_pred.astype(int), columns = ['RainToday'])
#df_sap.to_csv('myAns5.csv',  index_label = 'Id')