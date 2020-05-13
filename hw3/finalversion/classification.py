#models
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
#neccesities
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
#add ons
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,normalize
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.utils import resample
from functools import partial
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
#from imblearn.over_sampling import SMOTE, ADASYN
#print(df.describe())
#print(df.info())

def remove_nan(df, num):
    nan = np.array((df.isnull().sum(axis = 1)))
    count =0
    nana = dict()
    for i in nan:
        if i > num:
            nana[count] = nan[count]
        count += 1
    num = []
    for i in nana.keys():
        num.append(i)
    df = df.drop(df.index[num])
    return df

def find_best_features(df, label):
    df = df.drop(columns = ['Date'])
    Location_index = df.columns.get_loc("Location")
    WindGustDir_index = df.columns.get_loc("WindGustDir")
    WindDir9am_index = df.columns.get_loc("WindDir9am")
    WindDir3pm_index = df.columns.get_loc("WindDir3pm")
    labelencoder = LabelEncoder()
    df['Location'] = labelencoder.fit_transform(df['Location'].astype(str))
    df['WindGustDir'] = labelencoder.fit_transform(df['WindGustDir'].astype(str))
    df['WindDir9am'] = labelencoder.fit_transform(df['WindDir9am'].astype(str))
    df['WindDir3pm'] = labelencoder.fit_transform(df['WindDir3pm'].astype(str))
    #fill rest of nan
    df['WindGustDir'] = df['WindGustDir'].fillna(17)
    df['WindDir9am'] = df['WindDir9am'].fillna(17)
    df['WindDir3pm'] = df['WindDir3pm'].fillna(17)
    df = df.fillna(df.mean()) #use mean to fill the rest of data
    #sellect features
    score_func = partial(mutual_info_classif, discrete_features=[WindGustDir_index,WindDir9am_index,WindDir3pm_index,Location_index])
    selector = SelectKBest(score_func, k=10).fit(df, label)
    GetSupport= selector.get_support(True)
    return GetSupport

def isNaN(num):
    return num != num

def split_date(df):
    Date = df['Date'].tolist()
    #print(len(Date))
    Month = []
    for i in range(len(Date)):
        if not isNaN(Date[i]):
            Month.append(int(Date[i].split('-')[1]))
        else:
            Month.append(0)
    df['Month'] = Month 
    return df.drop(columns = ['Date'])

#find same month and Location average
def fillwithaverage(df, metric): 
    Month = df['Month'].tolist()
    Location = df['Location'].tolist()
    metric_list = df[metric].tolist()
    for i in range(len(metric_list)):
        if isNaN(metric_list[i]):
            count = 0
            ammount = 0
            for j in range(len(metric_list)):
                if (Month[j] == Month[i] and Location[j] == Location[i]):
                    if not isNaN(metric_list[j]):
                        #print(Month[j],Location[j],metric_list[j])
                        ammount += metric_list[j]
                        count += 1
            if count>0:
                metric_list[i] = ammount/count
    df[metric] = metric_list
    return df

labelencoder = LabelEncoder()
#preprocess data
def preprocess(df):
    #df = df.drop(columns = ['RISK_MM'])
    df = split_date(df)
    #hash wind 
    Location_index = df.columns.get_loc("Location")
    WindGustDir_index = df.columns.get_loc("WindGustDir")
    WindDir9am_index = df.columns.get_loc("WindDir9am")
    WindDir3pm_index = df.columns.get_loc("WindDir3pm")
    df['Location'] = labelencoder.fit_transform(df['Location'].astype(str))
    df['WindGustDir'] = labelencoder.fit_transform(df['WindGustDir'].astype(str))
    df['WindDir9am'] = labelencoder.fit_transform(df['WindDir9am'].astype(str))
    df['WindDir3pm'] = labelencoder.fit_transform(df['WindDir3pm'].astype(str))
    df['Location'] = df['Location'].fillna(50) #location max value 49
    df['WindGustDir'] = df['WindGustDir'].fillna(17) #location max value 16
    df['WindDir9am'] = df['WindDir9am'].fillna(17)
    df['WindDir3pm'] = df['WindDir3pm'].fillna(17)
    #fill rest of nan
    for col in df.columns:
        if col != "Location" and col != "Month":
            df = fillwithaverage(df, col)
    #df = df.drop(columns = ['Location'])
    df = df.fillna(df.mean()) #use mean to fill the rest of data
    #one hot encode wind
    #ct = ColumnTransformer([("wind", OneHotEncoder(), [WindGustDir_index,WindDir9am_index,WindDir3pm_index])], remainder = 'passthrough')
    #encoded = ct.fit_transform(df).toarray()
    #normalize column values
    #min_max_scaler = preprocessing.MinMaxScaler()
    #min_max_scaler = preprocessing.MaxAbsScaler()
    #encoded = min_max_scaler.fit_transform(encoded)
    encoded = normalize(df, norm='l2', axis=1) #best
    #scale colums
    #encoded = preprocessing.scale(encoded) #no good
    df = pd.DataFrame(encoded)
    #print(df.describe())
    return df

def process_and_save(X_ans, df_train):
    #split data set
    labels = df_train['RainToday']
    data = df_train.drop(columns = ['RainToday'])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=40)

    #remove too much nan
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data = remove_nan(train_data,7)
    #resample data
    negative = train_data[train_data.RainToday==0]
    positive = train_data[train_data.RainToday==1]
    pos_upsampled = resample(positive,
     replace=True, # sample with replacement
     n_samples=len(negative), # match number in majority class
     random_state=101) # reproducible results
    train_data = pd.concat([negative, pos_upsampled])
    train_data = shuffle(train_data)

    y_train = train_data['RainToday']
    X_train = train_data.drop(columns = ['RainToday'])
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    #GetSupport = find_best_features(X_train,y_train)
    #print("GetSupport : ",GetSupport)

    #preprocess
    X_train = preprocess(X_train)
    #print(X_train)
    X_ans = preprocess(X_ans)
    X_test = preprocess(X_test)

    X_train.to_csv('processed_X_train_cutall.csv', index = False)
    X_ans.to_csv('processed_X_ans_cutall.csv', index = False)
    X_test.to_csv('processed_X_test_cutall.csv', index = False)
    y_train.to_csv('processed_y_train_cutall.csv', index = False)
    y_test.to_csv('processed_y_test_cutall.csv', index = False)

def main(): 
    #X_ans = pd.read_csv('test.csv')
    #df_train = pd.read_csv('train.csv')
    #process_and_save(X_ans, df_train)
    
    X_train = pd.read_csv('processed_X_train_cutall.csv')
    X_ans = pd.read_csv('processed_X_ans_cutall.csv')
    X_test = pd.read_csv('processed_X_test_cutall.csv')
    y_train = pd.read_csv('processed_y_train_cutall.csv')
    y_test = pd.read_csv('processed_y_test_cutall.csv')

    #print(X_train.describe())
    #X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    #selector = SelectKBest(chi2, k=50).fit(X_train, y_train) #doesn't make difference
    #GetSupport= selector.get_support(True)
    #print("GetSupport : ",GetSupport)
    #X_train = X_train.iloc[:,GetSupport]
    #X_test = X_test.iloc[:,GetSupport]
    #X_ans = X_ans.iloc[:,GetSupport]

    #train model
    model = XGBClassifier(silent=0, learning_rate= 0.3, min_child_weight=7, max_depth=6, gamma=0.2, subsample=1, 
                            max_delta_step=0, colsample_bytree=1, reg_lambda=3, scale_pos_weight=0.9, n_estimators=100, seed=101)

    model.fit(X_train,y_train)

    #predict
    y_pred_decision = model.predict(X_test)
    print('Accuracy: %f' % accuracy_score(y_test, y_pred_decision))
    print('f1-score: %f' % f1_score(y_test, y_pred_decision))

    ans_pred = model.predict(X_ans)
    df_sap = pd.DataFrame(ans_pred.astype(int), columns = ['RainToday'])
    df_sap.to_csv('myAns_test.csv',  index_label = 'Id')

def test_models():
    X_train = pd.read_csv('processed_X_train_cut.csv')
    X_ans = pd.read_csv('processed_X_ans_cut.csv')
    X_test = pd.read_csv('processed_X_test_cut.csv')
    y_train = pd.read_csv('processed_y_train_cut.csv')
    y_test = pd.read_csv('processed_y_test_cut.csv')

    #train model
    model = XGBClassifier(silent=0, learning_rate= 0.3, min_child_weight=7, max_depth=6, gamma=0.2, subsample=1, 
                            max_delta_step=0, colsample_bytree=1, reg_lambda=3, scale_pos_weight=0.9, n_estimators=100, seed=101)
    #model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
    #model = DecisionTreeClassifier(random_state=101)
    #model = LogisticRegression(class_weight='balanced', random_state=101)

    model.fit(X_train,y_train)

    #predict
    y_pred_decision = model.predict(X_test)
    print('Accuracy: %f' % accuracy_score(y_test, y_pred_decision))
    print('f1-score: %f' % f1_score(y_test, y_pred_decision))

    ans_pred = model.predict(X_ans)
    df_sap = pd.DataFrame(ans_pred.astype(int), columns = ['RainToday'])
    df_sap.to_csv('myAns_LogisticRegression.csv',  index_label = 'Id')


if __name__ == '__main__':
    main()
    #test_models()