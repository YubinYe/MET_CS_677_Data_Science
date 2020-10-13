# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import  LabelEncoder

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/Users/yubinye/Downloads/677/hw5/data.csv')



'''
Question 4
'''

X = df[['LB', 'ALTV', 'Min', 'Mean']].values



le = LabelEncoder()
Y = le.fit_transform(df[['True_Label']].values)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.5, random_state=1)


df_new = pd.DataFrame(columns=['d=1','d=2',\
        'd=3','d=4','d=5'],index=range(1,11))

for i in range(1,11):
    for j in range(1,6):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,\
                test_size=0.5, random_state=11)
        model = RandomForestClassifier(n_estimators =i, max_depth =j,\
                                criterion ='entropy', random_state=12)
        model.fit(X_train, Y_train )
        prediction = model.predict(X_test)
        error_rate = np.mean(prediction != Y_test)
        df_new.iloc[i-1,j-1] = error_rate

print(df_new)



x=df_new.index
colors=['lightsteelblue','hotpink','slateblue','darkkhaki','gold']
plt.figure(figsize=(10,8)) 

for i in range(0,5):
        plt.plot(x,df_new[df_new.columns[i]],color=colors[i], linewidth=2,
        alpha=1,marker='o',markerfacecolor=colors[i],markersize=6)
        plt.legend(labels = df_new.columns, loc = 'best')        



plt.title('Random Forest for LB, ALTV, Min, Mean',fontsize=14)
plt.yticks(np.arange(0.1, 0.23, step=0.005))
plt.xticks(np.arange(1, 11, step=1))
plt.xlabel('number of estimators', fontsize=14)
plt.ylabel('error rate', fontsize=14)
plt.grid(alpha= 0.5)




X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5, random_state=1)
model = RandomForestClassifier(n_estimators =6, max_depth =5,criterion ='entropy', random_state=1)
model.fit(X_train, Y_train )
prediction = model.predict(X_test)
accuracy = np.mean(prediction == Y_test)
confusion_matrix_rf = confusion_matrix(Y_test, prediction) 

print('\naccuracy of Random Forest model is\n',round(accuracy,4))
print('\nconfusion_matrix of Random Forest model is\n',confusion_matrix_rf)


def tpr_tnr(predict,test):
    TP,FP,TN,FN,match=0,0,0,0,0
    for i in range(len(test)):
        true_lb=test[i]
        my_lb=predict[i]
        if true_lb == my_lb:
            match+=1
        if (true_lb == 0) & (my_lb == 0):
                TP+=1
        elif (true_lb == 1) & (my_lb == 0):
                FP+=1
        elif (true_lb == 1) & (my_lb == 1):
                TN+=1
        elif (true_lb == 0) & (my_lb == 1):
                FN+=1
        else:
            continue
    TPR=TP/(TP+FN)
    TNR = TN/(TN+FP)
    accuracy=match/len(test)
    dic=[TP,FP,TN,FN,round(accuracy,6),round(TPR,6),round(TNR,6)]
    result=pd.DataFrame(dic)
    result.index=['TP','FP','TN','FN','accuracy','TPR','TNR']
    return result

result_q4=tpr_tnr(prediction,Y_test)
print(result_q4)




'''
Question 5
'''



q2=pd.read_csv('/Users/yubinye/Downloads/677/hw5/q2.csv')
q3=pd.read_csv('/Users/yubinye/Downloads/677/hw5/q3.csv')

q23= pd.concat([q2,q3],axis=0)
q234= pd.concat([q23,result_q4.T],axis=0)

q234.drop(['Unnamed: 0'],axis=1,inplace=True)
q234.index=['naive bayesian','decision tree','random forest']
q234.index.name='model'
print(q234)

q234.to_csv('/Users/yubinye/Downloads/677/hw5/all.csv')