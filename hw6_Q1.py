# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df=pd.read_csv('seeds_dataset.txt',header=None,delimiter="\t+")
df.info()

df.columns= ['f1','f2','f3','f4','f5','f6','f7','class']
df.head()

df.to_csv('df.csv',index=False)

# My last digit of BUID is 5, then R=2.
df_R2 = df.loc[df['class'] != 2]
df_R2.to_csv('df_R2.csv',index=False)

X = df_R2[['f1','f2','f3','f4','f5','f6','f7']].values
le = LabelEncoder()
Y = le.fit_transform(df_R2['class'].values)

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


def compute_svm(method,n):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, random_state=10)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        svm_classifier = svm.SVC(kernel =method,degree=n)
        svm_classifier.fit(X_train,Y_train)
        prediction = svm_classifier.predict(X_test)
        accuracy = np.mean(prediction == Y_test)
        confusion_matrix_svm = confusion_matrix(Y_test, prediction) 
        result=tpr_tnr(prediction,Y_test)

        return {'accuracy':round(accuracy,4),
        'confusion_matrix_svm':confusion_matrix_svm,'result':result}


'''
Question 1.1
'''

print('result for linear kernel SVM is \n')
print(compute_svm('linear',1))

'''
Question 1.2
'''
print('result for Gaussian kernel SVM is \n')
print(compute_svm('rbf',1))

'''
Question 1.3
'''
print('result for polynomial kernel SVM is \n')
print(compute_svm('poly',3))



q11=compute_svm('linear',1)['result']
q12=compute_svm('rbf',1)['result']
q13=compute_svm('poly',3)['result']

q_1_ALL= pd.concat([pd.concat([q11.T,q12.T],axis=0),q13.T],axis=0)

q_1_ALL.index.name='index'
q_1_ALL.index=['linear kernel SVM','Gaussian kernel SVM','polynomial kernel SVM']

print(q_1_ALL)

q_1_ALL.to_csv('Q1_all.csv',index=False)