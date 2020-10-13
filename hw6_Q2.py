# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import  LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df_R2=pd.read_csv('df_R2.csv')

X = df_R2[['f1','f2','f3','f4','f5','f6','f7']].values
le = LabelEncoder()
Y = le.fit_transform(df_R2['class'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, random_state=10)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors = 9)
knn_classifier.fit(X_train, Y_train)
prediction = knn_classifier.predict(X_test)




# among all Ks` which has best accuracy: 9,11, 13, 15
# 9 is the smallest K, so I will use K=9


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



def compute_knn(k,Y_test,prediction):
        accuracy = np.mean(prediction == Y_test)
        confusion_matrix_knn = confusion_matrix(Y_test, prediction) 
        result=tpr_tnr(prediction,Y_test)

        return {'accuracy':round(accuracy,4),
        'confusion_matrix_svm':confusion_matrix_knn,'result':result}



print('result for knn is \n')
print(compute_knn(9,Y_test,prediction))

q2=compute_knn(9,Y_test,prediction)['result'].T
print(q2)
q1=pd.read_csv('Q1_all.csv')
q1

q_all=pd.concat([q1,q2],axis=0)


q_all.index.name='index'
q_all.index=['linear kernel SVM','Gaussian kernel SVM',
'polynomial kernel SVM','knn']

print(q_all)


q_all.to_csv('Q2.csv')
