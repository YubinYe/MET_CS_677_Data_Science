# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import  LabelEncoder
from sklearn import tree


df=pd.read_csv('/Users/yubinye/Downloads/677/hw5/data.csv')

df.head()

'''
Question 3
'''

X = df[['LB', 'ALTV', 'Min', 'Mean']].values


le = LabelEncoder()
Y = le.fit_transform(df[['True_Label']].values)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.5, random_state=1)

tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
tree_classifier = tree_classifier.fit(X_train, Y_train)

prediction = tree_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
confusion_matrix_tree = confusion_matrix(Y_test, prediction) 

print('\naccuracy of Decision Tree model is\n',round(accuracy,4))
print('\nconfusion_matrix of Decision Tree model is\n',confusion_matrix_tree)


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

result_q3=tpr_tnr(prediction,Y_test)
print(result_q3)

result_q3.T.to_csv('/Users/yubinye/Downloads/677/hw5/q3.csv')