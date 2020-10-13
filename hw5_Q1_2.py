# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import  LabelEncoder

df = pd.ExcelFile('/Users/yubinye/Downloads/677/hw5/CTG.xls')
df = pd.read_excel(df, 'Raw Data')
df.head(5)
#df.info()

# by df.info(), I find NaN value
df.drop([0],inplace = True)
df.dropna(how='any',inplace=True)

# re-range index
df.index=range(len(df))

'''
Question 1.2
'''
df['True_Label']=None
df['True_Label'][df['NSP'] == 1.0] = '1'
df['True_Label'][df['NSP'] > 1.0] = '0'
df.to_csv('/Users/yubinye/Downloads/677/hw5/data.csv')
'''
Question 2
'''

X = df[['LB', 'ALTV', 'Min', 'Mean']].values



le = LabelEncoder()
Y = le.fit_transform(df[['True_Label']].values)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.5, random_state=1)




NB_classifier = GaussianNB().fit(X_train, Y_train)
prediction = NB_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
confusion_matrix_NB = confusion_matrix(Y_test, prediction) 
print('\naccuracy of Naive Bayesian model is\n',round(accuracy,4))
print('\nconfusion_matrix of Naive Bayesian model is\n',confusion_matrix_NB)

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

result_q2=tpr_tnr(prediction,Y_test)
print(result_q2)

result_q2.T.to_csv('/Users/yubinye/Downloads/677/hw5/q2.csv')