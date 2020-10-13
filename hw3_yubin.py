# -*- coding: utf-8 -*-

import pandas as pd

df=pd.read_csv('/Users/yubinye/Downloads/677/hw3/data_banknote_authentication.txt',header = None)
df.columns=['f1','f2','f3','f4','class']
df.head()


'''
Question 1.1
'''
df['color']=df['class'].apply(lambda x: 'red' if x>0 else 'green')


'''
Question 1.2
'''
df0=df[df['class']==0]
df1=df[df['class']==1]

def u_std(data):
    u_list=[]
    count=0
    for i in df.columns[:4]:
        u=data[i].mean() 
        u_list.append(u)
        std=data[i].std()
        u_list.append(std)
    count+=1
    
    
    table = pd.DataFrame(u_list,index=['u(f1)','std(f1)',
    'u(f2)','std(f2)','u(f3)','std(f3)','u(f4)','std(f4)'])

    return table

tb=pd.concat([u_std(df0).T, u_std(df1).T], axis=0)
tb=pd.concat([tb, u_std(df).T], axis=0)
tb.index=['0','1','all']
tb.index.name='class'
tb=round(tb,2)
print('\nQ1.2\n')
print(tb)
tb.to_csv('/Users/yubinye/Downloads/677/hw3/tb.csv')



'''
Question 2.1
'''
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


train, test=train_test_split(df, test_size=0.5,random_state=1)

'''
plot 
'''
X_train_0=train[train['class']==0][['f1','f2','f3','f4']]
X_train_1=train[train['class']==1][['f1','f2','f3','f4']]

pic=sns.pairplot(X_train_0, markers="s",
                    plot_kws=dict(s=10, edgecolor="g", alpha=0.5))
pic.fig.suptitle("class=0", y=1.05)
pic.savefig("good bills.pdf")


pic1=sns.pairplot(X_train_1, markers="o",
                 plot_kws=dict(s=10, edgecolor="r", linewidth=1))
pic1.fig.suptitle("class=1", y=1.05)
pic1.savefig("fake bills.pdf")

sns.pairplot(train, hue="class")



'''
Question 2.2, 2.3
'''
def my_predict(data):
    data['predict']=None
    for i in data.index:
        if (data['f1'][i]>0) or (data['f2'][i]>10) or (data['f3'][i]<5 and data['f4'][i]<-10 ):
            data['predict'][i]=0
        else: data['predict'][i]=1
    return data['predict']

test['predict']=my_predict(test)

'''
Question 2.4
'''

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

result_2_4=tpr_tnr(test['predict'].values,test['class'].values)
print('\nQ2.4\n')
print(result_2_4,'\n')
result_2_4.T.to_csv('/Users/yubinye/Downloads/677/hw3/tb2.csv')


'''
Question 3.1
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

X_train= train[['f1','f2','f3','f4']].values
X_test = test[['f1','f2','f3','f4']].values


scaler = StandardScaler()
scaler.fit(X_train)

X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

le = LabelEncoder()
Y_train = le.fit_transform(train[['class']].values)
Y_test = test['class'].values

print('\nQ3.1\n')
accu=[]
for k in [3,5,7,9,11]:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, Y_train)
    pred_k = knn_classifier.predict(X_test)
    accu.append(np.mean(pred_k == Y_test))
    print('when k=%s, accuracy is'%k,round(accuracy,4))

'''
plot 
'''
import matplotlib.pyplot as plt

ax = plt.gca()
plt.plot(range(3,12,2), accu, color='red', linestyle='dashed',
marker='o', markerfacecolor='black', markersize=10)
plt.title('accuracy vs. k')
plt.xlabel('number of neighbors : k')
plt.ylabel('accuracy')



'''
Question 3.3
'''


def best_k_pred(n):
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_train, Y_train)
    pred_k = knn_classifier.predict(X_test)
    result = tpr_tnr(pred_k,Y_test)
    return result

print('\nQ3.3\n')
print('\nif best k = 7\n',best_k_pred(7))
print('\nif best k = 9\n',best_k_pred(9))

best_k_pred(7).T.to_csv('/Users/yubinye/Downloads/677/hw3/best7.csv')
best_k_pred(9).T.to_csv('/Users/yubinye/Downloads/677/hw3/best9.csv')



'''
Question 3.5
'''
bill_x=[[1,1,4,5]]
x=[1,1,4,5]
bill=pd.DataFrame(x).T
bill.columns=['f1','f2','f3','f4']

print('\nQ3.5\n')
print('\nmy predicted label is ',my_predict(bill)[0])
print('\nknn predicted label is',knn_classifier.predict(bill_x)[0])



'''
Question 4.1
'''

def pred_by_knn(cols):
    X = df[cols].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)

    le = LabelEncoder()
    Y = le.fit_transform(df['class'].values)

    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.5,random_state=1)

    knn_classifier = KNeighborsClassifier(n_neighbors=7)
    knn_classifier.fit(X_train, Y_train)
    pred_k = knn_classifier.predict(X_test)
    accracy=round(np.mean(pred_k == Y_test),4)
    return accracy

print('\nQ4.1\n')
print('if missing f1, accuracy is',pred_by_knn(['f2','f3','f4']))
print('if missing f2, accuracy is',pred_by_knn(['f1','f3','f4']))
print('if missing f3, accuracy is',pred_by_knn(['f1','f2','f4']))
print('if missing f4, accuracy is',pred_by_knn(['f1','f2','f3']))

'''
Question 5.1
'''
from sklearn.linear_model import LogisticRegression

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)
pred_log = log_reg_classifier.predict(X_test)

print('\nQ5.1\n')
print(tpr_tnr(pred_log,Y_test))
tpr_tnr(pred_log,Y_test).T.to_csv('/Users/yubinye/Downloads/677/hw3/tb5.csv')

'''
Question 5.5
'''
print('\nQ5.5\n')
print('\nlogistic regression label is',log_reg_classifier.predict(bill_x)[0])


'''
Question 6.1
'''

def pred_by_log(cols):
    X = df[cols].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)

    le = LabelEncoder()
    Y = le.fit_transform(df['class'].values)

    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.5)

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train,Y_train)
    pred_lg = log_reg_classifier.predict(X_test)
    accracy=round(np.mean(pred_lg == Y_test),4)
    return accracy

print('\nQ6.1\n')
print('if missing f1, accuracy is',pred_by_log(['f2','f3','f4']))
print('if missing f2, accuracy is',pred_by_log(['f1','f3','f4']))
print('if missing f3, accuracy is',pred_by_log(['f1','f2','f4']))
print('if missing f4, accuracy is',pred_by_log(['f1','f2','f3']))