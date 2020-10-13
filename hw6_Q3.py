# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix

df=pd.read_csv('df.csv')
df.shape

X = df.iloc[:,0:-1].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = df.iloc[:,-1].values.ravel()

'''
Question 3.1
'''
list1 = []
for k in range(1,9):
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(X)
    inertia = kmeans_classifier.inertia_
    list1.append(inertia)


plt.figure(figsize=(8,4))
plt.plot(range(1,9), list1, marker = 'o',color='slateblue')
plt.xticks(range(9))
plt.xlabel('K') 
plt.ylabel('Inertia Values') 
plt.title('The Elbow Method using Inertia')

plt.grid(True)
plt.show()


'''
Question 3.2
'''

kmeans_classifier = KMeans(n_clusters=3)  
y_kmeans = kmeans_classifier.fit_predict(X)
inertia = kmeans_classifier.inertia_
centroids = kmeans_classifier.cluster_centers_
centroids

feature_length = len(df.columns) - 1
feature_list = np.random.choice(feature_length, 2, replace=False)


plt.figure(figsize=(10,6) )
for i in range(3):
    scatter_color = ['gold','pink','y']
    plt.scatter(X[y_kmeans == i, feature_list[0]] , X[y_kmeans == i, feature_list[1]],
     s = 20, c = scatter_color[i], label = 'cluster %s' %(i+1))

    centroid_color = ['red','b','green']
    plt.scatter(centroids[:, feature_list[0]][i], centroids[:, feature_list[1]][i], s = 100, 
    c = centroid_color[i], label ='Centroid %s'%(i+1),marker='+')
    
x_label = df.columns[feature_list[0]]
y_label = df.columns[feature_list[1]]
plt.xlabel(x_label)
plt.ylabel(y_label)

plt.legend(loc='best')
plt.grid(True)
plt.show()


'''
Question 3.3
'''

entriod = {}
for i in range(1,4):
    label_list = Y[y_kmeans == (i-1)]
    final_label = int(statistics.mode(label_list))
    centroid = centroids[i-1]
    centroid_name = 'centroid %s'%i
    entriod.update({centroid_name:final_label})
    print('\ncluster %s'%i,
    '\nlabel is :',final_label,
    '\n',centroid)



'''
Question 3.4
'''
labels = []
for i in range(len(X)):
    d1 = distance.euclidean(X[i],centroids[0])
    d2 = distance.euclidean(X[i],centroids[1])
    d3 = distance.euclidean(X[i],centroids[2])

    if min(d1, d2, d3) == d1:
            labels.append(entriod['centroid 1'])
    elif min(d1, d2, d3) == d2:
            labels.append(entriod['centroid 2'])
    else:
        labels.append(entriod['centroid 3'])     

accuracy= np.mean(np.array(labels) == Y)
print('accuracy for k mean is', round(accuracy,4))


'''
Question 3.5
'''

df_r2 = pd.read_csv('df_R2.csv')

X_r2 = df_r2.iloc[:,0:-1].values
scaler = StandardScaler()
scaler.fit(X_r2)
X = scaler.transform(X_r2)
Y = df_r2.iloc[:,-1].values.ravel()



kmeans_classifier = KMeans(n_clusters=2)  
y_kmeans = kmeans_classifier.fit_predict(X)
inertia = kmeans_classifier.inertia_
centroids = kmeans_classifier.cluster_centers_
centroids


entriod = {}
for i in range(1,3):
    label_list = Y[y_kmeans == (i-1)]
    final_label = int(statistics.mode(label_list))
    centroid = centroids[i-1]
    centroid_name = 'centroid %s'%i
    entriod.update({centroid_name:final_label})
    print('\ncluster %s'%i,
    '\nlabel is :',final_label,
    '\n',centroid)



labels= []
for i in range(len(X)):
    d1 = distance.euclidean(X[i],centroids[0])
    d2 = distance.euclidean(X[i],centroids[1])

    if min(d1, d2) == d1:
            labels.append(entriod['centroid 1'])
    elif min(d1, d2) == d2:
            labels.append(entriod['centroid 2'])


accuracy= np.mean(np.array(labels) == Y)
print('accuracy is', round(accuracy,4))

confusion_matrix = confusion_matrix(Y, labels) 
print('confusion_matrix  is', confusion_matrix)


TN, FP, FN, TP = confusion_matrix.ravel()
TPR = TP/(TP + FN)
TNR = TN/(TN + FP)
dic=[TP,FP,TN,FN,round(accuracy,6),round(TPR,6),round(TNR,6)]
result=pd.DataFrame(dic)
result.index=['TP','FP','TN','FN','accuracy','TPR','TNR']

q35=result.T

q2=pd.read_csv('Q2.csv')
q2.drop(['Unnamed: 0'],axis=1,inplace=True)


q_all=pd.concat([q2,q35],axis=0)


q_all.index.name='index'
q_all.index=['linear kernel SVM','Gaussian kernel SVM',
'polynomial kernel SVM','knn','k mean']

print(q_all)

q_all.to_csv('Q3.csv')
