# -*- coding: utf-8 -*-

import pandas as pd

df=pd.read_csv('/Users/yubinye/Downloads/677/hw4/heart_failure_clinical_records_dataset.csv')
df_0=df[df['DEATH_EVENT']==0]
df_1=df[df['DEATH_EVENT']==1]


'''
Question 2
'''

import numpy  as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



def plot_y_x(degree):
    weights = np.polyfit(X_train,Y_train, degree)
    model = np.poly1d(weights)
    predicted = model(X_test)

    plt.scatter(X_test, Y_test, color ='pink')
    plt.plot(X_test, predicted, color='skyblue')
    plt.title('degree=%s'%degree)
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    plt.legend(labels = ['predicted','actual'], loc = 'best')
    plt.grid(alpha= 0.3)
    plt.show()

    rss=np.sum((predicted - Y_test)**2)
    print('\nweights',np.round(weights,2))
    print('\nsse',np.round(rss,2))


    return 


def plot_y_lnx(degree):
    weights = np.polyfit(np.log(X_train),Y_train, degree)
    model = np.poly1d(weights)
    predicted = model(np.log(X_test))

    # d)
    plt.scatter(np.log(X_test), Y_test, color ='pink')
    plt.plot(np.log(X_test), predicted, color='green')
    plt.title('Y = a log(X) + b')
    plt.xlabel('log(X)')
    plt.ylabel('Y', rotation=0)
    plt.legend(labels = ['predicted','actual'], loc = 'best')
    plt.grid(alpha= 0.3)
    plt.show()

    rss=np.sum((predicted - Y_test)**2)
    print('\nweights',np.round(weights,2))
    print('\nsse',np.round(rss,2))


    return 


def plot_lny_lnx(degree):

    weights = np.polyfit(np.log(X_train),np.log(Y_train), degree)
    model = np.poly1d(weights)
    predicted = model(np.log(X_test))

    # d)
    plt.scatter(np.log(X_test), np.log(Y_test), color ='pink')
    plt.plot(np.log(X_test), predicted, color='blue')
    plt.title('log(Y) = a log(X) + b')
    plt.xlabel('log(X)')
    plt.ylabel('log(Y)', rotation=0)
    plt.legend(labels = ['predicted','actual'], loc = 'best')
    plt.grid(alpha= 0.3)
    plt.show()

    rss=np.sum((predicted - np.log(Y_test))**2)
    print('\nweights',np.round(weights,4))
    print('\nsse',np.round(rss,4))

    return 







# death event=0

dt=df_0

X=dt['creatinine_phosphokinase']
Y=dt['platelets']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=15)

X_test, Y_test= zip(*sorted(zip(X_test, Y_test)))


if __name__ == '__main__':
    print('\n1. Y = a X + b\n')
    plot_y_x(1)
    print('\n2. Y = a X**2 + b X + c\n')
    plot_y_x(2)
    print('\n3. Y = a X**3 + b X**2 + c X + d\n')
    plot_y_x(3)
    print('\n4. Y = a log(X) + b\n')
    plot_y_lnx(1)
    print('\n5. log(Y) = a log(X) + b\n')
    plot_lny_lnx(1)




# death event=1

dt=df_1

X=dt['creatinine_phosphokinase']
Y=dt['platelets']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=15)

X_test, Y_test= zip(*sorted(zip(X_test, Y_test)))


if __name__ == '__main__':
    print('\n1. Y = a X + b\n')
    plot_y_x(1)
    print('\n2. Y = a X**2 + b X + c\n')
    plot_y_x(2)
    print('\n3. Y = a X**3 + b X**2 + c X + d\n')
    plot_y_x(3)
    print('\n4. Y = a log(X) + b\n')
    plot_y_lnx(1)
    print('\n5. log(Y) = a log(X) + b\n')
    plot_lny_lnx(1)

# question 3

import numpy  as np
from sklearn.model_selection import train_test_split

'''
Question3
'''

def compute_y_x(degree):
    weights = np.polyfit(X_train,Y_train, degree)
    model = np.poly1d(weights)
    predicted = model(X_test)
    rss=np.sum((predicted - Y_test)**2)
    return {'weights':weights,'sse':rss}

def compute_y_lnx(num):
    weights = np.polyfit(np.log(X_train),Y_train, 1)
    model = np.poly1d(weights)
    predicted = model(np.log(X_test))
    rss=np.sum((predicted - Y_test)**2)
    return {'weights':weights,'sse':rss}

def compute_lny_lnx(num):
    weights = np.polyfit(np.log(X_train),np.log(Y_train), 1)
    model = np.poly1d(weights)
    predicted = model(np.log(X_test))
    rss=np.sum((predicted- np.log(Y_test))**2)
    return {'weights':weights,'sse':rss}

def sse(num):
    sse1=compute_y_x(1)['sse']
    sse2=compute_y_x(2)['sse']
    sse3=compute_y_x(3)['sse']
    sse4=compute_y_lnx(1)['sse']
    sse5=compute_lny_lnx(1)['sse']

    sse_list = [sse1,sse2,sse3,sse4,sse5]
    sse=pd.DataFrame(sse_list)
    sse.index=['Y = a X + b',
                'Y = a X**2 + b X + c',
                'Y = a X**3 + b X**2 + c X + d',
                'Y = a log(X) + b',
                'log(Y) = a log(X) + b']
    return sse

dt=df_0
X=dt['creatinine_phosphokinase']
Y=dt['platelets']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=15)
sse0=sse(0)


dt=df_1
X=dt['creatinine_phosphokinase']
Y=dt['platelets']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=15)
sse1=sse(1)


sse=pd.concat([sse0,sse1], axis=1)
sse.columns=['death event=0','death event=1']
print(sse)



sse.to_csv('sse.csv')