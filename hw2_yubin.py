# -*- coding: utf-8 -*-

import pandas as pd

'''
there is only one script for both SPY and my Stock Y, 

I could change file path and file name, and get different result.

'''

df=pd.read_csv('/Users/yubinye/Downloads/677/hw2/SPY.csv')


'''
Question 1.1
'''
df['TrueLabel']=df['Return'].apply(lambda x:'-' if x < 0 else '+')

test=df[df['Year']>2017]
train=df[df['Year']<=2017]


'''
Question 1.2
'''
# how many up days 
up=len(train['TrueLabel'][train['TrueLabel']== '+'])

# probability of up days
p1=up/len(train['TrueLabel'])
print('\nQuestion 1.2:\n')
print('the probability is ',p1)

'''
Question 1.3
'''

# transfer labels into a string in train sample
s_train=''.join(train['TrueLabel'])

# define a function to count specific piece of string in a long string
def num_of_patterns(astr,pattern):
    astr, pattern = astr.strip(), pattern.strip()
    if pattern == '': return 0

    ind, count, start_flag = 0,0,0
    while True:
        try:
            if start_flag == 0:
                ind = astr.index(pattern)
                start_flag = 1
            else:
                ind += 1 + astr[ind+1:].index(pattern)
            count += 1
        except:
            break
    return count

# the first part a string pattern
down_pat=['-','--','---']

def probability(pat):
    prob={}
    count=0
    for i in range(len(down_pat)):
        # combine first part with + we get: .+, ..+, ...+
        # combine first part with - we get: .-, ..-, ...-
        pat1=down_pat[i]+'+'
        pat2=down_pat[i]+'-'
        # compute how many short string in long string respectively
        up=num_of_patterns(s_train,pat1)
        down=num_of_patterns(s_train,pat2)
        p=up/(up+down)
        prob[pat1]=p

    count+=1
    return prob

print('\n the probability respectively is ',probability(down_pat),'\n')


'''
Question 1.4
'''
# the first part a string pattern
up_pat=['+','++','+++']
print('\n the probability respectively is ',probability(up_pat),'\n')


'''
Question 2.1
'''
# transfer labels into a string in test sample
s_test=''.join(test['TrueLabel'])

# compute probability for ..+, ...+,...+ in train sample
def compute(k):
    list_p=[]
    pat=[]
    count=0

    for j in range(len(test))[k:]:
        # find a pair of n-day labels
        # combine the pair of labels into a single string

        if k==2:
            ss=list(test['TrueLabel'])[j-2:j]
            l=ss[0]+ss[1]
            pat.append(l)

        elif k==3:
            ss=list(test['TrueLabel'])[j-3:j]
            l=ss[0]+ss[1]+ss[2]

        elif k==4:
            ss=list(test['TrueLabel'])[j-4:j]
            l=ss[0]+ss[1]+ss[2]+ss[3]

        # comupte probabilities respectively
        up=num_of_patterns(s_train,l+'+')
        down=num_of_patterns(s_train,l+'-')
        p=up/(up+down)
        list_p.append(p)
    count+=1
    return list_p

# create a new column for w=2,3,4
# compute the probability for each pair of labels when w=2,3,4
# if the probability that end label is + bigger than 0.5, it will be assigned +, otherwise -

test['w2p']=None
test['w2p'][2:]=compute(2)
test['w2']=None
test['w2'][2:]=test['w2p'][2:].apply(lambda x: '+' if x > 0.5 else '-')

test['w3p']=None
test['w3p'][3:]=compute(3)
test['w3']=None
test['w3'][3:]=test['w3p'][3:].apply(lambda x: '+' if x > 0.5 else '-')

test['w4p']=None
test['w4p'][4:]=compute(4)
test['w4']=None
test['w4'][4:]=test['w4p'][4:].apply(lambda x: '+' if x > 0.5 else '-')


test.head(10)

'''
Question 2.2
'''

def accu(col,k):
    all_p=0
    posi=0
    neg=0
    for i in test.index[k:]:
        if (col[i]=='+') & (test['TrueLabel'][i] == '+') :
            posi+=1
            
        if (col[i]=='-') & (test['TrueLabel'][i] == '-'):
            neg+=1

        if col[i]==test['TrueLabel'][i]:
            all_p+=1

    p1=all_p/(len(col)-k)
    p2=posi/(len(col[col=="+"]))
    p3=neg/(len(col[col=="-"]))

    return {'all':p1,'+':p2,"-":p3}

print('\nQuestion 2.2:','\n',
'\naccuracy for w=2 is',accu(test['w2'],2),'\n',
'\naccuracy for w=3 is',accu(test['w3'],3),'\n',
'\naccuracy for w=4 is',accu(test['w4'],4),'\n')


'''
Question 3.1
'''
test['p']=(test['w2p']+test['w3p']+test['w4p'])/3
test['ensb']=None
test['ensb'][4:]=test['p'][4:].apply(lambda x: '+' if x > 0.5 else '-')

'''
Question 3.2
'''

print('\nQuestion 3.2: \n',
'accuracy for ensemble is',accu(test['ensb'],4),'\n')



'''
Question 4
'''
def tpr_tnr(col):
    TP,FP,TN,FN=0,0,0,0
    for i in test.index:
        true_lb=test['TrueLabel'][i]
        my_lb=col[i]
        if (true_lb == '+') & (my_lb == '+'):
                TP+=1
        elif (true_lb == '-') & (my_lb == '+'):
                FP+=1
        elif (true_lb == '-') & (my_lb == '-'):
                TN+=1
        elif (true_lb == '+') & (my_lb == '-'):
                FN+=1
        else:
            continue
    TPR=TP/(TP+FN)
    TNR = TN/(TN+FP)
    return {'TP':TP,'FP':FP,'TN':TN,'FN':FN,'TPR':TPR,'TNR':TNR}


print('\nQuestion 4:','\n',
'\nw=2',tpr_tnr(test['w2']),'\n',
'\nw=3',tpr_tnr(test['w3']),'\n',
'\nw=4',tpr_tnr(test['w4']),'\n',
'\nw=ensemble',tpr_tnr(test['ensb']),'\n')


'''
Question 5
'''
def gain(col):
    money=[]
    gain=100
    count=0
    for i in test.index:
        R = test['Return'][i]

        if col[i] == '+':
            gain*=(1+R)
            money.append(gain)
        else:
            gain=gain
            money.append(gain)
        
    count+=1

    return money


test['$w']=None
test['$w']=gain(test['w2'])

test['$ensb']=None
test['$ensb']=gain(test['ensb'])


def buy_and_hold(col):
    money=[]
    gain=100
    count=0
    for i in test.index:
        R = test['Return'][i]
        gain*=(1+R)
        money.append(gain)
        
    count+=1
    return money

test['$buy_hold']=None
test['$buy_hold']=buy_and_hold(test['TrueLabel'])

test.tail()



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.figure(figsize=(15,5))
x=test['Year_Week']
best_w= plt.plot(x,test['$w'],color='darkkhaki', linewidth=1, alpha=1,
                 marker='o',markerfacecolor='darkkhaki',markersize=1)
ensemble = plt.plot(x,test['$ensb'],color='lightsteelblue', linewidth=1,
                  alpha=1,marker='o',markerfacecolor='lightsteelblue',markersize=1)
buy_hold = plt.plot(x,test['$buy_hold'],color='pink', linewidth=.7,alpha=1)

plt.ylabel('Year_Week',fontsize = 14)

# change name for sticker SPY or my stock Y
plt.title('growth of SPY amount',fontsize = 18)
plt.xticks(test['Year_Week'][::50])

plt.legend(labels = ['best W','ensemble' ,'buy and hold'], loc = 'best')
plt.grid(alpha= 0.3)

