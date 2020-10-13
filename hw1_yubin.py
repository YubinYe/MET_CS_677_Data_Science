# -*- coding: utf-8 -*-
import os

ticker='SPY'
input_dir = r'/Users/yubinye/Downloads/677/hw1'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
        SPY = [i.split(',') for i in lines][1:]
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

ticker='YY'
input_dir = r'/Users/yubinye/Downloads/677/hw1'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
        YY = [i.split(',') for i in lines][1:]
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

'''
Question 1.1
'''


# seperate by weekdays

def subset_day(data):
    mon = []
    tue = []
    wed = []
    thu= []
    fri = []
    for i in data:
        if i[4] == 'Monday':
            mon.append(i)
        elif i[4] == 'Tuesday':
            tue.append(i)
        elif i[4] == 'Wednesday':
            wed.append(i)
        elif i[4] == 'Thursday':
            thu.append(i)
        elif i[4] == 'Friday':
            fri.append(i)
    day_list = {'Monday' : mon, 'Tuesday':tue,
    'Wednesday':wed,'Thursday':thu,'Friday':fri}

    return day_list

#test print(subset_day(YY))

# seperate by sign

def subset_sign(data):
    R_all=[]
    R_neg=[]
    R_posi=[]
    
    count = 0

    for i in range(len(data)):
        R = float(data[i][-3])
        R_all.append(R)
        if R<0:
            R_neg.append(R)
        else :
            R_posi.append(R)

    count+=1
    
    R_list = {'' : R_all, 
            '-' : R_neg,
            '+' : R_posi}
    
    return R_list

#test print(subset_day(YY))



def compute(data,day,sign):


    R = subset_sign(subset_day(data)[day])[sign]
            
    n = len(R)
    sum_r=sum(R)
            
    # mean
    ave = sum_r/n
            
    # standard deviation
    s=0
    for j in R:
        s += (j**2)
    std= (s/n - ave**2)**0.5  
    
    frame = {'u(%sR)'%sign : ave, 
            'std(%sR)'%sign : std,
            '|%sR|'%sign :n }
        
    return frame


#test print(compute(YY,'Monday','-'))

'''
Question 1.2
'''


# Here I use pandas only for save table
import pandas as pd

days=['Monday', 'Tuesday','Wednesday','Thursday','Friday']
def table_days(data):

    all_r=[]
    neg_r=[]
    pos_r=[]
    count=0
    for day in days:
        all_r.append(compute(data,day,''))
        neg_r.append(compute(data,day,'-'))
        pos_r.append(compute(data,day,'+'))

    count+=1

    all_r = pd.DataFrame(all_r,index=days)
    neg_r = pd.DataFrame(neg_r,index=days)
    pos_r= pd.DataFrame(pos_r,index=days)

    table = pd.concat([all_r, neg_r], axis=1)
    table = pd.concat([table, pos_r], axis=1)   
    return table

print(table_days(YY))
print(table_days(SPY))
table_days(YY).to_csv('/Users/yubinye/Downloads/677/hw1/tbYY.csv')
table_days(SPY).to_csv('/Users/yubinye/Downloads/677/hw1/tbSPY.csv')



def subset_year(data,year):
    YY_year=[]
    count=0
    for i in range(len(data)):
        if data[i][1] == year:
            YY_year.append(data[i])
    count+=1
    return YY_year

#test print(subset_year(YY,'2015'))

def show_table(data,year):
    table=table_days(subset_year(data,year))
    table.index.name='%s'%year
    return table


show_table(YY,'2015').to_csv('/Users/yubinye/Downloads/677/hw1/tb2015.csv')
show_table(YY,'2016').to_csv('/Users/yubinye/Downloads/677/hw1/tb2016.csv')
show_table(YY,'2017').to_csv('/Users/yubinye/Downloads/677/hw1/tb2017.csv')
show_table(YY,'2018').to_csv('/Users/yubinye/Downloads/677/hw1/tb2018.csv')
show_table(YY,'2019').to_csv('/Users/yubinye/Downloads/677/hw1/tb2019.csv')



'''
Question 1.3
'''

def count_days(data):   
    neg_days=0
    non_neg_days=0

    for i in range(len(data)):
        R = float(data[i][-3])
        if R < 0 :
            neg_days+=1
        else:
            non_neg_days+=1
            
    count_days={'negative days':neg_days,
               'non_negative days':non_neg_days}
    return count_days

print('\n2015',count_days(subset_year(YY,'2015')),
      '\n2016',count_days(subset_year(YY,'2016')),
      '\n2017',count_days(subset_year(YY,'2017')),
      '\n2018',count_days(subset_year(YY,'2018')),
      '\n2019',count_days(subset_year(YY,'2019')))

print('\nall years',count_days(YY),'\n')


'''
Question 1.4
'''
def lose_gain(data):
    answer={}
    count=0
    years=['2015','2016','2017','2018','2019']

    for year in years:
        R = subset_sign(subset_year(data,year))['']
                
        n = len(R)
        sum_r=sum(R)
                
        # mean
        ave = sum_r/n

        if ave < 0:
            answer[year]='lose more'
        else:
            answer[year]='gain more'

    count+=1

    return answer

print('all years\n',lose_gain(YY),'\n')


'''
Question 1.5
'''
def lose_gain_year(data,year):
    answer={}
    count=0
    for day in days:
        u=compute(subset_year(data,year),day,'')['u(R)']
        if u < 0:
            answer[day]='lose more'
        else:
            answer[day]='gain more'

    count+=1

    return [year,answer]

print(lose_gain_year(YY,'2015'),'\n')
print(lose_gain_year(YY,'2016'),'\n')
print(lose_gain_year(YY,'2017'),'\n')
print(lose_gain_year(YY,'2018'),'\n')
print(lose_gain_year(YY,'2019'),'\n')



'''
Question 3
'''

print('\nYY\n',table_days(YY))
print('\nSPY\n',table_days(SPY))

'''
Question 4
'''

def count_gain(data):

    money=100

    for i in range(len(data)):
        R = float(data[i][-3])
        if R > 0:
            money*=(1+R)

    return money

print('my stock YY has $',count_gain(YY),'\n')
print('stock SPY has $',count_gain(SPY),'\n')


'''
Question 5
'''

def buy_and_hold(data):
    start_price = float(data[0][-4])
    share = 100 / start_price
    final_price = float(data[-1][-4])
    money = final_price * share
    return money

print('my stock YY has $',buy_and_hold(YY),'\n')
print('stock SPY has $',buy_and_hold(SPY),'\n')


'''
Question 6
'''

def sort_return(data):
    R_list=[]
    
    count = 0

    for i in range(len(data)):
        R = float(data[i][-3])
        R_list.append(R)
    count+=1
    R_list=sorted(R_list)

    return R_list

def wrong_oracle(data,left,right):
    
    R_all =sort_return(data)
    rest=count_gain(data)
    

    for R1 in R_all[:left]:
        rest*=(1+R1)

    for R2 in R_all[right:]:
        rest/=(1+R2)

    return rest

print('YY missed best 10 days',wrong_oracle(YY,0,-10))
print('YY missed worst 10 days',wrong_oracle(YY,10,len(YY)))
print('YY missed best and worst 5 days',wrong_oracle(YY,5,-5))


print('SPY missed best 10 days',wrong_oracle(SPY,0,-10))
print('SPY missed worst 10 days',wrong_oracle(SPY,10,len(YY)))
print('SPY missed best and worst 5 days',wrong_oracle(SPY,5,-5))