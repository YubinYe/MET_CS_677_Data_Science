# -*- coding: utf-8 -*-

import pandas as pd

df=pd.read_csv('/Users/yubinye/Downloads/677/hw4/heart_failure_clinical_records_dataset.csv')

df.head(20)

'''
Question 1.1
'''
df_0=df[df['DEATH_EVENT']==0]
df_1=df[df['DEATH_EVENT']==1]


'''
Question 1.2
'''
import seaborn as sns
import matplotlib.pyplot as plt

print('\nheatmap for df_0')
sns.heatmap(round(df_0.corr(),2),center=0,annot=True,alpha=0.9,annot_kws={"size":8},cmap='coolwarm')
sns.set_context("paper", font_scale=0.8)     
plt.show(sns)

print('\nheatmap for df_1')
sns.heatmap(round(df_1.corr(),2),center=0,annot=True,alpha=0.9,annot_kws={"size":8},cmap='coolwarm')
sns.set_context("paper", font_scale=0.8) 



