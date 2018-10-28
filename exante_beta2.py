# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:38:05 2018

@author: 장연숙
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

#Inv.iloc[0:1] select row
#Inv[0:1] select row
#Inv.loc['2002Q2'] select row
#Inv['삼성전자'] select column
#Inv.iloc[:,0] or Inv.iloc[:, 0:3] select column  

data_beta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI_D.xlsx')

MR = pd.read_excel(data_beta, 'KOSPI_Ri_D')
RF = pd.read_excel(data_beta, 'CD91_adj')
MKTR = pd.read_excel(data_beta, 'KOSPI_D')

MR = MR.set_index(list(MR.columns[[0]])) / 100  # set the first column as index
RF = RF.set_index(list(RF.columns[[0]])) / 100
MKTR = MKTR.set_index(list(MKTR.columns[[0]])) /100


# Calculate the market excess returns
mkexr = MKTR.copy()
mkexr.iloc[:,0] = MKTR.iloc[:,0] - RF.iloc[:,0]
mkexr_lag = mkexr.shift().fillna(0)

# Calculate individual stocks' excess returns
exr = MR.copy()
for i in range(len(exr.columns)):
    exr.iloc[:,i] = MR.iloc[:,i] - RF.iloc[:,0]

#exr = exr.drop(['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한국전자금융', '한화투자증권', '이베스트투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '에이티넘인베스트', 'SV인베스트먼트', '푸른저축은행', '제주은행', '한양증권', 'DSC인베스트먼트', '대성창투', '티에스인베스트먼트', '제미니투자'], axis=1)


#Regression for estimating pre-ranking beta
b0mtrx = exr.copy()
b1mtrx = exr.copy()
dates= pd.date_range('2000-01-01','2017-12-31' , freq='1M')-pd.offsets.MonthBegin(1)
dates= dates.tolist()
b0mtrx = b0mtrx.loc[dates][36:]
b1mtrx = b1mtrx.loc[dates][36:]

for i in range (len(exr.columns)): #i=stock
    alpha = []
    beta0 = []
    beta1 = []
    for j in range(len(b0mtrx)): #j = daily
        t = j * 30
        y = exr.iloc[t:t+1095,i]
        x = pd.DataFrame([mkexr.iloc[t:t+1095,0], mkexr_lag.iloc[t:t+1095,0]], index=['mkt', 'mkt-1']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b0, b1 = est.params
        alpha.append(a)
        beta0.append(b0)
        beta1.append(b1)
        #est.summary()
    b0mtrx.iloc[:,i] = beta0
    b1mtrx.iloc[:,i] = beta1

bmtrx = b0mtrx + b1mtrx

#writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\exante_beta2.xlsx')
##b0mtrx.to_excel(writer, 'beta0')
##b1mtrx.to_excel(writer, 'beta1')
#bmtrx.to_excel(writer,'beta')
#writer.save()
