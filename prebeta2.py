# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:11:43 2018

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

data_kospi = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI.xlsx')

ri_kospi = pd.read_excel(data_kospi, 'KOSPI_Ri')
r_kospi = pd.read_excel(data_kospi, 'KOSPI_R')

ri_kospi = ri_kospi.set_index(list(ri_kospi.columns[[0]])) / 100  # set the first column as index
r_kospi = r_kospi.set_index(list(r_kospi.columns[[0]])) /100


# Calculate std of individual stock and index (rolling window = 36 mon)
ri_std = ri_kospi[36:].copy().T
for i in range(len(ri_std.columns)):
    ri_std.iloc[:,i] = list(ri_kospi.iloc[i:i+36,:].std())
ri_std = ri_std.T

idx_std = r_kospi[36:].copy().T
for i in range(len(idx_std.columns)):
    idx_std.iloc[:,i] = list(r_kospi.iloc[i:i+36,:].std())
idx_std = idx_std.T

# Calculate correlation btwn indv stock and market index (rolling window = 36 mon)
ri_corr = ri_kospi[36:].copy()
for i in range(len(ri_corr)):
    rho = []
    for j in range(len(ri_corr.columns)):
        p = ri_kospi.iloc[i:i+36,j].corr(r_kospi.iloc[i:i+36,0])
        rho.append(p)
    ri_corr.iloc[i:i+1,:] = rho

mul = ri_corr * ri_std
bet = mul.copy()
for i in range(len(bet.columns)):
    bet.iloc[:,i] = idx_std.iloc[:,0]

beta = mul / bet
betas = beta*0.6 + (1-0.6)*1

writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\prebeta2.xlsx')
betas.to_excel(writer,'beta_shrink')
writer.save()
