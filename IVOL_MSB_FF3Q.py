# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:13:15 2018

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

dr = pd.read_excel(data_beta, 'KOSPI_Ri_D')[730:]
drf = pd.read_excel(data_beta, 'MSB_D_adj')[730:]
dmktr = pd.read_excel(data_beta, 'KOSPI_D')[730:]

dr = dr.set_index(list(dr.columns[[0]])) / 100  # set the first column as index
drf = drf.set_index(list(drf.columns[[0]])) / 100
dmktr = dmktr.set_index(list(dmktr.columns[[0]])) /100

'''Benchmark adjusted return'''
# Calculate the market excess returns
mkexr = dmktr.copy()
mkexr.iloc[:,0] = dmktr.iloc[:,0] - drf.iloc[:,0]

# Calculate individual stocks' excess returns
exr = dr.copy()
for i in range(len(exr.columns)):
    exr.iloc[:,i] = dr.iloc[:,i] - drf.iloc[:,0]

data_mktc = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\MKTCAP_D.xlsx')
dmktrcap = pd.read_excel(data_mktc, 'MKTCAP_D')[730:]
dmktrcap = dmktrcap.set_index(list(dmktrcap.columns[[0]]))

#Size sorting (each month), mktcapc = mktcap[common]
dmktcapc_size = dmktrcap.dropna(axis=1).copy() #sorting stocks each month by marketcap(Size)
#mktcapc_size = mktcapc_size[common]
for i in range(len(dmktcapc_size)): #t = 180
    rank = dmktcapc_size[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    dmktcapc_size[i:i+1] = [rank2.iloc[:,0].values]
dmktcapc_size.columns = list(range(len((dmktcapc_size.columns)))) #small(위) to big(아래)

#B/M = Total Equity / mktcap
data_accnt = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\BV.xlsx')
TE = pd.read_excel(data_accnt, 'BV_avg')[8:]
TE = TE.set_index(list(TE.columns[[0]]))
#TEc = TE

for i in range(len(TE)):
    for j in range(len(TE.columns)):
        if TE.iloc[i+1:i+2,j].isnull().values == True:
            TE.iloc[i+1:i+2,j] = TE.iloc[i:i+1,j]

mktcap_kospic = dmktrcap.dropna(axis=1).copy()
com = mktcap_kospic.columns.tolist()
com.remove('세아제강지주')
TE = TE[com]
BV = mktcap_kospic.copy()
del BV['세아제강지주']
for i in range(int(len(BV)/91)): # i= 16 years
    for m in range(91):
        n = i * 91
        BV[n+m:n+m+1] = TE[i:i+1].values
        
BM = BV.copy() 
del mktcap_kospic['세아제강지주']
for i in range(len(BM)):
    BM[i:i+1] = BV[i:i+1] / mktcap_kospic[i:i+1]

    
#sorting stocks each month by BM
BM_sort = BM.copy()
BM_sort = BM_sort.dropna(axis=1)
for i in range(len(BM_sort)):
    rank = BM_sort[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    BM_sort[i:i+1] = [rank2.iloc[:,0].values]
BM_sort.columns = list(range(len((BM_sort.columns)))) #Low to High B/M

# mktcapc_size -> Big Small, BM_sort -> H M L, generating 6 ptfs
half = round(len(dmktcapc_size.columns) / 2)
thr_pct = round(len(BM_sort.columns) * 0.3)
for_pct = round(len(BM_sort.columns) * 0.7)
PTF_FF3 = []
for t in range(len(dmktcapc_size)):    
    small = [item for sublist in dmktcapc_size.iloc[t:t+1,0:half].values.tolist() for item in sublist]
    big = [item for sublist in dmktcapc_size.iloc[t:t+1,half:].values.tolist() for item in sublist]
    
    low = [item for sublist in BM_sort.iloc[t:t+1,0:thr_pct].values.tolist() for item in sublist]
    middle = [item for sublist in BM_sort.iloc[t:t+1,thr_pct:for_pct].values.tolist() for item in sublist]
    high = [item for sublist in BM_sort.iloc[t:t+1,for_pct:].values.tolist() for item in sublist]
    
    size_list = [small, big]
    bm_list =[low, middle, high]
    ptf = [list(set(j).intersection(i)) for i in bm_list for j in size_list]
    PTF_FF3.append(ptf)

# Returns of 6 ptfs
rs_ptff3 = mktcap_kospic.iloc[:,:6].copy().T #PTF-FF3 returns (t x 6)
for i in range(len(PTF_FF3)): # t = 180
    rs_ff3 = []
    for j in range(len(PTF_FF3[i])):  #6 ptfs
        r_ptff3 = dr[PTF_FF3[i][j]][i:i+1]
#        mktcap_ptf = mktcap_kospic[PTF_FF3[i][j]][i:i+1]
#        tmktcap = mktcap_ptf.sum().sum()
#        r = (r_ptff3 * mktcap_ptf / tmktcap).sum().sum()
        r = r_ptff3.sum().sum() / len(PTF_FF3[i][j])
        rs_ff3.append(r)
    rs_ptff3.iloc[:, i] = rs_ff3
rs_ptff3 = rs_ptff3.T
rs_ptff3.columns = ['S/L', 'B/L', 'S/M', 'B/M', 'S/H', 'B/H']


#Calculation of SMB and HML (t X 2)
SMB = rs_ptff3.iloc[:,:1].copy()
SMB.columns = ['SMB']
for i in range(len(rs_ptff3)):
    SMB[i:i+1] = ((rs_ptff3.iloc[i:i+1]['S/L'] + rs_ptff3.iloc[i:i+1]['S/M'] + rs_ptff3.iloc[i:i+1]['S/H']) / 3) - ((rs_ptff3.iloc[i:i+1]['B/L'] + rs_ptff3.iloc[i:i+1]['B/M'] + rs_ptff3.iloc[i:i+1]['B/H']) / 3)
    
HML = rs_ptff3.iloc[:,:1].copy()
HML.columns = ['HML']
for i in range(len(rs_ptff3)):
    HML[i:i+1] = ((rs_ptff3.iloc[i:i+1]['S/H'] + rs_ptff3.iloc[i:i+1]['B/H']) / 2) - ((rs_ptff3.iloc[i:i+1]['S/L'] + rs_ptff3.iloc[i:i+1]['B/L']) / 2)

#Regression for estimating benchmark adjusted returns
bmr = exr.copy()
residuals = []
for i in range (len(exr.columns)):
    y = pd.DataFrame(exr.iloc[:,i].values)
    x = pd.DataFrame([mkexr.iloc[:,0].values, SMB.iloc[:,0].values, HML.iloc[:,0].values], index = ['MKT', 'SMB', 'HML']).T
    x = sm.add_constant(x)
    est = sm.OLS(y,x).fit()
    #al, pb = est.params
    #ttest = est.tvalues
    resi = est.resid
    #est.pvalues
    #alpha.append(al)
    #pbeta.append(pb)
    #tvalues.append(ttest)
    residuals.append(resi)
    #est.summary()
    bmr.iloc[:,i] = resi.values
    
writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\IVOL_MSB_FF3Q.xlsx')
bmr.to_excel(writer, 'benchmark_adj_r')
writer.save()

