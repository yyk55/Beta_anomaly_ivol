# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:48:58 2018

@author: 장연숙
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime

#Inv.iloc[0:1] select row
#Inv[0:1] select row
#Inv.loc['2002Q2'] select row
#Inv['삼성전자'] select column
#Inv.iloc[:,0] or Inv.iloc[:, 0:3] select column  

data_kospi = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI_D.xlsx')

ri_kospi = pd.read_excel(data_kospi, 'KOSPI_Ri_D')
r_kospi = pd.read_excel(data_kospi, 'KOSPI_D')
r_kospim = pd.read_excel(data_kospi, 'KOSPI_R_M')

ri_kospi = ri_kospi.set_index(list(ri_kospi.columns[[0]])) / 100  # set the first column as index
r_kospi = r_kospi.set_index(list(r_kospi.columns[[0]])) /100
r_kospim = r_kospim.set_index(list(r_kospim.columns[[0]])) /100

# Calculate std of individual stock and index (rolling window = 365 days)
ri_std = ri_kospi[365:].copy().T
for i in range(len(ri_std.columns)):
    ri_std.iloc[:,i] = list(ri_kospi.iloc[i:i+365,:].std())
ri_std = ri_std.T
ri_std = ri_std[730:]

idx_std = r_kospi[365:].copy().T
for i in range(len(idx_std.columns)):
    idx_std.iloc[:,i] = list(r_kospi.iloc[i:i+365,:].std())
idx_std = idx_std.T
idx_std = idx_std[730:]

# Calculate correlation btwn indv stock and market index (rolling window = 3 years)
ri_corr = ri_kospi[1095:].copy()
for i in range(len(ri_corr)):
    rho = []
    for j in range(len(ri_corr.columns)):
        p = ri_kospi.iloc[i:i+1095,j].corr(r_kospi.iloc[i:i+1095,0])
        rho.append(p)
    ri_corr.iloc[i:i+1,:] = rho

mul = ri_corr * ri_std
bet = mul.copy()
for i in range(len(bet.columns)):
    bet.iloc[:,i] = idx_std.iloc[:,0]

beta = mul / bet
betas = beta*0.6 + (1-0.6)*1

writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\exante_beta.xlsx')
betas.to_excel(writer,'beta_shrink_d')
writer.save()


'''Stock sorting by beta 5 ptfs'''
data_prebeta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\exante_beta.xlsx')
prebeta = pd.read_excel(data_prebeta, 'beta_shrink_d') #from 2003 to 2017
#misp_msr = pd.read_excel(data_mispmsr, 'misp_msr')
prebeta = prebeta.set_index(list(prebeta.columns[[0]]))
#misp_msr = misp_msr.set_index(list(misp_msr.columns[[0]]))
#prebeta = prebeta.dropna(axis=1)

#common = list(misp_msr.columns.intersection(prebeta.columns).values)
#misp_msrc = misp_msr[common]
#misp_msrn = misp_msrc.copy() #sorting stocks each month by mispricing measure
#for i in range(len(misp_msrc)):
#    rank = misp_msrc[i:i+1].T
#    rank1 = rank.sort_values(str(rank.columns.values[0]))
#    rank2 = rank1.copy()
#    rank2.iloc[:,0] = rank2.index
#    misp_msrn[i:i+1] = [rank2.iloc[:,0].values]
#misp_msrn.columns = list(range(len((misp_msrn.columns)))) #0:underpricied
dates= pd.date_range('2000-01-01','2017-12-31' , freq='1M')-pd.offsets.MonthBegin(1)
dates= dates.tolist()
prebeta = prebeta.loc[dates][36:]

prebetan = prebeta.copy().dropna(axis=1) #sorting stocks each month by pre-beta
#del prebetan['세아제강지주']
for i in range(len(prebetan)):
    rank = prebetan[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank1 = rank1.dropna()
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    prebetan[i:i+1] = [rank2.iloc[:,0].values]
prebetan.columns = list(range(len((prebetan.columns)))) # 1st = lowest beta


quin = round(len(rank1) / 5)
PTF = []
#CNT = []
for t in range(len(prebetan)):    
    b1 = [item for sublist in prebetan.iloc[t:t+1,0:quin].values.tolist() for item in sublist]
    b2 = [item for sublist in prebetan.iloc[t:t+1,quin:quin*2].values.tolist() for item in sublist]
    b3 = [item for sublist in prebetan.iloc[t:t+1,quin*2:quin*3].values.tolist() for item in sublist]
    b4 = [item for sublist in prebetan.iloc[t:t+1,quin*3:quin*4].values.tolist() for item in sublist]
    b5 = [item for sublist in prebetan.iloc[t:t+1,quin*4:].values.tolist() for item in sublist]
    b_list =[b1,b2,b3,b4, b5]
    #ptf = [list(set(j).intersection(i)) for i in b_list for j in msr_list]
    #cnt = list(len(ptf[c]) for c in range(len(ptf)))
    PTF.append(b_list)
    #CNT.append(cnt)

#df_cnt = round(pd.DataFrame(CNT).mean(axis=0))
#df_cntm = np.array(df_cnt).reshape(5,5)
#df_cntm = pd.DataFrame(df_cntm, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','4','Overpriced']).T

#data_market = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper\\mispricing.xlsx')
#mktcap = pd.read_excel(data_market, 'Market_marketcap(M,T)')[24:-6]
#mktcap = mktcap.set_index(list(mktcap.columns[[0]]))
#R = pd.read_excel(data_market, 'M_returns(M,T)')[24:-6] 
#R = R.set_index(list(R.columns[[0]]))/100
#mktcapc = mktcap[common]
#Rc = R[common]

#rfc = rf[25:]
#ri_kospis = ri_kospi[24:]
#mktcap_kospis = mktcap_kospi[24:]
#rs_ptf = prebetan.iloc[:,:5].copy().T # beta sorted 5ptf returns
#for i in range(len(PTF)): # t = 180
#    rs = []
#    for j in range(len(PTF[i])):  #5 ptfs
#        r_ptf = ri_kospis[PTF[i][j]][i:i+1]
#        #mktcap_ptf = mktcap_kospis[PTF[i][j]][i:i+1]
#        #tmktcap = mktcap_ptf.sum().sum()
#        r = r_ptf.sum().sum() / len(PTF[i][j])
#        rs.append(r)
#    rs_ptf.iloc[:, i] = rs
#rs_ptf = rs_ptf.T
#rs_ptf.describe()

data_kospi = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI.xlsx')

ri_kospi = pd.read_excel(data_kospi, 'KOSPI_Ri')[12:] 
mktcap_kospi = pd.read_excel(data_kospi, 'MKTCAP')[12:] 
#r_kospi = pd.read_excel(data_kospi, 'KOSPI_R')[11:]
rf = pd.read_excel(data_kospi, 'Rf_MSB_adj')[11:]
#bv_kospi = pd.read_excel(data_kospi, 'BV_Y')[1:] #2001-2017

ri_kospi = ri_kospi.set_index(list(ri_kospi.columns[[0]])) / 100  # set the first column as index
mktcap_kospi = mktcap_kospi.set_index(list(mktcap_kospi.columns[[0]])) 
#r_kospi = r_kospi.set_index(list(r_kospi.columns[[0]])) /100
rf = rf.set_index(list(rf.columns[[0]])) /100
#bv_kospi = bv_kospi.set_index(list(bv_kospi.columns[[0]]))


ri_kospis = ri_kospi[24:]
mktcap_kospis = mktcap_kospi[24:]
rs_ptf = prebetan.iloc[:,:5].copy().T # beta sorted 4ptf returns
for i in range(len(PTF)): # t = 180
    rs = []
    for j in range(len(PTF[i])):  # 4 ptfs
        r_ptf = ri_kospis[PTF[i][j]][i:i+1]
        mktcap_ptf = mktcap_kospis[PTF[i][j]][i:i+1]
        tmktcap = mktcap_ptf.sum().sum()
        r = (r_ptf * mktcap_ptf / tmktcap).sum().sum()
        rs.append(r)
    rs_ptf.iloc[:, i] = rs
rs_ptf = rs_ptf.T
rs_ptf.describe()


#CAPM
rfc = rf[25:]
mkexr = r_kospim[36:].copy()
mkexr.iloc[:,0]= mkexr.values - rfc.values
rs_ptf.index = mkexr.index #index 그냥 맞춰주기

alpha = []
ts = []
ps = []
for i in range (len(rs_ptf.columns)):
        y = pd.DataFrame(rs_ptf.iloc[:,i].values - rfc.iloc[:,0].values)
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0].values], index = ['MKT']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b = est.params
        ttest = est.tvalues
        pvalue = est.pvalues
        alpha.append(a)
        ts.append(ttest)
        ps.append(pvalue)
        #est.summary()

a = alpha.copy()
df_a = np.array(a).reshape(5,5)
df_a = pd.DataFrame(df_a, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','4','Overpriced']).T
#df_pbm['Highest-Lowest'] = df_pbm.iloc[:,-1] - df_pbm.iloc[:,0]


#FF-3factors
MKT = mkt_exr[24:] 
rfc = rf[25:]
mkexr = MKT.copy()
mkexr.iloc[:,0]= MKT.values - rfc.values


common = prebeta.dropna(axis=1)
common = common.columns.tolist()
common.remove('세아제강지주')
#Size sorting (each month), mktcapc = mktcap[common]
mktcapc_size = mktcap_kospi.copy()[24:] #sorting stocks each month by marketcap(Size)
mktcapc_size = mktcapc_size[common]
for i in range(len(mktcapc_size)): #t = 180
    rank = mktcapc_size[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    mktcapc_size[i:i+1] = [rank2.iloc[:,0].values]
mktcapc_size.columns = list(range(len((mktcapc_size.columns)))) #small(위) to big(아래)

#B/M = Total Equity / mktcap
data_accnt = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper\\Accounting_information.xlsx')
TE = pd.read_excel(data_accnt, 'TE')[4:-2]
TE = TE.set_index(list(TE.columns[[0]]))
TEc = TE[common]

mktcap_kospic = mktcap_kospi[common][24:]
BV = mktcap_kospic.copy()
for i in range(int(len(BV)/12)):
    for m in range(12):
        n = i * 12
        q = (i+1) * 4
        BV[n+m:n+m+1] = TEc[q-1:q].values
        
BM = BV.copy() 
for i in range(len(BM)):
    BM[i:i+1] = BV[i:i+1] / mktcap_kospic[i:i+1]
    
#sorting stocks each month by BM
BM_sort = BM.copy() 
for i in range(len(BM_sort)):
    rank = BM_sort[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    BM_sort[i:i+1] = [rank2.iloc[:,0].values]
BM_sort.columns = list(range(len((BM_sort.columns)))) #Low to High B/M

# mktcapc_size -> Big Small, BM_sort -> H M L, generating 6 ptfs
half = round(len(mktcapc_size.columns) / 2)
thr_pct = round(len(BM_sort.columns) * 0.3)
for_pct = round(len(BM_sort.columns) * 0.7)
PTF_FF3 = []
for t in range(len(mktcapc_size)):    
    small = [item for sublist in mktcapc_size.iloc[t:t+1,0:half].values.tolist() for item in sublist]
    big = [item for sublist in mktcapc_size.iloc[t:t+1,half:].values.tolist() for item in sublist]
    
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
        r_ptff3 = ri_kospis[PTF_FF3[i][j]][i:i+1]
        mktcap_ptf = mktcap_kospic[PTF_FF3[i][j]][i:i+1]
        tmktcap = mktcap_ptf.sum().sum()
        r = (r_ptff3 * mktcap_ptf / tmktcap).sum().sum()
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

#Regression for estimating alpha of 25 ptfs
alpha = []
ts = []
ps = []
for i in range (len(rs_ptf.columns)):
        y = rs_ptf.iloc[:,i] - rfc.iloc[:,0]
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0], SMB.iloc[:,0], HML.iloc[:,0]], index = ['MKT', 'SMB', 'HML']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b, smb, hml = est.params
        ttest = est.tvalues
        pvalue = est.pvalues
        alpha.append(a)
        ts.append(ttest)
        ps.append(pvalue)
        #est.summary()

a = alpha.copy()
df_a = np.array(a).reshape(5,5)
df_a = pd.DataFrame(df_a, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','4','Overpriced']).T
#df_pbm['Highest-Lowest'] = df_pbm.iloc[:,-1] - df_pbm.iloc[:,0]

#writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\Table2.xlsx')
#df_a.to_excel(writer, 'alpha')
#writer.save()