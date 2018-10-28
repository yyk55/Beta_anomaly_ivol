# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:32:02 2018

@author: na88555
"""

import pandas as pd
import numpy as np

#Inv.iloc[0:1] select row
#Inv[0:1] select row
#Inv.loc['2002Q2'] select row
#Inv['삼성전자'] select column
#Inv.iloc[:,0] or Inv.iloc[:, 0:3] select column  

data_accnt = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\accounting_infoQ.xlsx')

'''Investment-to-assets'''
Inv = pd.read_excel(data_accnt, 'Inv')
PPE = pd.read_excel(data_accnt, 'PPE')
TE = pd.read_excel(data_accnt, 'TE_notavg')

Inv = Inv.set_index(list(Inv.columns[[0]]))
PPE = PPE.set_index(list(PPE.columns[[0]]))
TE = TE.set_index(list(TE.columns[[0]]))

# fill nan values with the previous values
for i in range(len(Inv)):
    for j in range(len(Inv.columns)):
        if Inv.iloc[i+1:i+2,j].isnull().values == True:
            Inv.iloc[i+1:i+2,j] = Inv.iloc[i:i+1,j].values

for i in range(len(PPE)):
    for j in range(len(PPE.columns)):
        if PPE.iloc[i+1:i+2,j].isnull().values == True:
            PPE.iloc[i+1:i+2,j] = PPE.iloc[i:i+1,j].values

for i in range(len(TE)):
    for j in range(len(TE.columns)):
        if TE.iloc[i+1:i+2,j].isnull().values == True:
            TE.iloc[i+1:i+2,j] = TE.iloc[i:i+1,j].values
            
ItoA = []
for i in range(len(Inv)):
	#t1 = 4*(i+2)-1
	#t0 = 4*(i+1)-1
	a = (((Inv[i+1:i+2].values-Inv[i:i+1].values)+(PPE[i+1:i+2].values-PPE[i:i+1].values))/TE[i:i+1].values).tolist()	
	ItoA.append(a)

del ItoA[-1]

flat_ItoA = [item for sublist in ItoA for item in sublist]
labels = list(Inv.columns)
date_rng = pd.date_range(start='2000-04-01', end='2018', freq='Q')
df_ItoA = pd.DataFrame(flat_ItoA, index= date_rng, columns = labels)
fin = ['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한화투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '제주은행', '한양증권']
df_ItoA = df_ItoA.drop(fin, axis=1)

#without Nan
filtered_ItoA = df_ItoA.dropna(axis=1)
rank_ItoA = filtered_ItoA.copy().T

for i in range(len(rank_ItoA.columns)):
    rank_ItoA.iloc[:,i] = rank_ItoA.iloc[:,i].rank(pct=True)
Rank_ItoA = rank_ItoA.T

#with Nan
nan_ItoA = df_ItoA
r_ItoA = nan_ItoA.copy().T

for i in range(len(r_ItoA.columns)):
    r_ItoA.iloc[:,i] = r_ItoA.iloc[:,i].rank(pct=True)
R_ItoA = r_ItoA.T

#writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\ItoA.xlsx')
#Rank_ItoA.to_excel(writer, 'ItoA')
#R_ItoA.to_excel(writer, 'ItoA_nan')
#writer.save()
    

'''Return on Assets''' 
NI = pd.read_excel(data_accnt, 'NI')
TE = pd.read_excel(data_accnt, 'TE_notavg')

NI = NI.set_index(list(NI.columns[[0]]))
TE = TE.set_index(list(TE.columns[[0]]))

# fill nan values with the previous values
for i in range(len(NI)):
    for j in range(len(NI.columns)):
        if NI.iloc[i+1:i+2,j].isnull().values == True:
            NI.iloc[i+1:i+2,j] = NI.iloc[i:i+1,j].values

for i in range(len(TE)):
    for j in range(len(TE.columns)):
        if TE.iloc[i+1:i+2,j].isnull().values == True:
            TE.iloc[i+1:i+2,j] = TE.iloc[i:i+1,j].values
            
ROA = []
for i in range(len(NI)):
	#t1 = 4*(i+2)-1
	#t0 = 4*(i+1)-1
	b = (NI[i+1:i+2].values/TE[i:i+1].values).tolist()	
	ROA.append(b)
    
del ROA[-1]

flat_ROA = [item for sublist in ROA for item in sublist]
labels = list(NI.columns)
date_rng = pd.date_range(start='2000-04-01', end='2018', freq='Q')
df_ROA = pd.DataFrame(flat_ROA, index= date_rng, columns = labels)
fin = ['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한화투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '제주은행', '한양증권']
df_ROA = df_ROA.drop(fin, axis=1)

#without Nan
filtered_ROA = df_ROA.dropna(axis=1)
rank_ROA = filtered_ROA.copy().T

for i in range(len(rank_ROA.columns)):
    rank_ROA.iloc[:,i] = rank_ROA.iloc[:,i].rank(pct=True, ascending=False)
Rank_ROA = rank_ROA.T

#with Nan
nan_ROA = df_ROA
r_ROA = nan_ROA.copy().T

for i in range(len(r_ROA.columns)):
    r_ROA.iloc[:,i] = r_ROA.iloc[:,i].rank(pct=True, ascending=False)
R_ROA = r_ROA.T
    
#writer2 = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\ROA.xlsx')
#Rank_ROA.to_excel(writer2, 'ROA')
#R_ROA.to_excel(writer2, 'ROA_nan')
#writer2.save()
    
'''Asset Growth'''
TA = pd.read_excel(data_accnt, 'TA_notavg')
TA = TA.set_index(list(TA.columns[[0]]))

# fill nan values with the previous values
for i in range(len(TA)):
    for j in range(len(TA.columns)):
        if TA.iloc[i+1:i+2,j].isnull().values == True:
            TA.iloc[i+1:i+2,j] = TA.iloc[i:i+1,j].values

AG = []
for i in range(len(TA)):
	#t1 = 4*(i+2)-1
	#t0 = 4*(i+1)-1
	c = (TA[i+1:i+2].values/TA[i:i+1].values).tolist()	
	AG.append(c)
    
del AG[-1]
  
flat_AG = [item for sublist in AG for item in sublist]
labels = list(TA.columns)
date_rng = pd.date_range(start='2000-04-01', end='2018', freq='Q')
df_AG = pd.DataFrame(flat_AG, index= date_rng, columns = labels)
fin = ['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한화투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '제주은행', '한양증권']
df_AG = df_AG.drop(fin, axis=1)

#without Nan
filtered_AG = df_AG.dropna(axis=1)
rank_AG = filtered_AG.copy().T

for i in range(len(rank_AG.columns)):
    rank_AG.iloc[:,i] = rank_AG.iloc[:,i].rank(pct=True)
Rank_AG = rank_AG.T

#with Nan
nan_AG = df_AG
r_AG = nan_AG.copy().T

for i in range(len(r_AG.columns)):
    r_AG.iloc[:,i] = r_AG.iloc[:,i].rank(pct=True)
R_AG = r_AG.T
    
#writer3 = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\AG.xlsx')
#Rank_AG.to_excel(writer3, 'AG')
#R_AG.to_excel(writer3, 'AG_nan')
#writer3.save()

'''Gross Profitability Premium'''
GP = pd.read_excel(data_accnt, 'GP')
TA = pd.read_excel(data_accnt, 'TA_notavg')

GP = GP.set_index(list(GP.columns[[0]]))
TA = TA.set_index(list(TA.columns[[0]]))

# fill nan values with the previous values
for i in range(len(GP)):
    for j in range(len(GP.columns)):
        if GP.iloc[i+1:i+2,j].isnull().values == True:
            GP.iloc[i+1:i+2,j] = GP.iloc[i:i+1,j].values
            
for i in range(len(TA)):
    for j in range(len(TA.columns)):
        if TA.iloc[i+1:i+2,j].isnull().values == True:
            TA.iloc[i+1:i+2,j] = TA.iloc[i:i+1,j].values

GPP = []
for i in range(len(GP)):
	#t1 = 4*(i+2)-1
	#t0 = 4*(i+1)-1
	d = (GP[i:i+1].values/TA[i:i+1].values).tolist()	
	GPP.append(d)
	
del GPP[0]
 
flat_GPP = [item for sublist in GPP for item in sublist]
labels = list(GP.columns)
date_range = pd.date_range(start='2000-04-01', end='2018', freq='Q')
df_GPP = pd.DataFrame(flat_GPP, index= date_range, columns = labels)
fin = ['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한화투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '제주은행', '한양증권']
df_GPP = df_GPP.drop(fin, axis = 1)

                     
#without Nan
filtered_GPP = df_GPP.dropna(axis=1)
rank_GPP = filtered_GPP.copy().T

for i in range(len(rank_GPP.columns)):
    rank_GPP.iloc[:,i] = rank_GPP.iloc[:,i].rank(pct=True, ascending=False)
Rank_GPP = rank_GPP.T

#with Nan
nan_GPP = df_GPP
r_GPP = nan_GPP.copy().T

for i in range(len(r_GPP.columns)):
    r_GPP.iloc[:,i] = r_GPP.iloc[:,i].rank(pct=True, ascending=False)
R_GPP = r_GPP.T
    
#writer4 = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\GPP.xlsx')
#Rank_GPP.to_excel(writer4, 'GPP')
#R_GPP.to_excel(writer4, 'GPP_nan')
#writer4.save()
    
'''Momentum'''
data_market = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper\\mispricing.xlsx')

R = pd.read_excel(data_market, 'M_returns(M,T)')[:-6]
R = R.set_index(list(R.columns[[0]]))

MOM = []
for i in range(int(len(R-11-1))):
	e = (R[i:i+1].values+R[i+1:i+2].values+R[i+2:i+3].values+R[i+3:i+4].values+R[i+4:i+5].values+R[i+5:i+6].values+R[i+6:i+7].values+R[i+7:i+8].values+R[i+8:i+9].values+R[i+9:i+10].values+R[i+10:i+11].values).tolist()	
	MOM.append(e)
	
flat_MOM = [item for sublist in MOM for item in sublist]
labels = list(R.columns)
date_rngm = pd.date_range(start='2003-01-31', end='2017-12-31', freq='M')
df_MOM = pd.DataFrame(flat_MOM[:-2], index= date_rngm, columns = labels)
df_MOM = df_MOM.drop(['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한국전자금융', '한화투자증권', '이베스트투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '에이티넘인베스트', 'SV인베스트먼트', '푸른저축은행', '제주은행', '한양증권', 'DSC인베스트먼트', '대성창투', '티에스인베스트먼트', '제미니투자'], axis=1)

#without Nan
filtered_MOM = df_MOM.dropna(axis=1)
rank_MOM = filtered_MOM.copy().T

for i in range(len(rank_MOM.columns)):
    rank_MOM.iloc[:,i] = rank_MOM.iloc[:,i].rank(pct=True, ascending=False)
Rank_MOM = rank_MOM.T

#with Nan
nan_MOM = df_MOM
r_MOM = nan_MOM.copy().T

for i in range(len(r_MOM.columns)):
    r_MOM.iloc[:,i] = r_MOM.iloc[:,i].rank(pct=True, ascending=False)
R_MOM = r_MOM.T
    
#writer5 = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\MOM.xlsx')
#Rank_MOM.to_excel(writer5, 'MOM')
#R_MOM.to_excel(writer5, 'MOM_nan')
#writer5.save()


'''Consolidating the df of all mispricings'''
# 샘플기간 맞춰주기 (from 2003 - 2017)
R_AG = R_AG[10:-1]
R_GPP = R_GPP[10:-1]
R_ItoA = R_ItoA[10:-1]
R_ROA = R_ROA[10:-1]
cm = R_AG.columns.tolist()
cm.remove('하나제약')
cm.remove('세아제강지주')
R_MOM = R_MOM[cm]
dm = R_MOM.columns.tolist()
R_AG = R_AG[dm]
R_GPP = R_GPP[dm]
R_ItoA = R_ItoA[dm]
R_ROA = R_ROA[dm]

exp_ItoA = R_MOM.copy() # setting df as 180 monthly samples
for i in range(len(R_ItoA)):
    for m in range (3):
        n = 3*i
        exp_ItoA[n+m:n+m+1] = R_ItoA[i:i+1].values
        
exp_ROA = R_MOM.copy() # setting df as 180 monthly samples
for i in range(len(R_ROA)):
    for m in range (3):
        n = 3*i
        exp_ROA[n+m:n+m+1] = R_ROA[i:i+1].values
        
exp_AG = R_MOM.copy() # setting df as 180 monthly samples
for i in range(len(R_AG)):
    for m in range (3):
        n = 3*i
        exp_AG[n+m:n+m+1] = R_AG[i:i+1].values
        
exp_GPP = R_MOM.copy() # setting df as 180 monthly samples
for i in range(len(R_GPP)):
    for m in range (3):
        n = 3*i
        exp_GPP[n+m:n+m+1] = R_GPP[i:i+1].values

#counting the number of nan and no-nan
fna_ItoA = exp_ItoA.fillna(0).copy()
cnt_ItoA = fna_ItoA.where(fna_ItoA == 0, 1)

fna_ROA = exp_ROA.fillna(0).copy()
cnt_ROA = fna_ROA.where(fna_ROA == 0, 1)

fna_AG = exp_AG.fillna(0).copy()
cnt_AG = fna_AG.where(fna_AG == 0, 1)

fna_GPP = exp_GPP.fillna(0).copy()
cnt_GPP = fna_GPP.where(fna_GPP == 0, 1)

fna_MOM = R_MOM.fillna(0).copy()
cnt_MOM = fna_MOM.where(fna_MOM == 0, 1)

count = cnt_ItoA + cnt_ROA + cnt_AG + cnt_GPP + cnt_MOM

# calculation of total mispricing measure
sum_misp = fna_ItoA + fna_ROA + fna_AG + fna_GPP + fna_MOM
misp_msr = (sum_misp / count).dropna(axis=1)

writer6 = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\mispring_Q.xlsx')
misp_msr.to_excel(writer6, 'misp_msr')
writer6.save()