# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:54:57 2018

@author: 장연숙
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:44:52 2018

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

ri_kospi = pd.read_excel(data_kospi, 'KOSPI_Ri')[12:] 
mktcap_kospi = pd.read_excel(data_kospi, 'MKTCAP')[12:] 
r_kospi = pd.read_excel(data_kospi, 'KOSPI_R')[11:]
rf = pd.read_excel(data_kospi, 'Rf_CD91')[11:]
bv_kospi = pd.read_excel(data_kospi, 'BV_Y')[1:] #2001-2017

ri_kospi = ri_kospi.set_index(list(ri_kospi.columns[[0]])) / 100  # set the first column as index
mktcap_kospi = mktcap_kospi.set_index(list(mktcap_kospi.columns[[0]])) 
r_kospi = r_kospi.set_index(list(r_kospi.columns[[0]])) /100
rf = rf.set_index(list(rf.columns[[0]])) /100
bv_kospi = bv_kospi.set_index(list(bv_kospi.columns[[0]]))


# deleting negative book value and financial firm
a = bv_kospi.where(bv_kospi < 0)
nbv = [] #negative book value
for i in range(len(a.columns)):
    if a.iloc[:,i].any() == True:
        nbv.append(a.iloc[:,i].name)
fin = ['KB금융', '신한지주', '하나금융지주', '우리은행', '기업은행', '미래에셋대우', '한국금융지주', 'NH투자증권', '삼성증권', 'BNK금융지주', '메리츠종금증권', '키움증권', 'DGB금융지주', '메리츠금융지주', 'JB금융지주', '유안타증권', '대신증권', '한국자산신탁', '광주은행', '우리종금', '신영증권', '한화투자증권', 'SK증권', '현대차증권', '유진투자증권', 'KTB투자증권', '부국증권', 'DB금융투자', '유화증권', '골든브릿지증권', '제주은행', '한양증권']
remove = list(set(nbv + fin)) 
ri_kospi = ri_kospi.drop(remove, axis=1)
mktcap_kospi = mktcap_kospi.drop(remove, axis=1)

# Calculate the market excess returns
mkt_exr = r_kospi.copy()
mkt_exr.iloc[:,0] = r_kospi.iloc[:,0] - rf.iloc[:,0]
mktexr_lag = mkt_exr.shift().fillna(0)
mkt_exr = mkt_exr[1:]
mktexr_lag = mktexr_lag[1:]

# Calculate individual stocks' excess returns
ex_ri = ri_kospi.copy()
for i in range(len(ex_ri.columns)):
    ex_ri.iloc[:,i] = ri_kospi.iloc[:,i] - rf.iloc[:,0]

#Regression for estimating pre-ranking beta
b0mtrx = ex_ri.copy()[24:]
b1mtrx = ex_ri.copy()[24:]
for i in range (len(ex_ri.columns)):
    alpha = []
    beta0 = []
    beta1 = []
    for j in range(len(ex_ri.index)-24):
        y = ex_ri.iloc[j:j+24,i]
        x = pd.DataFrame([mkt_exr.iloc[j:j+24,0], mktexr_lag.iloc[j:j+24,0]], index=['mkt', 'mkt-1']).T
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

writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\pre_beta.xlsx')
b0mtrx.to_excel(writer, 'beta0')
b1mtrx.to_excel(writer, 'beta1')
bmtrx.to_excel(writer,'beta')
writer.save()


'''Stock sorting by beta and misp 15 ptfs'''
data_prebeta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\pre_beta.xlsx')
data_mispmsr = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper\\misp_msr.xlsx')
prebeta = pd.read_excel(data_prebeta, 'beta') #from 2003 to 2017
misp_msr = pd.read_excel(data_mispmsr, 'misp_msr')
prebeta = prebeta.set_index(list(prebeta.columns[[0]]))

common = prebeta.dropna(axis=1)
common = common.columns.tolist()
common.remove('세아제강지주')

prebetan = prebeta.copy().dropna(axis=1) #sorting stocks each month by pre-beta
del prebetan['세아제강지주']
for i in range(len(prebetan)):
    rank = prebetan[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank1 = rank1.dropna()
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    prebetan[i:i+1] = [rank2.iloc[:,0].values]
prebetan.columns = list(range(len((prebetan.columns)))) # 1st = lowest beta

'''Momentum'''
R = pd.read_excel(data_kospi, 'KOSPI_Ri')[24:]
R = R.set_index(list(R.columns[[0]])) / 100  # set the first column as index

MOM = []
for i in range(int(len(R-11-1))):
	e = (R[i:i+1].values+R[i+1:i+2].values+R[i+2:i+3].values+R[i+3:i+4].values+R[i+4:i+5].values+R[i+5:i+6].values+R[i+6:i+7].values+R[i+7:i+8].values+R[i+8:i+9].values+R[i+9:i+10].values+R[i+10:i+11].values).tolist()	
	MOM.append(e)
	
flat_MOM = [item for sublist in MOM for item in sublist]
labels = list(R.columns)
date_rngm = pd.date_range(start='2003-01-31', end='2017-12-31', freq='M')
df_MOM = pd.DataFrame(flat_MOM[:-2], index= date_rngm, columns = labels)
df_MOM = df_MOM[common] #cumulrative returns of 11 months 

#with Nan
nan_MOM = df_MOM
r_MOM = nan_MOM.copy().T

for i in range(len(r_MOM.columns)):
    r_MOM.iloc[:,i] = r_MOM.iloc[:,i].rank(pct=True, ascending=False)
R_MOM = r_MOM.T # ranking


misp_msrc = R_MOM
misp_msrn = misp_msrc.copy() #sorting stocks each month by mispricing measure
for i in range(len(misp_msrc)):
    rank = misp_msrc[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    misp_msrn[i:i+1] = [rank2.iloc[:,0].values]
misp_msrn.columns = list(range(len((misp_msrn.columns)))) #0:underpricied


#data_mispmsr = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper\\misp_msr.xlsx')
#misp_msr = pd.read_excel(data_mispmsr, 'misp_msr')


quin = round(len(rank1) / 5)
tri = round(len(rank1) / 3)
PTF = []
CNT = []
for t in range(len(prebetan)):
    msr1 = [item for sublist in misp_msrn.iloc[t:t+1,0:tri].values.tolist() for item in sublist]
    msr2 = [item for sublist in misp_msrn.iloc[t:t+1,tri:tri*2].values.tolist() for item in sublist]
    msr3 = [item for sublist in misp_msrn.iloc[t:t+1,tri*2:].values.tolist() for item in sublist]
    
    b1 = [item for sublist in prebetan.iloc[t:t+1,0:quin].values.tolist() for item in sublist]
    b2 = [item for sublist in prebetan.iloc[t:t+1,quin:quin*2].values.tolist() for item in sublist]
    b3 = [item for sublist in prebetan.iloc[t:t+1,quin*2:quin*3].values.tolist() for item in sublist]
    b4 = [item for sublist in prebetan.iloc[t:t+1,quin*3:quin*4].values.tolist() for item in sublist]
    b5 = [item for sublist in prebetan.iloc[t:t+1,quin*4:].values.tolist() for item in sublist]
    msr_list = [msr1,msr2,msr3]
    b_list =[b1,b2,b3,b4,b5]
    ptf = [list(set(j).intersection(i)) for i in b_list for j in msr_list]
    cnt = list(len(ptf[c]) for c in range(len(ptf)))
    PTF.append(ptf)
    CNT.append(cnt)

df_cnt = round(pd.DataFrame(CNT).mean(axis=0))
df_cntm = np.array(df_cnt).reshape(5,3)
df_cntm = pd.DataFrame(df_cntm, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','Overpriced']).T

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
    
rfc = rf[25:]
ri_kospis = ri_kospi[24:]
mktcap_kospis = mktcap_kospi[24:]
rs_ptf = prebetan.iloc[:,:15].copy().T # 15ptf returns
for i in range(len(PTF)): # t = 180
    rs = []
    for j in range(len(PTF[i])):  #5 ptfs
        r_ptf = ri_kospis[PTF[i][j]][i:i+1]
        mktcap_ptf = mktcap_kospis[PTF[i][j]][i:i+1]
        tmktcap = mktcap_ptf.sum().sum()
        r = (r_ptf * mktcap_ptf / tmktcap).sum().sum()
        rs.append(r)
    rs_ptf.iloc[:, i] = rs
rs_ptf = rs_ptf.T
rs_ptf.describe()

#Regression for estimating post-ranking beta (mkexr:INDv and exptfr:DPNv)
kospi03 = r_kospi[25:]
pbeta = []
tvalues = []
for i in range (len(rs_ptf.columns)): #i = 15 ptfs
        y = rs_ptf.iloc[:,i] 
        x = kospi03.iloc[:,0] 
        #x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        pb = est.params
        ttest = est.tvalues
        #est.pvalues
        #palpha.append(pa)
        pbeta.append(pb)
        tvalues.append(ttest)
        est.summary()

pb = df_cnt.copy()
pb[:] = pbeta
df_pbm = np.array(pb).reshape(5,3)
df_pbm = pd.DataFrame(df_pbm, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','Overpriced']).T
df_pbm['Highest-Lowest'] = df_pbm.iloc[:,-1] - df_pbm.iloc[:,0]


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
df_a = np.array(a).reshape(5,3)
df_a = pd.DataFrame(df_a, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','Overpriced']).T
df_a['Highest-Lowest'] = df_a.iloc[:,-1] - df_a.iloc[:,0]

#writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper\\Table2.xlsx')
#df_a.to_excel(writer, 'alpha')
#writer.save()
    
