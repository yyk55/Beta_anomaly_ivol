# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:08:28 2018

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

#data_beta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI_D.xlsx')
#
#dr = pd.read_excel(data_beta, 'KOSPI_Ri_D')[730:]
#drf = pd.read_excel(data_beta, 'CD91_adj')[730:]
#dmktr = pd.read_excel(data_beta, 'KOSPI_D')[730:]
#
#dr = dr.set_index(list(dr.columns[[0]])) / 100  # set the first column as index
#drf = drf.set_index(list(drf.columns[[0]])) / 100
#dmktr = dmktr.set_index(list(dmktr.columns[[0]])) /100
#
#'''Benchmark adjusted return'''
## Calculate the market excess returns
#mkexr = dmktr.copy()
#mkexr.iloc[:,0] = dmktr.iloc[:,0] - drf.iloc[:,0]
#
## Calculate individual stocks' excess returns
#exr = dr.copy()
#for i in range(len(exr.columns)):
#    exr.iloc[:,i] = dr.iloc[:,i] - drf.iloc[:,0]
#
##Regression for estimating benchmark adjusted returns
#bmr = exr.copy()
#residuals = []
#for i in range (len(exr.columns)):
#    y = exr.iloc[:,i]
#    x = mkexr.iloc[:,0]
#    x = sm.add_constant(x)
#    est = sm.OLS(y,x).fit()
#    #al, pb = est.params
#    #ttest = est.tvalues
#    resi = est.resid
#    #est.pvalues
#    #alpha.append(al)
#    #pbeta.append(pb)
#    #tvalues.append(ttest)
#    residuals.append(resi)
#    #est.summary()
#    bmr.iloc[:,i] = resi
#
###Regression for estimating benchmark adjusted returns
##bmr = exr.copy()
##for i in range (len(bmr)): #t = daily
##    residuals = []
##    #alpha = []
##    #pbeta = []
##    #tvalues = []
##    for j in range (len(exr.columns)):
##        y = exr.iloc[i,j]
##        x = mkexr.iloc[i,0]
##        x = sm.add_constant(x)
##        est = sm.OLS(y,x).fit()
##        #al, pb = est.params
##        #ttest = est.tvalues
##        resi = est.resid
##        #est.pvalues
##        #alpha.append(al)
##        #pbeta.append(pb)
##        #tvalues.append(ttest)
##        residuals.append(resi)
##        #est.summary()
##    bmr.iloc[i:i+1] = resi
#    
#writer = pd.ExcelWriter('C:\\Users\\장연숙\\Documents\\Paper2\\IVOL.xlsx')
#bmr.to_excel(writer, 'benchmark_adj_r')
#writer.save()

'''IVOL'''
data_bmr = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\IVOL_MSB_FF3Q.xlsx')
bmrs = pd.read_excel(data_bmr, 'benchmark_adj_r')
bmrs = bmrs.set_index(list(bmrs.columns[[0]])) 

data_beta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI_D.xlsx')
dr = pd.read_excel(data_beta, 'KOSPI_Ri_D')[730:]
dr = dr.set_index(list(dr.columns[[0]])) / 100 

# Calculate std of individual stock and index (rolling window = 365 days)
bmrs_std = dr[365:].copy().T
for i in range(len(bmrs_std.columns)):
    bmrs_std.iloc[:,i] = list(bmrs.iloc[i:i+365,:].std())
bmrs_std = bmrs_std.T

dates= pd.date_range('2000-01-01','2017-12-31' , freq='1M')-pd.offsets.MonthBegin(1)
dates= dates.tolist()
ivol = bmrs_std.loc[dates][36:]


'''Stock sorting by beta 5 ptfs'''
data_prebeta = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\exante_beta.xlsx')
data_mispmsr = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\mispring_Q6.xlsx')

prebeta = pd.read_excel(data_prebeta, 'beta_shrink_d') #from 2003 to 2017
misp_msr = pd.read_excel(data_mispmsr, 'misp_msr')
prebeta = prebeta.set_index(list(prebeta.columns[[0]]))
#prebeta = prebeta.dropna(axis=1)

data_kospi = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\KOSPI.xlsx')
ri_kospi = pd.read_excel(data_kospi, 'KOSPI_Ri')
mktcap_kospi = pd.read_excel(data_kospi, 'MKTCAP')
r_kospi = pd.read_excel(data_kospi, 'KOSPI_R')
rf = pd.read_excel(data_kospi, 'Rf_MSB_adj')
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
prebeta = prebeta.drop(remove, axis=1)
del prebeta['세아제강지주']

prebetan = prebeta.copy().dropna(axis=1) #sorting stocks each month by pre-beta
common = prebetan.columns.tolist()
for i in range(len(prebetan)):
    rank = prebetan[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank1 = rank1.dropna()
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    prebetan[i:i+1] = [rank2.iloc[:,0].values]
prebetan.columns = list(range(len((prebetan.columns)))) # 1st = lowest beta


misp_msrc = misp_msr[common]
misp_msrn = misp_msrc.copy() #sorting stocks each month by mispricing measure
for i in range(len(misp_msrc)):
    rank = misp_msrc[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    misp_msrn[i:i+1] = [rank2.iloc[:,0].values]
misp_msrn.columns = list(range(len((misp_msrn.columns)))) #0:underpricied

ivol_c = ivol[common]
ivol_n = ivol_c.copy() #sorting stocks each month by ivol
for i in range(len(ivol_c)):
    rank = ivol_c[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    ivol_n[i:i+1] = [rank2.iloc[:,0].values]
ivol_n.columns = list(range(len((ivol_n.columns)))) #0:low ivol


quin = round(len(rank1) / 5)
tri = round(len(rank1) / 3)
quar = round(len(rank1) / 4)
pct = round(len(rank1)*0.936)
PTF = []
CNT = []
PTFB = []
for t in range(len(prebetan)):
    high_b = [item for sublist in prebetan.iloc[t:t+1,pct:].values.tolist() for item in sublist]
    #overp = [item for sublist in misp_msrn.iloc[t:t+1,quar*3:].values.tolist() for item in sublist]
    #high_vol = [item for sublist in ivol_n.iloc[t:t+1,quar*3:].values.tolist() for item in sublist]
    #inter = [list(set(overp).intersection(high_vol))][0]
    
    bt = prebetan.iloc[t:t+1,:].values.tolist()
    bt = bt[0]
    interb = np.array([el for el in bt if el not in high_b]).tolist()
    
    mt = misp_msrn.iloc[t:t+1,:].values.tolist() 
    mt = mt[0]
    interm = np.array([el for el in mt if el not in high_b]).tolist()
    
    msr1 = interm[0:quar]
    msr2 = interm[quar:quar*2]
    msr3 = interm[quar*2:quar*3]
    msr4 = interm[quar*3:]
    
    b1 = interb[0:quin]
    b2 = interb[quin:quin*2]
    b3 = interb[quin*2:quin*3]
    b4 = interb[quin*3:quin*4]
    b5 = interb[quin*4:]
    
    msr_list = [msr1,msr2,msr3,msr4]
    b_list =[b1,b2,b3,b4,b5]
    ptf = [list(set(j).intersection(i)) for i in b_list for j in msr_list]
    cnt = list(len(ptf[c]) for c in range(len(ptf)))
    PTF.append(ptf)
    CNT.append(cnt)
    PTFB.append(b_list)
    
df_cnt = round(pd.DataFrame(CNT).mean(axis=0))
df_cntm = np.array(df_cnt).reshape(5,4)
df_cntm = pd.DataFrame(df_cntm, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','Overpriced']).T


ri_kospis = ri_kospi[36:]
mktcap_kospis = mktcap_kospi[36:]
rs_ptf = prebetan.iloc[:,:20].copy().T # 15ptf returns
for i in range(len(PTF)): # t = 180
    rs = []
    for j in range(len(PTF[i])):  #5 ptfs
        r_ptf = ri_kospis[PTF[i][j]][i:i+1]
#        mktcap_ptf = mktcap_kospis[PTF[i][j]][i:i+1]
#        tmktcap = mktcap_ptf.sum().sum()
#        r = (r_ptf * mktcap_ptf / tmktcap).sum().sum()
        r = r_ptf.sum().sum() / len(PTF[i][j])
        rs.append(r)
    rs_ptf.iloc[:, i] = rs
rs_ptf = rs_ptf.T
rs_ptf.describe()

rs_ptfb = prebetan.iloc[:,:5].copy().T # 5ptf returns
for i in range(len(PTFB)): # t = 180
    rs = []
    for j in range(len(PTFB[i])):  #5 ptfs
        r_ptfb = ri_kospis[PTFB[i][j]][i:i+1]
#        mktcap_ptf = mktcap_kospis[PTFB[i][j]][i:i+1]
#        tmktcap = mktcap_ptf.sum().sum()
#        r = (r_ptf * mktcap_ptf / tmktcap).sum().sum()
        r = r_ptfb.sum().sum() / len(PTFB[i][j])
        rs.append(r)
    rs_ptfb.iloc[:, i] = rs
rs_ptfb = rs_ptfb.T
rs_ptfb.describe()



rfc = rf[36:]
#Regression for estimating post-ranking beta (mkexr:INDv and exptfr:DPNv)
kospi03 = r_kospi[36:]
pbeta = []
tvalues = []
for i in range (len(rs_ptf.columns)): #i = 15 ptfs
        y =  pd.DataFrame(rs_ptf.iloc[:,i].values - rfc.iloc[:,0].values) 
        x =  pd.DataFrame(kospi03.iloc[:,0].values - rfc.iloc[:,0].values) 
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
df_pbm = np.array(pb).reshape(5,4)
df_pbm = pd.DataFrame(df_pbm, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','Overpriced']).T
df_pbm['Highest-Lowest'] = df_pbm.iloc[:,-1] - df_pbm.iloc[:,0]



#FF-3factors
# Calculate the market excess returns
mkt_exr = r_kospi.copy()
mkt_exr.iloc[:,0] = r_kospi.iloc[:,0] - rf.iloc[:,0]

MKT = mkt_exr[36:] 
mkexr = MKT.copy()

#Size sorting (each month), mktcapc = mktcap[common]
mktcapc_size = mktcap_kospi.copy()[36:] #sorting stocks each month by marketcap(Size)
mktcapc_size = mktcapc_size[common]
for i in range(len(mktcapc_size)): #t = 180
    rank = mktcapc_size[i:i+1].T
    rank1 = rank.sort_values(str(rank.columns.values[0]))
    rank2 = rank1.copy()
    rank2.iloc[:,0] = rank2.index
    mktcapc_size[i:i+1] = [rank2.iloc[:,0].values]
mktcapc_size.columns = list(range(len((mktcapc_size.columns)))) #small(위) to big(아래)

#B/M = Total Equity / mktcap
data_accnt = pd.ExcelFile('C:\\Users\\장연숙\\Documents\\Paper2\\BV.xlsx')
TE = pd.read_excel(data_accnt, 'BV_avg')[12:]
TE = TE.set_index(list(TE.columns[[0]]))
TEc = TE[common]

for i in range(len(TEc)):
    for j in range(len(TEc.columns)):
        if TEc.iloc[i+1:i+2,j].isnull().values == True:
            TEc.iloc[i+1:i+2,j] = TEc.iloc[i:i+1,j]

mktcap_kospic = mktcap_kospi[common][36:]
BV = mktcap_kospic.copy()
for i in range(int(len(BV)/3)): # i = 60 quarters 
    for m in range(3):
        n = i * 3 
        BV[n+m:n+m+1] = TEc[i:i+1].values

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
#        r = r_ptff3.sum().sum() / len(PTF_FF3[i][j])
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


#Regression for estimating alpha of 20 ptfs
alpha = []
ts = []
ps = []
for i in range (len(rs_ptf.columns)):
        y = pd.DataFrame(rs_ptf.iloc[:,i].values - rfc.iloc[:,0].values)
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0].values, SMB.iloc[:,0].values, HML.iloc[:,0].values], index = ['MKT', 'SMB', 'HML']).T
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
df_a = np.array(a).reshape(5,4)
df_a = pd.DataFrame(df_a, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','Overpriced']).T
df_a['Highest-Lowest'] = df_a.iloc[:,-1] - df_a.iloc[:,0]

p = [ps[i][0] for i in range(len(ps)) ].copy()
df_p = np.array(p).reshape(5,4)
df_p = pd.DataFrame(df_p, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','Overpriced']).T

t = [ts[i][0] for i in range(len(ts)) ].copy()
df_t = np.array(t).reshape(5,4)
df_t = pd.DataFrame(df_t, index=['Lowest','2','3','4','Highest'], columns=['Underpriced','2','3','Overpriced']).T


#Regression for calculating t-value of the diffrence btwn hihg and low of 20 ptfs
alphat = []
tst = []
pst = []
for i in range (int(len(rs_ptf.columns)-16)):
        y = pd.DataFrame((rs_ptf.iloc[:,i+16].values - rfc.iloc[:,0].values)-(rs_ptf.iloc[:,i].values - rfc.iloc[:,0].values))
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0].values, SMB.iloc[:,0].values, HML.iloc[:,0].values], index = ['MKT', 'SMB', 'HML']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b, smb, hml = est.params
        ttest = est.tvalues
        pvalue = est.pvalues
        alphat.append(a)
        tst.append(ttest)
        pst.append(pvalue)
        #est.summary()

at = alphat.copy()
df_at = np.array(at)
df_at = pd.DataFrame(df_at, index=['Underpriced','2','3','Overpriced'], columns=['Highest-Lowest'])

pt = [pst[i][0] for i in range(len(pst)) ].copy()
df_pt = np.array(pt)
df_pt = pd.DataFrame(df_pt, index=['Underpriced','2','3','Overpriced'], columns=['Highest-Lowest'])

tt = [tst[i][0] for i in range(len(tst)) ].copy()
df_tt = np.array(tt)
df_tt = pd.DataFrame(df_tt, index=['Underpriced','2','3','Overpriced'], columns=['Highest-Lowest'])


#Regression for estimating alpha of 5 ptfs
alphab = []
tsb = []
psb = []
for i in range (len(rs_ptfb.columns)):
        y = pd.DataFrame(rs_ptfb.iloc[:,i].values - rfc.iloc[:,0].values)
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0].values, SMB.iloc[:,0].values, HML.iloc[:,0].values], index = ['MKT', 'SMB', 'HML']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b, smb, hml = est.params
        ttest = est.tvalues
        pvalue = est.pvalues
        alphab.append(a)
        tsb.append(ttest)
        psb.append(pvalue)
        #est.summary()

b = alphab.copy()
df_ab = np.array(b)
df_ab = pd.DataFrame(df_ab, index=['Lowest','2','3','4','Highest'], columns=['All stocks']).T
df_ab['Highest-Lowest'] = df_ab.iloc[:,-1] - df_ab.iloc[:,0]

pb = [psb[i][0] for i in range(len(psb)) ].copy()
df_pb = np.array(pb)
df_pb = pd.DataFrame(df_pb, index=['Lowest','2','3','4','Highest'], columns=['All stocks']).T


tb = [tsb[i][0] for i in range(len(tsb)) ].copy()
df_tb = np.array(tb)
df_tb = pd.DataFrame(df_tb, index=['Lowest','2','3','4','Highest'], columns=['All stocks']).T

#Regression for calculating t-value of the diffrence btwn hihg and low of 5 ptfs
alphabt = []
tsbt = []
psbt = []
for i in range (int(len(rs_ptfb.columns)-4)):
        y = pd.DataFrame((rs_ptfb.iloc[:,i+4].values - rfc.iloc[:,0].values) - (rs_ptfb.iloc[:,i].values - rfc.iloc[:,0].values))
        #y = rs_ptf.iloc[:,i] - RF.iloc[:,0]
        x = pd.DataFrame([mkexr.iloc[:,0].values, SMB.iloc[:,0].values, HML.iloc[:,0].values], index = ['MKT', 'SMB', 'HML']).T
        x = sm.add_constant(x)
        est = sm.OLS(y,x).fit()
        a, b, smb, hml = est.params
        ttest = est.tvalues
        pvalue = est.pvalues
        alphabt.append(a)
        tsbt.append(ttest)
        psbt.append(pvalue)
        #est.summary()

abt = alphabt.copy()
df_abt = np.array(abt)
df_abt = pd.DataFrame(df_abt, index=['All stocks'], columns=['Highest-Lowest'])

pbt = [psbt[i][0] for i in range(len(psbt)) ].copy()
df_pbt = np.array(pbt)
df_pbt = pd.DataFrame(df_pbt, index=['All stocks'], columns=['Highest-Lowest'])

tbt = [tsbt[i][0] for i in range(len(tsbt)) ].copy()
df_tbt = np.array(tbt)
df_tbt = pd.DataFrame(df_tbt, index=['All stocks'], columns=['Highest-Lowest'])


