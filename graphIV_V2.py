
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:31:05 2024

@author: dzrh
"""

# from CYL.StressTestNew import BarrierReport
import streamlit as stm
import rqdatac as rqd
from datetime import datetime,time
import time as sys_time
from CYL.StressTestNew import getRQcode
# from CYL.YieldChainAPI import YieldChainAPI
# rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
# rqd.init()
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.stats import norm
import os


# stm.session_state
rf=0.03
q=0
# 添加两条线
# def init_session():


def getBSMPriceArr(opttpye_ts,s_ts,k_ts,t_ts,iv_ts):
    b=0    
    d1=(np.log(s_ts/k_ts)+t_ts*0.5*iv_ts**2)/(iv_ts*np.sqrt(t_ts))
    d2=d1-iv_ts*np.sqrt(t_ts)
    
    s_dis=s_ts*np.exp((b-rf)*t_ts)
    k_dis=k_ts*np.exp(-1*rf*t_ts)
    
    c_p=s_dis*norm.cdf(d1)-k_dis*norm.cdf(d2)
    p=c_p+np.where(opttpye_ts=='C',0,1)*(k_dis-s_dis)
    return p

def getDeltaArr(opttype_ts,s_ts,k_ts,t_ts,iv_ts):
    '''
    return delta array like
    
    Parameters
    ----------
    opttype_ts : array
        
    s_ts : array

    k_ts : array
    
    t_ts : array
   
    iv_ts : array
    Returns
    -------
    delta_ts:array like

    '''
    b=0
    d1=(np.log(s_ts/k_ts)+t_ts*(b+0.5*iv_ts**2))/(iv_ts*np.sqrt(t_ts))
    delta_ts=norm.cdf(d1)+np.where(opttype_ts=='C',0,-1)*np.exp((b-rf)*t_ts)
    return delta_ts

def getIVArr(opttype_ts,s_ts,k_ts,t_ts,target,dim):
    '''
    return iv on an array type

    Parameters
    ----------
    opttpye_ts : TYPE
        DESCRIPTION.
    s_ts : TYPE
        DESCRIPTION.
    k_ts : TYPE
        DESCRIPTION.
    t_ts : TYPE
        DESCRIPTION.
    target : array_list
        option price array
    dim : int
        dimention of data.

    Returns
    -------
    iv_ts : TYPE
        DESCRIPTION.

    '''
    
    high = 2*np.ones(dim)
    low = np.zeros(dim)
    i=0
    while np.any((high-low) > 1.0e-5):
        if i>=1000:
            break
        else:
           i+=1
           # print(i+1)
           p=getBSMPriceArr(opttype_ts, s_ts, k_ts, t_ts,(high + low) / 2)
           mid=(high+low)/2
            
           high=np.where(p>target,mid,high)
           low=np.where(p<target,mid,low)
   
    iv_ts = (high + low) / 2
    return iv_ts    

def getRule():
    loc=r'D:\chengyilin\work\2.Trading\system\OptionStrikeRule.xlsx'
    dfrule=pd.read_excel(loc,index_col=0)
    dfrule.insert(loc=3,column='Level4',value=dfrule['Level3']*5)
    dfrule['Level3'].fillna(5*dfrule['Level2'],inplace=True)
    dfrule['Level4'].fillna(dfrule['Level3'],inplace=True)
    dfrule['Dk4'].fillna(dfrule['Dk3'],inplace=True)
    
    # dfrule.insert(loc=2,column='Level3',value=dfrule['Level2']*5)
  
    return dfrule
def findCode(underlying):
    '''
    Return Varity Code of Underlying

    Parameters
    ----------
    underlying : TYPE
        DESCRIPTION.

    Returns
    -------
    code : str
    '''
    # if underlying=="000300.XSHG":
    #     code="IO"
    # elif underlying=="000905.XSHG":
    #     code="IC"
    # else:
    code=underlying[0].upper() if underlying[1].isdigit() else underlying[:2].upper()
    return code
def find_dk(rule,p):
    a=np.where(p<=rule[0],1,0)
    dk=((a-np.insert(a[:-1],0,0))*rule[1]).sum()
    return dk

def find_k(rule,last_p,rng):
    k1=np.arange(0,rule[0,0]+rule[1,0],rule[1,0])
    k2=np.arange(rule[0,0]+rule[1,1],rule[0,1]+rule[1,1],rule[1,1])
    k3=np.arange(rule[0,1]+rule[1,2],rule[0,2]+rule[1,2],rule[1,2])
    k_ts=np.hstack((k1,k2,k3))
    dk_0=find_dk(rule,last_p)
    atm_k=round(last_p/dk_0,0)*dk_0
    atm_idx=np.argwhere(k_ts==atm_k)[0][0]
    # sel_k=k_ts[atm_idx-rng:atm_idx+rng]
    if atm_k>last_p:
        sel_k=k_ts[atm_idx-rng:atm_idx+rng]
    else :
        sel_k=k_ts[atm_idx:atm_idx+rng+1]
    return sel_k

def formatOptionCode(underlying,k_ts,optiontype,con):
    '''
    Parameters
    ----------
    underlying : eg i2301.DCE
    k_ts : list
        strike list.
    optiontype : 'C' or 'P'
        call or put
    con : str

    Returns
    -------
    option_code : list
        generated option codes

    '''
    # exchange=underlying[-4:]
    # if underlying=="000300.XSHG":
        # option_code=list(map(lambda x:underlying+con+optiontype+con+str(int(x)),k_ts))
    option_code=list(map(lambda x:underlying+con+optiontype+con+str(int(x)),k_ts))
    return option_code

def getOptionCodes(underlyingCode,s,rng):
      dfrule=getRule()
      code=findCode(underlyingCode)
      rule=dfrule.loc[code].values[:-1].reshape((2,4))  
      k_ts=find_k(rule,s,rng)
      # con=findConnaction(dfrule, code)
      con=""
      option_code_call=formatOptionCode(underlyingCode, k_ts, 'C', con)
      option_code_put=formatOptionCode(underlyingCode, k_ts, 'P', con)
      option_code=option_code_call+option_code_put
      return option_code
  
# def getLatestIV(underlyingCode):
#     snp=rqd.current_snapshot(underlyingCode)
    
#     options_contracts=getOptionCodes(underlyingCode,snp.last,1)
#     option_instruments=rqd.instruments(options_contracts)

#     optlast_ts=np.array([rqd.current_snapshot(opt).last for opt in options_contracts])
    
#     opttype_ts=np.array([inst.option_type for inst in option_instruments])
#     k_ts=np.array([inst.strike_price for inst in option_instruments])
    
#     cdt_time=[snp.datetime.time()<time(9,0,0)
#               ,snp.datetime.time()<time(11,30,0)
#               ,snp.datetime.time()<time(13,30,0)
#               ,snp.datetime.time()<time(15,0,0)
#               ,snp.datetime.time()<time(21,0,0)
#               ,snp.datetime.time()<=time(23,0,0)]
#     cho_intra=[4/6
#                 ,2/6
#                 ,1.5/6
#                 ,0.5/6
#                 ,0
#                 ,5/6]
#     intr_hours=np.select(cdt_time,cho_intra,0)
#     ttm_days=intr_hours+stm.session_state.trading_list.index(pd.to_datetime(option_instruments[0].maturity_date))-stm.session_state.trading_list.index(cur_trd_date)

#     cur_iv=np.mean(getIVArr(opttype_ts,np.ones(opttype_ts.shape)*snp.last,k_ts,np.ones(opttype_ts.shape)*ttm_days/252,optlast_ts,1))
#     # idx=snp.datetime
#     # res=pd.Series(data=round(cur_iv,4)*100,index=idx)
#     # return res
#     return round(cur_iv,4)*100



def getLatestIV(underlyingCode):
    snp=rqd.current_snapshot(underlyingCode)
    cdt_time=[snp.datetime.time()<time(9,0,0)
              ,snp.datetime.time()<time(11,30,0)
              ,snp.datetime.time()<time(13,30,0)
              ,snp.datetime.time()<time(15,0,0)
              ,snp.datetime.time()<time(21,0,0)
              ,snp.datetime.time()<=time(23,0,0)]
    cho_intra=[4/6
                ,2/6
                ,1.5/6
                ,0.5/6
                ,0
                ,5/6]
    intr_hours=np.select(cdt_time,cho_intra,0)
    # dfcur_iv=pd.DataFrame(columns=[0.95,1,1.05],index='cur_iv')
    # dic_level={0.95:pd.DataFrame(),1:pd.DataFrame(),1.05:pd.DataFrame()}
    cur_iv_list=[]
    for col in moneyness:
        options_contracts=getOptionCodes(underlyingCode,snp.last*col,1)
        if col>1:
            options_contracts=options_contracts[:2]
        elif col<1:
            options_contracts=options_contracts[2:]
        else:
            pass
    
        option_instruments=rqd.instruments(options_contracts)
    
        optasks_ts=[rqd.current_snapshot(opt).asks[0] for opt in options_contracts]
        optbids_ts=[rqd.current_snapshot(opt).bids[0] for opt in options_contracts]
        
        opttype_ts=[inst.option_type for inst in option_instruments]
        k_ts=[inst.strike_price for inst in option_instruments]
        ttm_days=intr_hours+stm.session_state.trading_list.index(pd.to_datetime(option_instruments[0].maturity_date))-stm.session_state.trading_list.index(cur_trd_date)
        # ttm_days=intr_hours+trading_list.index(pd.to_datetime(option_instruments[0].maturity_date))-trading_list.index(cur_trd_date)
        
        cur_iv=np.mean(getIVArr(np.array(opttype_ts*2),np.ones(len(opttype_ts)*2)*snp.last
                                ,np.array(k_ts*2)
                                ,np.ones(len(opttype_ts)*2)*ttm_days/252,np.array(optasks_ts+optbids_ts),1))
        
        cur_iv_list.append(round(cur_iv,4)*100)
        

    return np.array(cur_iv_list).reshape(1,3)



def getHistIV(underlyingCode,start_date,end_date,given_delta=50,optionList=""):
    trading_date_list=stm.session_state.trading_list
    # trading_date_list=trading_list
    freq='1m'
  
        
    if optionList=="":
        option_list=rqd.options.get_contracts(underlyingCode)
    else:
        option_list=optionList
        
    tic=datetime.now()
    try:
        wd=rqd.get_price(option_list,start_date,end_date,freq,fields=['close','volume','trading_date']).reset_index()
    except:
        stm.write(underlyingCode)
    # wd.drop(index=wd[wd.volume==0].index,inplace=True)
    wd['datetime']=pd.to_datetime(wd.datetime)
    wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
    option_instruments=rqd.instruments(wd.order_book_id)
      
    wd['strike']=list(map(lambda x:x.strike_price,option_instruments))
    wd['optiontype']=list(map(lambda x:x.option_type,option_instruments))
    wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
    # wd['expire_date']=np.where(wd.expire_date==datetime(2024,2,13).date(),datetime(2024,2,7).date(),wd.expire_date)
    
    wd['underlying']=list(map(lambda x:x.underlying_order_book_id,option_instruments))
    print("Format time with = ", datetime.now() - tic, "s")
    
    
    
    tic=datetime.now()
    wd['exp_idx']=wd.expire_date.apply(lambda x:trading_date_list.index(x))
    wd['trd_idx']=wd.trading_date.apply(lambda x:trading_date_list.index(x))
    wd['trd_time']=wd.datetime.apply(lambda x:x.time())
    cdt_time=[wd.trd_time<time(9,0,0)
              ,wd.trd_time<time(11,30,0)
              ,wd.trd_time<time(13,30,0)
              ,wd.trd_time<time(15,0,0)
              ,wd.trd_time<time(21,0,0)
              ,wd.trd_time<=time(23,0,0)]
    cho_intra=[4/6
               ,2/6
               ,1.5/6
               ,0.5/6
               ,0
               ,5/6]
    wd['intra_hours']=np.select(cdt_time,cho_intra,0)
    wd['t']=wd['intra_hours']+wd['exp_idx']-wd['trd_idx']
    print("Format New ttm time with = ", datetime.now() - tic, "s")
    
    
    spot_price=rqd.get_price(wd.underlying,start_date,end_date,freq,'close').reset_index()     
    spot_price.rename(columns={'order_book_id':'underlying'},inplace=True)
    wd=pd.merge(wd,spot_price,how='outer',on=['underlying','datetime'],suffixes=('_option','_underlying'))
    wd.dropna(inplace=True)
    
    tic=datetime.now()
    wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.close_option,wd.shape[0])
    print("Runing IV time = ", datetime.now() - tic, "s")
    # wd['delta']=getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.iv)*100
     
    wd['moneyness']=(wd.strike/wd.close_underlying).round(2)
    
    # wd['delta_round']=wd.delta.round(0)
    
    wd['iv_volume']=wd['iv']*wd['volume']
    # dfiv_delta=pd.DataFrame()
    # for given_delta in [-20,50,20]:
    #     wd_delta=wd.where((wd.delta_round>=given_delta-5)&(wd.delta<=given_delta+5)).dropna()
    #     iv_delta=wd_delta.groupby('datetime')['iv'].mean() if start_date==end_date else wd_delta.groupby('trading_date')['iv_volume'].sum()/wd_delta.groupby('trading_date')['volume'].sum()
    #     dfiv_delta[given_delta]=iv_delta.round(4)*100
    
    dfiv_moneyness=pd.DataFrame()
    for given_m in moneyness:
        wd_m=wd.where((wd['moneyness']>=given_m-0.01)&(wd['moneyness']<=given_m+0.01)).dropna()
        if given_m<1:
            wd_m.drop(wd_m[wd_m.optiontype=='C'].index.tolist(),inplace=True)
        elif given_m>1:
            wd_m.drop(wd_m[wd_m.optiontype=='P'].index.tolist(),inplace=True)
        else:
            pass
        iv_delta=wd_m.groupby('datetime')['iv'].mean() if start_date==end_date else wd_m.groupby('trading_date')['iv_volume'].sum()/wd_m.groupby('trading_date')['volume'].sum()
        dfiv_moneyness[given_m]=iv_delta.round(4)*100
    return dfiv_moneyness.values

          

def getdata(ass_num_idx,underlyingCode):
    
    if stm.session_state[ass_num_idx]['underlyingCode']!=getRQcode(underlyingCode):
        # stm.session_state['latest_list']=[]
        stm.session_state[ass_num_idx]['underlyingCode']=getRQcode(underlyingCode)
        stm.session_state[ass_num_idx]['cur_iv']=0
        # stm.session_state['cur_time']=0
        stm.session_state[ass_num_idx]['hist_iv']=getHistIV(stm.session_state[ass_num_idx]['underlyingCode'],cur_trd_date,cur_trd_date)
    else:
        stm.session_state[ass_num_idx]['cur_iv']=getLatestIV(stm.session_state[ass_num_idx]['underlyingCode'])
        stm.session_state[ass_num_idx]['hist_iv']=np.append(stm.session_state[ass_num_idx]['hist_iv'],stm.session_state[ass_num_idx]['cur_iv'],axis=0)



def update(t):
    # t=date_time.time()
    cdt=[t<time(9,0,30)#f
         ,t<time(10,14,30)#t
         ,t<time(10,30,20)#f
         ,t<time(11,29,30)#t
         ,t<time(13,30,20)#f
         ,t<time(14,59,00)#t
         ,t<time(20,55,0)#f
         ,t<time(22,59,00)#t
         ]
    cho=[False,True,False,True,False,True,False,True]
    return np.select(cdt,cho,False)



def getfigure(x,y,title_name):
    
    fig = go.Figure()
    for idx,m in enumerate(moneyness):
        fig.add_trace(go.Scatter(x=x
                                  , y=y[:,idx]
                                  , mode='lines'
                                  ,name=str(m)
                                  ))
    # fig.update_layout(yaxis_range=[data.min(), data.max()])
    for idx,m in enumerate(moneyness):
        title_name+="("+str(m)+": "+str(round(y[-1][idx],2))+")  "
    fig.update_layout(title='IV: '+title_name
                      ,width=800
                      ,height=400
                      )
    
    fig.update_xaxes(tickangle=45)
    return fig
 


def main_during(ass_num_idx,underlyingCode):
    getdata(ass_num_idx,underlyingCode)
    fig=getfigure(x=np.arange(0,stm.session_state[ass_num_idx]['hist_iv'].shape[0])
                  # x=list(map(lambda t:str(t.time()),stm.session_state[ass_num_idx]['hist_iv'].index.tolist()))
                  , y=stm.session_state[ass_num_idx]['hist_iv']
                  # , title_name=underlyingCode+" "+str(stm.session_state[ass_num_idx]['cur_iv'])
                  ,title_name=underlyingCode+"  "
                  )
    stm.plotly_chart(fig)
        
def main_after(ass_num_idx,underlyingCode):
    if stm.session_state[ass_num_idx]['underlyingCode']!=getRQcode(underlyingCode):
        stm.session_state[ass_num_idx]['underlyingCode']=getRQcode(underlyingCode)
        stm.session_state[ass_num_idx]['cur_iv']=0
        # stm.session_state['cur_time']=0
        stm.session_state[ass_num_idx]['hist_iv']=getHistIV(stm.session_state[ass_num_idx]['underlyingCode'],cur_trd_date,cur_trd_date)
 
    fig=getfigure(x=np.arange(0,stm.session_state[ass_num_idx]['hist_iv'].shape[0])
                  # x=list(map(lambda t:str(t.time()),stm.session_state[ass_num_idx]['hist_iv'].index.tolist()))
                  , y=stm.session_state[ass_num_idx]['hist_iv']
                  , title_name=underlyingCode+"  ")
    # fig.update_layout(yaxis_range=[data.min(), data.max()])
    stm.plotly_chart(fig)
     
def saveIV():
    file_path=r'D:\chengyilin\ivtick'+'\/'+str(cur_trd_date)
    if os.path.exists(file_path):
        print("Exists!")
    else:
        os.makedirs(file_path, exist_ok=True)
        absolute_path = os.path.abspath(file_path)
        for ass in ass_lists:
            dfiv=pd.DataFrame(columns=moneyness,data=stm.session_state['ass_'+ass]['hist_iv'])
            dfiv.to_csv(absolute_path+'\/'+stm.session_state['ass_'+ass]['underlyingCode']+'.csv')


def getSimleIV(underlyingCode):
# underlyingCode="AG2412"
    option_list=rqd.options.get_contracts(underlyingCode)
    opt_bid=list(map(lambda x:rqd.current_snapshot(x)['bids'][0],option_list))
    opt_ask=list(map(lambda x:rqd.current_snapshot(x)['asks'][0],option_list))
    # wd.drop(index=wd[wd.volume==0].index,inplace=True)
    # wd['datetime']=pd.to_datetime(wd.datetime)
    # wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
    option_instruments=rqd.instruments(option_list)
      
    strike_list=list(map(lambda x:x.strike_price,option_instruments))
    optiontype_list=list(map(lambda x:x.option_type,option_instruments))
    expire_date_list=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
    # wd['expire_date']=np.where(wd.expire_date==datetime(2024,2,13).date(),datetime(2024,2,7).date(),wd.expire_date)
    
    # underlying_list=list(map(lambda x:x.underlying_order_book_id,option_instruments))
    # print("Format time with = ", datetime.now() - tic, "s")
    # tic=datetime.now()
    # wd=pd.DataFrame()
    
    # trading_date_list=rqd.get_trading_dates("2024-01-01","2025-12-31")
    exp_idx=stm.session_state['trading_list'].index(expire_date_list[0])
    trd_idx=stm.session_state['trading_list'].index(cur_trd_date)
    trd_time=datetime.now().time()
    # wd['exp_idx']=(lambda x:trading_date_list.index(x))
    # wd['trd_idx']=wd.trading_date.apply(lambda x:trading_date_list.index(x))
    # wd['trd_time']=wd.datetime.apply(lambda x:x.time())
    cdt_time=[trd_time<time(9,0,0)
              ,trd_time<time(11,30,0)
              ,trd_time<time(13,30,0)
              ,trd_time<time(15,0,0)
              ,trd_time<time(21,0,0)
              ,trd_time<=time(23,0,0)]
    cho_intra=[4/6
               ,2/6
               ,1.5/6
               ,0.5/6
               ,0
               ,5/6]
    intra_hours=np.select(cdt_time,cho_intra,0)
    t=intra_hours+exp_idx-trd_idx
    
    s_ts=np.ones(shape=(len(optiontype_list)))*rqd.current_snapshot(underlyingCode).last
    
    
    
    wd=pd.DataFrame(columns=['strike','opttype'])
    wd['strike']=strike_list
    wd['opttype']=optiontype_list
    wd['iv_bid']=getIVArr(np.array(optiontype_list)
             , s_ts
             , np.array(strike_list)
             , np.ones(shape=(len(optiontype_list)))*t/252
             , np.array(opt_bid), 1)*100
    wd['iv_ask']=getIVArr(np.array(optiontype_list)
             , s_ts
             , np.array(strike_list)
             , np.ones(shape=(len(optiontype_list)))*t/252
             , np.array(opt_ask), 1)*100
    return wd
# wd_c=wd[wd['opttype']=='C']
# wd_c=wd[wd['opttype']=='C']

def fig_simle(strike_ts,given_vol,title_name):
     # fig=getfigure(x=strike_ts
     #               # x=list(map(lambda t:str(t.time()),stm.session_state[ass_num_idx]['hist_iv'].index.tolist()))
     #               , y=given_vol
     #               # , title_name=underlyingCode+" "+str(stm.session_state[ass_num_idx]['cur_iv'])
     #                ,title_name=underlyingCode+"  "
     #               )
     colors=dict({'ask':'red','bid':'blue'})
     fig = go.Figure()
     for idx,m in enumerate(['bid','ask']):
         fig.add_trace(go.Scatter(x=[str(k) for k in strike_ts]
                                   , y=given_vol[:,idx]
                                   , mode='lines'
                                   ,name=str(m)
                                   ,line=dict(color=colors.get(idx))
                                   ))
     # fig.update_layout(yaxis_range=[data.min(), data.max()])
     # for idx,m in enumerate(['bid','ask']):
         # title_name+="("+str(m)+": "+str(round(y[-1][idx],2))+")  "
       
     fig.update_layout(title='Smile IV: '+title_name
                       ,width=600
                       ,height=400
                       )
     
     fig.update_xaxes(tickangle=45)
     stm.plotly_chart(fig)


stm.set_page_config(page_title="Plot of ATM IV",layout='wide') 
stm.session_state

if 'trading_list' not in stm.session_state.keys():
    stm.session_state['trading_list']=rqd.get_trading_dates('2024-01-01','2025-12-31')
    # trading_list=rqd.get_trading_dates('2024-01-01','2025-12-31')
    stm.write(stm.session_state['trading_list'])
    
cur_trd_date=stm.date_input("Current Trade Date:",value=rqd.get_future_latest_trading_date())
moneyness=stm.text_input("Moneyness:",value="0.95,1,1.05")
moneyness=list(map(lambda m:float(m),moneyness.split(",")))

#----------------------------------------------------------------------------------------------------
asset_lists="AU2412,AU2502,AG2412,AG2502,CF501,CF505,MA412,MA501,EG2412,EG2501,EB2412,V2412,V2501,PP2412,PP2501,L2501,SH501,AL2412"
asset_lists_input=stm.text_input("Underlyings:",)

if asset_lists==asset_lists_input or asset_lists_input=="":
    pass
else:
    asset_lists=asset_lists_input
stm.write(asset_lists)

ass_lists=list(map(lambda x:getRQcode(x),asset_lists.split(",")))

ass_num=len(ass_lists)
stm.write(str(ass_num)+" underlyings")


if 'container' not in stm.session_state:
    stm.session_state.container = []
    stm.session_state['row_num'] = 0

for ass in ass_lists:
    if 'ass_'+ass not in stm.session_state:
        stm.session_state['ass_'+ass]={'underlyingCode':""
                                    ,'cur_iv':0
                                    ,'hist_iv':[]}
        row=stm.container()
        stm.session_state.container.append({'row':row,'ass_idx':'ass_'+ass})
        stm.session_state['row_num']+=1




if update(datetime.now().time()):
    print('During:'+str(datetime.now()))
    # stm.write(datetime.now())
    for i in range(stm.session_state['row_num']):
        col11, col12= stm.columns(2)
        with col11:
            main_during(stm.session_state.container[i]['ass_idx']
            ,stm.session_state.container[i]['ass_idx'][4:])
        with col12:
            wd=getSimleIV(stm.session_state[stm.session_state.container[i]['ass_idx']]['underlyingCode'])
            fig_simle(wd[wd['opttype']=='C'].strike.values
                      ,wd[wd['opttype']=='C'][['iv_bid','iv_ask']].values
                      ,stm.session_state[stm.session_state.container[i]['ass_idx']]['underlyingCode'])
  
    sys_time.sleep(10)
    stm.rerun()
elif datetime.now().time()>=time(15,0,0) and datetime.now().time()<time(21,0,0):
    for ass in ass_lists:
        main_after('ass_'+ass,ass)
    # saveIV()
elif datetime.now().time()>time(23,0,0) or datetime.now().time()<time(8,0,0):
    for ass in ass_lists:
        main_after('ass_'+ass,ass)
    sys_time.sleep(3600)
    print('Before 8 AM:'+str(datetime.now()))
    stm.rerun()
elif datetime.now().time()<time(9,1,0):
    print('Before 9 AM:'+str(datetime.now()))
    for ass in ass_lists:
        main_after('ass_'+ass,ass)
    sys_time.sleep(60)
    stm.rerun()
else:
    for ass in ass_lists:
        main_after('ass_'+ass,ass)
    sys_time.sleep(60)
    print('Break:'+str(datetime.now()))
    stm.rerun()

    
    
    