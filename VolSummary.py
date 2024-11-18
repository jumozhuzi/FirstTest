# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:29:24 2024

@author: dzrh
"""

import streamlit as stm
import xlwings as xw
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# import tushare as tu
from datetime import timedelta,datetime,time
# import time
# import iFinDPy as fd
# import os
from CYL.OptionPricing import BSM,calIV,calTradttm,LinearInterpVol
# from CYL.YieldChainAPI import YieldChainAPI 
# from CYL.StressTest import StressTest
from CYL.StressTestNew_api import StressTestNew,getRQcode
import plotly.graph_objs as go       
# from CYL.pythonAPI_pyfunctions4newDll_3 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer
# from CYL.StressTestNew import LinearInterpVol 
# from CYL.AccCalculator import CalAcc 
import rqdatac as rqd
# rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")


# # @xw.func
# def LogIn():
#     result=fd.THS_iFinDLogin("Dzqh165", "CYLcyl0208")
#     if result==0:
#         print("登录成功")
#         # return "登录成功"
#     elif result==-201:
#         print("重复登录")
#     elif result==-2:
#         print("用户名或密码错误")
#     else:
#         print("未知错误")
    
# LogIn()
rf=0.03
q=0
annual_coeff=252

def getRule():
    loc=r'D:\chengyilin\work\system\OptionStrikeRule.xlsx'
    dfrule=pd.read_excel(loc,index_col=0)
    dfrule.insert(loc=3,column='Level4',value=dfrule['Level3']*5)
    dfrule['Level3'].fillna(5*dfrule['Level2'],inplace=True)
    dfrule['Level4'].fillna(dfrule['Level3'],inplace=True)
    dfrule['Dk4'].fillna(dfrule['Dk3'],inplace=True)
    
    # dfrule.insert(loc=2,column='Level3',value=dfrule['Level2']*5)
  
    return dfrule


def calCalttm(expire_date):
    '''
    Return Calendar Year Days

    Parameters
    ----------
    expire_date : str 
        yyyy-mm-dd.
   
        
    Returns
    ------
    calttm: float
    '''
    calttm=(pd.to_datetime(expire_date+" 15:00:00")-datetime.now())/np.timedelta64(1,'D')
 
    return calttm





def findMultiplier(optioncode):
    '''
    Find multiplier of option contract

    Parameters
    ----------
    optioncode : TYPE
        DESCRIPTION.

    Returns
    -------
    multiplier : float

    '''
    multiplier=rqd.instruments(optioncode).contract_multiplier
    return multiplier

def findExpireDate(optioncode):
    '''
    

    Parameters
    ----------
    optioncode : TYPE
        DESCRIPTION.

    Returns
    -------
    expire_date : str

    '''

    expire_date=rqd.instruments(optioncode).de_listed_date
    return expire_date

def findConnaction(dfrule,code):
    try :
        if np.isnan(dfrule.loc[code][-1]):
            con=""
    except:
        con=dfrule.loc[code][-1]
    return con

# @xw.func
def findLatestPrice(underlying):
    if type(underlying)==list:
        return list(map(lambda x:x.last,rqd.current_snapshot(underlying)))
    else:
        if underlying[:2]=="IO":
            #000300
          return  rqd.current_snapshot("IF"+underlying[2:]).last
        elif underlying[:2]=="MO":
            #中证1000
           return     rqd.current_snapshot("IM"+underlying[2:]).last
        elif underlying[:2]=="HO":
            #上证50
            return    rqd.current_snapshot("IH"+underlying[2:]).last
        else:
            return rqd.current_snapshot(underlying).last



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
    sel_k=k_ts[atm_idx-rng:atm_idx+rng]
    # sel_k=k_ts[atm_idx:atm_idx+rng+1]
    return sel_k

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

def formatT(wd,underlying,expire_date):
    """
    

    Parameters
    ----------
    wd : TYPE
        DESCRIPTION.
    expire_date: str
              
    Returns
    -------
    wdT : dataframe
        

    """
    # wd=calIVandGreeks(underlying,rng=15,annual_coff=365)
    wdc=wd.groupby('optiontype').get_group('C')
    wdp=wd.groupby('optiontype').get_group('P')
    sel_col=['vega','gamma','theta','delta','askvol','bidvol']
    wdT=pd.merge(wdc[sel_col+['strike']]
                 ,wdp[sel_col[::-1]+['strike']]
                 ,on='strike'
                 ,suffixes=('_Call','_Put'))
    wdT.rename(columns={'strike':underlying[:-4]+expire_date},inplace=True)
    return wdT

def getOptionCodes(underlying,s,rng):
      dfrule=getRule()
      code=findCode(underlying)
      rule=dfrule.loc[code].values[:-1].reshape((2,4))  
      k_ts=find_k(rule,s,rng)
      # con=findConnaction(dfrule, code)
      con=""
      option_code_call=formatOptionCode(underlying, k_ts, 'C', con)
      option_code_put=formatOptionCode(underlying, k_ts, 'P', con)
      option_code=option_code_call+option_code_put
      return option_code
  
    
# @xw.func
def RQcalIVandGreeks(underlying,annual_coeff,rng=15):
    '''
    

    Parameters
    ----------
    dfrule : dataframe
        strike rule of underlying.
    underlying : str
        underlying code.
    code : str
        variety type.
    rng : int
        arange of strikes .
    annual_coeff: int
                 yearly coefficient
    Returns
    -------
    wd : dataframe.

    '''
    # tic = time.time()
    # dfrule=getRule()
    # code=findCode(underlying)
    
    # rule=dfrule.loc[code].values[:-1].reshape((2,4))  
    # con=findConnaction(dfrule, code)
    last_p=findLatestPrice(underlying)
    # last_p=findLatestPrice("IF2304")
    # k_ts=find_k(rule,last_p,rng)
    # con=""
    # option_code_call=formatOptionCode(underlying, k_ts, 'C', con)
    # option_code_put=formatOptionCode(underlying, k_ts, 'P', con)
    # option_code=option_code_call+option_code_put
    
    option_code=getOptionCodes(underlying, last_p,rng)
    option_snap=rqd.current_snapshot(option_code)
    # print("Running time = ", time.time() - tic, "s")
    

    wd=pd.DataFrame()
    wd['ask1']=list(map(lambda snap:snap.asks[0],option_snap))
    wd['bid1']=list(map(lambda snap:snap.bids[0],option_snap))
    wd['code']=list(map(lambda x:x.order_book_id,option_snap))
    wd['strike']=list(map(lambda x:x.strike_price,rqd.instruments(wd.code)))
    wd['optiontype']=list(map(lambda x:x.option_type,rqd.instruments(wd.code)))

    
    multiplier=findMultiplier(wd.code[0])
    expire_date=findExpireDate(wd.code[0])
 
    
    # annual_coff=365
    trading_dates=rqd.get_trading_dates(datetime.today(),datetime.today()+timedelta(days=365))
    if annual_coeff==365:
        t=calCalttm(trading_dates,datetime.now(),expire_date)/annual_coeff
    else:
        t=calTradttm(trading_dates,datetime.now(),datetime.strptime(expire_date,"%Y-%m-%d").date())/annual_coeff
    #calculat iv
    wd['bidvol']=list(map(lambda opttype,k,target:calIV(opttype, last_p, k, t, rf,q, target)
         ,wd.optiontype.values
         ,wd.strike.values
         ,wd.bid1))
    wd['askvol']=list(map(lambda opttype,k,target:calIV(opttype, last_p, k, t, rf,q, target)
         ,wd.optiontype.values
         ,wd.strike.values
         ,wd.ask1))
    
    #delta%
    wd['delta']=list(map(lambda k,sigma,opttype:BSM(last_p,k,t,rf,q,sigma,opttype).delta()*100
             ,wd.strike.values
             ,wd[['bidvol','askvol']].mean(axis=1)
             ,wd.optiontype.values))
    #每手总量对应的theta
    wd['theta']=list(map(lambda k,sigma,opttype:(BSM(last_p,k,t,rf,q,sigma,opttype).theta(1/365))*multiplier
             ,wd.strike.values
             ,wd[['bidvol','askvol']].mean(axis=1)
             ,wd.optiontype.values))
    #gamma手数（没有乘以multiplier，即为了直接乘以成交手数得出gamma手数）
    wd['gamma']=list(map(lambda k,sigma,opttype:(BSM(last_p,k,t,rf,q,sigma,opttype).gamma())
             ,wd.strike.values
             ,wd[['bidvol','askvol']].mean(axis=1)
             ,wd.optiontype.values))
    #每手总量对应的vega
    wd['vega']=list(map(lambda k,sigma,opttype:(BSM(last_p,k,t,rf,q,sigma+0.01,opttype).price()-BSM(last_p,k,t,rf,q,sigma,opttype).price())*multiplier
              ,wd.strike.values
              ,wd[['bidvol','askvol']].mean(axis=1)
              ,wd.optiontype.values))
    
    
    wdT=formatT(wd,underlying,expire_date)
    wdT[['askvol_Put','askvol_Call','bidvol_Put','bidvol_Call']]=100*wdT[['askvol_Put','askvol_Call','bidvol_Put','bidvol_Call']]
    midvol=wdT[['askvol_Put','askvol_Call','bidvol_Put','bidvol_Call']].mean(axis=1)
    midvol.index=wdT.delta_Call
    moneyness_ts=np.log(wdT[wdT.columns[6]]/last_p)
    delta_ts=np.where(moneyness_ts>=0,wdT.delta_Call,wdT.delta_Put)
    
    # fig=plt.figure(figsize=(15,12))
    fs=8
    fig=plt.figure(figsize=(5,3))
    ax=fig.add_subplot(111)
    ax.plot(moneyness_ts,wdT.bidvol_Call,marker='o',ms=1,linewidth=1,label='call_bid')
    ax.plot(moneyness_ts,wdT.bidvol_Put,marker='o',ms=1,linewidth=1,label='put_bid')
    ax.plot(moneyness_ts,midvol,marker='o',ms=1,linewidth=1,label='midvol')
    ax.grid(True)
    ax.set_xbound(lower=-0.12,upper=0.12)
    idx_atm=wdT.where(wdT.iloc[:,6]>=last_p).dropna().index[0]
    yb_lower=wdT.loc[idx_atm,['askvol_Call','bidvol_Call','askvol_Put','bidvol_Put']].max()*0.5
    yb_upper=wdT.loc[idx_atm,['askvol_Call','bidvol_Call','askvol_Put','bidvol_Put']].max()*1.5
    ax.set_ybound(lower=yb_lower,upper=yb_upper)
    ax.set_xlabel('Moneyness(log(K/S))',fontsize=fs)
    ax.set_ylabel('IV(%)',fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs,loc='upper left')
    ax.set_title(underlying,fontsize=fs)
    ax2=ax.twinx()
    ax2.plot(moneyness_ts,delta_ts,linestyle='--',linewidth=3,color='black',label='delta')
    ax2.legend(fontsize=15,loc='upper right')
    ax2.set_xbound(lower=-0.12,upper=0.12)
    ax2.set_ybound(lower=-50,upper=50)
    ax2.set_ylabel('Delta(%)',fontsize=fs)
    ax2.tick_params(labelsize=fs)
    # stm.pyplot(fig)
  
    return wdT ,fig


def formatOptionSnap(option_code):
    '''
    Format Dataframe of option list

    Parameters
    ----------
    option_code : list
        list of options.

    Returns
    -------
    wd : dataframe
    '''
    option_snap=rqd.current_snapshot(option_code)
    wd=pd.DataFrame()
    wd['ask1']=list(map(lambda snap:snap.asks[0],option_snap))
    wd['bid1']=list(map(lambda snap:snap.bids[0],option_snap))
    wd['code']=list(map(lambda x:x.order_book_id,option_snap))
    return wd

# @xw.func
def RQcalBidAskVol(underlyings,given_delta,given_t,annual_coeff,rng=15,short_adjvol=0,long_adjvol=0,show_expiredate=True):
    # LogIn()
    
    dfrule=getRule()
    code=findCode(underlyings[0])
    rule=dfrule.loc[code].values[:-1].reshape((2,4))  
    con=""

    
    last_p=findLatestPrice(underlyings)
    k_ts=list(map(lambda x:find_k(rule,x,rng),last_p))
    
    optiontype='C' if given_delta>0 else 'P'
    option_code=list(map(lambda underlying,k:formatOptionCode(underlying, k, optiontype, con),underlyings,k_ts))
    
    wd=list(map(lambda x:formatOptionSnap(x),option_code))
    

    k_ts=[[]]
    k_ts[0]=list(map(lambda x:x.strike_price,rqd.instruments(option_code[0])))
    k_ts.append(list(map(lambda x:x.strike_price,rqd.instruments(option_code[1]))))
    
    
    expire_date=list(map(lambda x:findExpireDate(x.code[0]),wd))   

    trading_dates=rqd.get_trading_dates(datetime.today(),datetime.today()+timedelta(days=365))
    
    if annual_coeff==365:
        t_days=list(map(lambda x:calCalttm(x),expire_date))
    else:
        t_days=list(map(lambda x:calTradttm(trading_dates,datetime.now(),datetime.strptime(x,"%Y-%m-%d").date()),expire_date))
    
    t=np.array(t_days)/annual_coeff
    #calculat iv
    wd[0]['bidvol']=list(map(lambda k,target:calIV(optiontype, last_p[0], k, t[0], rf,q, target)
          ,k_ts[0]
          ,wd[0].bid1))
    wd[1]['bidvol']=list(map(lambda k,target:calIV(optiontype, last_p[1], k, t[1], rf,q, target)
          ,k_ts[1]
          ,wd[1].bid1))
    
    wd[0]['askvol']=list(map(lambda k,target:calIV(optiontype, last_p[0], k, t[0],rf,q, target)
          ,k_ts[0]
          ,wd[0].ask1))
    wd[1]['askvol']=list(map(lambda k,target:calIV(optiontype, last_p[1], k, t[1], rf,q, target)
          ,k_ts[1]
          ,wd[1].ask1))
    
    #delta%
    wd[0]['delta']=list(map(lambda k,sigma:BSM(last_p[0],k,t[0],rf,q,sigma,optiontype).delta()*100
              ,k_ts[0]
              ,wd[0][['bidvol','askvol']].mean(axis=1)
              ))
    wd[1]['delta']=list(map(lambda k,sigma:BSM(last_p[1],k,t[1],rf,q,sigma,optiontype).delta()*100
              ,k_ts[1]
              ,wd[1][['bidvol','askvol']].mean(axis=1)
              ))
    idx_short=np.where(wd[0].delta>given_delta)[0][-1]
    idx_long=np.where(wd[1].delta>given_delta)[0][-1]
    
    vol_short=wd[0].loc[idx_short,['bidvol','askvol']]+short_adjvol    
    vol_long=wd[1].loc[idx_long,['bidvol','askvol']]+long_adjvol
    
    
    vol=pd.DataFrame(index=underlyings,columns=['bidvol','askvol'])
    vol.loc[underlyings[0]]=vol_short.values
    vol.loc[underlyings[1]]=vol_long.values
    # vol['midvol']=vol[['bidvol','askvol']].mean(axis=1)
 
    findvol=vol_short+(vol_long-vol_short)/(t_days[-1]-t_days[0])*(given_t-t_days[0])
    res=pd.concat([vol,pd.DataFrame(findvol).T])
    res.index=underlyings+['ResultVol']
    res['midvol']=res[['bidvol','askvol']].mean(axis=1)
    res=res*100
    t_days.append(given_t)
    res['TradingTTM']=t_days
    if show_expiredate:
        res['ExpiredDate']=expire_date+[""]
        return res
    else:
         return res

    
  
# @xw.func
# @xw.arg('rolling_days',range)
def RQcalRealisedVol(underlying,maxret_period,end_date,annual_coeff,period="FULL",fig_flag="SHOW"):
    '''
    Calculate calendar-year-volatility. 

    Parameters
    ----------
    underlying : TYPE
        DESCRIPTION.
    sheets_name : TYPE
        DESCRIPTION.
    cell_address: str
        the address of the picture showed. eg "A1"
    fullperiod : TYPE, optional
        DESCRIPTION. The default is True.
    showfig : TYPE, optional
        DESCRIPTION. The default is False.
    annual_coff : TYPE, optional
        DESCRIPTION. The default is 365.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    maxret_period=int(maxret_period)
    rolling_days=[10,21,42,63]
    # end_date=(datetime.today().date()).strftime("%Y-%m-%d")
    start_date=(end_date-timedelta(days=365)).strftime("%Y-%m-%d")
    

    #----------------------------------------------RQDATA
    wd=rqd.get_price(underlying,start_date,end_date)[['open','high','low','close']]
    wd['time']=list(map(lambda x:x[-1].date(),wd.index))
    wdtoday=pd.DataFrame(index=[underlying])    
    snap=rqd.current_snapshot(underlying)
    wdtoday[['latest','high','low']]=snap.last,snap.high,snap.low
    
    
 
    if datetime.now().time()>time(21,0,0):        
          # current_trd=fd.THS_Date_Offset('212001','dateType:0,period:D,offset:1,dateFormat:0,output:singledate',end_date).data
          # time_ts=pd.concat([wd.time,pd.Series(pd.to_datetime(current_trd).date())],ignore_index=True)
          time_ts=pd.concat([wd.time,pd.Series(datetime.today().date()+timedelta(days=1))],ignore_index=True)

          close_ts=pd.concat([wd.close,pd.Series(wdtoday.loc[underlying].latest)],ignore_index=True)
          high_ts=pd.concat([wd.high,pd.Series(wdtoday.loc[underlying].high)],ignore_index=True)
          low_ts=pd.concat([wd.low,pd.Series(wdtoday.loc[underlying].low)],ignore_index=True)
    elif datetime.now().time()>time(15,0,0):
          close_ts,high_ts,low_ts=wd.close,wd.high,wd.low
          time_ts=wd.time
    else:
          time_ts=pd.concat([wd.time,pd.Series(datetime.today().date())],ignore_index=True)
          close_ts=pd.concat([wd.close,pd.Series(wdtoday.loc[underlying].latest)],ignore_index=True)
          high_ts=pd.concat([wd.high,pd.Series(wdtoday.loc[underlying].high)],ignore_index=True)
          low_ts=pd.concat([wd.low,pd.Series(wdtoday.loc[underlying].low)],ignore_index=True)
    
    
    # max high-low to close
    max_abs_idx=np.argmax([np.abs(high_ts/close_ts.shift(1)-1)
                               ,np.abs(low_ts/close_ts.shift(1)-1)
                               ,np.abs(close_ts/close_ts.shift(1)-1)],axis=0)
    ret_matrix=np.array([(high_ts/close_ts.shift(1)-1),(low_ts/close_ts.shift(1)-1),(close_ts/close_ts.shift(1)-1)]).T
    max_ret_ts=np.array([ret_matrix[i,c] for i,c in enumerate(max_abs_idx)])

    # max_ret_ts=np.max([np.abs(high_ts/close_ts.shift(1)-1),np.abs(low_ts/close_ts.shift(1)-1),np.abs(close_ts/close_ts.shift(1)-1)],axis=0)

    #close to close
    if annual_coeff==365:
        ret_ts=(close_ts/close_ts.shift(1)-1)/(time_ts-time_ts.shift(1)).apply(lambda x:np.sqrt(x.days))
        max_ret_ts=(max_ret_ts/(time_ts-time_ts.shift(1)).apply(lambda x:np.sqrt(x.days)))*np.sqrt(annual_coeff)*100
    else:
        ret_ts=close_ts/close_ts.shift(1)-1
        # max_ret_ts=pd.Series(max_ret_ts*np.sqrt(annual_coeff)*100)
        max_ret_ts=pd.Series(max_ret_ts)


    # close-to-close vol    
    dfvol_std=pd.DataFrame(map(lambda days:ret_ts.rolling(days).std()*np.sqrt(annual_coeff)*100,rolling_days)).T
    dfvol_std.columns=rolling_days
    dfvol_std.index=time_ts
    dfvol_std.sort_index(ascending=False,inplace=True)
    
    dfvol_mean=pd.DataFrame(map(lambda days:(ret_ts.abs()*np.sqrt(annual_coeff)*100).rolling(days).mean(),rolling_days)).T
    dfvol_mean.columns=rolling_days
    dfvol_mean.index=time_ts
    dfvol_mean.sort_index(ascending=False,inplace=True)
    
    # max high-low to close vol
    # dfvol_max=pd.DataFrame(map(lambda days:(max_ret_ts).rolling(days).mean(),rolling_days)).T
    dfvol_max=pd.DataFrame(map(lambda days:(max_ret_ts).rolling(days).std()*np.sqrt(annual_coeff)*100,rolling_days)).T

    dfvol_max.columns=rolling_days
    dfvol_max.index=time_ts
    dfvol_max.sort_index(ascending=False,inplace=True)
    
 
    dfvol=pd.merge(dfvol_std, dfvol_mean,how='inner',on=dfvol_std.index,suffixes=('Std','Mean')).drop(columns='key_0')
    dfvol.index=dfvol_max.index
    dfvol=pd.merge(dfvol,dfvol_max,how='inner',on=dfvol.index).drop(columns='key_0')
    dfvol.columns=dfvol.columns.tolist()[:8]+[str(col)+'Max' for col in dfvol.columns[-4:].tolist()]
    if period.upper()=="FULL":
        res=dfvol.describe().iloc[1:,:].T
    else:
        res=dfvol.iloc[:100,:].describe().iloc[1:,:].T
    res.insert(0,'Latest',dfvol.iloc[0,:])
    res=(res.T).round(2)


    return res,dfvol_std

def Plt_RQcalRealisedVol(underlying,dfvol_std):
    fig=plt.figure(figsize=(5,2))
    ax=fig.add_subplot(111)
    for col in dfvol_std.columns:
        ax.plot(dfvol_std[col],label=col,)
    ax.legend(fontsize=5) 
    ax.tick_params(labelsize=5)
    plt.title(label=underlying+" Std Vol",fontsize=5)
    stm.pyplot(fig)
    
  


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

def getBSMPriceArr(opttpye_ts,s_ts,k_ts,t_ts,iv_ts):
    b=0    
    d1=(np.log(s_ts/k_ts)+t_ts*0.5*iv_ts**2)/(iv_ts*np.sqrt(t_ts))
    d2=d1-iv_ts*np.sqrt(t_ts)
    
    s_dis=s_ts*np.exp((b-rf)*t_ts)
    k_dis=k_ts*np.exp(-1*rf*t_ts)
    
    c_p=s_dis*norm.cdf(d1)-k_dis*norm.cdf(d2)
    p=c_p+np.where(opttpye_ts=='C',0,1)*(k_dis-s_dis)
    return p

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


# @xw.func
def getSurface(varity,trd_date,delta_rng=[-10,-20,-30,-40,50,40,30,20,10]):
      '''
        Get weighted average implied volatility surface with volumes as the weights for a given trade day.

        Parameters
        ----------
        varity :str
            eg 'au'
        trd_date : str
        
        Returns
        -------
        surface : dataframe
        iv surface

      '''
      trading_date_list=rqd.get_trading_dates('2019-01-01','2026-12-31')
      # varity='I'
      varity=varity.upper()
      freq='10m'
       # trd_date="2023-07-07"
      option_id_list=rqd.options.get_contracts(varity,trading_date=trd_date)
      
      if option_id_list==[]:
          print('No Listed Contracts')
          return []
      else:
            wd=rqd.get_price(option_id_list,trd_date,trd_date,freq,fields=['close','volume','open_interest','trading_date']).reset_index()
            # wd.drop(index=wd[wd.volume==0].index,inplace=True)
            wd['datetime']=pd.to_datetime(wd.datetime)
            wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
            option_instruments=rqd.instruments(wd.order_book_id)
               
            wd['strike']=list(map(lambda x:x.strike_price,option_instruments))
            wd['optiontype']=list(map(lambda x:x.option_type,option_instruments))
            wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
            wd['underlying']=list(map(lambda x:x.underlying_order_book_id,option_instruments))

            max_volume=wd.groupby('expire_date')['volume'].sum().max()
            drop_exp_date=wd.groupby('expire_date')['volume'].sum()[(wd.groupby('expire_date')['volume'].sum()<max_volume*0.5/100)].index
            wd.index=wd.expire_date
            wd.drop(drop_exp_date,inplace=True)
            
            tic=datetime.now()
            wd['exp_idx']=wd.expire_date.apply(lambda x:trading_date_list.index(x))
            wd['trd_idx']=wd.trading_date.apply(lambda x:trading_date_list.index(x))
            wd['trd_time']=wd.datetime.apply(lambda x:x.time())
            wd['intra_hours']=np.select([wd.trd_time<time(9,0,0)
                                      ,wd.trd_time<time(11,30,0)
                                      ,wd.trd_time<time(13,30,0)
                                      ,wd.trd_time<time(15,0,0)
                                      ,wd.trd_time==time(15,0,0)
                                      ,wd.trd_time<=time(23,0,0)]
                                        ,[4/6
                                        ,2/6
                                        ,1.5/6
                                        ,0.5/6
                                        ,0
                                        ,5/6],0)
            wd['t']=wd['intra_hours']+wd['exp_idx']-wd['trd_idx']
            print("Format New ttm time with = ", datetime.now() - tic, "s")
    
            # wd['t']=list(map(lambda trd_t,exp_d:calTradttm(trading_date_list, trd_t, exp_d),wd.datetime,wd.expire_date))
            # wd.drop(index=wd[wd.t<=3].index,inplace=True)
             
            spot_price=rqd.get_price(wd.underlying,trd_date,trd_date,freq,'close').reset_index()     
            spot_price.rename(columns={'order_book_id':'underlying'},inplace=True)
            wd=pd.merge(wd,spot_price,how='outer',on=['underlying','datetime'],suffixes=('_option','_underlying'))
            wd.dropna(inplace=True)
            # wd['t']=list(map(lambda trd_d,exp_d:trading_date_list.index(exp_d)-trading_date_list.index(trd_d)+0.5,wd.trading_date,wd.expire_date))
             
            
            # tic = time.time()
            wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/annual_coeff,wd.close_option,wd.shape[0])
    
            # wd['iv']=list(map(lambda opttype, s, k, t, target:calIV(opttype, s, k, t, rf, q, target)
            #                   ,wd.optiontype, wd.close_underlying
            #                 , wd.strike, wd.t/annual_coeff
            #                 , wd.close_option))
    
            # print("Running time = ", time.time() - tic, "s")
    
            wd['delta']=getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/annual_coeff,wd.iv)*100
            
            
            surface=pd.DataFrame(index=wd.expire_date.drop_duplicates().sort_values().values,columns=delta_rng)
            for given_delta in delta_rng:
                # given_delta=-20
                sel_delta=wd.where((wd.delta>=given_delta-3)&(wd.delta<=given_delta+3)).dropna()
                sel_delta['iv_vol']=sel_delta.iv*sel_delta.volume
                surface[given_delta]=sel_delta.groupby('expire_date')['iv_vol'].sum()/sel_delta.groupby('expire_date')['volume'].sum()
                # surface[given_delta]=sel_delta.groupby('expire_date')['iv'].mean()*100
            surface.fillna(method='ffill',inplace=True)
            
            return surface*100
    
    # tic = time.time()
    # a=getSurface(varity, trd_date)
    # print("Running time = ", time.time() - tic, "s")
    # tic = time.time()
    # aa=getSurface_v2(varity, trd_date)
    # print("Running time = ", time.time() - tic, "s")


def getGivenDeltaIV(varity,trd_d,given_delta):
    if abs(given_delta)<=1:
        given_delta=given_delta*100
    
    option_id_list=rqd.options.get_contracts(varity,trading_date=trd_d)          
    if option_id_list==[]:
        return "" 
    else:
         wd=rqd.options.get_greeks(option_id_list,trd_d,trd_d).reset_index()
         wd.delta=wd.delta*100
         corresponding_iv=wd.where((wd.delta>=given_delta-3)&(wd.delta<=given_delta+3)).dropna()['iv'].mean()
         print(trd_d,given_delta)
         return  corresponding_iv
     
# def ReLoadIV(path,end_d):
#     # end_d='2023-07-26'
#     if end_d=="":
#         end_d=rqd.get_previous_trading_date(datetime.today().date(), 1)
#     file_list=os.listdir(path)
#     for file in file_list:
#         a=pd.read_csv(path+'\/'+file,index_col='Unnamed: 0')    
#         last_date=a.index.tolist()[-1]
#         start_d=rqd.get_next_trading_date(last_date, 1)
#         trd_date_list=rqd.get_trading_dates(start_d,end_d)
#         if trd_date_list==[]:
#             continue
#         dfiv=pd.DataFrame(index=trd_date_list,columns=a.columns)
#         for col in dfiv.columns[:-2]:
#             dfiv[col]=[getGivenDeltaIV(file[:-4],trd_d,float(col)) for trd_d in dfiv.index]
#         dfiv['close']=rqd.get_price(file[:-4].upper()+'99',start_d,end_d,'1d','close').reset_index()['close'].values
#         a=pd.concat([a,dfiv],axis=0)
#         a.to_csv(path+'\/'+file)
#     print('IV Reload Finished')
 
    
# @xw.func() 
def HistIVDescribe(sheets_name,cell_address,varity,delta_1,delta_2,given_delta=str(50),start_d="",end_d=""):
    '''

    Parameters
    ----------
    varity : TYPE
        DESCRIPTION.
    delta_1 : str
     
    delta_2 : str
       

    Returns
    -------
    None.

    '''
    varity=varity.upper()
    delta_1=str(delta_1)
    delta_2=str(delta_2)
    
    path=r'D:\chengyilin\ivdata'
    file_name=varity.upper()+'.CSV'
    dfiv=pd.read_csv(path+'\/'+file_name,index_col='trading_date')    
    dfiv['ret']=(dfiv.close/dfiv.close.shift(1)-1).rolling(21).std()*np.sqrt(annual_coeff)
    dfiv.index=pd.to_datetime(dfiv.index)
    dfiv.fillna(method='ffill',inplace=True)
    
    

    if start_d not in dfiv.index or start_d=="":
        start_d=dfiv.index[0]
    if end_d not in dfiv.index or end_d=="":
        end_d=dfiv.index[-1]


    close_ts=dfiv.loc[start_d:end_d,'close']
    skew_ts=(dfiv.loc[start_d:end_d,delta_1]-dfiv.loc[start_d:end_d,delta_2])*100
    
    fig=SkewGraph(varity, delta_1, delta_2, skew_ts, close_ts)
    # get_figure(varity, delta_1, delta_2, skew_ts, close_ts)

    # given_delta=str(20)
    
    givendelta_iv_ts=dfiv.loc[start_d:end_d,given_delta]*100
    atm_iv_ts=dfiv.loc[start_d:end_d,str(50)]*100
    ret_ts=dfiv.loc[start_d:end_d,'ret']*100
    
    
    fig2=plt.figure(figsize=(20,20))
    ax=fig2.add_subplot(111)
    ax.plot(givendelta_iv_ts,'r',linewidth=3,label=given_delta+' IV')
    ax.plot(atm_iv_ts,'b',linewidth=2.5,label='ATM IV')
    ax.plot(ret_ts,'k',linewidth=2.5,label='20D Vol')
    ax.tick_params(labelsize=30)
    plt.xticks(rotation=30)
    ax.set_ylabel('Volatility(%)',fontsize=30)
    ax.legend(fontsize=30)
    ax.grid()
    ax.set_title(label=varity+' '+given_delta+' Delta IV',fontsize=35)

    
    
    wb=xw.Book.caller()
    sht=wb.sheets(sheets_name)
    sht.pictures.add(fig,name='skew'
                      ,update=True
                      ,left=sht.range(cell_address).left
                      ,top=sht.range(cell_address).top
                      ,width=300
                      ,height=300
                      )
    sht.pictures.add(fig2,name='50 iv'
                      ,update=True
                      ,left=sht.range(cell_address).left
                      ,top=sht.range(cell_address).top
                      ,width=300
                      ,height=300
                      )
    
    return  ((dfiv.loc[start_d:end_d].describe()).drop(columns='close').drop(index='count'))*100


# @xw.func() 
def HistIVDescribeSurf(varity,dfiv,delta_1,delta_2,given_delta=str(50),start_d="",end_d=""):
    '''

    Parameters
    ----------
    varity : TYPE
        DESCRIPTION.
    delta_1 : str
     
    delta_2 : str
       

    Returns
    -------
    None.

    '''
    varity=varity.upper()
    delta_1=str(delta_1)
    delta_2=str(delta_2)
   
    # if sheets_name=="":
    dfiv.index=list(map(lambda x:datetime.strptime(x,"%Y-%m-%d").date(),dfiv.index))
    # else:
    #     dfiv.index=pd.to_datetime(dfiv.index)
    dfiv['ret']=((dfiv.close.groupby(dfiv.index).last()/dfiv.close.groupby(dfiv.index).last().shift(1)-1).rolling(21).std()*np.sqrt(annual_coeff)*100).loc[dfiv.index]
    dfiv.expire_date=dfiv.expire_date.apply(lambda x:pd.to_datetime(x).date())

    # dfiv['ret']=(dfiv.close/dfiv.close.shift(1)-1).rolling(21).std()*np.sqrt(annual_coeff)
    # dfiv.index=pd.to_datetime(dfiv.index)
    # dfiv.fillna(method='ffill',inplace=True)
    # dfiv['exp']
    

    if start_d not in dfiv.index or start_d=="":
        start_d=dfiv.index.values[0]
    if end_d not in dfiv.index or end_d=="":
        end_d=dfiv.index.values[-1]
    
    dfiv=dfiv.loc[start_d:end_d]
    
    close_ts=dfiv.loc[start_d:end_d,'close'].groupby(dfiv.index).last()
    # skew_ts=(dfiv.loc[start_d:end_d,delta_1].groupby(dfiv.index).mean()-dfiv.loc[start_d:end_d,delta_2].groupby(dfiv.index).mean())
    skew_ts=(dfiv.loc[start_d:end_d,delta_1].groupby(dfiv.index).median()-dfiv.loc[start_d:end_d,delta_2].groupby(dfiv.index).median())

    # skew_ts=(dfiv.loc[start_d:end_d,delta_1].groupby(dfiv.index).last()-dfiv.loc[start_d:end_d,delta_2].groupby(dfiv.index).last())
    fig_skw=SkewGraph(varity, delta_1, delta_2, skew_ts, close_ts)
    get_skew(varity, delta_1, delta_2, skew_ts, close_ts)

    givendelta_iv_ts=dfiv.loc[start_d:end_d,given_delta].groupby(dfiv.index).median()
    atm_iv_ts=dfiv.loc[start_d:end_d,str(50)].groupby(dfiv.index).median()
    ret_ts=dfiv.loc[start_d:end_d,'ret'].groupby(dfiv.index).last()
    # ret_ts=(close_ts/close_ts.shift(1)-1).rolling(21)
    
    fs=5
    fig2=plt.figure(figsize=(4,3))
    ax=fig2.add_subplot(111)
    ax.plot(givendelta_iv_ts,'-o',ms=1,color='r',label=given_delta+' IV')
    ax.plot(atm_iv_ts,'-o',ms=1,color='b',label='ATM IV')
    ax.plot(ret_ts,'k',linewidth=1,label='20D Vol')
    ax.tick_params(labelsize=fs)
    plt.xticks(rotation=fs)
    ax.set_ylabel('Volatility(%)',fontsize=fs)
    ax.legend(fontsize=fs)
    ax.grid()
    ax.set_title(label=varity+' '+given_delta+' Delta IV',fontsize=fs)
    # stm.pyplot(fig)
    
    
    res=((dfiv.drop(columns="expire_date").loc[start_d:end_d].groupby(dfiv.index).mean().describe()).drop(columns='close').drop(index='count'))

    return  res,fig2,fig_skw


def SkewGraph(varity,delta_1,delta_2,skew_ts,close_ts):
    fs=10
    mean,per_25,per_75=skew_ts.describe()[['mean','25%','75%']].round(2)
    # fig=plt.figure(figsize=(20,20))
    # fig=plt.figure(figsize=(10,5))
    fig=plt.figure(figsize=(4,3))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(skew_ts,'r'
            ,linewidth=1
            ,label='Skew')
    ax.tick_params(labelsize=fs)
    # ax.set_ylim(np.percentile(skew_ts,98),np.percentile(skew_ts,2))

    ax.set_ylabel('Vol Spread (%)'
                   ,fontsize=fs
                  )
    plt.xticks(rotation=fs)
    xmin,xmax=skew_ts.index[0],skew_ts.index[-1]
    
    ax.hlines(mean,xmin,xmax,linewidth=1,color='k',label='Mean:'+str(mean))
    # ax.text(xmax,mean,'ave:'+str(round(mean,2)),fontsize=30,color='b')
    ax.hlines(per_25,xmin,xmax,linestyle='-',color='y',label='25 Per:'+str(per_25))
    # ax.text(xmax,per_25,'25per:'+str(round(per_25,2)),fontsize=30,color='b')
    ax.hlines(per_75,xmin,xmax,linestyle='-',color='g',label='75 Per:'+str(per_75))
    # ax.text(xmax,per_75,'75per:'+str(round(per_75,2)),fontsize=30,color='b')
    ax.legend(fontsize=fs)    
    ax.grid()
    ax.set_title(label=varity+' ('+delta_1+'%) - ('+delta_2+'%) Delta Skew'
                  ,fontsize=fs)    
    ax2=ax.twinx()
    ax2.plot(close_ts,label='Close')
    ax2.set_ylabel('Close',fontsize=fs)
    ax2.tick_params(labelsize=fs)
    plt.tight_layout()
    
    return fig


        
def get_skew(varity,delta_1,delta_2,skew_ts,close_ts):
    # 创建Plotly图表
    mean,per_25,per_75=skew_ts.describe()[['mean','25%','75%']].round(2)

    fig = go.Figure()
    
    # 添加百分率数据，使用左侧纵坐标
    fig.add_trace(go.Scatter(x=skew_ts.index
                             ,y=skew_ts.values
                             ,name='skew'
                             ,yaxis='y1'
                             ,line=dict(color='red')))
    
    # 添加绝对价格数据，使用右侧纵坐标
    fig.add_trace(go.Scatter(x=close_ts.index
                             ,y=close_ts.values
                             ,name='close'
                             ,yaxis='y2'
                             ,line=dict(color='blue')))
    
    # fig.add_hline(y=per_25, line_dash="dot", line_color="green", name='25th Percentile')
    fig.add_trace(go.Scatter(
                            y=[per_25]*skew_ts.shape[0],  # 创建一个与df长度相同的列表，所有值都是q25
                            x=skew_ts.index,
                            mode='lines',
                            line=dict(color='green', dash='dash'),
                            name='25th Per',
                            showlegend=True
                            ))
    
    fig.add_trace(go.Scatter(
                            y=[per_75]*skew_ts.shape[0],  # 创建一个与df长度相同的列表，所有值都是q25
                            x=skew_ts.index,
                            mode='lines',
                            line=dict(color='orange', dash='dash'),
                            name='75th Per',
                            showlegend=True
                            ))
    fig.add_trace(go.Scatter(
                            y=[mean]*skew_ts.shape[0],  # 创建一个与df长度相同的列表，所有值都是q25
                            x=skew_ts.index,
                            mode='lines',
                            line=dict(color='black', dash='dash'),
                            name='Mean',
                            showlegend=True
                            ))


    fig.add_hline(y=mean, line_dash="dot", line_color="black", name='mean')

    # fig.add_hline(y=per_75, line_dash="dot", line_color="yellow", name='75th Percentile')

    
    # 更新布局，设置两个纵坐标
    fig.update_layout(
        title=varity+'('+str(delta_1)+'%)-('+str(delta_2)+'%)Delta Skew'
        ,xaxis_title='Date'
        ,yaxis=dict(title='Vol Spread (%)', side='left')
        ,yaxis2=dict(title='Price', side='right', overlaying='y', position=0.95)
        ,width=400  # 设置图表宽度
        ,height=600  # 设置图表高度
        ,legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1)
    )
    
    # 在Streamlit中展示图表
    stm.plotly_chart(fig, use_container_width=True)
    return fig



def get_figure(suf):
    # 创建Plotly图表
    fig = go.Figure()
    for idx in suf.index:
        fig.add_trace(go.Scatter(x=suf.columns
                                  , y=suf.loc[idx,:].values
                                  , mode='lines'
                                  ,name=str(idx)
                                  ))
    # fig.update_layout(yaxis_range=[data.min(), data.max()])
    # for idx,m in enumerate(moneyness):
    #     title_name+="("+str(m)+": "+str(round(y[-1][idx],2))+")  "
    fig.update_layout(title='IV'
                      ,width=800
                      ,height=400
                      )
    
    fig.update_xaxes(tickangle=45)
    return fig

# @xw.func
# def STonVarity(sheets_name,varity,days,next_end_date):
#     delta_s_arr=np.arange(0.9,1.11,0.01).round(2)
#     t_decay_arr=np.arange(1,int(days))
#     # next_end_time=datetime.combine(next_end_date,time(15,0,0))
#     next_end_time=next_end_date
#     st=StressTest()
#     st_res=st.calStressTestArr(varity,delta_s_arr,t_decay_arr,datetime.now(),next_end_time)
   
#     st_theta=st_res['theta']     
#     st_vega=st_res['vega'] 
#     # st_pnl=st_res['pnl'].sum()
    
#     fig=st_res['fig']
#     wb=xw.Book.caller()
#     sht=wb.sheets(sheets_name)
#     sht.pictures.add(fig,name='pnl'
#                       ,update=True
#                         # ,left=sht.range(cell_address).left
#                         # ,top=sht.range(cell_address).top
#                         ,left=sht.range("A1").left
#                         ,top=sht.range("A1").top
#                         ,width=250
#                         ,height=500
#                       )
    
    
#     st_total=pd.concat([st_theta,st_vega]).T
#     st_total.insert(int(days)-1,"Vega",delta_s_arr)
#     st_total=st_total.T
#     st_total.index.name="Theta"    
    

#     return st_total

@xw.func
def STonVarityNew(sheets_name,varity,current_trade_date,next_end_time):
    delta_s_arr=np.arange(0.9,1.11,0.01).round(2)
    # t_decay_arr=np.arange(1,int(days))
    # next_end_time=datetime.combine(next_end_date,time(15,0,0))
    # next_end_time=next_end_date
    st=StressTestNew(current_trade_date.date())
    # st_res=st.calStressTestArrNew(varity,delta_s_arr,decay_days,datetime.now(),next_end_time)
    st_res=st.calStressTestNew(varity, delta_s_arr, 20, datetime.now(), next_end_time)
    st_theta=st_res['theta']     
    st_vega=st_res['vega'] 
    # st_pnl=st_res['pnl'].sum()
    
    fig=st_res['fig']
    wb=xw.Book.caller()
    sht=wb.sheets(sheets_name)
    sht.pictures.add(fig,name='pnl'
                      ,update=True
                        # ,left=sht.range(cell_address).left
                        # ,top=sht.range(cell_address).top
                        ,left=sht.range("A1").left
                        ,top=sht.range("A1").top
                        ,width=250
                        ,height=500
                      )
    
    
    st_total=pd.concat([st_theta,st_vega]).T
    st_total.insert(20,"Vega",delta_s_arr)
    st_total=st_total.T
    st_total.index.name="Theta"    
    

    return st_total



# def getpdobList(underlyingCode,trade_date,first_obsdate,expired_date,pricing_time):
#     pdobList=pd.DataFrame(index=rqd.get_trading_dates(first_obsdate,expired_date)
#                           , columns=['close'],data=np.nan)
#     if pricing_time.time()<time(15,0,0) and first_obsdate<trade_date:
#         close_ts=rqd.get_price(underlyingCode,first_obsdate,expired_date,'1d','close').loc[underlyingCode]
#         pdobList.loc[close_ts.index.tolist(),'close']=close_ts.close
#     elif pricing_time.time()<time(15,0,0) and first_obsdate>=trade_date:
#         pdobList.fillna(0,inplace=True)
#     elif pricing_time.time()>=time(15,0,0) and first_obsdate>trade_date:
#         pdobList.fillna(0,inplace=True)
#     elif pricing_time.time()>=time(15,0,0) and first_obsdate==trade_date:
#         pdobList.loc[trade_date]=findLatestPrice(underlyingCode)
#     elif pricing_time.time()>=time(15,0,0) and first_obsdate<trade_date:
#         close_ts=rqd.get_price(underlyingCode,first_obsdate,expired_date,'1d','close').loc[underlyingCode]
#         if np.isnan(pdobList.loc[trade_date.date(),'close']):
#             # print("11")
#             pdobList.loc[trade_date.date(),'close']=findLatestPrice(underlyingCode) 
#     else:
#         pass
#     pdobList.fillna(0,inplace=True)
#     pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]
#     return pdobList

# @xw.func
# def calKOAccmulator(pricing_time,underlyingCode,trade_date,first_obsdate,expired_date
#                     ,opttype,s_0,s_t,K,B,REB,coupon,qty_freq,isCashSettle,lev_daily,lev_expire,cust_bs):
    
#     # print(str(pricing_time)[:19])
#     # print(pricing_time)[:19])
#     # cols=['S_t','pv', 'delta', 'gamma', 'vega_percentage', 'theta_per_day', 'rho_percentage'
#     #       , 'dividend_rho_percentage'
#     #       ,'accumulated_position', 'accumulated_payment', 'accumulated_pnl']
#     # res=pd.DataFrame(columns=cols,index)
#     # print(str(pricing_time)[:19])
#     LiV=LinearInterpVol(underlyingCode, trade_date)    
#     a=pyAIKOAccumulatorPricer(opttype
#                             , -1 if cust_bs=='B' else 1
#                             , s_t
#                             , K
#                             , datetime2timestamp(str(pricing_time)[:19])
#                             , datetime2timestamp(str(expired_date.date())+" 15:00:00")
#                             , s_0
#                             , int(qty_freq)
#                             , int(isCashSettle)
#                             , float(lev_daily)
#                             , float(lev_expire)
#                             , float(coupon)
#                             , float(B)
#                             , float(REB)
#                             , getpdobList(underlyingCode,trade_date,first_obsdate,expired_date,pricing_time)
#                             , rf, rf
#                             , 0 #const_sgm
#                             , getpdobList(underlyingCode,trade_date,first_obsdate,expired_date,pricing_time).shape[0]
#                             , LiV.getVolsurfacejson(underlyingCode)
#                             # ,vol
#                             )
#     a[1]=a[1]/findMultiplier(getRQcode(underlyingCode))
#     a.insert(0,s_t)
    
#     # res=pd.DataFrame(data=a,index=cols).T
#     # res.index=res.S_t
#     return a

def getflag(flag):
    """
    Parameters
    ----------
    flag : TYPE
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    if flag=="熔断累购":
        return 'b_acccall'
    elif flag=="熔断累沽":
        return 'b_accput'
    elif flag=="熔断增强累购":
        return 'b_acccallplus'
    elif flag=="熔断增强累沽":
        return 'b_accputplus'
    elif flag=="熔断固陪累购":
        return 'b_fpcall'
    elif flag=="熔断固陪累沽":
        return 'b_fpput'
    elif flag=="累购":
        return 'acccall'
    elif flag=="累沽":
        return 'accput'
    elif flag=="固定赔付累购":
        return 'fpcall'
    elif flag=="固定赔付累沽":
        return 'fpput'
    else:
        print("Wrong option type!")




# @xw.func
# def IterationAcc(ask_Inter,ask_Coupon,underlyingCode,flag,direction
#                 , s_t,s_0,strike,barrier,rebate,coupon
#                 , pricing_time, trade_date,first_obsdate,expired_date,obs_days
#                 , daily_amt, leverage, leverage_expire
#                 , isCashsettle,strike_ramp, barrier_ramp
#                 , const_sgm):
      
    
#       # ask_Inter=ask_Inter
#       # ask_Coupon=ask_Coupon
#       # underlyingCode=underlyingCode
#       # flag=flag
#       # direction= 
#       #  s_t,
#       #  s_0,
#       #  strike,
#       #  barrier,
#       #  rebate,
#       #  coupon, 
#       #  pricing_time, 
#       #  trade_date,
#       #  first_obsdate,
#       #  expired_date,
#       #  obs_days, 
#       #  daily_amt, 
#       #  leverage, 
#       #  leverage_expire, 
#       #  isCashsettle,
#       #  strike_ramp, 
#       #  barrier_ramp, 
#       #  const_sgm
      
    
#       # if s_t=="":
#        s_t=findLatestPrice(underlyingCode)
#        s_0=s_t
#        direction="B" if direction=="买" else "S"
#        ca=CalAcc(underlyingCode
#                  , getflag(flag)
#                  , direction, float(s_t), float(s_0)
#                  , strike, barrier, rebate, coupon
#                  , pricing_time
#                  , trade_date, first_obsdate, expired_date, int(obs_days)
#                  , int(daily_amt), int(leverage), int(leverage_expire), int(isCashsettle)
#                  , float(strike_ramp), float(barrier_ramp), float(const_sgm)
#                  )
#        # res=ca.getRes(ask_Inter, ask_Coupon)
#        ca.expired_date
#        return ca.expired_date




# @xw.func
# def LoadBarContracts():
#     bc=BarrierContracts(datetime(2024,3,21).date())
#     # posLive_Bar=bc.dfbar
#     wb=xw.Book.caller()
#     sht=wb.sheets('BarrierContracts')
#     sht.range("a1")=bc.dfbar
#     # return posLive_Bar

# a=LoadBarContracts()

# def IntraHours(now_time):
#     '''
    

#     Parameters
#     ----------
#     trading_dates : list with datetime.date 
#     start_time : datetime.time
#         start trading time.
#     expire_date : datetime.date
#         expiration.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     '''
#     if now_time<time(9,0,0):
#         intra_hours=4/6
#         # trd_date=start_time.date()
#     elif now_time<time(11,30,0):
#         intra_hours=(datetime(start_time.year,start_time.month,start_time.day,11,30,0)-start_time).seconds/3600
#         intra_hours=(intra_hours+1.5)/6
#         # trd_date=start_time.date()
#     elif now_time<time(13,30,0):
#         intra_hours=1.5/6
#         # trd_date=start_time.date()
#     elif now_time<time(15,0,0):
#         intra_hours=(datetime(start_time.year,start_time.month,start_time.day,15,0,0)-start_time).seconds/3600
#         intra_hours=intra_hours/6
#         # trd_date=start_time.date()
#     elif now_time<time(21,0,0):
#         intra_hours=0
#         # trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
#     elif now_time<time(23,0,0):
#         intra_hours=(datetime(start_time.year,start_time.month,start_time.day,23,0,0)-start_time).seconds/3600
#         intra_hours=(intra_hours+4)/6
#         # trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
#     else:
#         intra_hours=0

#     return intra_hours



def Plot_IV(start_date,end_date,underlyingCode,given_deltas,optionList="",skew=False):
    '''
    Show the trend of implied volatility of given options or given delta during then given dates.
    It draws the implied volatility of selected during current trading date when the start_date 
    is equal to the end_date. 
    
    1.Draw given delta IV based on given unlderyingers or varity.Can be within one day or
      during a selected period.
    2.Draw selected option contracts IV,Delta and Skew.Can be within one day or during a selected period.
    
    Parameters
    ----------
    start_date : Datetime.date
        Start trading date.
    end_date : Datetime.date
        End trading date.
    underlyingCode : str
        Could be underlyingers or varity.
    given_delta : int, optional
        The select delta that the plot will show. The default is 50.
    optionList : List[str], optional
        The select option code(s) that the plot will show. The default is [].
    skew : Bool, optional
        Only can effect when the optionList is not empty.The default is False.
    Returns
    -------
    None.

    '''
    trading_date_list=rqd.get_trading_dates('2019-01-01','2026-12-31')
    if start_date==end_date:
        freq='1m'
    else:
        freq="1m"
        
    if optionList=="":
        option_list=rqd.options.get_contracts(underlyingCode)
    else:
        option_list=optionList
        
    tic=datetime.now()
    wd=rqd.get_price(option_list,start_date,end_date,freq,fields=['close','volume','trading_date']).reset_index()
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
    
    # tic=datetime.now()
    # wd['t']=list(map(lambda trd_t,exp_d:calTradttm_0(trading_date_list, trd_t, exp_d),wd.datetime,wd.expire_date))
    # print("Format ttm time with 0 = ", datetime.now() - tic, "s")
    
    # tic=datetime.now()
    # wd['t']=list(map(lambda trd_t,exp_d:calTradttm(trading_date_list, trd_t, exp_d),wd.datetime,wd.expire_date))
    # # wd['t']=(wd.expire_date-wd.trading_date).apply(lambda x:x.days)
    # print("Format ttm time with = ", datetime.now() - tic, "s")
    
    
    tic=datetime.now()
    wd['exp_idx']=wd.expire_date.apply(lambda x:trading_date_list.index(x))
    wd['trd_idx']=wd.trading_date.apply(lambda x:trading_date_list.index(x))
    wd['trd_time']=wd.datetime.apply(lambda x:x.time())
    cdt_time=[wd.trd_time<time(9,0,0)
              ,wd.trd_time<time(11,30,0)
              ,wd.trd_time<time(13,30,0)
              ,wd.trd_time<time(15,0,0)
              ,wd.trd_time==time(15,0,0)
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
    wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/annual_coeff,wd.close_option,wd.shape[0])
    print("Runing IV time = ", datetime.now() - tic, "s")
    wd['delta']=getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/annual_coeff,wd.iv)*100
    
    # wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/365,wd.close_option,wd.shape[0])
    # wd['delta']=getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/365,wd.iv)*100

    wd['delta_round']=wd.delta.round(0)
    wd['iv_volume']=wd['iv']*wd['volume']

    if optionList=="":
        # Draw given delta trend
        fig=plt.figure(figsize=(5,3))
        ax=fig.add_subplot(111)
        for given_delta in given_deltas:
            wd_delta=wd.where((wd.delta_round>=given_delta-2)&(wd.delta<=given_delta+2)).dropna()
            iv_delta=wd_delta.groupby('datetime')['iv'].mean().values if start_date==end_date else wd_delta.groupby('trading_date')['iv_volume'].sum()/wd_delta.groupby('trading_date')['volume'].sum()
            ax.plot(iv_delta,'-o',ms=2,label=str(given_delta)+' '+underlyingCode)
            plt.xticks(rotation=30)
            plt.legend(fontsize=5)
            ax.grid(True)
        ax.set_title(label='Given_delta IV',fontsize=5)
        ax.tick_params(labelsize=5)
        # stm.pyplot(fig)
            
    elif start_date==end_date:
        #Draw given options trend within the selected trading date.
        if skew:
            fig=plt.figure(figsize=(8,15))
            ax=fig.add_subplot(211)
            for col in optionList:
                ax.plot(
                        wd[wd['order_book_id']==col]['iv'].values
                        ,label=col)
                ax.grid(True)
                ax.legend()
            ax=fig.add_subplot(212)
            skew_iv=(wd[wd['order_book_id']==option_list[0]]['iv'].values-wd[wd['order_book_id']==option_list[1]]['iv'].values)*100
            ax.plot(skew_iv,label='skew')
            ax.grid(True)
            ax.set_title(option_list[0]+' - '+option_list[1]+' Skew')
            # stm.pyplot(fig)
        else:
            fig=plt.figure(figsize=(5,2))
            ax=fig.add_subplot(111)
            for col in optionList:
                ax.plot(wd[wd['order_book_id']==col]['iv'].values
                        ,label=col)
                ax.grid(True)
                ax.legend(fontsize=5)
            ax.set_title('IV Within '+str(start_date),fontsize=8)
            ax.tick_params(labelsize=5)
    else:
        # Draw given options trend during the given period.
        wd_grouper=wd.groupby(['order_book_id','trading_date'])
        wd_res=wd_grouper['iv_volume'].sum()/wd_grouper['volume'].sum()
        wd_res=wd_res.unstack().T
        if skew:
            fig=plt.figure(figsize=(8,15))
            ax=fig.add_subplot(311)
            for col in wd_res.columns:
                ax.plot(wd_res[col],'-o',label=col)
                ax.legend()    
                ax.grid(True)
            ax.set_title('IV')
            ax=fig.add_subplot(312)
            for col in wd_res.columns:
                ax.plot(wd[wd['order_book_id']==col].groupby('trading_date')['delta_round'].last(),label=col+'Delta')
                ax.grid(True)
                ax.legend()
            ax.set_title('Delta')
            ax=fig.add_subplot(313)
            skew_iv=(wd_res[option_list[0]]-wd_res[option_list[1]])*100
            ax.plot(skew_iv,'-o',label='skew')
            ax.grid(True)
            ax.set_title('Skew '+option_list[0]+' - '+option_list[1])
            # stm.pyplot(fig)
        else:
            fig=plt.figure(figsize=(8,10))
            ax=fig.add_subplot(211)
            for col in wd_res.columns:
                ax.plot(wd_res[col],'-o',label=col)
                ax.legend()    
                ax.grid(True)
            ax.set_title('IV')
            ax=fig.add_subplot(212)
            for col in wd_res.columns:
                ax.plot(wd[wd['order_book_id']==col].groupby('trading_date')['delta_round'].last(),'-o',label=col+'Delta')
                ax.grid(True)
                ax.legend()
            ax.set_title('Delta')
            # stm.pyplot(fig)
    return fig
            
def hl_k(k):
    if k/findLatestPrice(underlyingCode)>=1:
        return 'background-color: tomato'
    else:
        return'background-color: yellowgreen'

def bidvol_style(vol_col):
    if vol_col>=0:
        return 'background-color: turquoise'
    else:
        return 'background-color: turquoise'
    
def askvol_style(vol_col):
    if vol_col>=0:
        return 'background-color: '
    else:
        return 'background-color: '
    
def delta_style(delta_col):
    if delta_col>=0:
        return 'background-color: ivory'
    else:
        return 'background-color: ivory'
# def ivgre_style(cols):
    
#     style_map={'bidvol_Call':bidvol_style
#                ,'delta_Call':delta_style
#                ,iv_gre.columns[6]:hl_k
#                }
#     return style_map(cols)

#     iv_gre.style.applymap(hl_k,subset=iv_gre.columns[6])
#     iv_gre.style.applymap(bidvol_style,subset=iv_gre[['bidvol_Call','bidvol_Put']])
#     iv_gre.style.applymap(delta_style,subset=iv_gre[['delta_Call','delta_Put']])
#     return iv_gre




if __name__=="__main__":
    import warnings 
    warnings.filterwarnings("ignore")
    # annaul_coeff=annual_coeff
    # import time as sys_time
    
    # stm.set_page_config(page_title="Volatility Summary") 

    
#%%
stm.set_page_config(layout='wide')
with stm.sidebar:
        # with col1:
        cur_trd_date=stm.date_input("Current Trade Date:",value=rqd.get_future_latest_trading_date())
        varity=stm.text_input("Choose Varity", value="AU").upper()
        trd_contracts=rqd.futures.get_contracts(varity,cur_trd_date)
        a=rqd.get_price(trd_contracts,rqd.get_previous_trading_date(cur_trd_date,1)
                      ,rqd.get_previous_trading_date(cur_trd_date,1),'1d')['volume'].unstack()
        trading_contracts=a.sort_values(a.columns[0],ascending=False)[:2].index.tolist()

        # underlyingCodes=stm.text_input("Choose Underlying", value=trading_contracts[:1])
        underlyingCodes=stm.text_input("Choose Underlyings")
        if len(underlyingCodes)>0:
            trading_contracts=list(map(lambda c:getRQcode(c),underlyingCodes.split(",")))
        else:
            pass
        # underlyingCodes=underlyingCodes.split(",")
        # stm.write(underlyingCodes)
        delta_1 =stm.text_input("Give a delta_1 for skew", value=10, placeholder="")
        delta_2 =  stm.text_input("Give a delta_2 for skew", value=-10, placeholder="")
        given_delta= stm.text_input("Give a delta for IV", value=20, placeholder="")
        start_d=stm.date_input("Start Date for Historical IV(Coule be empty):",value=datetime(2022,12,26))
        end_d=stm.date_input("End Date for Historical IV(Coule be empty):",value=None)

         
for underlyingCode in trading_contracts:
    with stm.container():
        res,dfvol_std=RQcalRealisedVol(underlyingCode,15,cur_trd_date,annual_coeff)
        col1, col2 = stm.columns(2,gap='small')
        with col1:
            stm.header(underlyingCode+" Realized Vol Distribution")
            stm.write(res.astype(float).round(2))
        with col2:
            Plt_RQcalRealisedVol(underlyingCode,dfvol_std)

# try:
suf=getSurface(varity, cur_trd_date)   
suf['LeftCalDays']=pd.Series(suf.index).apply(lambda x:(x-cur_trd_date).days).values

stm.header("**"+varity.upper()+"**"+" Implied Volitilty surface in "+str(cur_trd_date))
col1, col2 = stm.columns(2)
with col1:
    stm.write(suf.round(2))
with col2:
    stm.plotly_chart(get_figure(suf.iloc[:,:-1]))
# except:
    # stm.subheader("No ITC Surface")

try:
    @stm.cache_data
    def load_data(path,file_name):
        dfiv=pd.read_csv(path+'\/'+file_name,index_col='trade_date')  
        return dfiv
    
    with stm.container():
        path=r'D:\chengyilin\ivdata_suf'
        file_name=varity.upper()+'_SUF.CSV'
        dfiv=load_data(path,file_name)
     
        res,fig2,fig_skw=HistIVDescribeSurf(varity,dfiv,delta_1,delta_2, given_delta,start_d,end_d)
        stm.header("**"+varity.upper()+"**"+" IV Distribution During Given Period on Each Delta")
        stm.write(res.round(2))
        col1, col2 = stm.columns(2,gap='small')
        with col1:
            stm.pyplot(fig2)
            # stm.plotly_chart(fig2)
        with col2:
            stm.pyplot(fig_skw)
except:
    stm.subheader("No Historical IV")




# 应用样式函数到DataFrame
# styled_df = df.style.apply(style_by_column, axis=1)

try:    
            # stm.button("Refresh")
    # while True:
            # stm.rerun()  # 
        for underlyingCode in trading_contracts:
            with stm.container():
                    iv_gre,fig_iv=RQcalIVandGreeks(underlyingCode,annual_coeff,rng=25)
                    iv_gre=iv_gre.round(4)
                    stm.header(underlyingCode+" IV and Greeks")
                    stm.write(iv_gre.style.applymap(hl_k,subset=iv_gre.columns[6]))
                    # style_dict = {
                    #                 'bidvol_Call': {'background-color': 'yellow'},  # 列 'A' 的单元格将变为红色字体
                    #                 'delta_Call': {'background-color': 'yellow'},  # 列 'B' 的单元格将有黄色背景
                    #                 # : {'font-weight': 'bold'}  # 列 'C' 的单元格将变为粗体
                    #             }
                    # stm.write(iv_gre.style.apply(style_by_column, axis=1))
                    # functions=[lambda col:bidvol_style(col)
                    #            ,lambda col:delta_style(col)]
                    # stm.write(iv_gre.style.apply([map(funs,cols) for funs,cols in zip(functions,iv_gre[['bidvol_Call','delta_Call']])]))
                    # stm.write(iv_gre)
            
                    col1, col2 = stm.columns(2,gap='small')
                    with col1:
                        stm.pyplot(fig_iv)
                    fig_atm=Plot_IV(start_date=cur_trd_date
                            ,end_date=cur_trd_date
                            ,underlyingCode=underlyingCode #RQD CODE
                            ,given_deltas=[50]
                            # ,optionList=getOptionCodes(underlyingCode, findLatestPrice(underlyingCode),1)#RQD C
                            )
                    with col2:
                        stm.pyplot(fig_atm)
                                # else:
                                    # pass
        # time.sleep(10)  # 等待半秒
# except KeyboardInterrupt:
#         pass  # 允许用户通过键盘中断来停止循环
except:
      stm.subheader("No ITC") 
    
#%%










