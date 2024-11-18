# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:22:03 2024

@author: dzrh
"""



import xlwings as xw
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns
# import tushare as tus
from datetime import date,timedelta,datetime,time
# import time
# import iFinDPy as fd
import os
import copy
from CYL.OptionPricing import BSM,calIV,calTradttm,AccOption,StandardBarrierArr,BarrierAccOption
# from CYL.YieldChainAPI import YieldChainAPI
import bisect
import itertools
import rqdatac as rqd
from CYL.pythonAPI_pyfunctions4newDll_3 import pyAIAccumulatorPricer,pyAIKOAccumulatorPricer,jsonvolSurface2cstructure,jsonvolSurface2cstructure_selfapi,pyAILinearInterpVolSurface
from ast import literal_eval
import json
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
from CYL.OTCAPI import SelfAPI
import re


rf=0.03
q=0
annual_coeff=252
# user='chengyl'
# passwd='CYLcyl0208@'
# YL=YieldChainAPI(user,passwd)
api=SelfAPI()
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
    code=underlying[0].upper() if underlying[1].isdigit() else underlying[:2].upper()
    return code


def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code


    # optioncode="AP410C7800"
    # optioncode="ag2410C8200"
def getRQoptcode(optioncode):
    optioncode=optioncode.upper()
    if "-" not in optioncode:
        idx=re.search(r'[^\d](?=.*\d)', optioncode[::-1]).start()
        optioncode=optioncode if optioncode[len(optioncode)-idx-1-4].isdigit() else optioncode[:re.search(r'\d', optioncode).start()]+"2"+optioncode[re.search(r'\d', optioncode).start():]
    else:
        optioncode=optioncode.replace("-","")
    return optioncode



def datetime2timestamp(datetime_t: str):
    '''
    The function used to convert a string of Beijing time (GMT + 8) to timestamp
    :param datetime_t: a string of time in the format '%Y-%m-%d %H:%M:%S'
    :return: a timestamp as an integer
    '''
    timestamp_t = int((datetime.strptime(datetime_t, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1) - timedelta(hours=8)) / timedelta(seconds=1))
    return timestamp_t

def getpdobList(underlyingCode,trade_date,first_obsdate,expired_date,pricing_time,show_end_date=""):
    pdobList=pd.DataFrame(index=rqd.get_trading_dates(first_obsdate,expired_date)
                          , columns=['close'],data=np.nan)
    if pricing_time.time()<time(15,0,0) and first_obsdate<trade_date:
        close_ts=rqd.get_price(underlyingCode,first_obsdate,expired_date,'1d','close').loc[underlyingCode]
        pdobList.loc[close_ts.index.tolist(),'close']=close_ts.close
    elif pricing_time.time()<time(15,0,0) and first_obsdate>=trade_date:
        pdobList.fillna(0,inplace=True)
    elif pricing_time.time()>=time(15,0,0) and first_obsdate>trade_date:
        pdobList.fillna(0,inplace=True)
    elif pricing_time.time()>=time(15,0,0) and first_obsdate==trade_date:
        pdobList.loc[trade_date]=rqd.current_snapshot(underlyingCode).last
    elif pricing_time.time()>=time(15,0,0) and first_obsdate<trade_date:
        close_ts=rqd.get_price(underlyingCode,first_obsdate,expired_date,'1d','close').loc[underlyingCode]
        pdobList.loc[close_ts.index.tolist()]=close_ts.values
        if np.isnan(pdobList.loc[trade_date,'close']):
            # print("11")
            pdobList.loc[trade_date,'close']=rqd.current_snapshot(underlyingCode).last 
    else:
        pass
    pdobList.fillna(0,inplace=True)
    if show_end_date!="":
        pdobList.loc[show_end_date]=0
    pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]
    return pdobList
     
class LinearInterpVol():
    def __init__(self,asset_list,valuedate):
        self.valuedate=str(valuedate)
        self.dict_vol=dict(zip(asset_list,[api.getVol(self.valuedate,x)["mid"].pivot_table(values="vol",index="expire",columns="strike")/100 for x in asset_list]))
        self.ttmdays_arr=np.array([1,7,14,30,60,90,183,365])
        
    
    def getVolsurfacejson(self,UnderlyingCode):
        # if type(self.asset)==str:
        #     underlyingCodes=[self.asset]
        # else:
        #     underlyingCodes=self.asset
            # vfe=api.getVol_json(str(self.trade_date), self.underlyingCode)['mid']
            # vfe=json.dumps(literal_eval(str(vfe)))
            # self.cSV = jsonvolSurface2cstructure_selfapi(vfe)
        vfe=api.getVol_json(self.valuedate,UnderlyingCode)['mid']
        vfe=json.dumps(literal_eval(str(vfe)))
        cvolSurface = jsonvolSurface2cstructure_selfapi(vfe)
        # vfe=vfe.replace("S","s").replace("E","e").replace("V","v").replace("D","")
        # cvolSurface = jsonvolSurface2cstructure(vfe)
        return cvolSurface
    
    def calVol(self,single_ass,k_s,exp_ttm):
         if k_s==0:
             return 0
         elif exp_ttm<=0:
             return 0
         else:
             ks_ratio=np.arange(0.8,1.22,0.02).round(2)
         
             # volsurface=(self.dict_vol[single_ass].pivot_table(values="vol",index="expire",columns="strike")/100).values
             volsurface=self.dict_vol[single_ass].values
             idx_t=bisect.bisect_left(self.ttmdays_arr,exp_ttm)
             idx_ks=bisect.bisect_left(ks_ratio,k_s) 
    
    
             condition=[k_s<=ks_ratio[0]
                        ,k_s>ks_ratio[-1]]
             choice=[(0,1),(20,21)]     
             cols=np.select(condition,choice,(idx_ks-1,idx_ks+1))
             
             
             if exp_ttm<=self.ttmdays_arr[0]:
                 vol_slice=volsurface[0,cols[0]:cols[1]]
             elif exp_ttm>self.ttmdays_arr[-1]:
                 vol_slice=volsurface[-1,cols[0]:cols[1]]
             else:
                 vol_slice=volsurface[idx_t-1:idx_t+1,cols[0]:cols[1]]
    
    
             if vol_slice.shape==(1,):
                 return vol_slice[0]
             elif vol_slice.shape==(2,):
                 ratio_arr=ks_ratio[cols[0]:cols[1]] 
                 return (np.diff(vol_slice,axis=0)/np.diff(ratio_arr)*(k_s-ratio_arr[0])+vol_slice[0])[0]
             elif vol_slice.shape==(2,1):
                 t_arr=self.ttmdays_arr[idx_t-1:idx_t+1] 
                 return ((np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0])[0]
             else:
                 t_arr=self.ttmdays_arr[idx_t-1:idx_t+1] 
                 ratio_arr=ks_ratio[cols[0]:cols[1]] 
                 term_diff_vol=(np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0,:]
                 return (np.diff(term_diff_vol)/np.diff(ratio_arr)*(k_s-ratio_arr[0])+term_diff_vol[0])[0]

class AccOptionArr():
    '''
    Accumulation option pricing. Containing AccCall,AccPut,AccCall/Put with Fixed Income
    Only give present value and greeks of the remainding option part!
    Parameters
    ----------
    trading_dates_list : list with datetime-type items
        Trading Dates List from excel.
    acctype : str
        c for acccall,p for accput,cfi for acc call fixed income,pfi for acc put fixed income.
    S_0 : float
        entry price.
    sigma : float
        vol.
    strike : float
        k.
    barrier : float
        barrier level.
    fixed_income : float
        fixed income customer recieved.
    startobs_date : str
        start observation date eg "2022-01-01".
    endobs_date : str
        end observation date eg "2022-01-01".
    trading_time: datetime
        current time for calculating
    qty_freq : float
        quantity traded each day.
    customer_bs: str
        B or S for the whole structure!
    leverage : int
        leverage amount.
    delta_barrier : float
        barrier hedge width.
    delta_strike : float
        strike hedge width.
    show : bool, opional
        whether to show the result. The default is False.

    Returns
    -------
    
    '''
    __annual_coeff=annual_coeff
    __rf=0.03
    __q=0
    def __init__(self,trading_dates_list,acctype,asset,S_0,sigma
                 ,strike,barrier,fixed_income
                 ,startobs_date,endobs_date,trading_time
                 ,qty_freq,customer_bs,leverage
                 ,delta_strike,delta_barrier
                 ,show=False):
     
        self.trading_dates_list=trading_dates_list
        self.acctype=acctype.upper()
        self.asset=asset
        self.S_0=S_0
        self.sigma=sigma
        self.strike=strike
        self.barrier=barrier
        self.fixed_income=fixed_income
        self.startobs_date=startobs_date
        self.endobs_date=endobs_date
        self.trading_time=trading_time
        self.qty_freq=abs(qty_freq)
        self.customer_bs=customer_bs.upper()
        self.leverage=leverage
        self.delta_strike=delta_strike
        self.delta_barrier=delta_barrier
        self.show=show

        if self.delta_barrier=="" or self.delta_barrier==0:
            self.delta_barrier=0.00001
        if self.delta_strike=="" or self.delta_strike==0:
            self.delta_strike=0.00001
        
    
        # self.trading_days=self.findTradingDays(self.trading_dates_list, self.startobs_date, self.endobs_date)
        self.trading_days=self.trading_dates_list.index(self.endobs_date)-self.trading_dates_list.index(self.startobs_date)+1
        self.obs_dates_list=self.trading_dates_list[self.trading_dates_list.index(self.startobs_date):self.trading_dates_list.index(self.endobs_date)+1]
        if self.__annual_coeff==365:
             self.ttm_list=list(map(lambda t:(pd.to_datetime(datetime.strftime(t,'%Y-%m-%d')+" 15:00:00")-trading_time)/np.timedelta64(1,'D'),self.obs_dates_list))
      
        else:
            self.ttm_list=list(map(lambda t:calTradttm(self.trading_dates_list,self.trading_time,t),self.obs_dates_list))
        self.ttm_list=np.array(self.ttm_list)

        
    def getCall(self):
          if self.barrier==0:
              self.buy_sell=['S','B']
              self.opttype=['P','C']
              self.strikes=[self.strike,self.strike]
              self.ratio=[self.leverage,1]
          else:
              self.buy_sell=['S','B','S','B']
              self.opttype=['P','C','C','C']
              self.strikes=[self.strike,self.strike,self.barrier,self.barrier+self.delta_barrier] if self.customer_bs=="B" else [
                  self.strike,self.strike,self.barrier-self.delta_barrier,self.barrier]
              knockout_qty=(self.barrier-self.strike)/ self.delta_barrier if self.customer_bs=="B" else (self.barrier-self.strike-self.delta_barrier)/ self.delta_barrier 
              self.ratio=[ self.leverage,1,knockout_qty+1,knockout_qty]
                  
    def getCallFI(self):
          if self.barrier==0:
             self.buy_sell=['S','B','S']
             self.opttype=['P','C','C']
             self.strikes=[self.strike,self.strike-self.delta_strike,self.strike]
             knockout_qty_strike=self.fixed_income/self.delta_strike
             self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike]
          else:
              self.buy_sell=['S','B','S','S','B']
              self.opttype=['P','C','C','C','C']
              self.strikes=[self.strike,self.strike-self.delta_strike,self.strike,
                       self.barrier,self.barrier+self.delta_barrier] if self.customer_bs=="B" else [
                           self.strike,self.strike,self.strike+self.delta_strike,
                                    self.barrier-self.delta_barrier,self.barrier]
              knockout_qty_strike=self.fixed_income/self.delta_strike
              knockout_qty_barrier=self.fixed_income/self.delta_barrier
              self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike,
                     knockout_qty_barrier,knockout_qty_barrier]
       
    def getPut(self):
          if self.barrier==0:
              self.buy_sell=['S','B']
              self.opttype=['C','P']
              self.strikes=[self.strike,self.strike]
              self.ratio=[self.leverage,1]
          else:
              self.buy_sell=['S','B','S','B']
              self.opttype=['C','P','P','P']
              self.strikes=[self.strike,self.strike,self.barrier,self.barrier-self.delta_barrier] if self.customer_bs=="B" else [self.strike,self.strike,self.barrier+self.delta_barrier,self.barrier] 
              knockout_qty=(self.strike-self.barrier)/ self.delta_barrier if self.customer_bs=="B" else (self.strike-self.barrier-self.delta_barrier)/ self.delta_barrier
              self.ratio=[self.leverage,1,knockout_qty+1,knockout_qty]
            
    def getPutFI(self):
          if self.barrier==0:
             self.buy_sell=['S','B','S']
             self.opttype=['C','P','P']
             self.strikes=[self.strike,self.strike+self.delta_strike,self.strike]
             knockout_qty_strike=self.fixed_income/self.delta_strike
             self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike]
          else:
              self.buy_sell=['S','B','S','S','B']
              self.opttype=['C','P','P','P','P']
              self.strikes=[self.strike,self.strike+self.delta_strike,self.strike,
                       self.barrier,self.barrier-self.delta_barrier] if self.customer_bs=="B" else [
                           self.strike,self.strike,self.strike-self.delta_strike,self.barrier+self.delta_barrier,self.barrier]
              knockout_qty_strike=self.fixed_income/self.delta_strike
              knockout_qty_barrier=self.fixed_income/self.delta_barrier
              self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike,
                     knockout_qty_barrier,knockout_qty_barrier]
    
    def getACC(self):
        if self.acctype=='ACCCALL':
            self.getCall()
        elif self.acctype=='ACCPUT':
            self.getPut()
        elif self.acctype=='FPCALL':
            self.getCallFI()
        elif self.acctype=='FPPUT':
           self.getPutFI()
        else:
            print("Wrong Acc Option Type!")
        
        if self.sigma=="":
            print('There is no volitality!')
        else:
            self.sigma_arr=[self.sigma]*len(self.opttype)*self.trading_days
            
        self.ttm_list=self.ttm_list.repeat(len(self.opttype)) #due to ttm_list is array_like
        self.buy_sell=self.buy_sell*self.trading_days
        self.opttype=self.opttype*self.trading_days
        self.strikes=self.strikes*self.trading_days
        self.ratio=self.ratio*self.trading_days
        self.s_arr=self.S_0*np.ones(self.ttm_list.shape)
        
        
        # acc2.sigma_arr=[acc2.sigma]*len(acc2.opttype)*acc2.trading_days
        # acc2.ttm_list=acc2.ttm_list.repeat(len(acc2.opttype)) #due to ttm_list is array_like
        # acc2.buy_sell=acc2.buy_sell*acc2.trading_days
        # acc2.opttype=acc2.opttype*acc2.trading_days
        # acc2.strikes=acc2.strikes*acc2.trading_days
        # acc2.ratio=acc2.ratio*acc2.trading_days
        # acc2.s_arr=acc2.S_0*np.ones(acc2.ttm_list.shape)
        
        
        # bsm_list=list(map(lambda k,t,vol,opt_type:BSM(a.S_0,k,t,a._AccOption__rf,0,vol,opt_type)
        #           , a.strikes
        #           ,a.ttm_list/a._AccOption__annual_coeff
        #           ,a.sigma_arr,a.opttype))

        #    bsm_arr=BSM_ARR(b.s_arr
        #                    ,np.array(b.strikes)
        #                    ,b.ttm_list/b._AccOptionArr__annual_coeff,b._AccOptionArr__rf,b._AccOptionArr__q
        #                    ,np.array(b.sigma_arr)
        #                    ,np.array(b.opttype))
        
        
        
        bsm_arr=BSM_ARR(self.s_arr
                        ,np.array(self.strikes)
                        ,self.ttm_list/AccOptionArr.__annual_coeff,AccOptionArr.__rf,AccOptionArr.__q
                        ,np.array(self.sigma_arr)
                        ,np.array(self.opttype))
        
        dfbsm=pd.DataFrame([bsm_arr.price(),bsm_arr.delta(),bsm_arr.gamma(),bsm_arr.vega()/100,bsm_arr.theta(1/AccOptionArr.__annual_coeff)]).T
        dfbsm['ttm']=self.ttm_list
        dfbsm.columns=['cashflow','delta','gamma','vega','theta','ttm']
        bs_idx_greeks=np.where(np.array(self.buy_sell)=="B",-1,1) #for greeks calculation AND book pv!
        bs_idx_pv=np.where(np.array(self.buy_sell)=="B",1,-1) #b represent cash flow
        
        bs_idx=np.repeat(bs_idx_greeks,dfbsm.shape[-1]).reshape(dfbsm.shape)
        bs_idx[:,0]=bs_idx_pv
        
        qty_ratio=np.array(self.ratio)*self.qty_freq
        bsm_total=dfbsm*bs_idx
        bsm_total=bsm_total.multiply(qty_ratio,axis=0)
        bsm_total=bsm_total.drop(bsm_total[bsm_total.ttm==0].index,axis=0)
        bsm_total.drop(columns='ttm',inplace=True)
        
        result=bsm_total.sum()
        result['cashflow unit']=result['cashflow']/np.abs(self.qty_freq)/self.trading_days
        result['bookpv']=-1*result['cashflow']

        if self.show:
            if result.cashflow>0:
                print("{name}:{value}".format(name="Issuer Received", value=round(result['cashflow'],2)))
                print("{name}:{value}".format(name="Issuer Received Unit", value=round(result['cashflow unit'],2)))
            else:
                print("{name}:{value}".format(name="Issuer Paid", value=round(result['cashflow'],2)))
                print("{name}:{value}".format(name="Issuer Paid Unit", value=round(result['cashflow unit'],2)))
            print("{name}:{value}".format(name="Delta(Issuer)", value=round(result.delta,2)))
            print("{name}:{value}".format(name="Gamma(Issuer)", value=round(result.gamma,2)))
            print("{name}:{value}".format(name="Vega(Issuer)", value=round(result.vega,2)))
            print("{name}:{value}".format(name="Theta(Issuer)", value=round(result.theta,2)))
    
        return result
    
    def getAS(self):
        if self.acctype=="ASCALL":
           opt_type_list=['C']*len(self.ttm_list)
        elif self.acctype=="ASPUT":
            opt_type_list=['P']*len(self.ttm_list)
        else:
            print("Wrong AS Option Type!")
        
      
        # if self.sigma=="":
        #     bsm_list=list(map(lambda t:BSM(self.S_0,self.strike,t,AccOption.__rf,0,LinearInterpVol(self.asset,self.S_0,self.strike,t*AccOption.__annual_coeff),opt_type)
        #               ,self.ttm_list/AccOption.__annual_coeff))
        # else:
        #     bsm_list=list(map(lambda t:BSM(self.S_0,self.strike,t,AccOption.__rf,0,self.sigma,opt_type)
        #               ,self.ttm_list/AccOption.__annual_coeff))
    
        bsm_arr=BSM_ARR(self.S_0*np.ones(self.ttm_list.shape)
                          ,self.strike*np.ones(self.ttm_list.shape)
                          ,self.ttm_list/AccOptionArr.__annual_coeff,AccOptionArr.__rf,AccOptionArr.__q
                          ,self.sigma*np.ones(self.ttm_list.shape)
                          ,opt_type_list)
        
        bsm_arr=pd.DataFrame([bsm_arr.price(),bsm_arr.delta(),bsm_arr.gamma(),bsm_arr.vega()/100,bsm_arr.theta(1/AccOptionArr.__annual_coeff)])

        bsm_arr.columns=['cashflow','delta','gamma','vega','theta']
        
        bs_idx_greeks=-1 if self.customer_bs=="B" else 1
        bs_idx_pv=1 if self.customer_bs=="B" else -1
        bsm_total=bsm_arr.multiply(self.qty_freq,axis=0)

        result=bsm_total.sum()
        result['cashflow']=result.cashflow*bs_idx_pv
        result['cashflow unit']=result['cashflow']/self.qty_freq/self.trading_days
        result[['delta','gamma','vega','theta']]=result[['delta','gamma','vega','theta']]*bs_idx_greeks
        result['bookpv']=-1*result['cashflow']
        if self.show:
            if result.cashflow>0:
                print("{name}:{value}".format(name="Issuer Received", value=round(result['cashflow'],2)))
                print("{name}:{value}".format(name="Issuer Received Unit", value=round(result['cashflow unit'],2)))
            else:
                print("{name}:{value}".format(name="Issuer Paid", value=round(result['cashflow'],2)))
                print("{name}:{value}".format(name="Issuer Paid Unit", value=round(result['cashflow unit'],2)))
            print("{name}:{value}".format(name="Delta(Issuer)", value=round(result.delta,2)))
            print("{name}:{value}".format(name="Gamma(Issuer)", value=round(result.gamma,2)))
            print("{name}:{value}".format(name="Vega(Issuer)", value=round(result.vega,2)))
            print("{name}:{value}".format(name="Theta(Issuer)", value=round(result.theta,2)))
        
        return result
      
    def getResult(self):
        if self.acctype[:2]=="AS":
            return self.getAS()
        else:
            return self.getACC()
        


  


class BSM_ARR():
    '''
    Generalized BSM. Can be used to price European option on stocks,stocks paying a continuous
    dividend yield, options on future, and currency options.

    Parameters
    ----------
    s_arr : 
        spot price           
    k_arr : float
        strike price
    t_arr : float
        The time to maturity.
    r_arr : float
        risk free rate
    q_arr  : float
          dividend rate for stocks or foreign interest rate for currency option

    sigma_arr : float
            implied volatility
    opttype_arr : str
              option type
    Returns
    -------
    None.

    '''
    
   
    def __init__(self,s_arr,k_arr,t_arr,r,q,sigma_arr,opttype_arr):
          self.s_arr=s_arr
          self.k_arr=k_arr
          self.r_arr=r*np.ones(self.s_arr.shape)
          self.q_arr=q*np.ones(self.s_arr.shape)
          self.sigma_arr=sigma_arr
          self.t_arr=np.where(t_arr<0,0,t_arr)
          self.indicator=np.where(t_arr<0,0,1)
          # self.opttype_arr=np.array([opt.upper()for opt in opttype_arr])
          self.opttype_arr=opttype_arr
          self.condition=[self.opttype_arr=='C',self.opttype_arr=='P']
          # if self.t>0 and self.sigma!=0:
          self.__b_future()
          # np.seterr(divide='ignore')
          np.seterr(divide='ignore',invalid='ignore')
          self.d1_arr=(np.log(self.s_arr/self.k_arr)+self.t_arr*(self.b_arr+0.5*self.sigma_arr**2))/(self.sigma_arr*np.sqrt(self.t_arr))
          self.d2_arr=self.d1_arr-self.sigma_arr*np.sqrt(self.t_arr)


    def __b_stock(self):
        self.b_arr=self.r_arr
        
    def __b_stock_div(self):
        self.b_arr=self.r_arr-self.q_arr
     
    def __b_future(self):
          self.b_arr=np.zeros(self.s_arr.shape)
    
    def __b_margined_future(self):
        self.b_arr=np.zeros(self.s_arr.shape)
        self.r_arr=np.zeros(self.s_arr.shape)
        
    def __b_currency(self):
        self.b_arr=self.r_arr-self.q_arr
        
        
    def price(self):
        # np.seterr(divide='ignore')
        call_part=self.s_arr*np.exp((self.b_arr-self.r_arr)*self.t_arr)*norm.cdf(self.d1_arr)-self.k_arr/np.exp(self.r_arr*self.t_arr)*norm.cdf(self.d2_arr)
       
        choice=[0,self.k_arr/np.exp(self.r_arr*self.t_arr)-self.s_arr*np.exp((self.b_arr-self.r_arr)*self.t_arr)]
        pv_arr=call_part+np.select(self.condition,choice,default=np.nan)
        pv_arr=np.nan_to_num(pv_arr, nan=0.0)
        # pv_arr=np.where(self.opttype_arr=='C',0,))
        return pv_arr*self.indicator
    
    def delta(self):
     
        choice=[0,-1*np.exp((self.b_arr-self.r_arr)*self.t_arr)]
        delta_arr=np.exp((self.b_arr-self.r_arr)*self.t_arr)*norm.cdf(self.d1_arr)+np.select(self.condition,choice,default=np.nan)
        return delta_arr*self.indicator
        
    def gamma(self):   
        # np.seterr(divide='ignore')
        gamma_arr=norm.pdf(self.d1_arr)/(self.s_arr*self.sigma_arr*np.sqrt(self.t_arr))
        return gamma_arr*self.indicator
    
    def vega(self):
        # np.seterr(divide='ignore')
        vega_arr=(self.s_arr*norm.pdf(self.d1_arr)*np.sqrt(self.t_arr))/np.exp(self.r_arr*self.t_arr)
        return vega_arr*self.indicator
    
    def theta(self,delta_t):

        p_0=self.price()
        delta_t_arr=delta_t*np.ones(self.t_arr.shape)
        condition=[self.t_arr>=1/annual_coeff
                    ,(self.t_arr>0)&(self.t_arr<1/annual_coeff)
                    ,self.t_arr<=0]
        choice=[BSM_ARR(self.s_arr,self.k_arr,self.t_arr-delta_t_arr,self.r_arr,self.q_arr,self.sigma_arr,self.opttype_arr).price()-p_0
                ,np.max([(self.s_arr-self.k_arr)*np.select(self.condition,[1,-1],default=np.nan),np.zeros(self.s_arr.shape)],axis=0)-p_0
                ,0]
        theta_arr=np.select(condition,choice)    
        
        

        
        return theta_arr*self.indicator
    
    

class AccOptionArrSelectItems():
    __annual_coeff=annual_coeff
    __rf=0.03
    __q=0
    def __init__(self,item_select,trading_dates_list,decaydays,acctype,asset,S_0,sigma
                 ,strike,barrier,fixed_income
                 ,startobs_date,endobs_date,trading_time
                 ,qty_freq,customer_bs,leverage
                 ,delta_strike,delta_barrier
                 ):
        self.item_select=np.array(item_select)
        self.trading_dates_list=trading_dates_list
        self.decaydays=1 if decaydays=="" or decaydays==0 else decaydays
        self.acctype=acctype.upper()
        self.asset=asset
        self.S_0=S_0
        self.sigma=sigma
        self.strike=strike
        self.barrier=barrier
        self.fixed_income=fixed_income
        self.startobs_date=startobs_date
        self.endobs_date=endobs_date
        self.trading_time=trading_time
        self.qty_freq=qty_freq
        self.customer_bs=customer_bs.upper()
        self.leverage=leverage
        self.delta_strike=delta_strike
        self.delta_barrier=delta_barrier
  

        if self.delta_barrier=="" or self.delta_barrier==0:
            self.delta_barrier=0.00001
        if self.delta_strike=="" or self.delta_strike==0:
            self.delta_strike=0.00001
        
        
        # obs_dates_list: contains startobs_date!!
        # self.obs_dates_list=self.trading_dates_list[self.trading_dates_list.index(pd.to_datetime(self.startobs_date)):self.trading_dates_list.index(pd.to_datetime(self.endobs_date))+1]
        self.obs_dates_list=self.trading_dates_list[self.trading_dates_list.index(self.startobs_date):self.trading_dates_list.index(self.endobs_date)+1]

        if self.__annual_coeff==365:
             self.ttm_list=list(map(lambda t:(pd.to_datetime(datetime.strftime(t,'%Y-%m-%d')+" 15:00:00")-trading_time)/np.timedelta64(1,'D'),self.obs_dates_list))
      
        else:
            self.ttm_list=list(map(lambda t:calTradttm(self.trading_dates_list,self.trading_time,t),self.obs_dates_list))
        self.ttm_list=np.array(self.ttm_list)
        self.ttm_list=self.ttm_list[self.ttm_list>0]
        
        
    def getResult(self):
            if self.acctype[:2]=="AS":
               return self.getAS()
            else:
                return self.getACC()
            
    def getCall(self):
          if self.barrier==0:
              self.buy_sell=['S','B'] if self.customer_bs=="B" else ['B','S']
              self.opttype=['P','C']
              self.strikes=[self.strike,self.strike]
              self.ratio=[self.leverage,1]
          else:
              self.buy_sell=['S','B','S','B'] if self.customer_bs=="B" else ['B','S','B','S']
              self.opttype=['P','C','C','C']
              self.strikes=[self.strike,self.strike,self.barrier,self.barrier+self.delta_barrier] if self.customer_bs=="B" else [
                  self.strike,self.strike,self.barrier-self.delta_barrier,self.barrier]
              knockout_qty=(self.barrier-self.strike)/ self.delta_barrier if self.customer_bs=="B" else (self.barrier-self.strike-self.delta_barrier)/ self.delta_barrier 
              self.ratio=[ self.leverage,1,knockout_qty+1,knockout_qty]
                  
    def getCallFI(self):
          if self.barrier==0:
             self.buy_sell=['S','B','S'] if self.customer_bs=="B" else ['B','S','B']
             self.opttype=['P','C','C']
             self.strikes=[self.strike,self.strike-self.delta_strike,self.strike]
             knockout_qty_strike=self.fixed_income/self.delta_strike
             self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike]
          else:
              self.buy_sell=['S','B','S','S','B'] if self.customer_bs=="B" else ['B','S','B','B','S']
              self.opttype=['P','C','C','C','C']
              self.strikes=[self.strike,self.strike-self.delta_strike,self.strike,
                       self.barrier,self.barrier+self.delta_barrier] if self.customer_bs=="B" else [
                           self.strike,self.strike,self.strike+self.delta_strike,
                                    self.barrier-self.delta_barrier,self.barrier]
              knockout_qty_strike=self.fixed_income/self.delta_strike
              knockout_qty_barrier=self.fixed_income/self.delta_barrier
              self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike,
                     knockout_qty_barrier,knockout_qty_barrier]
       
    def getPut(self):
          if self.barrier==0:
              self.buy_sell=['S','B'] if self.customer_bs=="B" else ['B','S']
              self.opttype=['C','P']
              self.strikes=[self.strike,self.strike]
              self.ratio=[self.leverage,1]
          else:
              self.buy_sell=['S','B','S','B'] if self.customer_bs=="B" else ['B','S','B','S']
              self.opttype=['C','P','P','P']
              self.strikes=[self.strike,self.strike,self.barrier,self.barrier-self.delta_barrier] if self.customer_bs=="B" else [self.strike,self.strike,self.barrier+self.delta_barrier,self.barrier] 
              knockout_qty=(self.strike-self.barrier)/ self.delta_barrier if self.customer_bs=="B" else (self.strike-self.barrier-self.delta_barrier)/ self.delta_barrier
              self.ratio=[self.leverage,1,knockout_qty+1,knockout_qty]
            
    def getPutFI(self):
          if self.barrier==0:
             self.buy_sell=['S','B','S'] if self.customer_bs=="B" else ['B','S','B']
             self.opttype=['C','P','P']
             self.strikes=[self.strike,self.strike+self.delta_strike,self.strike]
             knockout_qty_strike=self.fixed_income/self.delta_strike
             self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike]
          else:
              self.buy_sell=['S','B','S','S','B'] if self.customer_bs=="B" else ['B','S','B','B','S']
              self.opttype=['C','P','P','P','P']
              self.strikes=[self.strike,self.strike+self.delta_strike,self.strike,
                       self.barrier,self.barrier-self.delta_barrier] if self.customer_bs=="B" else [
                           self.strike,self.strike,self.strike-self.delta_strike,self.barrier+self.delta_barrier,self.barrier]
              knockout_qty_strike=self.fixed_income/self.delta_strike
              knockout_qty_barrier=self.fixed_income/self.delta_barrier
              self.ratio=[self.leverage,knockout_qty_strike,knockout_qty_strike,
                     knockout_qty_barrier,knockout_qty_barrier]
  
    
    def getItem(self,item):
        if item=='delta':
            return self.bsm_arr.delta()
        elif item=='gamma':
            return self.bsm_arr.gamma()
        elif item=='theta':
            return self.bsm_arr.theta(1/AccOptionArrSelectItems.__annual_coeff)
        elif item=='vega':
            return self.bsm_arr.vega()/100
        elif item=='price':
            return self.bsm_arr.price()
        else:
            return np.nan
      
    def getACC(self):
        if self.acctype=='ACCCALL':
            self.getCall()
        elif self.acctype=='ACCPUT':
            self.getPut()
        elif self.acctype=='FPCALL':
            self.getCallFI()
        elif self.acctype=='FPPUT':
           self.getPutFI()
        else:
            print("Wrong Acc Option Type!")
            
        self.qty_freq=abs(self.qty_freq)
        t_shape=self.ttm_list.shape[0]
        group_shape=len(self.strikes)
    
        self.s_arr=list(self.S_0.repeat(t_shape*group_shape))*self.decaydays
        self.decay_arr=np.arange(1,self.decaydays+1,1).repeat(group_shape*t_shape*self.S_0.shape[0])
        self.ttm_list=np.array(list(self.ttm_list.repeat(group_shape))*self.S_0.shape[0]*(self.decaydays))-self.decay_arr
        self.strikes=self.strikes*(t_shape*self.S_0.shape[0])*self.decaydays
        self.opttype=self.opttype*(t_shape*self.S_0.shape[0])*self.decaydays
        self.buy_sell=self.buy_sell*(t_shape*self.S_0.shape[0]*self.decaydays)
        self.ratio=self.ratio*(t_shape*self.S_0.shape[0]*self.decaydays)


        # tic=datetime.now()
        # if self.sigma=="":
        #     k_s=np.array(self.strikes)/np.array(self.s_arr)
        #     self.sigma_arr=list(map(lambda ks,t:st.LiV.calVol(self.asset,ks,t)
        #              ,k_s,np.array(self.ttm_list)))
        # else:
        self.sigma_arr=[self.sigma]*len(self.s_arr)
        # print('run vol time=:',datetime.now()-tic,"s")

        self.bsm_arr=BSM_ARR(np.array(self.s_arr)
                        ,np.array(self.strikes)
                        ,np.array(self.ttm_list)/AccOptionArrSelectItems.__annual_coeff,AccOptionArrSelectItems.__rf,AccOptionArrSelectItems.__q
                        ,np.array(self.sigma_arr)
                        ,np.array(self.opttype))
    
        self.dfbsm=pd.DataFrame(columns=self.item_select)
        for item in self.item_select:
            self.dfbsm[item]=self.getItem(item)
        bs_idx_greeks=np.where(np.array(self.buy_sell)=="B",-1,1) #for greeks calculation AND book pv!
        bs_idx=np.repeat(bs_idx_greeks,self.dfbsm.shape[-1]).reshape(self.dfbsm.shape)        
        qty_ratio=np.array(self.ratio)*self.qty_freq
        self.bsm_total=self.dfbsm*bs_idx
        self.bsm_total=self.bsm_total.multiply(qty_ratio,axis=0)
        self.bsm_total['s']=self.s_arr
        self.bsm_total['decays']=self.decay_arr
        result=(self.bsm_total.groupby(['decays','s'])[self.dfbsm.columns].sum()).unstack()

        return result
 
    def getAS(self):

        t_shape=self.ttm_list.shape[0]
      
        self.s_arr=list(self.S_0.repeat(t_shape))*self.decaydays
        self.decay_arr=np.arange(1,self.decaydays+1,1).repeat(t_shape*self.S_0.shape[0])
        self.ttm_list=np.array(list(self.ttm_list)*self.S_0.shape[0]*self.decaydays)-self.decay_arr
        self.strikes=[self.strike]*(t_shape*self.S_0.shape[0])*self.decaydays
        if self.acctype=="ASCALL":
           self.opttype=['C']*(t_shape*self.S_0.shape[0])*self.decaydays
        elif self.acctype=="ASPUT":
            self.opttype=['P']*(t_shape*self.S_0.shape[0])*self.decaydays
        else:
            print("Wrong AS Option Type!")
        # tic=datetime.now()
        # if self.sigma=="":
        #     k_s=np.array(self.strikes)/np.array(self.s_arr)
        #     self.sigma_arr=list(map(lambda ks,t:st.LiV.calVol(self.asset,ks,t)
        #              ,k_s,np.array(self.ttm_list)))
        # else:
        self.sigma_arr=[self.sigma]*len(self.s_arr)
        # print('run vol time=:',datetime.now()-tic,"s"
        self.bsm_arr=BSM_ARR(np.array(self.s_arr)
                          ,np.array(self.strikes)
                          ,np.array(self.ttm_list)/AccOptionArrSelectItems.__annual_coeff,AccOptionArrSelectItems.__rf,AccOptionArrSelectItems.__q
                          ,np.array(self.sigma_arr)
                          ,np.array(self.opttype))
        
        self.dfbsm=pd.DataFrame(columns=self.item_select)
        for item in self.item_select:
             self.dfbsm[item]=self.getItem(item)
        
        # self.bsm_arr=pd.DataFrame([self.bsm_arr.price(),self.bsm_arr.delta(),self.bsm_arr.gamma(),self.bsm_arr.vega()/100,self.bsm_arr.theta(1/AccOptionArr2.__annual_coeff)])

        # self.bsm_arr.columns=['cashflow','delta','gamma','vega','theta']
        
        bs_idx_greeks=-1 if self.customer_bs=="B" else 1
        # bs_idx_pv=1 if self.customer_bs=="B" else -1
        
        self.bsm_total=self.dfbsm.multiply(self.qty_freq,axis=0)*bs_idx_greeks
        # self.bsm_total['cashflow']=self.bsm_total['cashflow']*bs_idx_pv
        # self.bsm_total['bookpv']=self.bsm_total['cashflow']*-1
        self.bsm_total['s']=self.s_arr
        self.bsm_total['decays']=self.decay_arr
     
        # result=(self.bsm_total.groupby(['decays','s'])[['bookpv','delta','gamma','theta','vega']].sum()).unstack()
        result=(self.bsm_total.groupby(['decays','s'])[self.dfbsm.columns].sum()).unstack()
        return result

class BarrierAccOptionSelectItems(BarrierAccOption):
    def __init__(self,item_select,trading_list,decaydays,opttype, s, s_0, strike, barrier, sigma, fix_income, reb, dt, lev_day,cust_bs,qty_freq, lev_exp
                 , next_obs_date, end_obs_date, trading_time):
        super().__init__(trading_list, opttype, s, s_0, strike, barrier, sigma
                          , fix_income, reb, dt, lev_day, cust_bs,qty_freq, lev_exp, next_obs_date, end_obs_date, trading_time)
        self.item_select=item_select
        self.decaydays=decaydays
        
    def getItem(self,func,item):
        if item=='delta':
            return func.delta()
        elif item=='gamma':
            return func.gamma()
        elif item=='theta':
            return func.theta(1/annual_coeff)
        elif item=='vega':
            return func.vega()
        elif item=='price':
            return func.price()
        else:
            return np.nan
        
        
    def dailyPart(self):
            if self.opttype=='acccall':
                self.acccall()
            elif self.opttype=='accput':
                self.accput()
            elif self.opttype=='acccallplus':
                self.acccall_forward()
            elif self.opttype=='accputplus':
                self.accput_forward()
            elif self.opttype=='fpcall':
                self.fpcall()
            elif self.opttype=='fpput':
                self.fpput()
            else:
                print('Wrong Type!')
            
            
            
            self.group_num=len(self.opt_list)
            self.s_length=self.s.shape[0]
            self.ttm_list=np.arange(self.ttm,0,-1)[:self.left_obs_days]
            self.t_shape=self.ttm_list.shape[0]
            if self.decaydays>0:
                self.s_arr=list(self.s.repeat(self.t_shape*self.group_num))*self.decaydays
                self.decay_arr=np.arange(1,self.decaydays+1,1).repeat(self.group_num*self.t_shape*self.s.shape[0])
                self.ttm_arr=np.array(list(self.ttm_list.repeat(self.group_num))*self.s.shape[0]*(self.decaydays))-self.decay_arr
            else:
                self.s_arr=list(self.s.repeat(self.t_shape*self.group_num))
                # self.decay_arr=np.arange(1,self.decaydays+1,1).repeat(self.group_num*self.t_shape*self.s.shape[0])
                self.ttm_arr=np.array(list(self.ttm_list.repeat(self.group_num))*self.s.shape[0])

            self.total_shape=self.ttm_arr.shape[0]
            # self.buy_sell=self.custbs_idx*int(self.total_shape/self.group_num)
            self.ratio=self.ratio*int(self.total_shape/self.group_num)
    
        
            
            self.sb=StandardBarrierArr(self.opt_list*int(self.total_shape/self.group_num)
                                    , self.dirt_list*int(self.total_shape/self.group_num)
                                    , self.move_list*int(self.total_shape/self.group_num)
                                    , self.s_arr
                                    , self.strikes*int(self.total_shape/self.group_num) #k
                                    , self.barriers*int(self.total_shape/self.group_num) #barrier
                                    , self.reb_list*int(self.total_shape/self.group_num)#rebate
                                    , [self.dt]*self.total_shape# dt
                                    # , (np.arange(self.ttm,0,-1)[:self.left_obs_days]/annual_coeff).repeat(self.group_num*self.s_lenght) #t 
                                    ,self.ttm_arr/annual_coeff
                                    ,[self.sigma]*self.total_shape #sigma
                                    , self.ttm%1 # intra_to_next_obs
                                    , rf
                                    , q)
        
            self.dfbsm=pd.DataFrame(columns=self.item_select)
            for item in self.item_select:
                self.dfbsm[item]=self.getItem(self.sb,item)
                
            # bs_idx_greeks=np.array(self.custbs_idx)*-1 #for greeks calculation AND book pv!
            
            bs_idx=np.array(list(np.array(self.custbs_idx)*self.cust_bs*-1)*int(self.total_shape/self.group_num))
            # bs_idx=np.repeat(bs_idx_greeks,self.dfbsm.shape[-1]).reshape(self.dfbsm.shape)        
            qty_ratio=np.array(self.ratio)*self.qty_freq
            self.bsm_total=self.dfbsm.multiply(bs_idx,axis=0)
            self.bsm_total=self.bsm_total.multiply(qty_ratio,axis=0)
            self.bsm_total['s']=self.s_arr
            self.bsm_total['decays']=self.decay_arr if self.decaydays>0 else self.decaydays
        
            return (self.bsm_total.groupby(['decays','s'])[self.dfbsm.columns].sum()).unstack()
        
    def expPart(self):
        
        if self.opttype=='acccall' or self.opttype=='acccallplus' or self.opttype=='fpcall':
            self.exp_type=['put']
            self.exp_dirt=['up']
            self.exp_move=['out']
        elif self.opttype=='accput'or self.opttype=='accputplus' or self.opttype=='fpput':
            self.exp_type=['call']
            self.exp_dirt=['down']
            self.exp_move=['out']
        else:
            print('Wrong type!')
            
        self.exp_custbs_idx=-1
        if self.decaydays>0:
            self.s_arr_exp=list(self.s)*self.decaydays
            self.decay_arr_exp=np.arange(1,self.decaydays+1,1).repeat(self.s.shape[0])
            self.ttm_arr_exp=self.ttm-self.decay_arr_exp
        else:
            self.s_arr_exp=list(self.s)
            # self.decay_arr_exp=np.arange(1,self.decaydays+1,1).repeat(self.s.shape[0])
            self.ttm_arr_exp=self.ttm


        self.exp_sb=StandardBarrierArr(self.exp_type*len(self.s_arr_exp)
                                    , self.exp_dirt*len(self.s_arr_exp)
                                    , self.exp_move*len(self.s_arr_exp)
                                    , self.s_arr_exp
                                    , [self.strike]*len(self.s_arr_exp)
                                    , [self.barrier]*len(self.s_arr_exp) #barrier
                                    , [0]*len(self.s_arr_exp)        #rebate
                                    , [self.dt]*len(self.s_arr_exp)# dt
                                    , (self.ttm_arr_exp)/annual_coeff #t 
                                    , [self.sigma]*len(self.s_arr_exp)   #sigma
                                    , self.ttm%1
                                    , rf
                                    , q)
        
        
        self.dfexp=pd.DataFrame(columns=self.item_select)
        for item in self.item_select:
            self.dfexp[item]=self.getItem(self.exp_sb,item)
        
        self.exp_total=self.dfexp*self.lev_exp*self.qty_freq*self.exp_custbs_idx*self.cust_bs*-1
        self.exp_total['s']=self.s_arr_exp
        self.exp_total['decays']=self.decay_arr_exp if self.decaydays>0 else self.decaydays
        return (self.exp_total.groupby(['decays','s'])[self.dfexp.columns].sum()).unstack()
        
    def getResult(self):
        if self.lev_exp>0:
            return self.dailyPart()+self.expPart() 
        else:
            return self.dailyPart()
        
        
def getTotalPnLData(file_path,end_d):
    df=pd.read_excel(file_path)
    # df['settleDate']=df.settleDate.apply(lambda x:x.date())
    periods=rqd.get_trading_dates(df.settleDate.unique()[-1],end_d)[1:]
    if len(periods)==0:
        return print("Latest Data !")
    else:
        start_date=periods[0]
        end_date=periods[-1]
    
        api=SelfAPI()
        dftotal=api.getTotalOTCTrade(str(start_date), str(end_date))
        dftotal.drop(index=dftotal[dftotal.optionType=='AIForwardPricer'].index.tolist(),inplace=True)
        dftotal['notional']=dftotal.notionalPrincipal/100000000
        dftotal.varietyName=dftotal.underlyingCode.apply(findCode)
        dftotal.tradeDate=pd.to_datetime(dftotal.tradeDate)
        # dftotal['month']=dftotal.tradeDate.apply(lambda x:x.month)
        dfnotion=dftotal.groupby(['tradeDate','varietyName'])[['notional','day1PnL']].sum().unstack()
        dfnotion.fillna(0,inplace=True)
      
        dfpnl=pd.DataFrame()
        for trd in dftotal.tradeDate.unique():
            dfrisk=api.getTradeRisk(str(trd)[:10])
            dfrisk['varietyName']=dfrisk['underlyingCode'].apply(findCode)
            # dfrisk['day1PnL'].fillna("0.00",inplace=True)
            for col in ['todayProfitLoss','theta','vega']:
                dfrisk[col]=dfrisk[col].apply(lambda x:float(x.replace(",","")))
            dfrisk=dfrisk.groupby('varietyName')[['todayProfitLoss','theta','vega']].sum().reset_index()
            dfrisk['settleDate']=str(trd)[:10]
            dfrisk['notion']=dfrisk['varietyName'].map(dfnotion.loc[trd]['notional'])
            dfrisk['day1']=dfrisk['varietyName'].map(dfnotion.loc[trd]['day1PnL'])
            dfpnl=pd.concat([dfpnl,dfrisk])
            
        dfpnl.notion.fillna(0,inplace=True)
        dfpnl.settleDate=pd.to_datetime(dfpnl.settleDate)
        df=pd.concat([df,dfpnl])
        df['settleDate']=df.settleDate.apply(lambda x:x.date())
        # df.settleDate=pd.to_datetime(df.settleDate)
        df.to_excel(file_path,index=False)
        return print("Finished!")    

        
               
class StressTestNew():
    def __init__(self,current_trade_date):
      
        # self.user='chengyl'
        # self.passwd='123456'
        # self.YL=YieldChainAPI(user,passwd)
        # self.api=SelfAPI()
        self.total_tradingdates=rqd.get_trading_dates('2023-01-01','2025-12-31')
        self.curren_trade_date=current_trade_date
        self.start_idx=self.total_tradingdates.index(self.curren_trade_date)
        # self.select_greeks=select_greeks

    # def __addPropertys(self):
    #     pp_l=self.dfotc.Propertys.apply(lambda x:len(x))
    #     pp_idx=pp_l.where(pp_l>0).dropna().index.tolist()
    #     pp_ts=self.dfotc.loc[pp_idx,'Propertys']
        
    #     p_names=pp_ts.apply(lambda x:pd.DataFrame(x)['name']).stack().unique()
    #     dfpp=pd.DataFrame(index=p_names,columns=pp_idx)
        
    #     for p,col in zip(pp_ts,pp_idx):
    #         dfpp.loc[pd.DataFrame(p)['name'],col]=pd.DataFrame(p)['value'].values
    #     dfpp=dfpp.T
    #     dfpp.rename(columns={"累计敲出价格":"barrier"
    #                          ,"固定赔付区间上沿":"fp_up"
    #                          ,"固定赔付区间下沿":"fp_dw"
    #                          ,"固定赔付":"fp"
    #                          ,"单倍系数":"interval_leverage"
    #                          ,"多倍系数":"leverage"
    #                          ,"敲出赔付":"rebate"
    #                          ,"到期倍数":"exp_leverage"},inplace=True)
    
    #     self.dfotc.loc[:,dfpp.columns]=np.nan
    #     self.dfotc.loc[pp_idx,dfpp.columns]=dfpp
    #     self.dfotc['barrier']=np.where(self.dfotc.StructureType=="固定赔付累购",self.dfotc.fp_up.astype(float),self.dfotc.barrier.astype(float))
    #     self.dfotc['barrier']=np.where(self.dfotc.StructureType=="固定赔付累沽",self.dfotc.fp_dw.astype(float),self.dfotc.barrier.astype(float))
        
        
    #     self.dfotc.loc[pp_idx,'FirstObservationDate']=self.dfotc.loc[pp_idx].ObservationDates.apply(lambda x:x.split(",")[0])
    #     self.dfotc.loc[pp_idx,'qty_freq']=self.dfotc.loc[pp_idx,'TradeAmount']/self.dfotc.loc[pp_idx].ObservationDates.apply(lambda x:len(x.split(",")))



    def getInfo(self):
        # self.dfinfo=self.YL.get_listInfo()
        # self.dfinfo.index=self.dfinfo.Code
        
        
        # self.dfitc=self.YL.get_listPosition()[['BookName','UnderlyingCode','ExchangeOptionCode','Volume']]
        # self.dfitc=self.dfitc[self.dfitc['BookName']=='场外交易簿']
        # self.dfitc['Varity']=list(map(lambda x:findCode(x),self.dfitc.UnderlyingCode))
        # self.dfitc.drop(self.dfitc[self.dfitc.Volume==0].index,axis=0,inplace=True)
        
        self.dfitc=pd.DataFrame(api.getITCLive_onDate(str(self.curren_trade_date)))
        self.dfitc['ExchangeOptionCode']=self.dfitc.index
        self.dfitc.reset_index(drop=True,inplace=True)
        self.dfitc['RQcode']=self.dfitc['ExchangeOptionCode'].apply(lambda x:getRQoptcode(x) if len(x)>6 else getRQcode(x))
        self.dfitc['Variety']=self.dfitc['RQcode'].apply(findCode)
        
        # self.dfotc=self.YL.get_tradeDetail(['确认成交','新增待确认'])
        # self.dfotc['Varity']=list(map(lambda x:findCode(x),self.dfotc.UnderlyingCode))
        # self.dfotc.drop(self.dfotc[self.dfotc.TradeAmount==0].index,axis=0,inplace=True)
        # tradetype_dic={'香草期权':'V','远期':'F','自定义交易':'SD','亚式期权':'AS','雪球期权':'SnowBall'}
        # self.dfotc.TradeType=self.dfotc.TradeType.map(tradetype_dic)   
        # self.__addPropertys()
        
        self.dfotc=api.getOTC_LiveTrade(str(self.curren_trade_date))
        self.dfotc['Variety']=self.dfotc.underlyingCode.apply(findCode)
        
  
    def getITC(self,varity_list):
          # pos_itc=self.dfitc[self.dfitc['Variety']==varity.upper()]
          pos_itc=self.dfitc[self.dfitc['Variety'].isin([v.upper()for v in varity_list])]

          pos_itc=pos_itc.copy()
          pos_itc['UnderlyingCode']=list(map(lambda x:rqd.instruments(x).trading_code.upper(),pos_itc.RQcode.apply(lambda x:rqd.instruments(x).underlying_order_book_id.upper() if len(x)>6 else x)))
          pos_itc['Strike']=pos_itc.RQcode.apply(lambda x:rqd.instruments(x).strike_price if len(x)>6 else 0)
          pos_itc['OptionType']=pos_itc.RQcode.apply(lambda x:rqd.instruments(x).option_type if len(x)>6 else "F")
          pos_itc['Expire_Date']=pos_itc.RQcode.apply(lambda x:rqd.instruments(x).maturity_date)
          self.pos_itc=pos_itc
          
    def getOTC(self,varity_list):
          # pos_otc=self.dfotc[self.dfotc['Variety']==varity.upper()]
          pos_otc=self.dfotc[self.dfotc['Variety'].isin([v.upper()for v in varity_list])]
          pos_otc=pos_otc.copy()
          pos_otc['ExerciseDate']=pos_otc.maturityDate.apply(lambda x:pd.to_datetime(x).date())
          pos_otc['TradeDate']=pos_otc.tradeDate.apply(lambda x:pd.to_datetime(x).date())
          pos_otc['CurAmount']=np.where(pos_otc.buyOrSell=='buy',1,-1)*pos_otc.availableVolume
          opttypes=pos_otc.callOrPut.map({'call':'C','put':'P'})
          acctypes=pos_otc.optionType.map({"AICallAccPricer":"acccall"
                                              ,"AIPutAccPricer":"accput"
                                              ,"AICallFixAccPricer":"fpcall"
                                              ,"AIPutFixAccPricer":"fpput"
                                              ,"AICallKOAccPricer":"b_acccall"
                                              ,"AIPutKOAccPricer":"b_accput"
                                              ,"AICallFixKOAccPricer":"b_fpcall"
                                              ,"AIPutFixKOAccPricer":"b_fpput"
                                              ,"AIEnCallKOAccPricer":"b_acccallplus"
                                              ,"AIEnPutKOAccPricer":"b_accputplus"
                                              ,'AIForwardPricer':"F"})
          pos_otc['OptionType']=acctypes.fillna(opttypes)
          self.pos_otc=pos_otc

    def getPos(self,varity_list,pos_simu_dic,additional_filter):
          self.getInfo()
          self.getITC(varity_list)
          self.getOTC(varity_list)
          
          
          pos=pd.DataFrame(columns=['TradeNumber','UnderlyingCode','UnderlyingPrice'
                            ,'ExerciseDate','OptionType','TradeAmount'
                            ,'Strike','StrikeRamp','barrier','BarrierRamp','fp','leverage','FirstObservationDate'
                            ,'qty_freq','rebate','exp_leverage','InitialSpotPrice','isCashSettle'])
          
          pos['TradeNumber']=pd.concat([self.pos_otc['tradeCode'],self.pos_itc['ExchangeOptionCode']],ignore_index=True)
          pos['UnderlyingCode']=pd.concat([self.pos_otc['underlyingCode'],self.pos_itc['UnderlyingCode']],ignore_index=True)
          
          
          pos['Strike']=pd.concat([self.pos_otc['strike'],self.pos_itc.Strike],ignore_index=True)
          pos['ExerciseDate']=pd.to_datetime(pd.concat([self.pos_otc.ExerciseDate,self.pos_itc.Expire_Date],ignore_index=True))
          
          pos['FirstObservationDate']=pd.concat([self.pos_otc.startObsDate,pd.Series([np.nan]*self.pos_itc.shape[0])],ignore_index=True)

          pos['OptionType']=pd.concat([self.pos_otc.OptionType,self.pos_itc.OptionType],ignore_index=True)
          pos['TradeAmount']=pd.concat([self.pos_otc.CurAmount,self.pos_itc.volume],ignore_index=True)
          pos['TradeAmount']=pos['TradeAmount'].astype(float)
          
          pos.index=pos.TradeNumber
          
          pos.loc[self.pos_otc.tradeCode,'qty_freq']=self.pos_otc.basicQuantity.astype(float).values
       

          pos.loc[self.pos_otc.tradeCode,'StrikeRamp']=self.pos_otc.strikeRamp.values
          pos.loc[self.pos_otc.tradeCode,'BarrierRamp']=self.pos_otc.barrierRamp.values
          pos.loc[self.pos_otc.tradeCode,'fp']=self.pos_otc.fixedPayment.values
          pos.loc[self.pos_otc.tradeCode,'leverage']=self.pos_otc.leverage.values
          pos.loc[self.pos_otc.tradeCode,'barrier']=self.pos_otc.barrier.values
          pos.loc[self.pos_otc.tradeCode,'rebate']=self.pos_otc.knockoutRebate.values
          pos.loc[self.pos_otc.tradeCode,'exp_leverage']=self.pos_otc.expireMultiple.values
          pos.loc[self.pos_otc.tradeCode,'InitialSpotPrice']=self.pos_otc.entryPrice.values
          pos.loc[self.pos_otc.tradeCode,'isCashSettle']=self.pos_otc.settleType.map({"physical":"0","cash":"1","mix":"2"})
          pos=pos.loc[pos.OptionType.dropna().index.tolist()]
          
          if pos_simu_dic:
              pos=pd.concat([pos,pd.DataFrame(pos_simu_dic)])
          else:
              pass
          # if additional_filter:
          #     pos[pos[additional_filter.keys]=]
          #     # pos=pd.concat([pos,pd.DataFrame(pos_simu_dic)])
          # else:
          #     pass

          
          asset_list=pos.UnderlyingCode.unique().tolist()
          
          self.LiV=LinearInterpVol(asset_list, self.curren_trade_date)
          # self.LiV=LinearInterpVol(self.curren_trade_date)
          pos.reset_index(drop=True,inplace=True)

          pos['RQCode']=pos.UnderlyingCode.apply(getRQcode)
          pos.UnderlyingPrice=list(map(lambda x:x.last,pos.RQCode.apply(rqd.current_snapshot)))
          
          
          self.pos_exotic=pos[pos.OptionType.apply(lambda x:len(x)>1)]
          self.pos_exotic['cust_bs']=self.pos_exotic.TradeAmount.apply(lambda x:'B' if x<0 else 'S')
          # self.pos_exotic.qty_freq.fillna(self.pos_exotic.TradeAmount,inplace=True)         
          
          self.pos=pos.drop(self.pos_exotic.index)
    
    def __formatParams(self,delta_s_arr,decay_days,start_time):
          self.start_time=start_time
          self.s_ts=self.pos.UnderlyingPrice
          self.exp_t=[rqd.get_next_trading_date(exp) if exp not in self.total_tradingdates else exp for exp in self.pos.ExerciseDate]
     
          self.ttm=list(map(lambda x:calTradttm(self.total_tradingdates,self.start_time, x),self.exp_t))
          self.ttm=np.where(self.pos.OptionType=='F',0,self.ttm)
          self.intra_hours=self.ttm[0]-int(self.ttm[0])
          
          self.decay_dates=self.total_tradingdates[self.start_idx+1:self.start_idx+decay_days+1]
          self.ttm_decay_calendar_days=((pd.to_datetime(self.decay_dates)-pd.to_datetime(self.curren_trade_date)).days).values
          ttm_calendar_days=((pd.to_datetime(self.pos.ExerciseDate.values)-pd.to_datetime(self.curren_trade_date)).days).values
          self.s_matr=((pd.DataFrame(np.ones((self.pos.shape[0],delta_s_arr.shape[0]))*delta_s_arr)).multiply(self.pos.UnderlyingPrice.values,axis=0)).values
          self.k_matr=(np.ones((delta_s_arr.shape[0],self.pos.shape[0]))*self.pos.Strike.values).transpose()
          self.k_div_s=self.k_matr/self.s_matr
          #calendar ttm used to find vol
          self.ttm_cal_matr=(np.ones((delta_s_arr.shape[0],self.pos.shape[0]))*ttm_calendar_days).transpose()
          #trading date ttm used to find pv and greeks
          self.ttm_matr=(np.ones((delta_s_arr.shape[0],self.pos.shape[0]))*self.ttm).transpose()
          self.types_matr=np.asmatrix(self.pos.OptionType.values).repeat(delta_s_arr.shape[0],axis=0).transpose()
          self.barrier_idx=self.pos_exotic[self.pos_exotic.OptionType.apply(lambda x:"_" in x)].index.tolist()
          self.norm_idx=self.pos_exotic[self.pos_exotic.OptionType.apply(lambda x:"_" not in x)].index.tolist()




    def __calvolmatrix(self,ass_arr,k_div_s_matrix,used_t_matr):
         col=k_div_s_matrix.shape[-1]
         v_matr=np.zeros(k_div_s_matrix.shape)
         tic=datetime.now()
         for c in np.arange(0,col,1):
                 v_matr[:,c]=list(map(lambda ass,k_s,t:self.LiV.calVol(ass,k_s,t)
                           ,ass_arr,k_div_s_matrix[:,c],used_t_matr[:,c]))
         print("Running Vol time = ", datetime.now() - tic, "s")
         return v_matr
     
        
    def calStressTestNew(self,varity_list,delta_s_arr,decay_days,start_time,next_end_time,pos_simu_dic=dict(),additional_filter=dict(),flag_pv=True,fig=True):
        self.calGreeks(varity_list, delta_s_arr, decay_days, start_time, pos_simu_dic,additional_filter)
        st_result=dict()
        st_result=self.greeks_dic
        if flag_pv:
            pv_0=self.__calPvArr(1,self.start_time)
            pv_exotic_0=self.__calExoticPvArr(1,self.start_time)
            df=pd.DataFrame(index=self.pos.UnderlyingCode,columns=delta_s_arr)
            dfexo=pd.DataFrame(index=self.pos_exotic.loc[self.norm_idx].UnderlyingCode,columns=delta_s_arr)
            for ds in delta_s_arr:
                df[ds]=self.__calPvArr(ds,next_end_time).values
                dfexo[ds]=self.__calExoticPvArr(ds,next_end_time).values
                
                
            df=df.subtract(pv_0,axis='index')
            df=df.multiply(self.pos.TradeAmount.values,axis='index')
            
            dfexo=dfexo.subtract(pv_exotic_0,axis='index')
            df=pd.concat([df,dfexo])
            
            if self.barrier_idx:
                pv_barrier_0=self.__calBarrierPvArr(np.array([1]), self.start_time)
                pv_barrier_1=self.__calBarrierPvArr(delta_s_arr, next_end_time)
    
                dfbar=pv_barrier_1.subtract(pv_barrier_0.values,axis='index')
                df=pd.concat([df,dfbar.astype(float)])
            else:
                pass    
            st_result['pnl']=df.groupby('UnderlyingCode').sum()
        
        if fig:
            pnl=df.sum(axis=0)
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,7))
            # ax=fig.add_subplot(2,1,1)
            ax1.plot(pnl/10000)
            ax1.set_ylabel('Pnl(w)')
            ax1.grid(True)
            varity=",".join(varity_list)
            ax1.set_title(varity.upper())
      
            ax2.plot(pnl.loc[0.97:1.03],'-o',ms=10,mfc='orange')
            # ax2.set_xlabel('1 PNL: '+str(pnl.loc[1].round(0))+' 0.99 PNL: '+str(pnl.loc[0.99].round(0))+' 1.01 PNL:'+str(pnl.loc[1.01].round(0)))
            # ax2.annotate(str(pnl.loc[1].round(0)),xy=(1,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
            # ax2.annotate(str(pnl.loc[0.99].round(0)),xy=(0.99,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
            # ax2.annotate(str(pnl.loc[1.01].round(0)),xy=(1.01,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
            
            ax2.annotate(str(round(pnl.loc[1],0)),xy=(1,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
            ax2.annotate(str(round(pnl.loc[0.99],0)),xy=(0.99,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
            ax2.annotate(str(round(pnl.loc[1.01],0)),xy=(1.01,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')

            
            ax2.set_ylabel('Pnl')
            ax2.grid(True)
            # ax2.set_title(varity)
            
            st_result['fig']=fig
 
        return st_result
        
    def calGreeks(self,varity,delta_s_arr,decay_days,start_time,pos_simu_dic,additional_filter):
        
         self.getPos(varity,pos_simu_dic,additional_filter)
         self.__formatParams(delta_s_arr, decay_days, start_time)
         greeks_sel=['theta','vega']
         self.greeks_dic=dict()
         for key in greeks_sel:
             self.greeks_dic[key]=pd.DataFrame(index=self.decay_dates,columns=delta_s_arr)
         # self.Theta=pd.DataFrame(index=self.decay_dates,columns=delta_s_arr)
         # self.Vega=pd.DataFrame(index=self.decay_dates,columns=delta_s_arr)
         tic=datetime.now()
         
         for t_decay,idx in zip(self.ttm_decay_calendar_days,self.decay_dates):
               bsm_arr=BSM_ARR(self.s_matr,self.k_matr
                               ,(self.ttm_matr-np.argwhere(self.ttm_decay_calendar_days==t_decay).item()-1)/annual_coeff
                               , rf ,q
                               ,self.__calvolmatrix(self.pos.UnderlyingCode,self.k_div_s,self.ttm_cal_matr-t_decay+self.intra_hours)
                               ,self.types_matr)
               
               # self.Theta.loc[idx]=np.dot(self.pos.TradeAmount.values.reshape((1,self.pos.shape[0])),np.nan_to_num(bsm_arr.theta(1/annual_coeff),0))
               # self.Vega.loc[idx]=np.dot(self.pos.TradeAmount.values.reshape((1,self.pos.shape[0])),np.nan_to_num(bsm_arr.vega()/100,0))
               self.greeks_dic['theta'].loc[idx]=np.dot(self.pos.TradeAmount.values.reshape((1,self.pos.shape[0])),np.nan_to_num(bsm_arr.theta(1/annual_coeff),0))
               self.greeks_dic['vega'].loc[idx]=np.dot(self.pos.TradeAmount.values.reshape((1,self.pos.shape[0])),np.nan_to_num(bsm_arr.vega()/100,0))
               # self.greeks_dic['delta'].loc[idx]=np.dot(self.pos.TradeAmount.values.reshape((1,self.pos.shape[0])),np.nan_to_num(bsm_arr.delta(),0))

          #       bsm_arr=BSM_ARR(st.s_matr,st.k_matr
          #                         ,(st.ttm_matr-np.argwhere(st.ttm_decay_calendar_days==1).item()-1)/annual_coeff
          #                         , rf ,q
          #                         ,st._StressTestNew__calvolmatrix(st.pos.UnderlyingCode,st.k_div_s,st.ttm_cal_matr-1+st.intra_hours)
          #                         , st.types_matr)
          # aa=np.dot(st.pos.TradeAmount.values.reshape((1,st.pos.shape[0])),np.nan_to_num(bsm_arr.vega()/100,0))
         print("Euro time = ", datetime.now() - tic, "s")
         self.pos_exotic.fillna(0,inplace=True)
         tic=datetime.now()
         
         # self.pos_exotic['cust_bs']=np.where(self.pos_exotic.TradeAmount<0,'B','S')
         for idx in self.norm_idx:
                 tic_i=datetime.now()
                 acc=AccOptionArrSelectItems(greeks_sel,self.total_tradingdates,decay_days
                              ,self.pos_exotic.loc[idx,'OptionType'],self.pos_exotic.loc[idx,'UnderlyingCode']
                              ,self.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
                           ,self.LiV.calVol(self.pos_exotic.loc[idx,'UnderlyingCode'],self.pos_exotic.loc[idx,'Strike']/self.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(self.pos_exotic.loc[idx,'ExerciseDate'])-self.start_time).days+1) 
                           ,self.pos_exotic.loc[idx,'Strike'],self.pos_exotic.loc[idx,'barrier'], float(self.pos_exotic.loc[idx,'fp'])
                           ,max(pd.to_datetime(self.pos_exotic.loc[idx,'FirstObservationDate']).date(),self.curren_trade_date)
                           ,self.pos_exotic.loc[idx,'ExerciseDate'].date()
                           ,self.start_time
                           ,self.pos_exotic.loc[idx,'qty_freq'],self.pos_exotic.loc[idx,'cust_bs'], float(self.pos_exotic.loc[idx,'leverage'])
                           ,float(self.pos_exotic.loc[idx,'StrikeRamp']), float(self.pos_exotic.loc[idx,'BarrierRamp']))
                 res=acc.getResult()[greeks_sel]
                 # self.Theta+=res['theta'].values
                 # self.Vega+=res['vega'].values
                 self.greeks_dic['theta']+=res['theta'].values
                 self.greeks_dic['vega']+=res['vega'].values
                 # self.greeks_dic['delta']+=res['delta'].values

                 print("Exotic each time = ", datetime.now() - tic_i, "s") 
                 
                 
               
          # self.Theta=self.Theta.astype(float).round(0)
          # self.Vega=self.Vega.astype(float).round(0)
         for key in self.greeks_dic.keys():
             self.greeks_dic[key]=self.greeks_dic[key].astype(float).round(0)
         print("Exotic Total time = ", datetime.now() - tic, "s")         
         
         
         for idx in self.barrier_idx:
                tic_i=datetime.now()
                bsb=BarrierAccOptionSelectItems(greeks_sel,self.total_tradingdates,decay_days
                                      , self.pos_exotic.loc[idx,'OptionType'][2:]
                                      , self.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
                                      , self.pos_exotic.loc[idx,'InitialSpotPrice']
                                      , self.pos_exotic.loc[idx,'Strike']
                                      , self.pos_exotic.loc[idx,'barrier']
                                      , self.LiV.calVol(self.pos_exotic.loc[idx,'UnderlyingCode'],self.pos_exotic.loc[idx,'Strike']/self.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(self.pos_exotic.loc[idx,'ExerciseDate'])-self.start_time).days+1) 
                                      , float(self.pos_exotic.loc[idx,'fp'])
                                      , float(self.pos_exotic.loc[idx,'rebate'])
                                      , 1/annual_coeff
                                      , float(self.pos_exotic.loc[idx,'leverage'])
                                      , self.pos_exotic.loc[idx,'cust_bs']
                                      , self.pos_exotic.loc[idx,'qty_freq']
                                      , float(self.pos_exotic.loc[idx,'exp_leverage'])
                                      , self.curren_trade_date  # next_obs_date
                                      , self.pos_exotic.loc[idx,'ExerciseDate'].date()
                                      , self.start_time)
                res=bsb.getResult()[greeks_sel]
                # self.Theta+=res['theta'].values
                # self.Vega+=res['vega'].values
                self.greeks_dic['theta']+=res['theta'].values
                self.greeks_dic['vega']+=res['vega'].values
                # self.greeks_dic['delta']+=res['delta'].values
                print("Barrier each time = ", datetime.now() - tic_i, "s")   
                
         # self.Theta=self.Theta.astype(float).round(0)
         # self.Vega=self.Vega.astype(float).round(0)
         
         for key in self.greeks_dic.keys():
             self.greeks_dic[key]=self.greeks_dic[key].astype(float).round(0)

         print("Barrier total time = ", datetime.now() - tic, "s")         

         
         
     
    def __calPvArr(self,delta_s,trade_time):
        ttm_0=np.array(list(map(lambda x:calTradttm(self.total_tradingdates,trade_time, x),self.exp_t)))
        sigma_arr=np.array(list(map(lambda ass,s,k,exp:self.LiV.calVol(ass,k/(s*delta_s),(pd.to_datetime(exp)-trade_time).days+1)
                            ,self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,self.pos.ExerciseDate))).round(4)

        bsm_arr=BSM_ARR(self.pos.UnderlyingPrice.values*delta_s
                        ,self.pos.Strike.values
                        ,ttm_0/annual_coeff, rf, q,sigma_arr,self.pos.OptionType.values)
        pv=bsm_arr.price()
        pv=np.where(self.pos.OptionType=='F',self.pos.UnderlyingPrice*delta_s,pv)
        pv=pd.Series(index=self.pos.UnderlyingCode,data=pv,dtype=float)
        return pv
    
    
    def __calExoticPvArr(self,delta_s,trade_time):
        # self.pos_exotic.ExerciseDate=self.pos_exotic.ExerciseDate.apply(lambda x:x.date())
        acc_list=list(map(lambda acctype,cust_bs,ass,s,k,b,delta_strike,delta_barrier,fp,lev
                    ,qty,startobs_date, endobs_date:
                        AccOptionArr(self.total_tradingdates, acctype, ass, s*delta_s
                              , self.LiV.calVol(ass,k/(s*delta_s),calTradttm(self.total_tradingdates,trade_time,endobs_date.date())) 
                              ,k,b, float(fp), pd.to_datetime(startobs_date).date(), endobs_date.date()
                              ,trade_time
                              ,qty,cust_bs, float(lev), float(delta_strike), float(delta_barrier)).getResult()
                   
                ,self.pos_exotic.loc[self.norm_idx].OptionType,self.pos_exotic.loc[self.norm_idx].cust_bs,self.pos_exotic.loc[self.norm_idx].UnderlyingCode,self.pos_exotic.loc[self.norm_idx].UnderlyingPrice
                ,self.pos_exotic.loc[self.norm_idx].Strike,self.pos_exotic.loc[self.norm_idx].barrier
                ,self.pos_exotic.loc[self.norm_idx].StrikeRamp,self.pos_exotic.loc[self.norm_idx].BarrierRamp,self.pos_exotic.loc[self.norm_idx].fp,self.pos_exotic.loc[self.norm_idx].leverage
                ,self.pos_exotic.loc[self.norm_idx].qty_freq,self.pos_exotic.loc[self.norm_idx].FirstObservationDate,self.pos_exotic.loc[self.norm_idx].ExerciseDate))
        
     
        pv_exotic=[acc.bookpv for acc in acc_list]
        pv_exotic=pd.Series(index=self.pos_exotic.loc[self.norm_idx].UnderlyingCode,data=pv_exotic,dtype=float)
        pv_exotic=np.where(self.pos_exotic.loc[self.norm_idx].cust_bs=='B',1,-1)*pv_exotic
        return pv_exotic
    

    
    def __calBarrierPvArr(self,delta_s_arr,trade_time):
        # pv_barrier=pd.DataFrame(columns=delta_s_arr)
        pv_barrier=pd.DataFrame(index=self.barrier_idx,columns=delta_s_arr)
        for idx in self.barrier_idx:
            bsb=BarrierAccOptionSelectItems(['price']
                                            ,self.total_tradingdates,0
                                          , self.pos_exotic.loc[idx,'OptionType'][2:]
                                          , self.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
                                          , self.pos_exotic.loc[idx,'InitialSpotPrice']
                                          , self.pos_exotic.loc[idx,'Strike']
                                          , self.pos_exotic.loc[idx,'barrier']
                                          , self.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
                                          , float(self.pos_exotic.loc[idx,'fp'])
                                          , float(self.pos_exotic.loc[idx,'rebate'])
                                          , 1/annual_coeff
                                          , float(self.pos_exotic.loc[idx,'leverage'])
                                          , self.pos_exotic.loc[idx,'cust_bs']
                                          , self.pos_exotic.loc[idx,'qty_freq']
                                          , float(self.pos_exotic.loc[idx,'exp_leverage'])
                                          , self.curren_trade_date  # next_obs_date
                                          , (self.pos_exotic.loc[idx,'ExerciseDate'].date())
                                          , trade_time)
            pv_barrier.loc[idx]=bsb.getResult()['price'].loc[0].values
            
        pv_barrier.index=self.pos_exotic.loc[self.barrier_idx,'UnderlyingCode']
        pv_barrier.rename(index={'decaydays':'UnderlyingCode'},inplace=True)
        # pv_barrier=pv_barrier.sum(axis=0)
        # pv_barrier=[bsb.getResult()['price'] for bsb in bsb_list]
        
        # pv_barrier=pd.Series(index=self.pos_exotic.loc[self.barrier_idx].UnderlyingCode,data=pv_exotic,dtype=float)
        return pv_barrier.astype(float)

class BarrierReport(StressTestNew):
    def __init__(self,current_trade_date,var_list,pricing_time):
        super().__init__(current_trade_date)
        self.cur_tr_d=current_trade_date
        self.pt=pricing_time
        # self.getPos(var_list, "","")
        self.getInfo()
        self.getOTC(var_list)
        
        bar_idx=self.pos_otc.OptionType.dropna()[self.pos_otc.OptionType.dropna().str.contains("b_")].index.tolist()
        self.dfbar=self.pos_otc.loc[bar_idx]
        self.dfbar['RQCode']=self.dfbar.underlyingCode.apply(getRQcode)
        self.dfbar['UnderlyingPrice']=list(map(lambda x:x.last,self.dfbar.RQCode.apply(rqd.current_snapshot)))
        

    def docal(self):    
        self.dfbar['pv']=0
        self.dfbar['delta']=0
        # self.dfres=pd.DataFrame(idx=self.dfbar.index.tolist(),columns=['pv','delta'])
        for idx in self.dfbar.index:
            pdobList=getpdobList(self.dfbar.loc[idx,'RQCode']
                              , self.cur_tr_d
                              , pd.to_datetime(self.dfbar.loc[idx,'startObsDate']).date()
                              , pd.to_datetime(self.dfbar.loc[idx,'maturityDate']).date()
                              , self.pt)
            # self.LiV=LinearInterpVol(self.dfbar.loc[idx,'UnderlyingCode'], self.cur_tr_d)
            vfe=api.getVol_json(str(self.cur_tr_d), self.dfbar.loc[idx,'underlyingCode'])['mid']
            vfe=json.dumps(literal_eval(str(vfe)))
            self.cSV = jsonvolSurface2cstructure_selfapi(vfe)
            res=pyAIKOAccumulatorPricer(self.dfbar.loc[idx,'OptionType'][2:]
                                     , -1 if self.dfbar.loc[idx,'buyOrSell']=="sell" else 1
                                     , self.dfbar.loc[idx,'UnderlyingPrice']
                                     , self.dfbar.loc[idx,'strike']
                                     , datetime2timestamp(self.pt.strftime('%Y-%m-%d %H:%M:%S'))
                                     , datetime2timestamp(self.dfbar.loc[idx,'maturityDate']+" 15:00:00")
                                     , self.dfbar.loc[idx,'entryPrice']
                                     , self.dfbar.loc[idx,'basicQuantity']
                                     # , int(self.dfbar.loc[idx,'isCashSettle'])
                                     , 0
                                     , float(self.dfbar.loc[idx,'leverage'])
                                     , float(self.dfbar.loc[idx,'expireMultiple'])
                                     , float(self.dfbar.loc[idx,'fixedPayment'])
                                     , float(self.dfbar.loc[idx,'barrier'])
                                     , float(self.dfbar.loc[idx,'knockoutRebate'])
                                     , pdobList
                                     , rf, rf
                                     , 0 #const_sgm
                                     , pdobList.shape[0]
                                     # , self.LiV.getVolsurfacejson(self.dfbar.loc[idx,'UnderlyingCode'])
                                     ,self.cSV
                                     # ,vol
                                     )
            self.dfbar.loc[idx,'pv']=res[0]
            
            self.dfbar.loc[idx,'delta']=round(res[1]/rqd.instruments(self.dfbar.loc[idx,'RQCode']).contract_multiplier,0)
            
        
        self.dfbar['B-S_abs']=(self.dfbar.barrier-self.dfbar.UnderlyingPrice).abs()
        self.dfbar['B-S_rate']=(self.dfbar['B-S_abs']/self.dfbar.UnderlyingPrice*100).round(2)

        self.dfbar=self.dfbar[['tradeCode','clientName','OptionType','underlyingCode','UnderlyingPrice','barrier','B-S_abs','B-S_rate','delta','pv']]
        cdt=[(self.dfbar.OptionType.str.contains("put")) &(self.dfbar.UnderlyingPrice>self.dfbar.barrier)
             ,(self.dfbar.OptionType.str.contains("call")) &(self.dfbar.UnderlyingPrice<self.dfbar.barrier)
             ]
        cho=['Live'
             ,'Live']
        self.dfbar['Situation']=np.select(cdt, cho,'Break')
        cdt=[(self.dfbar.Situation=='Live')&(self.dfbar.OptionType.str.contains("put"))
             ,(self.dfbar.Situation=='Live')&(self.dfbar.OptionType.str.contains("call"))
             ]
        cho=["下","上"]
        self.dfbar['Dirt']=np.select(cdt, cho)
        
        
    def show(self):
        self.docal()
        cols=['tradeCode','clientName','OptionType'
               ,'underlyingCode','delta','UnderlyingPrice','Dirt','B-S_abs','barrier'
               ,'B-S_rate','Situation','pv']
        # self.dfbar= self.dfbar[self.dfbar['varity'].isin([v.upper() for v in var_list])]
        self.dfbar=self.dfbar.loc[:,cols]
        self.dfbar=self.dfbar.sort_values(['B-S_rate'])
        if 'Break' in self.dfbar['Situation'].values:
            print('Break!!!!')
            # self.dfbar[self.dfbar.Situation=='Break'].groupby(['UnderlyingCode','ClientName'])['delta'].sum()
        return self.dfbar


class BarrierContracts(StressTestNew):
    def __init__(self,current_trade_date,pricing_time):
        super().__init__(current_trade_date)
        self.getInfo()
        # self.getPos()
        bar_idx=self.dfotc['StructureType'].dropna()
        bar_idx=bar_idx[bar_idx.str.contains("熔断")].index.tolist()
        self.dfbar=self.dfotc.loc[bar_idx]
        self.dfbar.fillna(0,inplace=True)
        self.dfbar['ExerciseDate']=self.dfbar['ExerciseDate'].apply(lambda x:str(x)[:10])
        self.dfbar.StructureType=self.dfbar.StructureType.map({
                                            # "累购期权":"acccall"
                                            # ,"累沽期权":"accput"
                                            # ,"固定赔付累购":"fpcall"
                                            # ,"固定赔付累沽":"fpput"
                                            "熔断累购期权":"acccall"
                                            ,"熔断累沽期权":"accput"
                                            ,"熔断固赔累购":"fpcall"
                                            ,"熔断固赔累沽":"fpput"
                                            ,"熔断增强累购":"acccallplus"
                                            ,"熔断增强累沽":"accputplus"})
     
        
        


def HeatMap(st_result):
    fig_g=plt.figure(figsize=(14,9))
    ax_t=fig_g.add_subplot(211)
    ax_t=sns.heatmap((st_result['theta']/10000).round(2)
                      ,linewidth=0.5
                      ,annot=True
                      ,annot_kws={'fontsize':8}
                      ,cbar=False
                     )
    ax_t.xaxis.tick_top()
    ax_t.tick_params(labelsize=8)
    v=",".join(varity)
    ax_t.set_title(v.upper()+" Theta(w)")
    ax_v=fig_g.add_subplot(212)
    ax_v=sns.heatmap((st_result['vega']/10000).round(2),linewidth=0.5
                     ,annot=True
                     ,annot_kws={'fontsize':8},cbar=False,cmap='crest')
    ax_v.xaxis.tick_top()
    ax_v.tick_params(labelsize=8)
    ax_v.set_title(v.upper()+" Vega(w)")
    plt.tight_layout()
    
    
def TotalFig(v,st_each):
      fig = plt.figure(figsize=(17,9),tight_layout=True)
      gs = gridspec.GridSpec(2, 3)
    
      ax_t=fig.add_subplot(gs[0,:2])
      ax_t=sns.heatmap((st_each['theta']/10000).round(2),linewidth=0.5
                       ,annot=True
                       ,annot_kws={'fontsize':7},cbar=False)
      ax_t.xaxis.tick_top()
      ax_t.tick_params(labelsize=7)
      ax_t.set_title(v.upper()+" Theta(w)")
      
      ax_v=fig.add_subplot(gs[1,:2])
      ax_v=sns.heatmap((st_each['vega']/10000).round(2),linewidth=0.5
                       ,annot=True
                       ,annot_kws={'fontsize':7},cbar=False,cmap='crest')
      ax_v.xaxis.tick_top()
      ax_v.tick_params(labelsize=7)
      ax_v.set_title(v.upper()+" Vega(w)")
      # plt.tight_layout()
      
      pnl=st_each['pnl'].sum()
      ax_pnl=fig.add_subplot(gs[0,2])
      ax_pnl.plot(pnl/10000)
      ax_pnl.set_ylabel('Pnl(w)')
      ax_pnl.grid(True)
      ax_pnl.set_title(v.upper())
      
      ax2=fig.add_subplot(gs[1,2])
      ax2.plot(pnl.loc[0.97:1.03],'-o',ms=8,mfc='orange')
      # ax2.set_xlabel('1 PNL: '+str(pnl.loc[1].round(0))+' 0.99 PNL: '+str(pnl.loc[0.99].round(0))+' 1.01 PNL:'+str(pnl.loc[1.01].round(0)))
      ax2.annotate(str(pnl.loc[1].round(0)),xy=(1,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
      ax2.annotate(str(pnl.loc[0.99].round(0)),xy=(0.99,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
      ax2.annotate(str(pnl.loc[1.01].round(0)),xy=(1.01,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
      ax2.set_ylabel('Pnl')
      ax2.grid(True)

def formatPos_simu_dic(info_dic,units):
        under_list,strike_list,exp_list,opttype_list,trdamt_list=[],[],[],[],[]
        for k in info_dic.keys():
            if info_dic[k]=="":
                pass
            else:
                under_list+=info_dic[k]['underlyings']*len(info_dic[k]['strikes'])
                strike_list+=info_dic[k]['strikes']
                exp_list+=info_dic[k]['exp_d']*len(info_dic[k]['strikes'])
                opttype_list+=info_dic[k]['opt_types'] if len(info_dic[k]['opt_types'])==len(info_dic[k]['strikes']) else info_dic[k]['opt_types']*len(info_dic[k]['strikes'])
                trdamt_list+=info_dic[k]['trade_amt'] if len(info_dic[k]['trade_amt'])==len(info_dic[k]['strikes']) else info_dic[k]['trade_amt']*len(info_dic[k]['strikes'])
        
        pos_simu_dic={'UnderlyingCode': under_list,
                      'Strike':np.array(strike_list),
                      'ExerciseDate':pd.to_datetime(np.array(exp_list)),
                      'OptionType':opttype_list,
                      'TradeAmount':np.array(trdamt_list)*units}
        return pos_simu_dic
    
def CheckForward(cur_d,client_name,variety_list=[],underlying_list=[],flag=1):
    pd.set_option('display.unicode.ambiguous_as_wide',True)
    pd.set_option('display.unicode.east_asian_width',True)
    pd.set_option('display.width',200)
    if flag==1:
        dffor=api.getOTC_LiveTrade(cur_d,variety_list,underlying_list,["f"])
        dffor['ClientDirt']=np.where(dffor.buyOrSellName=="买入","空头","多头")
        dffor['Lots']=dffor.availableVolume/dffor.underlyingCode.apply(getRQcode).apply(findMultiplier)
        dffor['availableVolume']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.availableVolume
        dffor['Lots']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.Lots
        a=dffor[dffor.clientName.str.contains(client_name)].groupby(['underlyingCode',"ClientDirt",'strike'])['availableVolume','Lots'].sum()
        b=a.reset_index().groupby(['underlyingCode','ClientDirt']).apply(lambda x:(x['strike']*x['Lots']).sum()/x['Lots'].sum()).round(2)
        a.index.rename({"underlyingCode":"标的合约"
                    ,"ClientDirt":"客户方向"
                    ,"strike":"开仓价格"},inplace=True)
        a.rename(columns={"availableVolume":"存续数量"
                          ,"Lots":"存续手数"}
                         ,inplace=True)
        
        b.index.rename({"underlyingCode":"标的合约"
                    ,"ClientDirt":"客户方向"},inplace=True)
        b.name='持仓均价'

    else:
        dffor=api.getOTC_closed_onDate(cur_d,underlying_list,["f"])
        # dffor['CloseDirt']=np.where(dffor['closeVolume']<0,"平空","平多")
        dffor['Lots']=dffor.closeVolume/dffor.underlyingCode.apply(getRQcode).apply(findMultiplier)
        a=dffor[dffor.clientName.str.contains(client_name)].groupby(["underlyingCode"])['closeVolume',"Lots"].sum()
        # a.index.rename({"underlyingCode":"标的合约"
        #             # ,"CloseDirt":"客户方向"
        #             },inplace=True)
        # a.rename(columns={"closeVolume":"平仓数量"
        #                   ,"Lots":"平仓手数"}
        #                  ,inplace=True)
    
    print("------------------"+dffor[dffor.clientName.str.contains(client_name)].clientName.unique()[0]+"------------------------------------------------")
    print("")
    print(a)
    print("")
    print("")
    print("------------------"+"各标的多空合计"+"--------------------------------------------------------")
    print("")
    aa=a.groupby(["标的合约","客户方向"]).sum()
    bb=pd.merge(b,aa,on=b.index)
    bb.rename(columns={"key_0":"key_1"
                      }
                     ,inplace=True)
    bb=pd.merge(bb[bb.columns[0]].apply(pd.Series),bb,on=bb.index)
    bb.drop(columns=['key_0','key_1'],inplace=True)
    bb.rename(columns={0:'标的合约',1:'客户方向'},inplace=True)
    print(bb.groupby(["标的合约","客户方向"]).last())
    
    
    
    return a,bb
    
if __name__=='__main__':
        delta_s_arr=np.arange(0.9,1.1,0.01).round(2)
        # delta_s_arr=np.arange(0.99,1.02,0.002).round(3)
        api=SelfAPI()
        
      #%%
if __name__=='__main__':
        end_trading_time=datetime(2024,11,19,14,59,0)
        cur_trad_date=datetime(2024,11,18).date() 
        st=StressTestNew(cur_trad_date)
        #%% stresstest
if __name__=='__main__':
        tic=datetime.now()
        varity=['V']
        # if len(varity)>2:
        #     units=rqd.instruments(getRQcode(varity)).contract_multiplier
        # else:
        #     units=rqd.instruments(varity.upper()+'99').contract_multiplier
        units=[rqd.instruments(v.upper()+'99').contract_multiplier for v in varity][0]
        info_dic={
                        # 'conts_f':{'underlyings':["UR501"]
                        #               ,'strikes':[1824]
                        #               ,'exp_d':[datetime(2024,12,27).date()]
                        #               , 'opt_types':['F']
                        #               ,'trade_amt':[24]
                        #               }
                    
                        # 'conts_1':{'underlyings':["CF501"]
                        #                       ,'strikes':[13600,13800,14000]
                        #                       ,'exp_d':[datetime(2024,12,11).date()]
                        #                       , 'opt_types':['P']
                        #                       ,'trade_amt':[-300]
                        #                         }
                        'conts_1':{'underlyings':['V2502']
                                    ,'strikes':[5200,5300,5400,5500]
                                    ,'exp_d':[datetime(2025,1,17).date()]
                                    , 'opt_types':['P']*2+['C']*2
                                    ,'trade_amt':[-300]
                                    }
                    #     'conts_1':{'underlyings':['AG2412']
                    #                 ,'strikes':[7100,7700,7800]
                    #                 ,'exp_d':[datetime(2024,11,25).date()]
                    #                 , 'opt_types':['P']
                    #                 ,'trade_amt':[40]+[20]*2
                    #                 }
                    # ,
                    # 'conts_2':{'underlyings':["AG2502"]
                    #                       ,'strikes':[8100,8200,8300]
                    #                       ,'exp_d':[datetime(2025,1,24).date()]
                    #                       , 'opt_types':['C']
                    #                       ,'trade_amt':[-10]
                    #                       }
                    # 'conts_2':{'underlyings':["CF503"]
                    #                       ,'strikes':[14000,13800]
                    #                       ,'exp_d':[datetime(2025,2,12).date()]
                    #                       , 'opt_types':['P']
                    #                       ,'trade_amt':[-150]
                    #                         }
                        # 'conts_2':{'underlyings':["AU2411"]
                        #                       ,'strikes':[576,592,584]
                        #                       ,'exp_d':[datetime(2024,10,25).date()]
                        #                       , 'opt_types':['C']
                        #                       ,'trade_amt':[-20]
                        #                       }|
                    }
        # start_time=datetime(2024,10,23,22,50,0)
        start_time=datetime.now()
        decay_days=20
        # additional_filter={'TradeNumber':["20240118-TXF-01-2","20240118-TXF-01-1"]}
        st_res_arr=st.calStressTestNew(varity, delta_s_arr, decay_days, start_time, end_trading_time
                                    # ,pos_simu_dic=formatPos_simu_dic(info_dic,units)
                                    # ,additional_filter
                                ,flag_pv=False
                                ,fig=False
                                )
        
        print("Running time = ", datetime.now() - tic, "s")
        st_theta_arr=st_res_arr['theta']
        st_vega_arr=st_res_arr['vega'] 
        st.pos.UnderlyingPrice.groupby(st.pos.UnderlyingCode).last()*0.98
        
        
        
        
      # %% heatmap
if __name__=='__main__':
        HeatMap(st_res_arr)
    
       #%% BarrierCheck
if __name__=='__main__':
        brt=BarrierReport(cur_trad_date
                          ,['PR','CF','MA','EG','EB','UR','V','PP','L','AG','AU','AL','SH']
                          ,datetime.now())
        dfbar=brt.show()
        pd.set_option('display.unicode.ambiguous_as_wide',True)
        pd.set_option('display.unicode.east_asian_width',True)
        pd.set_option('display.width',200)
        dfbar[dfbar.Situation=='Break'].groupby(['underlyingCode','clientName','barrier','OptionType'])['delta'].sum()
     
    #%%['MA','EG','EB','UR','V','PP','L','AG','AU','AL']
if __name__=='__main__':
        decay_days=20    
        for varity in ['CF'
                        ,'MA','EG'
                         ,'EB'
                          ,'UR','PR'
                          ,'V','PP','L','SH','AG'
                       # ,'AU','AL'
                       ]:
            # start_time=datetime.now()
            try:
                st_each=st.calStressTestNew([varity], delta_s_arr, decay_days
                                            , datetime.now(), end_trading_time
                                            # ,pos_simu_dic=formatPos_simu_dic(info_dic,units)
                                            # ,additional_filter
                                            ,fig=False
                                        )
                TotalFig(varity,st_each)
            except:
                continue
#%%    
# getTotalPnLData(file_path=r'D:\chengyilin\work\2.Trading\总结\pnl.xlsx'
#                     , end_d="2024-09-20")

    
#%% Client Forward Check!
if __name__=='__main__':
        client_name="健创"
        variety_list=["SI"]
        underlying_list=[]
        flag=1
        # a,bb=CheckForward(str(cur_trad_date),client_name,variety_list,underlying_list,flag)
        a,bb=CheckForward("2024-09-30",client_name,variety_list,underlying_list,flag)

# %%   Expired Forward Check!
if __name__=='__main__':
        varietyList=['PR','CF','MA','EG','EB','UR','V','PP','L','SH','AG']
        # client_name="健创"
        end_date="2024-11-14"
        dflive_fut=api.getOTC_LiveTrade(str(cur_trad_date),varietyList=varietyList
                                        # ,optiontype_Name=['f']
                                        )
        dfexpired_fut=dflive_fut[dflive_fut.maturityDate==end_date]
        dfexpired_fut['ClientDirt']=np.where(dfexpired_fut.buyOrSellName=="买入","空头","多头")
        dfexpired_fut['Lots']=dfexpired_fut.availableVolume/dfexpired_fut.underlyingCode.apply(getRQcode).apply(findMultiplier)
        dfexpired_fut['Lots']=np.where(dfexpired_fut['ClientDirt']=="空头",-1,1)*dfexpired_fut.Lots
        # dfexpired_fut.groupby(['clientName','underlyingCode'])['availableVolume'].sum()
        dfexpired_fut.groupby(['underlyingCode','clientName','strike'])['Lots'].sum()

        
        # variety_list=["EG"]
        # underlying_list=[]
#%%
if __name__=='__main__':
    open_date=datetime(2024,10,25).date()
    opt_num=3
    
    params={'s_0':[1845]*opt_num
            ,'strike':[1845,1801,1801]
            ,'opttype':['P']*opt_num
            ,'buysell':[1,-1,-1]
            ,'exp_date':[datetime(2024,11,25).date()]*opt_num
            ,'open_vol':[0.17]*opt_num
            ,'close_vol':[0.17]*opt_num
            }
    
    s_ts=np.arange(1780,1820,1)
    # delta_arr=np.arange(0.8,1.2,0.04)
    # s_ts=delta_arr*params['s_0'][0]
    
    
    trd_idx=rqd.get_trading_dates(open_date, params['exp_date'][0])
    t=calTradttm(trd_idx,datetime.now(), params['exp_date'][0])/annual_coeff
    bsm_arr=BSM_ARR(np.array(params['s_0'])
                    , np.array(params['strike'])
                    , np.ones(opt_num)*t
                    , rf, q
                    , np.array(params['open_vol'])
                    , np.array(params['opttype'])
                    )
    
    # premium=(bsm_arr.price()*np.array(params['buysell'])).sum()
    premium=0
    
    
    
    ttm_ts=list(map(lambda t:calTradttm(trd_idx,datetime.now(),t),trd_idx))[::-1]
    ttm_ts.append(0)
    ttm_ts=np.array(ttm_ts)/annual_coeff
    t_shape=ttm_ts.shape[0]
    s_arr=s_ts.repeat(opt_num)
    s_shape=s_ts.shape[0]
    k_arr=params['strike']*int(s_shape)
    s_arr=s_arr.repeat(t_shape)
    k_arr=np.array(k_arr*t_shape)
    t_arr=np.array((ttm_ts.repeat(opt_num)).tolist()*int(s_shape))
    sigma_arr=np.array(params['close_vol']*int(t_shape*s_shape))
    opttype_arr=np.array(params['opttype']*int(t_shape*s_shape))
    
    bsm_arr=BSM_ARR(s_arr
                    , k_arr
                    , t_arr
                    , rf, q
                    , sigma_arr
                    , opttype_arr
                    )
    close_p=bsm_arr.price()*np.array(params['buysell']*int(t_shape*s_shape))
    close_prem=close_p.reshape(int(t_shape*s_shape),opt_num).sum(axis=1)
    
    
    trd_idx.append(params['exp_date'][0])
    res=pd.DataFrame(columns=(s_ts).round(0),index=trd_idx
                      ,data=close_prem.reshape(s_shape,t_shape).T)
    res=res.round(2)-premium
    # final_pnl=res.iloc[np.arange(52,0,-5)[::-1]]
       
#%% pnl 
if __name__=='__main__':
    def thetaPnL(s_arr,k_arr,t_arr,sigma_arr,opttype_arr):
        return BSM_ARR(s_arr, k_arr, t_arr/annual_coeff, rf, q, sigma_arr, opttype_arr).price()-BSM_ARR(s_arr,k_arr
                            , (t_arr+1)/annual_coeff, rf, q,sigma_arr, opttype_arr).price()
    
    def  gammaPnL(s_t_arr,s_0_arr,k_arr,t_arr,r,q,sigma_arr,opttype_arr,pos_arr,delta_arr):
        return (BSM_ARR(s_t_arr, k_arr, t_arr, r, q, sigma_arr, opttype_arr).price()-BSM_ARR(s_0_arr, k_arr, t_arr, r, q, sigma_arr, opttype_arr).price())*pos_arr-delta_arr
    
    def getSettle(settle_type):
        if settle_type=="cash":
            return 1
        elif settle_type=='physical':
            return 0
        elif settle_type=='mix':
            return 2
        else:
            return "wrong settle type!"
    def getVolsurfacejson(trade_date,underlyingCode):
      
        vfe=api.getVol_json(str(trade_date),underlyingCode)['mid']
        vfe=json.dumps(literal_eval(str(vfe)))
        return jsonvolSurface2cstructure_selfapi(vfe)
    variety=['MA']

    pre_trade_d="2024-11-15"
    cur_trade_d="2024-11-18"
    trading_date_list=rqd.get_trading_dates(cur_trade_d,"2025-12-31")

    itc_trading=api.getITCTrade_onDate(cur_trade_d)
    # itc_trading['instrumentId']=itc_trading['instrumentId'].str.upper()
    itc_trading['amt']=itc_trading.price*itc_trading.volumeCount
    cur_trading=api.getCurrentRisk(variety)
    underlyingCode_list=cur_trading.underlyingCode.unique().tolist()

    last_trading=api.getTradeRisk(pre_trade_d,underlyingCodeList=underlyingCode_list,live_contract=True)
    last_trading['delta']=last_trading.delta.astype(float)
    last_trading['deltaLots']= last_trading.deltaLots.astype(float)
    df_s=rqd.get_price([getRQcode(u)for u in underlyingCode_list],pre_trade_d,pre_trade_d,fields='close').reset_index()[['order_book_id','close']]
    df_s['last']=[i.last for i in rqd.current_snapshot([getRQcode(u)for u in underlyingCode_list])]
    df_s['mult']=[i.contract_multiplier for i in rqd.instruments([getRQcode(u)for u in underlyingCode_list])]
    # df_s['underlyingCode']=[i.trading_code for i in rqd.instruments([getRQcode(u)for u in underlyingCode_list])]
    df_s.rename(columns={'order_book_id':'underlyingCode'},inplace=True)
    df_s.set_index('underlyingCode',inplace=True)
    last_trading.set_index('id',inplace=True)


    last_trading.delta.fillna(last_trading.underlyingCode.map(df_s['mult']).values*last_trading.deltaLots,inplace=True)

 
    cur_trading['opttype']=cur_trading.optionType.map({"AICallAccPricer":"acccall"
                                        ,"AIPutAccPricer":"accput"
                                        ,"AICallFixAccPricer":"fpcall"
                                        ,"AIPutFixAccPricer":"fpput"
                                        ,"AICallKOAccPricer":"bacccall"
                                        ,"AIPutKOAccPricer":"baccput"
                                        ,"AICallFixKOAccPricer":"bfpcall"
                                        ,"AIPutFixKOAccPricer":"bfpput"
                                        ,"AIEnCallKOAccPricer":"bacccallplus"
                                        ,"AIEnPutKOAccPricer":"baccputplus"
                                        ,'AIForwardPricer':"F"
                                        ,'AIVanillaPricer':"V"})
    cur_trading['opttype'].fillna('F',inplace=True)
    cur_trading.callOrPut.fillna('Other',inplace=True)
    cur_trading.optionType.fillna('F',inplace=True)
    
    cdt=[cur_trading.callOrPut=='call'
         ,cur_trading.callOrPut=='put']
    cho=['C','P']
    cur_trading.callOrPut=np.select(cdt,cho,default=cur_trading.callOrPut)
    

    
    
   
    
    cur_trading['delta_0']=cur_trading['id'].map(last_trading.delta)
    cur_trading['s_0']=cur_trading['id'].map(last_trading.lastPrice)
    cur_trading['vol_0']=cur_trading['id'].map(last_trading.nowVol)
    cur_trading['pos_0']=cur_trading['id'].map(last_trading.availableVolume)*np.where(cur_trading.buyOrSell=='buy',1,-1)
    
    cur_trading['deltaPnL']=cur_trading.delta_0*(cur_trading.lastPrice-cur_trading.s_0)
    
    cur_trading['maturityDate'].fillna(datetime.now().date(),inplace=True)
    cur_trading['maturityDate']=cur_trading.maturityDate.apply(lambda x:pd.to_datetime(x).date())
    
    cur_trading['ttm']=cur_trading.maturityDate.apply(lambda x: calTradttm(trading_date_list, datetime.now(),x))
    cur_trading.loc[cur_trading['opttype'] == 'F', 'ttm']=0
    cur_trading.strike.fillna(cur_trading.lastPrice,inplace=True)
    cur_trading['thetaPnL']=0
    cur_trading.loc[cur_trading.opttype=='V','thetaPnL']=thetaPnL(cur_trading[cur_trading['opttype'] == 'V']['s_0'].values
         ,cur_trading[cur_trading['opttype'] == 'V']['strike'].values
         ,cur_trading[cur_trading['opttype'] == 'V']['ttm'].values
         ,cur_trading[cur_trading['opttype'] == 'V']['vol_0'].values/100
         , cur_trading[cur_trading['opttype'] == 'V']['callOrPut'].values)
    cur_trading['thetaPnL']=cur_trading['thetaPnL']*cur_trading['pos_0']
    #pv_t-pv(s_0,v_0,t)
    cur_trading['vaegPnL']=0
    cur_trading.loc[cur_trading.opttype=='V','vegaPnL']=cur_trading[cur_trading['opttype'] == 'V'].availablePremium.abs()-BSM_ARR(cur_trading[cur_trading['opttype'] == 'V'].lastPrice.values
                                                                              ,cur_trading[cur_trading['opttype'] == 'V'].strike.values
                                                                              ,cur_trading[cur_trading['opttype'] == 'V'].ttm.values/annual_coeff
                                                                              ,rf,q
                                                                              ,cur_trading[cur_trading['opttype'] == 'V'].vol_0.values/100
                                                                              ,cur_trading[cur_trading['opttype'] == 'V'].callOrPut.values).price()
    cur_trading['vegaPnL']=cur_trading['vegaPnL']*cur_trading['pos_0']
                                                                        
    
    cur_trading['gammaPnL']=0
    cur_trading.loc[cur_trading.opttype=='V','gammaPnL']=gammaPnL(cur_trading[cur_trading['opttype'] == 'V'].lastPrice.values
             ,cur_trading[cur_trading['opttype'] == 'V'].s_0.values
             ,cur_trading[cur_trading['opttype'] == 'V'].strike.values
             ,cur_trading[cur_trading['opttype'] == 'V'].ttm.values/annual_coeff
             , rf, q
             ,cur_trading[cur_trading['opttype'] == 'V'].vol_0.values/100
             ,cur_trading[cur_trading['opttype'] == 'V'].callOrPut.values
             ,cur_trading[cur_trading['opttype'] == 'V'].pos_0.values
             ,cur_trading[cur_trading['opttype'] == 'V'].deltaPnL.values)
    
    cur_trading['amt']=cur_trading.instrumentId.map(itc_trading.groupby('instrumentId')['amt'].sum())
    cur_trading['volumeCount']=cur_trading.instrumentId.map(itc_trading.groupby('instrumentId')['volumeCount'].sum())
    cur_trading['PnLtrade']=cur_trading.volumeCount*cur_trading.availablePremium.abs()-cur_trading.amt
    
    
    
    
    
    
    
    
    for idx in cur_trading.index:
        pdob_0=getpdobList(getRQcode(cur_trading.loc[idx,'underlyingCode'])
                    ,pd.to_datetime(pre_trade_d).date()
                    ,pd.to_datetime(cur_trading.loc[idx,'startObsDate']).date()
                    ,cur_trading.loc[idx,'maturityDate']
                    ,datetime.combine(pd.to_datetime(pre_trade_d).date(),time(15,0,0))
                    ,show_end_date=pd.to_datetime(cur_trade_d).date())
        pdob_t=getpdobList(getRQcode(cur_trading.loc[idx,'underlyingCode'])
                    ,pd.to_datetime(cur_trade_d).date()
                    ,pd.to_datetime(cur_trading.loc[idx,'startObsDate']).date()
                    ,cur_trading.loc[idx,'maturityDate']
                    ,datetime.combine(pd.to_datetime(cur_trade_d).date(),time(15,0,0))
                    ,show_end_date="")
        
        v_0=getVolsurfacejson(pre_trade_d,cur_trading.loc[idx,'underlyingCode'])
        v_t=getVolsurfacejson(cur_trade_d,cur_trading.loc[idx,'underlyingCode'])
        s_0=cur_trading.loc[idx,'s_0']
        s_t=cur_trading.loc[idx,'lastPrice']
        t_0=datetime2timestamp(pre_trade_d+" 15:00:00")
        t_t=datetime2timestamp(cur_trade_d+" 15:00:00")
        #theta pnl= pv(s_0,v_0,t_t)-pv(s_0,v_0,t_0)
        p1=pyAIAccumulatorPricer(cur_trading.loc[idx].opttype
                                   , -1 if cur_trading.loc[idx,'buyOrSell']=='sell' else 1
                                   , s_0
                                   , cur_trading.loc[idx,'strike']
                                   # , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , t_t#datetime2timestamp(pre_trade_d+" 15:00:00")
                                   , datetime2timestamp(str(cur_trading.loc[idx,'maturityDate'])[:10]+" 15:00:00")
                                   , cur_trading.loc[idx,'basicQuantity']
                                   , getSettle(cur_trading.loc[idx,'settleType'])
                                   , cur_trading.loc[idx,'leverage']
                                   , cur_trading.loc[idx,'fixedPayment']
                                   , cur_trading.loc[idx,'barrier']
                                   , cur_trading.loc[idx,'strikeRamp']
                                   , cur_trading.loc[idx,'barrierRamp']
                                   , pdob_0
                                   , rf
                                   , 0
                                   , pdob_0.shape[0]
                                   , v_0
                                   , 0
                                   )[0]
        
        # p2=pyAIAccumulatorPricer(cur_trading.loc[idx].opttype
        #                            , -1 if cur_trading.loc[idx,'buyOrSell']=='sell' else 1
        #                            , s_0
        #                            , cur_trading.loc[idx,'strike']
        #                            # , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                            , t_0#datetime2timestamp(cur_trade_d+" 15:00:00")
        #                            , datetime2timestamp(str(cur_trading.loc[idx,'maturityDate'])[:10]+" 15:00:00")
        #                            , cur_trading.loc[idx,'basicQuantity']
        #                            , getSettle(cur_trading.loc[idx,'settleType'])
        #                            , cur_trading.loc[idx,'leverage']
        #                            , cur_trading.loc[idx,'fixedPayment']
        #                            , cur_trading.loc[idx,'barrier']
        #                            , cur_trading.loc[idx,'strikeRamp']
        #                            , cur_trading.loc[idx,'barrierRamp']
        #                            , pdob_0
        #                            , rf
        #                            , 0
        #                            , pdob_0.shape[0]
        #                            , v_0
        #                            , 0
        #                            )[0]
        thetapnl=p1-p2
        thetapnl=p1-last_trading.loc[cur_trading.loc[idx,'tradeCode']].availablePremium
        #vega pnl=pv_t(s_t,v_t,t_t)-pv(s_t,v_0,t_t)
        p3=pyAIAccumulatorPricer(cur_trading.loc[idx].opttype
                                   , -1 if cur_trading.loc[idx,'buyOrSell']=='sell' else 1
                                   , cur_trading.loc[idx,'s_0']
                                   , cur_trading.loc[idx,'strike']
                                   # , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , datetime2timestamp(cur_trade_d+" 15:00:00")
                                   , datetime2timestamp(str(cur_trading.loc[idx,'maturityDate'])[:10]+" 15:00:00")
                                   , cur_trading.loc[idx,'basicQuantity']
                                   , getSettle(cur_trading.loc[idx,'settleType'])
                                   , cur_trading.loc[idx,'leverage']
                                   , cur_trading.loc[idx,'fixedPayment']
                                   , cur_trading.loc[idx,'barrier']
                                   , cur_trading.loc[idx,'strikeRamp']
                                   , cur_trading.loc[idx,'barrierRamp']
                                   , pdob_t
                                   , rf
                                   , 0
                                   , pdob_t.shape[0]
                                   , getVolsurfacejson(pre_trade_d,cur_trading.loc[idx,'underlyingCode'])
                                   , 0
                                   )[0]
        cur_trading.loc[idx].availablePremium-p3
        
    
    
    
    
    
    
    
    
    res=cur_trading.groupby(['underlyingCode'])[['deltaPnL','gammaPnL','thetaPnL','vegaPnL']].sum()
    # res['pnl_itc']=0
    res=pd.merge(res, cur_trading.groupby('underlyingCode')['PnLtrade'].sum().reset_index(), left_index=True, right_on='underlyingCode', how='left')    
#%%
# a=api.getTradeRisk("2024-10-28",optiontype_Name=['van'],live_contract=True)
# t_list=list(map(lambda t:calTradttm(st.total_tradingdates,datetime(2024,10,28,15,0,0),pd.to_datetime(t)),a.maturityDate))
# s_arr=a.lastPrice.values
# k_arr=a.strike.values
# t_arr=np.array(t_list)/244
# sigma_arr=a.nowVol.values/100
# opttype_arr=np.where(a.callOrPut=='call','C','P')

# p_new=BSM_ARR(s_arr,k_arr,t_arr,rf,q,sigma_arr,opttype_arr).price()


# amt_new=np.where(a.buyOrSellName=='买入',-1,1)*p_new*a.availableVolume
# (amt_new-a.availableAmount).sum()




#%% 查看客户收盘价了结生成头寸
if __name__=='__main__':
    api=SelfAPI()
    dftotal=api.getOTC_LiveTrade("2024-10-21",varietyList=['EG'])
    dfacc=dftotal[dftotal.optionType.str.contains("Acc")]
    client_list=["诚汇金"
                  ]
    tradecode_list=[#'20240717-SHKY-04','20240717-SHKY-07','20240717-SHKY-05'
                    ]
    
    
    dfselect=pd.DataFrame()
    if len(client_list)>0:
        for cl in client_list:
            dfselect=pd.concat([dfselect,dfacc[dfacc.clientName.str.contains(cl)]])
    elif len(tradecode_list)>0:
        for cl in tradecode_list:
            dfselect=pd.concat([dfselect,dfacc[dfacc.tradeCode.str.contains(cl)]])
    else:
        dfselect=dfacc
    
    strike_ts=dfselect.strike.values
    barrier_ts=dfselect.barrier.values
    settle_ts=dfselect.settleType.values
    qty_ts=dfselect.basicQuantity.values
    lev_ts=dfselect.leverage.values        
    s_ts=dfselect.underlyingCode.apply(lambda x:rqd.current_snapshot(getRQcode(x)).last)
            
    cdt=[(dfselect.optionTypeName.str.contains("累购"))&(s_ts>=strike_ts)&(s_ts<barrier_ts)&(settle_ts=="physical")
            ,(dfselect.optionTypeName.str.contains("累购"))&(s_ts<strike_ts)&(settle_ts!="cash")
    
            ,(dfselect.optionTypeName.str.contains("累沽"))&(s_ts<=strike_ts)&(s_ts>barrier_ts)&(settle_ts=="physical")
            ,(dfselect.optionTypeName.str.contains("累沽"))&(s_ts>strike_ts)&(settle_ts!="cash")
        
            ]
            
    cho=[qty_ts
            ,qty_ts*lev_ts
            ,qty_ts
            ,qty_ts*lev_ts
    
            ]
            
    dfselect['Generate_fut']=np.select(cdt,cho,default=0)
    cdt=[(dfselect.optionTypeName.str.contains("固定"))&(s_ts>=strike_ts)&(s_ts<barrier_ts)
          ,(dfselect.optionTypeName.str.contains("固定"))&(s_ts<=strike_ts)&(s_ts>barrier_ts)]        
    cho=[0,0]
    dfselect['Generate_fut']=np.select(cdt,cho,1)*dfselect.Generate_fut/dfselect.underlyingCode.apply(lambda x:rqd.instruments(getRQcode(x)).contract_multiplier).values
    dfselect['Generate_fut_direct']=np.where(dfselect.optionTypeName.str.contains("购"),"多头(手数)","空头(手数)")
    a=dfselect.groupby(['underlyingCode','clientName','Generate_fut_direct'])['Generate_fut'].sum()
    a=a.unstack(fill_value=0)
    pd.set_option('display.unicode.ambiguous_as_wide',True)
    pd.set_option('display.unicode.east_asian_width',True)
    pd.set_option('display.width',200)
    # print(a)
    a.groupby(level='underlyingCode').sum()
            

#%%
        # api=SelfAPI() 
        # dftotal=api.getTotalOTCTrade("2024-01-01", "2024-06-06")
        # dftotal.drop(index=dftotal[dftotal.optionType=='AIForwardPricer'].index.tolist(),inplace=True)
        # dftotal['notional']=dftotal.notionalPrincipal/100000000
        # dftotal.varietyName=dftotal.underlyingCode.apply(findCode)
        # dftotal.tradeDate=pd.to_datetime(dftotal.tradeDate)
        # # dftotal['month']=dftotal.tradeDate.apply(lambda x:x.month)
        # dfnotion=dftotal.groupby(['tradeDate','varietyName'])[['notional','day1PnL']].sum().unstack()
        # dfnotion.fillna(0,inplace=True)
        # dfpnl=pd.DataFrame()
        # for trd in dftotal.tradeDate.unique():
        #     dfrisk=api.getTradeRisk(str(trd)[:10])
        #     dfrisk['varietyName']=dfrisk['underlyingCode'].apply(findCode)
        #     # dfrisk['day1PnL'].fillna("0.00",inplace=True)
        #     for col in ['todayProfitLoss','theta','vega']:
        #         dfrisk[col]=dfrisk[col].apply(lambda x:float(x.replace(",","")))
        #     dfrisk=dfrisk.groupby('varietyName')[['todayProfitLoss','theta','vega']].sum().reset_index()
        #     dfrisk['settleDate']=str(trd)[:10]
        #     dfrisk['notion']=dfrisk['varietyName'].map(dfnotion.loc[trd]['notional'])
        #     dfrisk['day1']=dfrisk['varietyName'].map(dfnotion.loc[trd]['day1PnL'])
        #     dfpnl=pd.concat([dfpnl,dfrisk])
        # dfpnl.notion.fillna(0,inplace=True)
        # dfpnl.to_excel(r'D:\chengyilin\work\2.Trading\总结\pnl.xlsx')

#%%

# pre_risk=api.getTradeRisk("2024-10-28",underlyingCodeList=['CF505'],live_contract=True)
# cur_risk=api.getTradeRisk("2024-10-29",underlyingCodeList=['CF505'],live_contract=True)

# itc_trading=api.getITCTrade_onDate("2024-10-29")
# itc_trading['TradeAmount']=itc_trading.price*itc_trading.volumeCount
# itc_trading_ts=itc_trading.groupby('instrumentId')['TradeAmount'].sum()

# pre_risk.optionType



        
        #%%
        # 
        # def get_volsuf(trade_date,underlyingCode,vol_type):
        #     '''
        #     Parameters
        #     ----------
        #     trade_date : str
        #                  eg. "2024-05-13"
        #     underlyingCode : str
        #     vol_type : str
        #         "ask","bid","mid"

        #     Returns
        #     -------
        #     None.

        #     '''
        #     vol=api.getVol_json(trade_date, underlyingCode)[vol_type]
        #     vfe=json.dumps(literal_eval(str(vol)))
        #     cSV = jsonvolSurface2cstructure_selfapi(vfe)
        #     return cSV
        
        # api=SelfAPI() 
        # underlyingCode="V2501"
        # cur_trd_date=datetime(2024,5,27).date()
        # # cSV=get_volsuf(str(cur_trd_date),underlyingCode,"mid")
        # dfotc=api.getOTC_LiveTrade(str(rqd.get_previous_trading_date(cur_trd_date))
        #                            , underlyingCode)
        # # dfotc['Varity']=dfotc.underlyingCode.apply(lambda x:findCode(x))
        # # v='v'
        # # dfotc=dfotc[dfotc.Varity==v.upper()]
        # dfotc.maturityDate=dfotc.maturityDate.apply(lambda t:pd.to_datetime(t).date())
        # dfotc['ttm']=dfotc.maturityDate.apply(lambda x:st.total_tradingdates.index(x)-st.total_tradingdates.index(cur_trd_date))
        # dfotc.buyOrSell=dfotc.buyOrSell.apply(lambda x:1 if x=='buy' else -1)
        # dfotc['cal_ttm']=dfotc.maturityDate.apply(lambda x:(x-cur_trd_date).days)
         
        
        # dfticks=rqd.get_ticks(underlyingCode).reset_index()
        # dfticks=dfticks[['datetime', 'trading_date', 'last']]
        # hedge_interval=15
        # preclose=4520
        
        # for idx in dfticks.index:
        #     ((dfticks['last']-preclose).abs()>=hedge_interval)*dfticks['last']








        # cdt=[dfotc.optionTypeName=="香草期权"
        #      ,dfotc.optionTypeName=="远期"]
        # cho=[True,True]
        # np.select(cdt,cho,default=False)
        # dfotc_euro=dfotc[np.select(cdt,cho,default=False)]
        # dfotc_euro['callOrPut']=dfotc_euro.callOrPut.map({"call":"C","put":"P"}).fillna('F')
        # dfs_ts=rqd.get_price( underlyingCode,cur_trd_date,cur_trd_date,"1m","close").reset_index()[['datetime','close']]
        # dfs_ts['intra']=dfs_ts.datetime.apply(lambda t:calTradttm(st.total_tradingdates, t, cur_trd_date))
        # # dfotc_euro['ttm']=dfotc_euro.maturityDate.apply(lambda t: calTradttm(st.total_tradingdates, datetime.combine(cur_trd_date,time(15,0,0)), t))
        # s_arr=dfs_ts.close.repeat(dfotc_euro.shape[0]).values
        # k_arr=np.array(dfotc_euro.strike.tolist()*dfs_ts.shape[0])
        # opttype_arr=np.array(dfotc_euro.callOrPut.tolist()*dfs_ts.shape[0])
        # t_arr=(dfs_ts.intra.repeat(dfotc_euro.shape[0])+np.array(dfotc_euro.ttm.tolist()*dfs_ts.shape[0]))/annual_coeff
        # cal_t_arr=(dfs_ts.intra.repeat(dfotc_euro.shape[0])+np.array(dfotc_euro.cal_ttm.tolist()*dfs_ts.shape[0]))
        
        
        # vol_str=str(api.getVol_json(str(cur_trd_date), underlyingCode)['mid'])
        # vfe=json.dumps(literal_eval(vol_str))
        # sigma_arr=np.array(list(map(lambda k,s,t:pyAILinearInterpVolSurface(vfe, t
        #                            , k/s),k_arr,s_arr,cal_t_arr)))
        # # sigma_arr=np.array(sigma_arr)
        # bsm=BSM_ARR(s_arr, k_arr, t_arr, rf, 0, sigma_arr, opttype_arr)
        # bsm.delta()
        # dfotc_exotic=dfotc.drop(dfotc_euro.index.tolist())
        # dfotc_exotic=dfotc_exotic.drop(dfotc_exotic[dfotc_exotic.optionTypeName=='远期'].index.tolist())
        
        
        
        
        
        # # dfsel=dfotc_exotic[dfotc_exotic.underlyingCode=='V2501']
        
        # api.getVol(str(cur_trd_date), 'V2501')['mid']
        
        
        
        
        # pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
        #                         , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                         , s_ts.loc[t,'close']
        #                         , st.pos_exotic.loc[idx,'Strike']
        #                         , datetime2timestamp(t.strftime('%Y-%m-%d %H:%M:%S'))
        #                         , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                         , st.pos_exotic.loc[idx,'InitialSpotPrice']
        #                         , st.pos_exotic.loc[idx,'qty_freq']
        #                         , int(st.pos_exotic.loc[idx,'isCashSettle'])
        #                         , float(st.pos_exotic.loc[idx,'leverage'])
        #                         , float(st.pos_exotic.loc[idx,'exp_leverage'])
        #                         , float(st.pos_exotic.loc[idx,'fp'])
        #                         , float(st.pos_exotic.loc[idx,'barrier'])
        #                         , float(st.pos_exotic.loc[idx,'rebate'])
        #                         , pdobList
        #                         , rf, rf
        #                         , 0 #const_sgm
        #                         , pdobList.shape[0]
        #                         , pre_Liv.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                         )



         
        # LV=LinearInterpVol(dfsel.underlyingCode.unique().tolist(),cur_trd_date)
        # s_ts=rqd.get_price(dfotc.underlyingCode.unique().tolist(),cur_trd_date,cur_trd_date,"1m",fields=['close','high','low']).reset_index()
        # s_ts.index=s_ts.datetime
        # for t in s_ts.index:
        #     # t=s_ts.index[0]
        #     t_arr=dfotc_euro.maturityDate.apply(lambda x:calTradttm(st.total_tradingdates,t,x)).values
        #     dfotc_euro['k_s']=dfotc_euro.strike/s_ts.loc[t,'close']
            
            
        #     bsm=BSM_ARR(np.ones(t_arr.shape)*s_ts.loc[t,'close']
        #             , dfotc_euro.strike.values
        #             , t_arr/annual_coeff, rf, q
        #             , np.array(list(map(lambda ass,x,y:LV.calVol(ass,x,y+t_arr[0]-int(t_arr[0])),dfotc_euro.underlyingCode,dfotc_euro.k_s,dfotc_euro.exp_calendar_days)))
        #             , dfotc_euro.callOrPut.apply(lambda x:x[0].upper()).values)
        #     dfotc_euro['delta']=dfotc_euro.availableVolume*dfotc_euro.buyOrSell*bsm.delta()
        #     s_ts.loc[t,'delta']=dfotc_euro.delta.sum()
        
        # last_h_price=rqd.get_price("EG2409",rqd.get_previous_trading_date(cur_trd_date),rqd.get_previous_trading_date(cur_trd_date),'1d','close').reset_index().close.values[0]
        # hedge_vol=0.1
        # hedge_interval=hedge_vol/np.sqrt(annual_coeff)*last_h_price
        
        # s_ts.high.apply(lambda x:1 if abs(x-last_h_price)>=hedge_interval else 0)
        
        
        # dfotc_exotic['OptionType']=dfotc_exotic.optionTypeName.map({"累购期权":"acccall"
        #                                     ,"累沽期权":"accput"
        #                                     ,"固定赔付累购":"fpcall"
        #                                     ,"固定赔付累沽":"fpput"
        #                                     ,"熔断累购期权":"b_acccall"
        #                                     ,"熔断累沽期权":"b_accput"
        #                                     ,"熔断固赔累购":"b_fpcall"
        #                                     ,"熔断固赔累沽":"b_fpput"
        #                                     ,"熔断增强累购":"b_acccallplus"
        #                                     ,"熔断增强累沽":"b_accputplus"})
        
        # for idx in dfotc_exotic.index:
        #     idx=dfotc_exotic.index[0]
        #     py

        
     
        
#%%
        # cols=['pv', 'delta', 'gamma', 'vega_percentage', 'theta_per_day', 'rho_percentage'
        #       , 'dividend_rho_percentage'
        #       ,'accumulated_position', 'accumulated_payment', 'accumulated_pnl']
 
        # v='v'
        # st.getPos(v,'','')
        # st.pos_exotic.fillna(0,inplace=True)

        
        # idx=st.pos_exotic[st.pos_exotic.TradeNumber==tradenumber].index[0]
        # # s_ts=rqd.get_ticks('V2409',cur_trad_date,cur_trad_date)['last'].reset_index()
        # s_ts=rqd.get_price('V2409',cur_trad_date,cur_trad_date,'1m','close').reset_index()
        # s_ts.index=s_ts.datetime
        # pdobList=getpdobList(getRQcode(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                       , rqd.get_previous_trading_date(cur_trad_date,1)
        #                       , pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate']).date()
        #                       , st.pos_exotic.loc[idx,'ExerciseDate'].date()
        #                       , s_ts.datetime[0]
        #                       ,cur_trad_date)
        # pre_Liv=LinearInterpVol(st.pos_exotic.loc[idx,'UnderlyingCode'],rqd.get_previous_trading_date(cur_trad_date,1))
        # dfres=pd.DataFrame(index=s_ts.index,columns=cols)
        # for t in s_ts.index:
        #     res=pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
        #                             , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                             , s_ts.loc[t,'close']
        #                             , st.pos_exotic.loc[idx,'Strike']
        #                             , datetime2timestamp(t.strftime('%Y-%m-%d %H:%M:%S'))
        #                             , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                             , st.pos_exotic.loc[idx,'InitialSpotPrice']
        #                             , st.pos_exotic.loc[idx,'qty_freq']
        #                             , int(st.pos_exotic.loc[idx,'isCashSettle'])
        #                             , float(st.pos_exotic.loc[idx,'leverage'])
        #                             , float(st.pos_exotic.loc[idx,'exp_leverage'])
        #                             , float(st.pos_exotic.loc[idx,'fp'])
        #                             , float(st.pos_exotic.loc[idx,'barrier'])
        #                             , float(st.pos_exotic.loc[idx,'rebate'])
        #                             , pdobList
        #                             , rf, rf
        #                             , 0 #const_sgm
        #                             , pdobList.shape[0]
        #                             , pre_Liv.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                             )
        #     dfres.loc[t]=res
        # dfres['deltalots']=dfres.delta/rqd.instruments('V2409').contract_multiplier
        # dfres.pv.plot()
        # pre_close=pdobList[pdobList>0]['close'].dropna().values[-1]
        
       
       
       
       
       
       
        #%%
        # cols=['pv', 'delta', 'gamma', 'vega_percentage', 'theta_per_day', 'rho_percentage'
        #       , 'dividend_rho_percentage'
        #       ,'accumulated_position', 'accumulated_payment', 'accumulated_pnl']

        # v='v'
        # st.getPos(v,'','')
        # st.pos_exotic.fillna(0,inplace=True)

        # tradenumber="20240419-JBJS-01"
        # idx=st.pos_exotic[st.pos_exotic.TradeNumber==tradenumber].index[0]
        
        # pdobList=pd.DataFrame(index=rqd.get_trading_dates(st.pos_exotic.loc[idx,'FirstObservationDate'],st.pos_exotic.loc[idx,'ExerciseDate'])
        #                       , columns=['close'],data=np.nan)
        # if pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate'])<=st.curren_trade_date:
        #     close_ts=rqd.get_price(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'FirstObservationDate'],st.curren_trade_date,'1d','close').loc[st.pos_exotic.loc[idx,'UnderlyingCode']]
        #     pdobList.loc[close_ts.index.tolist(),'close']=close_ts.close
        # pdobList.fillna(0,inplace=True)
        # pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]
        
        # # pricing_time=datetime.now()
        
        # s_ts=rqd.get_ticks('V2409',cur_trad_date,cur_trad_date)['last'].reset_index()
        
        
        
        # pricing_time=datetime.now()
        # pdobList=getpdobList(getRQcode(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                       , cur_trad_date
        #                       , pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate']).date()
        #                       , st.pos_exotic.loc[idx,'ExerciseDate'].date()
        #                       , pricing_time)
        # res=pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
        #                         , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                         , st.pos_exotic.loc[idx,'UnderlyingPrice']
        #                         , st.pos_exotic.loc[idx,'Strike']
        #                         , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                         , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                         , st.pos_exotic.loc[idx,'InitialSpotPrice']
        #                         , st.pos_exotic.loc[idx,'qty_freq']
        #                         , int(st.pos_exotic.loc[idx,'isCashSettle'])
        #                         , float(st.pos_exotic.loc[idx,'leverage'])
        #                         , float(st.pos_exotic.loc[idx,'exp_leverage'])
        #                         , float(st.pos_exotic.loc[idx,'fp'])
        #                         , float(st.pos_exotic.loc[idx,'barrier'])
        #                         , float(st.pos_exotic.loc[idx,'rebate'])
        #                         , pdobList
        #                         , rf, rf
        #                         , 0 #const_sgm
        #                         , pdobList.shape[0]
        #                         , st.LiV.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                         # ,vol
        #                         )
        # res=pd.DataFrame(index=cols,data=res)
        # pre_pv=29203.12
        # pre_deltaquant=-681.1878
        # s_t=st.pos_exotic.loc[idx,'UnderlyingPrice']
        # pre_close=2434
        # pre_Liv=LinearInterpVol(st.pos_exotic.loc[idx,'UnderlyingCode'],rqd.get_previous_trading_date(cur_trad_date,1))
        # delta_pnl=pre_deltaquant*(s_t-pre_close)      
        # pre_pdobList=getpdobList(getRQcode(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                       , rqd.get_previous_trading_date(cur_trad_date,1)
        #                       , pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate']).date()
        #                       , st.pos_exotic.loc[idx,'ExerciseDate'].date()
        #                       , datetime.combine(rqd.get_previous_trading_date(cur_trad_date,1),time(15,1,0)))
        
        # pre_pricing_time=datetime.combine(rqd.get_previous_trading_date(cur_trad_date,1),time(15,0,0))
        # theta_pnl=pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
        #                         , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                         , pre_close
        #                         , st.pos_exotic.loc[idx,'Strike']
        #                         , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                         , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                         , st.pos_exotic.loc[idx,'InitialSpotPrice']
        #                         , st.pos_exotic.loc[idx,'qty_freq']
        #                         , int(st.pos_exotic.loc[idx,'isCashSettle'])
        #                         , float(st.pos_exotic.loc[idx,'leverage'])
        #                         , float(st.pos_exotic.loc[idx,'exp_leverage'])
        #                         , float(st.pos_exotic.loc[idx,'fp'])
        #                         , float(st.pos_exotic.loc[idx,'barrier'])
        #                         , float(st.pos_exotic.loc[idx,'rebate'])
        #                         , pre_pdobList
        #                         , rf, rf
        #                         , 0 #const_sgm
        #                         , pdobList.shape[0]
        #                         , pre_Liv.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                         # ,vol
        #                         )[0]-pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
        #                                                 , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                                                 , pre_close
        #                                                 , st.pos_exotic.loc[idx,'Strike']
        #                                                 , datetime2timestamp(pre_pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                                                 , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                                                 , st.pos_exotic.loc[idx,'InitialSpotPrice']
        #                                                 , st.pos_exotic.loc[idx,'qty_freq']
        #                                                 , int(st.pos_exotic.loc[idx,'isCashSettle'])
        #                                                 , float(st.pos_exotic.loc[idx,'leverage'])
        #                                                 , float(st.pos_exotic.loc[idx,'exp_leverage'])
        #                                                 , float(st.pos_exotic.loc[idx,'fp'])
        #                                                 , float(st.pos_exotic.loc[idx,'barrier'])
        #                                                 , float(st.pos_exotic.loc[idx,'rebate'])
        #                                                 , pre_pdobList
        #                                                 , rf, rf
        #                                                 , 0 #const_sgm
        #                                                 , pdobList.shape[0]
        #                                                 , pre_Liv.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
        #                                                 # ,vol
        #                                                 )[0]
        
        
        
        
        
        
        # url=r"C:\Users\dzrh\Desktop\eg05barrier.xlsx"
        # df=pd.read_excel(url)
        
        
# #%%
# user='chengyl'
# passwd='CYLcyl0208@'
# yl=YieldChainAPI(user,passwd)
# trade_date="2024/03/29"
# pos_live=yl.get_tradeDetail('确认成交')
# # pos_live[pos_live.TradeNumber==tradenumber]
   
# #%%
# url=r'C:\Users\dzrh\Desktop\risk_L2405.xlsx'
# df=pd.read_excel(url)        
# df.columns=list(map(lambda x:x.replace(" ",""),df.columns))
   
# trd_num="20240318-DZRHNY-01"
# idx=df[df['交易编号']==trd_num].index[0]

# pdobList=pd.DataFrame(index=rqd.get_trading_dates(df.loc[idx,'FirstObservationDate'],st.pos_exotic.loc[idx,'ExerciseDate'])
#                       , columns=['close'],data=np.nan)
# if pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate'])<=st.curren_trade_date:
#     close_ts=rqd.get_price(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'FirstObservationDate'],st.curren_trade_date,'1d','close').loc[st.pos_exotic.loc[idx,'UnderlyingCode']]
#     pdobList.loc[close_ts.index.tolist(),'close']=close_ts.close
# pdobList.fillna(0,inplace=True)
# pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]


# for t in df.index:
#         a=pyAIKOAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType'][2:]
#                                 , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
#                                 , close_ts.loc[t].close
#                                 , st.pos_exotic.loc[idx,'Strike']
#                                 , datetime2timestamp(t.strftime('%Y-%m-%d %H:%M:%S'))
#                                 , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
#                                 , st.pos_exotic.loc[idx,'InitialSpotPrice']
#                                 , st.pos_exotic.loc[idx,'qty_freq']
#                                 , int(st.pos_exotic.loc[idx,'isCashSettle'])
#                                 , float(st.pos_exotic.loc[idx,'leverage'])
#                                 , float(st.pos_exotic.loc[idx,'exp_leverage'])
#                                 , float(st.pos_exotic.loc[idx,'fp'])
#                                 , float(st.pos_exotic.loc[idx,'barrier'])
#                                 , float(st.pos_exotic.loc[idx,'rebate'])
#                                 , pdobList
#                                 , rf, rf
#                                 , 0 #const_sgm
#                                 , pdobList.shape[0]
#                                 , st.LiV.getVolsurfacejson(st.pos_exotic.loc[idx,'UnderlyingCode'])
#                                 # ,vol
#                                 )
        
#         df.loc[t,'close']=close_ts.loc[t].close
#         # df.loc[t,cols[1:]]=a    


        
        
        
     
     #%%
        # idx=7
        # indexList = [datetime2timestamp(str(t) + ' 15:00:00') for t in rqd.get_trading_dates(st.pos_exotic.loc[idx,'FirstObservationDate'],st.pos_exotic.loc[idx,'ExerciseDate'])]
        # pyAIAccumulatorPricer(st.pos_exotic.loc[idx,'OptionType']
        #                         , -1 if st.pos_exotic.loc[idx,'cust_bs']=="B" else 1
        #                         , st.pos_exotic.loc[idx,'UnderlyingPrice']
        #                         , st.pos_exotic.loc[idx,'Strike']
        #                         , datetime2timestamp(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #                         , datetime2timestamp(str(st.pos_exotic.loc[idx,'ExerciseDate'])[:10]+" 15:00:00")
        #                         , st.pos_exotic.loc[idx,'qty_freq']
        #                         , 0 #isCashsettle
        #                         , float(st.pos_exotic.loc[idx,'leverage'])
        #                         , float(st.pos_exotic.loc[idx,'fp'])
        #                         , float(st.pos_exotic.loc[idx,'barrier'])
        #                         , float(st.pos_exotic.loc[idx,'StrikeRamp'])
        #                         , float(st.pos_exotic.loc[idx,'BarrierRamp'])
        #                         # , float(st.pos_exotic.loc[idx,'rebate'])
        #                         # , pd.DataFrame(index=[datetime2timestamp(str(t) + ' 15:00:00') for t in rqd.get_trading_dates(st.pos_exotic.loc[idx,'FirstObservationDate'],st.pos_exotic.loc[idx,'ExerciseDate'])]
        #                         #                , columns=['close']
        #                         #                ,data=rqd.get_price(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'FirstObservationDate'],st.curren_trade_date,'1d','close').close.values)
        #                         ,pd.DataFrame(index= indexList 
        #                                            ,columns=['close'],data=1)
        #                         , rf
        #                         , 0 #const_sgm
        #                         , len(indexList)
        #                         , st.LiV.getVolsurfacejson(),0
        #                         )
  
     
        #%%
        # idx=10
        # for idx in st.norm_idx:
        #     acc=AccOptionArrSelectItems(['theta','vega'],st.total_tradingdates,decay_days
        #                  ,st.pos_exotic.loc[idx,'OptionType']
        #                  ,st.pos_exotic.loc[idx,'UnderlyingCode']
        #                   ,st.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
        #                  # ,st.pos_exotic.loc[idx,'UnderlyingPrice']
        #               ,st.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
        #               ,st.pos_exotic.loc[idx,'Strike'],st.pos_exotic.loc[idx,'barrier'], float(st.pos_exotic.loc[idx,'fp'])
        #               ,max(pd.to_datetime(st.pos_exotic.loc[idx,'FirstObservationDate']).date(),st.curren_trade_date)
        #               ,st.pos_exotic.loc[idx,'ExerciseDate']
        #               ,datetime.now()
        #               ,st.pos_exotic.loc[idx,'qty_freq'],st.pos_exotic.loc[idx,'cust_bs'], float(st.pos_exotic.loc[idx,'leverage'])
        #               ,float(st.pos_exotic.loc[idx,'StrikeRamp']), float(st.pos_exotic.loc[idx,'BarrierRamp']))
        #     res=acc.getResult()[['price','gamma','theta','vega']]
        
        # res['price'].T.plot()
        #%%
        # pv_exotic=pd.DataFrame(index=st.norm_idx,columns=delta_s_arr)
        # for idx in st.norm_idx:
        #         acc=AccOptionArrSelectItems(['price'],st.total_tradingdates,""
        #                      ,st.pos_exotic.loc[idx,'OptionType'],st.pos_exotic.loc[idx,'UnderlyingCode']
        #                      ,st.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
        #                   ,st.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
        #                   ,st.pos_exotic.loc[idx,'Strike'],st.pos_exotic.loc[idx,'barrier'], float(st.pos_exotic.loc[idx,'fp'])
        #                   ,st.curren_trade_date,st.pos_exotic.loc[idx,'ExerciseDate']
        #                   ,st.start_time
        #                   ,st.pos_exotic.loc[idx,'qty_freq'],st.pos_exotic.loc[idx,'cust_bs'], float(st.pos_exotic.loc[idx,'leverage'])
        #                   ,float(st.pos_exotic.loc[idx,'StrikeRamp']), float(st.pos_exotic.loc[idx,'BarrierRamp']))
        #         pv_exotic.loc[idx]=acc.getResult()['price'].loc[0].values

 
        
        #%%
# res_1=pd.DataFrame(index=np.arange(1,decay_days+1,1),columns=delta_s_arr,data=0)
# res_2=pd.DataFrame(index=np.arange(1,decay_days+1,1),columns=delta_s_arr,data=0)
# # for idx in st.barrier_idx:
# # for idx in st.pos_exotic[st.pos_exotic.OptionType.str.contains('b_acccallplus')].index.tolist():
# idx=24
# bsb=BarrierAccOptionSelectItems(['theta','price'],st.total_tradingdates,decay_days
#                       , st.pos_exotic.loc[idx,'OptionType'][2:]
#                       , st.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
#                       , st.pos_exotic.loc[idx,'InitialSpotPrice']
#                       , st.pos_exotic.loc[idx,'Strike']
#                       , st.pos_exotic.loc[idx,'barrier']
#                       , st.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
#                       , float(st.pos_exotic.loc[idx,'fp'])
#                       , float(st.pos_exotic.loc[idx,'rebate'])
#                       , 1/annual_coeff
#                       , float(st.pos_exotic.loc[idx,'leverage'])
#                         ,st.pos_exotic.loc[idx,'cust_bs']
#                       # ,"b"
#                       , st.pos_exotic.loc[idx,'qty_freq']
#                       , float(st.pos_exotic.loc[idx,'exp_leverage'])
#                         # , st.curren_trade_date 
#                         ,datetime(2024,6,17).date()# next_obs_date
#                       , st.pos_exotic.loc[idx,'ExerciseDate']
#                         # , st.start_time
#                         ,datetime(2024,6,17,14,0,0)
#                       )
# res=bsb.getResult()[['theta','price']]
# res_theta=res['theta']
# res_pv=res['price']



# aa=copy.copy(bsb.bsm_total)
# aa['pv_exp']=bsb.sb.pv_exp
# aa['pv_0']=bsb.sb.price()


# aa['opt_type']=bsb.opt_list*int(bsb.total_shape/bsb.group_num)
# aa['dirt']=bsb.dirt_list*int(bsb.total_shape/bsb.group_num)
# aa['move']=bsb.move_list*int(bsb.total_shape/bsb.group_num)
# aa['bsm_call']=bsb.sb.bsm_call



# aa['price_out']=bsb.sb.price_out()
# res_1.T.plot()


# res_1_end=pd.DataFrame(index=np.arange(1,decay_days+1,1),columns=delta_s_arr,data=0)
# res_2_end=pd.DataFrame(index=np.arange(1,decay_days+1,1),columns=delta_s_arr,data=0)

# bsb_end=BarrierAccOptionSelectItems(['theta','price'],st.total_tradingdates,decay_days
#                       , st.pos_exotic.loc[idx,'OptionType'][2:]
#                       , st.pos_exotic.loc[idx,'UnderlyingPrice']*delta_s_arr
#                       , st.pos_exotic.loc[idx,'InitialSpotPrice']
#                       , st.pos_exotic.loc[idx,'Strike']
#                       , st.pos_exotic.loc[idx,'barrier']
#                       , st.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
#                       , float(st.pos_exotic.loc[idx,'fp'])
#                       , float(st.pos_exotic.loc[idx,'rebate'])
#                       , 1/annual_coeff
#                       , float(st.pos_exotic.loc[idx,'leverage'])
#                        ,st.pos_exotic.loc[idx,'cust_bs']
#                       # ,"b"
#                       , st.pos_exotic.loc[idx,'qty_freq']
#                       , float(st.pos_exotic.loc[idx,'exp_leverage'])
#                         # , st.curren_trade_date 
#                         ,datetime(2024,5,16).date()# next_obs_date
#                       , st.pos_exotic.loc[idx,'ExerciseDate']
#                        # , st.start_time
#                        ,datetime(2024,5,15,16,59,0)
#                       )
# res=bsb.getResult()[['theta','price']]
# res_1_end+=res['theta'].values
# res_1_end+=res['price'].values
# res_1_end.T.plot()






# # print("Barrier each time = ", datetime.now() - tic_i, "s")     
#  # self.Theta=self.Theta.astype(float).round(0)
#  # self.Vega=self.Vega.astype(float).round(0)

# idx=10
# bsb=BarrierAccOption(st.total_tradingdates
#                       , st.pos_exotic.loc[idx,'OptionType'][2:]
#                       , st.pos_exotic.loc[idx,'UnderlyingPrice']*0+7210
#                       , st.pos_exotic.loc[idx,'InitialSpotPrice']
#                       , st.pos_exotic.loc[idx,'Strike']
#                       , st.pos_exotic.loc[idx,'barrier']
#                       , st.LiV.calVol(st.pos_exotic.loc[idx,'UnderlyingCode'],st.pos_exotic.loc[idx,'Strike']/st.pos_exotic.loc[idx,'UnderlyingPrice'],(pd.to_datetime(st.pos_exotic.loc[idx,'ExerciseDate'])-st.start_time).days+1) 
#                       , float(st.pos_exotic.loc[idx,'fp'])
#                       , float(st.pos_exotic.loc[idx,'rebate'])
#                       , 1/annual_coeff
#                       , float(st.pos_exotic.loc[idx,'leverage'])
#                        ,st.pos_exotic.loc[idx,'cust_bs']
#                       # ,"b"
#                       , st.pos_exotic.loc[idx,'qty_freq']
#                       , float(st.pos_exotic.loc[idx,'exp_leverage'])
#                       , datetime(2024,5,16).date()  # next_obs_date
#                       , st.pos_exotic.loc[idx,'ExerciseDate']
#                       # , st.start_time
#                       ,datetime(2024,5,15,22,30,0)
#                       )


 # bsb.price()








        #%%
        
      
        # s_0=8589
        # strike=s_0-250
        # barrier=s_0+250
        # s_ts=(np.arange(80,120,1)/100*s_0).round(0)
        # sigma=0.2
        # fixed_income=130
        # firstobs_date=datetime(2023,12,21).date()
        # endobs_date=datetime(2024,1,31).date()
        # # trading_time=datetime.now()
        # qty_freq=10
        # leverage=2
        # # lev_exp=1
        # delta_strike=50
        # delta_barrier=50
        # dic_acc={datetime(2023,12,21).date():datetime(2023,12,21,12,3,0)
        #          ,datetime(2024,1,15).date():datetime(2024,1,15,12,3,0)
        #          ,datetime(2024,1,29).date():datetime(2024,1,29,12,3,0)
        #          }
        # i=0
        # fig=plt.figure(figsize=(15,15))
        # for startobs_date in dic_acc.keys():
        #     left_days=(endobs_date-startobs_date).days
        #     trading_time=dic_acc[startobs_date]
        #     df1=pd.DataFrame(columns=s_ts)
        #     df2=pd.DataFrame(columns=s_ts)
        #     df3=pd.DataFrame(columns=s_ts)
        #     df4=pd.DataFrame(columns=s_ts)
        #     for s in s_ts:
        #         acc1=AccOptionArr(st.total_tradingdates,'ACCCALL',varity,s, sigma, strike, barrier, fixed_income
        #                       , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
        #         acc2=AccOptionArr(st.total_tradingdates,'ACCCALL',varity,s, sigma, strike, 0, fixed_income
        #                       , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
        #         acc3=AccOptionArr(st.total_tradingdates,'FPCALL',varity,s, sigma, strike, barrier, fixed_income
        #                       , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
        #         acc4=AccOptionArr(st.total_tradingdates,'FPCALL',varity,s, sigma, strike, 0, fixed_income
        #                       , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)

        #         #              , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
        #         df1[s]=acc1.getResult()
        #         df2[s]=acc2.getResult()
        #         df3[s]=acc3.getResult()
        #         df4[s]=acc4.getResult()
            
        #     for c in ['delta','gamma','theta','vega']:
        #         i+=1
        #         ax=fig.add_subplot(3,4,i)
        #         ax.plot(df1.loc[c,:],label='acc_bar',linestyle='--',linewidth=1,color='k')
        #         ax.plot(df2.loc[c,:],label='acc_no_bar',color='k')
        #         ax.plot(df3.loc[c,:],label='fp_bar',linestyle='--',linewidth=1,color='r')
        #         ax.plot(df4.loc[c,:],label='fp_no_bar',color='r')
        #         min_=np.min([df1.loc[c,:].min(),df2.loc[c,:].min(),df3.loc[c,:].min(),df4.loc[c,:].min()])
        #         max_=np.max([df1.loc[c,:].max(),df2.loc[c,:].max(),df3.loc[c,:].max(),df4.loc[c,:].max()])
        #         ax.vlines(strike,min_,max_,linewidth=1,linestyle='--',color='y',label='K')
        #         ax.vlines(barrier,min_,max_,linewidth=1,linestyle='--',color='b',label='Barrier')
        #         plt.legend()
        #         ax.set_title(c+' '+str(left_days)+' days left')
        # plt.tight_layout()
        #     # dic_acc[startobs_date]=df
        
        # # dic_facc={datetime(2023,12,5).date():""
        # #          ,datetime(2024,1,2).date():""
        # #          ,datetime(2024,1,29).date():""
        # #          }
        # # total_obs_days=st.total_tradingdates.index(endobs_date)-st.total_tradingdates.index(firstobs_date)+1
        # # for startobs_date in dic_facc.keys():
        # #     # df=pd.DataFrame(columns=s_ts)
        # #     # for s in s_ts:
        # #     #     acc=AccOptionArr(st.total_tradingdates,'FPCALL',varity,s, sigma, strike, barrier, fixed_income
        # #     #                  , startobs_date, endobs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
        # #     #     df[s]=acc.getResult()
        # #     # dic_facc[startobs_date]=df
        # #     left_obs_days=st.total_tradingdates.index(endobs_date)-st.total_tradingdates.index(startobs_date)+1
        # #     print(left_obs_days)
        # #     # df=pd.DataFrame(columns=s_ts)
        # #     # for s in s_ts:
        # #     #     bsb=BarrierAccOption('acccall',s,strike,barrier,sigma,fixed_income,0,leverage,lev_exp,qty_freq
        # #     #                          ,left_obs_days,total_obs_days,trading_time)      
        # #     #     df[s]=bsb.getResult()*-1   
        # #     # dic_facc[startobs_date]=df
        
        
        # # for k in dic_acc.keys():
        # #     fig=plt.figure(figsize=(15,4))
        # #     ttm=(endobs_date-k).days
        # #     fig.suptitle(str(ttm)+" Days Left")
        # #     i=0
        # #     for c in ['delta','gamma','theta','vega']:
        # #         i+=1
        # #         ax=fig.add_subplot(1,4,i)
        # #         ax.plot(dic_acc[k].loc[c,:],label='acc',color='k')
        # #         ax.plot(dic_facc[k].loc[c,:],label='bsb',color='b')
        # #         if c=='gamma':
        # #             max_g=dic_acc[k].loc[c,:].max()*2
        # #             min_g=dic_acc[k].loc[c,:].min()*2
        # #             ax.set_ylim(min_g,max_g)
        # #         ax.vlines(strike,min(dic_acc[k].loc[c,:].min(),dic_facc[k].loc[c,:].min()),max(dic_acc[k].loc[c,:].max(),dic_facc[k].loc[c,:].max()),linewidth=1,linestyle='--',color='y',label='K')
        # #         ax.vlines(barrier,min(dic_acc[k].loc[c,:].min(),dic_facc[k].loc[c,:].min()),max(dic_acc[k].loc[c,:].max(),dic_facc[k].loc[c,:].max()),linewidth=1,linestyle='--',color='b',label='Barrier')
        # #         plt.legend()
        # #         ax.set_title(c)
        # #     plt.tight_layout()
            
        # # fig=plt.figure(figsize=(15,4))   
        # #        # fig.suptitle(str(ttm)+" Days Left")
        # # i=0
        # # for c in ['delta','gamma','theta','vega']:
        # #     i+=1
        # #     ax=fig.add_subplot(1,4,i)
        # #     for k in dic_acc.keys():
        # #         ttm=(endobs_date-k).days
        # #         ax.plot(dic_acc[k].loc[c,:],label=str(ttm))
        # #         # ax.plot(dic_acc[k].loc[c,:],label='facc',color='b')
        # #     ax.vlines(strike,dic_acc[k].loc[c,:].min(),dic_acc[k].loc[c,:].max(),linewidth=1,linestyle='--',color='y',label='K')
        # #     ax.vlines(barrier,dic_acc[k].loc[c,:].min(),dic_acc[k].loc[c,:].max(),linewidth=1,linestyle='--',color='b',label='Barrier')
        # #     plt.legend()
        # #     ax.set_title(c)
        # # plt.tight_layout()
        
        # #%%
        
     
        
     

       
        
        
        
        
        
        
