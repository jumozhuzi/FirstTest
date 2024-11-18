# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:18:01 2023

@author: dzrh
"""

import xlwings as xw
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# import tushare as tus
from datetime import date,timedelta,time,datetime
import time
import iFinDPy as fd
import os
import copy
from CYL.OptionPricing import BSM,calIV,calTradttm,AccOption

from CYL.YieldChainAPI import YieldChainAPI
import bisect
import itertools
import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")

rf=0.03
q=0
user='chengyl'
passwd='CYLcyl0208@'
YL=YieldChainAPI(user,passwd)


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
    

    
class LinearInterpVol():
    def __init__(self,asset,valuedate):
        self.valuedate=str(valuedate)
        self.asset=asset
        self.__getVolsurface()
        self.ttmdays_arr=np.array([1,7,14,30,60,90,183,365])
        
    def __getVolsurface(self):
        '''
        '''
        if type(self.asset)==str:
            underlyingCodes=[self.asset]
        else:
            underlyingCodes=self.asset
            
        self.volsurface_dic=YL.get_surface(underlyingCodes, self.valuedate)
        # self.volsurface=volsurface.values
        # self.ttmdays_arr=np.array(volsurface.index.tolist())
       
    
    def calVol(self,single_ass,s,k,exp_ttm):
         if k==0:
             return 0
         elif exp_ttm<=0:
             return 0
         else:
             ks_ratio=np.arange(0.8,1.22,0.02).round(2)
         
             volsurface=self.volsurface_dic.get(single_ass).values
             
             idx_t=bisect.bisect_left(self.ttmdays_arr,exp_ttm)
             idx_ks=bisect.bisect_left(ks_ratio,k/s) 
    
    
             condition=[k/s<=ks_ratio[0]
                        ,k/s>ks_ratio[-1]]
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
                 return (np.diff(vol_slice,axis=0)/np.diff(ratio_arr)*(k/s-ratio_arr[0])+vol_slice[0])[0]
             elif vol_slice.shape==(2,1):
                 t_arr=self.ttmdays_arr[idx_t-1:idx_t+1] 
                 return ((np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0])[0]
             else:
                 t_arr=self.ttmdays_arr[idx_t-1:idx_t+1] 
                 ratio_arr=ks_ratio[cols[0]:cols[1]] 
                 term_diff_vol=(np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0,:]
                 return (np.diff(term_diff_vol)/np.diff(ratio_arr)*(k/s-ratio_arr[0])+term_diff_vol[0])[0]

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
    __annual_coeff=252
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
        self.obs_dates_list=self.trading_dates_list[self.trading_dates_list.index(pd.to_datetime(self.startobs_date)):self.trading_dates_list.index(pd.to_datetime(self.endobs_date))+1]
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
        


     
    
# idx=2
# tic = time.time()
# acc2=AccOption2(st.total_tradingdates
#           ,pos_exotic.loc[idx,'OptionType'],pos_exotic.loc[idx,'UnderlyingCode']
#           ,pos_exotic.loc[idx,'UnderlyingPrice']
#           ,0.2,pos_exotic.loc[idx,'Strike'],pos_exotic.loc[idx,'barrier']
#           ,float(pos_exotic.loc[idx,'fp']),pd.to_datetime(pos_exotic.loc[idx,'FirstObservationDate'])
#           ,pos_exotic.loc[idx,'ExerciseDate'],datetime.now()
#           ,pos_exotic.loc[idx,'qty_freq'],"B",float(pos_exotic.loc[idx,'leverage'])
#           ,float(pos_exotic.loc[idx,'StrikeRamp']),float(pos_exotic.loc[idx,'BarrierRamp'])
#           )
# acc2.getResult()
# print("Running time = ", time.time() - tic, "s")

# tic = time.time()
# acc=AccOption(st.total_tradingdates
#           ,pos_exotic.loc[idx,'OptionType'],pos_exotic.loc[idx,'UnderlyingCode']
#           ,pos_exotic.loc[idx,'UnderlyingPrice']
#           ,0.2,pos_exotic.loc[idx,'Strike'],pos_exotic.loc[idx,'barrier']
#           ,float(pos_exotic.loc[idx,'fp']),pd.to_datetime(pos_exotic.loc[idx,'FirstObservationDate'])
#           ,pos_exotic.loc[idx,'ExerciseDate'],datetime.now()
#           ,pos_exotic.loc[idx,'qty_freq'],"B",float(pos_exotic.loc[idx,'leverage'])
#           ,float(pos_exotic.loc[idx,'StrikeRamp']),float(pos_exotic.loc[idx,'BarrierRamp'])
#           )
# acc.getResult()
# print("Running time = ", time.time() - tic, "s")
def IntraHours(start_time):
    '''
    

    Parameters
    ----------
    trading_dates : list with datetime.date 
    start_time : datetime.time
        start trading time.
    expire_date : datetime.date
        expiration.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    # today=datetime.today()
    # trading_dates=rqd.get_trading_dates(datetime.today(),datetime.today()+timedelta(days=365))
    
    # now_time=datetime.now().time()
    now_time=start_time.time()
    if now_time<time(9,0,0):
        intra_hours=4/6
        # trd_date=start_time.date()
    elif now_time<time(11,30,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,11,30,0)-start_time).seconds/3600
        intra_hours=(intra_hours+1.5)/6
        # trd_date=start_time.date()
    elif now_time<time(13,30,0):
        intra_hours=1.5/6
        # trd_date=start_time.date()
    elif now_time<time(15,0,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,15,0,0)-start_time).seconds/3600
        intra_hours=intra_hours/6
        # trd_date=start_time.date()
    elif now_time<time(21,0,0):
        intra_hours=0
        # trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    elif now_time<time(23,0,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,23,0,0)-start_time).seconds/3600
        intra_hours=(intra_hours+4)/6
        # trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    else:
        intra_hours=0
        # trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    

    return intra_hours
   
        
class StressTest():
    def __init__(self):
      
        self.user='chengyl'
        self.passwd='CYLcyl0208@'
        self.total_tradingdates=rqd.get_trading_dates('2023-01-01','2024-12-31')
      

        
    def __addPropertys(self):
        pp_l=self.dfotc.Propertys.apply(lambda x:len(x))
        pp_idx=pp_l.where(pp_l>0).dropna().index.tolist()
        pp_ts=self.dfotc.loc[pp_idx,'Propertys']
        
        p_names=pp_ts.apply(lambda x:pd.DataFrame(x)['name']).stack().unique()
        dfpp=pd.DataFrame(index=p_names,columns=pp_idx)
        
        for p,col in zip(pp_ts,pp_idx):
            dfpp.loc[pd.DataFrame(p)['name'],col]=pd.DataFrame(p)['value'].values
        dfpp=dfpp.T
        dfpp.rename(columns={"累计敲出价格":"barrier","固定赔付区间上沿":"fp_up"
                             ,"固定赔付区间下沿":"fp_dw","固定赔付":"fp"
                             ,"单倍系数":"interval_rebate","多倍系数":"leverage"},inplace=True)
    
        self.dfotc.loc[:,dfpp.columns]=np.nan
        self.dfotc.loc[pp_idx,dfpp.columns]=dfpp
        self.dfotc['barrier']=np.where(self.dfotc.StructureType=="固定赔付累购",self.dfotc.fp_up.astype(float),self.dfotc.barrier.astype(float))
        self.dfotc['barrier']=np.where(self.dfotc.StructureType=="固定赔付累沽",self.dfotc.fp_dw.astype(float),self.dfotc.barrier.astype(float))
        
        
        self.dfotc.loc[pp_idx,'FirstObservationDate']=self.dfotc.loc[pp_idx].ObservationDates.apply(lambda x:x.split(",")[0])
        self.dfotc.loc[pp_idx,'qty_freq']=self.dfotc.loc[pp_idx,'TradeAmount']/self.dfotc.loc[pp_idx].ObservationDates.apply(lambda x:len(x.split(",")))



    def getInfo(self):
        self.dfinfo=YL.get_listInfo()
        self.dfinfo.index=self.dfinfo.Code
        self.dfitc=YL.get_listPosition()[['BookName','UnderlyingCode','ExchangeOptionCode','Volume']]
        self.dfitc=self.dfitc[self.dfitc['BookName']=='场外交易簿']
        self.dfitc['Varity']=list(map(lambda x:findCode(x),self.dfitc.UnderlyingCode))
        self.dfitc.drop(self.dfitc[self.dfitc.Volume==0].index,axis=0,inplace=True)
        
        
        self.dfotc=YL.get_tradeDetail()
        self.dfotc['Varity']=list(map(lambda x:findCode(x),self.dfotc.UnderlyingCode))
        self.dfotc.drop(self.dfotc[self.dfotc.TradeAmount==0].index,axis=0,inplace=True)
        tradetype_dic={'香草期权':'V','远期':'F','自定义交易':'SD','亚式期权':'AS','雪球期权':'SnowBall'}
        self.dfotc.TradeType=self.dfotc.TradeType.map(tradetype_dic)   
        self.dfotc.drop(self.dfotc[self.dfotc.TradeNumber=='20230822-NJCQ-01'].index,inplace=True)
        self.__addPropertys()
        
    
    def calStressTest(self,varity,delta_s_arr,t_decay_arr,start_time,next_end_time,pos_simu_dic=dict()):
        
        self.getPos(varity,start_time,pos_simu_dic)
        params_list=list(itertools.product(delta_s_arr,t_decay_arr))
        result=pd.DataFrame()
        for p in params_list:
            # print(p)
            result=pd.concat([result,self.__calGreeks(p[0],p[1])])
        
        st_result=dict()
        st_result['org']=result
        st_result['theta']=result.groupby(['ExpireDate','delta_s'])['Theta'].sum().unstack()
        st_result['vega']=result.groupby(['ExpireDate','delta_s'])['Vega'].sum().unstack()
    
        pv_0=self.__calPv(1,self.start_time)
        pv_exotic_0=self.__calExoticPv(1,self.start_time)
        
        df=pd.DataFrame(index=self.pos.UnderlyingCode,columns=delta_s_arr)
        dfexo=pd.DataFrame(index=self.pos_exotic.UnderlyingCode,columns=delta_s_arr)
        for  ds in delta_s_arr:
            df.loc[:,ds]=self.__calPv(ds,next_end_time).values
            dfexo.loc[:,ds]=self.__calExoticPv(ds,next_end_time).values
            
        df=df.subtract(pv_0,axis='index')
        df=df.multiply(self.pos.TradeAmount.values,axis='index')
        
        dfexo=dfexo.subtract(pv_exotic_0,axis='index')
        df=pd.concat([df,dfexo])
  
        st_result['pnl']=df.groupby('UnderlyingCode').sum()
        
        pnl=df.sum(axis=0)
        fig=plt.figure(figsize=(6,12))
        ax=fig.add_subplot(2,1,1)
        ax.plot(pnl/10000)
        ax.set_ylabel('Pnl(w)')
        ax.grid(True)
        ax.set_title(varity)
        ax=fig.add_subplot(2,1,2)
        ax.plot(pnl.loc[0.96:1.04])
        ax.set_ylabel('Pnl')
        ax.grid(True)
        ax.set_title(varity)
        
        st_result['fig']=fig
        
        return  st_result
    
    def calStressTestArr(self,varity,delta_s_arr,t_decay_arr,start_time,next_end_time,pos_simu_dic=dict()):
        # tic=datetime.now()
        self.getPos(varity,start_time,pos_simu_dic)
        params_list=list(itertools.product(delta_s_arr,t_decay_arr))
        result=pd.DataFrame()
        for p in params_list:
            # print(p)
            result=pd.concat([result,self.__calGreeksArr(p[0],p[1])])
        # print("total time :" , datetime.now() - tic, "s")
        
        st_result=dict()
        st_result['org']=result
        st_result['theta']=result.groupby(['ExpireDate','delta_s'])['Theta'].sum().unstack()
        st_result['vega']=result.groupby(['ExpireDate','delta_s'])['Vega'].sum().unstack()
      
    
     
        pv_0=self.__calPvArr(1,self.start_time)
        pv_exotic_0=self.__calExoticPvArr(1,self.start_time)
        df=pd.DataFrame(index=self.pos.UnderlyingCode,columns=delta_s_arr)
        dfexo=pd.DataFrame(index=self.pos_exotic.UnderlyingCode,columns=delta_s_arr)
        for  ds in delta_s_arr:
            df.loc[:,ds]=self.__calPvArr(ds,next_end_time).values
            dfexo.loc[:,ds]=self.__calExoticPvArr(ds,next_end_time).values
   
            
        df=df.subtract(pv_0,axis='index')
        df=df.multiply(self.pos.TradeAmount.values,axis='index')
        
        dfexo=dfexo.subtract(pv_exotic_0,axis='index')
        df=pd.concat([df,dfexo])
      
        st_result['pnl']=df.groupby('UnderlyingCode').sum()
        
        pnl=df.sum(axis=0)
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,7))
        # ax=fig.add_subplot(2,1,1)
        ax1.plot(pnl/10000)
        ax1.set_ylabel('Pnl(w)')
        ax1.grid(True)
        ax1.set_title(varity)
  
        ax2.plot(pnl.loc[0.97:1.03],'-o',ms=10,mfc='orange')
        # ax2.set_xlabel('1 PNL: '+str(pnl.loc[1].round(0))+' 0.99 PNL: '+str(pnl.loc[0.99].round(0))+' 1.01 PNL:'+str(pnl.loc[1.01].round(0)))
        ax2.annotate(str(pnl.loc[1].round(0)),xy=(1,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
        ax2.annotate(str(pnl.loc[0.99].round(0)),xy=(0.99,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
        ax2.annotate(str(pnl.loc[1.01].round(0)),xy=(1.01,pnl.loc[0.97:1.03].max()),xytext=(0,-50),textcoords='offset points')
        ax2.set_ylabel('Pnl')
        ax2.grid(True)
        # ax2.set_title(varity)
        
        st_result['fig']=fig
        
        return  st_result
    
   
    def getPos(self,varity,start_time,pos_simu_dic):
          self.getInfo()
          self.start_time=start_time
          pos_itc=self.dfitc[self.dfitc['Varity']==varity.upper()]
          pos_itc=pos_itc.fillna(method='ffill',axis=1)
          pos_itc=pd.merge(pos_itc[['ExchangeOptionCode','Volume','Varity']]
                   ,self.dfinfo.loc[pos_itc.ExchangeOptionCode][['Strike','OptionType','Expire_Date','Underlying_Code']]
                   ,on=pos_itc.ExchangeOptionCode)
          pos_itc['OptionType']=pos_itc['OptionType'].map({0:'F',1:'C',2:'P'})
          pos_itc['Underlying_Code']=np.where(pos_itc.Strike==0,pos_itc.ExchangeOptionCode,pos_itc.Underlying_Code)
          
          
          pos_otc=self.dfotc[self.dfotc['Varity']==varity.upper()]
          pos_otc=pos_otc.copy()
          pos_otc['ExerciseDate']=pos_otc.ExerciseDate.apply(lambda x:pd.to_datetime(x).date())
          pos_otc['TradeDate']=pos_otc.TradeDate.apply(lambda x:pd.to_datetime(x).date())
          pos_otc['CurAmount']=np.where(pos_otc.BuySell=='买入',1,-1)*pos_otc.TradeAmount
          opttypes=pos_otc.CallPut.map({'Call':'C','Put':'P'})
          acctypes=pos_otc.StructureType.map({"累购期权":"acccall","累沽期权":"accput","固定赔付累购":"fpcall","固定赔付累沽":"fpput"})
          # pos_otc['OptionType']=np.select([pos_otc.TradeType=='F'
          #                                 ,(pos_otc.TradeType=='AS')&(pos_otc.CallPut=='Call')
          #                                 ,(pos_otc.TradeType=='AS')&(pos_otc.CallPut=='Put')]
          #                                ,['F','ascall','asput'],default=acctypes.fillna(opttypes))
          pos_otc['OptionType']=np.where(pos_otc.TradeType=='F','F',acctypes.fillna(opttypes))
          pos=pd.DataFrame(columns=['TradeNumber','UnderlyingCode','UnderlyingPrice'
                            ,'ExerciseDate','OptionType','TradeAmount'
                            ,'Strike','StrikeRamp','barrier','BarrierRamp','fp','leverage','FirstObservationDate','qty_freq'])
          
          pos['TradeNumber']=pos_otc['TradeNumber'].append(pos_itc['ExchangeOptionCode'])
          pos['UnderlyingCode']=pos_otc['UnderlyingCode'].append(pos_itc['Underlying_Code'])
          
          
          pos['Strike']=pos_otc['Strike'].append(pos_itc.Strike)
          pos['ExerciseDate']=pd.to_datetime(pos_otc.ExerciseDate.append(pos_itc.Expire_Date))
          
          pos['FirstObservationDate']=pos_otc.FirstObservationDate

          pos['OptionType']=pos_otc.OptionType.append(pos_itc.OptionType)
          pos['TradeAmount']=pos_otc.CurAmount.append(pos_itc.Volume)
          pos['TradeAmount']=pos['TradeAmount'].astype(float)
          
          pos.index=pos.TradeNumber
          
          pos.loc[pos_otc.TradeNumber,'qty_freq']=pos_otc.qty_freq.astype(float).values
          # pos['Exchange']=self.dfinfo.loc[pos.UnderlyingCode].Exchange.values
          
          
          pos.loc[pos_otc.TradeNumber,'StrikeRamp']=pos_otc.StrikeRamp.values
          pos.loc[pos_otc.TradeNumber,'BarrierRamp']=pos_otc.BarrierRamp.values
          pos.loc[pos_otc.TradeNumber,'fp']=pos_otc.fp.values
          pos.loc[pos_otc.TradeNumber,'leverage']=pos_otc.leverage.values
          pos.loc[pos_otc.TradeNumber,'barrier']=pos_otc.barrier.values
          
          if pos_simu_dic:
              pos=pd.concat([pos,pd.DataFrame(pos_simu_dic)])
          else:
              pass
          
          asset_list=pos.UnderlyingCode.unique().tolist()
          
          self.LiV=LinearInterpVol(asset_list, datetime.today().date())
          pos.reset_index(drop=True,inplace=True)

          pos['RQCode']=pos.UnderlyingCode.apply(getRQcode)
          pos.UnderlyingPrice=list(map(lambda x:x.last,pos.RQCode.apply(rqd.current_snapshot)))
          
          
          self.pos_exotic=pos[pos.qty_freq>0]
          # self.pos_exotic.fillna(0,inplace=True)
          # self.pos_exotic.barrier= self.pos_exotic.barrier.fillna(0)
      
          # self.pos_exotic.BarrierRamp=self.pos_exotic.BarrierRamp.fillna(0)
          # self.pos_exotic.StrikeRamp=self.pos_exotic.StrikeRamp.fillna(0)
          # self.pos_exotic.fp=self.pos_exotic.fp.fillna(0)
          
          
          self.pos=pos.drop(self.pos_exotic.index)
          
          self.s_ts=self.pos.UnderlyingPrice
          self.exp_t=[rqd.get_next_trading_date(exp) if exp not in self.total_tradingdates else exp for exp in self.pos.ExerciseDate]
     
          self.ttm=list(map(lambda x:calTradttm(self.total_tradingdates,self.start_time, x),self.exp_t))
          self.ttm=np.where(self.pos.OptionType=='F',0,self.ttm)
         
          
    def __calGreeks(self,delta_s,t_decay):
        
        bsm=list(map(lambda ass,s,k,t,opttype:BSM(s*delta_s,k,(t-t_decay)/252,rf,q
                                                  ,self.LiV.calVol(ass,s*delta_s, k,t-t_decay),opttype),
        self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,self.ttm,self.pos.OptionType))
         
        self.pos['Theta']=[b.theta(1/252) for b in bsm]
        self.pos['Vega']=[b.vega()/100 for b in bsm]
        self.pos[['Theta','Vega']]=self.pos[['Theta','Vega']].multiply(self.pos.TradeAmount,axis=0)
   
          
        res_euro=self.pos.groupby('UnderlyingCode').sum()[['Theta','Vega']].round(0)
 
        acc_list=list(map(lambda acctype,ass,s,k,b,delta_strike,delta_barrier,fp,lev
                  ,qty,startobs_date, endobs_date:
                      AccOption(self.total_tradingdates, acctype, ass, s*delta_s
                            , self.LiV.calVol(ass,s*delta_s, k,calTradttm(self.total_tradingdates,self.start_time,endobs_date)-t_decay) 
                            ,k,b, float(fp), pd.to_datetime(startobs_date), endobs_date
                            ,datetime.combine(rqd.get_next_trading_date(datetime.today().date(),t_decay),self.start_time.time())
                            ,qty,"B", float(lev), float(delta_strike), float(delta_barrier)).getResult()
                 
              ,self.pos_exotic.OptionType,self.pos_exotic.UnderlyingCode,self.pos_exotic.UnderlyingPrice
              ,self.pos_exotic.Strike,self.pos_exotic.barrier
              ,self.pos_exotic.StrikeRamp,self.pos_exotic.BarrierRamp,self.pos_exotic.fp,self.pos_exotic.leverage
              ,self.pos_exotic.qty_freq,self.pos_exotic.FirstObservationDate,self.pos_exotic.ExerciseDate))
            
        self.pos_exotic['Theta']=[acc.theta for acc in acc_list]
        self.pos_exotic['Vega']=[acc.vega for acc in acc_list]
  
         
        res_acc=self.pos_exotic.groupby('UnderlyingCode').sum()[['Theta','Vega']].round(0)
        
        result=pd.concat([res_euro,res_acc],axis=0).groupby('UnderlyingCode')[['Theta','Vega']].sum()
        result['delta_s']=delta_s
        result['ExpireDate']=rqd.get_next_trading_date(datetime.today().date(),t_decay)
        
        return result
    
    def __calGreeksArr(self,delta_s,t_decay):
        # sigma_arr=np.array(list(map(lambda ass,s,k,exp:self.LiV.calVol(ass,s*delta_s, k,(exp-trade_time).days+1)
        #                    ,self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,self.pos.ExerciseDate))).round(4)

        sigma_arr=np.array(list(map(lambda ass,s,k,t:self.LiV.calVol(ass,s*delta_s, k,t-t_decay)
                           ,self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,self.ttm)))
        
        bsm_arr=BSM_ARR(self.pos.UnderlyingPrice.values*delta_s,self.pos.Strike.values,(self.ttm-t_decay)/252, rf, q,sigma_arr,self.pos.OptionType.values)
        
        self.pos['Theta']=bsm_arr.theta(1/252)*self.pos.TradeAmount
        self.pos['Vega']=bsm_arr.vega()/100*self.pos.TradeAmount
   

        res_euro=self.pos.groupby('UnderlyingCode').sum()[['Theta','Vega']].round(0)
      
        
      
        acc_list=list(map(lambda acctype,ass,s,k,b,delta_strike,delta_barrier,fp,lev
                      ,qty,startobs_date, endobs_date:
                          AccOptionArr(self.total_tradingdates, acctype, ass, s*delta_s
                                , self.LiV.calVol(ass,s*delta_s, k,calTradttm(self.total_tradingdates,self.start_time,endobs_date)-t_decay) 
                                ,k,b, float(fp), pd.to_datetime(startobs_date), endobs_date
                                ,datetime.combine(rqd.get_next_trading_date(datetime.today().date(),t_decay),self.start_time.time())
                                ,qty,"B", float(lev), float(delta_strike), float(delta_barrier)).getResult()
                     
                  ,self.pos_exotic.OptionType,self.pos_exotic.UnderlyingCode,self.pos_exotic.UnderlyingPrice
                  ,self.pos_exotic.Strike,self.pos_exotic.barrier
                  ,self.pos_exotic.StrikeRamp,self.pos_exotic.BarrierRamp,self.pos_exotic.fp,self.pos_exotic.leverage
                  ,self.pos_exotic.qty_freq,self.pos_exotic.FirstObservationDate,self.pos_exotic.ExerciseDate))

        self.pos_exotic['Theta']=[acc.theta for acc in acc_list]
        self.pos_exotic['Vega']=[acc.vega for acc in acc_list]
        res_acc=self.pos_exotic.groupby('UnderlyingCode').sum()[['Theta','Vega']].round(0)
     
        
        result=pd.concat([res_euro,res_acc],axis=0).groupby('UnderlyingCode')[['Theta','Vega']].sum()
        result['delta_s']=delta_s
        result['ExpireDate']=rqd.get_next_trading_date(datetime.today().date(),t_decay)
        
        return result
    

    def __calPv(self,delta_s,trade_time):
        ttm_0=list(map(lambda x:calTradttm(self.total_tradingdates,trade_time, x),self.exp_t))      
        bsm=list(map(lambda ass,s,k,t,opttype:BSM(s*delta_s,k,(t)/252,rf,q
                                                  ,self.LiV.calVol(ass,s*delta_s, k,t),opttype),
        self.pos.UnderlyingCode,
        self.pos.UnderlyingPrice,self.pos.Strike,ttm_0,self.pos.OptionType))

        pv=[b.price() for b in bsm]
        pv=np.where(self.pos.OptionType=='F',self.pos.UnderlyingPrice*delta_s,pv)
        pv=pd.Series(index=self.pos.UnderlyingCode,data=pv,dtype=float)
        return pv
    
    def __calExoticPv(self,delta_s,trade_time):
        
        acc_list=list(map(lambda acctype,ass,s,k,b,delta_strike,delta_barrier,fp,lev
                    ,qty,startobs_date, endobs_date:
                        AccOption(self.total_tradingdates, acctype, ass, s*delta_s
                              , self.LiV.calVol(ass,s*delta_s, k,calTradttm(self.total_tradingdates,trade_time,endobs_date)) 
                              ,k,b, float(fp), pd.to_datetime(startobs_date), endobs_date
                              ,trade_time
                              ,qty,"B", float(lev), float(delta_strike), float(delta_barrier)).getResult()
                   
                ,self.pos_exotic.OptionType,self.pos_exotic.UnderlyingCode,self.pos_exotic.UnderlyingPrice
                ,self.pos_exotic.Strike,self.pos_exotic.barrier
                ,self.pos_exotic.StrikeRamp,self.pos_exotic.BarrierRamp,self.pos_exotic.fp,self.pos_exotic.leverage
                ,self.pos_exotic.qty_freq,self.pos_exotic.FirstObservationDate,self.pos_exotic.ExerciseDate))
        
    
        pv_exotic=[acc.bookpv for acc in acc_list]
        pv_exotic=pd.Series(index=self.pos_exotic.UnderlyingCode,data=pv_exotic,dtype=float)

   
        return pv_exotic
       
    def __calPvArr(self,delta_s,trade_time):
        ttm_0=np.array(list(map(lambda x:calTradttm(self.total_tradingdates,trade_time, x),self.exp_t)))
        # sigma_arr=np.array(list(map(lambda ass,s,k,t:self.LiV.calVol(ass,s*delta_s, k,t)
        #                     ,self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,ttm_0))).round(4)
   
        sigma_arr=np.array(list(map(lambda ass,s,k,exp:self.LiV.calVol(ass,s*delta_s, k,(pd.to_datetime(exp)-trade_time).days+1)
                            ,self.pos.UnderlyingCode,self.pos.UnderlyingPrice,self.pos.Strike,self.pos.ExerciseDate))).round(4)

        bsm_arr=BSM_ARR(self.pos.UnderlyingPrice.values*delta_s
                        ,self.pos.Strike.values
                        ,ttm_0/252, rf, q,sigma_arr,self.pos.OptionType.values)
        pv=bsm_arr.price()
        pv=np.where(self.pos.OptionType=='F',self.pos.UnderlyingPrice*delta_s,pv)
        pv=pd.Series(index=self.pos.UnderlyingCode,data=pv,dtype=float)
        return pv
        
    def __calExoticPvArr(self,delta_s,trade_time):
        acc_list=list(map(lambda acctype,ass,s,k,b,delta_strike,delta_barrier,fp,lev
                    ,qty,startobs_date, endobs_date:
                        AccOptionArr(self.total_tradingdates, acctype, ass, s*delta_s
                              , self.LiV.calVol(ass,s*delta_s, k,calTradttm(self.total_tradingdates,trade_time,endobs_date)) 
                              ,k,b, float(fp), pd.to_datetime(startobs_date), endobs_date
                              ,trade_time
                              ,qty,"B", float(lev), float(delta_strike), float(delta_barrier)).getResult()
                   
                ,self.pos_exotic.OptionType,self.pos_exotic.UnderlyingCode,self.pos_exotic.UnderlyingPrice
                ,self.pos_exotic.Strike,self.pos_exotic.barrier
                ,self.pos_exotic.StrikeRamp,self.pos_exotic.BarrierRamp,self.pos_exotic.fp,self.pos_exotic.leverage
                ,self.pos_exotic.qty_freq,self.pos_exotic.FirstObservationDate,self.pos_exotic.ExerciseDate))
        
     
        pv_exotic=[acc.bookpv for acc in acc_list]
        pv_exotic=pd.Series(index=self.pos_exotic.UnderlyingCode,data=pv_exotic,dtype=float)

   
        return pv_exotic
          

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
          self.opttype_arr=np.array([opt.upper()for opt in opttype_arr])
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
        condition=[self.t_arr>=1/252
                    ,(self.t_arr>0)&(self.t_arr<1/252)
                    ,self.t_arr<=0]
        choice=[BSM_ARR(self.s_arr,self.k_arr,self.t_arr-delta_t_arr,self.r_arr,self.q_arr,self.sigma_arr,self.opttype_arr).price()-p_0
                ,np.max([(self.s_arr-self.k_arr)*np.select(self.condition,[1,-1],default=np.nan),np.zeros(self.s_arr.shape)],axis=0)-p_0
                ,0]
        theta_arr=np.select(condition,choice)    
        
        
        # p_0=bsm_arr.price()
        # delta_t_arr=1/252*np.ones(bsm_arr.t_arr.shape)

        # condition=[bsm_arr.t_arr>=1/252
        #             ,(bsm_arr.t_arr>=0)&(bsm_arr.t_arr<1/252)
        #             ,bsm_arr.t_arr<0]
        # choice=[BSM_ARR(bsm_arr.s_arr,bsm_arr.k_arr,bsm_arr.t_arr-delta_t_arr,bsm_arr.r_arr,bsm_arr.q_arr,bsm_arr.sigma_arr,bsm_arr.opttype_arr).price()-p_0
        #         ,np.max([(bsm_arr.s_arr-bsm_arr.k_arr)*np.where(bsm_arr.opttype_arr=='C',1,-1),np.zeros(bsm_arr.s_arr.shape)],axis=0)-p_0
        #         ,0]
        # theta_arr=np.select(condition,choice)
        
        
        
        return theta_arr*self.indicator
    
        
if __name__=='__main__':
    

        st=StressTest()
        delta_s_arr=np.arange(0.9,1.11,0.01).round(2)
        t_decay_arr=np.arange(1,20)
        end_trading_time=datetime(2023,11,24,14,59,0)
        
        
       
        
        varity='UR'
    
        # pos_simu_dic={'UnderlyingCode':["EG2401"],
        #               'Strike':np.array([4123]),
        #               'ExerciseDate':np.array([datetime(2023,12,23).date()]),
        #               'OptionType':['F'],
        #               'TradeAmount':np.array([-10*10])}
        pos_simu_dic={'UnderlyingCode':["MA405"],
                      'Strike':np.array([2480]),
                      'ExerciseDate':np.array([datetime(2023,12,18).date()]),
                      'OptionType':['C'],
                      'TradeAmount':np.array([-3000])}
        tic=datetime.now()
        start_time=datetime.now()
        st_res_arr=st.calStressTestArr(varity,delta_s_arr,t_decay_arr,start_time,end_trading_time
                                # ,pos_simu_dic
                              )
        print("Running time = ", datetime.now() - tic, "s")
        st_theta_arr=st_res_arr['theta']
        st_vega_arr=st_res_arr['vega'] 
        st_pnl_arr=st_res_arr['pnl'].sum()
        
        
        
        
        
        
        
        
        
        
     