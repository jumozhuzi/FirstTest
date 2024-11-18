# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:31:07 2021

@author: chengyilin
"""

import time
# from WindPy import w
# w.start()
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.special import zeta
from math import exp,sqrt
from datetime import date,timedelta,time,datetime
import bisect
import rqdatac as rqd
from chinese_calendar import is_holiday, get_workdays
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")

annual_coeff=252
def pathGeneration(simulations,s_0,mu,sigma,period,freq=1,annual_coeff=annual_coeff):
    '''
    generates spot roads

    Parameters
    ----------
    simulations : int
        Simulations.
    s_0 : float
        The initial spot.
    t : int
        maturity.
    mu : float
        drift.
    sigma : float
        volatility.

    Returns
    -------
    None.

    '''
    # period=np.int(period)
    # simulations=np.int(simulations)
    dt=freq/annual_coeff
    ep=np.random.normal(size=(simulations,period))
    a=np.exp((mu- 0.5 * sigma** 2) * dt +sigma* np.sqrt(dt) *ep)
    s_mat=s_0*np.cumproduct(a,axis=1)
    return s_mat


def LinearInterpVol(asset,s,k,exp_ttm):
     '''
     Parameters
     ----------
     asset: str
          the underlying code
     s : float
         underlying price.
     k : float
         strike
     exp_ttm : float
         calander days to expiration.


\
     Returns
     -------
     vol : float
         IV.

     '''
      # exp_ttm=t_0
      # s=s_0
     ks_ratio=np.arange(0.8,1.22,0.02).round(2)
     volsurface,ttmdays_arr=getVolsurface(asset)
     
     
     idx_t=bisect.bisect_left(ttmdays_arr,exp_ttm)
     idx_ks=bisect.bisect_left(ks_ratio,k/s) 


     condition=[k/s<=ks_ratio[0]
                ,k/s>ks_ratio[-1]]
     choice=[(0,1),(20,21)]     
     cols=np.select(condition,choice,(idx_ks-1,idx_ks+1))
     
     
     if exp_ttm<=ttmdays_arr[0]:
         vol_slice=volsurface[0,cols[0]:cols[1]]
     elif exp_ttm>ttmdays_arr[-1]:
         vol_slice=volsurface[-1,cols[0]:cols[1]]
     else:
         vol_slice=volsurface[idx_t-1:idx_t+1,cols[0]:cols[1]]


     if vol_slice.shape==(1,):
         return vol_slice[0]
     elif vol_slice.shape==(2,):
         ratio_arr=ks_ratio[cols[0]:cols[1]] 
         return (np.diff(vol_slice,axis=0)/np.diff(ratio_arr)*(k/s-ratio_arr[0])+vol_slice[0])[0]
     elif vol_slice.shape==(2,1):
         t_arr=ttmdays_arr[idx_t-1:idx_t+1] 
         return ((np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0])[0]
     else:
         t_arr=ttmdays_arr[idx_t-1:idx_t+1] 
         ratio_arr=ks_ratio[cols[0]:cols[1]] 
         term_diff_vol=(np.diff(vol_slice,axis=0)[0]/np.diff(t_arr))*(exp_ttm-t_arr[0])+vol_slice[0,:]
         return (np.diff(term_diff_vol)/np.diff(ratio_arr)*(k/s-ratio_arr[0])+term_diff_vol[0])[0]


class BSM():
    '''
    Generalized BSM. Can be used to price European option on stocks,stocks paying a continuous
    dividend yield, options on future, and currency options.

    Parameters
    ----------
    s : float
        spot price           
    k : float
        strike price
    t : float
        The time to maturity.
    r : float
        risk free rate
    q  : float
         dividend rate for stocks or foreign interest rate for currency option

    sigma : float
            implied volatility
    opttype : str
              option type
    Returns
    -------
    None.

    '''
    
   
    def __init__(self,s,k,t,r,q,sigma,opttype):
         self.s=s
         self.k=k
         self.r=r
         self.q=q
         self.sigma=sigma
         self.t=t
         self.opttype=opttype.upper()
         
         if self.t>0 and self.sigma!=0:
             self.__b_future()
             self.d1=(np.log(self.s/self.k)+self.t*(self.b+0.5*self.sigma**2))/(self.sigma*np.sqrt(self.t))
             self.d2=self.d1-self.sigma*np.sqrt(self.t)
     
    def __b_stock(self):
        self.b=self.r
        
    def __b_stock_div(self):
        self.b=self.r-self.q
     
    def __b_future(self):
         self.b=0
    
    def __b_margined_future(self):
        self.b=0
        self.r=0
        
    def __b_currency(self):
        self.b=self.r-self.q
        
   
         
    def price(self):
          
         if self.t<0:
             return 0
         elif self.t==0:
             return np.where(self.opttype=='C',max(self.s-self.k,0),max(self.k-self.s,0))
         else:
             if self.opttype=='C' :
                 return self.s*np.exp((self.b-self.r)*self.t)*norm.cdf(self.d1)-self.k*np.exp(-1*self.r*self.t)*norm.cdf(self.d2)
             elif self.opttype=='P':
                 return self.k*np.exp(-1*self.r*self.t)*norm.cdf(-1*self.d2)-self.s*np.exp((self.b-self.r)*self.t)*norm.cdf(-1*self.d1)
             else:
                 print('Wrong option type input')
             
    def delta(self):
         if self.t>0:
             if self.opttype=='C':
                 return norm.cdf(self.d1)*np.exp((self.b-self.r)*self.t)
             elif self.opttype=='P':
                 return (norm.cdf(self.d1)-1)*np.exp((self.b-self.r)*self.t)
             else:
                 print('Wrong option type input')
         else:
             # drop delta when expired
             return 0
     
    def gamma(self):
         if self.t>0:
             return norm.pdf(self.d1)/(self.sigma*self.s*np.sqrt(self.t))
         else:
             return 0
         
    def vega(self):
             '''
             单位是1，即波动率变化1，期权价格变化多少，所以转成成1%就是vega/100
             Returns
             -------
             TYPE
                 DESCRIPTION.
 
             '''
             if self.t>0:
               return (self.s*norm.pdf(self.d1)*np.sqrt(self.t))/np.exp(self.r*self.t)
             else:
                 return 0
         
    def theta(self,delta_t):
         '''
         用bs公式算：单位是年，即时间变化一年，期权价格降低多少，所以转换成天就是theta/245
         用差分算则可直接用
 
         Parameters
         ----------
         delta_t : float
             yearly delta time to for theta, generally is 1/365 .
 
         Returns
         -------
         '''
         p_0=self.price()
         if self.t>=1/annual_coeff :
             p_1=BSM(self.s,self.k,self.t-delta_t,self.r,self.q,self.sigma,self.opttype).price()
             return p_1-p_0
         elif self.t>0:
              p_2=BSM(self.s,self.k,0,self.r,self.q,self.sigma,self.opttype).price()
              return p_2-p_0
         else:
             return 0



def calTradttm_NEW(trading_dates,start_time,expire_date):
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
    cdt_time=[now_time<time(9,0,0)
              ,(now_time>=time(9,0,0) )& (now_time<time(11,30,0))
              ,(now_time>=time(11,30,0)) & (now_time<time(13,30,0))
              ,(now_time>=time(13,30,0)) & (now_time<time(15,0,0))
              ,(now_time>=time(15,0,0)) & (now_time<time(21,0,0))
              ,(now_time>=time(21,0,0)) & (now_time<time(23,0,0))
              ]
    
    cho_intra=[4/6
               ,((datetime(start_time.year,start_time.month,start_time.day,11,30,0)-start_time).seconds/3600+1.5)/6
               ,1.5/6
               ,((datetime(start_time.year,start_time.month,start_time.day,15,0,0)-start_time).seconds/3600)/6
               ,1
               ,((datetime(start_time.year,start_time.month,start_time.day,23,0,0)-start_time).seconds/3600+4)/6
               ]
    cho_trd_date=[start_time.date()
                  ,start_time.date()
                  ,start_time.date()
                  ,start_time.date()
                  ,np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
                  ,np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
                  ]
    
    intra_hours=np.select(cdt_time,cho_intra)
    trd_date=np.select(cdt_time,cho_trd_date,np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0])
    
    # if now_time<time(9,0,0):
    #     intra_hours=4/6
    #     trd_date=start_time.date()
    # elif now_time<time(11,30,0):
    #     intra_hours=(datetime(start_time.year,start_time.month,start_time.day,11,30,0)-start_time).seconds/3600
    #     intra_hours=(intra_hours+1.5)/6
    #     trd_date=start_time.date()
    # elif now_time<time(13,30,0):
    #     intra_hours=1.5/6
    #     trd_date=start_time.date()
    # elif now_time<time(15,0,0):
    #     intra_hours=(datetime(start_time.year,start_time.month,start_time.day,15,0,0)-start_time).seconds/3600
    #     intra_hours=intra_hours/6
    #     trd_date=start_time.date()
    # elif now_time<time(21,0,0):
    #     intra_hours=1
    #     trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    # elif now_time<time(23,0,0):
    #     intra_hours=(datetime(start_time.year,start_time.month,start_time.day,23,0,0)-start_time).seconds/3600
    #     intra_hours=(intra_hours+4)/6
    #     trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    # else:
    #     intra_hours=0
    #     trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    
    if start_time.date() not in trading_dates:
        next_idx=np.argmax(np.array( trading_dates)>=start_time.date())
        total_days=trading_dates.index(expire_date)-next_idx    
    else:
        total_days=trading_dates.index(expire_date)-trading_dates.index(trd_date)

    
    return total_days+intra_hours

def calTradttm(trading_dates,start_time,expire_date):
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
        trd_date=start_time.date()
    elif now_time<time(11,30,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,11,30,0)-start_time).seconds/3600
        intra_hours=(intra_hours+1.5)/6
        trd_date=start_time.date()
    elif now_time<time(13,30,0):
        intra_hours=1.5/6
        trd_date=start_time.date()
    elif now_time<time(15,0,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,15,0,0)-start_time).seconds/3600
        intra_hours=intra_hours/6
        trd_date=start_time.date()
    elif now_time<time(21,0,0):
        intra_hours=1
        trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    elif now_time<=time(23,0,0):
        intra_hours=(datetime(start_time.year,start_time.month,start_time.day,23,0,0)-start_time).seconds/3600
        intra_hours=(intra_hours+4)/6
        trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    else:
        intra_hours=0
        trd_date=np.array(trading_dates)[np.array(trading_dates)>start_time.date()][0]
    
    if start_time.date() not in trading_dates:
        next_idx=np.argmax(np.array( trading_dates)>=start_time.date())
        total_days=trading_dates.index(expire_date)-next_idx    
    else:
        total_days=trading_dates.index(expire_date)-trading_dates.index(trd_date)

    
    return total_days+intra_hours

class AccOption():
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
        
        # if self.barrier==0 or self.barrier=="":
        #     self.barrier==""
       
        
       # if acc.barrier==0 or acc.barrier=="":
       #     acc.barrier==""
        # self.trading_days=list(map(lambda x:datetime.strftime(x,"%Y-%m-%d"),self.trading_days))
    
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

        # acc.sigma_arr=[acc.sigma]*len(acc.opttype)*acc.trading_days
        # acc.ttm_list=acc.ttm_list.repeat(len(acc.opttype)) #due to ttm_list is array_like
        # acc.buy_sell=acc.buy_sell*acc.trading_days
        # acc.opttype=acc.opttype*acc.trading_days
        # acc.strikes=acc.strikes*acc.trading_days
        # acc.ratio=acc.ratio*acc.trading_days
        
        
        # dfparams=pd.DataFrame()
        # dfparams['sigma_arr']=acc.sigma_arr  
        # dfparams['ttm_list']=acc.ttm_list
        # dfparams['buy_sell']=acc.buy_sell
        # dfparams['opttype']=acc.opttype
        # dfparams['strikes']=acc.strikes
        # dfparams['ratio']=acc.ratio
        
        # dfparmas=dfparams.drop(dfparams[dfparams.ttm_list<=0].index)
        
  
        if self.sigma=="":
            bsm_list=list(map(lambda k,t,opt_type:BSM(self.S_0,k,t,AccOption.__rf,0
                                                      ,LinearInterpVol(self.asset,self.S_0,k,t*AccOption.__annual_coeff),opt_type)
                      , self.strikes
                      ,self.ttm_list/AccOption.__annual_coeff
                      ,self.opttype))
        else:
            bsm_list=list(map(lambda k,t,vol,opt_type:BSM(self.S_0,k,t,AccOption.__rf,0,vol,opt_type)
                      , self.strikes
                      ,self.ttm_list/AccOption.__annual_coeff
                      ,self.sigma_arr,self.opttype))

        
        
        
            # bsm_list=list(map(lambda k,t,vol,opt_type:BSM(acc.S_0,k,t,0.03,0,vol,opt_type)
            #           , dfparams.strikes
            #           ,dfparams.ttm_list/annual_coeff
            #           ,dfparams.sigma_arr,dfparams.opttype))
        

        
        bsm_arr=pd.DataFrame([[bsm.price(),bsm.delta(),bsm.gamma(),bsm.vega()/100,bsm.theta(1/AccOption.__annual_coeff)] for bsm in bsm_list])
        bsm_arr['ttm']=self.ttm_list
        # bsm_arr=pd.DataFrame([[bsm.price(),bsm.delta(),bsm.gamma(),bsm.vega()/100,bsm.theta(1/annual_coeff)] for bsm in bsm_list])
        # bsm_arr['ttm']=acc.ttm_list
        
        bsm_arr.columns=['cashflow','delta','gamma','vega','theta','ttm']
        bs_idx_greeks=np.where(np.array(self.buy_sell)=="B",-1,1) #for greeks calculation AND book pv!
        bs_idx_pv=np.where(np.array(self.buy_sell)=="B",1,-1) #b represent cash flow
        
        bs_idx=np.repeat(bs_idx_greeks,bsm_arr.shape[-1]).reshape(bsm_arr.shape)
        bs_idx[:,0]=bs_idx_pv
        
        qty_ratio=np.array(self.ratio)*self.qty_freq
        
        # bs_idx=np.where(np.array(acc.buy_sell)=="B",-1,1) 
        # qty_ratio=np.array(acc.ratio)*abs(acc.qty_freq)
        
        bsm_total=bsm_arr*bs_idx
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
           opt_type='C'
        elif self.acctype=="ASPUT":
            opt_type='P'
        else:
            print("Wrong AS Option Type!")
            
        if self.sigma=="":
            bsm_list=list(map(lambda t:BSM(self.S_0,self.strike,t,AccOption.__rf,0,LinearInterpVol(self.asset,self.S_0,self.strike,t*AccOption.__annual_coeff),opt_type)
                      ,self.ttm_list/AccOption.__annual_coeff))
        else:
            bsm_list=list(map(lambda t:BSM(self.S_0,self.strike,t,AccOption.__rf,0,self.sigma,opt_type)
                      ,self.ttm_list/AccOption.__annual_coeff))
 
        bsm_arr=pd.DataFrame([[bsm.price(),bsm.delta(),bsm.gamma(),bsm.vega()/100,bsm.theta(1/AccOption.__annual_coeff)] for bsm in bsm_list])

        # bsm_list=list(map(lambda t:BSM(acc.S_0,acc.strike,t,0.03,0,acc.sigma,'P')
        #           ,acc.ttm_list/annual_coeff))
   
        # bsm_arr=pd.DataFrame([[bsm.price(),bsm.delta(),bsm.gamma(),bsm.vega()/100,bsm.theta(1/annual_coeff)] for bsm in bsm_list])



        bsm_arr.columns=['cashflow','delta','gamma','vega','theta']
        
        bs_idx_greeks=-1 if self.customer_bs=="B" else 1
        bs_idx_pv=1 if self.customer_bs=="B" else -1
        
        
        bsm_total=bsm_arr.multiply(self.qty_freq,axis=0)
        
        # bsm_total=bsm_arr.multiply(100,axis=0)
        
        
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
        
def calIV(opttype, s, k, t, rf,q ,target):
    '''
    

    Parameters
    ----------
    opttype : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    target : float
        option price.

    Returns
    -------
    iv : TYPE
        DESCRIPTION.

    '''

    high = 2
    low = 0
    # iv = (high + low) / 2
    while (high-low) > 1.0e-5:
   
        p=BSM(s,k,t,rf,q,(high + low) / 2,opttype).price()
        if p>target:
            high = (high + low) / 2
        else:
            low = (high + low) / 2
    iv = (high + low) / 2
    return iv    

     
def StandardBarrier(typeFlag, s,k, h, t,sigma,reb,dt,rf,b=0):
    '''
    Parameters
    ----------
    typeFlag : str
              option type,only support CUI,CUO,CDI,CDO,PUI,PUO,PDI,PDO
    s : float
        underlying spot
    k : float
        strike
    h : float
        barrier level
    reb : float
        rebate 
    t : float
        time to maturity,yearly
    dt : float, optional
        The time between monitoring events
    rf : flaot
        risk free rate
    sigam : float
            implied volatility
    b : float
        any cost rate. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # typeFlag = "cdo"
    # s = 88
    # k = 110
    # h = 90
    # reb =10
    # t = 1/annual_coeff
    # rf = 0
    # b = 0
    # sigma = 0.2
    # dt = 1/365
    
    typeFlag=typeFlag.upper()
    beta=zeta(0.5)/sqrt(2*np.pi)*-1
    
    if h > s:
        H_adj = h * np.exp(beta *sigma*np.sqrt(dt))
    elif h<s:
        H_adj = h * np.exp(-1*beta *sigma*np.sqrt(dt))
    else:
        H_adj=''
        

    mu = (b-sigma**2/2)/(sigma**2)
    lamb = np.sqrt(mu ** 2 + 2 * rf / sigma ** 2)
    sigma_sqrt_t=sigma*np.sqrt(t)
    mult_1=(1 + mu) *sigma_sqrt_t
    
    x1 = np.log(s / k) /sigma_sqrt_t + mult_1
    x2 = np.log(s / H_adj) / sigma_sqrt_t +mult_1
    y1 = np.log(H_adj ** 2 / (s * k)) /  sigma_sqrt_t + mult_1
    y2 = np.log(H_adj / s) / sigma_sqrt_t +mult_1
    z = np.log(H_adj / s) / sigma_sqrt_t+ lamb * sigma_sqrt_t
    
    if typeFlag == "CDI" or typeFlag == "CDO":
        eta = 1
        phi = 1
    elif typeFlag == "CUI" or typeFlag == "CUO" :
        eta = -1
        phi = 1
    elif typeFlag == "PDI" or typeFlag == "PDO" :
        eta = 1
        phi = -1
    elif typeFlag == "PUI" or typeFlag == "PUO" :
        eta = -1
        phi = -1
    else:
        print('Wrong Type')
        return 0
    
    mult_2=phi * s * np.exp((b - rf) * t)
    discount=np.exp(-rf*t) 
    
    f1 = mult_2 * norm.cdf(phi * x1) - phi * k *discount * norm.cdf(phi * x1 - phi * sigma_sqrt_t)
    f2 = mult_2 * norm.cdf(phi * x2) - phi *k* discount* norm.cdf(phi * x2 - phi *sigma_sqrt_t)
    f3= mult_2* (H_adj / s) **(2 * (mu + 1)) * norm.cdf(eta * y1) - phi * k * discount* (H_adj / s)**(2 * mu) * norm.cdf(eta * y1 - eta *  sigma_sqrt_t)
    f4 = mult_2* (H_adj / s) **(2 * (mu + 1)) *norm.cdf(eta * y2) - phi * k *discount* (H_adj / s) **(2 * mu) * norm.cdf(eta * y2 - eta *  sigma_sqrt_t)
    f5= reb *discount * (norm.cdf(eta * x2 - eta *sigma_sqrt_t) - (H_adj / s) **(2 * mu) * norm.cdf(eta * y2 - eta *sigma_sqrt_t))
    f6 = reb * ((H_adj / s)**(mu + lamb) * norm.cdf(eta * z) + (H_adj / s)**(mu - lamb) *norm.cdf(eta * z - 2 * eta * lamb *  sigma_sqrt_t))
    
    
    if k > H_adj:
            if typeFlag=='CDI':
                return f3 + f5
            elif typeFlag=="CUI":
                return f1 + f5
            elif typeFlag=="PDI":
                return f2 - f3 + f4 + f5
            elif typeFlag=="PUI":
                return f1 - f2 + f4 + f5
            elif typeFlag=="CDO":
                return f1 - f3 + f6
            elif typeFlag=="CUO":
                return  f6
            elif typeFlag=="PDO":
                return  f1 - f2 + f3 - f4 + f6
            else: 
            # typeFlag=="puo"
                return f2 - f4 + f6
    else: 
    # k< H_adj :
            if typeFlag=='CDI':
                return f1 - f2 + f4 + f5
            elif typeFlag=="CUI":
                return f2 - f3 + f4 + f5
            elif typeFlag=="PDI":
                return  f1 + f5
            elif typeFlag=="PUI":
                return f3 + f5
            elif typeFlag=="CDO":
                return  f2 + f6 - f4
            elif typeFlag=="CUO":
                return   f1 - f2 + f3 - f4 + f6
            elif typeFlag=="PDO":
                return  f6
            else: 
            # typeFlag=="puo"
                return  f1 - f3 + f6
            
            

            

def EuroOnSpread(s,k,ttm_cal,ttm_trd,vol,opType,rf):
    '''
    

    Parameters
    ----------
    s : float
        Spot price.
    k : float
        Strike.
    ttm_cal : TYPE
        日历日调整，n/365
    ttm_trd : TYPE
        交易日调整，n/annual_coeff
    vol : float
        Absulote value of divation.
    opType : str
        The category of option.
    rf : float
        Risk free rate.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    annual_coeff=annual_coeff
    volatility = vol *np.sqrt(ttm_trd * annual_coeff)
    
    disValueFac = np.exp(-1*rf*ttm_cal)
    
    standarization = (k - s) / volatility
    
    opType=opType.upper()
    if opType =="C":
       return disValueFac*((k-s)*(norm.cdf(standarization)-1)+volatility/np.sqrt(2*np.pi)*np.exp(-1*standarization**2/2))
  
    else:
        return disValueFac * ((k-s) *norm.cdf(standarization) + volatility / np.sqrt(2 *np.pi)* np.exp(-1*standarization**2 / 2))

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
        pv_arr=call_part+np.select(self.condition,choice,default=0)
        
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
    
class StandardBarrierArr():
    
    def __init__(self,opttype,direction,move,s,strike,barrier,reb,dt,t,sigma,intra_to_next_obs,rf,b,annual_coeff=annual_coeff):
        '''
        Standard barrier option. Consider rebate only when it's knock-out option.
        Parameters
        ----------
        opttype : List[str] or Str
            Like call or put. Small latter.
        direction : List[str] or Str
            Like up or down. Small latter.
        move : List[str] or str
            Like out or in. Small latter.
        s : Array_like or float
            Spot.
        strike : Array_like or float
            option strike.
        barrier : Array_like or float
            Barrier .
        reb : Array_like or float
            Option rebate.
        dt : Array_like or float
            Time between monitoring events.
        t : Array_like or float
            Time to maturity.
        sigma : Array_like or float
            Volatility.
        rf : float
            risk free rate.
        b : float
            cost rate.

        Returns
        -------
        None.

        '''
        self.type=np.array(opttype)
        self.direction=np.array(direction)
        self.move=np.array(move)
        self.s=np.array(s)
        self.strike=np.array(strike)
        self.barrier=np.array(barrier)
        self.reb=np.array(reb)
        self.dt=np.array(dt)
        self.t=np.array(t)
        self.sigma=np.array(sigma)
        # self.current_obs_time=current_obs_time
        self.intra_to_next_obs=np.ones(shape=self.t.shape)*intra_to_next_obs
        # self.obs_point_list=np.array(obs_point_list)
        self.rf=rf
        self.b=b
        self.annual_coeff=annual_coeff
        self.d=0.0005
                 
        self.eta=np.select([self.direction=='up',self.direction=='down'], [-1,1],0)
        self.phi=np.select([self.type=='call',self.type=='put'],[1,-1],0)
        # self.intrahour=getIntraHours(self.current_obs_time)
        # self.t=(self.t+self.intrahour)/self.annual_coeff
    
    
        mult_2=self.phi * self.s * np.exp((self.b - self.rf) * self.t)
        discount=np.exp(-self.rf*self.t) 
        beta=zeta(0.5)/np.sqrt(2*np.pi)*-1
   
    
        cdt_b_adj=[self.barrier>=self.s,
                    self.barrier<self.s]
        cho_b_adj=[self.barrier * np.exp(beta *self.sigma*np.sqrt(self.dt))
                ,self.barrier* np.exp(-1*beta *self.sigma*np.sqrt(self.dt))
                ]
        
        self.B_adj=np.where(self.dt==0,self.barrier,np.select(cdt_b_adj,cho_b_adj,10000))

        mu = (self.b-self.sigma**2/2)/(self.sigma**2)
        lamb = np.sqrt(mu ** 2 + 2 * self.rf / self.sigma ** 2)
        sigma_sqrt_t=self.sigma*np.sqrt(self.t)
        mult_1=(1 + mu) *sigma_sqrt_t
        
        x1 = np.log(self.s / self.strike) /sigma_sqrt_t + mult_1
        x2 = np.log(self.s / self.B_adj) / sigma_sqrt_t +mult_1
        y1 = np.log( self.B_adj ** 2 / (self.s * self.strike)) /  sigma_sqrt_t + mult_1
        y2 = np.log(self.B_adj / self.s) / sigma_sqrt_t +mult_1
        z = np.log(self.B_adj / self.s) / sigma_sqrt_t+ lamb * sigma_sqrt_t
    
    
    
        self.f1 = mult_2 * norm.cdf(self.phi * x1) - self.phi * self.strike *discount * norm.cdf(self.phi * x1 - self.phi * sigma_sqrt_t)
        self.f2 = mult_2 * norm.cdf(self.phi * x2) - self.phi *self.strike * discount* norm.cdf(self.phi * x2 - self.phi *sigma_sqrt_t)
        self.f3= mult_2* (self.B_adj  / self.s) **(2 * (mu + 1)) * norm.cdf(self.eta * y1) - self.phi * self.strike * discount* (self.B_adj  / self.s)**(2 * mu) * norm.cdf(self.eta * y1 - self.eta *  sigma_sqrt_t)
        self.f4= mult_2* (self.B_adj  / self.s) **(2 * (mu + 1)) *norm.cdf(self.eta * y2) - self.phi * self.strike * discount* (self.B_adj  / self.s) **(2 * mu) * norm.cdf(self.eta * y2 - self.eta *  sigma_sqrt_t)
        self.f5= self.reb *discount * (norm.cdf(self.eta * x2 - self.eta *sigma_sqrt_t) - (self.B_adj  / self.s) **(2 * mu) * norm.cdf(self.eta * y2 - self.eta *sigma_sqrt_t))
        self.f6 = self.reb * ((self.B_adj  / self.s)**(mu + lamb) * norm.cdf(self.eta * z) + (self.B_adj  / self.s)**(mu - lamb) *norm.cdf(self.eta * z - 2 * self.eta * lamb *  sigma_sqrt_t))
        
        self.bsm_call=BSM_ARR(self.s, self.strike,self.t, self.rf, self.b, self.sigma,'C').price()
        self.bsm_put=BSM_ARR(self.s, self.strike,self.t, self.rf, self.b, self.sigma,'P').price()



    def price_out(self):

        cdt_type=[(self.type=='call') & (self.direction=='up') 
                    # ,(self.type=='call')& (self.direction=='up') &(self.move=='in')
                    ,(self.type=='call')& (self.direction=='down') 
                    # ,(self.type=='call')& (self.direction=='down') &(self.move=='in')
                    ,(self.type=='put')& (self.direction=='up') 
                    # ,(self.type=='put')& (self.direction=='up') &(self.move=='in')
                    ,(self.type=='put')& (self.direction=='down')
                    # ,(self.type=='put')& (self.direction=='down') &(self.move=='in')
                    ]
       
        choice_1=[self.f6
                # ,self.f1 + self.f5
                ,self.f1- self.f3+ self.f6
                # ,self.f3 + self.f5
                ,self.f2 - self.f4 + self.f6
                # ,self.f1 - self.f2 + self.f4 + self.f5
                ,self.f1 - self.f2+ self.f3 - self.f4 + self.f6
                # ,self.f2 - self.f3 + self.f4 + self.f5
                ]
        choice_2=[self.f1 - self.f2 + self.f3 - self.f4 + self.f6
                # ,self.f2 - self.f3 + self.f4 + self.f5
                ,self.f2 - self.f4 + self.f6
                # ,self.f1 - self.f2 + self.f4 + self.f5
                ,self.f1 - self.f3 + self.f6
                # ,self.f3 + self.f5
                ,self.f6
                # ,self.f1 + self.f5 
                ]
    
        
        cdt_k_badj=[self.strike>self.B_adj
                   ,self.strike<=self.B_adj
                   ]
        
        cho_pv=[np.select(cdt_type,choice_1,0)
                ,np.select(cdt_type,choice_2,0)]
        self.norm_pv=np.select(cdt_k_badj,cho_pv,0)
         
        cho_exp_pv=[np.where(self.s>=self.barrier,self.reb,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s<=self.barrier,0,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    ,np.where(self.s<=self.barrier,self.reb,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s>=self.barrier,0,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
            
                    ,np.where(self.s>=self.barrier,self.reb,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s<=self.barrier,0,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    ,np.where(self.s<=self.barrier,self.reb,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s>=self.barrier,0,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    ]
     
        #expired pv only for out type of option!
        self.pv_exp_out=np.select(cdt_type,cho_exp_pv)
        
        cdt_expire=[self.t==0
                    ,self.t<0
                    ,self.t>0]

        cho_pv=[self.pv_exp_out
                ,0
                ,self.norm_pv]
        self.pv=np.select(cdt_expire,cho_pv)

        # 如果是连续观察dt==0，直接判断是否敲出敲入
        # 如果是离散观察dt!=0，先判断是否是观察点，再判断是否敲入敲出
        cdt_out=[(self.direction=='up') & (self.move=='out') & (self.s>=self.barrier)
                    ,(self.direction=='down') & (self.move=='out') &(self.s<=self.barrier)] 
        # cho_out=[self.reb*np.exp(-self.rf*(self.intrahour/self.annual_coeff)),self.reb*np.exp(-self.rf*(self.intrahour/self.annual_coeff))]
        cho_out=[self.reb,self.reb]
        self.out_pv=np.select(cdt_out,cho_out,self.pv)      
        
        # self.obs_bool=[self.current_obs_time in obs_list for obs_list in self.obs_point_list]
        # self.discret_pv=np.where(self.obs_bool*self.out_pv==0,self.pv,self.obs_bool*self.out_pv)
        
        
        # cdt_obspoint=[self.current_obs_time in self.obs_point_list]
        # cho_obspoint=[self.out_pv]
        
        cho_obspoint=[self.reb*np.exp(-self.rf*(self.intra_to_next_obs/self.annual_coeff))
                      ,self.reb*np.exp(-self.rf*(self.intra_to_next_obs/self.annual_coeff))]
       
        # cho_obspoint=[self.s*exp(self.b-self.rf)*(self.intra_to_next_obs/self.annual_coeff)*(norm.cdf(self.get_d1(self.strike))-norm.cdf(self.get_d1(self.B_adj)))-self.strike*(1-norm.cdf(self.get_d2(self.B_adj))-norm.cdf(self.get_d2(self.strike)))+self.reb*exp(-self.rf*self.intra_to_next_obs)*norm.cdf(self.get_d2(self.B_adj))
        #               ,self.reb]
        # cho_obspoint=[StandardBarrier(typeFlag, s, k, h, t, sigma, reb, dt, rf)]
        self.discret_pv=np.select(cdt_out,cho_obspoint,self.pv)
        
        
        cdt_dt=[self.dt==0,self.dt>0]
        cho_dis_pv=[self.out_pv,self.discret_pv]
        # cho_dis_pv=[self.out_pv,self.pv]
        final_pv=np.select(cdt_dt,cho_dis_pv)
        
        final_pv=np.where(final_pv<0,0,final_pv)
        final_pv=np.where(self.t<0,0,final_pv)
        return final_pv
    
    def price_exp(self):
        
        cdt_type=[
            # (self.type=='call') & (self.direction=='up') 
                    (self.type=='call')& (self.direction=='up') &(self.move=='in')
                    # ,(self.type=='call')& (self.direction=='down') 
                    ,(self.type=='call')& (self.direction=='down') &(self.move=='in')
                    # ,(self.type=='put')& (self.direction=='up') 
                    ,(self.type=='put')& (self.direction=='up') &(self.move=='in')
                    # ,(self.type=='put')& (self.direction=='down')
                    ,(self.type=='put')& (self.direction=='down') &(self.move=='in')
                    ]
        cho_exp_pv=[
        # np.where(self.s>=self.barrier,self.reb,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    np.where(self.s<=self.barrier,0,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s<=self.barrier,self.reb,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
                    ,np.where(self.s>=self.barrier,0,np.max([self.s-self.strike,np.zeros(self.s.shape)],axis=0))
            
                    # ,np.where(self.s>=self.barrier,self.reb,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    ,np.where(self.s<=self.barrier,0,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    # ,np.where(self.s<=self.barrier,self.reb,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    ,np.where(self.s>=self.barrier,0,np.max([self.strike-self.s,np.zeros(self.s.shape)],axis=0))
                    ]
     
        
        self.pv_exp_in=np.select(cdt_type,cho_exp_pv)
        
        cdt_move=[self.move=='out'
                  ,self.move=='in']
        cho_exp=[self.pv_exp_out,self.pv_exp_in]
        self.pv_exp=np.select(cdt_move,cho_exp)
        
        
    def price(self):
        cdt_type=[(self.type=='call') & (self.move=='out')
                   ,(self.type=='call')& (self.move=='in')
                   ,(self.type=='put')&(self.move=='out')
                   ,(self.type=='put')& (self.move=='in')
                  ]
        cho_price=[self.price_out()
                   ,self.bsm_call-self.price_out()
                   ,self.price_out()
                   ,self.bsm_put-self.price_out()
                   ]
        pv=np.select(cdt_type,cho_price)
        return pv
 
    
    def delta(self):
        # pv_0=self.pv()
      
        pv_1=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1+self.d)
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()
        pv_2=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1-self.d)
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()
        
        delta=(pv_1-pv_2)/(self.s*2*self.d)
        return delta
    
    def gamma(self):
        
        delta_1=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1+self.d)
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).delta()
        delta_2=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1-self.d)
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).delta()
        gamma=(delta_1-delta_2)/(2*self.s*self.d)
        
        # pv_1=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1+self.d)
        #                         ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()
        # pv_2=StandardBarrierArr(self.type,self.direction,self.move,self.s*(1-self.d)
        #                         ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()
        # pv_0=StandardBarrierArr(self.type,self.direction,self.move,self.s
        #                         ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()
        # gamma=(pv_1+pv_2-2*pv_0)/(self.s*self.d**2)
        
        return gamma
    
    def theta(self,delta_t):
        p_0=self.price()
        self.price_exp()
        delta_t_arr=delta_t*np.ones(self.t.shape)
        condition=[self.t>=1/annual_coeff 
                    ,(self.t>0)&(self.t<1/annual_coeff)
                    ,self.t<=0]
        choice=[StandardBarrierArr(self.type,self.direction,self.move,self.s
                                ,self.strike,self.barrier,self.reb,self.dt,self.t-delta_t_arr,self.sigma,self.intra_to_next_obs,self.rf,self.b).price()-p_0
                ,self.pv_exp-p_0
                ,0]
        theta_arr=np.select(condition,choice)    
        return theta_arr
    
    def vega(self):
        p_0=self.price()
        pv_1=StandardBarrierArr(self.type,self.direction,self.move,self.s
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma+0.0001,self.intra_to_next_obs,self.rf,self.b).price()
        # pv_2=StandardBarrierArr(self.type,self.direction,self.move,self.s
        #                         ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma-0.01,self.current_obs_time,self.obs_point_list,self.rf,self.b).price()
        # vega=(pv_1-pv_2)/0.02
        vega=(pv_1-p_0)*100
        return vega
    
    def vomma(self):
        # p_0=self.price()
        vega_1=StandardBarrierArr(self.type,self.direction,self.move,self.s
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma+0.0001,self.intra_to_next_obs,self.rf,self.b).vega()
        vega_2=StandardBarrierArr(self.type,self.direction,self.move,self.s
                                ,self.strike,self.barrier,self.reb,self.dt,self.t,self.sigma-0.0001,self.intra_to_next_obs,self.rf,self.b).vega()

        vomma=(vega_1-vega_2)/0.0002
        return vomma

def getIntraHours(trd_time):
    cdt_time=[trd_time<time(9,0,0)
              ,trd_time<time(11,30,0)
              ,trd_time<time(13,30,0)
              ,trd_time<time(15,0,0)
              ,trd_time==time(15,0,0)
              ,trd_time>=time(23,0,0)]
    cho_intra=[4/6
               ,2/6
               ,1.5/6
               ,0.5/6
               ,0
               ,5/6]
    return np.select(cdt_time,cho_intra,0)

class BarrierAccOption():
    __annual_coeff=annual_coeff
    __rf=0.03
    __q=0
    def __init__(self,trading_list,opttype,s,s_0,strike,barrier,sigma,fix_income,reb,dt,lev_day,cust_bs,qty_freq,lev_exp
                  ,next_obs_date,end_obs_date,trading_time):
        self.opttype=opttype
        self.s=s
        self.s_0=s_0
        self.strike=strike
        self.barrier=barrier
        self.sigma=sigma
        self.reb=reb
        self.dt=dt
        self.lev_day=lev_day
        self.lev_exp=lev_exp
        self.cust_bs=1 if cust_bs.upper()=="B" else -1
        self.qty_freq=qty_freq
        self.trading_time=trading_time
        self.fix_income=fix_income
    
        # if self.trading_time.time()>time(15,0,0):
        self.left_obs_days=trading_list.index(end_obs_date)-trading_list.index(next_obs_date)+1
        self.ttm=calTradttm(trading_list,self.trading_time, end_obs_date)
        self.ramp=0.00001
        
        # if type(self.s)==list:
        #     self.s_len=len(self.s)
        # else:
        #     list(self.s)
        # self.s_list=list(self.s)
        
         
    
    
    def acccall(self):
        self.opt_list  =['call','put']
        self.dirt_list =['up','up']
        self.move_list =['out','out']
        self.custbs_idx=[1,-1]
        self.reb_list  =[self.reb,0]
        self.strikes=[self.strike]*2
        self.barriers=[self.barrier]*2
        self.ratio=[1,self.lev_day]
        
    def accput(self):
        self.opt_list  =['put','call']
        self.dirt_list =['down','down']
        self.move_list =['out','out']
        self.custbs_idx=[1,-1]
        self.reb_list  =[self.reb,0]
        self.strikes=[self.strike]*2
        self.barriers=[self.barrier]*2
        self.ratio=[1,self.lev_day]
    
    def acccall_forward(self):
        self.opt_list  =['call','put','call','put']
        self.dirt_list =['up','up','up','up']
        self.move_list =['out','out','in','in']
        self.custbs_idx=[1,-1,1,-1]
        self.reb_list  =[0]*4
        self.strikes=[self.strike,self.strike,self.s_0,self.s_0]
        self.barriers=[self.barrier]*4
        self.ratio=[1,self.lev_day,1,1]
        
    def accput_forward(self):
        self.opt_list  =['put','call','put','call']
        self.dirt_list =['down','down','down','down']
        self.move_list =['out','out','in','in']
        self.custbs_idx=[1,-1,1,-1]
        self.reb_list  =[0]*4
        self.strikes=[self.strike,self.strike,self.s_0,self.s_0]
        self.barriers=[self.barrier]*4
        self.ratio=[1,self.lev_day,1,1]
        
        
    def fpcall(self):
         
         self.opt_list  =['call','call','put']
         self.dirt_list =['up','up','up']
         self.move_list =['out','out','out']
         self.custbs_idx=[1,-1,-1]
         self.reb_list  =[self.reb/(self.fix_income/self.ramp),0,0]
         self.strikes   =[self.strike-self.ramp,self.strike,self.strike]
         self.barriers  =[self.barrier]*3
         self.ratio     =[self.fix_income/self.ramp,self.fix_income/self.ramp,self.lev_day]
        
    def fpput(self):
        self.opt_list  =['put','put','call']
        self.dirt_list =['down','down','down']
        self.move_list =['out','out','out']
        self.custbs_idx=[1,-1,-1]
        self.reb_list  =[self.reb/(self.fix_income/self.ramp),0,0]
        self.strikes   =[self.strike+self.ramp,self.strike,self.strike]
        self.barriers  =[self.barrier]*3
        self.ratio     =[self.fix_income/self.ramp,self.fix_income/self.ramp,self.lev_day]
 
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
        
        group_num=len(self.opt_list)
        # self.s=list(np.repeat(self.s,self.left_obs_days))
        
        self.sb=StandardBarrierArr(self.opt_list*self.left_obs_days
                                , self.dirt_list*self.left_obs_days
                                , self.move_list*self.left_obs_days
                                , [self.s]*group_num*self.left_obs_days
                                , self.strikes*self.left_obs_days #k
                                , self.barriers*self.left_obs_days#barrier
                                , self.reb_list*self.left_obs_days#rebate
                                , [self.dt]*group_num*self.left_obs_days# dt
                                # , (np.arange(self.ttm,0,-1)/BarrierAccOption.__annual_coeff).repeat(group_num) #t 
                                , (np.arange(self.ttm,0,-1)[:self.left_obs_days]/BarrierAccOption.__annual_coeff).repeat(group_num) #t 
                                ,[self.sigma]*group_num*self.left_obs_days #sigma
                                , self.ttm%1 # intra_to_next_obs
                                # , [self.trading_time.time()]*group_num*self.left_obs_days
                                # , [time(15,0,0)]*group_num*self.left_obs_days
                                , BarrierAccOption.__rf
                                , BarrierAccOption.__q)
        
        dfsb=pd.DataFrame([self.sb.price(),self.sb.delta(),self.sb.gamma(),self.sb.theta(1/BarrierAccOption.__annual_coeff),self.sb.vega()]).T
        dfsb.columns= ['pv','delta','gamma','theta','vega']
        # res=dfsb.multiply(np.array(list(np.array(self.custbs_idx)*-1*np.array([1,self.lev_day])*self.qty_freq)*self.left_obs_days),axis=0).sum()
        res=dfsb.multiply(np.array(list(np.array(self.custbs_idx)*self.cust_bs*-1*np.array(self.ratio)*self.qty_freq)*self.left_obs_days),axis=0).sum()

        return res
    
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
        self.exp_sb=StandardBarrierArr(self.exp_type
                                    , self.exp_dirt
                                    , self.exp_move
                                    , self.s
                                    , self.strike
                                    , self.barrier#barrier
                                    , 0        #rebate
                                    , self.dt# dt
                                    , (self.ttm)/BarrierAccOption.__annual_coeff #t 
                                    , self.sigma  #sigma
                                    , self.ttm%1
                                    , BarrierAccOption.__rf
                                    , BarrierAccOption.__q)
        dfsb=pd.DataFrame([self.exp_sb.price(),self.exp_sb.delta(),self.exp_sb.gamma(),self.exp_sb.theta(1/BarrierAccOption.__annual_coeff),self.exp_sb.vega()]).T
        dfsb.columns=['pv','delta','gamma','theta','vega']
        res=dfsb*self.lev_exp*self.qty_freq*self.exp_custbs_idx*self.cust_bs*-1
        return res.iloc[0]
        # return dfsb

    def getResult(self):
        if self.lev_exp>0:
            return self.dailyPart()+self.expPart() 
        else:
            return self.dailyPart()
        
if __name__=="__main__":
    import rqdatac as rqd
    rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
    trading_list=rqd.get_trading_dates('2023-01-01', '2025-12-31')
    #%%
if __name__=="__main__":
    bsb=BarrierAccOption(trading_list=trading_list
                     , opttype='fpcall'
                     , s=4723
                     , s_0=4723
                     , strike=4673
                     , barrier=4773
                     , sigma=0.10
                     , fix_income=120
                     , reb=50
                     , dt=1/annual_coeff
                     , lev_day=2
                     , cust_bs="B"
                     , qty_freq=50
                     , lev_exp=1
                     , next_obs_date=datetime(2024,7,4).date()
                     , end_obs_date=datetime(2024,8,5).date()
                     , trading_time=datetime.now())
    bsb.getResult()
    
    
    #%%
if __name__=="__main__":
    s_0=1
    opttype='accput'
    strike=1.03
    barrier=0.97
    s_ts=(np.arange(0.8,1.2,0.001)*s_0).round(3)
    sigma=0.3
    reb=abs(barrier-s_0)
    fix_income=0
    dt=1/annual_coeff
    qty_freq=1
    lev_day=2
    lev_exp=0
    # first_obs_date=datetime(2023,12,19).date()
    next_obs_date=datetime(2024,8,27).date()
    end_obs_date=datetime(2024,9,9).date()    
    
    trading_time=datetime(2024,8,26,17,30,0)
    bsb=BarrierAccOption(trading_list
                     , opttype
                     , s_0
                     , s_0
                     , strike
                     , barrier
                     , sigma
                     , fix_income
                     , reb
                     , dt
                     , lev_day
                     , "B"
                     , qty_freq
                     , lev_exp
                     , next_obs_date, end_obs_date, trading_time)
    bsb.getResult()
    
    
    
    # total_obs_days=trading_list.index(last_obs_date)-trading_list.index(first_obs_date)+1
    # # left_obs_days=trading_list.index(last_obs_date)-trading_list.index(next_obs_date)+1
 
    # bsb=BarrierAccOptionSelectItems(['delta','gamma']
    #                                 ,trading_list,20
    #                                 ,opttype,s_ts,s_0,strike,barrier
    #                                 ,sigma,fix_income,reb,dt
    #                                 ,lev_day,qty_freq,lev_exp
    #                                 ,next_obs_date,end_obs_date
    #                                 ,datetime.now())      
    # bsb.dailyPart()
    #%%


    
    #%%
if __name__=="__main__":   
    opt_type='put'
    drt_type='down'
    move_type='out'
    s=4700
    k=4700
    barrier=s-500
    sigma=0.25
    reb=0
    dt=1/annual_coeff
    expire_date=datetime(2024,4,15).date()
    sb=StandardBarrierArr(opt_type
                            , drt_type
                            , move_type
                            , s
                            , k #k
                            , barrier #barrier
                            , reb #rebate
                            , dt # dt
                            , calTradttm(trading_list, datetime.now(), expire_date)/annual_coeff #t 
                            , sigma   #sigma
                            , 0# intra_to_next_obs
                            , 0.03, 0)
    print(sb.price())
    
    
    
    #%%
if __name__=="__main__":   
    
    
    s_ts=np.arange(50,150,0.2).round(1)/100*3800
    # for s in range(80,145,5):
    opt_type='put'
    drt_type='down'
    move_type='out'
    k=3800
    barrier=3800-380
    dt=1/annual_coeff
    reb=0
    t=20/annual_coeff
    sigma=0.15
    sb=StandardBarrierArr([opt_type]*s_ts.shape[0]
                            , [drt_type]*s_ts.shape[0]
                            , [move_type]*s_ts.shape[0]
                            , s_ts
                            , [k]*s_ts.shape[0] #k
                            , [barrier]*s_ts.shape[0]#barrier
                            , [reb]*s_ts.shape[0]#rebate
                            , [dt]*s_ts.shape[0]# dt
                            ,[t]*s_ts.shape[0] #t 
                            ,[sigma]*s_ts.shape[0]    #sigma
                            ,0# intra_to_next_obs
                            ,0.03, 0)
    # s_ts2=np.arange(1,103,0.02).round(2)
    # sb=StandardBarrierArr([opt_type]*s_ts2.shape[0]
    #                         , [drt_type]*s_ts2.shape[0]
    #                         , [move_type]*s_ts2.shape[0]
    #                         , s_ts2
    #                         , [k]*s_ts2.shape[0] #k
    #                         , [barrier]*s_ts2.shape[0]#barrier
    #                         , [reb]*s_ts2.shape[0]#rebate
    #                         , [dt]*s_ts2.shape[0]# dt
    #                         ,[0.25]*s_ts2.shape[0] #t 
    #                         ,[sigma]*s_ts2.shape[0]    #sigma
    #                         ,0.25# intra_to_next_obs
    #                         , 0.03, 0)
    # dfsb_in=pd.DataFrame([sb.price(),sb.delta(),sb.gamma(),sb.theta(1/annual_coeff),sb.vega()]).T
    # dfsb_out=pd.DataFrame([sb.price(),sb.delta(),sb.gamma(),sb.theta(1/annual_coeff),sb.vega()]).T
    # dfsb_in.index=s_ts
    # dfsb_out.index=s_ts
    dfsb=pd.DataFrame([sb.price(),sb.delta(),sb.gamma(),sb.theta(1/annual_coeff),sb.vega(),sb.vomma()]).T
    dfsb.columns= ['pv','delta','gamma','theta','vega','vomma']
    dfsb.index=s_ts
    # df=pd.DataFrame(index=s_ts,columns=['pv','delta','gamma','theta','vega'])
    # for s in s_ts:
    #     bsm=BSM(s,k,t,0.03,0,sigma,opt_type[0].upper())
    #     df.loc[s,'pv']=bsm.price()
    #     df.loc[s,'delta']=bsm.delta()
    #     df.loc[s,'gamma']=bsm.gamma()
    #     df.loc[s,'theta']=bsm.theta(1/annual_coeff)
    #     df.loc[s,'vega']=bsm.vega()/100
    # df=df.astype(float)
    
    fig=plt.figure(figsize=(15,4))
    fig.suptitle(opt_type+' '+drt_type+' '+move_type+" K:"+str(k)+" B:"+str(barrier))
    for c in dfsb.columns:
        i=dfsb.columns.tolist().index(c)+1
        ax=fig.add_subplot(1,6,i)
        ax.plot(dfsb[c],label='barrier',color='r')
        # ax.plot(sb.pv,label='pv',color='k')
        # ax.plot(df[c],label='euro',linestyle='--')
        # ax.vlines(k,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='y',label='K')
        # ax.vlines(barrier,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='b',label='Barrier')
        plt.legend()
        ax.set_title(c)
    plt.tight_layout()
    
    
    sigma_ts=np.arange(0.02,0.5,0.02)
    s=100
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for t in [200/annual_coeff,60/annual_coeff,5/annual_coeff]:
        sb=StandardBarrierArr([opt_type]*sigma_ts.shape[0]
                                , [drt_type]*sigma_ts.shape[0]
                                , [move_type]*sigma_ts.shape[0]
                                , [s]*sigma_ts.shape[0]
                                , [k]*sigma_ts.shape[0] #k
                                , [barrier]*sigma_ts.shape[0]#barrier
                                , [reb]*sigma_ts.shape[0]#rebate
                                , [dt]*sigma_ts.shape[0]# dt
                                , [t]*sigma_ts.shape[0] #t 
                                , sigma_ts   #sigma
                                , 0.05# intra_to_next_obs
                                , 0.03, 0)
        dfsb=pd.DataFrame([sb.price(),sb.delta(),sb.gamma(),sb.theta(1/annual_coeff),sb.vega()]).T
        dfsb.columns= ['pv','delta','gamma','theta','vega']
        dfsb.index=sigma_ts
        
        ax.plot(dfsb['vega'],label=str(t*annual_coeff)+' days')
        plt.legend()
    ax.set_title('vega')
    plt.tight_layout()

    
    


    #%%
if __name__=="__main__":
    opt_type='put'
    drt_type='down'
    move_type='out'
    s=4700
    k=4700
    barrier=4200
    dt=1/annual_coeff
    reb=0
    t=20/annual_coeff
    sigma=0.225
    sig_ts=np.arange(1,500,2).round(1)/100*sigma
    sb=StandardBarrierArr([opt_type]*sig_ts.shape[0]
                            , [drt_type]*sig_ts.shape[0]
                            , [move_type]*sig_ts.shape[0]
                            , [s]*sig_ts.shape[0]
                            , [k]*sig_ts.shape[0] #k
                            , [barrier]*sig_ts.shape[0]#barrier
                            , [reb]*sig_ts.shape[0]#rebate
                            , [dt]*sig_ts.shape[0]# dt
                            ,[t]*sig_ts.shape[0] #t 
                            ,sig_ts   #sigma
                            ,0# intra_to_next_obs
                            ,0.03, 0)
    dfsb=pd.DataFrame([sb.price(),sb.delta(),sb.gamma(),sb.theta(1/annual_coeff),sb.vega(),sb.vomma()]).T
    dfsb.columns= ['pv','delta','gamma','theta','vega','vomma']
    dfsb.index=sig_ts
    
    fig=plt.figure(figsize=(15,4))
    fig.suptitle(opt_type+' '+drt_type+' '+move_type+" K:"+str(k)+" B:"+str(barrier))
    for c in dfsb.columns:
        i=dfsb.columns.tolist().index(c)+1
        ax=fig.add_subplot(1,6,i)
        ax.plot(dfsb[c],label='barrier',color='r')
        # ax.plot(sb.pv,label='pv',color='k')
        # ax.plot(df[c],label='euro',linestyle='--')
        # ax.vlines(k,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='y',label='K')
        # ax.vlines(barrier,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='b',label='Barrier')
        plt.legend()
        ax.set_title(c)
    plt.tight_layout()


    
    #%%
if __name__=="__main__":
    # import rqdatac as rqd
    # rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
    # trading_list=rqd.get_trading_dates('2023-01-01', '2024-12-31')
    s_0=7380
    opttype='acccall_forward'
    strike=s_0-135
    barrier=s_0+135
    s_ts=(np.arange(98,102,0.02)/100*s_0).round(3)
    sigma=0.125
    reb=abs(barrier-s_0)
    fix_income=0
    dt=1/annual_coeff
    qty_freq=5
    lev_day=1
    lev_exp=65
    # first_obs_date=datetime(2023,12,19).date()
    next_obs_date=datetime(2024,1,9).date()
    end_obs_date=datetime(2024,4,17).date()    
    # total_obs_days=trading_list.index(last_obs_date)-trading_list.index(first_obs_date)+1
    # left_obs_days=trading_list.index(last_obs_date)-trading_list.index(next_obs_date)+1
 
    bsb=BarrierAccOption(trading_list,opttype,s_0,s_0,strike,barrier
                         ,sigma,fix_income,reb,dt
                         ,lev_day,qty_freq,lev_exp
                         ,next_obs_date,end_obs_date
                         ,datetime.now())      
    bsb.getResult()
    
    df_dic={datetime(2024,1,8).date():datetime(2024,1,3,14,55,0)
            ,datetime(2024,3,8).date():datetime(2024,1,4,14,55,0)
            ,datetime(2024,4,17).date():datetime(2024,1,3,14,50,0)
            }

    df_res=dict()
    for next_obs_date in df_dic.keys():
        df=pd.DataFrame(columns=s_ts)
        for s in s_ts:
            bsb=BarrierAccOption(trading_list,opttype,s,s_0,strike,barrier
                                  ,sigma,fix_income,reb,dt
                                  ,lev_day,qty_freq,lev_exp
                                  ,next_obs_date,end_obs_date
                                  ,df_dic[next_obs_date])      
            df[s]=bsb.getResult()
        df_dic[next_obs_date]=df.T    
    
    # df_dic={datetime(2024,1,3).date():datetime(2024,1,3,14,30,0)
    #         ,datetime(2024,2,2).date():datetime(2024,2,2,14,30,0)
    #         ,datetime(2024,3,4).date():datetime(2024,3,4,14,30,0)
    #         ,datetime(2024,4,15).date():datetime(2024,4,15,14,30,0)
    #         }
    # for next_obs_date in df_dic.keys():
    #     trading_time=df_dic[next_obs_date]
    #     df=pd.DataFrame(columns=s_ts)
    #     for s in s_ts:
    #         bsb=BarrierAccOption(trading_list,opttype,s,s_0,strike,barrier
    #                               ,sigma,fix_income,reb,dt
    #                               ,lev_day,qty_freq,lev_exp
    #                               ,next_obs_date,end_obs_date
    #                               ,trading_time)      
    #         df[s]=bsb.getResult()
    #     df_dic[next_obs_date]=df.T   
        
    # fig=plt.figure(figsize=(15,4))
    # # fig.suptitle(opt_type+' '+drt_type+' '+move_type+" K:"+str(k)+" B:"+str(barrier))
    # i=0
    # for c in df.columns:
    #     i+=1
    #     ax=fig.add_subplot(1,5,i)
    #     ax.plot(df[c],label=c,color='r')
    #     # ax.vlines(k,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='y',label='K')
    #     # ax.vlines(barrier,min(dfsb[c].min(),df[c].min()),max(dfsb[c].max(),df[c].max()),linewidth=1,linestyle='--',color='b',label='Barrier')
    #     plt.legend()
    #     ax.set_title(c)
    # plt.tight_layout()
    
    fig=plt.figure(figsize=(15,4))   
    i=0
    for c in ['pv','delta','gamma','theta','vega']:
        i+=1
        ax=fig.add_subplot(1,5,i)
        for k in df_dic.keys():
            ax.plot(df_dic[k][c],label=str(k))
            # ax.plot(dic_acc[k].loc[c,:],label='facc',color='b')
        if c=='delta':
            max_g=df_dic[k][c].max()*0.5
            min_g=df_dic[k][c].min()*1
            ax.set_ylim(min_g,max_g)
        if c=='gamma' :
            max_g=df_dic[k][c].max()*0.01
            min_g=df_dic[k][c].min()*0.01
            ax.set_ylim(min_g,max_g)
        ax.vlines(strike,df_dic[k][c].min(),df_dic[k][c].max(),linewidth=1,linestyle='--',color='y',label='K')
        ax.vlines(barrier,df_dic[k][c].min(),df_dic[k][c].max(),linewidth=1,linestyle='--',color='b',label='Barrier')
        plt.legend()
        ax.set_title(c)
    plt.tight_layout()
 
    #%%
if __name__=="__main__":
    strike=2305
    barrier=2455
    s_0=2387
    reb=0
    dt=1/annual_coeff
    s_ts=(np.arange(80,120,1)/100*s_0).round(0)
    sigma=0.20
    fixed_income=10
    end_obs_date=datetime(2024,1,17).date()   
    # trading_time=datetime.now()
    qty_freq=1
    lev_day,leverage=2,2
    lev_exp=20
    delta_strike=2
    delta_barrier=2
    dic_acc={datetime(2023,12,20).date():datetime(2023,12,20,9,50,0)
            ,datetime(2024,1,3).date():datetime(2024,1,3,9,50,0)
            ,datetime(2024,1,16).date():datetime(2024,1,16,9,50,0)
            }
    # total_obs_days=st.total_tradingdates.index(endobs_date)-st.total_tradingdates.index(firstobs_date)+1
    for startobs_date in dic_acc.keys():
        trading_time=dic_acc[startobs_date]
        df=pd.DataFrame(index=['delta','gamma','theta','vega'],columns=s_ts)
        # left_obs_days=st.total_tradingdates.index(endobs_date)-st.total_tradingdates.index(startobs_date)+1
        for s in s_ts:
            acc=AccOption(trading_list,'ACCCALL','EB',s, sigma, strike, barrier, fixed_income
                         , startobs_date,end_obs_date, trading_time, qty_freq,"B", leverage, delta_strike, delta_barrier)
            # acc2=AccOptionArr(st.total_tradingdates,'ACCCALL',varity,s, sigma, s_0, 0, 0
            #              , startobs_date, endobs_date, trading_time, qty_freq,"B", 0, delta_strike, delta_barrier)
            bsb=BarrierAccOption(trading_list,'acccall',s,strike,barrier          
                                 ,sigma,reb,dt
                                 ,lev_day,qty_freq,lev_exp
                                 ,startobs_date,end_obs_date
                                 ,trading_time)      
            df[s]=acc.getResult()[df.index.tolist()]+sb.getResult()[df.index.tolist()]
        dic_acc[startobs_date]=df


    fig=plt.figure(figsize=(15,4))
    # fig.suptitle(str(ttm)+" Days Left")
    i=0
    for c in ['delta','gamma','theta','vega']:
        i+=1
        ax=fig.add_subplot(1,4,i)
        for k in dic_acc.keys():
            ttm=(endobs_date-k).days
            ax.plot(dic_acc[k].loc[c,:],label=str(ttm)+" Days Left")
        # ax.plot(dic_facc[k].loc[c,:],label='facc',color='b')
        ax.vlines(strike,dic_acc[k].loc[c,:].min(),dic_acc[k].loc[c,:].max(),linewidth=1,linestyle='--',color='y',label='K')
        ax.vlines(barrier,dic_acc[k].loc[c,:].min(),dic_acc[k].loc[c,:].max(),linewidth=1,linestyle='--',color='b',label='Barrier')
        plt.legend()
        ax.set_title(c)
    plt.tight_layout()
        
