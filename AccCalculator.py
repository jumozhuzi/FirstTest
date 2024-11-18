# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:24:46 2024

@author: dzrh
"""

import xlwings as xw
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# import tushare as tu
from datetime import timedelta,datetime,time
# import time
# import iFinDPy as fd
import os
from CYL.OptionPricing import BSM,calIV,calTradttm,LinearInterpVol
# from CYL.YieldChainAPI import YieldChainAPI 
from CYL.OTCAPI import SelfAPI
import streamlit as stm
from CYL.StressTestNew import StressTestNew,BarrierContracts
from CYL.pythonAPI_pyfunctions4newDll_3 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer,jsonvolSurface2cstructure_selfapi
# from CYL.pythonAPI_pyfunctions4newDll_3_2 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer,jsonvolSurface2cstructure_selfapi

from CYL.StressTestNew import LinearInterpVol 
import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
import json
from ast import literal_eval
    
def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code


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
            
def getpdobList(underlyingCode,trade_date,first_obsdate,expired_date,pricing_time):
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
    pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]
    return pdobList


# user='chengyl'
# passwd='CYLcyl0208@'
# YL=YieldChainAPI(user,passwd)
api=SelfAPI()
def getflag_reverse(flag):
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
        return 'bacccall'
    elif flag=="熔断累沽":
        return 'baccput'
    elif flag=="熔断增强累购":
        return 'bacccallplus'
    elif flag=="熔断增强累沽":
        return 'baccputplus'
    elif flag=="熔断固陪累购":
        return 'bfpcall'
    elif flag=="熔断固陪累沽":
        return 'bfpput'
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
    if flag=='bacccall':
        return "熔断累购"
    elif flag=='baccput':
        return "熔断累沽"
    elif flag=='bacccallplus':
        return "熔断增强累购"
    elif flag=='baccputplus':
        return "熔断增强累沽"
    elif flag=='bfpcall':
        return "熔断固陪累购"
    elif flag=='bfpput':
        return "熔断固陪累沽"
    elif flag=='acccall':
        return "累购"
    elif flag=='accput':
        return "累沽"
    elif flag=='fpcall':
        return "固定赔付累购"
    elif flag=='fpput':
        return "固定赔付累沽"
    else:
        print("Wrong option type!")
        
def getVolsurfacejson(trade_date,underlyingCode):
  
    vfe=api.getVol_json(str(trade_date),underlyingCode)['mid']
    vfe=json.dumps(literal_eval(str(vfe)))
    return jsonvolSurface2cstructure_selfapi(vfe)
    
class CalAcc():
    # def __init__(self,ask_Interval,ask_Coupon):
    #     super().__init__(ask_Interval,ask_Coupon)
 
    # trade_date="2024/03/26"
    def __init__(self,underlyingCode,flag, direction
                    , s_t,s_0,strike,barrier,rng_k,rng_b
                    , rebate,coupon
                    , pricing_time, trade_date,first_obsdate,expired_date,obs_days
                    , daily_amt, leverage, leverage_expire
                    , isCashsettle,strike_ramp, barrier_ramp
                    ,  const_sgm,mult_tag,rf=0.03):
           
           self.underlyingCode=underlyingCode
           self.flag=flag
           self.direction=-1 if direction=="B" else 1
           self.s_t=findLatestPrice(getRQcode(self.underlyingCode)) if s_t=="" else s_t
           self.s_0=self.s_t if s_0=="" else s_0
           self.rng_k=0 if rng_k=="" else rng_k
           self.rng_b=0 if rng_b=="" else rng_b
           self.strike=strike # could be ""
           # if barrier=="NO":
               # self.barrier=100000000
           self.barrier=100000000 if barrier=="NO" else barrier
           # self.barrier=self.barrier if 'call' in self.flag else 0
           self.rebate=rebate #coulde be ""
           self.coupon=0  if coupon=="" else coupon
           self.pricing_time=pricing_time
           self.trade_date=pd.to_datetime(trade_date).date()
           self.first_obsdate=self.trade_date if first_obsdate=="" else pd.to_datetime(first_obsdate).date()
           self.obs_days=obs_days
           # self.expired_date=rqd.get_next_trading_date(self.first_obsdate,self.obs_days-1) if expired_date=="" else pd.to_datetime(expired_date).date()
           self.expired_date=rqd.get_next_trading_date(self.first_obsdate,int(self.obs_days)-1) if expired_date=="" else pd.to_datetime(expired_date).date()
           if self.expired_date not in rqd.get_trading_dates(self.first_obsdate,"2026-01-01"):
               self.expired_date=rqd.get_previous_trading_date(self.expired_date,1)
           self.obs_days=len(rqd.get_trading_dates(self.first_obsdate,self.expired_date))
            
           self.daily_amt=1 if daily_amt=="" else daily_amt
           self.leverage=leverage
           self.leverage_expire=leverage_expire
           self.isCashsettle=isCashsettle
           self.strike_ramp=strike_ramp
           self.barrier_ramp=barrier_ramp
           self.const_sgm=const_sgm
           self.rf=rf
           # self.LiV=LinearInterpVol(self.underlyingCode, self.trade_date)
           self.pdobList=getpdobList(getRQcode(self.underlyingCode), self.trade_date, self.first_obsdate, self.expired_date, self.pricing_time)
            # LiV=LinearInterpVol(underlyingCode,trade_date)
           self.obs_days=self.pdobList.shape[0] if self.obs_days=="" else int(self.obs_days)
           self.__getVolsurfacejson()
           self.credit=0
           self.tick_size=rqd.instruments(getRQcode(self.underlyingCode)).tick_size()
           self.times=0
           self.pv=0
           self.inital_ratio=0.00001
           self.mult_tag=mult_tag
           
    # def __getVolsurfacejson(self):
  
    #       vfe=YL.get_surface_json([self.underlyingCode],self.trade_date)['data'][0]['VolTable']
    #       vfe=json.dumps(literal_eval(str(vfe)))
    #       vfe=vfe.replace("S","s").replace("E","e").replace("V","v").replace("D","")
    #       self.cSV = jsonvolSurface2cstructure(vfe)
    
    def __getVolsurfacejson(self):
  
          vfe=api.getVol_json(str(self.trade_date), self.underlyingCode)['mid']
          vfe=json.dumps(literal_eval(str(vfe)))
          self.cSV = jsonvolSurface2cstructure_selfapi(vfe)
          
    
    def getRes(self,flag_Inter,flag_Coupon,mult):
        if flag_Inter==1:
            self.__calInter_BarAcc(mult) if 'b' in self.flag else self.__calInter_Acc(mult)
        elif flag_Coupon==1:
            self.__calCoupon_BarAcc(mult) if 'b' in self.flag else self.__calCoupon_Acc(mult)
        else:
            print("Wrong Flag!")
            
        df=pd.DataFrame(columns=['要素'])
        df.loc['交易日期']=self.trade_date    
        df.loc['到期日期']=self.expired_date     
        df.loc['标的合约']=self.underlyingCode
        df.loc['入场价格']=self.s_0
        df.loc['期权类型']=getflag(self.flag)
        df.loc['行权价格']=str(round(self.strike))+" ("+str(round(self.strike)-self.s_0)+")"
        # df.loc['行权价格']=str(self.strike)
        df.loc['敲出价格']="不敲出" if (self.barrier==100000000 or self.barrier==0) else str(round(self.barrier))+" ("+str(round(self.barrier)-self.s_0)+")"
        # df.loc['敲出价格']="不敲出" if self.barrier==100000000 else str(self.barrier)

        if 'plus' in self.flag:
            df.loc['敲出赔付']="收盘价-入场价"
        else:
            df.loc['敲出赔付']="无" if self.rebate=="" else round(self.rebate)
        df.loc['区间赔付']="线性" if self.coupon==0 else int(self.coupon)
        # if self.coupon==0 and 'put' in self.flag:
        #     df.loc['区间内赔付（每日)']=str(self.strike)+"-收盘价"
        # elif self.coupon==0 and 'call' in self.flag:
        #     df.loc['区间内赔付（每日)']="收盘价-"+str(self.strike)
        # else:
        #     df.loc['区间内赔付（每日)']=self.coupon
        df.loc['起始观察日']=self.first_obsdate
        
        df.loc['观察次数']=self.obs_days
        df.loc['杠杆系数']=self.leverage
        df.loc['期末额外倍数']=self.leverage_expire
        df.loc['期权价格']=0
        df.loc['预估保证金(1单位/天)']=round(self.getmargin(),2)
        # df.loc['Credit']=0 if self.credit==0 else 1
        df.loc['OptPrice']=round(self.pv*-1,2)
        df.index.name='要素'
        print("Done")
        return df.astype(str)
  
    
    def getmargin(self):
        
        snp=rqd.current_snapshot(getRQcode(self.underlyingCode))
        self.ratio=round(snp.limit_up/snp.prev_settlement-1,2)
        if 'call' in self.flag:
            self.ratio=-1*self.ratio
        elif 'put' in self.flag:
            self.ratio=1*self.ratio
        else:
            pass
        
        if 'b' in self.flag:
            return pyAIKOAccumulatorPricer(self.flag[1:]
                                        ,self.direction
                                        ,self.s_0*(1+self.ratio)
                                        ,self.strike
                                        , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                        , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                        ,self.s_0
                                        ,self.daily_amt
                                        ,self.isCashsettle
                                        ,self.leverage
                                        ,self.leverage_expire
                                        ,self.coupon
                                        ,self.barrier
                                        ,self.rebate
                                        ,self.pdobList
                                        ,self.rf,self.rf
                                        ,0
                                        ,self.pdobList.shape[0]
                                        ,self.cSV)[0]*1.1
        else:
            return pyAIAccumulatorPricer(self.flag
                                       , self.direction
                                       , self.s_0*(1+self.ratio)
                                       , self.strike
                                       , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                       , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                       , self.daily_amt
                                       , self.isCashsettle
                                       , self.leverage
                                       , self.coupon
                                       , self.barrier
                                       , self.strike_ramp
                                       , self.barrier_ramp
                                       , self.pdobList
                                       , self.rf
                                       , 0
                                       , self.pdobList.shape[0]
                                       , self.cSV
                                       , 0
                                       )[0]*1.1
    
    def __getInitialRng(self):
        
        if self.rng_k==0 and self.rng_b!=0: 
            rng_k=self.s_t*self.inital_ratio
            rng_b=self.rng_b
        elif self.rng_k!=0 and self.rng_b==0:
            rng_k=self.rng_k
            rng_b=self.s_t*self.inital_ratio
        else:
            rng_k,rng_b=self.s_t*self.inital_ratio,self.s_t*self.inital_ratio
        return round(rng_k,5),round(rng_b,5)
    
    def __getInitialCoupon(self):
        self.coupon=round(self.s_t*self.inital_ratio,2)
    
    # def __getcondition(self,price,theta,mult):
    #     if price<0:
    #         self.pv=price
    #         print("Credit!!")
    #     # return abs(price)/abs(theta)>mult and price>=0
    #     return abs(price)/abs(theta)>mult
    def __getcondition(self,price,theta,mult,barrier,strike):
        if self.mult_tag:
            if price<0:
                self.pv=price
                print("Credit!!")
                return False
            else:
                # return True
                if 'call' in self.flag:
                    # return barrier>self.s_t and strike<self.s_t and abs(price)/abs(theta)>mult
                    return barrier>self.s_t and strike<self.s_t and abs(abs(price)/abs(theta)-mult)>0.2
        
                elif 'put' in self.flag:
                    # return barrier<self.s_t and strike>self.s_t and abs(price)/abs(theta)>mult
                    return barrier<self.s_t and strike>self.s_t and abs(abs(price)/abs(theta)-mult)>0.2
        
                else:
                    return abs(abs(price)/abs(theta)-mult)>0.5
        else:
            if price<0:
                self.pv=price
                print("Credit!!")
                return False
            else:
                return True

    # def __getcondition(self,price,theta,mult):
    #     if price<0:
    #         self.pv=price
    #         print("Credit!!")
    #     return price/theta>mult and price/theta>0
            # return True
        # return abs(price)/abs(theta)>mult and price>=0
        # return abs(abs(price)/abs(theta)-mult)>0.5
    
    # def __getcondition(self,price,theta,mult):
    #     self.times=self.times+1
    #     if price<0:
    #         print("Credit!!")
    #         self.credit=1
    #     if self.times>100:
    #         return False
    #     else:
    #         return abs(abs(price/theta)-mult)>0.1
    
    def __getK_B(self,rng_k,rng_b):
        if 'call' in self.flag:
            strike=self.s_t-rng_k if self.strike=="" else self.strike
            barrier=self.s_t+rng_b if self.barrier=="" else self.barrier
        elif 'put' in self.flag:
            strike=self.s_t+rng_k if self.strike=="" else self.strike
            barrier=self.s_t-rng_b if self.barrier=="" else self.barrier
        else:
            pass
        return strike,barrier
    
    def __calInter_Acc(self,mult):
       
   
        rng_k,rng_b=self.__getInitialRng()
        # strike,barrier=self.__getK_B(rng_k,rng_b) 
        strike,barrier=self.__getK_B(rng_k,rng_b) 
        res=pyAIAccumulatorPricer(self.flag
                                   , self.direction
                                   , self.s_t
                                   , strike
                                   , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                   , self.daily_amt
                                   , self.isCashsettle
                                   , self.leverage
                                   , self.coupon
                                   , barrier
                                   , self.strike_ramp
                                   , self.barrier_ramp
                                   , self.pdobList
                                   , self.rf
                                   , self.const_sgm
                                   , self.pdobList.shape[0]
                                   , self.cSV
                                   , 0
                                   )
        if type(res)==bytes:
            return print(res)
        else:
            price,theta=res[0],res[4]
            while self.__getcondition(price,theta,mult,barrier,strike):
     
                if price>0:
                     rng_k=rng_k+1 if self.rng_k==0 else  self.rng_k
                     rng_b=rng_b+1 if self.rng_b==0 else  self.rng_b
                else:
                     rng_k=rng_k-1 if self.rng_k==0 else  self.rng_k
                     rng_b=rng_b-1 if self.rng_b==0 else  self.rng_b
                    
                strike,barrier=self.__getK_B(rng_k,rng_b)
                # res=pyAIAccumulatorPricer(self.flag
                #                             , self.direction
                #                             , self.s_t
                #                             , strike
                #                             , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                #                             , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                #                             , self.daily_amt
                #                             , self.isCashsettle
                #                             , self.leverage
                #                             , self.coupon#coupon
                #                             , barrier
                #                             , self.strike_ramp
                #                             , self.barrier_ramp
                #                             , self.pdobList
                #                             , self.rf
                #                             , self.const_sgm
                #                             , self.pdobList.shape[0]
                #                             , self.cSV
                #                             , 0
                #                             )
                price=pyAIAccumulatorPricer(self.flag
                                           , self.direction
                                           , self.s_t
                                           , strike
                                           , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                           , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                           , self.daily_amt
                                           , self.isCashsettle
                                           , self.leverage
                                           , self.coupon#coupon
                                           , barrier
                                           , self.strike_ramp
                                           , self.barrier_ramp
                                           , self.pdobList
                                           , self.rf
                                           , self.const_sgm
                                           , self.pdobList.shape[0]
                                           , self.cSV
                                           , 0
                                           )[0]

                theta=pyAIAccumulatorPricer(self.flag
                                           , self.direction
                                           , self.s_t
                                           , strike
                                           , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                           , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                           , self.daily_amt
                                           , self.isCashsettle
                                           , self.leverage
                                           , self.coupon#coupon
                                           , barrier
                                           , self.strike_ramp
                                           , self.barrier_ramp
                                           , self.pdobList
                                           , self.rf
                                           # , self.const_sgm
                                           ,0
                                           , self.pdobList.shape[0]
                                           , self.cSV
                                           , 0
                                           )[4]

                # price,theta=res[0],res[4]
            self.strike,self.barrier=strike,barrier
            
            return res
        
        
            
    def __calInter_BarAcc(self,mult):
        rng_k,rng_b=self.__getInitialRng()
        # strike,barrier=getK_B(self.flag[2:], self.s_t, rng_k,rng_b)
        strike,barrier=self.__getK_B(rng_k,rng_b)
        rebate=abs(self.s_0-barrier) if self.rebate=="" else self.rebate
        res=pyAIKOAccumulatorPricer(self.flag[1:]
                                    ,self.direction
                                    ,self.s_t
                                    , strike
                                    , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                    , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                    ,self.s_0
                                    ,self.daily_amt
                                    ,self.isCashsettle
                                    ,self.leverage
                                    ,self.leverage_expire
                                    ,self.coupon
                                    ,barrier
                                    ,rebate
                                    ,self.pdobList
                                    ,self.rf,self.rf
                                    ,self.const_sgm
                                    ,self.pdobList.shape[0]
                                    ,self.cSV)
        if type(res)==bytes:
            return print(res)
        else:
            price,theta=res[0],res[4]
            while self.__getcondition(price,theta,mult,barrier,strike):
                if price>0:
                     rng_k=rng_k+1 if self.rng_k==0 else  self.rng_k
                     rng_b=rng_b+1 if self.rng_b==0 else  self.rng_b
                else:
                     rng_k=rng_k-1 if self.rng_k==0 else  self.rng_k
                     rng_b=rng_b-1 if self.rng_b==0 else  self.rng_b
                    
                # strike,barrier=getK_B(self.flag, self.s_t, rng_k,rng_b)
                strike,barrier=self.__getK_B(rng_k,rng_b)
                rebate=abs(self.s_0-barrier) if self.rebate=="" else self.rebate
                # res=pyAIKOAccumulatorPricer(self.flag[1:]
                #                             ,self.direction
                #                             ,self.s_t
                #                             , strike
                #                             , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                #                             , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                #                             ,self.s_0
                #                             ,self.daily_amt
                #                             ,self.isCashsettle
                #                             ,self.leverage
                #                             ,self.leverage_expire
                #                             ,self.coupon
                #                             ,barrier
                #                             ,rebate
                #                             ,self.pdobList
                #                             ,self.rf,self.rf
                #                             ,self.const_sgm
                #                             ,self.pdobList.shape[0]
                #                             ,self.cSV)
                # price,theta=res[0],res[4]
                price=pyAIKOAccumulatorPricer(self.flag[1:]
                                            ,self.direction
                                            ,self.s_t
                                            , strike
                                            , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                            , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                            ,self.s_0
                                            ,self.daily_amt
                                            ,self.isCashsettle
                                            ,self.leverage
                                            ,self.leverage_expire
                                            ,self.coupon
                                            ,barrier
                                            ,rebate
                                            ,self.pdobList
                                            ,self.rf,self.rf
                                            ,self.const_sgm
                                            ,self.pdobList.shape[0]
                                            ,self.cSV)[0]
                
                theta=pyAIKOAccumulatorPricer(self.flag[1:]
                                            ,self.direction
                                            ,self.s_t
                                            , strike
                                            , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                            , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                            ,self.s_0
                                            ,self.daily_amt
                                            ,self.isCashsettle
                                            ,self.leverage
                                            ,self.leverage_expire
                                            ,self.coupon
                                            ,barrier
                                            ,rebate
                                            ,self.pdobList
                                            ,self.rf,self.rf
                                            # ,self.const_sgm
                                            ,0
                                            ,self.pdobList.shape[0]
                                            ,self.cSV)[4]
            self.strike,self.barrier,self.rebate=strike,barrier,rebate
            return res

    def __calCoupon_Acc(self,mult):
        self.__getInitialCoupon()
        if self.strike=="" and self.barrier=="":
             self.strike,self.barrier=self.__getK_B(self.rng_k,self.rng_b)
        elif self.strike=="" and self.barrier!="":
            self.strike=self.__getK_B(self.rng_k, self.rng_b)[0]
        elif self.strike!="" and self.barrier=="":
            self.barrier=self.__getK_B(self.rng_k, self.rng_b)[-1]
        elif self.strike!="" and self.barrier!="":
              pass
        else:
              print("Wrong Input Interval!")
        res=pyAIAccumulatorPricer(self.flag
                                   , self.direction
                                   , self.s_t
                                   , self.strike
                                   , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                   , self.daily_amt
                                   , self.isCashsettle
                                   , self.leverage
                                   , self.coupon
                                   , self.barrier
                                   , self.strike_ramp
                                   , self.barrier_ramp
                                   , self.pdobList
                                   , self.rf
                                   , self.const_sgm
                                   , self.pdobList.shape[0]
                                   , self.cSV
                                   , 0
                                   )
        if type(res)==bytes:
            return print(res)
        else:
            price,theta=res[0],res[4]
            while self.__getcondition(price,theta,mult,self.barrier,self.strike) and self.coupon>0:
                if price>0:
                    self.coupon+=1 
                else:
                    self.coupon-=1
                # res=pyAIAccumulatorPricer(self.flag
                #                            , self.direction
                #                            , self.s_t
                #                            , self.strike
                #                            , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                #                            , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                #                            , self.daily_amt
                #                            , self.isCashsettle
                #                            , self.leverage
                #                            , self.coupon#coupon
                #                            , self.barrier
                #                            , self.strike_ramp
                #                            , self.barrier_ramp
                #                            , self.pdobList
                #                            , self.rf
                #                            , self.const_sgm
                #                            , self.pdobList.shape[0]
                #                            , self.cSV
                #                            , 0
                #                            )
                # price,theta=res[0],res[4]
                price=pyAIAccumulatorPricer(self.flag
                                           , self.direction
                                           , self.s_t
                                           , self.strike
                                           , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                           , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                           , self.daily_amt
                                           , self.isCashsettle
                                           , self.leverage
                                           , self.coupon#coupon
                                           , self.barrier
                                           , self.strike_ramp
                                           , self.barrier_ramp
                                           , self.pdobList
                                           , self.rf
                                           , self.const_sgm
                                           , self.pdobList.shape[0]
                                           , self.cSV
                                           , 0
                                           )[0]
                theta=pyAIAccumulatorPricer(self.flag
                                           , self.direction
                                           , self.s_t
                                           , self.strike
                                           , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                           , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                           , self.daily_amt
                                           , self.isCashsettle
                                           , self.leverage
                                           , self.coupon#coupon
                                           , self.barrier
                                           , self.strike_ramp
                                           , self.barrier_ramp
                                           , self.pdobList
                                           , self.rf
                                           # , self.const_sgm
                                           ,0
                                           , self.pdobList.shape[0]
                                           , self.cSV
                                           , 0
                                           )[4]
            return res

    def __calCoupon_BarAcc(self,mult):
        self.__getInitialCoupon()
        if self.strike=="" and self.barrier=="":
             self.strike,self.barrier=self.__getK_B(self.rng_k,self.rng_b)
        elif self.strike=="" and self.barrier!="":
            self.strike=self.__getK_B(self.rng_k, self.rng_b)[0]
        elif self.strike!="" and self.barrier=="":
            self.barrier=self.__getK_B(self.rng_k, self.rng_b)[-1]
        elif self.strike!="" and self.barrier!="":
            pass
        else:
            print("Wrong Input Interval!")
        self.rebate=abs(self.s_0-self.barrier) if self.rebate=="" else self.rebate
        res=pyAIKOAccumulatorPricer(self.flag[1:]
                                    ,self.direction
                                    ,self.s_t
                                    , self.strike
                                    , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                    , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                    ,self.s_0
                                    ,self.daily_amt
                                    ,self.isCashsettle
                                    ,self.leverage
                                    ,self.leverage_expire
                                    ,self.coupon
                                    ,self.barrier
                                    ,self.rebate
                                    ,self.pdobList
                                    ,self.rf,self.rf
                                    ,self.const_sgm
                                    ,self.pdobList.shape[0]
                                    ,self.cSV)
        if type(res)==bytes:
            return print(res)
        else:
            price,theta=res[0],res[4]
            while self.__getcondition(price,theta,mult,self.barrier,self.strike) and self.coupon>0:
                if price>0:
                    self.coupon+=1 
                else:
                    self.coupon-=1
                # res=pyAIKOAccumulatorPricer(self.flag[1:]
                #                             ,self.direction
                #                             ,self.s_t
                #                             , self.strike
                #                             , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                #                             , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                #                             ,self.s_0
                #                             ,self.daily_amt
                #                             ,self.isCashsettle
                #                             ,self.leverage
                #                             ,self.leverage_expire
                #                             ,self.coupon
                #                             ,self.barrier
                #                             ,self.rebate
                #                             ,self.pdobList
                #                             ,self.rf,self.rf
                #                             ,self.const_sgm
                #                             ,self.pdobList.shape[0]
                #                             ,self.cSV)
                # price,theta=res[0],res[4]
                price=pyAIKOAccumulatorPricer(self.flag[1:]
                                            ,self.direction
                                            ,self.s_t
                                            , self.strike
                                            , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                            , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                            ,self.s_0
                                            ,self.daily_amt
                                            ,self.isCashsettle
                                            ,self.leverage
                                            ,self.leverage_expire
                                            ,self.coupon
                                            ,self.barrier
                                            ,self.rebate
                                            ,self.pdobList
                                            ,self.rf,self.rf
                                            ,self.const_sgm
                                            ,self.pdobList.shape[0]
                                            ,self.cSV)[0]
                theta=pyAIKOAccumulatorPricer(self.flag[1:]
                                            ,self.direction
                                            ,self.s_t
                                            , self.strike
                                            , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                            , datetime2timestamp(str(self.expired_date)+" 15:00:00")
                                            ,self.s_0
                                            ,self.daily_amt
                                            ,self.isCashsettle
                                            ,self.leverage
                                            ,self.leverage_expire
                                            ,self.coupon
                                            ,self.barrier
                                            ,self.rebate
                                            ,self.pdobList
                                            ,self.rf,self.rf
                                            # ,self.const_sgm
                                            ,0
                                            ,self.pdobList.shape[0]
                                            ,self.cSV)[4]
            return res


        
def calculate(trade_date,contracts_num,mult,params,mult_tag):
    params['flag_Coupon']=[1-f for f in params['flag_Inter']]
    params['barrier_ramp']=params['strike_ramp'] if params['barrier_ramp']==[] else params['barrier_ramp']
    for k in params.keys():
          if len(params[k])==0:
              params[k]=[""]
          if len(params[k])!=contracts_num:
              params[k]=params[k]*contracts_num
    df=pd.DataFrame()
    for i in range(len(params['s_0'])):
        ca=CalAcc(underlyingCode=params['underlyingCode'][i]
                  , flag=params['flag'][i], direction=params['direction'][i]
                  , s_t=params['s_t'][i], s_0=params['s_0'][i]
                  , rng_k=params['rng_k'][i],rng_b=params['rng_b'][i]
                  , strike=params['strike'][i], barrier=params['barrier'][i]
                  , rebate=params['rebate'][i],coupon=params['coupon'][i]
                   , pricing_time=datetime.now()
                  # , pricing_time=datetime(2024,10,14,14,30,0)
                  , trade_date=trade_date, first_obsdate=params['first_obsdate'][i]
                  , expired_date=params['expired_date'][i]
                  , obs_days=params['obs_day'][i]
                  , daily_amt=params['daily_amt'][i]
                  , leverage=params['leverage'][i], leverage_expire=params['leverage_expire'][i]
                  , isCashsettle=params['isCashsettle'][i]
                  , strike_ramp=params['strike_ramp'][i], barrier_ramp=params['barrier_ramp'][i]
                  , const_sgm=params['const_sgm'][i]
                  , mult_tag=mult_tag)
        res=ca.getRes(flag_Inter=params['flag_Inter'][i], flag_Coupon=params['flag_Coupon'][i],mult=mult)
        df=pd.concat([df,res],ignore_index=True,axis=1)
    
    df.columns=list(map(lambda c:"结构"+str(c+1),df.columns))
    return df

if __name__=="__main__":
    pd.set_option('display.unicode.ambiguous_as_wide',True)
    pd.set_option('display.unicode.east_asian_width',True)
    pd.set_option('display.width',200)
    trade_date="2024-11-19"
#%% 同歆
if __name__=='__main__':
    mult=0.2
    mult_tag=True
    contracts_num=6
    params={'flag_Inter':[1]
              ,'flag_Coupon':[0]
              ,'underlyingCode':['EG2505']
              ,'flag':['baccputplus','bacccallplus','baccput','bacccall','acccall','accput']
              ,'direction':['B']
              ,'s_t':[]
              ,'s_0':[]
              ,'rng_k':[]
              ,'rng_b':[]
              ,'strike':[]
              ,'barrier':[] # NO means no barrier
              ,'rebate':[]
              ,'coupon':[]
              ,'first_obsdate':[]
              ,'expired_date':[]
              ,'obs_day':[20]
              ,'daily_amt':[1]
              ,'leverage':[1]
              ,'leverage_expire':[0]
              ,'isCashsettle':[0]
              ,'strike_ramp':[10]
              ,'barrier_ramp':[10]
              ,'const_sgm':[0.135]*4+[0.14]*2
              }
    df=calculate(trade_date,contracts_num, mult, params,mult_tag)
    print(df)
    print("Finished!")
#%% 健创
if __name__=='__main__':
    mult=0.3
    mult_tag=True
    contracts_num=12
    params={'flag_Inter':[1]
            # ,'flag_Coupon':[0]
            ,'underlyingCode':['MA501','PP2501','V2501','EG2501','L2501','PP2505']*2
            ,'flag':['bacccallplus']*6+['baccputplus']*6
            ,'direction':['B']
            ,'s_t':[]
            ,'s_0':[]
            ,'rng_k':[]
            ,'rng_b':[]
            ,'strike':[]
            ,'barrier':[]
            ,'rebate':[]
            ,'coupon':[]
            ,'first_obsdate':[]
            ,'expired_date':["2024-12-23","2024-12-20","2024-12-20","2024-12-20","2024-12-20",""]*2
            ,'obs_day':[""]*5+[60]+[""]*5+[60]
            ,'daily_amt':[1]
            ,'leverage':[2]
            ,'leverage_expire':[0]
            ,'isCashsettle':[0]
            ,'strike_ramp':[10]
            ,'barrier_ramp':[10]
            ,'const_sgm':[0.145,0.085,0.155,0.135,0.085,0.085]*2
            }
    df=calculate(trade_date,contracts_num, mult, params,mult_tag)
    print("Finished!")
#%% 健创2
if __name__=='__main__':
    mult=0.3
    mult_tag=True
    contracts_num=12
    params={'flag_Inter':[1]
            # ,'flag_Coupon':[0]
            ,'underlyingCode':['PP2501','EG2501','L2501','PP2505','EG2505','MA505']*2
            ,'flag':['bacccallplus']*6+['baccputplus']*6
            ,'direction':['B']
            ,'s_t':[]
            ,'s_0':[]
            ,'rng_k':[]
            ,'rng_b':[]
            ,'strike':[]
            ,'barrier':[]
            ,'rebate':[]
            ,'coupon':[]
            ,'first_obsdate':[]
            ,'expired_date':["2024-12-20"]*3+[""]*3+["2024-12-20"]*3+[""]*3
            ,'obs_day':[""]*3+[60]*3+[""]*3+[60]*3
            ,'daily_amt':[1]
            ,'leverage':[2]
            ,'leverage_expire':[0]
            ,'isCashsettle':[0]
            ,'strike_ramp':[10]
            ,'barrier_ramp':[10]
            ,'const_sgm':[0.075,0.135,0.075,0.075,0.135,0.14]*2
            }
    df=calculate(trade_date,contracts_num, mult, params,mult_tag)
    print("Finished!")
#%% 棉花
if __name__=='__main__':
    mult=0.2
    mult_tag=True
    contracts_num=1
    params={'flag_Inter':[1]
              ,'underlyingCode':['CF501']
              ,'flag':['bacccall']
              ,'direction':['B']
              ,'s_t':[14050]
              ,'s_0':[]
              ,'rng_k':[100]
              ,'rng_b':[]
              ,'strike':[]
              ,'barrier':[] #NO for call without barrier 0 for put without barrier
              ,'rebate':[100]
              ,'coupon':[]
              ,'first_obsdate':[]
              ,'expired_date':["2024-10-24"]
              ,'obs_day':[]
              ,'daily_amt':[1]
              ,'leverage':[2]
              ,'leverage_expire':[0]# 我司自己的概念
              ,'isCashsettle':[0]
              ,'strike_ramp':[100]
              ,'barrier_ramp':[100]
              ,'const_sgm':[0.12]
              }
    df=calculate(trade_date,contracts_num, mult, params,mult_tag)
    print(df)

#%% 其他
if __name__=='__main__':
    mult=0.1
    mult_tag=True
    contracts_num=1
    params={'flag_Inter':[1]
              ,'underlyingCode':['L2501']
              ,'flag':['baccput']
              ,'direction':['B']
              ,'s_t':[]
              ,'s_0':[]
              ,'rng_k':[300]
              ,'rng_b':[]
              ,'strike':[]
              ,'barrier':[] #NO for call without barrier 0 for put without barrier
              ,'rebate':[]
              ,'coupon':[]
              ,'first_obsdate':[]
              ,'expired_date':[]
              ,'obs_day':[50]
              ,'daily_amt':[1]
              ,'leverage':[2]
              ,'leverage_expire':[0]# 我司自己的概念
              ,'isCashsettle':[0]
              ,'strike_ramp':[30]
              ,'barrier_ramp':[]
              ,'const_sgm':[0.20]
              }
    df=calculate(trade_date,contracts_num, mult, params,mult_tag)
    print(df)
#%% regular acc plus
if __name__=='__main__':
    flag='acccall'
    underlyingCode='EB2501'
    direction=-1
    s_t=8288
    strike=s_t-200
    pricing_time=datetime.now()
    trade_date=datetime.now().date()
    first_obsdate=trade_date
    expired_date=datetime(2025,1,13).date()
    sigma=0.18
    
    pdob=getpdobList(underlyingCode, trade_date, first_obsdate, expired_date, pricing_time)
    cSV=getVolsurfacejson(trade_date, underlyingCode)
    
    # d_ts=np.arange(-10000,-100,50)
    d_ts=np.arange(0,1000,5)
    res=pd.DataFrame(index=d_ts,columns=['price','pv'])
    for d in d_ts :
        p1=pyAIAccumulatorPricer(flag
                                   , direction
                                   , s_t
                                   , strike
                                   , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , datetime2timestamp(str(expired_date)+" 15:00:00")
                                   , 1 #daily amt
                                   , 0 # iscashsettle
                                   , 2 #leverage
                                   , 0 #coupon
                                   , s_t+d
                                   , 0 #strike ramp
                                   , 0# barrier ramp
                                   , pdob
                                   , 0.03
                                   , sigma
                                   , pdob.shape[0]
                                   , cSV
                                   , 0
                                   )[0]
        p2=pyAIAccumulatorPricer(flag
                                   , direction
                                   , s_t
                                   , s_t+d
                                   , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                   , datetime2timestamp(str(expired_date)+" 15:00:00")
                                   , 1 #daily amt
                                   , 0 # iscashsettle
                                   , 0 #leverage
                                   , 0#coupon
                                   , 1000000000
                                   , 0 #strike ramp
                                   , 0# barrier ramp
                                   , pdob
                                   , 0.03
                                   , sigma
                                   , pdob.shape[0]
                                   , cSV
                                   , 0
                                   )[0]
        res.loc[d,'price']=p1+p2
        
    for d in d_ts :
           p1=pyAIAccumulatorPricer(flag
                                      , direction
                                      , s_t
                                      , strike
                                      , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                      , datetime2timestamp(str(expired_date)+" 15:00:00")
                                      , 1 #daily amt
                                      , 0 # iscashsettle
                                      , 2 #leverage
                                      , 0 #coupon
                                      , s_t+d
                                      , 0 #strike ramp
                                      , 0# barrier ramp
                                      , pdob
                                      , 0.03
                                      , 0# cosvol
                                      , pdob.shape[0]
                                      , cSV
                                      , 0
                                      )[0]
           p2=pyAIAccumulatorPricer(flag
                                      , direction
                                      , s_t
                                      , s_t+d
                                      , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                      , datetime2timestamp(str(expired_date)+" 15:00:00")
                                      , 1 #daily amt
                                      , 0 # iscashsettle
                                      , 0 #leverage
                                      , 0 #coupon
                                      , 1000000000
                                      , 0 #strike ramp
                                      , 0# barrier ramp
                                      , pdob
                                      , 0.03
                                      , 0 #cosvol
                                      , pdob.shape[0]
                                      , cSV
                                      , 0
                                      )[0]
           res.loc[d,'pv']=p1+p2
    # plt.plot(d_ts,p_ts)
    res.plot()
    #%%
    # def __calInter_BarAcc(self,mult):
        # rng_k,rng_b=ca._CalAcc__getInitialRng()
        # # strike,barrier=getK_B(self.flag[2:], self.s_t, rng_k,rng_b)
        # strike,barrier=ca._CalAcc__getK_B(rng_k,rng_b)
        # rebate=abs(ca.s_0-barrier) if ca.rebate=="" else ca.rebate
        # res=pyAIKOAccumulatorPricer(ca.flag[1:]
        #                             ,ca.direction
        #                             ,ca.s_t
        #                             , strike
        #                             , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                             , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
        #                             ,ca.s_0
        #                             ,ca.daily_amt
        #                             ,ca.isCashsettle
        #                             ,ca.leverage
        #                             ,ca.leverage_expire
        #                             ,ca.coupon
        #                             ,barrier
        #                             ,rebate
        #                             ,ca.pdobList
        #                             ,ca.rf,ca.rf
        #                             ,ca.const_sgm
        #                             ,ca.pdobList.shape[0]
        #                             ,ca.cSV)
        # if type(res)==bytes:
        #     return print(res)
        # else:
        #     price,theta=res[0],res[4]
        #     while ca._CalAcc__getcondition(price,theta,mult,barrier,strike):
        #         if price>0:
        #               rng_k=rng_k+1 if ca.rng_k==0 else  ca.rng_k
        #               rng_b=rng_b+1 if ca.rng_b==0 else  ca.rng_b
        #         else:
        #               rng_k=rng_k-1 if ca.rng_k==0 else  ca.rng_k
        #               rng_b=rng_b-1 if ca.rng_b==0 else  ca.rng_b
                    
        #         # strike,barrier=getK_B(self.flag, self.s_t, rng_k,rng_b)
        #         strike,barrier=ca._CalAcc__getK_B(rng_k,rng_b)
        #         rebate=abs(self.s_0-barrier) if self.rebate=="" else self.rebate
        #         # res=pyAIKOAccumulatorPricer(self.flag[1:]
        #         #                             ,self.direction
        #         #                             ,self.s_t
        #         #                             , strike
        #         #                             , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #         #                             , datetime2timestamp(str(self.expired_date)+" 15:00:00")
        #         #                             ,self.s_0
        #         #                             ,self.daily_amt
        #         #                             ,self.isCashsettle
        #         #                             ,self.leverage
        #         #                             ,self.leverage_expire
        #         #                             ,self.coupon
        #         #                             ,barrier
        #         #                             ,rebate
        #         #                             ,self.pdobList
        #         #                             ,self.rf,self.rf
        #         #                             ,self.const_sgm
        #         #                             ,self.pdobList.shape[0]
        #         #                             ,self.cSV)
        #         # price,theta=res[0],res[4]
        #         price=pyAIKOAccumulatorPricer(self.flag[1:]
        #                                     ,self.direction
        #                                     ,self.s_t
        #                                     , strike
        #                                     , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                                     , datetime2timestamp(str(self.expired_date)+" 15:00:00")
        #                                     ,self.s_0
        #                                     ,self.daily_amt
        #                                     ,self.isCashsettle
        #                                     ,self.leverage
        #                                     ,self.leverage_expire
        #                                     ,self.coupon
        #                                     ,barrier
        #                                     ,rebate
        #                                     ,self.pdobList
        #                                     ,self.rf,self.rf
        #                                     ,self.const_sgm
        #                                     ,self.pdobList.shape[0]
        #                                     ,self.cSV)[0]
                
        #         theta=pyAIKOAccumulatorPricer(self.flag[1:]
        #                                     ,self.direction
        #                                     ,self.s_t
        #                                     , strike
        #                                     , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                                     , datetime2timestamp(str(self.expired_date)+" 15:00:00")
        #                                     ,self.s_0
        #                                     ,self.daily_amt
        #                                     ,self.isCashsettle
        #                                     ,self.leverage
        #                                     ,self.leverage_expire
        #                                     ,self.coupon
        #                                     ,barrier
        #                                     ,rebate
        #                                     ,self.pdobList
        #                                     ,self.rf,self.rf
        #                                     # ,self.const_sgm
        #                                     ,0
        #                                     ,self.pdobList.shape[0]
        #                                     ,self.cSV)[4]
        #     self.strike,self.barrier,self.rebate=strike,barrier,rebate
        #     return res

    #%%
    # s_t=1
    # k=1.03
    # b=0.97
    # s_0=1
    # reb=0.01
    # # vfe=api.getVol_json("2024-08-26", "I2501")['mid']
    # # vfe=json.dumps(literal_eval(str(vfe)))
    # # cSV = jsonvolSurface2cstructure_selfapi(vfe)
    # pt=datetime(2024,8,27,14,0,0)
    # theta_ts=[]
    # for s in np.arange(0.9,1.1,0.001)*s_0:
    #     t= pyAIKOAccumulatorPricer("accput"
    #                                 ,-1
    #                                 ,s
    #                                 , k
    #                                 , datetime2timestamp(pt.strftime('%Y-%m-%d %H:%M:%S'))
    #                                 , datetime2timestamp("2024-08-27 15:00:00")
    #                                 ,s_0
    #                                 ,1
    #                                 ,0
    #                                 ,2
    #                                 ,10
    #                                 ,0
    #                                 ,b
    #                                 ,reb
    #                                 ,getpdobList("I2501","2024-08-26","2024-08-27","2024-08-27",pt)
    #                                 ,0.03,0.03
    #                                 ,0.3
    #                                 ,getpdobList("I2501","2024-08-26","2024-08-27","2024-08-27",pt).shape[0]
    #                                 ,cSV)[4]
    #     theta_ts.append(t)
       
       
       # pyAIKOAccumulatorPricer(self.flag[1:]
       #                             ,self.direction
       #                             ,self.s_t
       #                             , self.strike
       #                             , datetime2timestamp(self.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
       #                             , datetime2timestamp(str(self.expired_date)+" 15:00:00")
       #                             ,self.s_0
       #                             ,self.daily_amt
       #                             ,self.isCashsettle
       #                             ,self.leverage
       #                             ,self.leverage_expire
       #                             ,self.coupon
       #                             ,self.barrier
       #                             ,self.rebate
       #                             ,self.pdobList
       #                             ,self.rf,self.rf
       #                             # ,self.const_sgm
       #                             ,0
       #                             ,self.pdobList.shape[0]
       #                             ,self.cSV)[0]
       
    # sdate = datetime(2024,8,28).date()
    # edate =  datetime(2025,2,28).date()
    # now = datetime(2024, 8, 28, 13, 56, 29, 55385)
    # T = int(datetime.timestamp(datetime.combine(edate, time(15,0,0))))
    # t = int(datetime.timestamp(now))
    # calendar = pd.DataFrame([(int(datetime.timestamp(datetime.combine(i, time(15,0,0)))),0) for i in rqd.get_trading_dates(sdate,edate)],columns=['index','close'])
    # calendar = calendar.set_index('index')

    # result = []
    # for i in range(7160,7700):
    #     temp = pyAIKOAccumulatorPricer(flag='acccall',direction=-1,daily_amt=1,
    #                             isCashsettle=1,
    #                             pdobList=calendar,
    #                             r=0.03,q=0.03,cVS=cSV,
    #                             t=t,T=T,s_0=7942,
    #                             s=7942,
    #                             k=7160,
    #                             daily_leverage=2,expiry_leverage=0,
    #                             coupon=0,
    #                             barrier=7992,
    #                             rebate=50,
    #                             const_sgm=0.162,
    #                             N=len(calendar),)
    #     result.append(temp[0])
    # plt.plot(range(7160,7700),result)
       #%%
    # plt.plot(np.arange(0.9,1.1,0.001)*s_0,theta_ts)
#%% 计算
if __name__=='__main__':
    api=SelfAPI()
    import requests
    # import json

    # # none,
    # # strike,
    # # barrier,
    # # interval,
    # # payment,
    parameterX="payment"
    
    
    underlyingCode="EG2501"
    vol="17"
    barrier=0
    strike=4767-80
    barrier_ramp,strike_ramp=0,0
    s_0=4767
    lev_exp=0
    lev_daily=2
    expired_date="2024-10-16"
    trade_date="2024-10-14"
    opttype="AICallFixAccPricer"
    coupon=""
    rebate=""
    payload=json.dumps({
                "tradeType": "singleLeg",
                "openOrClose": "open",
                "optionCombType": "",
                "sort": "1",
             	"quoteList": [{
                  		# "tradeTime":"14:37:25"
                        "tradeVol": vol
                        ,"tradeVolume": 3
                     	,"basicQuantity": 1
                        # ,"ttm": 3
                        ,"underlyingCode": underlyingCode
            # 			"algorithmName": "",
            # 			"alreadyKnockedIn": true,
             			,"barrier": barrier
             			,"barrierRamp": barrier_ramp
            		
            # 			"bonusRateAnnulized": true,
            # 			"bonusRateStructValue": 0,
             			,"buyOrSell": "buy"
            # 			"callOrPut": "put",
            # 			"ceilFloor": "",
             			,"clientId": "1"
            # 			"discountRate": 0,
            # 			"enhancedStrike": 0,
             			,"entryPrice": s_0
            # 			"evaluationTime": "2024-08-01",
             			,"expireMultiple": lev_exp
             			,"fixedPayment": coupon,
                        
            # 			"isTest": true,
            # 			"knockinBarrierRelative": true,
            # 			"knockinBarrierShift": 0,
            # 			"knockinBarrierValue": 0,
             			"knockoutRebate": rebate,
             			"leverage": lev_daily,
             			"maturityDate": expired_date,
             			# "midVol": 0,
            # 			"obsNumber": 0,
            # 			"optionPremiumPercent": 0,
            # 			"optionPremiumPercentAnnulized": true,
             			"optionType": opttype,
             			"parameterX": parameterX,
             			"settleType": "cash",
            			
             			"strike": strike,
            # 			"strike2": 0,
            # 			"strike2OnceKnockedinRelative": true,
            # 			"strike2OnceKnockedinShift": 0,
            # 			"strike2OnceKnockedinValue": 0,
            # 			"strikeOnceKnockedinRelative": true,
            # 			"strikeOnceKnockedinShift": 0,
            # 			"strikeOnceKnockedinValue": 0,
             			"strikeRamp": strike_ramp,
             			"tradeCode": "",
             			"tradeDate": trade_date,
            # 			"tradeDividendYield": 0,
             			"tradeObsDateList": [
                             #{"obsDate": "2024-10-11","price":""}
                                              # ,{"obsDate": "2024-10-14","price":""}
                                              
                                              {"obsDate": "2024-10-15","price":""}
                                              ,{"obsDate": "2024-10-16","price":""}
                                              ]
                          }]
                              
    
    })
    requests.post(api.URL+api.ENDPOINT['Pricing'],headers=api.headers
        ,data=payload).json()

#%%
    # mult=0.5
    # df=pd.DataFrame()
    # for i in range(len(params['s_0'])):
    #     ca=CalAcc_2(underlyingCode=params['underlyingCode'][i]
    #               , flag=params['flag'][i], direction=params['direction'][i]
    #               , s_t=params['s_t'][i], s_0=params['s_0'][i]
    #               , rng_k=params['rng_k'][i],rng_b=params['rng_b'][i]
    #               , strike=params['strike'][i], barrier=params['barrier'][i]
    #               , rebate=params['rebate'][i],coupon=params['coupon'][i]
    #               , pricing_time=datetime.now()
    #               , trade_date=trade_date, first_obsdate=params['first_obsdate'][i]
    #               , expired_date=params['expired_date'][i]
    #               , obs_days=params['obs_day'][i]
    #               , daily_amt=params['daily_amt'][i]
    #               , leverage=params['leverage'][i], leverage_expire=params['leverage_expire'][i]
    #               , isCashsettle=params['isCashsettle'][i]
    #               , strike_ramp=params['strike_ramp'][i], barrier_ramp=params['barrier_ramp'][i]
    #               , const_sgm=params['const_sgm'][i])
    #     res=ca.getRes(flag_Inter=params['flag_Inter'][i], flag_Coupon=params['flag_Coupon'][i],mult=mult)
    #     df=pd.concat([df,res],ignore_index=True,axis=1)
    
    # df.columns=list(map(lambda c:"结构"+str(c+1),df.columns))

    #%%
    # import tkinter as tk
    # from tkinter import ttk
    # from pandastable import Table
    # root=tk.Tk()
    # root.title("Dataframe display")
    # frame=tk.Frame(root)
    # frame.pack(fill="both",expand=True)
    # table=
    # # # 运行主循环
    # root.mainloop()

    #%%
    
# # st_range=np.arange(0.99,1.01,0.0001)
# st_range=np.arange(0.92,1.08,0.01)
# # dfres=pd.DataFrame(columns=['pv', 'delta', 'gamma', 'vega_percentage', 'theta_per_day'])
# fig,axs=plt.subplots(3,5,figsize=(15,12))  
# for j,pricing_time in enumerate([datetime.now(),datetime(2024,6,21,11,0,0),datetime(2024,6,21,14,40,0)]):

#     dic_res=dict()
#     for col in  df.columns:
#         dfres=pd.DataFrame()
#         pdo=getpdobList(df[col]['标的合约'], df[col]['交易日期'], df[col]['起始观察日'], df[col]['到期日期'], pricing_time)
#         cSV=jsonvolSurface2cstructure_selfapi(json.dumps(literal_eval(str(api.getVol_json(str(df[col]['交易日期']),df[col]['标的合约'])['mid']))))
#         # dfres.index=index=np.arange(0.9,1.1,0.01)*df[col]['入场价格']
#         res=[]
#         for s_t in st_range*df[col]['入场价格']:
#             res+=pyAIKOAccumulatorPricer(getflag_reverse(df[col]['期权类型'])[2:]
#                                         ,-1
#                                         ,s_t
#                                         , float(df[col]['行权价格'][:5])
#                                         , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
#                                         , datetime2timestamp(str(df[col]['到期日期'])+" 15:00:00")
#                                         ,df[col]['入场价格']
#                                         ,100
#                                         ,0
#                                         ,2
#                                         ,0
#                                         ,0 if df[col]['区间赔付']=="线性" else float(df[col]['区间赔付'])
#                                         ,float(df[col]['敲出价格'][:5])
#                                         ,0 if type(df[col]['敲出赔付'])==str else float(df[col]['敲出赔付'])
#                                         ,pdo
#                                         ,0.03,0.03
#                                         ,0
#                                         ,pdo.shape[0]
#                                         ,cSV)[:5]
#         dfres['st']=(st_range*df[col]['入场价格']).repeat(5)
#         dfres['greeks']=res    
#         dfres['items']=['pv', 'delta', 'gamma', 'vega', 'theta']*int(dfres.shape[0]/5)
#         dic_res[col]=dfres
    

#     for i,g in enumerate(['pv', 'delta', 'gamma', 'vega', 'theta']):
#         # axs=fig.add_subplot(3,5,i+1)
#         for k in dic_res.keys():
#             # ax.plot(dic_res[k].groupby(['items','st'])['greeks'].last()[g],label=getflag_reverse(df[k]['期权类型'])[2:])
#             # axs[j,i].plot(dic_res[k].groupby(['items','st'])['greeks'].last()[g],label=getflag_reverse(df[k]['期权类型'])[2:])
#             axs[j,i].plot(dic_res[k].groupby(['items','st'])['greeks'].last()[g],label=df[k]['区间赔付'])
#             # axs[j,i].vlines(float(df[k]['敲出价格'][:5])
#             #                 ,dic_res[k].groupby(['items','st'])['greeks'].last()[g].min()
#             #                 ,dic_res[k].groupby(['items','st'])['greeks'].last()[g].max(),linestyle='--',label='B'+getflag_reverse(df[k]['期权类型'])[2:])

#             axs[j,i].set_title(g)   
#     plt.legend()
# plt.tight_layout()
    
    
    
#%%
# ca._CalAcc__getInitialCoupon()
# if ca.strike=="" and ca.barrier=="":
#       ca.strike,ca.barrier=ca._CalAcc__getK_B(ca.rng_k,ca.rng_b)
# elif ca.strike=="" and ca.barrier!="":
#     ca.strike=ca._CalAcc__getK_B(ca.rng_k, ca.rng_b)[0]
# elif ca.strike!="" and ca.barrier=="":
#     ca.barrier=ca._CalAcc__getK_B(ca.rng_k, ca.rng_b)[-1]
# elif ca.strike!="" and ca.barrier!="":
#       pass
# else:
#       print("Wrong Input Interval!")
# res=pyAIAccumulatorPricer(ca.flag
#                             , ca.direction
#                             , ca.s_t
#                             , ca.strike
#                             , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
#                             , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
#                             , ca.daily_amt
#                             , ca.isCashsettle
#                             , ca.leverage
#                             , ca.coupon
#                             , ca.barrier
#                             , ca.strike_ramp
#                             , ca.barrier_ramp
#                             , ca.pdobList
#                             , ca.rf
#                             , ca.const_sgm
#                             , ca.pdobList.shape[0]
#                             , ca.cSV
#                             , 0
#                             )
# price,theta=res[0],res[4]
# while ca._CalAcc__getcondition(price,theta,mult,ca.barrier,ca.strike) and ca.coupon>0:
#     if price>0:
#         ca.coupon+=1 
#     else:
#         ca.coupon-=1
#     print(ca.coupon)
#     res=pyAIAccumulatorPricer(ca.flag
#                                 , ca.direction
#                                 , ca.s_t
#                                 , ca.strike
#                                 , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
#                                 , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
#                                 , ca.daily_amt
#                                 , ca.isCashsettle
#                                 , ca.leverage
#                                 , ca.coupon#coupon
#                                 , ca.barrier
#                                 , ca.strike_ramp
#                                 , ca.barrier_ramp
#                                 , ca.pdobList
#                                 , ca.rf
#                                 , ca.const_sgm
#                                 , ca.pdobList.shape[0]
#                                 , ca.cSV
#                                 , 0
#                                 )
#     price,theta=res[0],res[4]
#%%
        # rng_k,rng_b=ca._CalAcc__getInitialRng()
        # strike,barrier=ca._CalAcc__getK_B(rng_k,rng_b) 
        # res=pyAIAccumulatorPricer(ca.flag
        #                               , ca.direction
        #                               , ca.s_t
        #                               , strike
        #                               , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                               , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
        #                               , ca.daily_amt
        #                               , ca.isCashsettle
        #                               , ca.leverage
        #                               , ca.coupon
        #                               , barrier
        #                               , ca.strike_ramp
        #                               , ca.barrier_ramp
        #                               , ca.pdobList
        #                               , ca.rf
        #                               , ca.const_sgm
        #                               , ca.pdobList.shape[0]
        #                               , ca.cSV
        #                               , 0
        #                               )
        #   if type(res)==bytes:
        #       print(res)
        #   else:
        #       price,theta=res[0],res[4]
        #       while ca._CalAcc__getcondition(price,theta,mult,barrier,strike):
     
        #           if price>0:
        #                 rng_k=rng_k+1 if ca.rng_k==0 else  ca.rng_k
        #                 rng_b=rng_b+1 if ca.rng_b==0 else  ca.rng_b
        #           else:
        #                 rng_k=rng_k-1 if ca.rng_k==0 else  ca.rng_k
        #                 rng_b=rng_b-1 if ca.rng_b==0 else  ca.rng_b
                    
        #           strike,barrier=ca._CalAcc__getK_B(rng_k,rng_b)
        #           # res=pyAIAccumulatorPricer(ca.flag
        #           #                             , ca.direction
        #           #                             , ca.s_t
        #           #                             , strike
        #           #                             , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #           #                             , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
        #           #                             , ca.daily_amt
        #           #                             , ca.isCashsettle
        #           #                             , ca.leverage
        #           #                             , ca.coupon#coupon
        #           #                             , barrier
        #           #                             , ca.strike_ramp
        #           #                             , ca.barrier_ramp
        #           #                             , ca.pdobList
        #           #                             , ca.rf
        #           #                             , ca.const_sgm
        #           #                             , ca.pdobList.shape[0]
        #           #                             , ca.cSV
        #           #                             , 0
        #           #                             )
        #           # price,theta=res[0],res[4]
        #           price=pyAIAccumulatorPricer(ca.flag
        #                                       , ca.direction
        #                                       , ca.s_t
        #                                       , strike
        #                                       , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                                       , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
        #                                       , ca.daily_amt
        #                                       , ca.isCashsettle
        #                                       , ca.leverage
        #                                       , ca.coupon#coupon
        #                                       , barrier
        #                                       , ca.strike_ramp
        #                                       , ca.barrier_ramp
        #                                       , ca.pdobList
        #                                       , ca.rf
        #                                       , ca.const_sgm
        #                                       , ca.pdobList.shape[0]
        #                                       , ca.cSV
        #                                       , 0
        #                                       )[0]

        #           theta=pyAIAccumulatorPricer(ca.flag
        #                                       , ca.direction
        #                                       , ca.s_t
        #                                       , strike
        #                                       , datetime2timestamp(ca.pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
        #                                       , datetime2timestamp(str(ca.expired_date)+" 15:00:00")
        #                                       , ca.daily_amt
        #                                       , ca.isCashsettle
        #                                       , ca.leverage
        #                                       , ca.coupon#coupon
        #                                       , barrier
        #                                       , ca.strike_ramp
        #                                       , ca.barrier_ramp
        #                                       , ca.pdobList
        #                                       , ca.rf
        #                                       # , ca.const_sgm
        #                                       ,0
        #                                       , ca.pdobList.shape[0]
        #                                       , ca.cSV
        #                                       , 0
        #                                       )[4]
        #           # price,theta=res[0],res[4]

    
   
  # #%%
  # # underyingCodes=['RU2409','PP2409','L2409','MA409','V2409','EB2406']
  # # obs_days=[25]
  # # flags=['b_acccallplus']
  
  # # dfparams=pd.DataFrame(columns=['underlyingCode','flag', 'direction',
  # #           , s_t="", s_0="",rng_k=20,rng_b=20
  # #           , strike="", barrier="", rebate="",coupon=0
  # #           , pricing_time=datetime.now()
  # #           , trade_date="2024-03-25"
  # #           , first_obsdate=""
  # #           , expired_date=""
  # #           , obs_days=42
  # #           , daily_amt=1, leverage=2, leverage_expire=0, isCashsettle=0
  # #           , strike_ramp=10, barrier_ramp=10
  # #           , const_sgm=0.138)])
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  