# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:56:08 2023

@author: dzrh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 22:22:41 2022

@author: chengyilin
"""
import xlwings as xw
import pandas as pd
import numpy as np
from scipy.stats import norm

from matplotlib import pyplot as plt
# import tushare as tus
from datetime import timedelta,time,datetime,date
# import time
# import iFinDPy as fd
import os
from CYL.OptionPricing import BSM,calIV,calTradttm


import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")

rf=0.03
q=0
trading_date_list=rqd.get_trading_dates('2010-01-01','2025-12-31')

def getGivenDeltaIV(varity,trd_d,given_delta):
    if abs(given_delta)<=1:
        given_delta=given_delta*100
    
    option_id_list=rqd.options.get_contracts(varity,trading_date=trd_d)          
    if option_id_list==[]:
        return "" 
    else:
         wd=rqd.options.get_greeks(option_id_list,trd_d,trd_d).reset_index() #can only return pretrade date
         wd.delta=wd.delta*100
         corresponding_iv=wd.where((wd.delta>=given_delta-3)&(wd.delta<=given_delta+3)).dropna()['iv'].mean()
         print(trd_d,given_delta)
         return  corresponding_iv

def getLastTradeDate():
    now_time=datetime.now().time()
    if time(16,0,0)<=now_time<time(21,0,0):
        return rqd.get_latest_trading_date()
    elif now_time>=time(21,0,0):
        return rqd.get_latest_trading_date()
    elif now_time<time(16,0,0):
        return rqd.get_previous_trading_date(rqd.get_latest_trading_date(), 1)
    
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

def getDeltaIV(varity,start_d,end_d):
        
        freq='60m'
        option_list=rqd.options.get_contracts(varity)
   
            
        tic=datetime.now()
        wd=rqd.get_price(option_list,start_d,end_d,freq,fields=['close','volume','trading_date']).reset_index()
        # wd.drop(index=wd[wd.volume==0].index,inplace=True)
        wd['datetime']=pd.to_datetime(wd.datetime)
        wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
        option_instruments=rqd.instruments(wd.order_book_id)
          
        wd['strike']=list(map(lambda x:x.strike_price,option_instruments))
        wd['optiontype']=list(map(lambda x:x.option_type,option_instruments))
        wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
        wd['expire_date']=np.where(wd.expire_date==datetime(2024,2,13).date(),datetime(2024,2,7).date(),wd.expire_date)
        wd['underlying']=list(map(lambda x:x.underlying_order_book_id,option_instruments))
        print("Format time with = ", datetime.now() - tic, "s")
        
        
        
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

        spot_price=rqd.get_price(wd.underlying,start_d,end_d,freq,'close').reset_index()     
        spot_price.rename(columns={'order_book_id':'underlying'},inplace=True)
        wd=pd.merge(wd,spot_price,how='outer',on=['underlying','datetime'],suffixes=('_option','_underlying'))
        wd.dropna(inplace=True)
        
        tic=datetime.now()
        wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.close_option,wd.shape[0])
        print("Runing IV time = ", datetime.now() - tic, "s")
        wd['delta_round']=(getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.iv)*100).round(0)
        wd['iv_volume']=wd['iv']*wd['volume']
        wd['delta_signal']=np.select([(wd.delta_round>=0)&(wd.delta_round<=10)
                                       ,(wd.delta_round>15)&(wd.delta_round<=25)
                                       ,(wd.delta_round>25)&(wd.delta_round<=35)
                                       ,(wd.delta_round>35)&(wd.delta_round<=45)
                                       ,(wd.delta_round>45)&(wd.delta_round<=55)
                                       ,(wd.delta_round>=-55)&(wd.delta_round<=-45)
                                       ,(wd.delta_round>-45)&(wd.delta_round<=-35)
                                       ,(wd.delta_round>-35)&(wd.delta_round<=-25)
                                       ,(wd.delta_round>-25)&(wd.delta_round<=-15)
                                       ,(wd.delta_round>=-10)&(wd.delta_round<0)
                                       ]
                                     ,[10
                                        ,20
                                        ,30
                                        ,40
                                        ,50
                                        ,50
                                        ,-40
                                        ,-30
                                        ,-20
                                        ,-10]
                                     )
        result=(wd.groupby(['trading_date','delta_signal'])['iv_volume'].sum()/wd.groupby(['trading_date','delta_signal'])['volume'].sum()).unstack()
        result=result[[-10,-20,-30,-40,50,40,30,20,10]]
        result.fillna(method='ffill',inplace=True)
        return result



# for file in file_list:
        # a=pd.read_csv(path+'\/'+file,index_col='trading_date')
        # print(a.shape[-1])
        # if a.shape[-1]>10:
        #     print(file)
            # drop_c=['-10.1', '-20.1', '-30.1', '-40.1', '50.1', '40.1', '30.1', '20.1',
            #        '10.1']
            # a=a.drop(columns=a[drop_c])
            # a=a.drop(a.index[-1])
            # last_date=a.index.tolist()[-1]
            # start_d=rqd.get_next_trading_date(last_date, 1)
            # trd_date_list=rqd.get_trading_dates(start_d,end_d)
            # dfiv=getDeltaIV(file[:-4],start_d,end_d)
            # dfiv['close']=rqd.get_price(file[:-4].upper()+'99',start_d,end_d,'1d','close').reset_index()['close'].values
            # dfiv.columns=a.columns
            # a=pd.concat([a,dfiv],axis=0)
            # a.to_csv(path+'\/'+file)

def getSurface(varity,trd_date,delta_rng):
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
      # trading_date_list=rqd.get_trading_dates('2019-01-01','2024-12-31')
      # varity='I'
      varity=varity.upper()
      freq='1m'
       # trd_date="2023-07-07"
      option_id_list=rqd.options.get_contracts(varity,trading_date=trd_date)
      
      if option_id_list==[]:
          print('No Listed Contracts')
          return []
      else:
            wd=rqd.get_price(option_id_list,trd_date,trd_date,freq,fields=['close','volume','trading_date']).reset_index()
            # wd.drop(index=wd[wd.volume==0].index,inplace=True)
            wd['datetime']=pd.to_datetime(wd.datetime)
            wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
            option_instruments=rqd.instruments(wd.order_book_id)
               
            wd['strike']=list(map(lambda x:x.strike_price,option_instruments))
            wd['optiontype']=list(map(lambda x:x.option_type,option_instruments))
            wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
            # wd['expire_date']=np.where(wd.expire_date==datetime(2024,2,13).date(),datetime(2024,2,7).date(),wd.expire_date)
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
            wd['iv']=getIVArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.close_option,wd.shape[0])
    
            # wd['iv']=list(map(lambda opttype, s, k, t, target:calIV(opttype, s, k, t, rf, q, target)
            #                   ,wd.optiontype, wd.close_underlying
            #                 , wd.strike, wd.t/252
            #                 , wd.close_option))
    
            # print("Running time = ", time.time() - tic, "s")
    
            wd['delta']=getDeltaArr(wd.optiontype,wd.close_underlying,wd.strike,wd.t/252,wd.iv)*100
            
            
            surface=pd.DataFrame(index=wd.expire_date.drop_duplicates().sort_values().values,columns=delta_rng)
            for given_delta in delta_rng:
                # given_delta=-20
                sel_delta=wd.where((wd.delta>=given_delta-3)&(wd.delta<=given_delta+3)).dropna()
                sel_delta['iv_vol']=sel_delta.iv*sel_delta.volume
                surface[given_delta]=sel_delta.groupby('expire_date')['iv_vol'].sum()/sel_delta.groupby('expire_date')['volume'].sum()
                # surface[given_delta]=sel_delta.groupby('expire_date')['iv'].mean()*100
            surface.fillna(method='ffill',inplace=True)
  
            return surface*100
        
        
def ReLoadIV(path,end_d):
        # end_d='2023-07-26'
        delta_rng=[-10,-20,-30,-40,50,40,30,20,10]
        if end_d=="" :
            end_d=rqd.get_future_latest_trading_date()
        file_list=os.listdir(path)
        for file in file_list:
            # file=file_list[4]
            print(file)
            a=pd.read_csv(path+r'\/'+file,index_col='trade_date')    
            last_date=a.index.tolist()[-1]
            start_d=rqd.get_next_trading_date(last_date, 1)
            trd_date_list=rqd.get_trading_dates(start_d,end_d)
            if trd_date_list==[]:
                continue
            df=pd.DataFrame()
            for trd_date in trd_date_list:
                suf=getSurface(file[:-8],trd_date,delta_rng)
                suf['trade_date']=trd_date
                df=pd.concat([df,suf])
            df.fillna(method='ffill',inplace=True)
            df['expire_date']=df.index
            df.index=df.trade_date
            df=df.drop(columns='trade_date')
            close_ts=rqd.get_price(file[:-8].upper()+'99',start_d,end_d,'1d','close').reset_index().values[:,1:]
            close_ts=pd.DataFrame(index=close_ts[:,0],data=close_ts[:,1])
            df['close']=close_ts.loc[df.index]
            a.columns=df.columns
            a=pd.concat([a,df],axis=0)
            a.to_csv(path+r'\/'+file)
        print('IV Reload Finished')
        
if __name__=="__main__":

    path=r'D:\chengyilin\ivdata_suf'
    ReLoadIV(path,"")
    
    
    
    #%% load new asset
    # varity_dic={'AO':"2024-09-02"
    #             # ,'PF':0
    #             # ,'SF':0
    #             # ,'SH':0
    #             # ,'SM':0
    #             # ,'AL':0
     
    # }
      
     
            
    # end_d="2024-09-02"
    # delta_rng=[-10,-20,-30,-40,50,40,30,20,10]
    # for v in varity_dic.keys():
    #     df=pd.DataFrame()
    #     index=rqd.get_trading_dates(varity_dic[v],end_d)
    #     for t in index:
    # #        # print(t)
    #         suf=getSurface(v,t,delta_rng)
    #         suf['trade_date']=t
    #         df=pd.concat([df,suf])
    #     df.fillna(method='ffill',inplace=True)
    #     df['expire_date']=df.index
    #     df.index=df.trade_date
    #     df=df.drop(columns='trade_date')
    #     close_ts=rqd.get_price(v.upper()+'99',varity_dic[v],end_d,'1d','close').reset_index().values[:,1:]
    #     close_ts=pd.DataFrame(index=close_ts[:,0],data=close_ts[:,1])
    #     df['close']=close_ts.loc[df.index]
    #     df.to_csv(path+'\/'+v+'_SUF.csv')

    
