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
from datetime import date,timedelta,time,datetime
# import time
# import iFinDPy as fd
import os
from CYL.OptionPricing import BSM,calIV,calTradttm
from CYL.YieldChainAPI import YieldChainAPI

import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
rf=0.03
q=0
trading_date_list=rqd.get_trading_dates('2010-01-01','2025-12-31')

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
     
def LoadIV(path,end_d):
    # end_d='2023-07-26'
    if end_d=="":
        end_d=rqd.get_previous_trading_date(rqd.get_latest_trading_date(), 1)
    file_list=os.listdir(path)
    for file in file_list:
        a=pd.read_csv(path+'\/'+file,index_col='Unnamed: 0')    
        last_date=a.index.tolist()[-1]
        start_d=rqd.get_next_trading_date(last_date, 1)
        trd_date_list=rqd.get_trading_dates(start_d,end_d)
        if trd_date_list==[]:
            continue
        dfiv=pd.DataFrame(index=trd_date_list,columns=a.columns)
        for col in dfiv.columns[:-2]:
            dfiv[col]=[getDeltaIV(file[:-4],trd_d,float(col)) for trd_d in dfiv.index]
        dfiv['close']=rqd.get_price(file[:-4].upper()+'99',start_d,end_d,'1d','close').reset_index()['close'].values
        a=pd.concat([a,dfiv],axis=0)
        a.to_csv(path+'\/'+file)
    print('IV Reload Finished')

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
      # varity='I'
      varity=varity.upper()
      freq='10m'
       # trd_date="2023-07-07"
      option_id_list=rqd.options.get_contracts(varity,trading_date=trd_date)
      
      if option_id_list==[]:
          print('No Listed Contracts')
          return []
      else:
            # wd=rqd.options.get_greeks(option_id_list,trd_date,trd_date).reset_index()
            # wd['delta']=wd.delta*100
            # option_instruments=rqd.instruments(wd.order_book_id)
            # wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))

        # wd=rqd.get_price(option_id_list,trd_date,trd_date,freq,fields=['open','high','low','close','volume','trading_date']).reset_index()
        wd=rqd.get_price(option_id_list,trd_date,trd_date,freq,fields=['close','volume','trading_date']).reset_index()
        # wd.drop(index=wd[wd.volume==0].index,inplace=True)
        wd['datetime']=pd.to_datetime(wd.datetime)
        wd['trading_date']=wd.trading_date.apply(lambda d:d.date())
        option_instruments=rqd.instruments(wd.order_book_id)
           
        wd['strike']=list(map(lambda x:x.strike_price,option_instruments))
        wd['optiontype']=list(map(lambda x:x.option_type,option_instruments))
        wd['expire_date']=list(map(lambda x:datetime.strptime(x.maturity_date,'%Y-%m-%d').date(),option_instruments))
        wd['expire_date']=np.where(wd.expire_date==datetime(2024,2,13).date(),datetime(2024,2,7).date(),wd.expire_date)

        wd['underlying']=list(map(lambda x:x.underlying_order_book_id,option_instruments))
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
        wd.drop(index=wd[wd.t<=3].index,inplace=True)
         
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
            sel_delta=wd.where((wd.delta>=given_delta-5)&(wd.delta<=given_delta+5)).dropna()
            sel_delta['iv_vol']=sel_delta.iv*sel_delta.volume
            surface[given_delta]=sel_delta.groupby('expire_date')['iv_vol'].sum()/sel_delta.groupby('expire_date')['volume'].sum()
            # surface[given_delta]=sel_delta.groupby('expire_date')['iv'].mean()*100
                 
            
            
            
            
        surface.fillna(method='bfill',inplace=True)
        surface.fillna(method='ffill',inplace=True)
        return surface*100


if __name__=="__main__":

    path=r'D:\chengyilin\ivdata'
#%%
    varity_dic={'CF':"2019-1-28"
                # ,'CU':0
                # ,'EB':0
                # ,'EG':0
                # ,'I':0
                # ,'L':0
                # ,'MA':0
                # ,'PP':0
                # ,'PX':0
                # ,'RB':0
                # ,'RU':0
                # ,'SC':0
                # ,'TA':0
                # ,'V':0
    
    }
    # for v in varity_dic.keys():
    #     df=pd.read_csv(path+'\/'+v.upper()+'.csv')
    #     varity_dic[v]=df.iloc[0,0]

    # trading_date_list=rqd.get_trading_dates('2010-01-01','2024-12-31')
    
    # for v in varity_dic.keys():
    #     start_d= varity_dic[v]
    #     trd_date_list=rqd.get_trading_dates(start_d,'2023-11-15')
    #     dfiv=pd.DataFrame(index=trd_date_list,columns=[-10,-20,-30,50,30,20,10])
    #     for col in dfiv.columns:
    #         tic=datetime.now()
    #         dfiv[col]=[getGivenDeltaIV(v,trd_d,col) for trd_d in dfiv.index]
    #         print("Runing on delta time with = ", datetime.now() - tic, "s")
    #     dfiv['close']=rqd.get_price(v.upper()+'99',trd_date_list[0],trd_date_list[-1],'1d','close').reset_index()['close'].values
    #     dfiv.to_csv(path+'\/'+v.upper()+'.csv')
    # end_d="2023-11-15"
    # for v in varity_dic.keys():
    #     start_d=varity_dic[v]
    #     dfiv=getDeltaIV(v,start_d,end_d)
    #     dfiv['close']=rqd.get_price(v.upper()+'99',start_d,end_d,'1d','close').reset_index()['close'].values
    #     dfiv.to_csv(path+'\/'+v.upper()+'.csv')
    delta_rng=[-10,-20,-30,-40,50,40,30,20,10]
    end_d="2024-07-26"
    for v in varity_dic.keys():
        df=pd.DataFrame()
        index=rqd.get_trading_dates(varity_dic[v],end_d)
        for t in index:
           suf=getSurface(v,t,delta_rng)
           suf['trade_date']=t
           df=pd.concat([df,suf])
        close_ts=[]
        for t in df.trade_date:
            close_ts.append(rqd.get_price(v.upper()+'99',t,t,'1d','close').reset_index()['close'].values[0])
        df['close']=close_ts
        df.to_csv(path+'\/'+v.upper()+'.csv')
      