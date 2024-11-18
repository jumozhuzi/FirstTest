# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:50:55 2024

@author: dzrh
"""

import pandas as pd
import numpy as np
# from scipy.stats import norm
# from matplotlib import pyplot as plt
# import tushare as tu
from CYL.StressTestNew_api import getRQcode
from datetime import timedelta,datetime,time
from ast import literal_eval
import json
from CYL.pythonAPI_pyfunctions4newDll_3 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer,jsonvolSurface2cstructure_selfapi
import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
from CYL.OTCAPI import SelfAPI,findCode



def getrhvarity(underlying):
    v=underlying[:len(underlying)-4] if underlying[-4].isdigit() else underlying[:len(underlying)-3]
    return v

def getrhcode(underlying):
    
    code=underlying.lower() if underlying[-4].isdigit() else underlying
    return code

def main(hedge_info,file_path):
    
    # variety_list=list(hedge_info.keys())
    a=pd.DataFrame(columns=['varietyCode'],data=list(hedge_info.keys()))
    a['hedge_interval']=a['varietyCode'].map(lambda x: hedge_info.get(x, np.nan)[0])
    a['hedge_ratio']=a['varietyCode'].map(lambda x: hedge_info.get(x, np.nan)[1])
    a['sel_hedge_ass']=a['varietyCode'].map(lambda x: hedge_info.get(x, np.nan)[2])
    a.replace('',np.nan,inplace=True)
    api=SelfAPI()
    
    df=api.getCurrentRisk_Summary(variety_list=a['varietyCode'].tolist())
 
    
    var_gm_lots=df.groupby('varietyCode')['gammaLots'].sum()
    sel_hedge_ass=a['sel_hedge_ass'].dropna().tolist()
    dflots=pd.DataFrame(columns=['trade_lots'],index=sel_hedge_ass)
    dflots['varietyCode']=list(map(lambda x:findCode(x),sel_hedge_ass))
    for v,ass in zip(dflots.varietyCode,sel_hedge_ass):
        ratio=1 if hedge_info[v][1]=="" else hedge_info[v][1] 
        dflots.loc[ass,'trade_lots']=ratio*var_gm_lots[v]*df.groupby(['varietyCode','underlyingCode'])['balancedChanges'].last()[v][ass]
    dflots['underlyingCode']=dflots.index
    df['trade_lots']=df.underlyingCode.map(dflots.trade_lots)
    
    a['hedge_ratio'].fillna(1,inplace=True)
    df['gammaLots_new']=df.varietyCode.apply(lambda x:a[a['varietyCode']==x]['hedge_ratio'].values[0])*df.gammaLots
    df['balancedChanges_new']=df.varietyCode.apply(lambda x:a[a['varietyCode']==x]['hedge_interval'].values[0]).fillna(df.balancedChanges)
    df['short_p']=df.lastPrice+df.balancedChanges_new.round(0)
    df['long_p']=df.lastPrice-df.balancedChanges_new.round(0)

    df['trade_lots'].fillna(df.gammaLots_new*df.balancedChanges_new,inplace=True)
    idx_drop=[]
    for v in dflots.varietyCode:
        try:
            idx_drop+=df[(df.varietyCode==v)&(df.underlyingCode!=dflots[dflots.varietyCode==v].index[0])].index.tolist()
        except:
            idx_drop.append(df[(df.varietyCode==v)&(df.underlyingCode!=dflots[dflots.varietyCode==v].index[0])].index[0])

    df.drop(index=idx_drop,inplace=True)
    df['tick_size']=list(map(lambda x:rqd.instruments(x).tick_size(),df.underlyingCode.apply(getRQcode)))
    df['short_p']=df.short_p-df.short_p.mod(df.tick_size)
    df['long_p']=df.long_p+(df.tick_size-df.long_p.mod(df.tick_size))
    
    
    
    
    dfextra=pd.DataFrame(columns=df.columns)
    # for idx in df.index:
    #     lots=df.loc[idx,'trade_lots']
    #     if abs(df.loc[idx,'long_p']*df.loc[idx,'trade_lots']*rqd.instruments(getRQcode(df.loc[idx,'underlyingCode'])).contract_multiplier)>9000000:
    #         max_lots=int(9000000/df.loc[idx,'long_p']/rqd.instruments(getRQcode(df.loc[idx,'underlyingCode'])).contract_multiplier)
    #         df.loc[idx,'trade_lots']=max_lots
    #         nums,left_lots=divmod(lots,max_lots)
    #         dfextra=pd.concat([dfextra,pd.DataFrame([df.loc[idx]]*int(nums))])
    #         dfextra['trade_lots'].values[-1]=left_lots
    #     else:
    #         pass
        
    for idx in df.index:
        try:
         lots=df.loc[idx,'trade_lots']
         if lots>df_lim.loc[getrhvarity(df.loc[idx,"underlyingCode"])]['Amount']:
             max_lots=df_lim.loc[getrhvarity(df.loc[idx,"underlyingCode"])]['Amount']
             # int(9000000/df.loc[idx,'long_p']/rqd.instruments(getRQcode(df.loc[idx,'underlyingCode'])).contract_multiplier)
             df.loc[idx,'trade_lots']=max_lots
             nums,left_lots=divmod(lots,max_lots)
             dfextra=pd.concat([dfextra,pd.DataFrame([df.loc[idx]]*int(nums))])
             dfextra['trade_lots'].values[-1]=left_lots
         else:
             pass
        except:
            pass
        
    df=pd.concat([dfextra,df],ignore_index=True)

    

    
    dfres=pd.DataFrame()
    for i in ['long','short']:
        df_rh_orders=pd.DataFrame(columns=["账户","合约","C\S","预埋类型","买卖","开平","投保"
                                           ,"委托价","数量","互换","报单指令"])
    
        df_rh_orders['合约']=df.underlyingCode.apply(getrhcode)
        df_rh_orders['账户']=201711
        df_rh_orders['C\S']="云端"
        df_rh_orders['委托价']=df.short_p if i=='short' else df.long_p
        df_rh_orders['数量']=df.trade_lots.apply(int)
        df_rh_orders['预埋类型']="重新进入连续交易"
        df_rh_orders['投保']="投机"
        df_rh_orders['互换']="否"
        df_rh_orders['开平']="开仓"
        df_rh_orders['买卖']="卖" if i=='short' else "买"
        df_rh_orders['报单指令']="不做限制"
        dfres=pd.concat([dfres,df_rh_orders])
    dfres.sort_values('合约',inplace=True)
    dfres.to_csv(file_path,index=False,encoding='gbk')
    print('Finished!')


if __name__=="__main__":
   order_limi_path=r'D:\chengyilin\work\order_limitation.csv'
   df_lim=pd.read_csv(order_limi_path)    
   df_lim.index=df_lim.Varity.apply(lambda x:x.upper())


   #[hedge_interval,hedge_raCF501tio,sel_hedge_ass]
   hedge_info={ # ,'AU':[5,"","AU2412"]
                'MA':[25,0.7,""]
                ,"EB":[95,0.6,""]
               ,'PP':[40,0.7,""]
                ,'L':[40,0.7,"L2505"]
                ,'UR':["",0.7,""]
                # ,"EG":[38,0.7,"EG2501"]
                ,"V":[45,0.7,"V2505"]
                # ,"SH":[40,0.5,"SH501"]
                ,'CF':[85,0.7,"CF501"]
                ,'PR':[45,0.7,""]
               }
   
   main(hedge_info
         ,file_path=r'C:\Users\dzrh\Desktop\selforders.csv')








