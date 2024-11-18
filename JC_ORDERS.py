# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:53:48 2024

@author: dzrh
"""

import pandas as pd
import numpy as np
# from scipy.stats import norm
# from matplotlib import pyplot as plt
# import tushare as tu
from datetime import timedelta,datetime,time
from ast import literal_eval
import json
from CYL.pythonAPI_pyfunctions4newDll_3 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer,jsonvolSurface2cstructure_selfapi
import rqdatac as rqd
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
from CYL.OTCAPI import SelfAPI




api=SelfAPI()
rf=0.03

def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code

def getrhcode(underlying):
    
    code=underlying.lower() if underlying[-4].isdigit() else underlying
    return code

def getrhvarity(underlying):
    v=underlying[:len(underlying)-4] if underlying[-4].isdigit() else underlying[:len(underlying)-3]
    return v

def getpdobList(start_date,end_date):
    '''
    

    Parameters
    ----------
    start_date : str
    end_date : str

    Returns
    -------
    None.

    '''
    pdobList=pd.DataFrame(index=rqd.get_trading_dates(start_date,end_date)
                          , columns=['close'],data=np.nan)
    pdobList.fillna(0,inplace=True)
    pdobList.index= [datetime2timestamp(str(t) + ' 15:00:00') for t in pdobList.index.tolist()]
    return  pdobList


def get_volsuf(trade_date,underlyingCode,vol_type):
    '''
    Parameters
    ----------
    trade_date : str
                 eg. "2024-05-13"
    underlyingCode : str
    vol_type : str
        "ask","bid","mid"

    Returns
    -------
    None.

    '''
    vol=api.getVol_json(trade_date, underlyingCode)[vol_type]
    vfe=json.dumps(literal_eval(str(vol)))
    cSV = jsonvolSurface2cstructure_selfapi(vfe)
    return cSV

def getshort(df):

    cdt=[df.underlyingCode.str.contains("PVC")
            ]
    cho=["V"]
    return np.select(cdt,cho,df.underlyingCode.apply(findCode))


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

def main(file_path):
    conts=pd.read_excel(file_path+r'\JC_orders.xlsx') 
    conts.drop(conts[conts['标的进场价格']=="集合竞价"].index.tolist(),inplace=True)
    conts[conts.columns[-1]].fillna("-",inplace=True)
    dfinter=pd.read_excel(file_path+r'\JC_interval.xlsx')

    
    conts.columns=['drop_0','optiontype','underlyingCode','entryprice','qty'
                    ,'drop_1'
                   ]
    # conts.drop(columns=['drop_0','drop_1'],inplace=True)
    conts.drop(columns=['drop_0'],inplace=True)

    conts['underlyingCode']=conts['underlyingCode'].apply(lambda x:x.upper())
    
    
    conts['short_under']=getshort(conts)+conts.underlyingCode.apply(lambda x:x[-2:])
    dfinter['short_under']=getshort(dfinter)+dfinter.underlyingCode.apply(lambda x:x[-2:])
    
    conts_cdt=conts[conts['drop_1'].str.contains("触碰单")]
    conts.drop(columns=['drop_1'],inplace=True)
    conts.drop(conts_cdt.index,inplace=True)
    conts_fut=conts[conts.optiontype.str.contains("代持")]
    conts=conts[conts.optiontype.str.contains("增强")]
    
    
    conts['optiontype']=np.where(conts.optiontype.str.contains("累沽"),"accputplus","acccallplus")
    conts=conts.merge(dfinter,on='short_under',how='left')
    conts.dropna(inplace=True)
    conts['interval']=conts["熔断增强累沽"].where(conts.optiontype.str.contains("put"))
    conts['interval'].fillna(conts["熔断增强累购"].where(conts.optiontype.str.contains("call")).dropna(),inplace=True)
    conts['strike']=np.where(conts.optiontype.str.contains("call"),conts.entryprice-conts.interval,conts.entryprice+conts.interval)
    conts['barrier']=np.where(conts.optiontype.str.contains("call"),conts.entryprice+conts.interval,conts.entryprice-conts.interval)
    conts['underlyingCode']=conts.underlyingCode_y
    
    
    
    if conts_cdt.shape[0]==0:
        pass
    else:
        df_rh_cdt_orders=pd.DataFrame(index=conts_cdt.index,columns=["账户","合约id",r"C\S（本地；云端）","触发价类型（最新价；买一价；卖一价）"
                                                                     ,"触发价方向（向上突破；向下跌破；大于等于；小于等于；大于；小于）"
                                                                     ,"触发线","有效期（当日；本周；本月）"
                                                                  ,"买卖（买；卖）","开平（开仓；平仓；平今；平昨）"
                                                                  ,"委托价（最新价；对手价；市价；自定义）","超价","自定义委托价"
                                                                  ,"数量","追单（是；否）","投保（投机；套保；套利）"])
        df_rh_cdt_orders["合约id"]=conts_cdt.underlyingCode.apply(getrhcode)
        df_rh_cdt_orders['账户']=201711
    
        df_rh_cdt_orders[r'C\S（本地；云端）']="云端"
        df_rh_cdt_orders['触发价类型（最新价；买一价；卖一价）']="最新价"
        df_rh_cdt_orders['触发价方向（向上突破；向下跌破；大于等于；小于等于；大于；小于）']=np.where(conts_cdt.optiontype.str.contains("平空"),"大于等于","小于等于")
        df_rh_cdt_orders['触发线']=conts_cdt.entryprice
        df_rh_cdt_orders['有效期（当日；本周；本月）']="当日"
        df_rh_cdt_orders['买卖（买；卖）']=np.where(conts_cdt.optiontype.str.contains("平空"),"买","卖")
        df_rh_cdt_orders['开平（开仓；平仓；平今；平昨）']="开仓"
        df_rh_cdt_orders['委托价（最新价；对手价；市价；自定义）']="对手价"
        df_rh_cdt_orders['超价']=0
        df_rh_cdt_orders['自定义委托价']=conts_cdt.entryprice
        df_rh_cdt_orders['数量']=(conts_cdt.qty/conts_cdt.underlyingCode.apply(lambda x:rqd.instruments(x).contract_multiplier)).astype(int)
    
        df_rh_cdt_orders['追单（是；否）']="是"
        df_rh_cdt_orders['投保（投机；套保；套利）']="投机"
    
    df_rh_orders=pd.DataFrame(index=conts.index,columns=["账户","合约",r"C\S","预埋类型","买卖","开平","投保"
                                       ,"委托价","数量","互换","报单指令"])

    df_rh_orders['合约']=conts.underlyingCode.apply(getrhcode)
    df_rh_orders['账户']=201711
    df_rh_orders[r'C\S']="云端"
    df_rh_orders['委托价']=conts.entryprice
    df_rh_orders['预埋类型']="重新进入集合竞价"
    df_rh_orders['投保']="投机"
    df_rh_orders['互换']="否"
    df_rh_orders['开平']="开仓"
    df_rh_orders['买卖']=np.where(conts.optiontype.str.contains("call"),"买","卖")
    df_rh_orders['报单指令']="不做限制"
    
    
    pricing_time=datetime.now()
    for idx in conts.index:
        res=pyAIKOAccumulatorPricer(conts.loc[idx,'optiontype']
                                    ,-1
                                    ,conts.loc[idx,'entryprice']
                                    ,conts.loc[idx,'strike']
                                    , datetime2timestamp(pricing_time.strftime('%Y-%m-%d %H:%M:%S'))
                                    , datetime2timestamp(str(conts.loc[idx,'expire_date'])[:10]+" 15:00:00")
                                    ,conts.loc[idx,'entryprice']
                                    ,conts.loc[idx,'qty'] #daily_amt
                                    ,0 #iscashsettle
                                    ,2 #leverage
                                    ,0 #lev_exprie
                                    ,0 #coupon
                                    ,conts.loc[idx,'barrier'] #barrier
                                    ,0 #rebate
                                    ,getpdobList(conts.loc[idx,'start_date'],conts.loc[idx,'expire_date'])
                                    ,rf,rf
                                    ,0 # const_sgm
                                    ,getpdobList(conts.loc[idx,'start_date'],conts.loc[idx,'expire_date']).shape[0]
                                    ,get_volsuf(str(conts.loc[idx,'start_date'])[:10],conts.loc[idx,'underlyingCode'],"mid")
                                    )
        lots=abs(int(res[1]/rqd.instruments(getRQcode(conts.loc[idx,'underlyingCode'])).contract_multiplier))
        df_rh_orders.loc[idx,'数量']=lots

            
    
    if conts_fut.shape[0]==0:
        pass
    else:
        df_rh_orders_fut=pd.DataFrame(columns=df_rh_orders.columns)
        conts_fut['underlyingCode']=getshort(conts_fut)+conts_fut.underlyingCode.apply(lambda x:x[-4:])
        df_rh_orders_fut['合约']=conts_fut.underlyingCode.apply(lambda x:rqd.instruments(x).trading_code).apply(getrhcode)
        df_rh_orders_fut['账户']=201711
        df_rh_orders_fut[r'C\S']="云端"
        df_rh_orders_fut['委托价']=conts_fut.entryprice
        df_rh_orders_fut['预埋类型']="重新进入集合竞价"
        df_rh_orders_fut['投保']="投机"
        df_rh_orders_fut['互换']="否"
        df_rh_orders_fut['开平']="开仓"
        df_rh_orders_fut['买卖']=np.where(conts_fut.optiontype.str.contains("平空|开多"),"买","卖")
        df_rh_orders_fut['报单指令']="不做限制"
        df_rh_orders_fut['数量']=(conts_fut.qty/conts_fut.underlyingCode.apply(lambda x:rqd.instruments(x).contract_multiplier))
        df_rh_orders=pd.concat([df_rh_orders,df_rh_orders_fut])
        

    df_rh_orders["数量"]=df_rh_orders["数量"].astype(int)
    if conts_cdt.shape[0]==0:
        pass
    else:
        aa=pd.DataFrame(df_rh_orders.groupby(['合约','买卖','委托价']).sum()['数量'])
        bb=pd.DataFrame(df_rh_cdt_orders.groupby(['合约id','买卖（买；卖）','触发线']).sum()['数量'])
        cc=pd.merge(aa.reset_index(),bb.reset_index(),left_on=['合约','委托价'],right_on=['合约id','触发线'])
        cc['net_qty']=np.where(cc['买卖']=='买',cc['数量_x'],-1*cc['数量_x'])+np.where(cc['买卖（买；卖）']=='买',cc['数量_y'],-1*cc['数量_y'])
        cc['net_label']=np.where(cc['net_qty']>0,'买','卖')
        cc['net_qty']=cc['net_qty'].abs()
        
        
        cc_orders=cc[cc['买卖']==cc['net_label']]
        cc_cdt_orders=cc[cc['买卖（买；卖）']==cc['net_label']]    
        mapping=pd.merge(df_rh_orders[['合约','委托价']],cc_orders[['合约','委托价','net_qty']],on=['合约','委托价'],how='left').set_index(['合约','委托价'])['net_qty']
        mapping=mapping[mapping.index.unique()]
        mapping.dropna(inplace=True)
        if mapping.shape[0]>0:
            df_rh_orders['net_qty']=np.nan
            df_rh_orders['net_qty']=df_rh_orders.apply(lambda row: mapping.get((row['合约'], row['委托价']), row['net_qty']), axis=1)
            df_rh_orders['net_qty'].fillna(df_rh_orders['数量'],inplace=True)
            df_rh_orders['数量']=df_rh_orders['net_qty']
            df_rh_orders.drop(columns='net_qty',inplace=True)
        for p in cc_cdt_orders['触发线']:
            df_rh_orders.drop(df_rh_orders[df_rh_orders['委托价']==p].index,inplace=True)
        
        
        
        mapping=pd.merge(df_rh_cdt_orders[['合约id','触发线']],cc_cdt_orders[['合约id','触发线','net_qty']],on=['合约id','触发线'],how='left').set_index(['合约id','触发线'])['net_qty']
        mapping=mapping[mapping.index.unique()]
        mapping.dropna(inplace=True)
        if mapping.shape[0]>0:
            df_rh_cdt_orders['net_qty']=np.nan
            df_rh_cdt_orders['net_qty']=df_rh_cdt_orders.apply(lambda row: mapping.get((row['合约id'], row['触发线']), row['net_qty']), axis=1)
            df_rh_cdt_orders['net_qty'].fillna(df_rh_cdt_orders['数量'],inplace=True)
            df_rh_cdt_orders['数量']=df_rh_cdt_orders['net_qty']
            df_rh_cdt_orders.drop(columns='net_qty',inplace=True)
        for p in cc_orders['委托价']:
            df_rh_cdt_orders.drop( df_rh_cdt_orders[ df_rh_cdt_orders['触发线']==p].index,inplace=True)
        

    

    df_rh_orders_extra=df_rh_orders[df_rh_orders['数量']>df_lim.loc[df_rh_orders["合约"].apply(getrhvarity).values]['Amount'].values]
    if df_rh_orders_extra.shape[0]==0:
        pass
    else:
        for idx in df_rh_orders_extra.index:
            dfextra=pd.DataFrame(columns=df_rh_orders.columns)
    
            max_lots=df_lim.loc[getrhvarity(df_rh_orders_extra.loc[idx,"合约"])]['Amount']
            
            nums,left_lots=divmod(df_rh_orders_extra.loc[idx,'数量'],max_lots)
            df_rh_orders.loc[idx,'数量']=max_lots
            df_rh_orders_extra.loc[idx,'数量']=max_lots
            dfextra=pd.concat([dfextra,pd.DataFrame([df_rh_orders_extra.loc[idx]]*int(nums))])
            dfextra['数量'].values[-1]=left_lots
            df_rh_orders=pd.concat([dfextra,df_rh_orders])
    df_rh_orders["数量"]=df_rh_orders["数量"].astype(object)
    
    
    
    
    
    # df_rh_orders_1=pd.DataFrame(index=df_rh_orders.index,columns=["账户","合约","C\S","预埋类型","买卖","开平","投保"
    #                                    ,"委托价","数量","互换","报单指令"])

    # df_rh_orders_1['合约']=df_rh_orders['合约']
    # df_rh_orders_1['账户']=201711
    # df_rh_orders_1['C\S']="云端"
    # df_rh_orders_1['委托价']=df_rh_orders['委托价']
    # df_rh_orders_1['预埋类型']="重新进入集合竞价"
    # df_rh_orders_1['投保']="投机"
    # df_rh_orders_1['互换']="否"
    # df_rh_orders_1['开平']="开仓"
    # df_rh_orders_1['买卖']=df_rh_orders['买卖']
    # df_rh_orders_1['报单指令']="不做限制"
    # df_rh_orders_1['数量']=df_rh_orders["数量"].values
    
    
    
    
    
    
    
    
    df_rh_orders.to_csv(file_path+r'\JC_rh_orders.csv',index=False,encoding='gbk')  
    
    # if df_rh_cdt_orders.shape[0]>0:
    try:
        df_rh_cdt_orders.to_csv(file_path+r'\JC_rh_cdt_orders.csv',index=False,encoding='gbk')
        print("Condition Order Save!")
    except:
        print("No Condition Order!")
        
    
    print("Finish!")

if __name__=="__main__":
    
    file_path=r'C:\Users\dzrh\Desktop\JC'
    order_limi_path=r'D:\chengyilin\work\order_limitation.csv'
    df_lim=pd.read_csv(order_limi_path)    
    df_lim.index=df_lim.Varity

    main(file_path)
    
   
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    