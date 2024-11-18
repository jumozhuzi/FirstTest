# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:42:19 2024

@author: dzrh
"""

import pandas as pd
import numpy as np
# from scipy.stats import norm
from matplotlib import pyplot as plt
# import tushare as tu
from datetime import timedelta,datetime,time
from CYL.OptionPricing import BSM,calIV,calTradttm,LinearInterpVol
from CYL.AccCalculator import CalAcc
# from CYL.StressTestNew import BarrierReport,CheckForward
from CYL.OTCAPI import SelfAPI
# from CYL.YieldChainAPI import YieldChainAPI
import rqdatac as rqd
# rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
# import streamlit as stm
# import streamlit_searchbox as stm_box
import re
import requests
import json
api=SelfAPI()






import streamlit as st
st.set_page_config(layout='wide') 
# 初始化会话状态，用于存储列容器

def getTradeObsDateList(firstobs_date,end_date):
    obs_list=rqd.get_trading_dates(firstobs_date,end_date)
    obs_list=[str(d) for d in obs_list]
    price_list=[""]*len(obs_list)
    tradeObsDateList=[{'obsDate':obs,'price':p}for obs,p in zip(obs_list,price_list)]
    return tradeObsDateList





def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code

def getResult(underlyingCode,opttype,vol,barrier,strike,barrier_ramp,strike_ramp,s_0
            ,lev_daily,lev_exp
            ,expired_date,trade_date,tradeObsDateList
            ,coupon,rebate,parameterX):
        
            payload=json.dumps({
                        "tradeType": "singleLeg",
                        "openOrClose": "open",
                        "optionCombType": "",
                        "sort": "1",
                     	"quoteList": [{
                          		# "tradeTime":tradeTime
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
                     			"tradeObsDateList":tradeObsDateList
                                  }]
                                      
        
            })
            result=requests.post(api.URL+api.ENDPOINT['Pricing'],headers=api.headers
                ,data=payload).json()
            return result['data'][0]
        

style = """
<style>
div.row-widget.stRadio > div{flex-direction:row;}
</style>
"""

# 将 CSS 样式添加到 Streamlit 应用中

st.markdown(style, unsafe_allow_html=True)
opttyp_list=["累购","累沽","固定赔付累购","固定赔付累沽","熔断累购","熔断累沽","熔断增强累购","熔断增强累沽","熔断固陪累购","熔断固陪累沽"]
optionType_map={"累购":"AICallAccPricer"
            ,"累沽":"AIPutAccPricer"
            ,"固定赔付累购":"AICallFixAccPricer"
            ,"固定赔付累沽":"AIPutFixAccPricer"
            ,"熔断固陪累购":"AICallFixKOAccPricer"
            ,"熔断固陪累沽":"AIPutFixKOAccPricer"
            ,"熔断累购":"AICallKOAccPricer"
            ,"熔断累沽":"AIPutKOAccPricer"
            ,"熔断增强累购":"AIEnCallKOAccPricer"
            ,"熔断增强累沽":"AIEnPutKOAccPricer"
           }
optionType_map_self={"累购":"acccall"
            ,"累沽":"accput"
            ,"固定赔付累购":"fpcall"
            ,"固定赔付累沽":"fpput"
            ,"熔断固陪累购":"bfpcall"
            ,"熔断固陪累沽":"bfpput"
            ,"熔断累购":"bacccall"
            ,"熔断累沽":"baccput"
            ,"熔断增强累购":"bacccallplus"
            ,"熔断增强累沽":"baccputplus"
           }


def getParams(col_idx,trade_date):
        col_1,col_2 =st.columns(2)
        with col_1:
            underlyingCode = st.text_input("标的合约",value="",key="uc_"+str(col_idx))
        with col_2:
            try:
                default_v=rqd.current_snapshot(getRQcode(underlyingCode)).last
                # if st.session_state.columns[col_idx]['s_0']
                s_0=st.text_input("入场价格",key="s_0"+str(col_idx))
                # s_0=st.text_input("入场价格",key="s_"+str(col_idx))
                s_0=default_v if s_0=="" else float(s_0)
            except:
                s_0=st.text_input("入场价格",value="",key="s_"+str(col_idx))
                
            
        col_1,col_2,col_3 =st.columns([2,1,2])
        with col_1:
            # trade_date=st.date_input("成交日期:",value=rqd.get_future_latest_trading_date(),key=f"td_+{col_idx}")
            firstobs_date=st.date_input("起始观察日期:",value=trade_date,key=f"fd_+{col_idx}")
        with col_2:
            bias_date=st.text_input("时间偏移",value="",key=f"bd_+{col_idx}")
        with col_3:
            end_date=st.date_input("到期日期:",value=rqd.get_next_trading_date(trade_date,23) if bias_date=="" else rqd.get_next_trading_date(firstobs_date,int(bias_date)-1),key=f"ed_+{col_idx}") 

            # firstobs_date=st.date_input("起始观察日期:",value=trade_date,key=f"fd_+{col_idx}")
        # col_1,col_2 =st.columns([1,3])
        # with col_1:
        #     bias_date=st.text_input("时间偏移",value="",key=f"bd_+{col_idx}")
        # with col_2:
        #     end_date=st.date_input("到期日期:",value=rqd.get_next_trading_date(trade_date,23) if bias_date=="" else rqd.get_next_trading_date(trade_date,int(bias_date)-1),key=f"ed_+{col_idx}") 
        
        tradeObsDateList=getTradeObsDateList(firstobs_date,end_date)
        end_date=str(end_date)
        firstobs_date=str(firstobs_date)
        trade_date=str(trade_date)
        opttype = st.selectbox('期权类型：', opttyp_list,key=f"opt_+{col_idx}")

        col_1,col_2,col_3,col_4 =st.columns([1,2,1,2])
        with col_1:
            bias_k=st.text_input("Bias",placeholder="偏移",value="",key=f"bk_+{col_idx}")
        with col_2:
            strike=st.text_input("行权价:",value=st.session_state.columns[col_idx]['params_input']['strike'] if bias_k=="" else s_0+float(bias_k),key=f"k_+{col_idx}")
            # strike=st.text_input("行权价:",value="" if bias_k=="" else s_0+float(bias_k),key=f"k_+{col_idx}")

        strike=strike if strike=="" else float(strike)
        with col_3:
            bias_b=st.text_input("Bias",placeholder="偏移",value="",key=f"bb_+{col_idx}")
        with col_4:    
            barrier=st.text_input("敲出价:",value=st.session_state.columns[col_idx]['params_input']['barrier'] if bias_b=="" else s_0+float(bias_b),key=f"b_+{col_idx}") 
        barrier=barrier if barrier=="" else float(barrier)



        # col_1,col_2 =st.columns([1,3])
        # with col_1:
        #     bias_k=st.text_input("",placeholder="偏移",value="",key=f"bk_+{col_idx}")
        # with col_2:
        #     strike=st.text_input("行权价:",value=st.session_state.columns[col_idx]['params_input']['strike'] if bias_k=="" else s_0+float(bias_k),key=f"k_+{col_idx}")
        # strike=strike if strike=="" else float(strike)

        
        # col_1,col_2 =st.columns([1,3])
        # with col_1:
        #     bias_b=st.text_input("",placeholder="偏移",value="",key=f"bb_+{col_idx}")
        # with col_2:    
        #     barrier=st.text_input("敲出价:",value=st.session_state.columns[col_idx]['params_input']['barrier'] if bias_b=="" else s_0+float(bias_b),key=f"b_+{col_idx}") 
        # barrier=barrier if barrier=="" else float(barrier)
        col_1,col_2,col_3,col_4 =st.columns(4)
        with col_1:
            coupon=st.text_input("coupon",key=f"cp_+{col_idx}")
        with col_2:
            rebate=st.text_input("rebate",key=f"reb_+{col_idx}")
        with col_3:
            lev_list=st.text_input("lev",value="2,0",key=f"lev_+{col_idx}")
            lev_daily,lev_exp=[float(l)for l in lev_list.split(",")]
        with col_4:
            ramps=st.text_input("ramp",value="0",key=f"rp_+{col_idx}")
        strike_ramp,barrier_ramp=[float(ramps),float(ramps)]
        
        col_1,col_2,col_3 =st.columns(3)
        with col_1:
            vol=float(st.text_input("vol",value=0,key=f"v_+{col_idx}"))
        with col_2:
            mult_flag=st.selectbox("mult_flag",options=[True,False],key="mult_flag"+str(col_idx))
        with col_3:
            mult=float(st.text_input("mult",value=0.1,key=f"mult_+{col_idx}"))
        try:
            if "熔断" in opttype:
                rebate=abs(float(barrier)-s_0) if rebate=="" else rebate
        except:
                rebate=""
        
        return underlyingCode,opttype,vol,barrier,strike,barrier_ramp,strike_ramp,s_0,lev_daily,lev_exp,end_date,trade_date,tradeObsDateList,coupon,rebate,mult_flag,mult

def getinterval_self(i):
    # st.write(st.session_state.columns[i]['params_input']['underlyingCode'])
    # st.write(len(st.session_state.columns[i]['params_input']['underlyingCode']))
    # st.write(getRQcode(st.session_state.columns[i]['params_input']['underlyingCode']))
    params={'flag_Inter':1
             ,'underlyingCode':st.session_state.columns[i]['params_input']['underlyingCode']
             ,'flag':optionType_map_self[st.session_state.columns[i]['params_input']['opttype']]
             ,'direction':'B'
             ,'s_t':st.session_state.columns[i]['params_input']['s_0']
             ,'s_0':st.session_state.columns[i]['params_input']['s_0']
             ,'rng_k':""
             ,'rng_b':""
             ,'strike':st.session_state.columns[i]['params_input']['strike']
             ,'barrier':st.session_state.columns[i]['params_input']['barrier']
             ,'rebate':float(st.session_state.columns[i]['params_input']['rebate']) if len(st.session_state.columns[i]['params_input']['rebate'])>0 else ""
             ,'coupon':float(st.session_state.columns[i]['params_input']['coupon']) if len(st.session_state.columns[i]['params_input']['coupon'])>0 else st.session_state.columns[i]['params_input']['coupon']
             ,'first_obsdate':st.session_state.columns[i]['params_input']['tradeObsDateList'][0]["obsDate"]
             ,'expired_date':st.session_state.columns[i]['params_input']['end_date']
             ,'obs_day':""
             ,'daily_amt':1
             ,'leverage':st.session_state.columns[i]['params_input']['lev_daily']
             ,'leverage_expire':st.session_state.columns[i]['params_input']['lev_exp']
             ,'isCashsettle':0
             ,'strike_ramp':st.session_state.columns[i]['params_input']['strike_ramp']
             ,'barrier_ramp':st.session_state.columns[i]['params_input']['barrier_ramp']
             ,'const_sgm':st.session_state.columns[i]['params_input']['vol']/100
             ,'mult_flag':st.session_state.columns[i]['params_input']['mult_flag']
             }
    # st.write(st.session_state.columns[i]['params_input'])
    # st.write(params)
    ca=CalAcc(underlyingCode=params['underlyingCode']
                  , flag=params['flag'], direction=params['direction']
                  , s_t=params['s_t'], s_0=params['s_0']
                  , rng_k=params['rng_k'],rng_b=params['rng_b']
                  , strike=params['strike'], barrier=params['barrier']
                  , rebate=params['rebate'],coupon=params['coupon']
                  , pricing_time=datetime.now()
                  , trade_date=st.session_state.columns[i]['params_input']['trade_date']
                  , first_obsdate=params['first_obsdate']
                  , expired_date=params['expired_date']
                  , obs_days=params['obs_day']
                  , daily_amt=params['daily_amt']
                  , leverage=params['leverage']
                  , leverage_expire=params['leverage_expire']
                  , isCashsettle=params['isCashsettle']
                  , strike_ramp=params['strike_ramp'],barrier_ramp=params['barrier_ramp']
                  , const_sgm=params['const_sgm']
                  , mult_tag=params['mult_flag'])
    # st.write(ca._CalAcc__getVolsurfacejson())
    res=ca.getRes(flag_Inter=1, flag_Coupon=0,mult=st.session_state.columns[i]['params_input']['mult'])
    # res=""
    return getRESmds(i,res)

def getPayment_self(i):
    # st.write(st.session_state.columns[i]['params_input']['underlyingCode'])
    # st.write(len(st.session_state.columns[i]['params_input']['underlyingCode']))
    # st.write(getRQcode(st.session_state.columns[i]['params_input']['underlyingCode']))
    params={'flag_Inter':0
             ,'underlyingCode':st.session_state.columns[i]['params_input']['underlyingCode']
             ,'flag':optionType_map_self[st.session_state.columns[i]['params_input']['opttype']]
             ,'direction':'B'
             ,'s_t':st.session_state.columns[i]['params_input']['s_0']
             ,'s_0':st.session_state.columns[i]['params_input']['s_0']
             ,'rng_k':""
             ,'rng_b':""
             ,'strike':st.session_state.columns[i]['params_input']['strike']
             ,'barrier':st.session_state.columns[i]['params_input']['barrier']
             ,'rebate':float(st.session_state.columns[i]['params_input']['rebate']) if len(st.session_state.columns[i]['params_input']['rebate'])>0 else ""
             ,'coupon':float(st.session_state.columns[i]['params_input']['coupon']) if len(st.session_state.columns[i]['params_input']['coupon'])>0 else st.session_state.columns[i]['params_input']['coupon']
             ,'first_obsdate':st.session_state.columns[i]['params_input']['tradeObsDateList'][0]["obsDate"]
             ,'expired_date':st.session_state.columns[i]['params_input']['end_date']
             ,'obs_day':""
             ,'daily_amt':1
             ,'leverage':st.session_state.columns[i]['params_input']['lev_daily']
             ,'leverage_expire':st.session_state.columns[i]['params_input']['lev_exp']
             ,'isCashsettle':0
             ,'strike_ramp':st.session_state.columns[i]['params_input']['strike_ramp']
             ,'barrier_ramp':st.session_state.columns[i]['params_input']['barrier_ramp']
             ,'const_sgm':st.session_state.columns[i]['params_input']['vol']/100
             ,'mult_flag':st.session_state.columns[i]['params_input']['mult_flag']
             }
    # st.write(st.session_state.columns[i]['params_input'])
    # st.write(params)
    ca=CalAcc(underlyingCode=params['underlyingCode']
                  , flag=params['flag'], direction=params['direction']
                  , s_t=params['s_t'], s_0=params['s_0']
                  , rng_k=params['rng_k'],rng_b=params['rng_b']
                  , strike=params['strike'], barrier=params['barrier']
                  , rebate=params['rebate'],coupon=params['coupon']
                  , pricing_time=datetime.now()
                  , trade_date=st.session_state.columns[i]['params_input']['trade_date']
                  , first_obsdate=params['first_obsdate']
                  , expired_date=params['expired_date']
                  , obs_days=params['obs_day']
                  , daily_amt=params['daily_amt']
                  , leverage=params['leverage']
                  , leverage_expire=params['leverage_expire']
                  , isCashsettle=params['isCashsettle']
                  , strike_ramp=params['strike_ramp'],barrier_ramp=params['barrier_ramp']
                  , const_sgm=params['const_sgm']
                  , mult_tag=params['mult_flag'])
    # st.write(ca._CalAcc__getVolsurfacejson())
    res=ca.getRes(flag_Inter=0, flag_Coupon=1,mult=st.session_state.columns[i]['params_input']['mult'])
    # res=""
    return getRESmds(i,res)


def getinterval(i):
    res=getResult(st.session_state.columns[i]['params_input']['underlyingCode']
                  ,optionType_map[st.session_state.columns[i]['params_input']['opttype']]
              ,st.session_state.columns[i]['params_input']['vol']
              ,st.session_state.columns[i]['params_input']['barrier']
              ,st.session_state.columns[i]['params_input']['strike']
              ,st.session_state.columns[i]['params_input']['barrier_ramp']
              ,st.session_state.columns[i]['params_input']['strike_ramp']
              ,st.session_state.columns[i]['params_input']['s_0']
              ,st.session_state.columns[i]['params_input']['lev_daily']
              ,st.session_state.columns[i]['params_input']['lev_exp']
              ,st.session_state.columns[i]['params_input']['end_date']
              ,st.session_state.columns[i]['params_input']['trade_date']
              ,st.session_state.columns[i]['params_input']['tradeObsDateList']
              ,st.session_state.columns[i]['params_input']['coupon']
              ,st.session_state.columns[i]['params_input']['rebate']
              ,"interval")
    margin=res['margin']
    # st.write(st.session_state.columns[i]['params_input'])
    strike=st.session_state.columns[i]['params_input']['s_0']+res['solverValue'] if "沽" in st.session_state.columns[i]['params_input']['opttype'] else st.session_state.columns[i]['params_input']['s_0']-res['solverValue']
    barrier=st.session_state.columns[i]['params_input']['s_0']-res['solverValue'] if "沽" in st.session_state.columns[i]['params_input']['opttype'] else st.session_state.columns[i]['params_input']['s_0']+res['solverValue']
    
    # st.session_state.columns[i]['params_input']['strike']=strike
    # st.session_state.columns[i]['params_input']['barrier']=barrier
    # strike=str(st.session_state.columns[i]['params_input']['s_0']+res['solverValue'])+"(+"+str(res['solverValue'])+")" if "沽" in st.session_state.columns[i]['params_input']['opttype'] else str(st.session_state.columns[i]['params_input']['s_0']-res['solverValue'])+"(-"+str(res['solverValue'])+")"
    # barrier=str(st.session_state.columns[i]['params_input']['s_0']-res['solverValue'])+"(-"+str(res['solverValue'])+")" if "沽" in st.session_state.columns[i]['params_input']['opttype'] else str(st.session_state.columns[i]['params_input']['s_0']+res['solverValue'])+"(+"+str(res['solverValue'])+")"
    # st.session_state['k_'+str(i)].placeholder=str(strike)
    
    return getmds(i,strike,barrier,margin,res)

def getstrike(i):
    res=getResult(st.session_state.columns[i]['params_input']['underlyingCode']
                  ,optionType_map[st.session_state.columns[i]['params_input']['opttype']]
              ,st.session_state.columns[i]['params_input']['vol']
              ,st.session_state.columns[i]['params_input']['barrier']
              ,st.session_state.columns[i]['params_input']['strike']
              ,st.session_state.columns[i]['params_input']['barrier_ramp']
              ,st.session_state.columns[i]['params_input']['strike_ramp']
              ,st.session_state.columns[i]['params_input']['s_0']
              ,st.session_state.columns[i]['params_input']['lev_daily']
              ,st.session_state.columns[i]['params_input']['lev_exp']
              ,st.session_state.columns[i]['params_input']['end_date']
              ,st.session_state.columns[i]['params_input']['trade_date']
              ,st.session_state.columns[i]['params_input']['tradeObsDateList']
              ,st.session_state.columns[i]['params_input']['coupon']
              ,st.session_state.columns[i]['params_input']['rebate']
              ,"strike")
    margin=res['margin']
    # s_0=st.session_state.columns[i]['params_input']['s_0']
    barrier=st.session_state.columns[i]['params_input']['barrier']
    # opttype=st.session_state.columns[i]['params_input']['opttype']
    strike=res['solverValue']
    # st.session_state.columns[i]['params_input']['strike']=res['solverValue']
    # strike=str(res['solverValue'])+"(+"+str(res['solverValue']-s_0)+")" if "沽" in opttype else str(res['solverValue'])+"(-"+str(s_0-res['solverValue'])+")"
    # try:
        # barrier=str(barrier)+"(-"+str(s_0-float(barrier))+")" if "沽" in opttype else str(barrier)+"(+"+str(float(barrier)-s_0)+")"
    # except:
    #     barrier="无"
    return getmds(i,strike,barrier,margin,res)

def getbarrier(i):
    # st.write(st.session_state.columns[i]['params_input'])
    res=getResult(st.session_state.columns[i]['params_input']['underlyingCode']
                  ,optionType_map[st.session_state.columns[i]['params_input']['opttype']]
              ,st.session_state.columns[i]['params_input']['vol']
              ,st.session_state.columns[i]['params_input']['barrier']
              ,st.session_state.columns[i]['params_input']['strike']
              ,st.session_state.columns[i]['params_input']['barrier_ramp']
              ,st.session_state.columns[i]['params_input']['strike_ramp']
              ,st.session_state.columns[i]['params_input']['s_0']
              ,st.session_state.columns[i]['params_input']['lev_daily']
              ,st.session_state.columns[i]['params_input']['lev_exp']
              ,st.session_state.columns[i]['params_input']['end_date']
              ,st.session_state.columns[i]['params_input']['trade_date']
              ,st.session_state.columns[i]['params_input']['tradeObsDateList']
              ,st.session_state.columns[i]['params_input']['coupon']
              ,st.session_state.columns[i]['params_input']['rebate']
              ,"barrier")
    margin=res['margin']
    # s_0=st.session_state.columns[i]['params_input']['s_0']
    strike=st.session_state.columns[i]['params_input']['strike']
    # opttype=st.session_state.columns[i]['params_input']['opttype']
    barrier=res['solverValue']
    # st.session_state.columns[i]['params_input']['barrier']=barrier
    # strike=str(strike)+"(+"+str(float(strike)-s_0)+")" if "沽" in opttype else str(strike)+"(-"+str(s_0-float(strike))+")"
    # barrier=str(res['solverValue'])+"(-"+str(s_0-res['solverValue'])+")"  if "沽" in opttype else str(res['solverValue'])+"(+"+str(res['solverValue']-s_0)+")"
    return getmds(i,strike,barrier,margin,res)

def getpayment(i):
    # st.write(st.session_state.columns[i]['params_input'])
    res=getResult(st.session_state.columns[i]['params_input']['underlyingCode']
                  ,optionType_map[st.session_state.columns[i]['params_input']['opttype']]
              ,st.session_state.columns[i]['params_input']['vol']
              ,st.session_state.columns[i]['params_input']['barrier']
              ,st.session_state.columns[i]['params_input']['strike']
              ,st.session_state.columns[i]['params_input']['barrier_ramp']
              ,st.session_state.columns[i]['params_input']['strike_ramp']
              ,st.session_state.columns[i]['params_input']['s_0']
              ,st.session_state.columns[i]['params_input']['lev_daily']
              ,st.session_state.columns[i]['params_input']['lev_exp']
              ,st.session_state.columns[i]['params_input']['end_date']
              ,st.session_state.columns[i]['params_input']['trade_date']
              ,st.session_state.columns[i]['params_input']['tradeObsDateList']
              ,st.session_state.columns[i]['params_input']['coupon']
              ,st.session_state.columns[i]['params_input']['rebate']
              ,"payment")
    
    margin=res['margin']

    strike=st.session_state.columns[i]['params_input']['strike']
    barrier=st.session_state.columns[i]['params_input']['barrier']
    st.session_state.columns[i]['params_input']['coupon']=res['solverValue']
    # st.write(res)
    return getmds(i,strike,barrier,margin,res)
    

def getmds(i,strike,barrier,margin,res):
    if "熔断增强累沽" in st.session_state.columns[i]['params_input']['opttype']:
        rebate="入场价-收盘价"
    elif "熔断增强累购" in st.session_state.columns[i]['params_input']['opttype']:
        rebate="收盘价-入场价"
    elif "熔断" not in st.session_state.columns[i]['params_input']['opttype']:
        rebate="无"
    else:
        rebate=abs(float(barrier)-float(st.session_state.columns[i]['params_input']['s_0'])) if st.session_state.columns[i]['params_input']['rebate']=="" else st.session_state.columns[i]['params_input']['rebate']

        
    if "固" in st.session_state.columns[i]['params_input']['opttype']:
        coupon=st.session_state.columns[i]['params_input']['coupon']
        # coupon=res['solverValue']
    # elif "熔断累沽" in st.session_state.columns[i]['params_input']['opttype']:
    #     coupon="行权价-收盘价"
    # else :
    #     coupon="收盘价-行权价"
    else:
        coupon="线性"

    
    strike=str(strike)+" ("+str(round(strike)-st.session_state.columns[i]['params_input']['s_0'])+")"
    try:
        barrier=str(barrier)+" ("+str(barrier-st.session_state.columns[i]['params_input']['s_0'])+")"
    except:
        barrier="无"
        
    markdown_str = "####  ===== 报价：=====\n"
    markdown_str += f"##### 成交日期: {st.session_state.columns[i]['params_input']['trade_date']}\n"
    markdown_str += f"##### 到期日期: {st.session_state.columns[i]['params_input']['end_date']}\n"
    markdown_str += f"##### 标的合约: {st.session_state.columns[i]['params_input']['underlyingCode'].upper()}\n"
    markdown_str += f"##### 入场价格: {st.session_state.columns[i]['params_input']['s_0']}\n"
    markdown_str += f"##### 期权类型: {st.session_state.columns[i]['params_input']['opttype']}\n"
    markdown_str += f"##### 行权价格: {strike}\n"
    markdown_str += f"##### 敲出价格: {barrier}\n"
    markdown_str += f"##### 敲出赔付: {rebate}\n"
    markdown_str += f"##### 区间赔付:  {coupon}\n"
    markdown_str += f"##### 起始观察日:  {st.session_state.columns[i]['params_input']['tradeObsDateList'][0]['obsDate']}\n"
    markdown_str += f"##### 观察次数:  {len(st.session_state.columns[i]['params_input']['tradeObsDateList'])}\n"
    markdown_str += f"##### 杠杆系数:  {st.session_state.columns[i]['params_input']['lev_daily']}\n"
    markdown_str += f"##### 期末额外倍数:  {st.session_state.columns[i]['params_input']['lev_exp']}\n"
    markdown_str += f"##### 期权价格: {0}\n"
    markdown_str += f"##### 预估保证金:  {margin}\n"
    
    # st.write("theta:"+res['theta'])
    # markdown_res=""\
    markdown_str += " #### ================\n"
    markdown_str += f" ###### optionPremium:  {res['optionPremium']}\n"
    markdown_str += f" ###### delta:  {res['delta']}\n"
    markdown_str += f" ###### day1PnL:  {res['day1PnL']}\n"
    markdown_str += f" ###### theta:  {res['theta']}\n"
    markdown_str += f" ###### vega:  {res['vega']}\n"
    
    # st.write(res),markdown_res
    return markdown_str

def getRESmds(i,res):
    
    # st.session_state.columns[i]['params_input']['strike']=float(res.loc['行权价格','要素'].split(" (")[0])
    # st.session_state.columns[i]['params_input']['barrier']="" if res.loc['敲出价格','要素']=="无" else float(res.loc['敲出价格','要素'].split(" (")[0])
    # st.session_state.columns[i]['params_input']['rebate']="" if res.loc['敲出赔付','要素']=="无" else float(res.loc['敲出赔付','要素'])
    # try:
    #     st.session_state.columns[i]['params_input']['coupon']=float(res.loc['区间赔付','要素'])
    # except:
    #     st.session_state.columns[i]['params_input']['coupon']=""
    
    markdown_str = f"####  ===== 报价 {i+1}：=====\n"
    markdown_str += f"##### 成交日期: {res.loc['交易日期','要素']}\n"
    markdown_str += f"##### 到期日期: {res.loc['到期日期','要素']}\n"
    markdown_str += f"##### 标的合约: {res.loc['标的合约','要素']}\n"
    markdown_str += f"##### 入场价格: {res.loc['入场价格','要素']}\n"
    markdown_str += f"##### 期权类型: {res.loc['期权类型','要素']}\n"
    markdown_str += f"##### 行权价格: {res.loc['行权价格','要素']}\n"
    markdown_str += f"##### 敲出价格: {res.loc['敲出价格','要素']}\n"
    markdown_str += f"##### 敲出赔付: {res.loc['敲出赔付','要素']}\n"
    markdown_str += f"##### 区间赔付:  {res.loc['区间赔付','要素']}\n"
    markdown_str += f"##### 起始观察日:  {res.loc['起始观察日','要素']}\n"
    markdown_str += f"##### 观察次数:  {res.loc['观察次数','要素']}\n"
    markdown_str += f"##### 杠杆系数:  {res.loc['杠杆系数','要素']}\n"
    markdown_str += f"##### 期末额外倍数:  {res.loc['期末额外倍数','要素']}\n"
    markdown_str += f"##### 期权价格: {0}\n"
    markdown_str += f"##### 预估保证金:  {res.loc['预估保证金(1单位/天)','要素']}\n"
    # st.write(res)
    return markdown_str



if 'columns' not in st.session_state:
    st.session_state['columns'] = []
    st.session_state['col_num'] = 0


keys="underlyingCode,opttype,vol,barrier,strike,barrier_ramp,strike_ramp,s_0,lev_daily,lev_exp,end_date,trade_date,tradeObsDateList,coupon,rebate,mult_flag,mult".split(",")
if datetime.now().time()>time(15,0,0) and datetime.now().time()<time(21,0,0):
    trade_date=st.date_input("成交日期:",value=rqd.get_next_trading_date(datetime.now().date(),1))
else:
    trade_date=st.date_input("成交日期:",value=rqd.get_future_latest_trading_date())

if st.button('添加列容器'):
    # 每次点击按钮，就创建一个新的列容器
    col=st.columns(1)[0]
    st.session_state.columns.append({'container': col
                                     ,'params_input': dict(zip(keys,[""]*len(keys)))
                                     ,'output':""})
    st.session_state['col_num']+=1
    # if len(st.session_state.columns) >0:
    #         for nums in range(len(st.session_state.columns)):
    #             st.write(st.session_state.columns[nums]['output'])


if st.button("删除此列容器"):
    st.session_state.columns.pop(st.session_state['col_num']-1)
    # st.write(st.session_state)
    st.session_state['col_num'] -= 1 
    

for i, col in enumerate(st.columns(st.session_state['col_num'])):
    with col:
        # st.write(i)
        params_list=getParams(i,trade_date)
        st.session_state.columns[i]['params_input']=dict(zip(keys, params_list))
        # st.write(params_list)
        # st.write(st.session_state.columns[i]['params_input'])
        col11, col12, col13, col14,col15,col16 = st.columns(6)
        col11.button('Int', key="button_Int_"+str(i))
        col12.button('K', key="button_K_"+str(i))
        col13.button('B', key="button_B_"+str(i))
        col14.button('P', key="button_P_"+str(i))
        col15.button('Int_Self', key="button_Int_self_"+str(i))
        col16.button('Pay_Self', key="button_Pay_self_"+str(i))


        # if st.session_state.columns[i]['output']:
        #     st.markdown(st.session_state.columns[i]['output'])

        # st.write(st.session_state.columns[i]['output'])
        # st.write(st.session_state.columns[i]['params_input'])
        # if st.button("Int",key="button_Int_"+str(i)):
        
        if st.session_state["button_Int_"+str(i)]:
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getinterval(i)
            st.session_state.columns[i]['output']=mds
        # if st.button("K",key="button_K_"+str(i)):
        if st.session_state["button_K_"+str(i)]:
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getstrike(i)
            st.session_state.columns[i]['output']=mds
        # if st.button("B",key="button_B_"+str(i)):
        if st.session_state["button_B_"+str(i)]:
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getbarrier(i)
            st.session_state.columns[i]['output']=mds
        # if st.button("P",key="button_P_"+str(i)):
        if st.session_state["button_P_"+str(i)]:
            # st.write('here!!!')
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getpayment(i)
            st.session_state.columns[i]['output']=mds
        # if st.button("Int_Self",key="button_Int_self_"+str(i)):
        if st.session_state["button_Int_self_"+str(i)]:
        
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getinterval_self(i)
            st.session_state.columns[i]['output']=mds
        if st.session_state["button_Pay_self_"+str(i)]:
        
            # st.write(st.session_state.columns[i]['params_input'])
            mds=getPayment_self(i)
            st.session_state.columns[i]['output']=mds

            # st.write(st.session_state.columns[i]['params_input'])
        # else:
        #     pass
        # st.session_state.columns[i]['output']=mds
        # st.write(st.session_state.columns[i]['output'])
        if st.session_state.columns[i]['output']:
            st.markdown(st.session_state.columns[i]['output'])
  
            # st.write(st.session_state.columns[i]['params_input'])
            # st.session_state.columns[i]['output']=markdown_str
        # getButtons(i, st.session_state.columns[i]['params_input'])
        # st.write(st.session_state.columns[i])
        # 更新会话状态中的列内容
#         # st.session_state.columns_content[i] = st.session_state.columns_content[i]['params_input']
#         # 显示输入的内容
#         # st.write(f"第 {i + 1} 列的输出:", user_input)

#     # st.write(f"添加第 {col_num} 列容器")
#     # col = st.columns(1)[0]
#     # st.session_state['columns'].append({'container': col, 'params_input':{}, 'output': ""})
#     # # st.write(st.session_state)
#     # # st.write(st.session_state['col_num']['params_input']=getParams())
#     # # =
    

#     # st.session_state['col_num'] += 1     
#     # # st.session_state['columns'][st.session_state['col_num']-1]['params_input']=getParams(st.session_state['col_num']-1)
#     # st.write(st.session_state)
    

    # st.write(st.session_state['col_num'])
    
  
    
        # bias_date=st.text_input("时间偏移",value="")
        # end_date=st.date_input("到期日期:",value=rqd.get_next_trading_date(trade_date,23) if bias_date=="" else rqd.get_next_trading_date(trade_date,int(bias_date)-1)) 
        # tradeObsDateList=getTradeObsDateList(firstobs_date,end_date)
        
        
        # bias_k=st.text_input("行权价偏移",value="")
        # strike=st.text_input("行权价:",value="" if bias_k=="" else s_0+float(bias_k)) 
        # bias_b=st.text_input("敲出价偏移",value="")
        # barrier=st.text_input("敲出价:",value="" if bias_b=="" else s_0+float(bias_b)) 
        # lev_list=st.text_input("lev",value="2,0")
        # lev_daily,lev_exp=[float(l)for l in lev_list.split(",")]
        # ramp_list=st.text_input("ramp",value="0,0")
        # strike_ramp,barrier_ramp=[float(r)for r in ramp_list.split(",")]
        # coupon=st.text_input("coupon")
        # rebate=st.text_input("rebate")
        # opttype = st.selectbox('期权类型：', opttyp_list, index=0)
        # vol=st.text_input("vol")
        # if "熔断" in opttype:
        #     rebate=str(abs(float(barrier)-s_0)) if rebate=="" else rebate
        # else:
        #     rebate=""
# # 初始化会话状态变量，用于存储区块的状态
# if 'blocks' not in st.session_state:
#     st.session_state.blocks = [
#         {'input_value': '', 'output_value': '', 'active': False},
#         # 可以根据需要初始化更多的区块
#     ]

# # 定义一个函数来执行计算
# def calculate(block_idx):
#     # 这里添加你的计算逻辑
#     # 假设计算结果是输入值的反转
#     input_value = st.session_state.blocks[block_idx]['input_value']
#     result = input_value[::-1]
#     st.session_state.blocks[block_idx]['output_value'] = result
#     st.session_state.blocks[block_idx]['active'] = True

# # 定义一个函数来显示区块
# def display_block(block_idx):
#     st.text_input(f"区块 {block_idx + 1} 输入", value=st.session_state.blocks[block_idx]['input_value'], key=f"input_{block_idx}")
#     if st.button(f"计算区块 {block_idx + 1}"):
#         calculate(block_idx)
#     if st.session_state.blocks[block_idx]['active']:
#         st.write(f"区块 {block_idx + 1} 输出: ", st.session_state.blocks[block_idx]['output_value'])

# # 显示和处理区块
# for idx in range(len(st.session_state.blocks)):
#     with st.container():
#         display_block(idx)

# # 添加区块的按钮
# if st.button('添加区块'):
#     st.session_state.blocks.append({'input_value': '', 'output_value': '', 'active': False })

# # 删除区块的按钮
# if st.button('删除区块'):
#     if len(st.session_state.blocks) > 1:
#         st.session_state.blocks.pop()
       
    
# for col_idx, col_info in enumerate(st.session_state['columns']):
#     with col_info['container']:
#         st.write(f"这是第 {col_idx + 1} 列的内容")
        # params_input = st.text_input(f"第 {col_idx + 1} 列的输入", value=col_info['params_input'])
        # st.session_state['columns'][col_idx]['params_input'] = 
        # getParams()
        # st.write(f"第 {col_idx + 1} 列的输出：{col_info['output']}")
    
# col1= st.columns(1)
# with col1[0]:
# def getParams(col_idx):
#     st.session_state['columns'][col_idx]['params_input']={
#         'underylyingCode':st.text_input("标的合约")
#         ,'s_0':st.text_input("入场价格")
#         }
    # underlyingCode = st.text_input("标的合约")
    # s_0=st.text_input("入场价格")
    # s_0=rqd.current_snapshot(getRQcode(underlyingCode)).last if s_0=="" else float(s_0)
    # trade_date=st.date_input("成交日期:",value=rqd.get_future_latest_trading_date())
    # firstobs_date=st.date_input("起始观察日期:",value=trade_date)
    # bias_date=st.text_input("时间偏移",value="")
    # end_date=st.date_input("到期日期:",value=rqd.get_next_trading_date(trade_date,23) if bias_date=="" else rqd.get_next_trading_date(trade_date,int(bias_date)-1)) 
    # tradeObsDateList=getTradeObsDateList(firstobs_date,end_date)
    
    
    # bias_k=st.text_input("行权价偏移",value="")
    # strike=st.text_input("行权价:",value="" if bias_k=="" else s_0+float(bias_k)) 
    # bias_b=st.text_input("敲出价偏移",value="")
    # barrier=st.text_input("敲出价:",value="" if bias_b=="" else s_0+float(bias_b)) 
    # lev_list=st.text_input("lev",value="2,0")
    # lev_daily,lev_exp=[float(l)for l in lev_list.split(",")]
    # ramp_list=st.text_input("ramp",value="0,0")
    # strike_ramp,barrier_ramp=[float(r)for r in ramp_list.split(",")]
    # coupon=st.text_input("coupon")
    # rebate=st.text_input("rebate")
    # opttype = st.selectbox('期权类型：', opttyp_list, index=0)
    # vol=st.text_input("vol")
    # if "熔断" in opttype:
    #     rebate=str(abs(float(barrier)-s_0)) if rebate=="" else rebate
    # else:
    #     rebate=""

    
