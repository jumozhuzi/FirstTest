# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:29:14 2024

@author: dzrh
"""

# from Analyse import *
import pandas as pd
import numpy as np
# from scipy.stats import norm
from matplotlib import pyplot as plt
# import tushare as tu
from datetime import timedelta,datetime,time
from CYL.OptionPricing import BSM,calIV,calTradttm,LinearInterpVol
# from CYL.AccCalculator import CalAcc
from CYL.StressTestNew_api import BarrierReport
from CYL.OTCAPI import SelfAPI
# from CYL.YieldChainAPI import YieldChainAPI
import rqdatac as rqd
# rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
import streamlit as stm
import streamlit_searchbox as stm_box
import re
# import time as sys_time
rf=0.03
q=0
annual_coeff=252
api=SelfAPI()
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

def getRule():
    loc=r'D:\chengyilin\work\system\OptionStrikeRule.xlsx'
    dfrule=pd.read_excel(loc,index_col=0)
    dfrule.insert(loc=3,column='Level4',value=dfrule['Level3']*5)
    dfrule['Level3'].fillna(5*dfrule['Level2'],inplace=True)
    dfrule['Level4'].fillna(dfrule['Level3'],inplace=True)
    dfrule['Dk4'].fillna(dfrule['Dk3'],inplace=True)
    
    # dfrule.insert(loc=2,column='Level3',value=dfrule['Level2']*5)
  
    return dfrule

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

def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code

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

def hl_k(k):
    if k/findLatestPrice(underlyingCode)>=1:
        return 'background-color: tomato'
    else:
        return'background-color: yellowgreen'
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
     
def getfloat(s): 
    try:
        return float(s)
    except:
        return s
      
def formatParams(inter_params):
    try:
        return inter_params.split(",")
    except:
        return inter_params
    

clientName=api._SelfAPI__getClientList()
if "clientnames" not in stm.session_state:
    stm.session_state["clientnames"]=clientName
# stm.write(stm.session_state["clientnames"])


def search_function(query):
    # clientName=api.getClientList()
    # name_list=clientName.index.tolist()
    if query:
        pattern = re.compile('.*' + re.escape(query) + '.*', re.IGNORECASE)
        return [item for item in stm.session_state["clientnames"].keys() if pattern.match(item)]
    return "空"

if __name__=="__main__":
    
    stm.set_page_config(page_title="ITC IV",layout='wide') 
    tabs=stm.tabs(['Vol','ForwardDisplay'])
    
    with tabs[0]:
        # if stm.button('Refresh Tab 1'):
   
        if stm.button("Refresh"):
            stm.rerun
        else:
            pass
        col1, col2 = stm.columns([1.5,1])

        with col1:
            code_text=(stm.text_input("Choose Underlyings"
                                                    # , value="AU2408"
                                                    ))
            try:
                with stm.container():
                        underlyingCode=getRQcode(code_text)
                        iv_gre,fig_iv=RQcalIVandGreeks(underlyingCode,annual_coeff,rng=25)
                        iv_gre=iv_gre.round(4)
                        stm.header(underlyingCode+" IV and Greeks")
                        stm.write(iv_gre.style.applymap(hl_k,subset=iv_gre.columns[6]))
                        stm.pyplot(fig_iv)
            except:
                pass
         
        with col2:
                delta_str=stm.text_input("delta",value="50,-50")
                days_str=stm.text_input("days",value="22")
                contracts_list=stm.text_input("contracts",value="2412,2501").split(",")
                variety=stm.text_input("variety")
                underlyingCodes=[variety.upper()+contracts_list[0],variety.upper()+contracts_list[1]]
                try:
                    for g_d in delta_str.split(","):
                        for g_t in days_str.split(","):
                            g_d,g_t=float(g_d),int(g_t)
                            trading_contracts=list(map(lambda c:getRQcode(c),underlyingCodes))
                            res=RQcalBidAskVol(trading_contracts, g_d, g_t,annual_coeff)
                            stm.write("Delta: "+str(g_d)+" TradeDay: "+str(g_t))
                            stm.write(res.round(2))
                except:
                    pass
        
            
    
    with tabs[1]:
        # try:
            
        # varitey_list=stm.text_input("Variety select:",value='CF,MA,EG,EB,UR,V,PP,L,AG,AU,AL,SH').split(",")
        # if stm.button("CalBarrier"):
        #     api=SelfAPI()
        #     brt=BarrierReport(cur_trad_date
        #                       ,varitey_list
        #                       ,datetime.now())
        #     dfbar=brt.show()
        #     # if dfbar.Situation.
        #     stm.table(dfbar[dfbar.Situation=='Break'].groupby(['UnderlyingCode','ClientName','barrier','StructureType'])['delta'].sum()
        #               )
        #     # stm.button("CalBarrier",value=False)
        # else:
        #     pass
        
    # with tabs[2]:
        cur_trad_date=rqd.get_future_latest_trading_date()
        given_date=stm.date_input("Current Trade Date:",value=cur_trad_date)
        
        search_results=stm_box.st_searchbox(search_function,label='Search for client name')

        if search_results:
            client_name=search_results
            stm.write(client_name)
            variety_list=stm.text_input("variety list")
            
            underlying_list=stm.text_input("underlying list")
            variety_list=[] if variety_list=="" else [v.upper() for v in variety_list.split(",")]
            underlying_list=[] if underlying_list=="" else underlying_list.split(",")
            if stm.button("CalForward"):
    
                if given_date==cur_trad_date:
                    dffor=api.getOTC_LiveTrade(str(given_date),[client_name],variety_list,underlying_list,["f"])
                    dffor['ClientDirt']=np.where(dffor.buyOrSellName=="买入","空头","多头")

                else:
                    dffor=api.getTradeRisk(str(given_date),[client_name],underlying_list,["f"])
                    dffor['ClientDirt']=np.where(dffor.buyOrSellName=="买入","多头","空头")

                # dffor['ClientDirt']=np.where(dffor.buyOrSellName=="买入","空头","多头")
                dffor['Lots']=dffor.availableVolume/dffor.underlyingCode.apply(getRQcode).apply(findMultiplier)
                dffor['availableVolume']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.availableVolume
                dffor['Lots']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.Lots
                a=dffor.groupby(['underlyingCode',"ClientDirt",'strike'])[['availableVolume','Lots']].sum()
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
                aa=a.groupby(["标的合约","客户方向"]).sum()
                bb=pd.merge(b,aa,on=b.index)
                bb.rename(columns={"key_0":"key_1"
                                  }
                                 ,inplace=True)
                bb=pd.merge(bb[bb.columns[0]].apply(pd.Series),bb,on=bb.index)
                bb.drop(columns=['key_0','key_1'],inplace=True)
                bb.rename(columns={0:'标的合约',1:'客户方向'},inplace=True)
                # print(bb.groupby(["标的合约","客户方向"]).last())
                custom_style = """
                                <style>
                                table {
                                    font-family: Arial, sans-serif;
                                    font-size: 15px;
                                    border-collapse: collapse;
                                    width: 50%;
                                }
                                table, th, td {
                                    border: 1px solid black;
                                }
                                th {
                                    height: 20px;
                                    background-color: #f2f2f2;
                                }
                                td {
                                    text-align: center;
                                    padding: 8px;
                                }
                                </style>
                                """
                stm.markdown(custom_style,unsafe_allow_html=True)    
                stm.markdown("#### "+client_name+"远期明细 (截止"+str(given_date)+")：")          
                # a_md=a.to_markdown()
                stm.write(a)
                # stm.table(a)
                # stm.markdown(a_md,unsafe_allow_html=False)
                # print("")
                # print("")
                stm.markdown("##### 各标的多空合计：")
                stm.dataframe(bb.groupby(["标的合约","客户方向"]).last())
                # stm.table(bb.groupby(["标的合约","客户方向"]).last())
                # stm.session_state['client']['last']=client_name
                # stm.write(stm.session_state['client'])
                
            else:
                pass
    
            
        # except:
            # pass
    #     try:
    #         container_params=stm.container()
    #         container_show=stm.container()
    #         with container_params:
    #             col1,col2,col3,col4=stm.columns(4)
    #             with col1:
    #                 # trade_date=stm.date_input("Current Trade Date:",value=rqd.get_future_latest_trading_date())
    #                 trade_date=stm.text_input("Current Trade Date:",value=str(rqd.get_future_latest_trading_date()))
    #                 mult=stm.number_input("mult params:",value=0.5)
    #                 contracts_num=stm.number_input("contracts numbers:",value=1)
    #                 underlyingCode=stm.text_input("underlyingCode")
    #                 const_sgm=stm.text_input("const_sgm")
    #             with col2:
    #                 flag_Inter=stm.text_input("flag_Inter",value=1)
    #                 flag_Coupon=stm.text_input("flag_Coupon",value=0)
    #                 s_t=stm.text_input("s_t")
    #                 s_0=stm.text_input("s_0")
    #             with col3:
    #                 rng_k=stm.text_input("rng_k")
    #                 rng_b=stm.text_input("rng_b")
    #                 strike=stm.text_input("strike")
    #                 barrier=stm.text_input("barrier")
    #                 rebate=stm.text_input("rebate")
    #                 coupon=stm.text_input("coupon")
    #             with col4:
    #                 obs_day=stm.text_input("obs_day",value=20)
    #                 # ,'daily_amt':[1]
    #                 leverage=stm.text_input("leverage",value=2)
    #                 leverage_expire=stm.text_input("leverage_expire",value=0)# 我司自己的概念
    #                 # ,'isCashsettle':[0]
    #                 strike_ramp=stm.text_input("strike_ramp",value=10)
    #                 barrier_ramp=stm.text_input("barrier_ramp",value=10)
    #             flag=stm.text_input("flag")
    #             expired_date=stm.text_input("expired_date")
       
            
    #         params={'flag_Inter':[int(f) for f in formatParams(flag_Inter)]
    #                   ,'flag_Coupon':[int(f) for f in formatParams(flag_Coupon)]
    #                   ,'underlyingCode':formatParams(underlyingCode)
    #                   ,'flag':formatParams(flag)
    #                   ,'direction':['B']
    #                   ,'s_t':[getfloat(s) for s in formatParams(s_t)]
    #                   ,'s_0':[getfloat(s) for s in formatParams(s_0)]
    #                   ,'rng_k':[getfloat(s) for s in formatParams(rng_k)]
    #                   ,'rng_b':[getfloat(s) for s in formatParams(rng_b)]
    #                   ,'strike':[getfloat(s) for s in formatParams(strike)]
    #                   ,'barrier':[getfloat(s) for s in formatParams(barrier)]
    #                   ,'rebate':[getfloat(s) for s in formatParams(rebate)]
    #                   ,'coupon':[getfloat(s) for s in formatParams(coupon)]
    #                   ,'first_obsdate':[""]
    #                   ,'expired_date':[getfloat(s) for s in formatParams(expired_date)]
    #                   ,'obs_day':[getfloat(s) for s in formatParams(obs_day)]
    #                   ,'daily_amt':[1]
    #                   ,'leverage':[getfloat(s) for s in formatParams(leverage)]
    #                   ,'leverage_expire':[getfloat(s) for s in formatParams(leverage_expire)]# 我司自己的概念
    #                   ,'isCashsettle':[0]
    #                   ,'strike_ramp':[getfloat(s) for s in formatParams(strike_ramp)]
    #                   ,'barrier_ramp':[getfloat(s) for s in formatParams(barrier_ramp)]
    #                   ,'const_sgm':[getfloat(s) for s in formatParams(const_sgm)]
    #                   }
    #         for k in params.keys():
    #               if len(params[k])==0:
    #                   params[k]=[""]
    #               if len(params[k])!=contracts_num:
    #                   params[k]=params[k]*contracts_num
    #         stm.write(params)
       
    #         # if stm.button("Cal"):
    #         df=pd.DataFrame()
    #         for i in range(contracts_num):    
    #             # stm.write(params['s_t'][i])
    #             ca=CalAcc(underlyingCode=params['underlyingCode'][i]
    #                       , flag=params['flag'][i], direction=params['direction'][i]
    #                       , s_t=params['s_t'][i], s_0=params['s_0'][i]
    #                       , rng_k=params['rng_k'][i],rng_b=params['rng_b'][i]
    #                       , strike=params['strike'][i], barrier=params['barrier'][i]
    #                       , rebate=params['rebate'][i],coupon=params['coupon'][i]
    #                       , pricing_time=datetime.now()
    #                       , trade_date=trade_date, first_obsdate=params['first_obsdate'][i]
    #                       , expired_date=params['expired_date'][i]
    #                       , obs_days=params['obs_day'][i]
    #                       , daily_amt=params['daily_amt'][i]
    #                       , leverage=params['leverage'][i], leverage_expire=params['leverage_expire'][i]
    #                       , isCashsettle=params['isCashsettle'][i]
    #                       , strike_ramp=params['strike_ramp'][i], barrier_ramp=params['barrier_ramp'][i]
    #                       , const_sgm=params['const_sgm'][i])
    #             res=ca.getRes(flag_Inter=params['flag_Inter'][i], flag_Coupon=params['flag_Coupon'][i],mult=mult)
    #             df=pd.concat([df,res],ignore_index=True,axis=1)
    
    #         df.columns=list(map(lambda c:"结构"+str(c+1),df.columns))
    #         # with container_show:
    #             # container_show.write("here to show")
    #         stm.write(df)
    #         stm.stop()
       
            
    #     except:
    #         pass
        











