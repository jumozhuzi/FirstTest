# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:42:51 2024

@author: dzrh
"""
import pandas as pd
import numpy as np
import streamlit as stm
import streamlit_searchbox as stm_box
from CYL.OTCAPI import SelfAPI
import rqdatac as rqd
import re
rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")

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

def getRQcode(underlying):
    
    code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
    return code

clientName=api.getClientList()
if "clientnames" not in stm.session_state:
    stm.session_state["clientnames"]=clientName.to_dict()['id']
# stm.write(stm.session_state["clientnames"])


def search_function(query):
    # clientName=api.getClientList()
    # name_list=clientName.index.tolist()
    if query:
        pattern = re.compile('.*' + re.escape(query) + '.*', re.IGNORECASE)
        return [item for item in stm.session_state["clientnames"].keys() if pattern.match(item)]
    return "空"
search_results=stm_box.st_searchbox(search_function,label='Search for client name')

if search_results:
    client_name=search_results
    
    cur_trad_date=stm.date_input("Current Trade Date:",value=rqd.get_future_latest_trading_date())
    variety_list=stm.text_input("variety list")
    underlying_list=stm.text_input("underlying list")
    variety_list=[] if variety_list=="" else variety_list.split(",")
    underlying_list=[] if underlying_list=="" else underlying_list.split(",")
    clientid=stm.session_state["clientnames"][client_name]
    dffor=api.getOTC_LiveTrade(str(cur_trad_date),[clientid],variety_list,underlying_list,["f"])
    
    # age_filter = stm.slider('Filter by varity', value=(0, 100), step=1)
    # city_filter = stm.selectbox('Filter by strike', options=df['City'].unique())
    
    dffor['ClientDirt']=np.where(dffor.buyOrSellName=="买入","空头","多头")
    dffor['Lots']=dffor.availableVolume/dffor.underlyingCode.apply(getRQcode).apply(findMultiplier)
    dffor['availableVolume']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.availableVolume
    dffor['Lots']=np.where(dffor['ClientDirt']=="空头",-1,1)*dffor.Lots
    # a=dffor[dffor.clientName.str.contains(client_name)].groupby(['underlyingCode',"ClientDirt",'strike'])['availableVolume','Lots'].sum()
    a=dffor.groupby(['underlyingCode',"ClientDirt",'strike'])['availableVolume','Lots'].sum()
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
    # stm.markdown("#### "+dffor[dffor.clientName.str.contains(client_name)].clientName.unique()[0]+"远期明细：")
    stm.markdown("#### "+client_name+"远期明细：")

    # print("")
    # a_md=a.to_markdown()
    stm.write(a)
    # stm.markdown(a_md,unsafe_allow_html=False)
    # print("")
    # print("")
    stm.markdown("##### 各标的多空合计：")
    stm.dataframe(bb.groupby(["标的合约","客户方向"]).last())

        
    # else:
    #     pass

