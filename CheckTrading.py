# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:20:03 2024

@author: dzrh
"""

from typing import List
from datetime import datetime
import requests
# from requests.auth import HTTPBasicAuth
import pandas as pd
import base64
# from CYL.OTCAPI import SelfAPI
import json
class YieldChainAPI(object):
    URL = 'http://192.168.61.238:9000/'
    ENDPOINT = {'token': 'api/token?grant_type=client_credential',
                'clientInfo': 'api/v1/clientInfoList',
                'clientCard': 'api/v1/clientBankCardList',
                'clientDuty': 'api/v1/clientDutyList',
                'clientBalance': 'api/v1/clientLatestBalanceList',
                'otcPosition': 'api/v2/clientPositionList',# OTC position
                'riskCalc': 'api/v1/realtimeRiskCalc',
                'surface': 'api/v1/getUnderlyingVolSurface',
                'surface_post': 'api/v1/saveVolatility',
                'undInfo': 'api/v1/getUnderlyingList',#对
                'optInfo': 'api/v1/getExchangeOptionList',#对
                'otcMargin': 'api/v1/realtimeMargin',
                'singleVol': 'api/v1/getVol',
                'listPosition': 'api/v1/realtimeExchangePositions',#累计场内持仓（无数据）
                'todayListTrade': 'api/v1/todayExchangePositions',#当日场内交易（有数据）
                'todayITCtrading':'api/v1/exchange/todayExchangePositions',
                # 'otcTrade_post': 'api/v1/order/option',
                # 'otcTrade_post': 'api/v1/otc-option/tradeListExport',
                'otcStTrade_post': 'api/v1/order/structure_options',
                'otcGrTrade_post': 'api/v1/order/group_options',
                'otcGrTrade_update': 'api/v1/order/group_options_main_update',
                'optHedgeVol_update': 'api/v1/saveOtcTradeHedgingVol',
                'optSettleVol_update': 'api/v1/saveOptionTradeVol',
                'optClose_post': 'api/v1/TradeClose',
                'optClose': 'api/v1/tradeCloseInfoList',
                'optSettle_update': 'api/v1/updateCustomTradeRisk',
                'optMargin_update': 'api/v1/updateTradePositionMargin',
                'tradeDetail': 'api/v1/tradeDetailList',
                'tradeDetail_2': 'api/v2/tradeDetailList',
                'tradeRisk': 'api/v1/queryTradeRisk',
                'clientBalance2': 'api/v1/ClientBalance',  # 和 clientBalance 接口重复
                'clientCash_post': 'api/v1/ClientCashInCashOut',
                'clientCash_confirm': 'api/v1/ClientCashInCashOutConfirm',
                'clientCash_reject': 'api/v1/ClientCashInCashOutReject',
                'clientBalance_post': 'api/v1/ClientBalanceGap',
                'client_create': 'api/v1/CreateClient'
                }

    def __init__(self, user: str, passwd: str) -> None:
        self.user = user
        self.passwd = passwd
        # auth=HTTPBasicAuth(self.user, self.passwd)
        # self.headers = {'Content-Type':'application/json', HTTPBasicAuth(self.user, self.passwd)}
        # self.headers = {'Content-Type': 'application/json'
        #                 , 'Authorization': 'Basic'+str(base64.b64encode().encode('utf'))
        #                 # , 'Authorization': HTTPBasicAuth(self.user, self.passwd)
        #                 }
        self.headers_log_in = {'authorization':'Basic ' + str(base64.b64encode((self.user+':'+self.passwd).encode('utf-8')),'utf-8'),'Content-Type':"application/json"}
        msg = requests.post(self.URL + self.ENDPOINT['token']
                            # ,auth=HTTPBasicAuth(self.user, self.passwd)
                            ,headers=self.headers_log_in
                            ).json()
        try:
            self.accessToken = msg['accessToken']
            self.headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.accessToken}'}
            print(f'{self.user} 链接成功')
        except:
            print(f'{self.user} {msg["errmsg"]}')

   

    def get_otcPosition(self, clientId: int = None, clientNumber: str = None, status: List[int] = None) -> pd.DataFrame:
        # 获取客户持仓列表
        body = {'pageIndex': 1, 'pageSize': 100000,
                'ClientId': clientId,
                'ClientNumber': clientNumber, 
                'TradeStatus': status}
        msg = requests.post(self.URL + self.ENDPOINT['otcPosition'],
                            headers=self.headers, json=body).json()
        my_list = msg['data']['rows']
        # new_list = []
        # for i in my_list:
        #     dic = {}
        #     for key, value in i.items():
        #         if isinstance(value, dict):

        #             for k, v in value.items():
        #                 dic[f'{key}_{k}'] = v
        #         else:
        #             dic[key] = value
        #     new_list.append(dic)
        # res = pd.DataFrame(new_list)
        res=pd.DataFrame(my_list)
        res=res[['TradeStatus','TradeNumber','UnderlyingCode','BuySell','TradeType','StructureType'
                  ,'Strike','UnderlyingPrice','TradeDate','ExerciseDate','CallPut','InitialSpotPrice','TradePrice','TradeOriginalAmount','TradeAmount']]
        return res

  

    def get_surface(self, underlyingCodes, valuedate) -> pd.DataFrame:
        # 获取波动率曲面
   
        
        body = {'UnderlyingCodes':underlyingCodes
                , 'VolType':"交易"
                , 'SystemDate': str(valuedate)}
        msg = requests.post(self.URL + self.ENDPOINT['surface'],
                            headers=self.headers,json=body).json()
        res= pd.DataFrame(pd.DataFrame(msg['data']))
        res.index=res.UnderlyingCode
        res=res[['UnderlyingCode','VolTable']].T
        res=dict(res.drop(index='UnderlyingCode'))
        for key in res.keys():
            res_single=pd.DataFrame(res.get(key)[0])
            res_single['Expire']=list(map(lambda x:x.split("D")[0],res_single.Expire))
            res_single['Expire']=res_single.Expire.astype(int)
            res_single=pd.pivot(res_single,values="Vol",index=["Expire"],columns=["Strike"])
            res[key]=res_single
 
        return res
    
    def get_surface_json(self, underlyingCodes, valuedate) -> pd.DataFrame:
        # 获取波动率曲面
 
        body = {'UnderlyingCodes':underlyingCodes
                , 'VolType':"交易"
                , 'SystemDate': str(valuedate)}
        msg = requests.post(self.URL + self.ENDPOINT['surface'],
                            headers=self.headers,json=body).json()
        
 
        return msg

   
    def get_listInfo(self) -> pd.DataFrame:
        # 获取场内合约信息
        msg_und = requests.post(self.URL + self.ENDPOINT['undInfo'],
                                headers=self.headers).json()
        msg_opt = requests.post(self.URL + self.ENDPOINT['optInfo'],
                                headers=self.headers).json()
        
        res = pd.DataFrame(msg_und + msg_opt)
        return res



    def get_listPosition(self) -> pd.DataFrame:
    # def get_listPosition(self) :
        # 获取场内持仓信息
        msg = requests.post(self.URL + self.ENDPOINT['listPosition'],
                            headers=self.headers
                            ,json={'AssetBookNames':['奇异结构交易簿','场外交易簿']}).json()
        # res=msg
        # res =msg['data']
        res = pd.DataFrame(msg['data'])
        return res
   

    def get_listTrade_today(self) -> pd.DataFrame:
        # 获取当日场内fut，opt成交数据
        msg = requests.post(self.URL + self.ENDPOINT['todayListTrade'],
                            headers=self.headers).json()
        res = pd.DataFrame(msg['data'])
        return res




    def get_tradeDetail(self,status:str=None) -> pd.DataFrame:
        body = {'pageIndex': 1, 'pageSize': 100000
                ,'AssetBookNames':['奇异结构交易簿','场外交易簿']
                ,'TradeStatus':status
                }
        msg = requests.post(self.URL + self.ENDPOINT['tradeDetail'],
                            headers=self.headers,json=body).json()
        res = pd.DataFrame(msg['data']['rows'])
        # res=res[['TradeNumber','TradeStatus','UnderlyingCode','ClientName','BuySell','TradeType','StructureType'
        #           ,'Strike','UnderlyingPrice','TradeDate','ExerciseDate','CallPut'
        #           ,'InitialSpotPrice','TradePrice','TradeAmount','Propertys','OriginalStockEqvNotional']]
        return res
    
    def get_close(self,trd_date) -> pd.DataFrame:
        body = {'ValueDate':trd_date
                ,'AssetBookNames':['奇异结构交易簿','场外交易簿']
                # ,'TradeStatus':'确认成交'
                }
        msg = requests.post(self.URL + self.ENDPOINT['optClose'],
                            headers=self.headers,json=body).json()
        res = pd.DataFrame(msg['data'])
        # res=res[['TradeNumber','UnderlyingCode','BuySell','TradeType','StructureType'
        #           ,'Strike','UnderlyingPrice','TradeDate','ExerciseDate','CallPut'
        #           ,'InitialSpotPrice','TradePrice','TradeAmount','Propertys']]

        return res
    
   

class SelfAPI():
    URL = 'http://192.168.65.99:19999/'
    ENDPOINT = {'token': "api/userserver/user/login"
                ,"getAllClientList":"api/clientserver/client/list"
                ,'TradeList': "api/quoteserver/trade/queryTradeList"
                ,"CloseTradingList":"api/quoteserver/trade/settlementConfirmBookSelectByPage"
                ,'ITCTrade':"api/quoteserver/trade/exchangeTrade/selectListByPage"
                ,"ITCLive":"api/quoteserver/trade/risk/selectPosListByPage"
                ,"TradeRisk":"api/quoteserver/trade/risk/selectListByPage"
                ,"VolSurface":"api/quoteserver/volatility/queryByUnderlyingId"
                ,"CurrentRisk":"api/nettyserver/scenario/getTradeList"
                ,"CurrentRisk_Summary":"api/nettyserver/risk/getRiskInfoList"
                ,"Pricing":"api/quoteserver/quote/calculate"
                ,"variety":"api/dmserver/variety/list"
        
                }
    USER=''
    PASSWD=''
    PAGESIZE=10000
    PAGENO=1
    BOOK_LIST=[1,2,9]

    def __init__(self) -> None:       
        self.payload = json.dumps({
            "account": self.USER
            ,"password": self.PASSWD
            ,"loginFrom":1
        })
        self.headers_log_in = {
            'Authorization': '153C26883C664B3BBFC7E46326421DC7',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '192.168.64.231:8081',
            'Connection': 'keep-alive'
        }
        
        msg = requests.post(self.URL+self.ENDPOINT['token']
                               , headers=self.headers_log_in, data=self.payload).json()
        try:
            self.accessToken=msg['data']['token']
            print(f'{self.USER} {msg["message"]}')
        except:
            print(f'{self.USER} {msg["errmsg"]}')
            
        self.headers = {
               'Authorization': self.accessToken,
               'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
               'Content-Type': 'application/json',
               'Accept': '*/*',
               'Host': '192.168.64.231:8081',
               'Connection': 'keep-alive'
           }

        self.optionType_map={"pass1":"AISnowBallCallPricer,AISnowBallPutPricer,AIBreakEvenSnowBallCallPricer,AICustomPricer,AIEnAsianPricer,AIInsuranceAsianPricer,AILimitLossesSnowBallCallPricer,AILimitLossesSnowBallPutPricer,"
                    ,"亚式期权":"AIAsianPricer"
                    ,"acccall":"AICallAccPricer"
                    ,"accput":"AIPutAccPricer"
                    ,"fpcall":"AICallFixAccPricer"
                    ,"fpput":"AIPutFixAccPricer"
                    ,"bfpcall":"AICallFixKOAccPricer"
                    ,"bfpput":"AIPutFixKOAccPricer"
                    ,"bacccall":"AICallKOAccPricer"
                    ,"baccput":"AIPutKOAccPricer"
                    ,"bacccallplus":"AIEnCallKOAccPricer"
                    ,"baccputplus":"AIEnPutKOAccPricer"
                    ,"f":"AIForwardPricer"
                    ,"van":"AIVanillaPricer"}
        self.__getvarietyInfo()
    
    def __getvarietyInfo(self):
        msg = requests.post(self.URL + self.ENDPOINT['variety'],
                            headers=self.headers, data={}).json()
        self.varietyinfo= pd.DataFrame(msg['data'])
        self.varietyinfo.index=self.varietyinfo.varietyCode
        

    def getOTC_open_onDate(self,given_d:str,clientName_list=[],varietyList=[],underlyingCodeList=[],optiontype_Name=[],book_filter=True):
        #获取给定日期当日场外开仓明细
        book_list=self.BOOK_LIST if book_filter else []
        payload=json.dumps({"assetIdList":book_list
            # "optionTypeList": ["AIPutKOAccPricer"]
            	# ,"tradeStateList": ["partClosed","closed","exercised","expired"]
                ,"pageSize":self.PAGESIZE
                ,"pageNo":self.PAGENO
                ,'clientIdList':[self.__getClientList().get(cl_na)for cl_na in clientName_list]
                ,"startTradeDate":given_d
                ,"endTradeDate":given_d
                ,"optionTypeList":[self.optionType_map[opt] for opt in optiontype_Name]
                ,"underlyingCodeList":underlyingCodeList
                ,"varietyIdList":self.varietyinfo.loc[varietyList]['id'].values.tolist()
                })
    
        msg = requests.post(self.URL + self.ENDPOINT['TradeList'],
                            headers=self.headers, data=payload).json()
        res= pd.DataFrame(msg['data']['records'])
        return res
    
    def getOTC_closed_onDate(self,given_d:str,clientName_list=[],underlyingCodeList=[],optiontype_Name=[],book_filter=True):
            #获取给定日期当日平仓开仓明细
            book_list=self.BOOK_LIST if book_filter else []
            payload=json.dumps({"assetIdList":book_list
                    ,"pageSize":self.PAGESIZE
                    ,"pageNo":self.PAGENO
                    ,'clientIdList':[self.__getClientList().get(cl_na)for cl_na in clientName_list]
                    ,"startCloseDate":given_d
                    ,"endCloseDate":given_d
                    ,"optionTypeList":[self.optionType_map[opt] for opt in optiontype_Name]
                    ,"underlyingCodeList":underlyingCodeList
                    # ,"varietyIdList":self.varietyinfo.loc[varietyList]['id'].values.tolist()
                    })
       
            msg = requests.post(self.URL + self.ENDPOINT['CloseTradingList'],
                                headers=self.headers, data=payload).json()
            res= pd.DataFrame(msg['data']['records'])
            return res


def check(l_1,l_2):
    return list((set(l_1)-set(l_2))|(set(l_2)-set(l_1)))



if __name__=="__main__":
    user=''
    passwd=''
    yl=YieldChainAPI(user,passwd)
    trade_date="2024-11-04"
    api=SelfAPI()    
    
    yl_pos_live=yl.get_tradeDetail(['确认成交','新增待确认'])
    yl_pos_live=yl_pos_live[pd.to_datetime(yl_pos_live.StartDate)==trade_date]
    yl_pos_live['TradeType']=yl_pos_live['TradeType'].map({"远期":"F"}).fillna(yl_pos_live['CallPut'])
    yl_pos_live['StartDate']=pd.to_datetime(yl_pos_live['StartDate'])
    
    
    yl_pos_close=yl.get_close(trade_date)
    yl_pos_close=yl_pos_close[yl_pos_close.AssetBookName=='场外交易簿']
    yl_pos_close['TradeType']=yl_pos_close['TradeType'].map({'远期':"F"}).fillna(yl_pos_close['CallPut'])
    yl_pos_close['StartDate']=pd.to_datetime(yl_pos_close['StartDate'])
    
    used_cols=['TradeNumber','TradeType','StartDate']
    yl_trading=pd.concat([yl_pos_live[used_cols],yl_pos_close[used_cols]])
    
    close_self=api.getOTC_closed_onDate(trade_date)
    open_self=api.getOTC_open_onDate(trade_date)
    
    
    open_f=open_self[open_self.optionType=='AIForwardPricer'].shape[0]
    open_other=open_self.shape[0]-open_f
    close_f=close_self[close_self.optionType=='AIForwardPricer'].shape[0]
    close_other=close_self.shape[0]-close_f
    
    
    open_f_yl=yl_trading[(yl_trading.TradeType=='F')&(yl_trading.StartDate==pd.to_datetime(trade_date))].shape[0]
    open_other_yl=yl_trading[(yl_trading.TradeType!='F')&(yl_trading.StartDate==pd.to_datetime(trade_date))].shape[0]
    close_f_yl=yl_pos_close[yl_pos_close.TradeType=='F'].shape[0]
    close_other_yl=yl_pos_close.shape[0]-close_f_yl
    
    
    
    
    if open_f==open_f_yl:
        print(f"forward open ZY:{open_f}")#contain those closed today
        print(f"forward open YL:{open_f_yl}")
    else:
        l_1=open_self[open_self.optionType=='AIForwardPricer'].tradeCode.tolist()
        l_2=yl_trading[(yl_trading.TradeType=='F')&(yl_trading.StartDate==pd.to_datetime(trade_date))].TradeNumber.tolist()
        wrg=check(l_1,l_2)
        print(f"To Check Trading:{wrg}")
        
    if close_f==close_f_yl:
        print(f"forward finish ZY:{close_f}")#contain those closed today
        print(f"forward finish YL:{close_f_yl}")
    else:
        l_1=close_self[close_self.optionType=='AIForwardPricer'].tradeCode.tolist()
        l_2=yl_pos_close[yl_pos_close.TradeType=='F'].TradeNumber.tolist()
        wrg=check(l_1,l_2)
        print(f"To Check Trading:{wrg}")
    
    if open_other==open_other_yl:
        print(f"OTC open ZY:{open_other}")#contain those closed today
        print(f"OTC open YL:{open_other_yl}")
    else:
        l_1=open_self[open_self.optionType!='AIForwardPricer'].tradeCode.tolist()
        l_2=yl_trading[(yl_trading.TradeType!='F')&(yl_trading.StartDate==pd.to_datetime(trade_date))].TradeNumber.tolist()
        wrg=check(l_1,l_2)
        print(f"To Check Trading:{wrg}")
        
    if close_other==close_other_yl:
        print(f"OTC finish ZY:{close_other}")#contain those closed today
        print(f"OTC finish YL:{close_other_yl}")
    else:
        l_1=close_self[close_self.optionType!='AIForwardPricer'].tradeCode.tolist()
        l_2=yl_pos_close[yl_pos_close.TradeType!='F'].TradeNumber.tolist()
        wrg=check(l_1,l_2)
        print(f"To Check Trading:{wrg}")

















