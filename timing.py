# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:55:59 2024

@author: dzrh
"""

# import xlwings as xw
# import pandas as pd
# import numpy as np
# from scipy.stats import norm
from matplotlib import pyplot as plt
# # import tushare as tu
# from datetime import timedelta,datetime
# import time as sys_time
# # import time
# import iFinDPy as fd
# import os
# from CYL.OptionPricing import BSM,calIV,calTradttm,LinearInterpVol
# # from CYL.YieldChainAPI import YieldChainAPI 
# # from CYL.StressTest import StressTest
# from CYL.StressTestNew import StressTestNew,BarrierContracts,getRQcode
# from CYL.pythonAPI_pyfunctions4newDll_3 import datetime2timestamp,pyAIAccumulatorPricer,pyAIKOAccumulatorPricer
# from CYL.StressTestNew import LinearInterpVol 
# from CYL.AccCalculator import CalAcc 
# import rqdatac as rqd
# rqd.init("license","G9PESsRQROkWhZVu7fkn8NEesOQOAiHJGqtEE1HUQOUxcxPhuqW5DbaRB8nSygiIdwdVilRqtk9pLFwkdIjuUrkG6CXcr3RNBa3g8wrbQ_TYEh3hxM3tbfMbcWec9qjgy_hbtvdPdkghpmTZc_bT8JYFlAwnJ6EY2HaockfgGqc=h1eq2aEP5aNpVMACSAQWoNM5C44O1MI1Go0RYQAaZxZrUJKchmSfKp8NMGeAKJliqfYysIWKS22Z2Dr1rlt-2835Qz8n5hudhgi2EAvg7WMdiR4HaSqCsKGzJ0sCGRSfFhZVaYBNgkPDr87OlOUByuRiPccsEQaPngbfAxsiZ9E=")
# import streamlit as stm
# def getRQcode(underlying):
    
#     code=underlying.upper() if underlying[-4].isdigit() else underlying[:-3].upper()+'2'+underlying[-3:]
#     return code

# cur_trade_date=datetime(2024,3,25).date()
# underlyingCode='MA405'
# underlyingCode=getRQcode(underlyingCode)

# a=rqd.get_ticks(underlyingCode)[['trading_date','update_time','last','a1','b1']]

import streamlit as st
import numpy as np
import time
 
# 使能缓存
@st.cache_data()
def get_data():
    data = []
    return data
 
# 模拟实时数据生成器
def generate_data():
    while True:
        # 生成随机数据
        new_data = np.random.randn()
        # 更新数据列表
        data_buffer.append(new_data)
        # 每秒生成一个数据点
        time.sleep(1)
 
# 初始化数据
data_buffer = get_data()
 
# 启动实时数据生成器
# st.thread.Thread(target=generate_data).start()
 
# 设置刷新间隔（例如每2秒刷新一次）
# st.set_page_config(page_title="实时数据图像", initial_sidebar_state="expanded")
 
# 绘制实时数据折线图
with st.sidebar:
    st.subheader("实时数据图像")
    interval = st.number_input("刷新间隔（秒）", min_value=0, value=2, step=1)
 
# # 使用streamlit.repeat来保持组件的状态，并定期重绘图像
@st.cache()
def plot_data(data):
    # import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig
 
# # 使用streamlit.repeat来保持组件的状态，并定期重绘图像
plot_chart = plot_data(data_buffer)
chart_container = st.empty()
 
# # 定时更新图像
st.set_option('deprecation.showPyplotGlobalUse', False)
 
def update_chart():
    chart_container.pyplot(plot_data(data_buffer))
 
# 开始定时更新图像
st.set_interval(update_chart, interval * 1000)