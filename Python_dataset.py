# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:14:56 2020

@author: Assa
"""

pip install ccxt


from datetime import datetime

import plotly.graph_objects as go

import ccxt

import pandas as pd

#Affichage des différentes classes de ccxt
print (ccxt.exchanges) 

#Sélection de l'obejt btcmarkets
btcmarkets = ccxt.btcmarkets()

#Selection des échanges entre btc et aud
trading_pair = 'BTC/AUD' 


# fetch_ohclv permet de recupérer les données de ccxt et les mettre dans candles

candles = btcmarkets.fetch_ohlcv(trading_pair, '1h', 
                                 btcmarkets.parse8601('2020-07-08T00:00:00'))


#Construire les listes date, open_data etc.. qui contiennent la date et les prix                                 
dates = []
open_data = []
high_data = []
low_data = []
close_data = []

#Transformer les listes en  dictionnaire 
for candle in candles:

    dates.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
    open_data.append(candle[1])
    high_data.append(candle[2])
    low_data.append(candle[3])
    close_data.append(candle[4])
    
dt_btc_aud = {'dates' : dates, 'open' : open_data, 'high' : high_data, 
              'low' : low_data, 'close' : close_data}

#Transformer le dictionnaire en data frame 
df = pd.DataFrame(dt_btc_aud)

#Exporter la data frame en csv 
df.to_csv(r"C://Users/assa7/OneDrive/Documents/PYTHON_MASTER_2_TIDE", 
          sep = ",", index = False)


fig = go.Figure(data=[go.Candlestick(x=dates,
                      open=open_data, high=high_data,
                      low=low_data, close=close_data)])

fig.write_html("first_figure.html", auto_open = True)

#graphique des lignes 

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(x=dates, y=open_data, name='Open Price',
                         line=dict(color='firebrick', width=1)))
fig_line.add_trace(go.Scatter(x=dates, y=close_data, name = 'Close price',
                         line=dict(color='royalblue', width=1)))
fig_line.add_trace(go.Scatter(x=dates, y=high_data, name = 'Highest price',
                         line=dict(color='firebrick', width=1,
                              dash='dash')))


fig_line.write_html("line_figure.html", auto_open = True)



#Taux de variation entre open et close
variation_open_close = []

for i in range(len(open_data)) :
    taux = close_data[i] - open_data[i]
    variation_open_close.append(taux)














