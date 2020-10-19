# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:50:15 2020

@author: assa7
"""

pip install plotly==4.11.0
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


# I - Aperçu du Data Frame
#II - Analyse du data frame

#**************************************I- Aperçu du Data Frame******************************************

#*******************************************************************************************************


#Importation des données du data-frame df_coin 

df_coin=pd.read_csv('C:/Users/assa7/OneDrive/Documents/PYTHON_MASTER_2_TIDE/btc_aud.csv')

#Affichage du type de df_coin 

type(df_coin)

#Affichage des colonnes en utilisant l'attribut columns

df_coin.columns 

#Dimension du data frame 

df_coin.shape

#Affichage des 7 premières lignes en utilisant la méthode head 

df_coin.head(7)

#Affichage des 7 dernières lignes en utilisant la méthode tail

df_coin.tail(7)


#Affichage des informations globales du data frame (nbr de colonnes, nbr de lignes, nbr de valeurs manquantes..)

df_coin.info()

#Type de chaque colonne

df_coin.dtypes

#Statistiques globales sur toutes les variables numériques : méthode describe

df_coin.describe()




#****************************************** II-Analyse du data frame *****************************************

#**************************************************************************************************************


# graphique du packages Plotly : du prix le plus grand et du prix le plus petit en fonction
# de la date 

import numpy as np
np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage des figures 
fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=df_coin.high,
                    mode='lines',
                    name='high price'))
fig.add_trace(go.Scatter(x=random_x, y=df_coin.low,
                    mode='lines+markers',
                    name='low price'))

fig.update_layout(title=" Graphique intéractif représentant l'évolution du prix max et du prix min selon la date")

fig.show()


#courbe de densité


# On utilise ensuite la fonction map pour tracer les courbes des densités de la variable 'high' 
sns.kdeplot(df_coin.high)

