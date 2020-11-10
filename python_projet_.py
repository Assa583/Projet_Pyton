# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:50:15 2020

@author: assa7
"""

########################################## PROJET : Algorithme : Investissement d'action  #################################

###########################################################################################################################

pip install plotly==4.11.0
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import itertools
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import math

# I - Aperçu du Data Frame
# II - Analyse du data frame
# II - Estimation 

######################################### I- Aperçu du Data Frame##########################################################

###########################################################################################################################


#Importation des données du data-frame df_coin 

df_coin=pd.read_csv('C:/Users/assa7/OneDrive/Documents/PYTHON_MASTER_2_TIDE/btc_aud.csv')

#Affichage du type de df_coin 

type(df_coin)

#Affichage des colonnes en utilisant l'attribut columns

df_coin.columns 
print(df_coin)
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

###########################################################################################################################




######################################### II-Analyse graphique du data frame ##############################################

###########################################################################################################################




#On s'interessera  a la serie financiere qui correspond au prix a la clôture d'un actif financier contenue dans la librairie CCXT de python.

#On représente graphiquement le cours à la clôture en fonction du temps.


np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage de la figure

fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=df_coin.close,
                    mode='lines',
                    name='close price'))


fig.update_layout(title=" Graphique intéractif représentant l'evolution du prix à la cloture selon le temps")

#Commentaire :
# On observe qu'au cours du mois de juillet, le cours de clôture reste plus ou moins constant il n'y a pas de phénomène de tendance ni de phénomène de saisonnalité durant cette période.
#Ensuite,  entre le 19 juillet et le 16 août 2020 apparaît un péhnomène de tendance, on peut voir une augmentation du prix de clôture 
#On peut tracer l'ACF d'une telle série chronologique même si on sait à l'avance qu'étant donné que la série est non-stationnaire
#l'autocorrélation empirique estimera très mal l'autocorrélation par conséquent, on interprétera mal l'ACF. 

from statsmodels.graphics.tsaplots import plot_acf 
plot_acf (df_coin.close, lags = 90)

# Nous voyons bien toutes les limites de l'ACF. En effet, nous ne pouvons pas conclure sur le fait que les variables de la série soient indépendantes dès lors qu'il y a un phénomène de tendance qui apparaît
# Le but est  de retirer ces phénomènes de tendance et de saisonnalité afin de pouvoir estimer le cours de clôture.

###########################################################################################################################


# L'objectif de notre projet sera surtout de savoir quel est le moment opportun pour investir dans une telle action ?
#Essayons de mesurer la volatilité de l'actif c'est-à-dire le risque que supporte un investisseur en détenant un tel actif.
# Rappel : volatilité d'une action = propension à subir des mouvements de prix plus ou moins prononcés.
# A priori, on pourrait penser que le meilleur moment pour investir serait entre le 12 juillet et le 20 juillet car le prix est relativement stable
# puisque le prix est stable, l'actif sera très peu volatile,donc le risque faible.
#Voyons cela de plus près...

#On calcule les log-rendements de la série entre le prix_t et t-1

logR=np.log(df_coin["close"]/df_coin["close"].shift(1))*100

#On représente graphiquement le log-rendement en fonction du temps


np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage de la figure

fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=logR,
                    mode='lines',
                    name='log rendement'))


fig.update_layout(title=" Graphique intéractif représentant le log-rendement selon le temps")


#Commentaire : 
#On observe très clairement que la volatilité est faible durant le mois de juillet (volatilité autour de 0% / 1%) 
#Cela rejoint ce que vous avions vu dans le graphe précédent à savoir que le cours variait peu durant cette période.
#A partir du 26 juillet, la volatilité de l'actif devient de plus en plus forte.
#Enfin, juste après le 2 août on peut voir un grand pique , la volatilité est forte , donc le risque qu'encourt l'investisseur est important à ce moment


######################################### III- Estimation paramétrique du log-rendement ###################################

###########################################################################################################################



# PREDICTION AVEC L'ALGORITHME ARIMA #

#Pq ce choix d'algortihme ? L'algo ARIMA permet de faire des prédictions même si la série est non -stationnaire

# Explication : ARIMA est la combinaison de trois termes : 
# le terme autorégressif (AR), le terme de différenciation (I) et le terme de moyennes mobiles (MA)
# ARIMA (p,d,q):
# p est le nombre de termes auto-régressifs
# d est le nombre de différences
# q est le nombre de moyennes mobiles

mdl = sm.tsa.statespace.SARIMAX(logR,order=(0, 0, 0),seasonal_order=(2, 2, 1, 7))
res = mdl.fit()
print(res.summary())

#Observations : On constate que 4 coefficients ont été estimés :
# 2 coefficients du termes auto-régressifs : -0.6689 et -0.3628
# le terme de moyenne mobile : -0.9999
# et la variance du bruit : 0.3349

#Graphiques 
res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()

y = pd.DataFrame(logR)

# adapter le modèle aux données
res = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 0),
                                seasonal_order=(2, 2, 1, 7),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()
 
# Limites de la prédiction
pred = res.get_prediction(start = 430,
                          dynamic = False, 
                          full_results=True)

# Graphe de la prédiction et des données
fig = plt.figure(figsize=(19, 7))
ax = fig.add_subplot(111)
ax.plot(y[0:],color='#006699',linewidth = 3, label='Observation');
pred.predicted_mean.plot(ax=ax,linewidth = 3, linestyle='-', label='Prediction', alpha=.7, color='#ff5318', fontsize=18);
ax.set_xlabel('temps', fontsize=18)
ax.set_ylabel('logR', fontsize=18)
plt.legend(loc='upper left', prop={'size': 20})
plt.title('Prediction ARIMA', fontsize=22, fontweight="bold")
plt.show()



# Précision du modèle en calculant le RMSE
rmse = math.sqrt(((pred.predicted_mean.values.reshape(-1, 1) - y[430:].values) ** 2).mean())
print('rmse = '+ str(rmse))

#Conclusion : 
# On peut remarquer que la prédiction obtenue en utilisant l'algorithme ARIMA n'est pas si satisfaisante que cela.
#En effet, la prédiction rate souvent les piques du log-rendement, or ces piques sont très importants car cela nous
#renseigne que la volatilité de l'actif financier est très forte (sûrement du à un Krach Boursier) donc que le risque
#qu'encourt l'investisseur est grand à ce moment. 

#Essayons de trouver un autre algorithme qui estimera mieux la volatilité de l'actif.
