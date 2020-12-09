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
from statsmodels.tsa.stattools import adfuller
pip install arch
from arch import arch_model
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

df_coin=pd.read_csv('C:/Users/assa7/OneDrive/Documents/PYTHON_MASTER_2_TIDE/btc_aud_new.csv')

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


# Un moyen  d'investir est d'utiliser un actif financier

#On s'interessera donc a la serie financiere qui correspond au prix a la clôture d'un actif financier contenue dans la librairie CCXT de python.

#On représente graphiquement le cours à la clôture en fonction du temps.


np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage de la figure

fig = go.Figure()

fig.add_trace(go.Scatter(x=random_x, y=df_coin.close,
                    mode='lines',
                    name='close price'))


fig.update_layout(title=" Trajectoire représentant le prix à la cloture selon le temps")

#Commentaire:
    
#On observe que le cours de clôture a tendance a augmenté entre Avril 2019 jusqu'a fin Juin 2019 (le pic est atteint juste avant Juillet 2019).
#De juillet 2019 jusqu'à Octobre 2020, on observe des fluctuations; à certains moments il y a des augmentations du prix, et à d'autres moments des baisses du prix.
# On observe également un autre pic ; on observe une forte baisse du prix au mois de Mars 2020
#On peut tracer l'ACF d'une telle série chronologique même si on sait à l'avance qu'étant donné que la série est non-stationnaire
#l'autocorrélation empirique estimera très mal l'autocorrélation. Par conséquent, on interprétera mal l'ACF. 

#Le prix à la clôture n'est pas stationnaire car on pouvait observer un phénomène de tendance.
#Représentons le graphe de moyenne mobile, de l'ec-type mobile et de la série pour confirmer cela.

rolling_mean = df_coin.close.rolling(window = 25).mean()
rolling_std = df_coin.close.rolling(window = 25).std()
plt.plot(df_coin.close, color = 'blue', label = 'Données brutes')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show()


#On observe que la moyenne mobile n'est pas constante, elle augmente ou baisse avec le temps, bien que l'ec-type mobile reste plus ou moins constant
#Ce qui nous confirme que la série n'est pas stationnaire.

#l'ACF de la série chronologique

from statsmodels.graphics.tsaplots import plot_acf 
plot_acf (df_coin.close, lags = 90)

# Nous voyons bien toutes les limites de l'ACF. En effet, nous ne pouvons pas conclure sur le fait que les variables de la série soient indépendantes dès lors qu'il y a un phénomène de tendance qui apparaît
# Le but est  de retirer ces phénomènes de tendance et de saisonnalité afin de pouvoir estimer le cours de clôture.

###########################################################################################################################


# L'objectif de notre projet sera surtout de savoir quel est le moment opportun pour investir dans une telle action afin de réaliser un bénéfice.
#Essayons de mesurer la volatilité de l'actif c'est-à-dire le risque ( à la hausse ou à la perte ) que supporte un investisseur en détenant un tel actif.
# Rappel : volatilité d'une action = propension à subir des mouvements de prix plus ou moins prononcés.
# A priori,si on suppose qu'on achète l'actif en Mars 2019, on pourrait penser que le meilleur moment pour vendre notre actif serait aux alentours de fin Juin 2019 
#car le prix de clôture atteint son maximum, par conséquent on pourrait  réalisé un grand bénéfice ( de 15M de dollars ).
# (Et à contrario, le moment le moins opportun durant la période observé, serait en Mars 2020  car ferait seulement un bénéfice de 2M .
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


fig.update_layout(title=" Trajectoire représentant le log-rendement selon le temps")


#Commentaire : 

#Le log-rendement (mesure de la volatilité) tourne plus ou moins autour de 5%/7%, l'actif est globalement peu volatile.
#Cependant, on observe des pics ; il y a une très forte volatilité de l'actif autour du mois de Mars 2020 ( environ 19%), donc le risque qu'encourt l'investisseur est important à ce moment.
#On observe également un autre pic en fin Juin 2019, car le cours atteint son maximum le 26 Juin 2020 et ensuite chute le 27 Juin 2020.
#Ce qui rejoint ce que nous avions dit précédemment à savoir que plus l'amplitude des variations du prix est important, plus
#l'actif sera volatile donc plus l’espérance de gain (ou risque de perte) sera important.



######################################### III- Estimation paramétrique  ###################################################

###########################################################################################################################

#A ) Estimation paramétrique du log-rendement 

# PREDICTION AVEC L'ALGORITHME ARIMA #

#Pq ce choix d'algortihme ? L'algo ARIMA permet de faire des prédictions même si la série est non -stationnaire

# Explication : ARIMA est la combinaison de trois termes : 
# le terme autorégressif (AR), le terme de différenciation (I) et le terme de moyennes mobiles (MA)
# ARIMA (p,d,q):
# p est le nombre de termes auto-régressifs
# d est le nombre de différences
# q est le nombre de moyennes mobiles

# order=() ce sont respectivement les coefficients AR, I et MA je test le modèle avec order=( 1,1,1) 
mdl = sm.tsa.statespace.SARIMAX(logR,order=(1, 1, 1))
res = mdl.fit()
print(res.summary())

#Observations : On constate que 3 coefficients ont été estimés :
# 1 coefficient du terme auto-régressif : -0.0594
# le terme de moyenne mobile : -0.9998
# et la variance du bruit :  0.6241   

#Graphiques de la modélisation
res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()

y = pd.DataFrame(logR)



# adapter le modèle aux données
res = sm.tsa.statespace.SARIMAX(y,
                                order=(2, 1, 2),
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

# rmse = 0.912

#Conclusion : 
# On peut remarquer que la prédiction obtenue en utilisant l'algorithme ARIMA n'est pas du tout satisfaisante.
#En effet, la prédiction rate presque tout le temps les piques du log-rendement, or ces piques sont très importants car cela nous
#renseigne que la volatilité de l'actif financier est très forte (sûrement du à un Krach Boursier) donc que le risque
#qu'encourt l'investisseur est grand à ce moment. De plus, on peut observer sur le graphe des résidus standardisés, que
#les données du log-rendement sont trop bruitées et qu'il y beaucoup de volatilité dans les données.
#Ce qui explique le fait que la prédiction soit inintéressante. 

#Les ARMA ont une faiblesse dès lors qu'il y a de la volatilité dans les données.
#Il existe un modèle pour modéliser les log-rendements d'une série financière et qui tient compte de la volatilité.
#Ce sont les processus GARCH. 

#Pour appliquer un modèle GARCH, la série doit être stationnaire.On sait que la série du prix à la clôture n'est pas stationnaire
#Cependant, la série de du log-rendement l'est peut-être car en prenant le log d'une série on peut se ramener à qlq chose de stationnaire.

#Vérifions si la série du log-rendement est stationnaire.

def stationarity(timeseries):
    
    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=12).mean() #moyenne mobile
    
    # tracé statistiques mobiles
    original = plt.plot(timeseries, color='blue', label='Origine')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile') #tendance estimé par MA
    plt.legend(loc='best') 
    plt.title('Moyenne mobile')
    plt.show(block=False)
    
    # Test Dickey–Fuller : 
    result = adfuller(timeseries)
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


logR_2=logR 
logR_2.dropna(inplace=True) # car le test fonctionne si on enlève les valeurs NA
get_stationarity(logR_2) 

#Graphiquement : 
#La moyenne mobile qui estime la tendance de la série logR est bien constante donc la série logR est stationnaire.

#Résultat du test 
#Le test de Dickey-Fuller est définit comme suit; H0 :" La série n'est pas stationnaire" vs H1= "La série est stationnaire"
#On obtient une p-value de 0% donc on rejete H0 pour tout alpha. La serie du logR est bien stationnaire.
#On peut appliquer le modèle GARCH à ces données de log rendement.

garch = arch_model(logR, vol='GARCH', p=1, q=1)
resultat = garch.fit()
print(resultat.summary())



#PB INSTALLATION ARCH 








# Essayons uniquement de prendre en compte les prix à la clôture.

# B ) Estimation paramétrique du cours de clôture 

#Le prix à la clôture n'était  pas stationnaire car on pouvait observer un phénomène de tendance.


#On peut tout de même faire le modèle ARIMA.

mdl2 = sm.tsa.statespace.SARIMAX(df_coin["close"],order=(0, 0, 0),seasonal_order=(2, 2, 1, 7))
res2 = mdl2.fit()
print(res2.summary())

#Graphique des estimations
res2.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()


y = pd.DataFrame(df_coin["close"])
res2 = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 0),
                                seasonal_order=(2, 2, 1, 7),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()
 

 #Limites de la prédiction
pred = res2.get_prediction(start = 430,
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

rmse = math.sqrt(((pred.predicted_mean.values.reshape(-1, 1) - y[430:].values) ** 2).mean())
print('rmse = '+ str(rmse))

#On constate que la qualité d'ajustement du modèle est nettement meilleure. 
# L'algortihme a très bien estimé le prix de clôture.

# Cependant, l'objectif de notre projet c'est surtout un choix d'investissement à prendre
# La volatilité est donc un critère à prendre en compte .


#C )  Alogrithme LSTM permettant d'estimer le log-rendement.

###########################################################################################################################



######################################### IV- Prise de décision d'investissement  #########################################

###########################################################################################################################




