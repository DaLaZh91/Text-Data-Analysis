# In dieser Datei werden die Klassifikationsverfahren angewandt.

# =============================================================================
# Pakete laden & Funktionen einbinden
# =============================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dill 
import random
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score # ET
from sklearn.metrics import confusion_matrix 
# from sklearn.datasets import load_boston
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import Lasso, LassoCV # Lasso
from sklearn.metrics import mean_squared_error # lasso
# https://laurenliz22.github.io/nlp_random_forest_and_neural_network_classifiers 

os.chdir(r'W:\your_folder\Python')
from functions import *

# =============================================================================
# Daten einlesen 
# =============================================================================

os.chdir(r'W:\your_folder\Output')
dill.load_session('4B_04_28.pkl')

stop_words = get_stop_words('de')

# Da die Gründe einzeln betrachtet werden, könnte theoretisch jeder Grund
# einzeln abgehandelt werden, da aber die Wörter aus der Wortsuche gebraucht
# werden, kommen erst alle vier Wortsuchen und dann die Klassifikationsmodelle,
# getrennt nach Gründen.

# =============================================================================
# =============================================================================
# # Wortsuchen 
# =============================================================================
# =============================================================================

# =============================================================================
# financial
# =============================================================================

### Neue Daten erstellen, für jeden Grund ein Datensatz, in dem z.B. nur die
### Renten Renten sind, der Rest alles 'K'

y_train_financial = list(pd.DataFrame(y_train).replace(['J', 'R', 'D', 'O'], 'C')[0])
y_test_financial = list(pd.DataFrame(y_test).replace(['J', 'R', 'D', 'O'], 'C')[0])

y_test_10_F = []
for i in range(len(y_test)):
    if y_test[i] == 'F':
        y_test_10_F.append(1)
    else:
        y_test_10_F.append(0)
        
y_train_10_F = []
for i in range(len(y_train)):
    if y_train[i] == 'F':
        y_train_10_F.append(1)
    else:
        y_train_10_F.append(0)
                
        
### Wortsuche    

### da hier aufgrund der geringen Anzahl an Kündigungen pro Grund nur sehr wenige 
### Wörter vorkommen, ist es schwierig eine Wordcloud ohne Namen zu machen

### stattdessen werden die häufig vorkommenden Wörter und Bigramme so angeschaut
### und die, die mit einer Kündigung aus finanziellem Grund zu tun haben, 
### werden ausgewählt

[freq_F, freq_F_bi] = getFreqWordsWithSW(X_train_F, 200, 100)

f_words = np.sort(['finanziell', 'insolvenzverwalt', 'engpass', 'wirtschaft',
                    'finanziell engpass', 'gezwung', 'insolvenzverfahr', 
                    'immobiliendarlehn', 'einbuss', 'eigentum', 'wenig lohn'])


poss_f_words = getBestCombinations(f_words, X_train, y_train_financial, 'F')

# wähle die wenigstens Wörter, wenn mehrere: zufällig das erste
amount_word = []
for i in range(len(poss_f_words)):
    amount_word.append(len(poss_f_words[i]))
f_words_updated = poss_f_words[myEqual(amount_word, min(amount_word))[0]]
# f_words_updated = ['eigentum', 'finanziell', 'immobiliendarlehn',
#                    'insolvenzverfahr', 'wenig lohn', 'wirtschaft']
# ['eigentum',
#  'finanziell',
#  'immobiliendarlehn',
#  'insolvenzverfahr',
#  'wenig lohn',
#  'wirtschaft']

[cm_F, sensi_F, speci_F, accuracy_F, ypred_F] = getValues(f_words_updated, [], 
                                                           X_train, y_train_financial,
                                                           'F', 'C')

# array([[ 13,   5],
#        [ 10, 851]], dtype=int64)

# sensi_F # 0.7222222222222222
# speci_F # 0.9883855981416957
# accuracy_F # 0.9829351535836177

[cm_F_test, sensi_F_test, speci_F_test, accuracy_F_test, ypred_F_test] = getValues(f_words_updated, [], 
                                                           X_test, y_test_financial,
                                                           'F', 'C')


# array([[  2,   6],
#        [  3, 367]], dtype=int64)

# sensi_F_test # 0.25
# speci_F_test # 0.9918918918918919
# accuracy_F_test # 0.9761904761904762

# =============================================================================
# Berufswechsel
# =============================================================================

### Neue Daten erstellen, für jeden Grund ein Datensatz, in dem z.B. nur die
### Renten Renten sind, der Rest alles 'K'

y_train_jobchange = list(pd.DataFrame(y_train).replace(['F', 'R', 'D', 'O'], 'C')[0])
y_test_jobchange = list(pd.DataFrame(y_test).replace(['F', 'R', 'D', 'O'], 'C')[0])

y_test_10_J = []
for i in range(len(y_test)):
    if y_test[i] == 'J':
        y_test_10_J.append(1)
    else:
        y_test_10_J.append(0)
        
y_train_10_J = []
for i in range(len(y_train)):
    if y_train[i] == 'J':
        y_train_10_J.append(1)
    else:
        y_train_10_J.append(0)
        
### Wortsuche
[freq_J, freq_J_bi] = getFreqWordsWithSW(X_train_J, 200, 100)

j_words = np.sort(getStem(['beschaftigt', 'betrieb', 'alt arbeitgeb', 'arbeitet',
                           'mitarbeiterin', 'mitarbeit', 'aufheb vertrag']))

poss_j_words = getBestCombinations(j_words, X_train, y_train_jobchange, 'J')
   
# wähle die wenigstens Wörter, wenn mehrere: zufällig das erste
amount_word = []
for i in range(len(poss_j_words)):
    amount_word.append(len(poss_j_words[i]))
j_words_updated = poss_j_words[myEqual(amount_word, min(amount_word))[0]]
j_words_updated # ['alt arbeitgeb', 'aufheb vertrag', 'beschaftigt', 'betrieb', 'mitarbeiterin']

[cm_J, sensi_J, speci_J, accuracy_J, ypred_J] = getValues(j_words_updated, [], 
                                                           X_train, y_train_jobchange,
                                                           'J', 'C')

cm_B
# array([[  9,   7],
#        [ 48, 815]], dtype=int64)
# sensi_J # 0.5625
# speci_J # 0.944380069524913
# accuracy_J # 0.9374288964732651

[cm_J_test, sensi_J_test, speci_J_test, accuracy_J_test, ypred_J_test] = getValues(j_words_updated, [], 
                                                           X_test, y_test_jobchange,
                                                           'J', 'K')

cm_J_test
# array([[  1,   6],
#        [ 16, 355]], dtype=int64)
# sensi_J_test # 0.14285714285714285
# speci_J_test # 0.9568733153638814
# accuracy_J_test # 0.9417989417989417


# =============================================================================
# Rente
# =============================================================================


### Neue Daten erstellen, für jeden Grund ein Datensatz, in dem z.B. nur die
### Renten Renten sind, der Rest alles 'K'

y_train_Rente = list(pd.DataFrame(y_train).replace(['B', 'F', 'T', 'S'], 'K')[0])
y_test_Rente = list(pd.DataFrame(y_test).replace(['B', 'F', 'T', 'S'], 'K')[0])


y_test_10_R = []
for i in range(len(y_test)):
    if y_test[i] == 'R':
        y_test_10_R.append(1)
    else:
        y_test_10_R.append(0)
        
y_train_10_R = []
for i in range(len(y_train)):
    if y_train[i] == 'R':
        y_train_10_R.append(1)
    else:
        y_train_10_R.append(0)
        
### Wortsuche
[freq_R, freq_R_bi] = getFreqWordsWithSW(X_train_R, 100, 50)


r_words = np.sort(getStem(['rente', 'ruhestand', 'rentenbeginn', 'altersrente',
                   'regelaltersrent', 'altersversorg', 'rentenalt', 'ruhestand',
                   'mehr arbeitsfah']))

poss_r_words = getBestCombinations(r_words, X_train, y_train_Rente, 'K', 'R')
   
# wähle die wenigstens Wörter, wenn mehrere: zufällig das erste
amount_word = []
for i in range(len(poss_r_words)):
    amount_word.append(len(poss_r_words[i]))
r_words_updated = poss_r_words[myEqual(amount_word, min(amount_word))[0]]
r_words_updated # ['altersrent',
 # 'mehr arbeitsfah',
 # 'regelaltersrent',
 # 'rent',
 # 'rentenbeginn',
 # 'ruhestand']

[cm_R, sensi_R, speci_R, accuracy_R, ypred_R] = getValues(r_words_updated, [], 
                                                           X_train, y_train_10_R,
                                                          1, 0)

#[cm_R, sensi_R, speci_R, accuracy_R] = getValuesClass(y_train_10_R, ypred_R)

# cm_R
# # array([[ 11,   2],
# #        [105, 761]])
# sensi_R # 0.8461538461538461
# speci_R #  0.8787528868360277
# accuracy_R # 0.8782707622298066


[cm_R_test, sensi_R_test, speci_R_test, accuracy_R_test, ypred_R_test] = getValues(r_words_updated, [], 
                                                           X_test, y_test_10_R,
                                                           1, 0)

# [cm_R_test, sensi_R_test, speci_R_test, accuracy_R_test] = getValuesClass(y_test_10_R, ypred_R_test)

# cm_R_test
# # array([[  2,   3],
# #        [ 53, 320]])
# sensi_R_test # 0.4
# speci_R_test # 0.8579088471849866
# accuracy_R_test # 0.8518518518518519

# =============================================================================
# Todesfall
# =============================================================================


### Neue Daten erstellen, für jeden Grund ein Datensatz, in dem z.B. nur die
### Renten Renten sind, der Rest alles 'K'

y_train_Todesfall = list(pd.DataFrame(y_train).replace(['F', 'B', 'R', 'S'], 'K')[0])
y_test_Todesfall = list(pd.DataFrame(y_test).replace(['F', 'B', 'R', 'S'], 'K')[0])

y_test_10_T = []
for i in range(len(y_test)):
    if y_test[i] == 'T':
        y_test_10_T.append(1)
    else:
        y_test_10_T.append(0)
        
y_train_10_T = []
for i in range(len(y_train)):
    if y_train[i] == 'T':
        y_train_10_T.append(1)
    else:
        y_train_10_T.append(0)

### Wortsuche
[freq_T, freq_T_bi] = getFreqWordsWithSW(X_train_T, 100, 50)


t_words = np.sort(getStem(['todesfall', 'gestorben', 'verstorben', 
                           'sterbeurkunde', 'sterbegeld', 'todesfallversicher']))

poss_t_words = getBestCombinations(t_words, X_train, y_train_Todesfall, 'K', 'T')
   
# wähle die wenigstens Wörter, wenn mehrere: zufällig das erste
amount_word = []
for i in range(len(poss_t_words)):
    amount_word.append(len(poss_t_words[i]))
t_words_updated = poss_t_words[myEqual(amount_word, min(amount_word))[0]]
t_words_updated # ['gestorb', 'verstorb']

[cm_T, sensi_T, speci_T, accuracy_T, ypred_T] = getValues(k_words = t_words_updated, 
                                                          nk_words = [], 
                                                          Xtr = X_train, 
                                                          ytr = y_train_10_T,
                                                          positiv = 1, 
                                                          negativ = 0)

# [cm_T, sensi_T, speci_T, accuracy_T] = getValuesClass(y_train_10_T, ypred_T)

# cm_T
# # array([[ 10,   1],
# #        [  0, 868]])
# sensi_T # 0.9090909090909091
# speci_T # 1.0
# accuracy_T # 0.9988623435722411

[cm_T_test, sensi_T_test, speci_T_test, accuracy_T_test, ypred_T_test] = getValues(t_words_updated, [], 
                                                           X_test, y_test_10_T,
                                                           1, 0)

#[cm_T_test, sensi_T_test, speci_T_test, accuracy_T_test] = getValuesClass(y_test_10_T, ypred_T_test)

# cm_T_test
# # array([[  3,   2],
# #        [  1, 372]])
# sensi_T_test # 0.6
# speci_T_test # 0.9973190348525469
# accuracy_T_test # 0.9920634920634921

os.chdir(r'W:\your_folder\Output')
dill.dump_session('ws_5B_4_28.pkl')
dill.load_session('ws_5B_4_28.pkl')
# =============================================================================
# =============================================================================
# # Oversampling (für Random Forest und SVM)
# =============================================================================
# =============================================================================

dtm_train_old = dtm_train
dtm_test_old = dtm_test
y_train_old = y_train
y_test_old = y_test
random.seed(1802)
sampl_strat = {'B': B_len * 4, 'F': F_len * 4, 'R': R_len * 4,  'T': T_len * 4}
ros = RandomOverSampler(random_state=0, sampling_strategy = sampl_strat)
dtm_train, y_train_all = ros.fit_resample(np.array(dtm_train_old.T), np.array(y_train))
# dtm_test, y_test_all = ros.fit_resample(np.array(dtm_test_old.T), np.array(y_test))
dtm_test = dtm_test_old.T
y_test_all = y_test

dtm_train = pd.DataFrame(dtm_train)
dtm_train.columns = dtm_train_old.T.columns
dtm_train = dtm_train.T.sort_index().T

# dtm_test = pd.DataFrame(dtm_test)
# dtm_test.columns = dtm_test_old.T.columns
dtm_test = dtm_test.T.sort_index().T

# meine indizes

features = f_words_updated + b_words_updated + r_words_updated + t_words_updated
dtm_trT = dtm_tr.T
dtm_train_kl = dtm_trT[features]

dtm_teT = dtm_te.T
dtm_test_klein = dtm_teT[features]

random.seed(1802)
ros = RandomOverSampler(random_state=0, sampling_strategy = sampl_strat)
dtm_train_klein, y_train_klein = ros.fit_resample(np.array(dtm_train_kl), np.array(y_train))
# dtm_test_klein, y_test_klein = ros.fit_resample(np.array(dtm_test_kl), np.array(y_test))
y_test_klein = y_test

dtm_train_klein = pd.DataFrame(dtm_train_klein)
dtm_train_klein.columns = features

# dtm_test_klein = pd.DataFrame(dtm_test_klein)
dtm_test_klein.columns = features

# =============================================================================
# Aufteilung auf die vier Gründe
# =============================================================================
y_test_10_F = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'F':
        y_test_10_F.append(1)
    else:
        y_test_10_F.append(0)
        
y_train_10_F = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'F':
        y_train_10_F.append(1)
    else:
        y_train_10_F.append(0)

y_test_10_R = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'R':
        y_test_10_R.append(1)
    else:
        y_test_10_R.append(0)
        
y_train_10_R = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'R':
        y_train_10_R.append(1)
    else:
        y_train_10_R.append(0)
        
y_test_10_B = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'B':
        y_test_10_B.append(1)
    else:
        y_test_10_B.append(0)
        
y_train_10_B = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'B':
        y_train_10_B.append(1)
    else:
        y_train_10_B.append(0)
        
y_test_10_T = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'T':
        y_test_10_T.append(1)
    else:
        y_test_10_T.append(0)
        
y_train_10_T = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'T':
        y_train_10_T.append(1)
    else:
        y_train_10_T.append(0)


### kleine 

y_test_10_F_klein = []
for i in range(len(y_test_klein)):
    if y_test_klein[i] == 'F':
        y_test_10_F_klein.append(1)
    else:
        y_test_10_F_klein.append(0)
        
y_train_10_F_klein = []
for i in range(len(y_train_klein)):
    if y_train_klein[i] == 'F':
        y_train_10_F_klein.append(1)
    else:
        y_train_10_F_klein.append(0)

y_test_10_R_klein = []
for i in range(len(y_test_klein)):
    if y_test_klein[i] == 'R':
        y_test_10_R_klein.append(1)
    else:
        y_test_10_R_klein.append(0)
        
y_train_10_R_klein = []
for i in range(len(y_train_klein)):
    if y_train_klein[i] == 'R':
        y_train_10_R_klein.append(1)
    else:
        y_train_10_R_klein.append(0)
        
y_test_10_B_klein = []
for i in range(len(y_test_klein)):
    if y_test_klein[i] == 'B':
        y_test_10_B_klein.append(1)
    else:
        y_test_10_B_klein.append(0)
        
y_train_10_B_klein = []
for i in range(len(y_train_klein)):
    if y_train_klein[i] == 'B':
        y_train_10_B_klein.append(1)
    else:
        y_train_10_B_klein.append(0)
        
y_test_10_T_klein = []
for i in range(len(y_test_klein)):
    if y_test_klein[i] == 'T':
        y_test_10_T_klein.append(1)
    else:
        y_test_10_T_klein.append(0)
        
y_train_10_T_klein = []
for i in range(len(y_train_klein)):
    if y_train_klein[i] == 'T':
        y_train_10_T_klein.append(1)
    else:
        y_train_10_T_klein.append(0)




# =============================================================================
# =============================================================================
# # Random Forest und SVM
# =============================================================================
# =============================================================================

# =============================================================================
# financial
# =============================================================================

### Random Forest =============================================================

### große Matrix
random.seed(1802)
[cm_train_RF_F, sensi_train_RF_F, speci_train_RF_F, accuracy_train_RF_F, 
 y_pred_train_RF_F, mydf_train_RF_F, ts_RF_F, cm_train_new_RF_F, sensi_train_new_RF_F, 
 speci_train_new_RF_F, accuracy_train_new_RF_F, y_pred_train_new_RF_F,
       cm_test_RF_F, sensi_test_RF_F, speci_test_RF_F, accuracy_test_RF_F, 
       y_pred_test_RF_F, mydf_test_RF_F] = RFcomplete('F', dtm_train, dtm_test, y_train_10_F, y_test_10_F, X_train_ind, X_test_ind)                

cm_train_RF_F
# array([[ 72,   0],
#        [  0, 981]])



cm_test_RF_F
# array([[  0,   8],
#        [  0, 370]])
sensi_test_RF_F
speci_test_RF_F
sensi_test_RF_F + speci_test_RF_F
accuracy_test_RF_F


### meine Token
random.seed(1802)
[cm_train_RF_F_kl, sensi_train_RF_F_kl, speci_train_RF_F_kl, accuracy_train_RF_F_kl, 
 y_pred_train_RF_F_kl, mydf_train_RF_F_kl, ts_RF_F_kl, cm_train_new_RF_F_kl, sensi_train_new_RF_F_kl, 
 speci_train_new_RF_F_kl, accuracy_train_new_RF_F_kl, y_pred_train_new_RF_F_kl,
       cm_test_RF_F_kl, sensi_test_RF_F_kl, speci_test_RF_F_kl, accuracy_test_RF_F_kl, 
       y_pred_test_RF_F_kl, mydf_test_RF_F_kl] = RFcomplete('F', dtm_train_klein, dtm_test_klein, y_train_10_F_klein, y_test_10_F_klein, X_train_ind, X_test_ind, False)                

cm_train_RF_F_kl
# array([[ 52,  20],
#        [  6, 975]])
sensi_train_RF_F_kl # 0.7222222222222222
speci_train_RF_F_kl # 0.9938837920489296
sensi_train_RF_F_kl + speci_train_RF_F_kl # 1.7161060142711517

cm_train_new_RF_F_kl
# array([[ 58,  14],
#        [ 44, 937]])
sensi_train_new_RF_F_kl # 0.8055555555555556
speci_train_new_RF_F_kl # 0.9551478083588175
sensi_train_new_RF_F_kl + speci_train_new_RF_F_kl # 1.7607033639143732


cm_test_RF_F_kl # besser als das Modell auf der großen Matrix
# array([[  2,   6],
#        [ 11, 359]])
sensi_test_RF_F_kl # 0.25
speci_test_RF_F_kl # 0.9702702702702702
sensi_test_RF_F_kl + speci_test_RF_F_kl # 1.2202702702702704
accuracy_test_RF_F_kl # 0.955026455026455


### SVM =======================================================================
 
### große Matrix 
random.seed(1802)
[cm_train_SVM_F, sensi_train_SVM_F, speci_train_SVM_F, accuracy_train_SVM_F, 
 y_pred_train_SVM_F, mydf_train_SVM_F, ts_SVM_F, cm_train_new_SVM_F, sensi_train_new_SVM_F, 
 speci_train_new_SVM_F, accuracy_train_new_SVM_F, y_pred_train_new_SVM_F,
       cm_test_SVM_F, sensi_test_SVM_F, speci_test_SVM_F, accuracy_test_SVM_F, 
       y_pred_test_SVM_F, mydf_test_SVM_F] = SVMcomplete('F', dtm_train, dtm_test, y_train_10_F, y_test_10_F, X_train_ind, X_test_ind)                

cm_train_SVM_F
# array([[ 72,   0],
#        [  0, 981]])


cm_test_SVM_F
# array([[  0,   8],
#        [  0, 370]])


### meine Token
random.seed(1802)
[cm_train_SVM_F_kl, sensi_train_SVM_F_kl, speci_train_SVM_F_kl, accuracy_train_SVM_F_kl, 
 y_pred_train_SVM_F_kl, mydf_train_SVM_F_kl, ts_SVM_F_kl, cm_train_new_SVM_F_kl, sensi_train_new_SVM_F_kl, 
 speci_train_new_SVM_F_kl, accuracy_train_new_SVM_F_kl, y_pred_train_new_SVM_F_kl,
       cm_test_SVM_F_kl, sensi_test_SVM_F_kl, speci_test_SVM_F_kl, accuracy_test_SVM_F_kl, 
       y_pred_test_SVM_F_kl, mydf_test_SVM_F_kl] = SVMcomplete('F', dtm_train_klein, dtm_test_klein, y_train_10_F_klein, y_test_10_F_klein, X_train_ind, X_test_ind, False)                

cm_train_SVM_F_kl
# array([[ 52,  20],
#        [  6, 975]])


sensi_train_SVM_F_kl # 0.7222222222222222
speci_train_SVM_F_kl # 0.9938837920489296
accuracy_train_SVM_F_kl # 0.9753086419753086

cm_test_SVM_F_kl # besser als das Modell auf der großen Matrix
# array([[  1,   7],
#        [  0, 370]])
sensi_test_SVM_F_kl # 0.125
speci_test_SVM_F_kl # 1.0
accuracy_test_SVM_F_kl #  0.9814814814814815

### das jeweils beste Modell der drei möglichen
# 1. WS, 2. RF, 3. SVM
cm_F_test
# array([[  2,   6],
#        [  3, 367]], dtype=int64)
sensi_F_test + speci_F_test # 1.2418918918918918
cm_test_RF_F_kl
# array([[  2,   6],
#        [ 11, 359]])
sensi_test_RF_F_kl + speci_test_RF_F_kl #  1.2202702702702704
cm_test_SVM_F_kl 
# array([[  1,   7],
#        [  0, 370]])
sensi_test_SVM_F_kl + speci_test_SVM_F_kl # 1.125

# ===========================================================================
# Berufswechsel
# ===========================================================================

### Random Forest =============================================================

### große Matrix
random.seed(1802)
[cm_train_RF_B, sensi_train_RF_B, speci_train_RF_B, accuracy_train_RF_B, 
 y_pred_train_RF_B, mydf_train_RF_B, ts_RF_B, cm_train_new_RF_B, sensi_train_new_RF_B, 
 speci_train_new_RF_B, accuracy_train_new_RF_B, y_pred_train_new_RF_B,
       cm_test_RF_B, sensi_test_RF_B, speci_test_RF_B, accuracy_test_RF_B, 
       y_pred_test_RF_B, mydf_test_RF_B] = RFcomplete('B', dtm_train, dtm_test, y_train_10_B, y_test_10_B, X_train_ind, X_test_ind)                

cm_train_RF_B 
# array([[ 64,   0],
#        [  0, 989]])


cm_test_RF_B 
# array([[  0,   7],
#        [  0, 371]])

### meine Token
random.seed(1802)
[cm_train_RF_B_kl, sensi_train_RF_B_kl, speci_train_RF_B_kl, accuracy_train_RF_B_kl, 
 y_pred_train_RF_B_kl, mydf_train_RF_B_kl, ts_RF_B_kl, cm_train_new_RF_B_kl, sensi_train_new_RF_B_kl, 
 speci_train_new_RF_B_kl, accuracy_train_new_RF_B_kl, y_pred_train_new_RF_B_kl,
       cm_test_RF_B_kl, sensi_test_RF_B_kl, speci_test_RF_B_kl, accuracy_test_RF_B_kl, 
       y_pred_test_RF_B_kl, mydf_test_RF_B_kl] = RFcomplete('B', dtm_train_klein, dtm_test_klein, y_train_10_B_klein, y_test_10_B_klein, X_train_ind, X_test_ind, False)                

cm_train_RF_B_kl
# array([[ 13,  51],
#        [  2, 987]])
sensi_train_RF_B_kl # 0.203125
speci_train_RF_B_kl #0.9979777553083923
accuracy_train_new_RF_B_kl #0.936372269705603

cm_train_new_RF_B_kl
# array([[ 38,  26],
#        [ 41, 948]])

sensi_train_new_RF_B_kl # 0.59375
speci_train_new_RF_B_kl # 0.9585439838220424
accuracy_train_new_RF_B_kl # 0.936372269705603


cm_test_RF_B_kl
# array([[  0,   7],
#        [  5, 366]])
sensi_test_RF_B_kl
speci_test_RF_B_kl #  0.9865229110512129
sensi_test_RF_B_kl + speci_test_RF_B_kl
accuracy_test_RF_B_kl

### SVM =======================================================================
 
### große Matrix 
random.seed(1802)
[cm_train_SVM_B, sensi_train_SVM_B, speci_train_SVM_B, accuracy_train_SVM_B, 
 y_pred_train_SVM_B, mydf_train_SVM_B, ts_SVM_B, cm_train_new_SVM_B, sensi_train_new_SVM_B, 
 speci_train_new_SVM_B, accuracy_train_new_SVM_B, y_pred_train_new_SVM_B,
       cm_test_SVM_B, sensi_test_SVM_B, speci_test_SVM_B, accuracy_test_SVM_B, 
       y_pred_test_SVM_B, mydf_test_SVM_B] = SVMcomplete('B', dtm_train, dtm_test, y_train_10_B, y_test_10_B, X_train_ind, X_test_ind)                

cm_train_SVM_B
# array([[ 64,   0],
#        [  0, 989]])
cm_train_new_SVM_B

cm_test_SVM_B 
# array([[  0,   7],
#        [  0, 371]])

sensi_test_SVM_B
speci_test_SVM_B
sensi_test_SVM_B + speci_test_SVM_B
accuracy_test_SVM_B


### meine Token  
random.seed(1802)
[cm_train_SVM_B_kl, sensi_train_SVM_B_kl, speci_train_SVM_B_kl, accuracy_train_SVM_B_kl, 
 y_pred_train_SVM_B_kl, mydf_train_SVM_B_kl, ts_SVM_B_kl, cm_train_new_SVM_B_kl, sensi_train_new_SVM_B_kl, 
 speci_train_new_SVM_B_kl, accuracy_train_new_SVM_B_kl, y_pred_train_new_SVM_B_kl,
       cm_test_SVM_B_kl, sensi_test_SVM_B_kl, speci_test_SVM_B_kl, accuracy_test_SVM_B_kl, 
       y_pred_test_SVM_B_kl, mydf_test_SVM_B_kl] = SVMcomplete('B', dtm_train_klein, dtm_test_klein, y_train_10_B_klein, y_test_10_B_klein, X_train_ind, X_test_ind, False)                

cm_train_SVM_B_kl
# array([[ 13,  51],
#        [  2, 987]])
sensi_train_SVM_B_kl # 0.203125
speci_train_SVM_B_kl # 0.9979777553083923
accuracy_train_SVM_B_kl # 0.949667616334283

cm_train_new_SVM_B_kl
# array([[ 38,  26],
#        [249, 740]])
sensi_train_new_SVM_B_kl # 0.59375
speci_train_new_SVM_B_kl # 0.7482305358948432
accuracy_train_new_SVM_B_kl # 0.7388414055080722

cm_test_SVM_B_kl # TODO HIER IST WAS FALSCH
# array([[  0,  64],
#        [ 18, 503]])
sensi_test_SVM_B_kl
speci_test_SVM_B_kl #  0.9654510556621881
sensi_test_SVM_B_kl + speci_test_SVM_B_kl
accuracy_test_SVM_B_kl

## beste Modelle
# 1. WS, 2. RF/ SVM
# beim RF und SVM ist das große Modell, dass alles als keinen Grund klassifiziert
# am besten (Summe aus Sensi & speci = 1)
# Wortsuche:
cm_B_test
# array([[  1,   6],
#        [ 16, 355]], dtype=int64)
sensi_B_test + speci_B_test # 1.0997304582210243   

# ===========================================================================
# Rente
# ===========================================================================

### Random Forest =============================================================

### große Matrix
random.seed(1802)
[cm_train_RF_R, sensi_train_RF_R, speci_train_RF_R, accuracy_train_RF_R, 
 y_pred_train_RF_R, mydf_train_RF_R, ts_RF_R, cm_train_new_RF_R, sensi_train_new_RF_R, 
 speci_train_new_RF_R, accuracy_train_new_RF_R, y_pred_train_new_RF_R,
       cm_test_RF_R, sensi_test_RF_R, speci_test_RF_R, accuracy_test_RF_R, 
       y_pred_test_RF_R, mydf_test_RF_R] = RFcomplete('R', dtm_train, dtm_test, y_train_10_R, y_test_10_R, X_train_ind, X_test_ind)                

cm_train_RF_R 
# array([[  52,    0],
#        [   0, 1001]])


cm_test_RF_R
# array([[  0,   5],
#        [  0, 373]])
sensi_test_RF_R
speci_test_RF_R
sensi_test_RF_R + speci_test_RF_R
accuracy_test_RF_R


### meine Token
random.seed(1802)
[cm_train_RF_R_kl, sensi_train_RF_R_kl, speci_train_RF_R_kl, accuracy_train_RF_R_kl, 
 y_pred_train_RF_R_kl, mydf_train_RF_R_kl, ts_RF_R_kl, cm_train_new_RF_R_kl, sensi_train_new_RF_R_kl, 
 speci_train_new_RF_R_kl, accuracy_train_new_RF_R_kl, y_pred_train_new_RF_R_kl,
       cm_test_RF_R_kl, sensi_test_RF_R_kl, speci_test_RF_R_kl, accuracy_test_RF_R_kl, 
       y_pred_test_RF_R_kl, mydf_test_RF_R_kl] = RFcomplete('R', dtm_train_klein, dtm_test_klein, y_train_10_R_klein, y_test_10_R_klein, X_train_ind, X_test_ind, False)                

cm_train_RF_R_kl
# array([[  34,   18],
#        [   0, 1001]])
sensi_train_RF_R_kl # 0.6538461538461539
speci_train_RF_R_kl # 1.0
accuracy_train_RF_R_kl # 0.9829059829059829

cm_train_new_RF_R_kl
# array([[ 47,   5],
#        [ 84, 917]])
sensi_train_new_RF_R_kl # 0.9038461538461539
speci_train_new_RF_R_kl # 0.916083916083916
accuracy_train_new_RF_R_kl # 0.9154795821462488


cm_test_RF_R_kl # besser als im großen Modell
# array([[  2,   3],
#        [  2, 371]])
sensi_test_RF_R_kl # 0.4
speci_test_RF_R_kl # 0.9946380697050938
sensi_test_RF_R_kl + speci_test_RF_R_kl # 1.3946380697050937
accuracy_test_RF_R_kl # 0.9867724867724867 


### SVM =======================================================================
 
### große Matrix 
random.seed(1802)
[cm_train_SVM_R, sensi_train_SVM_R, speci_train_SVM_R, accuracy_train_SVM_R, 
 y_pred_train_SVM_R, mydf_train_SVM_R, ts_SVM_R, cm_train_new_SVM_R, sensi_train_new_SVM_R, 
 speci_train_new_SVM_R, accuracy_train_new_SVM_R, y_pred_train_new_SVM_R,
       cm_test_SVM_R, sensi_test_SVM_R, speci_test_SVM_R, accuracy_test_SVM_R, 
       y_pred_test_SVM_R, mydf_test_SVM_R] = SVMcomplete('R', dtm_train, dtm_test, y_train_10_R, y_test_10_R, X_train_ind, X_test_ind)                

cm_train_SVM_R
# array([[  52,    0],
#        [   0, 1001]])


cm_test_SVM_R 
# array([[  0,   5],
#        [  0, 373]])
sensi_test_SVM_R
speci_test_SVM_R
sensi_test_SVM_R + speci_test_SVM_R
accuracy_test_SVM_R


### meine Token
random.seed(1802)
[cm_train_SVM_R_kl, sensi_train_SVM_R_kl, speci_train_SVM_R_kl, accuracy_train_SVM_R_kl, 
 y_pred_train_SVM_R_kl, mydf_train_SVM_R_kl, ts_SVM_R_kl, cm_train_new_SVM_R_kl, sensi_train_new_SVM_R_kl, 
 speci_train_new_SVM_R_kl, accuracy_train_new_SVM_R_kl, y_pred_train_new_SVM_R_kl,
       cm_test_SVM_R_kl, sensi_test_SVM_R_kl, speci_test_SVM_R_kl, accuracy_test_SVM_R_kl, 
       y_pred_test_SVM_R_kl, mydf_test_SVM_R_kl] = SVMcomplete('R', dtm_train_klein, dtm_test_klein, y_train_10_R_klein, y_test_10_R_klein, X_train_ind, X_test_ind, False)                

cm_train_SVM_R_kl
# array([[  34,   18],
#        [   0, 1001]])
sensi_train_SVM_R_kl # 0.6538461538461539
speci_train_SVM_R_kl # 1.0
accuracy_train_SVM_R_kl # 0.9829059829059829

cm_test_SVM_R_kl # besser als bei den großen
# array([[  2,   3],
#        [  0, 373]])
sensi_test_SVM_R_kl # 0.1
speci_test_SVM_R_kl # 1
sensi_test_SVM_R_kl + speci_test_SVM_R_kl # 1.4
accuracy_test_SVM_R_kl # 0.9920634920634921

## beste Modelle
# 1. SVM, 2. RF, 3. WS
cm_R_test
sensi_R_test + speci_R_test  # 1.2579088471849866
cm_test_RF_R_kl
speci_test_RF_R_kl + sensi_test_RF_R_kl #  1.3946380697050937
cm_test_SVM_R_kl
sensi_test_SVM_R_kl + speci_test_SVM_R_kl # 1.4

# =============================================================================
# Todesfall
# =============================================================================

### Random Forest =============================================================

### große Matrix
random.seed(1802)
[cm_train_RF_T, sensi_train_RF_T, speci_train_RF_T, accuracy_train_RF_T, 
 y_pred_train_RF_T, mydf_train_RF_T, ts_RF_T, cm_train_new_RF_T, sensi_train_new_RF_T, 
 speci_train_new_RF_T, accuracy_train_new_RF_T, y_pred_train_new_RF_T,
       cm_test_RF_T, sensi_test_RF_T, speci_test_RF_T, accuracy_test_RF_T, 
       y_pred_test_RF_T, mydf_test_RF_T] = RFcomplete('T', dtm_train, dtm_test, y_train_10_T, y_test_10_T, X_train_ind, X_test_ind)                

cm_train_RF_T
# array([[  44,    0],
#        [   0, 1009]])

cm_test_RF_T
# array([[  0,   5],
#        [  0, 373]])
sensi_test_RF_T
speci_test_RF_T
sensi_test_RF_T + speci_test_RF_T
accuracy_test_RF_T


### meine Token
random.seed(1802)
[cm_train_RF_T_kl, sensi_train_RF_T_kl, speci_train_RF_T_kl, accuracy_train_RF_T_kl, 
 y_pred_train_RF_T_kl, mydf_train_RF_T_kl, ts_RF_T_kl, cm_train_new_RF_T_kl, sensi_train_new_RF_T_kl, 
 speci_train_new_RF_T_kl, accuracy_train_new_RF_T_kl, y_pred_train_new_RF_T_kl,
       cm_test_RF_T_kl, sensi_test_RF_T_kl, speci_test_RF_T_kl, accuracy_test_RF_T_kl, 
       y_pred_test_RF_T_kl, mydf_test_RF_T_kl] = RFcomplete('T', dtm_train_klein, dtm_test_klein, y_train_10_T_klein, y_test_10_T_klein, X_train_ind, X_test_ind, False)                

cm_train_RF_T_kl
# array([[  43,    1],
#        [   0, 1009]])
sensi_train_RF_T_kl # 0.9772727272727273
speci_train_RF_T_kl # 1.0
accuracy_train_RF_T_kl # 0.9990503323836657

cm_test_RF_T_kl # besser als die große Matrix
# array([[  3,   2],
#        [  1, 372]])
sensi_test_RF_T_kl # 0.6
speci_test_RF_T_kl # 0.9973190348525469
sensi_test_RF_T_kl + speci_test_RF_T_kl # 1.5973190348525468
accuracy_test_RF_T_kl # 0.9920634920634921

### SVM =======================================================================
 
### große Matrix 
random.seed(1802)
[cm_train_SVM_T, sensi_train_SVM_T, speci_train_SVM_T, accuracy_train_SVM_T, 
 y_pred_train_SVM_T, mydf_train_SVM_T, ts_SVM_T, cm_train_new_SVM_T, sensi_train_new_SVM_T, 
 speci_train_new_SVM_T, accuracy_train_new_SVM_T, y_pred_train_new_SVM_T,
       cm_test_SVM_T, sensi_test_SVM_T, speci_test_SVM_T, accuracy_test_SVM_T, 
       y_pred_test_SVM_T, mydf_test_SVM_T] = SVMcomplete('T', dtm_train, dtm_test, y_train_10_T, y_test_10_T, X_train_ind, X_test_ind)                

cm_train_SVM_T
# array([[  44,    0],
#        [   0, 1009]])


cm_test_SVM_T # beide kein Grund
# array([[  0,   5],
#        [  0, 373]])
sensi_test_SVM_T
speci_test_SVM_T
sensi_test_SVM_T + speci_test_SVM_T
accuracy_test_SVM_T


### meine Token
random.seed(1802)
[cm_train_SVM_T_kl, sensi_train_SVM_T_kl, speci_train_SVM_T_kl, accuracy_train_SVM_T_kl, 
 y_pred_train_SVM_T_kl, mydf_train_SVM_T_kl, ts_SVM_T_kl, cm_train_new_SVM_T_kl, sensi_train_new_SVM_T_kl, 
 speci_train_new_SVM_T_kl, accuracy_train_new_SVM_T_kl, y_pred_train_new_SVM_T_kl,
       cm_test_SVM_T_kl, sensi_test_SVM_T_kl, speci_test_SVM_T_kl, accuracy_test_SVM_T_kl, 
       y_pred_test_SVM_T_kl, mydf_test_SVM_T_kl] = SVMcomplete('T', dtm_train_klein, dtm_test_klein, y_train_10_T_klein, y_test_10_T_klein, X_train_ind, X_test_ind, False)                

cm_train_SVM_T_kl
# array([[  43,    1],
#        [   0, 1009]])

cm_test_SVM_T_kl 
# array([[  3,   2],
#        [  1, 372]])
sensi_test_SVM_T_kl
speci_test_SVM_T_kl
sensi_test_SVM_T_kl + speci_test_SVM_T_kl
accuracy_test_SVM_T_kl

# beste Modelle
# alle drei gleich

cm_T_test
# array([[  3,   2],
#        [  1, 372]])
sensi_T_test + speci_T_test # 1.5973190348525468
cm_test_RF_T_kl
# array([[  3,   2],
#        [  1, 372]])
sensi_test_RF_T_kl + speci_test_RF_T_kl #  1.5973190348525468
cm_test_SVM_T_kl
# array([[  3,   2],
#        [  1, 372]])

os.chdir(r'W:\your_folder\Output')
dill.dump_session('vorvgl_5B_04_29_03.pkl')
#dill.load_session('vorvgl_5B_04_29_02.pkl')

### alle thresholds:
ts_RF_F
ts_RF_F_kl
ts_RF_B
ts_RF_B_kl
ts_RF_R
ts_RF_R_kl
ts_RF_T
ts_RF_T_kl

ts_SVM_F
ts_SVM_F_kl
ts_SVM_B
ts_SVM_B_kl
ts_SVM_R
ts_SVM_R_kl
ts_SVM_T
ts_SVM_T_kl


ts_RF_F
# OUt[57]: 0.5498307717013564

ts_RF_F_kl
# OUt[58]: 0.14560109321478149

ts_RF_B
# OUt[59]: 0.5664887599937336

ts_RF_B_kl
# OUt[60]: 0.3490324881235741

ts_RF_R
# OUt[61]: 0.42320532731256444

ts_RF_R_kl
# OUt[62]: 0.10787467907495621

ts_RF_T
# OUt[63]: 0.436527812678036

ts_RF_T_kl
# OUt[64]: 1.0

ts_SVM_F
# OUt[65]: 0.6637802823758071

ts_SVM_F_kl
# OUt[66]: 0.5675058140507078

ts_SVM_B
# OUt[67]: 0.666959995753519

ts_SVM_B_kl
# OUt[68]: 0.05381885382301271

ts_SVM_R
# OUt[69]: 0.40954741851139737

ts_SVM_R_kl
# OUt[70]: 0.969051003048289

ts_SVM_T
# OUt[71]: 0.5702134788421716

ts_SVM_T_kl
# OUt[72]: 0.9781792928645842
# =============================================================================
# Übersichten für jedes Modell
# =============================================================================

# beachte: Spaltensummen ergeben nicht unbedingt anzahl an z.B. Kündigungen
# mit finanziellem Grund, da Dokumente zu mehreren Gründen
# zugeordnet werden können

### Wortsuche =================================================================

wortsuche_erg = getResults(ypred_F, ypred_B, ypred_R, ypred_T, X_train, y_train)
#         F klass  B klass  R klass  T klass  K klass
# F wahr       13        3        1        0        4
# B wahr        0        9        0        0        7
# R wahr        0        5       11        0        1
# T wahr        0        0        0       10        1
# K wahr       10       40      104        0      675


wortsuche_erg_test = getResults(ypred_F_test, ypred_B_test, ypred_R_test,
                                    ypred_T_test, X_test, y_test)
#         F klass  B klass  R klass  T klass  K klass
# F wahr        2        0        2        0        5
# B wahr        0        1        1        0        5
# R wahr        0        0        2        0        3
# T wahr        0        0        0        3        2
# K wahr        3       16       50        1      288

### Random Forest =============================================================
# klein & groß: 1053
## Train
# groß


RF_erg_gr = getResults(y_pred_train_RF_F, y_pred_train_RF_B,
                          y_pred_train_RF_R, y_pred_train_RF_T,
                          dtm_train, y_train_all)
#         F klass  B klass  R klass  T klass  K klass
# F wahr       72        0        0        0        0
# B wahr        0       64        0        0        0
# R wahr        0        0       52        0        0
# T wahr        0        0        0       44        0
# K wahr        0        0        0        0      821

# stimmt: alles wird als richtig klassifiziert

# klein
RF_erg_kl = getResults(y_pred_train_RF_F_kl, y_pred_train_RF_B_kl,
                          y_pred_train_RF_R_kl, y_pred_train_RF_T_kl,
                          dtm_train_klein, y_train_klein)

#         F klass  B klass  R klass  T klass  K klass
# F wahr       52        0        0        0       20
# B wahr        0       13        0        0       51
# R wahr        0        0       34        0       18
# T wahr        0        0        0       43        1
# K wahr        6        2        0        0      813


## Test
# groß
RF_erg_test_gr = getResults(y_pred_test_RF_F, y_pred_test_RF_B,
                          y_pred_test_RF_R, y_pred_test_RF_T,
                          dtm_test, y_test_all)
#         F klass  B klass  R klass  T klass  K klass
# F wahr        0        0        0        0        8
# B wahr        0        0        0        0        7
# R wahr        0        0        0        0        5
# T wahr        0        0        0        0        5
# K wahr        0        0        0        0      353

# stimmt: für jeden GeVo wird alles als 'K' klassifiziert

# klein
RF_erg_test_kl = getResults(y_pred_test_RF_F_kl, y_pred_test_RF_B_kl,
                          y_pred_test_RF_R_kl, y_pred_test_RF_T_kl,
                           dtm_test_klein, y_test_klein)
#         F klass  B klass  R klass  T klass  K klass
# F wahr        2        0        0        0        6
# B wahr        1        0        0        0        6
# R wahr        0        0        2        0        3
# T wahr        0        0        0        3        2
# K wahr       10        5        2        1      336

# stimmt

### SVM =======================================================================

## Train
# groß
SVM_erg_gr = getResults(y_pred_train_SVM_F, y_pred_train_SVM_B,
                          y_pred_train_SVM_R, y_pred_train_SVM_T,
                          dtm_train, y_train_all)
#         F klass  B klass  R klass  T klass  K klass
# F wahr       72        0        0        0        0
# B wahr        0       64        0        0        0
# R wahr        0        0       52        0        0
# T wahr        0        0        0       44        0
# K wahr        0        0        0        0      821

# stimmt: alles wird als richtig klassifiziert

# klein
SVM_erg_kl = getResults(y_pred_train_SVM_F_kl, y_pred_train_SVM_B_kl, 
                           y_pred_train_SVM_R_kl, y_pred_train_SVM_T_kl,
                           dtm_train_klein, y_train_klein)
#         F klass  B klass  R klass  T klass  K klass
# F wahr       52        0        0        0       20
# B wahr        0       13        0        0       51
# R wahr        0        0       34        0       18
# T wahr        0        0        0       43        1
# K wahr        6        2        0        0      813

# stimmt

## Test
# groß
SVM_erg_test_gr = getResults(y_pred_test_SVM_F, y_pred_test_SVM_B,
                          y_pred_test_SVM_R, y_pred_test_SVM_T,
                          dtm_test, y_test_all)
#         F klass  B klass  R klass  T klass  K klass
# F wahr        0        0        0        0        8
# B wahr        0        0        0        0        7
# R wahr        0        0        0        0        5
# T wahr        0        0        0        0        5
# K wahr        0        0        0        0      353

# stimmt

# klein
SVM_erg_test_kl = getResults(y_pred_test_SVM_F_kl, y_pred_test_SVM_B_kl,
                          y_pred_test_SVM_R_kl, y_pred_test_SVM_T_kl,
                          dtm_test_klein, y_test_klein)
#         F klass  B klass  R klass  T klass  K klass
# F wahr        7        9        0        0       56
# B wahr        0        0        0        0       64
# R wahr        0        0       17        0       35
# T wahr        0        0        0        0       44
# K wahr        0        9        0        0      344

# stimmt

### Der Vergleich erfolgt zwischen der Wortsuche und den kleinen
### Matrizen bei sowohl SVM als auch RF



# Wortsuche
sensis_WS = [sensi_F_test, sensi_B_test, sensi_R_test, sensi_T_test]
# [0.25, 0.14285714285714285, 0.4, 0.6]
specis_WS = [speci_F_test, speci_B_test, speci_R_test, speci_T_test]
# [0.9918918918918919, 0.9568733153638814, 0.8579088471849866, 0.9973190348525469]

# RF
sensis_RF = [sensi_test_RF_F_kl, sensi_test_RF_B_kl, sensi_test_RF_R_kl, sensi_test_RF_T_kl]
# [0.25, 0.0, 0.4, 0.6]
specis_RF = [speci_test_RF_F_kl, speci_test_RF_B_kl, speci_test_RF_R_kl, speci_test_RF_T_kl]
# [0.9702702702702702,
#  0.9865229110512129,
#  0.9946380697050938,
#  0.9973190348525469]

# SVM
sensis_SVM = [sensi_test_SVM_F_kl, sensi_test_SVM_B_kl, sensi_test_SVM_R_kl, sensi_test_SVM_T_kl]
# [0.125, 1.0, 0.4, 0.6]
specis_SVM = [speci_test_SVM_F_kl, sensi_test_SVM_B_kl, sensi_test_SVM_R_kl, sensi_test_SVM_T_kl]
# [1.0, 1.0, 0.4, 0.6]

os.chdir(r'W:\your_folder\Output')
dill.dump_session('5B_04_29_02.pkl')
# dill.load_session('5B_04_29.pkl')

# Insgesamt ist 3 mal die Wortsuche das beste Modell, daher wird sich für
# die Wortsuche entschieden.
