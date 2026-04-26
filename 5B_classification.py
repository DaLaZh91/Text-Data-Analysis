# This file applies the classification methods.

# =============================================================================
# Load packages & include functions
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
# https://laurenliz22.github.io/nlp_random_forest_and_neural_network_smassifiers 

os.chdir(r'W:\your_folder\Python')
from functions import *

# =============================================================================
# Read data 
# =============================================================================

os.chdir(r'W:\your_folder\Output')
dill.load_session('4B_04_28.pkl')

stop_words = get_stop_words('de')

# Since the reasons are considered individually, each reason could theoretically
# be handled separately. However, because the words from the word search are needed,
# all four word searches are performed first, followed by the classification models,
# separated by reasons.

# =============================================================================
# =============================================================================
# # Wordsearches 
# =============================================================================
# =============================================================================

# =============================================================================
# Financial
# =============================================================================

### Create new data for every reason in the dataset, whith i.e. only retirements and cancellations 

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
                
        
### Wordsearch    

### Since there are only very few words per reason due to the small number of terminations,
### it is difficult to create a word cloud without names

### Instead, the frequently occurring words and bigrams are examined directly,
### and those related to terminations for financial reasons are selected

[freq_F, freq_F_bi] = getFreqWordsWithSW(X_train_F, 200, 100)

f_words = np.sort(['finanziell', 'insolvenzverwalt', 'engpass', 'wirtschaft',
                    'finanziell engpass', 'gezwung', 'insolvenzverfahr', 
                    'immobiliendarlehn', 'einbuss', 'eigentum', 'wenig lohn'])


poss_f_words = getBestCombinations(f_words, X_train, y_train_financial, 'F')

# choose the rarest words, if there are more then one: choose the first one
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
# Job change
# =============================================================================

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
        
### Wordsearch
[freq_J, freq_J_bi] = getFreqWordsWithSW(X_train_J, 200, 100)

j_words = np.sort(getStem(['beschaftigt', 'betrieb', 'alt arbeitgeb', 'arbeitet',
                           'mitarbeiterin', 'mitarbeit', 'aufheb vertrag']))

poss_j_words = getBestCombinations(j_words, X_train, y_train_jobchange, 'J')
   
amount_word = []
for i in range(len(poss_j_words)):
    amount_word.append(len(poss_j_words[i]))
j_words_updated = poss_j_words[myEqual(amount_word, min(amount_word))[0]]
j_words_updated # ['alt arbeitgeb', 'aufheb vertrag', 'beschaftigt', 'betrieb', 'mitarbeiterin']

[cm_J, sensi_J, speci_J, accuracy_J, ypred_J] = getValues(j_words_updated, [], 
                                                           X_train, y_train_jobchange,
                                                           'J', 'C')

cm_J
# array([[  9,   7],
#        [ 48, 815]], dtype=int64)
# sensi_J # 0.5625
# speci_J # 0.944380069524913
# accuracy_J # 0.9374288964732651

[cm_J_test, sensi_J_test, speci_J_test, accuracy_J_test, ypred_J_test] = getValues(j_words_updated, [], 
                                                           X_test, y_test_jobchange,
                                                           'J', 'C')

cm_J_test
# array([[  1,   6],
#        [ 16, 355]], dtype=int64)
# sensi_J_test # 0.14285714285714285
# speci_J_test # 0.9568733153638814
# accuracy_J_test # 0.9417989417989417


# =============================================================================
# Retirement
# =============================================================================


y_train_Retirement = list(pd.DataFrame(y_train).replace(['J', 'F', 'D', 'O'], 'C')[0])
y_test_Retirement = list(pd.DataFrame(y_test).replace(['J', 'F', 'D', 'O'], 'C')[0])


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
        
### Wordsearch
[freq_R, freq_R_bi] = getFreqWordsWithSW(X_train_R, 100, 50)


r_words = np.sort(getStem(['rente', 'ruhestand', 'rentenbeginn', 'altersrente',
                   'regelaltersrent', 'altersversorg', 'rentenalt', 'ruhestand',
                   'mehr arbeitsfah']))

poss_r_words = getBestCombinations(r_words, X_train, y_train_Rente, 'C', 'R')
   
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
# Death
# =============================================================================


y_train_Death = list(pd.DataFrame(y_train).replace(['F', 'J', 'R', 'O'], 'C')[0])
y_test_Death = list(pd.DataFrame(y_test).replace(['F', 'J', 'R', 'O'], 'C')[0])

y_test_10_D = []
for i in range(len(y_test)):
    if y_test[i] == 'D':
        y_test_10_D.append(1)
    else:
        y_test_10_D.append(0)
        
y_train_10_D = []
for i in range(len(y_train)):
    if y_train[i] == 'D':
        y_train_10_D.append(1)
    else:
        y_train_10_D.append(0)

### Wordsearch
[freq_D, freq_D_bi] = getFreqWordsWithSW(X_train_D, 100, 50)


d_words = np.sort(getStem(['todesfall', 'gestorben', 'verstorben', 
                           'sterbeurkunde', 'sterbegeld', 'todesfallversicher']))

poss_d_words = getBestCombinations(d_words, X_train, y_train_Death, 'C', 'D')
   
amount_word = []
for i in range(len(poss_d_words)):
    amount_word.append(len(poss_d_words[i]))
d_words_updated = poss_d_words[myEqual(amount_word, min(amount_word))[0]]
d_words_updated # ['gestorb', 'verstorb']

[cm_D, sensi_D, speci_D, accuracy_D, ypred_D] = getValues(c_words = d_words_updated, 
                                                          nc_words = [], 
                                                          Xtr = X_train, 
                                                          ytr = y_train_10_D,
                                                          positiv = 1, 
                                                          negativ = 0)

# [cm_D, sensi_D, speci_D, accuracy_D] = getValuesClass(y_train_10_D, ypred_D)

# cm_D
# # array([[ 10,   1],
# #        [  0, 868]])
# sensi_D # 0.9090909090909091
# speci_D # 1.0
# accuracy_D # 0.9988623435722411

[cm_D_test, sensi_D_test, speci__test, accuracy_D_test, ypred_D_test] = getValues(d_words_updated, [], 
                                                           X_test, y_test_10_D,
                                                           1, 0)

#[cm_D_test, sensi_D_test, speci_D_test, accuracy_D_test] = getValuesClass(y_test_10_D, ypred_D_test)

# cm_D_test
# # array([[  3,   2],
# #        [  1, 372]])
# sensi_D_test # 0.6
# speci_D_test # 0.9973190348525469
# accuracy_D_test # 0.9920634920634921

os.chdir(r'W:\your_folder\Output')
dill.dump_session('ws_5B_4_28.pkl')
dill.load_session('ws_5B_4_28.pkl')
# =============================================================================
# =============================================================================
# # Oversampling (for Random Forest and SVM)
# =============================================================================
# =============================================================================

dtm_train_old = dtm_train
dtm_test_old = dtm_test
y_train_old = y_train
y_test_old = y_test
random.seed(1802)
sampl_strat = {'J': J_len * 4, 'F': F_len * 4, 'R': R_len * 4,  'D': D_len * 4}
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

# my indices

features = f_words_updated + b_words_updated + r_words_updated + t_words_updated
dtm_trT = dtm_tr.T
dtm_train_sm = dtm_trT[features]

dtm_teT = dtm_te.T
dtm_test_small = dtm_teT[features]

random.seed(1802)
ros = RandomOverSampler(random_state=0, sampling_strategy = sampl_strat)
dtm_train_small, y_train_small = ros.fit_resample(np.array(dtm_train_sm), np.array(y_train))
# dtm_test_small, y_test_small = ros.fit_resample(np.array(dtm_test_sm), np.array(y_test))
y_test_small = y_test

dtm_train_small = pd.DataFrame(dtm_train_small)
dtm_train_small.columns = features

# dtm_test_small = pd.DataFrame(dtm_test_small)
dtm_test_small.columns = features

# =============================================================================
# Separation into the four reasons
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
        
y_test_10_J = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'J':
        y_test_10_J.append(1)
    else:
        y_test_10_J.append(0)
        
y_train_10_J = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'J':
        y_train_10_J.append(1)
    else:
        y_train_10_J.append(0)
        
y_test_10_D = []
for i in range(len(y_test_all)):
    if y_test_all[i] == 'D':
        y_test_10_D.append(1)
    else:
        y_test_10_D.append(0)
        
y_train_10_D = []
for i in range(len(y_train_all)):
    if y_train_all[i] == 'D':
        y_train_10_D.append(1)
    else:
        y_train_10_D.append(0)


### small 

y_test_10_F_small = []
for i in range(len(y_test_small)):
    if y_test_small[i] == 'F':
        y_test_10_F_small.append(1)
    else:
        y_test_10_F_small.append(0)
        
y_train_10_F_small = []
for i in range(len(y_train_small)):
    if y_train_small[i] == 'F':
        y_train_10_F_small.append(1)
    else:
        y_train_10_F_small.append(0)

y_test_10_R_small = []
for i in range(len(y_test_small)):
    if y_test_small[i] == 'R':
        y_test_10_R_small.append(1)
    else:
        y_test_10_R_small.append(0)
        
y_train_10_R_small = []
for i in range(len(y_train_small)):
    if y_train_small[i] == 'R':
        y_train_10_R_small.append(1)
    else:
        y_train_10_R_small.append(0)
        
y_test_10_J_small = []
for i in range(len(y_test_small)):
    if y_test_small[i] == 'J':
        y_test_10_J_small.append(1)
    else:
        y_test_10_J_small.append(0)
        
y_train_10_J_small = []
for i in range(len(y_train_small)):
    if y_train_small[i] == 'J':
        y_train_10_J_small.append(1)
    else:
        y_train_10_J_small.append(0)
        
y_test_10_D_small = []
for i in range(len(y_test_small)):
    if y_test_small[i] == 'D':
        y_test_10_D_small.append(1)
    else:
        y_test_10_D_small.append(0)
        
y_train_10_D_small = []
for i in range(len(y_train_small)):
    if y_train_small[i] == 'D':
        y_train_10_D_small.append(1)
    else:
        y_train_10_D_small.append(0)




# =============================================================================
# =============================================================================
# # Random Forest and SVM
# =============================================================================
# =============================================================================

# =============================================================================
# financial
# =============================================================================

### Random Forest =============================================================

### big matrix
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


### my tokens
random.seed(1802)
[cm_train_RF_F_sm, sensi_train_RF_F_sm, speci_train_RF_F_sm, accuracy_train_RF_F_sm, 
 y_pred_train_RF_F_sm, mydf_train_RF_F_sm, ts_RF_F_sm, cm_train_new_RF_F_sm, sensi_train_new_RF_F_sm, 
 speci_train_new_RF_F_sm, accuracy_train_new_RF_F_sm, y_pred_train_new_RF_F_sm,
       cm_test_RF_F_sm, sensi_test_RF_F_sm, speci_test_RF_F_sm, accuracy_test_RF_F_sm, 
       y_pred_test_RF_F_sm, mydf_test_RF_F_sm] = RFcomplete('F', dtm_train_small, dtm_test_small, y_train_10_F_small, y_test_10_F_small, X_train_ind, X_test_ind, False)                

cm_train_RF_F_sm
# array([[ 52,  20],
#        [  6, 975]])
sensi_train_RF_F_sm # 0.7222222222222222
speci_train_RF_F_sm # 0.9938837920489296
sensi_train_RF_F_sm + speci_train_RF_F_sm # 1.7161060142711517

cm_train_new_RF_F_sm
# array([[ 58,  14],
#        [ 44, 937]])
sensi_train_new_RF_F_sm # 0.8055555555555556
speci_train_new_RF_F_sm # 0.9551478083588175
sensi_train_new_RF_F_sm + speci_train_new_RF_F_sm # 1.7607033639143732


cm_test_RF_F_sm # better than the model on the big matrix
# array([[  2,   6],
#        [ 11, 359]])
sensi_test_RF_F_sm # 0.25
speci_test_RF_F_sm # 0.9702702702702702
sensi_test_RF_F_sm + speci_test_RF_F_sm # 1.2202702702702704
accuracy_test_RF_F_sm # 0.955026455026455


### SVM =======================================================================
 
### big matrix 
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


### my tokens
random.seed(1802)
[cm_train_SVM_F_sm, sensi_train_SVM_F_sm, speci_train_SVM_F_sm, accuracy_train_SVM_F_sm, 
 y_pred_train_SVM_F_sm, mydf_train_SVM_F_sm, ts_SVM_F_sm, cm_train_new_SVM_F_sm, sensi_train_new_SVM_F_sm, 
 speci_train_new_SVM_F_sm, accuracy_train_new_SVM_F_sm, y_pred_train_new_SVM_F_sm,
       cm_test_SVM_F_sm, sensi_test_SVM_F_sm, speci_test_SVM_F_sm, accuracy_test_SVM_F_sm, 
       y_pred_test_SVM_F_sm, mydf_test_SVM_F_sm] = SVMcomplete('F', dtm_train_small, dtm_test_small, y_train_10_F_small, y_test_10_F_small, X_train_ind, X_test_ind, False)                

cm_train_SVM_F_sm
# array([[ 52,  20],
#        [  6, 975]])


sensi_train_SVM_F_sm # 0.7222222222222222
speci_train_SVM_F_sm # 0.9938837920489296
accuracy_train_SVM_F_sm # 0.9753086419753086

cm_test_SVM_F_sm # better than the model on the big matrix
# array([[  1,   7],
#        [  0, 370]])
sensi_test_SVM_F_sm # 0.125
speci_test_SVM_F_sm # 1.0
accuracy_test_SVM_F_sm #  0.9814814814814815

### the best of the three possible models
# 1. WS, 2. RF, 3. SVM
cm_F_test
# array([[  2,   6],
#        [  3, 367]], dtype=int64)
sensi_F_test + speci_F_test # 1.2418918918918918
cm_test_RF_F_sm
# array([[  2,   6],
#        [ 11, 359]])
sensi_test_RF_F_sm + speci_test_RF_F_sm #  1.2202702702702704
cm_test_SVM_F_sm 
# array([[  1,   7],
#        [  0, 370]])
sensi_test_SVM_F_sm + speci_test_SVM_F_sm # 1.125

# ===========================================================================
# Job change
# ===========================================================================

### Random Forest =============================================================

### big matrix
random.seed(1802)
[cm_train_RF_J, sensi_train_RF_J, speci_train_RF_J, accuracy_train_RF_J, 
 y_pred_train_RF_J, mydf_train_RF_J, ts_RF_J, cm_train_new_RF_J, sensi_train_new_RF_J, 
 speci_train_new_RF_J, accuracy_train_new_RF_J, y_pred_train_new_RF_J,
       cm_test_RF_J, sensi_test_RF_J, speci_test_RF_J, accuracy_test_RF_J, 
       y_pred_test_RF_J, mydf_test_RF_J] = RFcomplete('J', dtm_train, dtm_test, y_train_10_J, y_test_10_J, X_train_ind, X_test_ind)                

cm_train_RF_J 
# array([[ 64,   0],
#        [  0, 989]])


cm_test_RF_J 
# array([[  0,   7],
#        [  0, 371]])

### my tokens
random.seed(1802)
[cm_train_RF_J_sm, sensi_train_RF_J_sm, speci_train_RF_J_sm, accuracy_train_RF_J_sm, 
 y_pred_train_RF_J_sm, mydf_train_RF_J_sm, ts_RF_J_sm, cm_train_new_RF_J_sm, sensi_train_new_RF_J_sm, 
 speci_train_new_RF_J_sm, accuracy_train_new_RF_J_sm, y_pred_train_new_RF_J_sm,
       cm_test_RF_J_sm, sensi_test_RF_J_sm, speci_test_RF_J_sm, accuracy_test_RF_J_sm, 
       y_pred_test_RF_J_sm, mydf_test_RF_J_sm] = RFcomplete('J', dtm_train_small, dtm_test_small, y_train_10_J_small, y_test_10_J_small, X_train_ind, X_test_ind, False)                

cm_train_RF_J_sm
# array([[ 13,  51],
#        [  2, 987]])
sensi_train_RF_J_sm # 0.203125
speci_train_RF_J_sm #0.9979777553083923
accuracy_train_new_RF_J_sm #0.936372269705603

cm_train_new_RF_J_sm
# array([[ 38,  26],
#        [ 41, 948]])

sensi_train_new_RF_J_sm # 0.59375
speci_train_new_RF_J_sm # 0.9585439838220424
accuracy_train_new_RF_J_sm # 0.936372269705603


cm_test_RF_J_sm
# array([[  0,   7],
#        [  5, 366]])
sensi_test_RF_J_sm
speci_test_RF_J_sm #  0.9865229110512129
sensi_test_RF_J_sm + speci_test_RF_J_sm
accuracy_test_RF_J_sm

### SVM =======================================================================
 
### big matrix 
random.seed(1802)
[cm_train_SVM_J, sensi_train_SVM_J, speci_train_SVM_J, accuracy_train_SVM_J, 
 y_pred_train_SVM_J, mydf_train_SVM_J, ts_SVM_J, cm_train_new_SVM_J, sensi_train_new_SVM_J, 
 speci_train_new_SVM_J, accuracy_train_new_SVM_J, y_pred_train_new_SVM_J,
       cm_test_SVM_J, sensi_test_SVM_J, speci_test_SVM_J, accuracy_test_SVM_J, 
       y_pred_test_SVM_J, mydf_test_SVM_J] = SVMcomplete('J', dtm_train, dtm_test, y_train_10_J, y_test_10_J, X_train_ind, X_test_ind)                

cm_train_SVM_J
# array([[ 64,   0],
#        [  0, 989]])
cm_train_new_SVM_J

cm_test_SVM_J 
# array([[  0,   7],
#        [  0, 371]])

sensi_test_SVM_J
speci_test_SVM_J
sensi_test_SVM_J + speci_test_SVM_J
accuracy_test_SVM_J


### my tokens  
random.seed(1802)
[cm_train_SVM_J_sm, sensi_train_SVM_J_sm, speci_train_SVM_J_sm, accuracy_train_SVM_J_sm, 
 y_pred_train_SVM_J_sm, mydf_train_SVM_J_sm, ts_SVM_J_sm, cm_train_new_SVM_J_sm, sensi_train_new_SVM_J_sm, 
 speci_train_new_SVM_J_sm, accuracy_train_new_SVM_J_sm, y_pred_train_new_SVM_J_sm,
       cm_test_SVM_J_sm, sensi_test_SVM_J_sm, speci_test_SVM_J_sm, accuracy_test_SVM_J_sm, 
       y_pred_test_SVM_J_sm, mydf_test_SVM_J_sm] = SVMcomplete('J', dtm_train_small, dtm_test_small, y_train_10_J_small, y_test_10_J_small, X_train_ind, X_test_ind, False)                

cm_train_SVM_J_sm
# array([[ 13,  51],
#        [  2, 987]])
sensi_train_SVM_J_sm # 0.203125
speci_train_SVM_J_sm # 0.9979777553083923
accuracy_train_SVM_J_sm # 0.949667616334283

cm_train_new_SVM_J_sm
# array([[ 38,  26],
#        [249, 740]])
sensi_train_new_SVM_J_sm # 0.59375
speci_train_new_SVM_J_sm # 0.7482305358948432
accuracy_train_new_SVM_J_sm # 0.7388414055080722

cm_test_SVM_J_sm
# array([[  0,  64],
#        [ 18, 503]])
sensi_test_SVM_J_sm
speci_test_SVM_J_sm #  0.9654510556621881
sensi_test_SVM_J_sm + speci_test_SVM_J_sm
accuracy_test_SVM_J_sm

## best models
# 1. WS, 2. RF/ SVM
# for RF and SVM is the big model the one that classifies everything as no reason
# the best (sum form Sensi & speci = 1)
# Wordsearch:
cm_J_test
# array([[  1,   6],
#        [ 16, 355]], dtype=int64)
sensi_J_test + speci_J_test # 1.0997304582210243   

# ===========================================================================
# Rente
# ===========================================================================

### Random Forest =============================================================

### big matrix
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


### my tokens
random.seed(1802)
[cm_train_RF_R_sm, sensi_train_RF_R_sm, speci_train_RF_R_sm, accuracy_train_RF_R_sm, 
 y_pred_train_RF_R_sm, mydf_train_RF_R_sm, ts_RF_R_sm, cm_train_new_RF_R_sm, sensi_train_new_RF_R_sm, 
 speci_train_new_RF_R_sm, accuracy_train_new_RF_R_sm, y_pred_train_new_RF_R_sm,
       cm_test_RF_R_sm, sensi_test_RF_R_sm, speci_test_RF_R_sm, accuracy_test_RF_R_sm, 
       y_pred_test_RF_R_sm, mydf_test_RF_R_sm] = RFcomplete('R', dtm_train_small, dtm_test_small, y_train_10_R_small, y_test_10_R_small, X_train_ind, X_test_ind, False)                

cm_train_RF_R_sm
# array([[  34,   18],
#        [   0, 1001]])
sensi_train_RF_R_sm # 0.6538461538461539
speci_train_RF_R_sm # 1.0
accuracy_train_RF_R_sm # 0.9829059829059829

cm_train_new_RF_R_sm
# array([[ 47,   5],
#        [ 84, 917]])
sensi_train_new_RF_R_sm # 0.9038461538461539
speci_train_new_RF_R_sm # 0.916083916083916
accuracy_train_new_RF_R_sm # 0.9154795821462488


cm_test_RF_R_sm # better than in the big model
# array([[  2,   3],
#        [  2, 371]])
sensi_test_RF_R_sm # 0.4
speci_test_RF_R_sm # 0.9946380697050938
sensi_test_RF_R_sm + speci_test_RF_R_sm # 1.3946380697050937
accuracy_test_RF_R_sm # 0.9867724867724867 


### SVM =======================================================================
 
### big matrix 
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


### my tokens
random.seed(1802)
[cm_train_SVM_R_sm, sensi_train_SVM_R_sm, speci_train_SVM_R_sm, accuracy_train_SVM_R_sm, 
 y_pred_train_SVM_R_sm, mydf_train_SVM_R_sm, ts_SVM_R_sm, cm_train_new_SVM_R_sm, sensi_train_new_SVM_R_sm, 
 speci_train_new_SVM_R_sm, accuracy_train_new_SVM_R_sm, y_pred_train_new_SVM_R_sm,
       cm_test_SVM_R_sm, sensi_test_SVM_R_sm, speci_test_SVM_R_sm, accuracy_test_SVM_R_sm, 
       y_pred_test_SVM_R_sm, mydf_test_SVM_R_sm] = SVMcomplete('R', dtm_train_small, dtm_test_small, y_train_10_R_small, y_test_10_R_small, X_train_ind, X_test_ind, False)                

cm_train_SVM_R_sm
# array([[  34,   18],
#        [   0, 1001]])
sensi_train_SVM_R_sm # 0.6538461538461539
speci_train_SVM_R_sm # 1.0
accuracy_train_SVM_R_sm # 0.9829059829059829

cm_test_SVM_R_sm # better than in the big models
# array([[  2,   3],
#        [  0, 373]])
sensi_test_SVM_R_sm # 0.1
speci_test_SVM_R_sm # 1
sensi_test_SVM_R_sm + speci_test_SVM_R_sm # 1.4
accuracy_test_SVM_R_sm # 0.9920634920634921

## best models
# 1. SVM, 2. RF, 3. WS
cm_R_test
sensi_R_test + speci_R_test  # 1.2579088471849866
cm_test_RF_R_sm
speci_test_RF_R_sm + sensi_test_RF_R_sm #  1.3946380697050937
cm_test_SVM_R_sm
sensi_test_SVM_R_sm + speci_test_SVM_R_sm # 1.4

# =============================================================================
# Death
# =============================================================================

### Random Forest =============================================================

### big matrix
random.seed(1802)
[cm_train_RF_D, sensi_train_RF_D, speci_train_RF_D, accuracy_train_RF_D, 
 y_pred_train_RF_D, mydf_train_RF_D, ts_RF_D, cm_train_new_RF_D, sensi_train_new_RF_D, 
 speci_train_new_RF_D, accuracy_train_new_RF_D, y_pred_train_new_RF_D,
       cm_test_RF_D, sensi_test_RF_D, speci_test_RF_D, accuracy_test_RF_D, 
       y_pred_test_RF_D, mydf_test_RF_D] = RFcomplete('D', dtm_train, dtm_test, y_train_10_D, y_test_10_D, X_train_ind, X_test_ind)                

cm_train_RF_D
# array([[  44,    0],
#        [   0, 1009]])

cm_test_RF_D
# array([[  0,   5],
#        [  0, 373]])
sensi_test_RF_D
speci_test_RF_D
sensi_test_RF_D + speci_test_RF_D
accuracy_test_RF_D


### my tokens
random.seed(1802)
[cm_train_RF_D_sm, sensi_train_RF_D_sm, speci_train_RF_D_sm, accuracy_train_RF_D_sm, 
 y_pred_train_RF_D_sm, mydf_train_RF_D_sm, ts_RF_D_sm, cm_train_new_RF_D_sm, sensi_train_new_RF_D_sm, 
 speci_train_new_RF_D_sm, accuracy_train_new_RF_D_sm, y_pred_train_new_RF_D_sm,
       cm_test_RF_D_sm, sensi_test_RF_D_sm, speci_test_RF_D_sm, accuracy_test_RF_D_sm, 
       y_pred_test_RF_D_sm, mydf_test_RF_D_sm] = RFcomplete('D', dtm_train_small, dtm_test_small, y_train_10_D_small, y_test_10_D_small, X_train_ind, X_test_ind, False)                

cm_train_RF_D_sm
# array([[  43,    1],
#        [   0, 1009]])
sensi_train_RF_D_sm # 0.9772727272727273
speci_train_RF_D_sm # 1.0
accuracy_train_RF_D_sm # 0.9990503323836657

cm_test_RF_D_sm # better than the big model
# array([[  3,   2],
#        [  1, 372]])
sensi_test_RF_D_sm # 0.6
speci_test_RF_D_sm # 0.9973190348525469
sensi_test_RF_D_sm + speci_test_RF_D_sm # 1.5973190348525468
accuracy_test_RF_D_sm # 0.9920634920634921

### SVM =======================================================================
 
### big matrix 
random.seed(1802)
[cm_train_SVM_D, sensi_train_SVM_D, speci_train_SVM_D, accuracy_train_SVM_D, 
 y_pred_train_SVM_D, mydf_train_SVM_D, ts_SVM_D, cm_train_new_SVM_D, sensi_train_new_SVM_D, 
 speci_train_new_SVM_D, accuracy_train_new_SVM_D, y_pred_train_new_SVM_D,
       cm_test_SVM_D, sensi_test_SVM_D, speci_test_SVM_D, accuracy_test_SVM_D, 
       y_pred_test_SVM_D, mydf_test_SVM_D] = SVMcomplete('D', dtm_train, dtm_test, y_train_10_D, y_test_10_D, X_train_ind, X_test_ind)                

cm_train_SVM_D
# array([[  44,    0],
#        [   0, 1009]])


cm_test_SVM_D # both no reason
# array([[  0,   5],
#        [  0, 373]])
sensi_test_SVM_D
speci_test_SVM_D
sensi_test_SVM_D + speci_test_SVM_D
accuracy_test_SVM_D


### my tokens
random.seed(1802)
[cm_train_SVM_D_sm, sensi_train_SVM_D_sm, speci_train_SVM_D_sm, accuracy_train_SVM_D_sm, 
 y_pred_train_SVM_D_sm, mydf_train_SVM_D_sm, ts_SVM_D_sm, cm_train_new_SVM_D_sm, sensi_train_new_SVM_D_sm, 
 speci_train_new_SVM_D_sm, accuracy_train_new_SVM_D_sm, y_pred_train_new_SVM_D_sm,
       cm_test_SVM_D_sm, sensi_test_SVM_D_sm, speci_test_SVM_D_sm, accuracy_test_SVM_D_sm, 
       y_pred_test_SVM_D_sm, mydf_test_SVM_D_sm] = SVMcomplete('D', dtm_train_small, dtm_test_small, y_train_10_D_small, y_test_10_D_small, X_train_ind, X_test_ind, False)                

cm_train_SVM_D_sm
# array([[  43,    1],
#        [   0, 1009]])

cm_test_SVM_D_sm 
# array([[  3,   2],
#        [  1, 372]])
sensi_test_SVM_D_sm
speci_test_SVM_D_sm
sensi_test_SVM_D_sm + speci_test_SVM_D_sm
accuracy_test_SVM_D_sm

# best models
# all three equal

cm_D_test
# array([[  3,   2],
#        [  1, 372]])
sensi_D_test + speci_D_test # 1.5973190348525468
cm_test_RF_D_sm
# array([[  3,   2],
#        [  1, 372]])
sensi_test_RF_D_sm + speci_test_RF_D_sm #  1.5973190348525468
cm_test_SVM_D_sm
# array([[  3,   2],
#        [  1, 372]])

os.chdir(r'W:\your_folder\Output')
dill.dump_session('vorvgl_5B_04_29_03.pkl')
#dill.load_session('vorvgl_5B_04_29_02.pkl')

### all thresholds:
ts_RF_F
ts_RF_F_sm
ts_RF_J
ts_RF_J_sm
ts_RF_R
ts_RF_R_sm
ts_RF_D
ts_RF_D_sm

ts_SVM_F
ts_SVM_F_sm
ts_SVM_J
ts_SVM_J_sm
ts_SVM_R
ts_SVM_R_sm
ts_SVM_D
ts_SVM_D_sm


ts_RF_F
# OUt[57]: 0.5498307717013564

ts_RF_F_sm
# OUt[58]: 0.14560109321478149

ts_RF_J
# OUt[59]: 0.5664887599937336

ts_RF_J_sm
# OUt[60]: 0.3490324881235741

ts_RF_R
# OUt[61]: 0.42320532731256444

ts_RF_R_sm
# OUt[62]: 0.10787467907495621

ts_RF_D
# OUt[63]: 0.436527812678036

ts_RF_D_sm
# OUt[64]: 1.0

ts_SVM_F
# OUt[65]: 0.6637802823758071

ts_SVM_F_sm
# OUt[66]: 0.5675058140507078

ts_SVM_J
# OUt[67]: 0.666959995753519

ts_SVM_J_sm
# OUt[68]: 0.05381885382301271

ts_SVM_R
# OUt[69]: 0.40954741851139737

ts_SVM_R_sm
# OUt[70]: 0.969051003048289

ts_SVM_D
# OUt[71]: 0.5702134788421716

ts_SVM_D_sm
# OUt[72]: 0.9781792928645842
# =============================================================================
# Overview for each model
# =============================================================================

# Note: column sums do not necessarily equal the number of, for example, terminations
# for financial reasons, since documents can be assigned to multiple reasons

### Wordsearch =================================================================

wordsearch_res = getResults(ypred_F, ypred_J, ypred_R, ypred_D, X_train, y_train)
#         F class  J class  R class  D class  C class
# F true       13        3        1        0        4
# J true        0        9        0        0        7
# R true        0        5       11        0        1
# D true        0        0        0       10        1
# C true       10       40      104        0      675


wordsearch_res_test = getResults(ypred_F_test, ypred_J_test, ypred_R_test,
                                    ypred_D_test, X_test, y_test)
#         F class  J class  R class  D class  C class
# F true        2        0        2        0        5
# J true        0        1        1        0        5
# R true        0        0        2        0        3
# D true        0        0        0        3        2
# C true        3       16       50        1      288

### Random Forest =============================================================
# small & big: 1053
## train
# big


RF_erg_gr = getResults(y_pred_train_RF_F, y_pred_train_RF_J,
                          y_pred_train_RF_R, y_pred_train_RF_D,
                          dtm_train, y_train_all)
#         F class  J class  R class  D class  C class
# F true       72        0        0        0        0
# J true        0       64        0        0        0
# R true        0        0       52        0        0
# D true        0        0        0       44        0
# C true        0        0        0        0      821

# everything is correct classified

# small
RF_erg_sm = getResults(y_pred_train_RF_F_sm, y_pred_train_RF_J_sm,
                          y_pred_train_RF_R_sm, y_pred_train_RF_D_sm,
                          dtm_train_small, y_train_small)

#         F class  J class  R class  D class  C class
# F true       52        0        0        0       20
# J true        0       13        0        0       51
# R true        0        0       34        0       18
# D true        0        0        0       43        1
# C true        6        2        0        0      813


## test
# big
RF_erg_test_gr = getResults(y_pred_test_RF_F, y_pred_test_RF_J,
                          y_pred_test_RF_R, y_pred_test_RF_D,
                          dtm_test, y_test_all)
#         F class  J class  R class  D class  C class
# F true        0        0        0        0        8
# J true        0        0        0        0        7
# R true        0        0        0        0        5
# D true        0        0        0        0        5
# C true        0        0        0        0      353

# for every Gevo everything is classified as "C"

# small
RF_erg_test_sm = getResults(y_pred_test_RF_F_sm, y_pred_test_RF_J_sm,
                          y_pred_test_RF_R_sm, y_pred_test_RF_D_sm,
                           dtm_test_small, y_test_small)
#         F class  J class  R class  D class  C class
# F true        2        0        0        0        6
# J true        1        0        0        0        6
# R true        0        0        2        0        3
# D true        0        0        0        3        2
# C true       10        5        2        1      336


### SVM =======================================================================

## train
# big
SVM_erg_gr = getResults(y_pred_train_SVM_F, y_pred_train_SVM_J,
                          y_pred_train_SVM_R, y_pred_train_SVM_D,
                          dtm_train, y_train_all)
#         F class  J class  R class  D class  C class
# F true       72        0        0        0        0
# J true        0       64        0        0        0
# R true        0        0       52        0        0
# D true        0        0        0       44        0
# C true        0        0        0        0      821

# everything is correct classified

# small
SVM_erg_sm = getResults(y_pred_train_SVM_F_sm, y_pred_train_SVM_J_sm, 
                           y_pred_train_SVM_R_sm, y_pred_train_SVM_D_sm,
                           dtm_train_small, y_train_small)
#         F class  J class  R class  D class  C class
# F true       52        0        0        0       20
# J true        0       13        0        0       51
# R true        0        0       34        0       18
# D true        0        0        0       43        1
# C true        6        2        0        0      813


## test
# big
SVM_erg_test_gr = getResults(y_pred_test_SVM_F, y_pred_test_SVM_J,
                          y_pred_test_SVM_R, y_pred_test_SVM_D,
                          dtm_test, y_test_all)
#         F class  J class  R class  D class  C class
# F true        0        0        0        0        8
# J true        0        0        0        0        7
# R true        0        0        0        0        5
# D true        0        0        0        0        5
# C true        0        0        0        0      353


# small
SVM_erg_test_sm = getResults(y_pred_test_SVM_F_sm, y_pred_test_SVM_J_sm,
                          y_pred_test_SVM_R_sm, y_pred_test_SVM_D_sm,
                          dtm_test_small, y_test_small)
#         F class  J class  R class  D class  C class
# F true        7        9        0        0       56
# J true        0        0        0        0       64
# R true        0        0       17        0       35
# D true        0        0        0        0       44
# C true        0        9        0        0      344


### The comparison is made between the word search and the small
### matrices for both SVM and RF



# Wordsearch
sensis_WS = [sensi_F_test, sensi_J_test, sensi_R_test, sensi_D_test]
# [0.25, 0.14285714285714285, 0.4, 0.6]
specis_WS = [speci_F_test, speci_J_test, speci_R_test, speci_D_test]
# [0.9918918918918919, 0.9568733153638814, 0.8579088471849866, 0.9973190348525469]

# RF
sensis_RF = [sensi_test_RF_F_sm, sensi_test_RF_J_sm, sensi_test_RF_R_sm, sensi_test_RF_D_sm]
# [0.25, 0.0, 0.4, 0.6]
specis_RF = [speci_test_RF_F_sm, speci_test_RF_J_sm, speci_test_RF_R_sm, speci_test_RF_D_sm]
# [0.9702702702702702,
#  0.9865229110512129,
#  0.9946380697050938,
#  0.9973190348525469]

# SVM
sensis_SVM = [sensi_test_SVM_F_sm, sensi_test_SVM_J_sm, sensi_test_SVM_R_sm, sensi_test_SVM_D_sm]
# [0.125, 1.0, 0.4, 0.6]
specis_SVM = [speci_test_SVM_F_sm, sensi_test_SVM_J_sm, sensi_test_SVM_R_sm, sensi_test_SVM_D_sm]
# [1.0, 1.0, 0.4, 0.6]

os.chdir(r'W:\your_folder\Output')
dill.dump_session('5B_04_29_02.pkl')
# dill.load_session('5B_04_29.pkl')

# Overall, the word search is the best model three times, therefore
# the word search is chosen.
