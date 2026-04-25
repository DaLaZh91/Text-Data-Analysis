# In this file, the classification methods are applied.

# =============================================================================
# Load packages & include functions
# =============================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dill
import random
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score # ET
from sklearn.metrics import confusion_matrix
# from sklearn.datasets import load_boston
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import Lasso, LassoCV  # Lasso
from sklearn.metrics import mean_squared_error  # lasso
# https://laurenliz22.github.io/nlp_random_forest_and_neural_network_classifiers

os.chdir(r'W:\your_folder\Python')
from functions import *
# =============================================================================
# Read data & preprocessing
# =============================================================================
os.chdir(r'W:\your_folder\Output')
dill.load_session('dtms_all_04_28.pkl')


y_test_10 = []
for i in range(len(y_test)):
    if y_test[i] == 'N':
        y_test_10.append(0)
    else:
        y_test_10.append(1)
        
y_train_10 = []
for i in range(len(y_train)):
    if y_train[i] == 'N':
        y_train_10.append(0)
    else:
        y_train_10.append(1)
                
  
  
# =============================================================================
# Word search
# =============================================================================

C_ohneNC = set(X_train_C).difference(X_train_NC)
len(C_ohneNC)  # 862
NC_ohneC = set(X_train_NC).difference(X_train_C)
len(NC_ohneC)  # 7119
# far too much to look at everything

# german stopwords
stop_words_stem = getStem(get_stop_words('de'))
    
createWC(' '.join(X_train_C), stop_words_stem, 'cloudC_1.png')
createWC(' '.join(X_train_NC), stop_words_stem, 'cloudNC_1.png')

# remove all mono- and bigrams from the 200 most frequent ones, except
# those that could be related to one of the GeVos (or explicitly do not
# suggest a cancellation)

[often_sl_C, often_sl_C_bi] = getFreqWords(X_train_C)
often_sl_C

# german words relevant for Cancellations (monograms)

gevowords_c_mono = ['auszahl', 'beitragsfrei', 'gekundigt', 'kundig', 
                    'kundigungsbestat', 'kundigungstermin', 'ruckkaufswert', 
                    'ruckkaufwert', 'teil', 'teilkund', 'versicherungsnehm']
often_sl_sm_C = often_sl_C.difference(gevowords_c_mono)
set(often_sl_C_bi)

# german words relevant for Cancellations (bigrams)

gevowords_c_bi = ['anbei kundig', 'auszahl ruckkaufswert', 
                  'auszahl ruckkaufwert', 'bestat kundig', 'betreff kundig', 
                  'bitt auszahl', 'eingang kundig', 'erhalt kundig',
                  'erklar kundig', 'herr kundig', 'hiermit kundig', 
                  'kundig besteh', 'kundig bitt', 'kundig freundlich', 
                  'kundig firstgerecht', 'kundig genannt', 'kundig hiermit',
                  'kundig innerhalb', 'kundig lebensversicher', 
                  'kundig nachstmog', 'kundig oben', 'kundig og', 
                  'kundig schriftlich', 'kundig sofort', 'kundig versicher',
                  'kundig vertrag', 'kundigungsbestat angab', 
                  'kundigungsbestat freundlich',  'ruckkaufswert bekannt',
                  'ruckkaufswert bitt', 'ruckkaufswert erkenn', 
                  'ruckkaufswert folgend', 'ruckkaufswert konto',
                  'ruckkaufswert uberweis', 'schreib kundig',
                  'schriftlich kundigungsbestat', 'termin kundig',
                  'uberweis ruckkaufswert', 'versicher kundig', 
                  'vertrag kundig', 'zeitpunkt kundig']
often_sl_sm_bi_C = set(often_sl_C_bi).difference(gevowords_c_bi)

# german words which occur often in non-Cancellations

[often_sl_NC, often_sl_NC_bi] = getFreqWords(X_train_NC)
often_sl_NC

gevowords_nc_mono = ['arbeitgeb', 'arbeitnehm', 'ausschluss', 'beitrag', 
                     'beitragszahl', 'beschäftigungsverhältnis',
                     'gesundheitsdat', 'todesfall', 
                     'versicherungsnehmerwechsel', 'wechsel']
often_sl_sm_NC = often_sl_NC.difference(gevowords_nc_mono)

set(often_sl_NC_bi)
gevowords_nc_bi = ['abfrag gesundheitsdat', 'arbeitgeb ausgeschloss',
                   'arbeitgeb erteilt', 'arbeitgeb neu', 'arbeitgeb sof',
                   'arbeitgeb sowi', 'arbeitgeb teilweis', 
                   'arbeitgeb ubernahm', 'arbeitgeb weiterfuhr',
                   'arbeitgeberwechsel vertrag', 'arbeitnehm beginn',
                   'arbeitnehm nachfolg', 'arbeitnehm privat',
                   'arbeitnehm ubernomm', 'arbeitsrecht handelt',
                   'ausgeschied arbeitnehm',
                   'ausscheid beschaftigungsverhaltnis', 'ausschluss bestimmt',
                   'beschaftigungsverhaltnis beim', 
                   'beschaftigungsverhaltnis bestand', 'betriebsubergang bgb',
                   'bisher versicherungsnehm', 'dat abschluss',
                   'dat ausgewahlt', 'dat besteh',
 'dat moglich',
 'dat ruckversicher',
 'dat schweigepflicht',
 'ehemal arbeitgeb',
 'fortfuhr neu',
 'fortfuhr versichert',
 'geschutzt dat',
 'geschutzt information',
 'gesund dat',
 'gesund heitsdat',
 'gesundheitsdat schweigepflichtentbindungserklar',
 'gesundheitsdat stgb',
 'gesundheitsdat vertrag',
 'handelt ubernahm',
 'handelt versicherungsnehmerwechsel',
 'liegt betriebsubergang',
 'medizin gutacht',
 'nachfolg arbeitgeb',
 'neu arbeitgeb',
 'neu beschaftigungsverhaltnis',
 'neu versicherungsnehm',
 'personenbezog dat',
 'ubertrag versicher',
 'versicher gesundheitsdat',
 'versichert arbeitnehm',
 'versichert person',
 'vertragsfortfuhr arbeitnehm',
 'verwend gesundheitsdat',
 'vorher arbeitgeb',
 'weiterfuhr besteh',
 'weitergab gesundheitsdat',
     'zustimm versicherungsnehmerwechsel']

often_sl_sm_bi_NC = set(often_sl_NC_bi).difference(gevowords_nc_bi)

# remove frequently occurring words that should not be relevant
oft_KNK = ['"company name"',  'sehr', 'geehrte', 'damen', 'und', 'herren',
           'und herren', 'freundlichen grüßen', 'mit freundlichen', '"company name"',
           'versicherung', 'lebensversicherung', 'gruppe', 'bitte']

stop_new = stop_words + list(often_sl_sm_C) + list(often_sl_sm_bi_C) + list(often_sl_sm_NC) + list(often_sl_sm_bi_NC)
createWC(' '.join(X_train_C), stop_new, 'cloudC_2.png')
createWC(' '.join(X_train_NC), stop_new, 'cloudNC_2.png')


c_words = ['auszahl', 'gekundigt', 'kundig', 'kundigungsbestat',
           'kundigungstermin', 'ruckkaufswert', 'ruckkaufwert', 'teilkund']
c_bigrams = ['auszahl ruckkaufswert', 'auszahl ruckkaufwert', 
              'ruckkaufswert uberweis', 'ruckkaufwert uberweis']

c_test = c_words + c_bigrams

[inC, sumC, inNC, sumNC] =  getAmounts(c_words, X_train_C, X_train_NC)

inCproc = [round(j/len(X_train_C) * 100, 2) for j in inC]
inNCproc = [round(j/len(X_train_NC) * 100, 2) for j in inNC]
c_test_CNC = pd.DataFrame([c_words + ['In total'], inC + [sumC], inCproc, inNC + [sumNC], inNCproc]).T
c_test_CNC.columns = ['token', 'in C', 'in C proc', 'in NC', 'in NC proc']

#               token in C in C proc in NC in NC proc
# 0           auszahl  177     14.08   115       1.06
# 1         gekundigt   17      1.35    68       0.63
# 2            kundig  780     62.05   312       2.88
# 3  kundigungsbestat   97      7.72     4       0.04
# 4  kundigungstermin   25      1.99     6       0.06
# 5     ruckkaufswert  231     18.38   130        1.2
# 6      ruckkaufwert   59      4.69    11        0.1
# 7          teilkund   26      2.07     4       0.04
# 8          In total  819      None   475       None

# gekundigt is the only word stam, that occur more often in non-Cancellations than in Cancellations

[inC_bi, sumC_bi, inNC_bi, sumNC_bi] =  getAmounts(c_bigrams, X_train_C, X_train_NC)
c_test_CNC_bi = pd.DataFrame([c_bigrams + ['In total'], inC_bi + [sumC_bi], inNC_bi + [sumNC_bi]]).T
c_test_CNC_bi.columns = ['token', 'in C', 'in NC']
c_test_CNC_bi

#                     token in C in NC
# 0   auszahl ruckkaufswert   61    18
# 1    auszahl ruckkaufwert   23     2
# 2  ruckkaufswert uberweis   24     0
# 3   ruckkaufwert uberweis    3     0
# 4                In total  110    20

# these bigrams therefore occur more frequently in the cancellations than in the
# other documents. Nevertheless, the bigrams are not used further,
# because the goal is to identify as many cancellations as possible, and by
# omitting "auszahl" or "ruckkaufswert" in favor of the bigram,
# this is not achieved

# therefore, consider the model with the k_words

[cm_C, sensi_C, speci_C, accuracy_C, ypred_C] = getValues(c_words, [], 
                                                           X_train, y_train)

cm_C
# array([[ 819,   61],
#        [ 475, 7109]], dtype=int64)

sensi_C # 0.9306818181818182
speci_C # 0.9373681434599156
sensi_C + speci_C # 1.8680499616417339
accuracy_C # 0.9366729678638941

# and consider the model without "gekundigt"
c_words_wog = ['auszahl', 'kundig', 'kundigungsbestat', 'kundigungstermin',
              'ruckkaufswert', 'ruckkaufwert', 'teilkund']

[cm_C_wog, sensi_C_wog, speci_C_wog, riightCl_C_og, ypred_C_og] = getValues(C_words_wog, [], 
                                                           X_train, y_train)

cm_C_og
# array([[ 816,   64],
#        [ 441, 7143]], dtype=int64)

sensi_C_wog # 0.9272727272727272
speci_C_wog # 0.9418512658227848
sensi_C_wog + speci_C_wog # 1.8691239930955121
riightcl_C_wog # 0.940335538752363


# use the model without "gekundigt"

c_words_final = c_words_wog

#### NC ==============================

# frequently occurring words (that could belong to the other GeVos) and their
# variations, as well as words related to the other GeVos
nc_words = ['beitragsfreistell', 'beitragspaus','erhoh', 'geschutzt dat', 
            'gesundheitsdat', 'pausi', 'stell', 'versicherungsnehm', 
            'versichert person', 'weitergab', 'ubertrag']


# how often do the nk words occur in nk and how often in k (not in how many
# documents, but total frequency, since this is only meant to provide a rough count
# and the goal of the initial methods is that they work as quickly as possible)

# now consider only the documents that were incorrectly classified as cancellations,
# the rest are classified correctly anyway  
predC = myEqual(ypred_C_wog, 'C')
actN = myEqual(y_train, 'N')
actC = myEqual(y_train, 'C')
Nwrong = list(set(predC).intersection(actN))
Cright = list(set(predC).intersection(actC))
X_train_NC_wrong = list(np.array(X_train)[Nwrong])
X_train_C_right = list(np.array(X_train)[Cright])

[inC_right, sumC_right, inNK_wrong, sumNC_wrong] = getAmounts(nc_words, X_train_C_right, X_train_NC_wrong)
   
nc_test_CNC = pd.DataFrame([nk_words + ['In total'], inC_right + [sumC_right], inNC_wrong + [sumNC_wrong]]).T
nc_test_CNC.columns = ['token', 'in C', 'in NC']
nc_test_CNC

#                 token in C in NC
# 0   beitragsfreistell   12    47
# 1        beitragspaus    0    79
# 2               erhoh    2   141
# 3       geschutzt dat    0    59
# 4      gesundheitsdat    0    66
# 5               pausi    0     0
# 6               stell   18   108
# 7   versicherungsnehm   77   180
# 8   versichert person   72   153
# 9           weitergab    5    93
# 10           ubertrag    7   123
# 11           In total  134   317

# "pausi" can be removed, since it is 0 in both
# "beitragspaus", "geschutzt dat" and "gesundheitsdat" can be included,
# since they do not occur in K

### model with beitragspaus, geschutzt dat and gesundheitsdat ==================

[cm_bg, sensi_bg, speci_bg, accuracy_bg, ypred_bg] = getValues(c_words_final, ['beitragspaus', 'geschutzt dat', 'gesundheitsdat'], 
                                                           X_train, y_train)

cm_bg
# array([[ 816,   64],
#        [ 331, 7253]], dtype=int64)

sensi_bg + speci_bg # 1.8836282125047947

predC = myEqual(ypred_bg, 'C')
actN = myEqual(y_train, 'N')
actC = myEqual(y_train, 'C')
Nwrong = list(set(predK).intersection(actN))
Cright = list(set(predK).intersection(actC))
X_train_NC_wrong = list(np.array(X_train)[Nwrong])
X_train_C_right = list(np.array(X_train)[Cright])

nc_bg = ['beitragsfreistell', 'erhoh',  'stell', 
         'versichert person', 'versicherungsnehm', 'weitergab', 'ubertrag']
[inC_right, sumC_right, inNC_wrong, sumNC_wrong] = getAmounts(nc_bg, X_train_C_right, X_train_NC_wrong)
   
nc_test_CNC_bg = pd.DataFrame([nc_bg + ['In total'], inC_right + [sumC_right], inNC_wrong + [sumNC_wrong]]).T
nc_test_CNC_bg.columns = ['token', 'in C', 'in NC']
nc_test_CNC_bg['in C (same scale)'] = nc_test_CNC_bg['in C'] * len(X_train_NC)/len(X_train_C)
nc_test_CNC_bg

#                token in C in NC in C (same scale)
# 0  beitragsfreistell   12    38           103.418182
# 1              erhoh    2    64            17.236364
# 2              stell   18    42           155.127273
# 3  versichert person   72    87           620.509091
# 4  versicherungsnehm   77    86                663.6
# 5          weitergab    5    11            43.090909
# 6           ubertrag    7    61            60.327273
# 7            In total  134   207          1154.836364

len(X_train_NC)/len(X_train_C) # 8.62

# we want to improve sensitivity plus specificity; this is therefore only possible
# with those that occur at least 8.62 times as often in NK as in the
# cancellations, otherwise specificity does not increase sufficiently.
# this only applies to "erhoh"

### model with beitragspaus, geschutzt dat, gesundheitsdat, erhoh =============

[cm_bge, sensi_bge, speci_bge, accuracy_bge, ypred_bge] = getValues(c_words_final, ['beitragspaus', 'geschutzt dat', 'gesundheitsdat', 'erhoh'], 
                                                           X_train, y_train)

cm_bge
# array([[ 814,   66],
#        [ 267, 7317]], dtype=int64)

sensi_bge + speci_bge # 1.8897943037974683


predC = myEqual(ypred_bge, 'C')
actN = myEqual(y_train, 'N')
actC = myEqual(y_train, 'C')
Nwrong = list(set(predC).intersection(actN))
Cright = list(set(predC).intersection(actC))
X_train_NC_wrong = list(np.array(X_train)[Nwrong])
X_train_C_right = list(np.array(X_train)[Cright])

nc_bge = ['beitragsfreistell', 'stell', 
         'versichert person', 'versicherungsnehm', 'weitergab', 'ubertrag']
[inC_right, sumC_right, inNC_wrong, sumNC_wrong] = getAmounts(nc_bge, X_train_C_right, X_train_NC_wrong)
   
nk_test_CNC_bge = pd.DataFrame([nk_bge + ['In total'], inC_right + [sumC_right], inNC_wrong + [sumNC_wrong]]).T
nk_test_CNC_bge.columns = ['token', 'in C', 'in NC']
nk_test_CNC_bge['in C (same scale)'] = nk_test_CNC_bge['in C'] * len(X_train_NC)/len(X_train_C)
nk_test_CNC_bge
 
#                token in C in NC in C (same scale)
# 0  beitragsfreistell   11    23                 94.8
# 1              stell   18    24           155.127273
# 2  versichert person   71    64           611.890909
# 3  versicherungsnehm   76    55           654.981818
# 4          weitergab    5     8            43.090909
# 5           ubertrag    7    51            60.327273
# 6            In total  132   143               1137.6   

# nothing further can improve the sum of sensitivity and specificity here

### therefore, this is the final model:
    
nc_words_final = ['beitragspaus', 'geschutzt dat', 'gesundheitsdat', 'erhoh']

[cm_WS_train, sensi_WS_train, speci_WS_train, 
 accuracy_WS_train, ypred_WS_train] = getValues(c_words_final, nc_words_final,  
                                    X_train, y_train)
  
cm_WS_train
# array([[ 814,   66],
#        [ 267, 7317]], dtype=int64)

sensi_WS_train
# 0.925
speci_WS_train
# 0.9647943037974683
sensi_WS_train + speci_WS_train
# 1.8897943037974683
accuracy_WS_train
# 0.9606568998109641                                    
                                                
[cm_WS_test, sensi_WS_test, speci_WS_test, 
 accuracy_WS_test, ypred_WS_test] = getValues(c_words_final, nc_words_final,  
                                    X_test, y_test)       
                            
cm_WS_test
# array([[ 347,   30],
#        [ 110, 3141]], dtype=int64)

sensi_WS_test
# 0.9204244031830239
speci_WS_test
# 0.9661642571516457
sensi_WS_test + speci_WS_test
# 1.8865886603346695
accuracy_WS_test
# 0.9614112458654906    
                                           
os.chdir(r'W:\your_folder\Output')
# =============================================================================
# # ===========================================================================
# # dtm
# # ===========================================================================
# =============================================================================
dtm_test_old = dtm_test
y_test_old = y_test_10
dtm_train_old = dtm_train
y_train_old = y_train_10
# =============================================================================
# Oversampling
# =============================================================================

random.seed(1802)
ros = RandomOverSampler(random_state=0, sampling_strategy = 0.5)
dtm_train, y_train_10 = ros.fit_resample(np.array(dtm_train.T), np.array(y_train_10))
# dtm_test, y_test_10 = ros.fit_resample(np.array(dtm_test.T), np.array(y_test_10))

dtm_train = pd.DataFrame(dtm_train)
dtm_train.columns = dtm_train_old.T.columns
dtm_train = dtm_train.T.sort_index().T

# dtm_test = pd.DataFrame(dtm_test)
# dtm_test.columns = dtm_test_old.T.columns
dtm_test = dtm_test.T.sort_index()

# my indices

features = c_words_final + nc_words_final
dtm_trT = dtm_tr.T
dtm_train_cl = dtm_trT[features]

dtm_teT = dtm_te.T
dtm_test_small = dtm_teT[features]

random.seed(1802)
ros = RandomOverSampler(random_state=0, sampling_strategy = 0.5)
dtm_train_small, y_train_small = ros.fit_resample(np.array(dtm_train_sm), np.array(y_train_old))
# dtm_test_small, y_test_small = ros.fit_resample(np.array(dtm_test_sm), np.array(y_test_old))
y_test_small = y_test_old

dtm_train_small = pd.DataFrame(dtm_train_small)
dtm_train_small.columns = features

dtm_test_small = pd.DataFrame(dtm_test_small)
dtm_test_small.columns = features

# =============================================================================
# Random Forest
# =============================================================================

# cm now structured the same way as in the word search in terms of splitting
# threshold = False -> as Python would split it
# sensi = true positive
# speci = true negative = 1 - false positive rate

### with the large matrices ===================================================
random.seed(1802)
[cm_RF_train, sensi_RF_train, speci_RF_train, accuracy_RF_train, y_pred_RF_train, 
 mydf_RF_train] = RFAll(dtm_train, dtm_train, y_train_10, y_train_10, X_train_ind)
   
cm_RF_train
# array([[3792,    0],
#        [ 660, 6924]])
sensi_RF_train # 1.0
speci_RF_train # 0.9129746835443038
sensi_RF_train + speci_RF_train # 1.9129746835443038
accuracy_RF_train # 0.9419831223628692
 
probs_pos_RF_train = mydf_RF_train['Prob pos'][mydf_RF_train['real class'] == 1]
probs_neg_RF_train = mydf_RF_train['Prob pos'][mydf_RF_train['real class'] == 0]           
HistFuncC(probs_pos_RF_train, 'hist_C_RF_train.png')
HistFuncNC(probs_neg_RF_train, 'hist_NC_RF_train.png')

ROCFunc(y_train_10, mydf_RF_train['Prob pos'], 'ROC_RF_train.png')                   
auc_RF_train = metrics.roc_auc_score(y_train_10, mydf_RF_train['Prob pos'])             
  
th = pd.DataFrame(metrics.roc_curve(y_train_10, mydf_RF_train['Prob pos'])).T
th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th['speci'] = 1 - th['false_positive rate']
th['sensi + speci'] = th['speci'] + th['true_positive_rate']
th_sorted = th.sort_values('sensi + speci')
th_sorted

ts_RF = list(th['threshold'][th['sensi + speci'] == max(th['sensi + speci'])])[0]
ts_RF # 0.34608605846093965


if min(probs_pos_RF_train) > max(probs_neg_RF_train):
    random.seed(1802)
    ts_RF = random.uniform(max(probs_neg_RF_train), min(probs_pos_RF_train))


random.seed(1802)
[cm_RF_test, sensi_RF_test, speci_RF_test, accuracy_RF_test, y_pred_RF_test, 
 mydf_RF_test] = RFAll(dtm_train, dtm_test, y_train_10, y_test_10, X_test_ind,
                       threshold_RF = ts_RF)
       
cm_RF_test
# array([[ 187,  190],
#        [1616, 1635]])
                    
sensi_RF_test #  0.4960212201591512
speci_RF_test # 0.5029221777914488
sensi_RF_test + speci_RF_test #  0.9989433979505999
accuracy_RF_test  # 0.5022050716648291                 

probs_pos_RF_test = mydf_RF_test['Prob pos'][mydf_RF_test['real class'] == 1]
probs_neg_RF_test = mydf_RF_test['Prob pos'][mydf_RF_test['real class'] == 0]
HistFuncC(probs_pos_RF_test, 'hist_K_RF_test.png')
HistFuncNC(probs_neg_RF_test, 'hist_NK_RF_test.png')

ROCFunc(y_test_10, mydf_RF_test['Prob pos'], 'ROC_RF_test.png')                  
auc_RF_test = metrics.roc_auc_score(y_test_10, mydf_RF_test['Prob pos'])    

### with the small matrices ==================================================         

random.seed(1802)
[cm_RF_train_sm, sensi_RF_train_sm, speci_RF_train_sm, accuracy_RF_train_sm, 
 y_pred_RF_train_sm, mydf_RF_train_sm] = RFAll(dtm_train_small, dtm_train_small, y_train_small, y_train_small, X_train_ind)
   
cm_RF_train_sm
# array([[3503,  289],
#        [ 206, 7378]])
sensi_RF_train_sm # 0.9237869198312236
speci_RF_train_sm # 0.9728375527426161
sensi_RF_train_sm + speci_RF_train_sm #  1.8966244725738397
accuracy_RF_train_sm #  0.9564873417721519
 

probs_pos_RF_train_sm = mydf_RF_train_sm['Prob pos'][mydf_RF_train_sm['real class'] == 1]
probs_neg_RF_train_sm = mydf_RF_train_sm['Prob pos'][mydf_RF_train_sm['real class'] == 0]           
HistFuncC(probs_pos_RF_train_sm, 'hist_C_RF_train_sm.png')
HistFuncNC(probs_neg_RF_train_sm, 'hist_NC_RF_train_sm.png')

ROCFunc(y_train_small, mydf_RF_train_sm['Prob pos'], 'ROC_RF_train_sm.png')                  
auc_RF_train_sm = metrics.roc_auc_score(y_train_small, mydf_RF_train_sm['Prob pos'])             
  
th_sm = pd.DataFrame(metrics.roc_curve(y_train_small, mydf_RF_train_sm['Prob pos'])).T
th_sm.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th_sm['speci'] = 1 - th_sm['false_positive rate']
th_sm['sensi + speci'] = th_sm['speci'] + th_sm['true_positive_rate']
th_sm_sorted = th_sm.sort_values('sensi + speci')

ts_sm_RF = list(th_sm['threshold'][th_sm['sensi + speci'] == max(th_sm['sensi + speci'])])[0]
ts_sm_RF # 0.3344046048140943

if min(probs_pos_RF_train_sm) > max(probs_neg_RF_train_sm):
    random.seed(1802)
    ts_sm_RF = random.uniform(max(probs_neg_RF_train_sm), min(probs_pos_RF_train_sm))

y_pred_RF_train_sm_new = getPred(mydf_RF_train_sm['Prob pos'], ts_sm_RF)
[cm_RF_train_sm_new, sensi_RF_train_sm_new, speci_RF_train_sm_new,
 accuracy_RF_train_sm_new] = getValuesClass(y_train_small, y_pred_RF_train_sm_new)

# same as above

random.seed(1802)
[cm_RF_test_sm, sensi_RF_test_sm, speci_RF_test_sm, accuracy_RF_test_sm, 
 y_pred_RF_test_sm, mydf_RF_test_sm] = RFAll(dtm_train_small, dtm_test_small, y_train_small, y_test_small, X_test_ind,
                       threshold_RF = ts_sm_RF)
                                             
       
cm_RF_test_sm
# array([[ 344,   33],
#        [  77, 3174]])               
sensi_RF_test_sm # 0.9124668435013262
speci_RF_test_sm # 0.9763149800061519
sensi_RF_test_sm + speci_RF_test_sm # 1.888781823507478
accuracy_RF_test_sm  # 0.9696802646085998

probs_pos_RF_test_sm = mydf_RF_test_sm['Prob pos'][mydf_RF_test_sm['real class'] == 1]
probs_neg_RF_test_sm = mydf_RF_test_sm['Prob pos'][mydf_RF_test_sm['real class'] == 0]
HistFuncC(probs_pos_RF_test_sm, 'hist_K_RF_test_sm.png')
HistFuncNC(probs_neg_RF_test_sm, 'hist_NK_RF_test_sm.png')

ROCFunc(y_test_small, mydf_RF_test_sm['Prob pos'], 'ROC_RF_test_sm.png')                  
auc_RF_test_sm = metrics.roc_auc_score(y_test_small, mydf_RF_test_sm['Prob pos'])    

os.chdir(r'W:\your_folder\Output')
# dill.dump_session('onlyRF_5A_04_27.pkl')
# dill.load_session('onlyRF_5A_04_27.pkl')

#comparison big - small

cm_RF_train
# array([[3792,    0],
#        [ 660, 6924]])

cm_RF_train_sm
# array([[3503,  289],
#        [ 206, 7378]])

cm_RF_test
# array([[ 187,  190],
#        [1616, 1635]])
sensi_RF_test + speci_RF_test #0.9989433979505999

cm_RF_test_sm
# array([[ 344,   33],
#        [  77, 3174]]) 
sensi_RF_test_sm + speci_RF_test_sm # 1.888781823507478

# best RF: small model

# # =============================================================================
# SVM
# =============================================================================

### on my tokens =========================================================

random.seed(1802)
[cm_SVM_train_sm, sensi_SVM_train_sm, speci_SVM_train_sm, accuracy_SVM_train_sm, 
 y_pred_SVM_train_sm, mydf_SVM_train_sm] = SVMAll(dtm_train_small, dtm_train_small, y_train_small, y_train_small, X_train_ind)
   
cm_SVM_train_sm
# array([[3444,  348],
#        [ 239, 7345]])

sensi_SVM_train_sm #  0.9082278481012658 
speci_SVM_train_sm # 0.9684862869198312
sensi_SVM_train_sm + speci_SVM_train_sm # 1.876714135021097
accuracy_SVM_train_sm # 0.9484001406469761
 

probs_pos_SVM_train_sm = mydf_SVM_train_sm['Prob pos'][mydf_SVM_train_sm['real class'] == 1]
probs_neg_SVM_train_sm = mydf_SVM_train_sm['Prob pos'][mydf_SVM_train_sm['real class'] == 0]           
HistFuncC(probs_pos_SVM_train_sm, 'hist_C_SVM_train_sm.png')
HistFuncNC(probs_neg_SVM_train_sm, 'hist_NC_SVM_train_sm.png')

ROCFunc(y_train_small, mydf_SVM_train_sm['Prob pos'], 'ROC_SVM_train_sm.png')                 
auc_SVM_train_sm = metrics.roc_auc_score(y_train_small, mydf_SVM_train_sm['Prob pos'])             
  
th_sm = pd.DataFrame(metrics.roc_curve(y_train_small, mydf_SVM_train_sm['Prob pos'])).T
th_sm.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th_sm['speci'] = 1 - th_sm['false_positive rate']
th_sm['sensi + speci'] = th_sm['speci'] + th_sm['true_positive_rate']
th_sm_sorted = th_sm.sort_values('sensi + speci')

ts_sm_SVM = list(th_sm['threshold'][th_sm['sensi + speci'] == max(th_sm['sensi + speci'])])[0]

if min(probs_pos_SVM_train_sm) > max(probs_neg_SVM_train_sm):
    random.seed(1802)
    ts_sm_SVM = random.uniform(max(probs_neg_SVM_train_sm), min(probs_pos_SVM_train_sm))


y_pred_SVM_train_sm_new = getPred(mydf_SVM_train_sm['Prob pos'], ts_sm_SVM)
[cm_SVM_train_sm_new, sensi_SVM_train_sm_new, speci_SVM_train_sm_new,
 accuracy_SVM_train_sm_new] = getValuesClass(y_train_small, y_pred_SVM_train_sm_new)

cm_SVM_train_sm_new
# array([[3504,  288],
#        [ 288, 7296]])
sensi_SVM_train_sm_new # 0.9240506329113924  
speci_SVM_train_sm_new # 0.9620253164556962
sensi_SVM_train_sm_new + speci_SVM_train_sm_new # 1.8860759493670887
accuracy_SVM_train_sm_new # 0.9493670886075949

# therefore, the LOWER one is the final model!

random.seed(1802)
[cm_SVM_test_sm, sensi_SVM_test_sm, speci_SVM_test_sm, accuracy_SVM_test_sm, 
 y_pred_SVM_test_sm, mydf_SVM_test_sm] = SVMAll(dtm_train_small, dtm_test_small, y_train_small, y_test_small, X_test_ind,
                       threshold_SVM = ts_sm_SVM)
       
cm_SVM_test_sm
# array([[ 349,   28],
#        [ 104, 3147]])
sensi_SVM_test_sm #  0.9257294429708223
speci_SVM_test_sm #   0.9680098431251922
sensi_SVM_test_sm + speci_SVM_test_sm #  1.8937392860960145
accuracy_SVM_test_sm # 0.9636163175303197

probs_pos_SVM_test_sm = mydf_SVM_test_sm['Prob pos'][mydf_SVM_test_sm['real class'] == 1]
probs_neg_SVM_test_sm = mydf_SVM_test_sm['Prob pos'][mydf_SVM_test_sm['real class'] == 0]
HistFuncC(probs_pos_SVM_test_sm, 'hist_C_SVM_test_sm.png')
HistFuncNC(probs_neg_SVM_test_sm, 'hist_NC_SVM_test_sm.png')

ROCFunc(y_test_small, mydf_SVM_test_sm['Prob pos'], 'ROC_SVM_test_sm.png')          
                  
auc_SVM_test_sm = metrics.roc_auc_score(y_test_small, mydf_SVM_test_sm['Prob pos'])    

os.chdir(r'W:\your_folder\Output')

### on the full matrices ===================================================

random.seed(1802)
[cm_SVM_train, sensi_SVM_train, speci_SVM_train, accuracy_SVM_train, 
 y_pred_SVM_train, mydf_SVM_train] = SVMAll(dtm_train, dtm_train, y_train_10, y_train_10, X_train_ind)
   
cm_SVM_train
# array([[3792,    0],
#        [2106, 5478]])

sensi_SVM_train # 1.0
speci_SVM_train # 0.7223101265822784
sensi_SVM_train + speci_SVM_train # 1.7223101265822784
accuracy_SVM_train # 0.814873417721519
 

probs_pos_SVM_train = mydf_SVM_train['Prob pos'][mydf_SVM_train['real class'] == 1]
probs_neg_SVM_train = mydf_SVM_train['Prob pos'][mydf_SVM_train['real class'] == 0]           
HistFuncC(probs_pos_SVM_train, 'hist_K_SVM_train.png')
HistFuncNC(probs_neg_SVM_train, 'hist_NK_SVM_train.png')

ROCFunc(y_train_10, mydf_SVM_train['Prob pos'], 'ROC_SVM_train.png')                  
auc_SVM_train = metrics.roc_auc_score(y_train_10, mydf_SVM_train['Prob pos'])             
  
th = pd.DataFrame(metrics.roc_curve(y_train_10, mydf_SVM_train['Prob pos'])).T
th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th['speci'] = 1 - th['false_positive rate']
th['sensi + speci'] = th['speci'] + th['true_positive_rate']
th_sorted = th.sort_values('sensi + speci')

ts_SVM = list(th['threshold'][th['sensi + speci'] == max(th['sensi + speci'])])[0]

if min(probs_pos_SVM_train) > max(probs_neg_SVM_train):
    random.seed(1802)
    ts_SVM = random.uniform(max(probs_neg_SVM_train), min(probs_pos_SVM_train))


y_pred_SVM_train_new = getPred(mydf_SVM_train['Prob pos'], ts_SVM)
[cm_SVM_train_new, sensi_SVM_train_new, speci_SVM_train_new,
 accuracy_SVM_train_new] = getValuesClass(y_train_10, y_pred_SVM_train_new)

cm_SVM_train_new
# array([[3792,    0],
#        [ 660, 6924]])

sensi_SVM_train_new # 1.0
speci_SVM_train_new # 0.9129746835443038
sensi_SVM_train_new + speci_SVM_train_new #1.9129746835443038
accuracy_SVM_train_new # 0.9419831223628692

# the lower model is the better one

random.seed(1802)
[cm_SVM_test, sensi_SVM_test, speci_SVM_test, accuracy_SVM_test, 
 y_pred_SVM_test, mydf_SVM_test] = SVMAll(dtm_train, dtm_test, y_train_10, y_test_10, X_test_ind,
                       threshold_SVM = ts_SVM)
       
cm_SVM_test
# array([[  54,  323],
#        [ 444, 2807]])

sensi_SVM_test # 0.14323607427055704

speci_SVM_test # 0.8634266379575515

sensi_SVM_test + speci_SVM_test # 1.0066627122281084

accuracy_SVM_test # 0.7885887541345094

probs_pos_SVM_test = mydf_SVM_test['Prob pos'][mydf_SVM_test['real class'] == 1]
probs_neg_SVM_test = mydf_SVM_test['Prob pos'][mydf_SVM_test['real class'] == 0]
HistFuncC(probs_pos_SVM_test, 'hist_C_SVM_test.png')
HistFuncNC(probs_neg_SVM_test, 'hist_NC_SVM_test.png')

ROCFunc(y_test_10, mydf_SVM_test['Prob pos'], 'ROC_SVM_test.png')    
                   
auc_SVM_test = metrics.roc_auc_score(y_test_10, mydf_SVM_test['Prob pos'])    

os.chdir(r'W:\your_folder\Output')

## big and small comparison

cm_SVM_train_new
# array([[3792,    0],
#        [ 660, 6924]])

cm_SVM_train_sm_new
# array([[3504,  288],
#        [ 288, 7296]])

cm_SVM_test
# vorher
# array([[1617,    8],
#        [ 935, 2316]])
# jetzt
# array([[  54,  323],
#        [ 444, 2807]])
sensi_SVM_test + speci_SVM_test # 1.0066627122281084

cm_SVM_test_sm
# array([[ 349,   28],
#        [ 104, 3147]])
sensi_SVM_test_sm + speci_SVM_test_sm # 1.8937392860960145


### best models regarding sensi + speci on testdata

cm_WS_test
# array([[ 347,   30],
#        [ 110, 3141]], dtype=int64)

sensi_WS_test + speci_WS_test # 1.8865886603346695

cm_RF_test_sm
# array([[ 344,   33],
#        [  77, 3174]])

sensi_RF_test_sm + speci_RF_test_sm # 1.888781823507478
sensi_RF_test_sm # 0.9124668435013262

cm_SVM_test_sm
# array([[ 349,   28],
#        [ 104, 3147]])

sensi_SVM_test_sm + speci_SVM_test_sm # 1.8937392860960145
sensi_SVM_test_sm # 0.9257294429708223

# winning model: small SVM (but everything is extremely close together!)
