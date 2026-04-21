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

os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
from Funktionen import *
# =============================================================================
# Daten einlesen & Prep
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
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
# Wortsuche
# =============================================================================

K_ohneNK = set(X_train_K).difference(X_train_NK)
len(K_ohneNK)  # 862
NK_ohneK = set(X_train_NK).difference(X_train_K)
len(NK_ohneK)  # 7119
# viel zu viel um das alles anzugucken

stop_words_stem = getStem(get_stop_words('de'))
    
createWC(' '.join(X_train_K), stop_words_stem, 'wolkeK_1.png')
createWC(' '.join(X_train_NK), stop_words_stem, 'wolkeNK_1.png')

# entferne alle mono- und bigramme aus dem am häufigsten vorkommenden 200, außer
# denen, die etwas mit einem der GeVos zu tun haben könnten (oder explizit keine
# Kündigung vermuten lassen)

[haufig_sl_K, haufig_sl_K_bi] = getHäufigeWörter(X_train_K)
haufig_sl_K
gevowords_k_mono = ['auszahl', 'beitragsfrei', 'gekundigt', 'kundig', 
                    'kundigungsbestat', 'kundigungstermin', 'ruckkaufswert', 
                    'ruckkaufwert', 'teil', 'teilkund', 'versicherungsnehm']
haufig_sl_kl_K = haufig_sl_K.difference(gevowords_k_mono)
set(haufig_sl_K_bi)
gevowords_k_bi = ['anbei kundig', 'auszahl ruckkaufswert', 
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
haufig_sl_kl_bi_K = set(haufig_sl_K_bi).difference(gevowords_k_bi)

[haufig_sl_NK, haufig_sl_NK_bi] = getHäufigeWörter(X_train_NK)
haufig_sl_NK
gevowords_nk_mono = ['arbeitgeb', 'arbeitnehm', 'ausschluss', 'beitrag', 
                     'beitragszahl', 'beschäftigungsverhältnis',
                     'gesundheitsdat', 'todesfall', 
                     'versicherungsnehmerwechsel', 'wechsel']
haufig_sl_kl_NK = haufig_sl_NK.difference(gevowords_nk_mono)
set(haufig_sl_NK_bi)
gevowords_nk_bi = ['abfrag gesundheitsdat', 'arbeitgeb ausgeschloss',
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

haufig_sl_kl_bi_NK = set(haufig_sl_NK_bi).difference(gevowords_nk_bi)

# entferne häufig vorkommende Wörter, die keine Relevanz haben sollten
# oft_KNK = ['signal', 'iduna', 'sehr', 'geehrte', 'damen', 'und', 'herren',
#            'und herren', 'freundlichen grüßen', 'mit freundlichen', 'iduna gruppe',
#            'versicherung', 'lebensversicherung', 'gruppe', 'bitte']

stop_new = stop_words + list(haufig_sl_kl_K) + list(haufig_sl_kl_bi_K) + list(haufig_sl_kl_NK) + list(haufig_sl_kl_bi_NK)
createWC(' '.join(X_train_K), stop_new, 'wolkeK_2.png')
createWC(' '.join(X_train_NK), stop_new, 'wolkeNK_2.png')


k_words = ['auszahl', 'gekundigt', 'kundig', 'kundigungsbestat',
           'kundigungstermin', 'ruckkaufswert', 'ruckkaufwert', 'teilkund']
k_bigrams = ['auszahl ruckkaufswert', 'auszahl ruckkaufwert', 
              'ruckkaufswert uberweis', 'ruckkaufwert uberweis']

k_test = k_words + k_bigrams

[inK, summeK, inNK, summeNK] =  getAnzahlen(k_words, X_train_K, X_train_NK)

inKproz = [round(j/len(X_train_K) * 100, 2) for j in inK]
inNKproz = [round(j/len(X_train_NK) * 100, 2) for j in inNK]
k_test_KNK = pd.DataFrame([k_words + ['GESAMT'], inK + [summeK], inKproz, inNK + [summeNK], inNKproz]).T
k_test_KNK.columns = ['Token', 'in K', 'in K proz', 'in NK', 'in NK proz']

#               Token in K in K proz in NK in NK proz
# 0           auszahl  177     14.08   115       1.06
# 1         gekundigt   17      1.35    68       0.63
# 2            kundig  780     62.05   312       2.88
# 3  kundigungsbestat   97      7.72     4       0.04
# 4  kundigungstermin   25      1.99     6       0.06
# 5     ruckkaufswert  231     18.38   130        1.2
# 6      ruckkaufwert   59      4.69    11        0.1
# 7          teilkund   26      2.07     4       0.04
# 8            GESAMT  819      None   475       None

# gekundigt ist der einzige Wortstamm, der häufiger in den NKs als ins den Ks
# vorkommt

[inK_bi, summeK_bi, inNK_bi, summeNK_bi] =  getAnzahlen(k_bigrams, X_train_K, X_train_NK)
k_test_KNK_bi = pd.DataFrame([k_bigrams + ['GESAMT'], inK_bi + [summeK_bi], inNK_bi + [summeNK_bi]]).T
k_test_KNK_bi.columns = ['Token', 'in K', 'in NK']
k_test_KNK_bi

#                     Token in K in NK
# 0   auszahl ruckkaufswert   61    18
# 1    auszahl ruckkaufwert   23     2
# 2  ruckkaufswert uberweis   24     0
# 3   ruckkaufwert uberweis    3     0
# 4                  GESAMT  110    20

# diese bigramme kommen also alle häufiger in den Kündigungen als in den 
# anderen Dokumenten vor. Trotzdem werden die Bigramme nicht weiter verwendet, 
# weil das Ziel ist, möglichst viele Kündigungen zu finden, und durch das 
# weglassen von auszahl oder ruckkaufswert zugunsten des bigrams
# gelingt dies nicht

# daher betrachte das modell mit den k_words

[cm_K, sensi_K, spezi_K, richtigkl_K, ypred_K] = getWerte(k_words, [], 
                                                           X_train, y_train)

cm_K
# array([[ 819,   61],
#        [ 475, 7109]], dtype=int64)

sensi_K # 0.9306818181818182
spezi_K # 0.9373681434599156
sensi_K + spezi_K # 1.8680499616417339
richtigkl_K # 0.9366729678638941

# und betrachte das Modell ohne gekundigt
k_words_og = ['auszahl', 'kundig', 'kundigungsbestat', 'kundigungstermin',
              'ruckkaufswert', 'ruckkaufwert', 'teilkund']

[cm_K_og, sensi_K_og, spezi_K_og, richtigkl_K_og, ypred_K_og] = getWerte(k_words_og, [], 
                                                           X_train, y_train)

cm_K_og
# array([[ 816,   64],
#        [ 441, 7143]], dtype=int64)

sensi_K_og # 0.9272727272727272
spezi_K_og # 0.9418512658227848
sensi_K_og + spezi_K_og # 1.8691239930955121
richtigkl_K_og # 0.940335538752363


# das Modell ohne gekundigt wird genommen

k_words_final = k_words_og

#### NK ==============================

# häufig vorkommende Wörter (die zu den anderen GeVos gehören könnten) und davon
# abgewandelte Wörter und wörter die mit den anderen GeVos zu tun haben
nk_words = ['beitragsfreistell', 'beitragspaus','erhoh', 'geschutzt dat', 
            'gesundheitsdat', 'pausi', 'stell', 'versicherungsnehm', 
            'versichert person', 'weitergab', 'ubertrag']


#wie oft kommen die nk Wörter in nk und wie oft in k vor (nicht in wie vielen 
# Dokumenten sondern wie oft, da es nur eine grobe Anzahl ergeben soll und das 
# Ziel der ersten Methoden ist, dass sie möglichst schnell funktioniert)

# betrachte nun nur die Dokumente, die fälschlicherweise als Kündigung klassifiziert
# wurden, der Rest wird ja so oder so richtig klassifiziert   
predK = myGleich(ypred_K_og, 'K')
aktN = myGleich(y_train, 'N')
aktK = myGleich(y_train, 'K')
Nfalsch = list(set(predK).intersection(aktN))
Krichtig = list(set(predK).intersection(aktK))
X_train_NK_falsch = list(np.array(X_train)[Nfalsch])
X_train_K_richtig = list(np.array(X_train)[Krichtig])

[inK_richtig, summeK_richtig, inNK_falsch, summeNK_falsch] = getAnzahlen(nk_words, X_train_K_richtig, X_train_NK_falsch)
   
nk_test_KNK = pd.DataFrame([nk_words + ['GESAMT'], inK_richtig + [summeK_richtig], inNK_falsch + [summeNK_falsch]]).T
nk_test_KNK.columns = ['Token', 'in K', 'in NK']
nk_test_KNK

#                 Token in K in NK
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
# 11             GESAMT  134   317

# pausi kann raus, da bei beiden 0
# beitragspaus, geschutzt dat und gesundheitsdat können rein, da in K nicht vorkommt

### Modell mit beitragspaus, geschutzt dat und gesundheitdat ==================

[cm_bg, sensi_bg, spezi_bg, richtigkl_bg, ypred_bg] = getWerte(k_words_final, ['beitragspaus', 'geschutzt dat', 'gesundheitsdat'], 
                                                           X_train, y_train)

cm_bg
# array([[ 816,   64],
#        [ 331, 7253]], dtype=int64)

sensi_bg + spezi_bg # 1.8836282125047947

predK = myGleich(ypred_bg, 'K')
aktN = myGleich(y_train, 'N')
aktK = myGleich(y_train, 'K')
Nfalsch = list(set(predK).intersection(aktN))
Krichtig = list(set(predK).intersection(aktK))
X_train_NK_falsch = list(np.array(X_train)[Nfalsch])
X_train_K_richtig = list(np.array(X_train)[Krichtig])

nk_bg = ['beitragsfreistell', 'erhoh',  'stell', 
         'versichert person', 'versicherungsnehm', 'weitergab', 'ubertrag']
[inK_richtig, summeK_richtig, inNK_falsch, summeNK_falsch] = getAnzahlen(nk_bg, X_train_K_richtig, X_train_NK_falsch)
   
nk_test_KNK_bg = pd.DataFrame([nk_bg + ['GESAMT'], inK_richtig + [summeK_richtig], inNK_falsch + [summeNK_falsch]]).T
nk_test_KNK_bg.columns = ['Token', 'in K', 'in NK']
nk_test_KNK_bg['in K (gleiche Skala)'] = nk_test_KNK_bg['in K'] * len(X_train_NK)/len(X_train_K)
nk_test_KNK_bg

#                Token in K in NK in K (gleiche Skala)
# 0  beitragsfreistell   12    38           103.418182
# 1              erhoh    2    64            17.236364
# 2              stell   18    42           155.127273
# 3  versichert person   72    87           620.509091
# 4  versicherungsnehm   77    86                663.6
# 5          weitergab    5    11            43.090909
# 6           ubertrag    7    61            60.327273
# 7             GESAMT  134   207          1154.836364

len(X_train_NK)/len(X_train_K) # 8.62

# wir wollen Sensititvität plus Spezifität verbessern, das geht also nur mit 
# denen, die in den NK mindestens 8.62 mal so häufig vorkommen, wie in den 
# Kündigungen, da sonst die Spezifität nicht genug wächst. 
# das ist nur fuer erhoh der fall

### Modell mit beitragspaus, geschutzt dat, gesundheitsdat, erhoh =============

[cm_bge, sensi_bge, spezi_bge, richtigkl_bge, ypred_bge] = getWerte(k_words_final, ['beitragspaus', 'geschutzt dat', 'gesundheitsdat', 'erhoh'], 
                                                           X_train, y_train)

cm_bge
# array([[ 814,   66],
#        [ 267, 7317]], dtype=int64)

sensi_bge + spezi_bge # 1.8897943037974683


predK = myGleich(ypred_bge, 'K')
aktN = myGleich(y_train, 'N')
aktK = myGleich(y_train, 'K')
Nfalsch = list(set(predK).intersection(aktN))
Krichtig = list(set(predK).intersection(aktK))
X_train_NK_falsch = list(np.array(X_train)[Nfalsch])
X_train_K_richtig = list(np.array(X_train)[Krichtig])

nk_bge = ['beitragsfreistell', 'stell', 
         'versichert person', 'versicherungsnehm', 'weitergab', 'ubertrag']
[inK_richtig, summeK_richtig, inNK_falsch, summeNK_falsch] = getAnzahlen(nk_bge, X_train_K_richtig, X_train_NK_falsch)
   
nk_test_KNK_bge = pd.DataFrame([nk_bge + ['GESAMT'], inK_richtig + [summeK_richtig], inNK_falsch + [summeNK_falsch]]).T
nk_test_KNK_bge.columns = ['Token', 'in K', 'in NK']
nk_test_KNK_bge['in K (gleiche Skala)'] = nk_test_KNK_bge['in K'] * len(X_train_NK)/len(X_train_K)
nk_test_KNK_bge
 
#                Token in K in NK in K (gleiche Skala)
# 0  beitragsfreistell   11    23                 94.8
# 1              stell   18    24           155.127273
# 2  versichert person   71    64           611.890909
# 3  versicherungsnehm   76    55           654.981818
# 4          weitergab    5     8            43.090909
# 5           ubertrag    7    51            60.327273
# 6             GESAMT  132   143               1137.6   

# hier kann nichts mehr die summe von sensi und spezi verbessern

### demnach ist das Modell das finale :
    
nk_words_final = ['beitragspaus', 'geschutzt dat', 'gesundheitsdat', 'erhoh']

[cm_WS_train, sensi_WS_train, spezi_WS_train, 
 richtigkl_WS_train, ypred_WS_train] = getWerte(k_words_final, nk_words_final,  
                                    X_train, y_train)
  
cm_WS_train
# array([[ 814,   66],
#        [ 267, 7317]], dtype=int64)

sensi_WS_train
# 0.925
spezi_WS_train
# 0.9647943037974683
sensi_WS_train + spezi_WS_train
# 1.8897943037974683
richtigkl_WS_train
# 0.9606568998109641                                    
                                                
[cm_WS_test, sensi_WS_test, spezi_WS_test, 
 richtigkl_WS_test, ypred_WS_test] = getWerte(k_words_final, nk_words_final,  
                                    X_test, y_test)       
                            
cm_WS_test
# array([[ 347,   30],
#        [ 110, 3141]], dtype=int64)

sensi_WS_test
# 0.9204244031830239
spezi_WS_test
# 0.9661642571516457
sensi_WS_test + spezi_WS_test
# 1.8865886603346695
richtigkl_WS_test
# 0.9614112458654906    
                                           
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
# dill.dump_session('nachwortsuche_5A_WS_04_27.pkl')
# dill.load_session('nachwortsuche_5A_WS_04_27.pkl')
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
# mydf = pd.concat([pd.DataFrame(range(len(dtm_train.T))).T, pd.DataFrame(range(len(dtm_train.T))).T])

# random.seed(1802)
# ros = RandomOverSampler(random_state=0, sampling_strategy = 0.5)
# dtm_indizes, yps = ros.fit_resample(np.array(mydf.T), np.array(y_train_10))
# dtm_indizes = dtm_indizes.T[0]
# dtm_train_new = pd.DataFrame(np.array(dtm_train.T)[dtm_indizes])      
# dtm_train_new.columns = dtm_train.T.columns 
# dtm_train_new = dtm_train_new.T 
# dtm_train_new.columns = dtm_indizes                                   
# dtm_test, y_test_10 = ros.fit_resample(np.array(dtm_test.T), np.array(y_test_10))

# # so funktioniert das nicht

# counter = []
# for i in range(len(dtm_train_old)):
#     counter.append(list(dtm_indizes).count(i))

# count_len = []
# for i in range(20):
#     count_len.append(len(myGleich(counter, i)))    
# sum(count_len)

# dtm_train = pd.DataFrame(dtm_train)
# dtm_train.columns = dtm_train_old.T.columns
# dtm_train = dtm_train.T.sort_index().T

# dtm_test = pd.DataFrame(dtm_test)
# dtm_test.columns = dtm_test_old.T.columns
# dtm_test = dtm_test.T.sort_index().T

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

# meine indizes

features = k_words_final + nk_words_final
dtm_trT = dtm_tr.T
dtm_train_kl = dtm_trT[features]

dtm_teT = dtm_te.T
dtm_test_klein = dtm_teT[features]

random.seed(1802)
ros = RandomOverSampler(random_state=0, sampling_strategy = 0.5)
dtm_train_klein, y_train_klein = ros.fit_resample(np.array(dtm_train_kl), np.array(y_train_old))
# dtm_test_klein, y_test_klein = ros.fit_resample(np.array(dtm_test_kl), np.array(y_test_old))
y_test_klein = y_test_old

dtm_train_klein = pd.DataFrame(dtm_train_klein)
dtm_train_klein.columns = features

dtm_test_klein = pd.DataFrame(dtm_test_klein)
dtm_test_klein.columns = features

# =============================================================================
# Random Forest
# =============================================================================

# cm jetzt genauso wie in der Wortsuche von der Aufteilung
# threshold = False -> so wie pyhton aufteilen würde
# sensi = true positive
# spezi = true negative = 1 - false positive rate

### mit den großen Matrizen ===================================================
random.seed(1802)
[cm_RF_train, sensi_RF_train, spezi_RF_train, richtigkl_RF_train, y_pred_RF_train, 
 mydf_RF_train] = RFAll(dtm_train, dtm_train, y_train_10, y_train_10, X_train_ind)
   
cm_RF_train
# array([[3792,    0],
#        [ 660, 6924]])
sensi_RF_train # 1.0
spezi_RF_train # 0.9129746835443038
sensi_RF_train + spezi_RF_train # 1.9129746835443038
richtigkl_RF_train # 0.9419831223628692
 
wkeiten_pos_RF_train = mydf_RF_train['Wkeit pos'][mydf_RF_train['wahre Klasse'] == 1]
wkeiten_neg_RF_train = mydf_RF_train['Wkeit pos'][mydf_RF_train['wahre Klasse'] == 0]           
HistFuncK(wkeiten_pos_RF_train, 'hist_K_RF_train.png')
HistFuncNK(wkeiten_neg_RF_train, 'hist_NK_RF_train.png')

ROCFunc(y_train_10, mydf_RF_train['Wkeit pos'], 'ROC_RF_train.png')                   
auc_RF_train = metrics.roc_auc_score(y_train_10, mydf_RF_train['Wkeit pos'])             
  
th = pd.DataFrame(metrics.roc_curve(y_train_10, mydf_RF_train['Wkeit pos'])).T
th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th['spezi'] = 1 - th['false_positive rate']
th['sensi + spezi'] = th['spezi'] + th['true_positive_rate']
th_sorted = th.sort_values('sensi + spezi')
th_sorted

ts_RF = list(th['threshold'][th['sensi + spezi'] == max(th['sensi + spezi'])])[0]
ts_RF # 0.34608605846093965


if min(wkeiten_pos_RF_train) > max(wkeiten_neg_RF_train):
    random.seed(1802)
    ts_RF = random.uniform(max(wkeiten_neg_RF_train), min(wkeiten_pos_RF_train))


random.seed(1802)
[cm_RF_test, sensi_RF_test, spezi_RF_test, richtigkl_RF_test, y_pred_RF_test, 
 mydf_RF_test] = RFAll(dtm_train, dtm_test, y_train_10, y_test_10, X_test_ind,
                       threshold_RF = ts_RF)
       
cm_RF_test
# array([[ 187,  190],
#        [1616, 1635]])
                    
sensi_RF_test #  0.4960212201591512
spezi_RF_test # 0.5029221777914488
sensi_RF_test + spezi_RF_test #  0.9989433979505999
richtigkl_RF_test  # 0.5022050716648291                 

wkeiten_pos_RF_test = mydf_RF_test['Wkeit pos'][mydf_RF_test['wahre Klasse'] == 1]
wkeiten_neg_RF_test = mydf_RF_test['Wkeit pos'][mydf_RF_test['wahre Klasse'] == 0]
HistFuncK(wkeiten_pos_RF_test, 'hist_K_RF_test.png')
HistFuncNK(wkeiten_neg_RF_test, 'hist_NK_RF_test.png')

ROCFunc(y_test_10, mydf_RF_test['Wkeit pos'], 'ROC_RF_test.png')                  
auc_RF_test = metrics.roc_auc_score(y_test_10, mydf_RF_test['Wkeit pos'])    

### mit den kleinen Matrizen ==================================================         

random.seed(1802)
[cm_RF_train_kl, sensi_RF_train_kl, spezi_RF_train_kl, richtigkl_RF_train_kl, 
 y_pred_RF_train_kl, mydf_RF_train_kl] = RFAll(dtm_train_klein, dtm_train_klein, y_train_klein, y_train_klein, X_train_ind)
   
cm_RF_train_kl
# array([[3503,  289],
#        [ 206, 7378]])
sensi_RF_train_kl # 0.9237869198312236
spezi_RF_train_kl # 0.9728375527426161
sensi_RF_train_kl + spezi_RF_train_kl #  1.8966244725738397
richtigkl_RF_train_kl #  0.9564873417721519
 

wkeiten_pos_RF_train_kl = mydf_RF_train_kl['Wkeit pos'][mydf_RF_train_kl['wahre Klasse'] == 1]
wkeiten_neg_RF_train_kl = mydf_RF_train_kl['Wkeit pos'][mydf_RF_train_kl['wahre Klasse'] == 0]           
HistFuncK(wkeiten_pos_RF_train_kl, 'hist_K_RF_train_kl.png')
HistFuncNK(wkeiten_neg_RF_train_kl, 'hist_NK_RF_train_kl.png')

ROCFunc(y_train_klein, mydf_RF_train_kl['Wkeit pos'], 'ROC_RF_train_kl.png')                  
auc_RF_train_kl = metrics.roc_auc_score(y_train_klein, mydf_RF_train_kl['Wkeit pos'])             
  
th_kl = pd.DataFrame(metrics.roc_curve(y_train_klein, mydf_RF_train_kl['Wkeit pos'])).T
th_kl.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th_kl['spezi'] = 1 - th_kl['false_positive rate']
th_kl['sensi + spezi'] = th_kl['spezi'] + th_kl['true_positive_rate']
th_kl_sorted = th_kl.sort_values('sensi + spezi')

ts_kl_RF = list(th_kl['threshold'][th_kl['sensi + spezi'] == max(th_kl['sensi + spezi'])])[0]
ts_kl_RF # 0.3344046048140943

if min(wkeiten_pos_RF_train_kl) > max(wkeiten_neg_RF_train_kl):
    random.seed(1802)
    ts_kl_RF = random.uniform(max(wkeiten_neg_RF_train_kl), min(wkeiten_pos_RF_train_kl))

y_pred_RF_train_kl_new = getPred(mydf_RF_train_kl['Wkeit pos'], ts_kl_RF)
[cm_RF_train_kl_new, sensi_RF_train_kl_new, spezi_RF_train_kl_new,
 richtigkl_RF_train_kl_new] = getWerteKlassi(y_train_klein, y_pred_RF_train_kl_new)

# genau wie oben

random.seed(1802)
[cm_RF_test_kl, sensi_RF_test_kl, spezi_RF_test_kl, richtigkl_RF_test_kl, 
 y_pred_RF_test_kl, mydf_RF_test_kl] = RFAll(dtm_train_klein, dtm_test_klein, y_train_klein, y_test_klein, X_test_ind,
                       threshold_RF = ts_kl_RF)
                                             
       
cm_RF_test_kl
# array([[ 344,   33],
#        [  77, 3174]])               
sensi_RF_test_kl # 0.9124668435013262
spezi_RF_test_kl # 0.9763149800061519
sensi_RF_test_kl + spezi_RF_test_kl # 1.888781823507478
richtigkl_RF_test_kl  # 0.9696802646085998

wkeiten_pos_RF_test_kl = mydf_RF_test_kl['Wkeit pos'][mydf_RF_test_kl['wahre Klasse'] == 1]
wkeiten_neg_RF_test_kl = mydf_RF_test_kl['Wkeit pos'][mydf_RF_test_kl['wahre Klasse'] == 0]
HistFuncK(wkeiten_pos_RF_test_kl, 'hist_K_RF_test_kl.png')
HistFuncNK(wkeiten_neg_RF_test_kl, 'hist_NK_RF_test_kl.png')

ROCFunc(y_test_klein, mydf_RF_test_kl['Wkeit pos'], 'ROC_RF_test_kl.png')                  
auc_RF_test_kl = metrics.roc_auc_score(y_test_klein, mydf_RF_test_kl['Wkeit pos'])    

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
# dill.dump_session('nurRF_5A_04_27.pkl')
# dill.load_session('nurRF_5A_04_27.pkl')

#vgl groß - klein

cm_RF_train
# array([[3792,    0],
#        [ 660, 6924]])

cm_RF_train_kl
# array([[3503,  289],
#        [ 206, 7378]])

cm_RF_test
# array([[ 187,  190],
#        [1616, 1635]])
sensi_RF_test + spezi_RF_test #0.9989433979505999

cm_RF_test_kl
# array([[ 344,   33],
#        [  77, 3174]]) 
sensi_RF_test_kl + spezi_RF_test_kl # 1.888781823507478

# bester RF: kleines Modell

# # =============================================================================
# SVM
# =============================================================================

### auf meinen Token =========================================================

random.seed(1802)
[cm_SVM_train_kl, sensi_SVM_train_kl, spezi_SVM_train_kl, richtigkl_SVM_train_kl, 
 y_pred_SVM_train_kl, mydf_SVM_train_kl] = SVMAll(dtm_train_klein, dtm_train_klein, y_train_klein, y_train_klein, X_train_ind)
   
cm_SVM_train_kl
# array([[3444,  348],
#        [ 239, 7345]])

sensi_SVM_train_kl #  0.9082278481012658 
spezi_SVM_train_kl # 0.9684862869198312
sensi_SVM_train_kl + spezi_SVM_train_kl # 1.876714135021097
richtigkl_SVM_train_kl # 0.9484001406469761
 

wkeiten_pos_SVM_train_kl = mydf_SVM_train_kl['Wkeit pos'][mydf_SVM_train_kl['wahre Klasse'] == 1]
wkeiten_neg_SVM_train_kl = mydf_SVM_train_kl['Wkeit pos'][mydf_SVM_train_kl['wahre Klasse'] == 0]           
HistFuncK(wkeiten_pos_SVM_train_kl, 'hist_K_SVM_train_kl.png')
HistFuncNK(wkeiten_neg_SVM_train_kl, 'hist_NK_SVM_train_kl.png')

ROCFunc(y_train_klein, mydf_SVM_train_kl['Wkeit pos'], 'ROC_SVM_train_kl.png')                 
auc_SVM_train_kl = metrics.roc_auc_score(y_train_klein, mydf_SVM_train_kl['Wkeit pos'])             
  
th_kl = pd.DataFrame(metrics.roc_curve(y_train_klein, mydf_SVM_train_kl['Wkeit pos'])).T
th_kl.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th_kl['spezi'] = 1 - th_kl['false_positive rate']
th_kl['sensi + spezi'] = th_kl['spezi'] + th_kl['true_positive_rate']
th_kl_sorted = th_kl.sort_values('sensi + spezi')

ts_kl_SVM = list(th_kl['threshold'][th_kl['sensi + spezi'] == max(th_kl['sensi + spezi'])])[0]

if min(wkeiten_pos_SVM_train_kl) > max(wkeiten_neg_SVM_train_kl):
    random.seed(1802)
    ts_kl_SVM = random.uniform(max(wkeiten_neg_SVM_train_kl), min(wkeiten_pos_SVM_train_kl))


y_pred_SVM_train_kl_new = getPred(mydf_SVM_train_kl['Wkeit pos'], ts_kl_SVM)
[cm_SVM_train_kl_new, sensi_SVM_train_kl_new, spezi_SVM_train_kl_new,
 richtigkl_SVM_train_kl_new] = getWerteKlassi(y_train_klein, y_pred_SVM_train_kl_new)

cm_SVM_train_kl_new
# array([[3504,  288],
#        [ 288, 7296]])
sensi_SVM_train_kl_new # 0.9240506329113924  
spezi_SVM_train_kl_new # 0.9620253164556962
sensi_SVM_train_kl_new + spezi_SVM_train_kl_new # 1.8860759493670887
richtigkl_SVM_train_kl_new # 0.9493670886075949

# also ist das UNTERE das finale Modell!

random.seed(1802)
[cm_SVM_test_kl, sensi_SVM_test_kl, spezi_SVM_test_kl, richtigkl_SVM_test_kl, 
 y_pred_SVM_test_kl, mydf_SVM_test_kl] = SVMAll(dtm_train_klein, dtm_test_klein, y_train_klein, y_test_klein, X_test_ind,
                       threshold_SVM = ts_kl_SVM)
       
cm_SVM_test_kl
# array([[ 349,   28],
#        [ 104, 3147]])
sensi_SVM_test_kl #  0.9257294429708223
spezi_SVM_test_kl #   0.9680098431251922
sensi_SVM_test_kl + spezi_SVM_test_kl #  1.8937392860960145
richtigkl_SVM_test_kl # 0.9636163175303197

wkeiten_pos_SVM_test_kl = mydf_SVM_test_kl['Wkeit pos'][mydf_SVM_test_kl['wahre Klasse'] == 1]
wkeiten_neg_SVM_test_kl = mydf_SVM_test_kl['Wkeit pos'][mydf_SVM_test_kl['wahre Klasse'] == 0]
HistFuncK(wkeiten_pos_SVM_test_kl, 'hist_K_SVM_test_kl.png')
HistFuncNK(wkeiten_neg_SVM_test_kl, 'hist_NK_SVM_test_kl.png')

ROCFunc(y_test_klein, mydf_SVM_test_kl['Wkeit pos'], 'ROC_SVM_test_kl.png')          
                  
auc_SVM_test_kl = metrics.roc_auc_score(y_test_klein, mydf_SVM_test_kl['Wkeit pos'])    

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
# dill.dump_session('nachSVMmeine_5A_04_27.pkl')
# dill.load_session('nachSVMmeine_5A_04_27.pkl')

### auf den großen Matrizen ===================================================

random.seed(1802)
[cm_SVM_train, sensi_SVM_train, spezi_SVM_train, richtigkl_SVM_train, 
 y_pred_SVM_train, mydf_SVM_train] = SVMAll(dtm_train, dtm_train, y_train_10, y_train_10, X_train_ind)
   
cm_SVM_train
# array([[3792,    0],
#        [2106, 5478]])

sensi_SVM_train # 1.0
spezi_SVM_train # 0.7223101265822784
sensi_SVM_train + spezi_SVM_train # 1.7223101265822784
richtigkl_SVM_train # 0.814873417721519
 

wkeiten_pos_SVM_train = mydf_SVM_train['Wkeit pos'][mydf_SVM_train['wahre Klasse'] == 1]
wkeiten_neg_SVM_train = mydf_SVM_train['Wkeit pos'][mydf_SVM_train['wahre Klasse'] == 0]           
HistFuncK(wkeiten_pos_SVM_train, 'hist_K_SVM_train.png')
HistFuncNK(wkeiten_neg_SVM_train, 'hist_NK_SVM_train.png')

ROCFunc(y_train_10, mydf_SVM_train['Wkeit pos'], 'ROC_SVM_train.png')                  
auc_SVM_train = metrics.roc_auc_score(y_train_10, mydf_SVM_train['Wkeit pos'])             
  
th = pd.DataFrame(metrics.roc_curve(y_train_10, mydf_SVM_train['Wkeit pos'])).T
th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
th['spezi'] = 1 - th['false_positive rate']
th['sensi + spezi'] = th['spezi'] + th['true_positive_rate']
th_sorted = th.sort_values('sensi + spezi')

ts_SVM = list(th['threshold'][th['sensi + spezi'] == max(th['sensi + spezi'])])[0]

if min(wkeiten_pos_SVM_train) > max(wkeiten_neg_SVM_train):
    random.seed(1802)
    ts_SVM = random.uniform(max(wkeiten_neg_SVM_train), min(wkeiten_pos_SVM_train))


y_pred_SVM_train_new = getPred(mydf_SVM_train['Wkeit pos'], ts_SVM)
[cm_SVM_train_new, sensi_SVM_train_new, spezi_SVM_train_new,
 richtigkl_SVM_train_new] = getWerteKlassi(y_train_10, y_pred_SVM_train_new)

cm_SVM_train_new
# array([[3792,    0],
#        [ 660, 6924]])

sensi_SVM_train_new # 1.0
spezi_SVM_train_new # 0.9129746835443038
sensi_SVM_train_new + spezi_SVM_train_new #1.9129746835443038
richtigkl_SVM_train_new # 0.9419831223628692

# das UNTERE TRAIN MODELL DAS BESSERE

random.seed(1802)
[cm_SVM_test, sensi_SVM_test, spezi_SVM_test, richtigkl_SVM_test, 
 y_pred_SVM_test, mydf_SVM_test] = SVMAll(dtm_train, dtm_test, y_train_10, y_test_10, X_test_ind,
                       threshold_SVM = ts_SVM)
       
cm_SVM_test
# array([[  54,  323],
#        [ 444, 2807]])

sensi_SVM_test # 0.14323607427055704

spezi_SVM_test # 0.8634266379575515

sensi_SVM_test + spezi_SVM_test # 1.0066627122281084

richtigkl_SVM_test # 0.7885887541345094

wkeiten_pos_SVM_test = mydf_SVM_test['Wkeit pos'][mydf_SVM_test['wahre Klasse'] == 1]
wkeiten_neg_SVM_test = mydf_SVM_test['Wkeit pos'][mydf_SVM_test['wahre Klasse'] == 0]
HistFuncK(wkeiten_pos_SVM_test, 'hist_K_SVM_test.png')
HistFuncNK(wkeiten_neg_SVM_test, 'hist_NK_SVM_test.png')

ROCFunc(y_test_10, mydf_SVM_test['Wkeit pos'], 'ROC_SVM_test.png')    
                   
auc_SVM_test = metrics.roc_auc_score(y_test_10, mydf_SVM_test['Wkeit pos'])    

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
# dill.dump_session('5A_04_29_abends.pkl')
# dill.load_session('5A_04_29_abends.pkl')

## groß und klein vgl

cm_SVM_train_new
# array([[3792,    0],
#        [ 660, 6924]])

cm_SVM_train_kl_new
# array([[3504,  288],
#        [ 288, 7296]])

cm_SVM_test
# vorher
# array([[1617,    8],
#        [ 935, 2316]])
# jetzt
# array([[  54,  323],
#        [ 444, 2807]])
sensi_SVM_test + spezi_SVM_test # 1.0066627122281084

cm_SVM_test_kl
# array([[ 349,   28],
#        [ 104, 3147]])
sensi_SVM_test_kl + spezi_SVM_test_kl # 1.8937392860960145


### beste Modelle nach Sensi + Spezi in den Testdaten

cm_WS_test
# array([[ 347,   30],
#        [ 110, 3141]], dtype=int64)

sensi_WS_test + spezi_WS_test # 1.8865886603346695

cm_RF_test_kl
# array([[ 344,   33],
#        [  77, 3174]])

sensi_RF_test_kl + spezi_RF_test_kl # 1.888781823507478
sensi_RF_test_kl # 0.9124668435013262

cm_SVM_test_kl
# array([[ 349,   28],
#        [ 104, 3147]])

sensi_SVM_test_kl + spezi_SVM_test_kl # 1.8937392860960145
sensi_SVM_test_kl # 0.9257294429708223

# Gewinnermodell: SVM klein (aber alles extrem nah beieinander!)
