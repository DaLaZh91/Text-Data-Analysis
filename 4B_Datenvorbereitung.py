# =============================================================================
# Pakete laden & Funktionen einbinden 
# =============================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dill  
from stop_words import get_stop_words #fuer die deutschen stoppwords
from sklearn.feature_extraction.text import CountVectorizer # für die DTM Matrix
import random
from sklearn.model_selection import train_test_split
from datetime import datetime

os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
from Funktionen import *

# =============================================================================
# Texte auslesen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
dill.load_session('ohne_duplikate_04_26.pkl')
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
dill.load_session('3B_04_26.pkl')

alle_K = list(ks_uebersicht['Text'])

for i in range(len(alle_K)):
    if pd.isna(alle_K[i]):
        alle_K[i] = 'nan'

grund = ks_uebersicht['Grundgruppierung']

# loesche alles außer alle_K, confmat_uebersicht und grund
del(auswahlgrund, i, ind, j, K_duplikatfrei, k_table, k_true,
    K_vergleich, NK_duplikatfrei)
# =============================================================================
# 1. Stemming und Co
# =============================================================================

# aktuell auskommentiert, weil es so lange dauert!

### 1. Stemming ===============================================================

my_stem = getStem(alle_K)

### 2. Stoppwörter raus =======================================================

stop_words = get_stop_words('de')

stop_words_stem = getStem(stop_words)

texte_sl = []
for i in range(len(my_stem)):
    try:
        texte_sl.append(delWortvektor(my_stem[i], stop_words_stem))
    except:
        texte_sl.append('nan')
  
texte_sl_einzeln = ' '.join(texte_sl).split()

  
### 3. Zahlen raus ============================================================

texte_nl = []
for i in range(len(texte_sl)):
    try:
        texte_nl.append(delNumbers(texte_sl[i]))
    except:
        texte_nl.append('nan')
        
texte_nl_einzeln = ' '.join(texte_nl).split()




os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
dill.dump_session('grund_slnlstm_04_27.pkl')



# =============================================================================
# 2. Einteilung in Test- und Trainingsdaten   
# =============================================================================

random.seed(1104)
[X_train_ind, X_test_ind, y_train, y_test] = train_test_split(range(0, len(alle_K)), grund, test_size=0.3, random_state=1, stratify = grund)

y_train = list(y_train)
y_test = list(y_test)

X_test = str(list(np.array(texte_nl)[X_test_ind]))
X_test = X_test[2:(len(X_test) - 2)].split(sep = "', '")

X_train = str(list(np.array(texte_nl)[X_train_ind]))
X_train = X_train[2:(len(X_train) - 2)].split(sep = "', '")

X_train_B = str(list(np.array(X_train)[myGleich(y_train, 'B')]))
X_train_B = X_train_B[2:(len(X_train_B) - 2)].split(sep = "', '")

X_train_F = str(list(np.array(X_train)[myGleich(y_train, 'F')]))
X_train_F = X_train_F[2:(len(X_train_F) - 2)].split(sep = "', '")

X_train_K = str(list(np.array(X_train)[myGleich(y_train, 'K')]))
X_train_K = X_train_K[2:(len(X_train_K) - 2)].split(sep = "', '")

X_train_R = str(list(np.array(X_train)[myGleich(y_train, 'R')]))
X_train_R = X_train_R[2:(len(X_train_R) - 2)].split(sep = "', '")

X_train_T = str(list(np.array(X_train)[myGleich(y_train, 'T')]))
X_train_T = X_train_T[2:(len(X_train_T) - 2)].split(sep = "', '")

# =============================================================================
# 3. Lösche wenig vorkommende Wörter (aus Speichergründen hier)
# =============================================================================

vect = CountVectorizer() 

[keep_token, df_einzeln_1] = delSelteneWoerter2(vect, X_train_B, X_train_F, X_train_K, X_train_R, X_train_T, 0.01, y_train)
    

### Texte mit nur diesen Wörtern ==============================================

texte_final = [0] * len(texte_nl)
for j in range(len(texte_nl)):
    mainsplit = texte_nl[j].split()
    new = []
    for i in range(len(mainsplit)):
        if mainsplit[i] in keep_token:
            new.append(mainsplit[i])
    texte_final[j] = new
    

for i in range(len(texte_final)):
    texte_final[i] = ' '.join(texte_final[i])

texte_final_einzeln = ' '.join(texte_final).split()

### Lösche selten vorkommende Wörter ==========================================
# die daraus resultierend "kleingemachten" Test- und Trainingsdaten

X_test = str(list(np.array(texte_final)[X_test_ind]))
X_test = X_test[2:(len(X_test) - 2)].split(sep = "', '")

X_train = str(list(np.array(texte_final)[X_train_ind]))
X_train = X_train[2:(len(X_train) - 2)].split(sep = "', '")

X_train_B = str(list(np.array(X_train)[myGleich(y_train, 'B')]))
X_train_B = X_train_B[2:(len(X_train_B) - 2)].split(sep = "', '")

X_train_F = str(list(np.array(X_train)[myGleich(y_train, 'F')]))
X_train_F = X_train_F[2:(len(X_train_F) - 2)].split(sep = "', '")

X_train_K = str(list(np.array(X_train)[myGleich(y_train, 'K')]))
X_train_K = X_train_K[2:(len(X_train_K) - 2)].split(sep = "', '")

X_train_R = str(list(np.array(X_train)[myGleich(y_train, 'R')]))
X_train_R = X_train_R[2:(len(X_train_R) - 2)].split(sep = "', '")

X_train_T = str(list(np.array(X_train)[myGleich(y_train, 'T')]))
X_train_T = X_train_T[2:(len(X_train_T) - 2)].split(sep = "', '")


# =============================================================================
# 4. bei Bi- und Trigrammen ebenfalls die selten vorkommenden löschen
# =============================================================================
vect = CountVectorizer() 
[keep_einzeln, df_einzeln] = delSelteneWoerter2(vect, X_train_B, X_train_F, X_train_K, X_train_R, X_train_T, 0.005, y_train)
    
vect2 = CountVectorizer(ngram_range = (2, 2)) 
[keep_bigram, df_bigram] = delSelteneWoerter2(vect2, X_train_B, X_train_F, X_train_K, X_train_R, X_train_T, 0.005, y_train)
    
# vect3 = CountVectorizer(ngram_range = (3, 3)) 
# [keep_trigram, df_trigram] = delSelteneWoerter2(vect3, X_train_B, X_train_F, X_train_K, X_train_R, X_train_T, 0.005, y_train)
    
vect.fit_transform(X_train)
token = vect.get_feature_names() 
dtm_mono = getDTM(vect, X_train, keep_einzeln)

vect2.fit_transform(X_train)
token2 = vect2.get_feature_names() 
dtm_bi = getDTM(vect2, X_train, keep_bigram)

# vect3.fit_transform(X_train)
# token3 = vect3.get_feature_names() 
# dtm_tri = getDTM(vect3, X_train, keep_trigram)

# dtm = pd.concat([dtm_mono, dtm_bi, dtm_tri])
# df = pd.concat([df_einzeln, df_bigram, df_trigram])

dtm = pd.concat([dtm_mono, dtm_bi])
df = pd.concat([df_einzeln, df_bigram])

dtm_tr = dtm

# =============================================================================
# 5. darauf Gini Index und nur die x wichtigsten behalten (Trainingsdaten)
# =============================================================================

### Gini-Indizes berechnen ====================================================
B_len = len(myGleich(y_train, 'B'))
F_len = len(myGleich(y_train, 'F'))
K_len = len(myGleich(y_train, 'K'))
R_len = len(myGleich(y_train, 'R'))
T_len = len(myGleich(y_train, 'T'))
aufteilung = [B_len, F_len, K_len, R_len, T_len]

ginis = getGini2(df, aufteilung)

plt.plot(np.sort(ginis))

ginis_mono = getGini2(df_einzeln, aufteilung)
ginis_bi = getGini2(df_bigram, aufteilung)
# ginis_tri = getGini2(df_trigram, aufteilung)

plt.plot(np.sort(ginis_mono))
plt.plot(np.sort(ginis_bi))
# plt.plot(np.sort(ginis_tri))

### nur die wichtigsten behalten ==============================================

df['ginis'] = ginis
keeptok = df['token'][df['ginis'] == 1]
dtm_new_m = getDTM(vect, X_train, set(keeptok).intersection(keep_einzeln))
dtm_new_b = getDTM(vect2, X_train, set(keeptok).intersection(keep_bigram))
# dtm_new_t = getDTM(vect3, X_train, set(keeptok).intersection(keep_trigram))

# =============================================================================
# 6. Zusammenfügen zu einer dtm Matrix (Trainingsdaten)
# =============================================================================

# dtm_train = pd.concat([dtm_new_m, dtm_new_b, dtm_new_t])
dtm_train = pd.concat([dtm_new_m, dtm_new_b])

# =============================================================================
# 8. Für die Testdaten, dtm Matrix mit den Token erstellen
# =============================================================================

vect1_t = CountVectorizer() 
vect1_t.fit_transform(X_test)
token1_t = vect1_t.get_feature_names() 
dtm_test_mono = getDTM(vect1_t, X_test, set(token1_t).intersection(keeptok))

vect2_t = CountVectorizer(ngram_range = (2, 2)) 
vect2_t.fit_transform(X_test)
token2_t = vect2_t.get_feature_names() 
dtm_test_bi = getDTM(vect2_t, X_test, set(token2_t).intersection(keeptok))

# vect3_t = CountVectorizer(ngram_range = (3, 3)) 
# vect3_t.fit_transform(X_test)
# token3_t = vect3_t.get_feature_names() 
# dtm_test_tri = getDTM(vect3_t, X_test, set(token3_t).intersection(keeptok))

# dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi, dtm_test_tri])
dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi])


test_token = list(dtm_test_start.T.columns)

nur_train_token = list(set(keeptok).difference(test_token))
lg = len(nur_train_token)
if lg == 0:
    dtm_test = dtm_test_start
else:
    len_test = len(dtm_test_start.columns)
    
    test_0 = pd.DataFrame(np.zeros([lg, len_test]))
    dtm_add = setColRowNames(test_0,  ['Doc '+ str(i) for i in range(1, len_test + 1)], nur_train_token)

    dtm_test = pd.concat([dtm_test_start, dtm_add])
    
# =============================================================================
# 8. Für die Testdaten, dtm Matrix mit allen Token erstellen
# =============================================================================

vect1_t = CountVectorizer() 
vect1_t.fit_transform(X_test)
token1_t = vect1_t.get_feature_names() 
dtm_test_mono = getDTM(vect1_t, X_test, set(token1_t).intersection(list(dtm_tr.T.columns)))

vect2_t = CountVectorizer(ngram_range = (2, 2)) 
vect2_t.fit_transform(X_test)
token2_t = vect2_t.get_feature_names() 
dtm_test_bi = getDTM(vect2_t, X_test, set(token2_t).intersection(list(dtm_tr.T.columns)))

# vect3_t = CountVectorizer(ngram_range = (3, 3)) 
# vect3_t.fit_transform(X_test)
# token3_t = vect3_t.get_feature_names() 
# dtm_test_tri = getDTM(vect3_t, X_test, set(token3_t).intersection(keeptok))

# dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi, dtm_test_tri])
dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi])


test_token = list(dtm_test_start.T.columns)

nur_train_token = list(set(list(dtm_tr.T.columns)).difference(test_token))
lg = len(nur_train_token)
if lg == 0:
    dtm_te = dtm_test_start
else:
    len_test = len(dtm_test_start.columns)
    
    test_0 = pd.DataFrame(np.zeros([lg, len_test]))
    dtm_add = setColRowNames(test_0,  ['Doc '+ str(i) for i in range(1, len_test + 1)], nur_train_token)

    dtm_te = pd.concat([dtm_test_start, dtm_add])

# =============================================================================
# 9. tf - idf Tranformation
# =============================================================================
tfidf_train = getTFIDF(dtm_train)
tfidf_test = getTFIDF(dtm_test)

# =============================================================================
# Matrizen
# =============================================================================

tfidf_train
tfidf_test
dtm_train
dtm_test

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
dill.dump_session('4B_04_28.pkl')
