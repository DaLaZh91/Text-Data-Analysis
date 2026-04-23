# In this file, text mining is performed: stopwords are removed,
# a document-term matrix (DTM) is created, and the most important tokens
# are selected using the Gini index. Additionally, a tf-idf transformation
# is applied to the matrix.
# Here, the texts of all business processes are processed.

# =============================================================================
# Load packages & import functions 
# =============================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dill  
from stop_words import get_stop_words  # for German stopwords
from sklearn.feature_extraction.text import CountVectorizer  # for the DTM matrix
import random
from sklearn.model_selection import train_test_split
from datetime import datetime

os.chdir(r'W:\your_folder\Python')
from Funktionen import *

# =============================================================================
# Load texts
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('ohne_duplikate_04_26.pkl')
os.chdir(r'W:\your_folder\Output')
tab = pd.read_excel("table_04_26_NEW.xlsx")

all_texts = list(tab['text'])

for i in range(len(all_texts)):
    if pd.isna(all_texts[i]):
        all_texts[i] = 'nan'

NC_len = len(NC_no_duplicates)
C_len = len(C_no_duplicates)

del(NC_no_duplicates, C_no_duplicates, tab)

# =============================================================================
# 1. Stemming etc.
# =============================================================================

# currently commented out because it takes too long!

### 1. Stemming ===============================================================

my_stem = getStem(all_texts)

os.chdir(r'W:\your_folder\Output')
dill.dump_session('stem_04_27.pkl')

### 2. Remove stopwords =======================================================

stop_words = get_stop_words('de')

stop_words_stem = getStem(stop_words)

texts_sl = []
for i in range(len(my_stem)):
    try:
        texts_sl.append(delWortvektor(my_stem[i], stop_words_stem))
    except:
        texts_sl.append('nan')
  
texts_sl_single = ' '.join(texts_sl).split()

### 3. Remove numbers =========================================================

texts_nl = []
for i in range(len(texts_sl)):
    try:
        texts_nl.append(delNumbers(texts_sl[i]))
    except:
        texts_nl.append('nan')
        
texts_nl_single = ' '.join(texts_nl).split()

os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.dump_session('slnlstm_04_27.pkl')

del(i, stop_words, stop_words_stem, texts_sl, texts_sl_single, texts_nl_single)

# =============================================================================
# 2. Split into training and test data   
# =============================================================================

gevos = ['N'] * NC_len + ['C'] * C_len
random.seed(1104)
[X_train_ind, X_test_ind, y_train, y_test] = train_test_split(
    range(0, (NC_len + C_len)), gevos, test_size=0.3,
    random_state=1, stratify=gevos
)
del(gevos)

X_test = str(list(np.array(texts_nl)[X_test_ind]))
X_test = X_test[2:(len(X_test) - 2)].split(sep="', '")

X_train = str(list(np.array(texts_nl)[X_train_ind]))
X_train = X_train[2:(len(X_train) - 2)].split(sep="', '")

X_train_C = str(list(np.array(X_train)[myGleich(y_train, 'C')]))
X_train_C = X_train_C[2:(len(X_train_C) - 2)].split(sep="', '")

X_train_NC = str(list(np.array(X_train)[myGleich(y_train, 'N')]))
X_train_NC = X_train_NC[2:(len(X_train_NC) - 2)].split(sep="', '")

# =============================================================================
# 3. Remove rare words (for memory reasons)
# =============================================================================
C_len_train = len(myGleich(y_train, 'C'))
NC_len_train = len(myGleich(y_train, 'N'))

vect = CountVectorizer() 
[keep_token, df_single_1] = delSelteneWoerter(
    vect, X_train_C, X_train_NC, 0.01, C_len_train, NC_len_train
)

### Keep only these words =====================================================

texts_final = [0] * len(texts_nl)
for j in range(len(texts_nl)):
    mainsplit = texts_nl[j].split()
    new = []
    for i in range(len(mainsplit)):
        if mainsplit[i] in keep_token:
            new.append(mainsplit[i])
    texts_final[j] = new

for i in range(len(texts_final)):
    texts_final[i] = ' '.join(texts_final[i])

texts_final_single = ' '.join(texts_final).split()

### Apply reduction to train/test =============================================

X_test = str(list(np.array(texts_final)[X_test_ind]))
X_test = X_test[2:(len(X_test) - 2)].split(sep="', '")

X_train = str(list(np.array(texts_final)[X_train_ind]))
X_train = X_train[2:(len(X_train) - 2)].split(sep="', '")

X_train_C = str(list(np.array(X_train)[myGleich(y_train, 'C')]))
X_train_C = X_train_C[2:(len(X_train_C) - 2)].split(sep="', '")

X_train_NC = str(list(np.array(X_train)[myGleich(y_train, 'N')]))
X_train_NC = X_train_NC[2:(len(X_train_NC) - 2)].split(sep="', '")

# =============================================================================
# 4. Also remove rare words for bi- and trigrams
# =============================================================================
vect = CountVectorizer() 
[keep_single, df_single] = delSelteneWoerter(
    vect, X_train_C, X_train_NC, 0.005, C_len_train, NC_len_train
)

vect2 = CountVectorizer(ngram_range=(2, 2)) 
[keep_bigram, df_bigram] = delSelteneWoerter(
    vect2, X_train_C, X_train_NC, 0.01, C_len_train, NC_len_train
)

vect.fit_transform(X_train)
token = vect.get_feature_names() 
dtm_mono = getDTM(vect, X_train, keep_single)

vect2.fit_transform(X_train)
token2 = vect2.get_feature_names() 
dtm_bi = getDTM(vect2, X_train, keep_bigram)

dtm = pd.concat([dtm_mono, dtm_bi])
df = pd.concat([df_single, df_bigram])

dtm_tr = dtm

# =============================================================================
# Plots
# =============================================================================

def getDiff(df):
    diff = df['count NC'] - df['count C']
    share_C = df['count C'] / C_len_train
    share_NC = df['count NC'] / NC_len_train
    diff_shares = share_C - share_NC
    return diff_shares

diff_shares_mono = getDiff(df_single)
plt.plot(np.sort(diff_shares_mono))

diff_shares_bi = getDiff(df_bigram)
plt.plot(np.sort(diff_shares_bi))

# =============================================================================
# 5. Apply Gini index and keep most important tokens
# =============================================================================

ginis = getGini(df, [C_len_train, NC_len_train])

plt.plot(np.sort(ginis))

ginis_mono = getGini(df_single, [C_len_train, NC_len_train])
ginis_bi = getGini(df_bigram, [C_len_train, NC_len_train])

plt.plot(np.sort(ginis_mono))
plt.plot(np.sort(ginis_bi))

### Keep only most important tokens ===========================================

df['ginis'] = ginis
keeptok = df['token'][df['ginis'] == 1]

dtm_new_m = getDTM(vect, X_train, set(keeptok).intersection(keep_single))
dtm_new_b = getDTM(vect2, X_train, set(keeptok).intersection(keep_bigram))

# =============================================================================
# 6. Combine into final DTM (training data)
# =============================================================================

dtm_train = pd.concat([dtm_new_m, dtm_new_b])

# =============================================================================
# 7. Create DTM for test data (selected tokens)
# =============================================================================

vect1_t = CountVectorizer() 
vect1_t.fit_transform(X_test)
token1_t = vect1_t.get_feature_names() 
dtm_test = getDTM(vect1_t, X_test, set(token1_t).intersection(keeptok))

vect2_t = CountVectorizer(ngram_range=(2, 2)) 
vect2_t.fit_transform(X_test)
token2_t = vect2_t.get_feature_names() 
dtm_test_bi = getDTM(vect2_t, X_test, set(token2_t).intersection(keeptok))

dtm_test_start = pd.concat([dtm_test, dtm_test_bi])

test_token = list(dtm_test_start.T.columns)

train_only_token = set(keeptok).difference(test_token)
lg = len(train_only_token)

if lg == 0:
    dtm_test = dtm_test_start
else:
    len_test = len(dtm_test_start.columns)
    test_0 = pd.DataFrame(np.zeros([lg, len_test]))
    dtm_add = setColRowNames(
        test_0,
        ['Doc ' + str(i) for i in range(1, len_test + 1)],
        train_only_token
    )
    dtm_test = pd.concat([dtm_test_start, dtm_add])

# =============================================================================
# 8. tf-idf transformation
# =============================================================================
tfidf_train = getTFIDF(dtm_train)
tfidf_test = getTFIDF(dtm_test)

# =============================================================================
# Output matrices
# =============================================================================

tfidf_train
tfidf_test
dtm_train
dtm_test

os.chdir(r'W:\your_folder\Output')
dill.dump_session('dtms_all_04_28.pkl')
