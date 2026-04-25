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
from functions import *

# =============================================================================
# Load texts
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('without_duplicates_04_26.pkl')
os.chdir(r'W:\your_folder\Output')
dill.load_session('3B_04_26.pkl')

all_C = list(cs_overview['text'])

for i in range(len(all_C)):
    if pd.isna(all_C[i]):
        all_C[i] = 'nan'

labels = cs_overview['reasongrouping']

# delete everything except all_C, confmat_overview, and labels
for var in ['descreason', 'i', 'ind', 'j', 'C_duplicatefree',
            'c_table', 'c_true', 'C_vergleich', 'NC_duplicatefree']:
    if var in globals():
        del globals()[var]

# =============================================================================
# 1. Stemming etc.
# =============================================================================

### 1. Stemming ===============================================================

my_stem = getStem(all_C)

### 2. Remove stopwords =======================================================

stop_words = get_stop_words('de')
stop_words_stem = getStem(stop_words)

texts_sl = []
for i in range(len(my_stem)):
    try:
        texts_sl.append(delWordVector(my_stem[i], stop_words_stem))
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
dill.dump_session('reason_slnlstm_04_27.pkl')

# =============================================================================
# 2. Split into training and test data   
# =============================================================================

random.seed(1104)
[X_train_ind, X_test_ind, y_train, y_test] = train_test_split(
    range(0, len(all_C)),
    labels,
    test_size=0.3,
    random_state=1,
    stratify=labels
)

y_train = list(y_train)
y_test = list(y_test)

X_test = str(list(np.array(texts_nl)[X_test_ind]))
X_test = X_test[2:(len(X_test) - 2)].split(sep="', '")

X_train = str(list(np.array(texts_nl)[X_train_ind]))
X_train = X_train[2:(len(X_train) - 2)].split(sep="', '")

X_train_O = str(list(np.array(X_train)[myEqual(y_train, 'O')]))
X_train_O = X_train_O[2:(len(X_train_O) - 2)].split(sep="', '")

X_train_F = str(list(np.array(X_train)[myEqual(y_train, 'F')]))
X_train_F = X_train_F[2:(len(X_train_F) - 2)].split(sep="', '")

X_train_D = str(list(np.array(X_train)[myEqual(y_train, 'D')]))
X_train_D = X_train_D[2:(len(X_train_D) - 2)].split(sep="', '")

X_train_R = str(list(np.array(X_train)[myEqual(y_train, 'R')]))
X_train_R = X_train_R[2:(len(X_train_R) - 2)].split(sep="', '")

X_train_J = str(list(np.array(X_train)[myEqual(y_train, 'J')]))
X_train_J = X_train_J[2:(len(X_train_J) - 2)].split(sep="', '")

# =============================================================================
# 3. Remove infrequent words (for memory reasons)
# =============================================================================

vect = CountVectorizer() 

[keep_token, df_single_1] = delRareWords2(
    vect, X_train_O, X_train_F, X_train_D, X_train_R, X_train_J, 0.01, y_train
)

### Keep only selected words ==================================================

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

X_train_O = str(list(np.array(X_train)[myEqual(y_train, 'O')]))
X_train_O = X_train_O[2:(len(X_train_O) - 2)].split(sep="', '")

X_train_F = str(list(np.array(X_train)[myEqual(y_train, 'F')]))
X_train_F = X_train_F[2:(len(X_train_F) - 2)].split(sep="', '")

X_train_D = str(list(np.array(X_train)[myEqual(y_train, 'D')]))
X_train_D = X_train_D[2:(len(X_train_D) - 2)].split(sep="', '")

X_train_R = str(list(np.array(X_train)[myEqual(y_train, 'R')]))
X_train_R = X_train_R[2:(len(X_train_R) - 2)].split(sep="', '")

X_train_J = str(list(np.array(X_train)[myEqual(y_train, 'J')]))
X_train_J = X_train_J[2:(len(X_train_J) - 2)].split(sep="', '")

# =============================================================================
# 4. Also remove infrequent words for bi- and trigrams
# =============================================================================

vect = CountVectorizer() 
[keep_single, df_single] = delRareWords2(
    vect, X_train_O, X_train_F, X_train_D, X_train_R, X_train_J, 0.005, y_train
)

vect2 = CountVectorizer(ngram_range=(2, 2)) 
[keep_bigram, df_bigram] = delRareWords2(
    vect2, X_train_O, X_train_F, X_train_D, X_train_R, X_train_J, 0.005, y_train
)

vect.fit_transform(X_train)
token = vect.get_feature_names_out() 
dtm_mono = getDTM(vect, X_train, keep_single)

vect2.fit_transform(X_train)
token2 = vect2.get_feature_names_out() 
dtm_bi = getDTM(vect2, X_train, keep_bigram)

dtm = pd.concat([dtm_mono, dtm_bi])
df = pd.concat([df_single, df_bigram])

dtm_tr = dtm

# =============================================================================
# 5. Apply Gini index and keep most important tokens (training data)
# =============================================================================

O_len = len(myEqual(y_train, 'O'))
F_len = len(myEqual(y_train, 'F'))
D_len = len(myEqual(y_train, 'D'))
R_len = len(myEqual(y_train, 'R'))
J_len = len(myEqual(y_train, 'J'))

class_distribution = [O_len, F_len, D_len, R_len, J_len]

ginis = getGini2(df, class_distribution)

plt.plot(np.sort(ginis))

ginis_mono = getGini2(df_single, class_distribution)
ginis_bi = getGini2(df_bigram, class_distribution)

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
token1_t = vect1_t.get_feature_names_out() 
dtm_test_mono = getDTM(vect1_t, X_test, set(token1_t).intersection(keeptok))

vect2_t = CountVectorizer(ngram_range=(2, 2)) 
vect2_t.fit_transform(X_test)
token2_t = vect2_t.get_feature_names_out() 
dtm_test_bi = getDTM(vect2_t, X_test, set(token2_t).intersection(keeptok))

# vect3_t = CountVectorizer(ngram_range=(3, 3)) 
# vect3_t.fit_transform(X_test)
# token3_t = vect3_t.get_feature_names_out() 
# dtm_test_tri = getDTM(vect3_t, X_test, set(token3_t).intersection(keeptok))

# dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi, dtm_test_tri])
dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi])

test_token = list(dtm_test_start.T.columns)

train_only_token = list(set(keeptok).difference(test_token))
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
# 8. Create DTM for the test data using all training tokens
# =============================================================================

vect1_t = CountVectorizer() 
vect1_t.fit_transform(X_test)
token1_t = vect1_t.get_feature_names_out() 
dtm_test_mono = getDTM(
    vect1_t,
    X_test,
    set(token1_t).intersection(list(dtm_tr.T.columns))
)

vect2_t = CountVectorizer(ngram_range=(2, 2)) 
vect2_t.fit_transform(X_test)
token2_t = vect2_t.get_feature_names_out() 
dtm_test_bi = getDTM(
    vect2_t,
    X_test,
    set(token2_t).intersection(list(dtm_tr.T.columns))
)

# vect3_t = CountVectorizer(ngram_range=(3, 3)) 
# vect3_t.fit_transform(X_test)
# token3_t = vect3_t.get_feature_names_out() 
# dtm_test_tri = getDTM(vect3_t, X_test, set(token3_t).intersection(keeptok))

# dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi, dtm_test_tri])
dtm_test_start = pd.concat([dtm_test_mono, dtm_test_bi])

test_token = list(dtm_test_start.T.columns)

train_only_token = list(set(list(dtm_tr.T.columns)).difference(test_token))
lg = len(train_only_token)

if lg == 0:
    dtm_te = dtm_test_start
else:
    len_test = len(dtm_test_start.columns)
    
    test_0 = pd.DataFrame(np.zeros([lg, len_test]))
    dtm_add = setColRowNames(
        test_0,
        ['Doc ' + str(i) for i in range(1, len_test + 1)],
        train_only_token
    )

    dtm_te = pd.concat([dtm_test_start, dtm_add])


# =============================================================================
# 9. tf-idf transformation
# =============================================================================

tfidf_train = getTFIDF(dtm_train)
tfidf_test = getTFIDF(dtm_test)

# =============================================================================
# Matrices
# =============================================================================

tfidf_train
tfidf_test
dtm_train
dtm_test

os.chdir(r'W:\your_folder\Output')
dill.dump_session('4B_04_28.pkl')
