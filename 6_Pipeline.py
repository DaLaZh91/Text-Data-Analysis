# =============================================================================
# Load packages & import functions
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
# from sklearn.model_selection import cross_val_score  # ET
from sklearn.metrics import confusion_matrix
# from sklearn.datasets import load_boston
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import Lasso, LassoCV  # Lasso
from sklearn.metrics import mean_squared_error  # Lasso
# https://laurenliz22.github.io/nlp_random_forest_and_neural_network_classifiers

os.chdir(r'W:\your_folder\Python')
from functions import *

# =============================================================================
# Load data & preparation
# =============================================================================
os.chdir(r'W:\your_folder\Output')
dill.load_session('dtms_all_04_28.pkl')


# Convert labels to binary (N = 0, others = 1)
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


# Keyword lists (final; german)
k_words_final = ['auszahl', 'kundig', 'kundigungsbestat', 'kundigungstermin',
                 'ruckkaufswert', 'ruckkaufwert', 'teilkund']

nk_words_final = ['beitragspaus', 'geschutzt dat', 'gesundheitsdat', 'erhoh']


# Random Forest classification
random.seed(1802)
[cm_1_PL, sensi_1_PL, spezi_1_PL, richtigkl_1_PL,
 y_pred_1_PL, mydf_1_PL] = RFAll(
    dtm_train_klein,
    dtm_test_klein,
    y_train_klein,
    y_test_klein,
    X_test_ind,
    threshold_RF=ts_kl_RF
)


# Select predictions classified as class 1
k_preds = myEqual(y_pred_1_PL, 1)

X_test_K_preds = list(np.array(X_test)[k_preds])


# =============================================================================
# Classification into reason categories (with german words)
# =============================================================================

f_words = ['eigentum', 'finanziell', 'immobiliendarlehn',
           'insolvenzverfahr', 'wenig lohn', 'wirtschaft']

j_words = ['alt arbeitgeb', 'aufheb vertrag', 'beschaftigt', 'betrieb', 'mitarbeiterin']

r_words = ['altersrent', 'mehr arbeitsfah', 'regelaltersrent',
           'rent', 'rentenbeginn', 'ruhestand']

d_words = ['gestorb', 'verstorb']


GrList = []
for i in range(len(X_test_C_preds)):
    dok_C = []
    dok = X_test_C_preds[i].split()

    # Create bigrams
    dok_bigram = [
        ' '.join(b)
        for l in [' '.join(dok)]
        for b in zip(l.split(' ')[:-1], l.split(' ')[1:])
    ]

    # Financial reasons
    if any(map(lambda v: v in f_words, dok)):
        dok_C += 'F'
    elif any(map(lambda v: v in f_words, dok_bigram)):
        dok_C += 'F'

    # Job-related reasons
    if any(map(lambda v: v in j_words, dok)):
        dok_C += 'J'
    elif any(map(lambda v: v in j_words, dok_bigram)):
        dok_C += 'J'

    # Retirement reasons
    if any(map(lambda v: v in r_words, dok)):
        dok_C += 'R'
    elif any(map(lambda v: v in r_words, dok_bigram)):
        dok_C += 'R'

    # Death-related reasons
    if any(map(lambda v: v in d_words, dok)):
        dok_C += 'D'
    elif any(map(lambda v: v in d_words, dok_bigram)):
        dok_C += 'D'

    # Default category
    if dok_C == []:
        dok_C = ['C']

    GrList.append(dok_C)
