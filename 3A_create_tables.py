# In this file, the table is created (e.g., containing name and text)
# and a keyword-based classification is performed. This classification
# does not require further text processing and is therefore done here.

# =============================================================================
# Load packages & import functions - UPDATE IF FILE NAME CHANGES
# =============================================================================
import os
import pandas as pd
import dill   
from sklearn.metrics import confusion_matrix

os.chdir(r'W:\your_folder\Python')
from Funktionen import *

# =============================================================================
# Load texts
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('without_duplicates_04_26.pkl')

all_texts = NC_no_duplicates + C_no_duplicates
    
# =============================================================================
# Create table
# =============================================================================
os.chdir(r'W:\your_folder\Output')
writeTable(all_texts, 'run_04_26_NEW.xlsx')

# =============================================================================
# Read table
# =============================================================================
os.chdir(r'W:\your_folder\Output')
table = pd.read_excel('run_04_26_NEW.xlsx')

table['true_label'] = ['N'] * len(NC_no_duplicates) + ['C'] * len(C_no_duplicates)

confusion_matrix(table['true_label'], table['predicted_label'])

# array([[ 1085,   272],
#        [ 1478, 11150]], dtype=int64)

table.to_excel("table_04_26_NEW.xlsx")
