# In this file, a table is created (e.g., containing name and text),
# and a keyword-based classification is performed. This classification
# does not require further text processing and is therefore done here.

# =============================================================================
# Load packages & import functions - UPDATE IF FILE NAME CHANGES
# =============================================================================
import os
import pandas as pd
import numpy as np
import dill   
import time
from sklearn.metrics import confusion_matrix

os.chdir(r'W:\your_folder\Python')
from Funktionen import *


# =============================================================================
# Load texts
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('without_duplicates_04_26.pkl')

del(NC_no_duplicates)


# =============================================================================
# Create table
# =============================================================================
os.chdir(r'W:\your_folder\Output')
writeTable(C_no_duplicates, 'run_gr_04_26.xlsx')

# =============================================================================
# Read table and extract cancellations
# =============================================================================
os.chdir(r'W:\your_folder\Output')
c_table = pd.read_excel('run_gr_04_26.xlsx')


# =============================================================================
# Add reasons (from table)
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected')
cust = pd.read_excel('Label_final.xlsx', sheet_name="Cancellations")

os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('c_nc_04_26.pkl')

# PROBLEM: TXT_CNC is read in ascending order of import time

# therefore: 
os.chdir(r'W:\your_folder\data - unprotected')
 
Cs = next(os.walk('Cancellations'))[2]

path = r'W:\your_folder\data - unprotected\Cancellations'
timestamps = []
for i in range(len(Cs)):
    t = os.path.getctime(path + r'\\' + Cs[i])
    timestamps.append(time.ctime(t))

time_df = pd.DataFrame([Cs, timestamps, txt_CNC]).T
time_df.columns = ['pdfname', 'time', 'texts']

time_df_sorted = time_df.sort_values('pdfname')

txt_CNC = list(time_df_sorted['texts'])

cust['text'] = createTable(seitenZusammen(txt_CNC))[4::5]
C_compare = createTable(C_no_duplicates)[4::5]

# need the indices of the cancellations that are still present for the cust table
# essentially searching for which(txt_CNC == C_no_duplicates), but only once each time,
# i.e., always the first match

mapping = {text: idx for idx, text in enumerate(cust['text'])}

ind = []
for text in C_compare:
    if text in mapping:
        ind.append(mapping[text])

c_true = cust.T[ind].T

cs_overview = pd.DataFrame([
    list(c_true['reasongroup']),
    list(c_true['reason 2']), 
    list(c_true['reasongrouping']),
    list(c_true['text'])
]).T 

cs_overview.columns = ['main_group', 'sub_group', 'grouping', 'text']

np.unique(list(cs_overview['sub_group']))
array(['Corona', 'nan', 'privat'], dtype='<U32')
# these are all reasons I do not analyze in detail,
# therefore the second reason is not considered further

del(cs_overview['sub_group'], cs_overview['main_group'])

prog_reasons = []
for i in range(len(cs_overview)):
    try:
        prog_reasons.append(reasonComparison(cs_overview['text'][i]))
    except:
        prog_reasons.append('nan')
        
cs_overview['prog_reasons'] = prog_reasons

# select cancellations that belong to one of my target reasons
selected_reason = []
for i in range(len(cs_overview)):
    if cs_overview['grouping'][i] in [
        'financial', 'death', 'job change',
        'retirement'
    ]:
        selected_reason.append(i)

mapping = {
    'others': 'O',
    'financial': 'F',
    'retirement': 'R',
    'death': 'D',
    'job change': 'J'
}

cs_overview['grouping'] = cs_overview['grouping'].replace(mapping)


for i in range(len(cs_overview)):
    if cs_overview['prog_reasons'][i] not in ['O', 'F', 'R', 'D', 'J']:
        cs_overview.loc[i, 'prog_reasons'] = 'C'
    if cs_overview['grouping'][i] not in ['O', 'F', 'R', 'D', 'J']:
        cs_overview.loc[i, 'grouping'] = 'C'
        

        
# # temporary solution, just to check if confusion matrix would work
# confmat_overview = cs_overview
# for i in range(len(confmat_overview)):
#     if cs_overview['prog_reasons'][i] in ['O', 'C']:
#         cs_overview.loc[i, 'prog_reasons'] = 'C'
#     if cs_overview['grouping'][i] in ['O', 'C']:
#         cs_overview.loc[i, 'grouping'] = 'C'
        

# confusion_matrix(list(confmat_overview['grouping']), list(confmat_overview['prog_reasons']))       
# len(myVectorEqual(confmat_overview['grouping'], confmat_overview['prog_reasons']))
# # 1180
# # 87% correct (no distinction between no reason and other reasons)

# # even more naive estimator: everything as "other"
# len(myVectorEqual(confmat_overview['grouping'], ['C'] * 1357))
# #1299
# # only 58 have a reason, which is problematic for the analysis

os.chdir(r'W:\your_folder\Output')
dill.dump_session('3B_04_26.pkl')
