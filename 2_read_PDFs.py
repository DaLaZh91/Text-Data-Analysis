# In this file, PDFs are read and the texts are sorted and cleaned
# so that each text appears only once and is classified into
# cancellation and non-cancellation.

# =============================================================================
# Load packages & import functions - UPDATE IF FILE NAME CHANGES
# =============================================================================
import os
import pandas as pd
import dill   
import time

os.chdir(r'W:\your_folder\Python')
from functions import *

# =============================================================================
# Read in & extract text
# =============================================================================

# # 1. Cancellations

[txt_C, duration_C] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Cancellations')
os.chdir(r'W:\your_folder\data - unprotected\texts')   
dill.dump_session('c_03_09.pkl')

# # 2. Other business processes

# # Contribution break
[txt_CB, duration_CB] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Contribution_break')
os.chdir(r'W:\your_folder\data - unprotected\texts')    
dill.dump_session('cb_03_01.pkl')

# Premium waiver
[txt_PW, duration_PW] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Premium_waiver')
os.chdir(r'W:\your_folder\data - unprotected\texts')   
dill.dump_session('pw_03_01.pkl')

# # Increase
[txt_I, duration_I] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Increase')
os.chdir(r'W:\your_folder\data - unprotected\texts')   
dill.dump_session('i_03_01.pkl')

# # Policyholder change
[txt_PC1, duration_PC1] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Policyholder_change_part_1')
os.chdir(r'W:\your_folder\data - unprotected\texts')    
dill.dump_session('pc1_03_01.pkl')

# [txt_PC2, duration_PC2] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Policyholder_change_part_2')
os.chdir(r'W:\your_folder\data - unprotected\texts')   
dill.dump_session('pc2_03_01.pkl')

# [txt_PC3, duration_PC3] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Policyholder_change_part_3')
os.chdir(r'W:\your_folder\data - unprotected\texts')    
dill.dump_session('pc3_03_01.pkl')

# [txt_PC4, duration_PC4] = ordnerEinlesen(r'W:\your_folder\data - unprotected\Policyholder_change_part_4')
os.chdir(r'W:\your_folder\data - unprotected\texts')     
dill.dump_session('pc4_03_01.pkl')

# =============================================================================
# Save everything into one workspace
# =============================================================================

os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.load_session('c_03_09.pkl')
dill.load_session('cb_03_01.pkl')
dill.load_session('pw_03_01.pkl')
dill.load_session('i_03_01.pkl')
dill.load_session('pc1_03_01.pkl')
dill.load_session('pc2_03_01.pkl')
dill.load_session('pc3_03_01.pkl')
dill.load_session('pc4_03_01.pkl')

del(duration_CB, duration_PW, duration_I, duration_PC1, duration_PC2, duration_PC3, duration_PC4, duration_C)
txt_PC = txt_PC1 + txt_PC2 + txt_PC3 + txt_PC4
del(txt_PC1, txt_PC2, txt_PC3, txt_PC4)

dill.dump_session('texts_03_09.pkl')
dill.load_session('texts_03_09.pkl')

# =============================================================================
# Sort cancellations
# =============================================================================
os.chdir(r'W:\your_folder\data - unprotected')
C_tab = pd.read_excel('Label_final.xlsx', sheet_name='Cancellations')

# PROBLEM: TXT_C is read in ascending order of import time

# therefore:
os.chdir(r'W:\your_folder\data - unprotected')
txt_C_old = txt_C
Cs = next(os.walk('Cancellations'))[2]

path = r'W:\your_folder\data - unprotected\Cancellations'
timestamps = []
for i in range(len(Cs)):
    t = os.path.getctime(path + r'\\' + Cs[i])
    timestamps.append(time.ctime(t))

time_df = pd.DataFrame([Cs, timestamps, txt_C]).T
time_df.columns = ['pdfname', 'time', 'texts']

time_df_sorted = time_df.sort_values('pdfname')

txt_C = list(time_df_sorted['texts'])

# done individually because sometimes there are handwritten additions
# (e.g., checkmarks), which we want to keep
# additional 6: probably because no contract number was given

C_tab['texts'] = txt_C

# selection = 2 means it is a cancellation to be reviewed
Cs = myGleich(C_tab['selection'], 2)
duplicates = myGleich(C_tab['dubble_ind (0 = is a duplicate)'], 0)

txt_C = list(C_tab['Texts'][C_tab['selection'] == 2])
txt_NC = list(C_tab['Texts'][C_tab['selection 2'] == 1])

all_NC = txt_CB + txt_PW + txt_I + txt_PC + txt_NC
all_texts = all_NC + txt_C

del(duplicates, i, C_tab, Cs, txt_CB, txt_PW, txt_I, txt_PC)

os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.dump_session('c_nc_04_26.pkl')

new_all = seitenZusammen(all_texts)
NC_combined = seitenZusammen(all_NC)
C_combined = seitenZusammen(txt_C)

# =============================================================================
# Remove duplicates
# =============================================================================

NC_unique = pd.unique(NC_combined)

NC_no_duplicates = []
for i in range(len(NC_unique)):
    NC_no_duplicates.append(NC_unique[i])
    
C_unique = pd.unique(C_combined)

C_no_duplicates = []
for i in range(len(C_unique)):
    C_no_duplicates.append(C_unique[i])
    
del(all_NC, all_texts, i, C_unique, C_combined, NC_unique, NC_combined)

os.chdir(r'W:\your_folder\data - unprotected\texts')
dill.dump_session('without_duplicates_04_26.pkl')
