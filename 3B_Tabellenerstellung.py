# In dieser Datei wird die Tabelle erstellt, in der bspw. Name und Text stehen
# und es wird eine Wortsuchenklassifikation durchgeführt. Diese Klassifikation
# benötigt keine weitere Bearbeitung der Texte und wird daher hier schon 
# durchgeführt. 

# =============================================================================
# Pakete laden & Funktionen einbinden - AKTUALISIEREN WENN DATEINAME NEU
# =============================================================================
import os
import pandas as pd
import dill   
import time
from sklearn.metrics import confusion_matrix

os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
from Funktionen import *


# =============================================================================
# Texte auslesen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
dill.load_session('ohne_duplikate_04_26.pkl')

del(NK_duplikatfrei)


# =============================================================================
# Tabelle erstellen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
tabelleSchreiben(K_duplikatfrei, 'durchlauf_gr_04_26.xlsx')

# =============================================================================
# Tabelle auslesen und Kündigungen rausholen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
k_table = pd.read_excel('durchlauf_gr_04_26.xlsx')


# =============================================================================
# Gründe hinzufügen (aus Tabelle)
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt')
kund = pd.read_excel('Label_final_endgültig.xlsx', sheet_name= "Kündigungen")

# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
# dill.load_session('k_nk_04_26.pkl')

# # PROBLEM: TXT_KNK liest nach wachsendem einlesedatum ein

# # daher: 
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt')
 
# ks = next(os.walk('Kündigung'))[2]

# path = r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Kündigung'
# uhrzeit = []
# for i in range(len(ks)):
#     zeit = os.path.getctime(path + r'\\' + ks[i])
#     uhrzeit.append(time.ctime(zeit))

# zeitendf = pd.DataFrame([ks, uhrzeit, txt_KNK]).T
# zeitendf.columns = ['pdfname', 'zeit', 'texte']

# zeitendf_sorted = zeitendf.sort_values('pdfname')

# txt_KNK = list(zeitendf_sorted['texte'])

kund['Text'] = createTable(seitenZusammen(txt_KNK))[4::5]
K_vergleich = createTable(K_duplikatfrei)[4::5]

# brauche die Indizes der Kuendigungen, die noch dabei sind fuer die kund Tabelle
# suche quasi which(txt_KNK == K_duplikatfrei), aber immer nur eins, also immer nur
# die ersten?
ind = []
for i in range(len(K_vergleich)):
    for j in range(len(kund['Text'])):
        if K_vergleich[i] == kund['Text'][j]:
            ind.append(j)
            break
        
k_true = kund.T[ind].T

ks_uebersicht = pd.DataFrame([list(k_true['Grundgruppe']), list(k_true['Grund 2']), 
                           list(k_true['Grundgruppierung']), list(k_true['Text'])]).T 

ks_uebersicht.columns = ['Grundgruppe', 'Grund 2', 'Grundgruppierung', 'Text']

np.unique(list(ks_uebersicht['Grund 2']))
#  array(['Corona', 'nan', 'privat'], dtype='<U32')
# das sind alles keine Gründe, die ich genauer anschaue, deshalb wird der 
# zweite Grund nicht weiter beachtet

del(ks_uebersicht['Grund 2'], ks_uebersicht['Grundgruppe'])

# prog_gruende = []
# for i in range(len(ks_uebersicht)):
#     try:
#         prog_gruende.append(grundVergleich(ks_uebersicht['Text'][i]))
#     except:
#         prog_gruende.append('nan')
        
# ks_uebersicht['prog. Gründe'] = prog_gruende

# suche die Kündigungen raus, die zu einem meiner Gründe gehören
auswahlgrund = []
for i in range(len(ks_uebersicht)):
    if ks_uebersicht['Grundgruppierung'][i] in ['finanziell', 'Todesfall', 'Berufswechsel (bzw Abgang)',
                            'Rente']:
        auswahlgrund.append(i)
        
ks_uebersicht['Grundgruppierung'] = ks_uebersicht['Grundgruppierung'].replace('Sonstiges', 'S')
ks_uebersicht['Grundgruppierung'] = ks_uebersicht['Grundgruppierung'].replace('finanziell', 'F')
ks_uebersicht['Grundgruppierung'] = ks_uebersicht['Grundgruppierung'].replace('Rente', 'R')
ks_uebersicht['Grundgruppierung'] = ks_uebersicht['Grundgruppierung'].replace('Todesfall', 'T')
ks_uebersicht['Grundgruppierung'] = ks_uebersicht['Grundgruppierung'].replace('Berufswechsel (bzw Abgang)', 'B')

for i in range(len(ks_uebersicht)):
    # if ks_uebersicht['prog. Gründe'][i] not in ['S', 'F', 'R', 'T', 'B']:
    #     ks_uebersicht['prog. Gründe'][i] = 'G'
    if ks_uebersicht['Grundgruppierung'][i] not in ['S', 'F', 'R', 'T', 'B']:
        ks_uebersicht['Grundgruppierung'][i] = 'K'
        

        
# # Übergangslösung, will nur sehen ob confusionmatrix klappen würde
# # würde es - aber sieht halt kacke aus
# confmat_uebersicht = ks_uebersicht
# for i in range(len(confmat_uebersicht)):
#     if ks_uebersicht['prog. Gründe'][i] in ['S', 'G']:
#         ks_uebersicht['prog. Gründe'][i] = 'K'
#     if ks_uebersicht['Grundgruppierung'][i] in ['S', 'K']:
#         ks_uebersicht['Grundgruppierung'][i] = 'K'
        

# confusion_matrix(list(confmat_uebersicht['Grundgruppierung']), list(confmat_uebersicht['prog. Gründe']))       
# len(myVektorGleich(confmat_uebersicht['Grundgruppierung'], confmat_uebersicht['prog. Gründe']))
# # 1180
# # 87% richtig (dabei keine Unterscheidung zwischen kein Grund und Sonstige Gründe)

# # noch naiverer schätzer, alles auf sonstiges
# len(myVektorGleich(confmat_uebersicht['Grundgruppierung'], ['K'] * 1357))
# #1299
# # nur 58 haben einen Grund, das ist natürlich sehr problematisch für meine
# # Analyse

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
dill.dump_session('3B_04_26.pkl')