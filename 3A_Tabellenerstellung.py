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
from sklearn.metrics import confusion_matrix

os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
from Funktionen import *

# =============================================================================
# Texte auslesen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
dill.load_session('ohne_duplikate_04_26.pkl')

alle_texte = NK_duplikatfrei + K_duplikatfrei
    
# =============================================================================
# Tabelle erstellen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
tabelleSchreiben(alle_texte, 'durchlauf_04_26_NEW.xlsx')

# =============================================================================
# Tabelle auslesen
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
table = pd.read_excel('durchlauf_04_26_NEW.xlsx')

table['wahrer GeVo'] =  ['N'] * len(NK_duplikatfrei) + ['K'] * len(K_duplikatfrei)

confusion_matrix(table['wahrer GeVo'], table['GeVos'])

# array([[ 1085,   272],
#        [ 1478, 11150]], dtype=int64)

table.to_excel("table_04_26_NEW.xlsx")

