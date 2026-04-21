# In dieser Datei werden zum einen die PDFs ausgelesen und zum anderen
# die Texte so sortiert und geloescht, dass jeder Text nur einmal vorkommt
# und es eine Einordnung in Kuedigung und keine Kuendigung gibt.

# =============================================================================
# Pakete laden & Funktionen einbinden - AKTUALISIEREN WENN DATEINAME NEU
# =============================================================================
import os
import pandas as pd
import dill   
import time

os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
from Funktionen import *

# =============================================================================
# Einlesen & Text extrahieren
# =============================================================================

# # 1. Kuendigungen

# [txt_KNK, dauer_KNK] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Kündigung')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')   
# dill.dump_session('knk_03_09.pkl')

# # 2. andere GeVos

# # Beitragspause
# [txt_BP, dauer_BP] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Beitragspause')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')    
# dill.dump_session('bp_03_01.pkl')

# Beitragsfreistellung
# [txt_BF, dauer_BF] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Beitragsfreistellung')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')   
# dill.dump_session('bf_03_01.pkl')

# # Erhöhung
# [txt_E, dauer_E] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Erhöhung')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')   
# dill.dump_session('e_03_01.pkl')

# # VN - Wechsel
# [txt_VN1, dauer_VN1] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\VN-Wechsel-Teil1')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')    
# dill.dump_session('vn1_03_01.pkl')

# [txt_VN2, dauer_VN2] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\VN-Wechsel-Teil2')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')   
# dill.dump_session('vn2_03_01.pkl')

# [txt_VN3, dauer_VN3] = ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\VN-Wechsel-Teil3')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')    
# dill.dump_session('vn3_03_01.pkl')

# [txt_VN4, dauer_VN4] =ordnerEinlesen(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\VN-Wechsel-Teil4')
# os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')     
# dill.dump_session('vn4_03_01.pkl')

# =============================================================================
# alles in einen Workspace speichern
# =============================================================================

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
# dill.load_session('knk_03_09.pkl')
# dill.load_session('bf_03_01.pkl')
# dill.load_session('bp_03_01.pkl')
# dill.load_session('e_03_01.pkl')
# dill.load_session('vn1_03_01.pkl')
# dill.load_session('vn2_03_01.pkl')
# dill.load_session('vn3_03_01.pkl')
# dill.load_session('vn4_03_01.pkl')

# del(dauer_BF, dauer_BP, dauer_E, dauer_VN1, dauer_VN2, dauer_VN3, dauer_VN4, dauer_KNK)
# txt_VN = txt_VN1 + txt_VN2 + txt_VN3 + txt_VN4
# del(txt_VN1, txt_VN2, txt_VN3, txt_VN4)


# dill.dump_session('texte_03_09.pkl')
dill.load_session('texte_03_09.pkl')

# =============================================================================
# Kündigungen sortieren
# =============================================================================
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt')
k_tab = pd.read_excel('Label_final_endgültig.xlsx', sheet_name = 'Kündigungen')

# PROBLEM: TXT_KNK liest nach wachsendem einlesedatum ein

# daher: 
os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt')
txt_KNK_alt = txt_KNK
ks = next(os.walk('Kündigung'))[2]

path = r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Kündigung'
uhrzeit = []
for i in range(len(ks)):
    zeit = os.path.getctime(path + r'\\' + ks[i])
    uhrzeit.append(time.ctime(zeit))

zeitendf = pd.DataFrame([ks, uhrzeit, txt_KNK]).T
zeitendf.columns = ['pdfname', 'zeit', 'texte']

zeitendf_sorted = zeitendf.sort_values('pdfname')

txt_KNK = list(zeitendf_sorted['texte'])

# hier einzeln weil: manchmal ist noch was per hand drauf geschreiben (zB kreuze)
# und dann nehme ich das gerne mit
# zusätzliche 6: wsl weil keine VNR angegeben war (oder kein RINR) --> 
k_tab['Texte'] = txt_KNK
# auswahl = 2 bedeutet in der Tabelle, dass es eine Kündigung ist, 
# die wir anschauen
ks = myGleich(k_tab['auswahl'], 2)
doppelt = myGleich(k_tab['doppelt_ind (0 = ist ein doppeltes)'], 0)


txt_K = list(k_tab['Texte'][k_tab['auswahl'] == 2])
txt_NK = list(k_tab['Texte'][k_tab['auswahl 2'] == 1])

alle_NK = txt_BF + txt_BP + txt_E + txt_VN + txt_NK
alle_texte = alle_NK + txt_K

del(doppelt, i, k_tab, ks, txt_BF, txt_BP, txt_E, txt_VN)

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
# dill.dump_session('k_nk_04_26.pkl')

# neu_all = seitenZusammen(alle_texte)
NK_zusammen = seitenZusammen(alle_NK)
K_zusammen = seitenZusammen(txt_K)

# =============================================================================
# Duplikate löschen
# =============================================================================

NK_unique = pd.unique(NK_zusammen)

NK_duplikatfrei = []
for i in range(len(NK_unique)):
    NK_duplikatfrei.append(NK_unique[i])
    
K_unique = pd.unique(K_zusammen)

K_duplikatfrei = []
for i in range(len(K_unique)):
    K_duplikatfrei.append(K_unique[i])
    
del(alle_NK, alle_texte, i, K_unique, K_zusammen, NK_unique, NK_zusammen)

os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Daisy-Daten\Daten - ungeschützt\Texte')
dill.dump_session('ohne_duplikate_04_26.pkl')
   
