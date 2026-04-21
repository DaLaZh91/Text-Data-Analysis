
# import os
# os.chdir(r'W:\Sonder\lva-93300\Masterarbeiten\Marie Punsmann\Python')
# from Funktionen import *

# =============================================================================
# Pakete laden
# =============================================================================

import pandas as pd # zB für die tf-idf Matrix
import numpy as np
import random
import os # zum setzen des working directories
import pytesseract as py # fuer die OCR
py.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pikepdf # zum speichern der pdf dateien als ungeschützte dateien
from pdf2image import convert_from_path #, convert_from_bytes
# import dill
from datetime import datetime
from Levenshtein import distance as lev
from collections import defaultdict # für die Häufigkeiten der GeVos
import collections as cl # für häufigkeiten auch
import xlsxwriter # um in excel tabelle zu schreiben
from wordcloud import WordCloud #fuer wordclouds
import itertools
from collections import Counter
import snowballstemmer # fürs stemming
from stop_words import get_stop_words #fuer die deutschen stoppwords
from sklearn.ensemble import RandomForestClassifier # für den Random Forest
from sklearn.metrics import confusion_matrix
from sklearn import svm # fuer svm
import warnings # fuer svm 
from sklearn.ensemble import ExtraTreesClassifier # ET
from sklearn import metrics # ROC 
import matplotlib.pyplot as plt # ROC
import matplotlib

# =============================================================================
# random.seed setzen, für den Fall dass es mal gebraucht wird 
# =============================================================================

random.seed(1921)

# =============================================================================
# =============================================================================
# # pdf einlesen + OCR (für 1_PDFs_konvertieren und 2_PDFs_auslesen)
# =============================================================================
# =============================================================================

# =============================================================================
# umspeichern von geschützten pdfs zu ungeschützten
# =============================================================================

def pdfUmspeichern(datei, datenordner, pdfordner):
    '''
    speichert pdf "datei" um als ungeschützte pdf
    
    Inputs:     datei - geschützte pdf Datei
                datenordner - da liegt die geschützte pdf
                pdfordner - zielordner für die ungeschützte pdf
    
    Outputs:    kein Output in python, ungeschützte pdf kommt in pdfordner
    '''
    os.chdir(datenordner)
    pdf = pikepdf.open(datei, allow_overwriting_input=True)
    os.chdir(pdfordner)
    pdf.save(datei)
    

def vielUmspeichern(datenordner, pdfordner):
    '''
    speichert alle geschützen pdfs aus datenordner in geschützte pdfs in 
    pdfordner um
    
    Inputs:     datenordner - da liegen die geschützten pdfs
                pdfordner - zielordner für die ungeschützten pdfs
    
    Outputs:    kein Output in python, ungeschützte pdfs kommen in pdfordner
    '''
    for i in next(os.walk(datenordner))[2]:
        pdfUmspeichern(i, datenordner, pdfordner)
        
# wird möglicherweise nicht mehr benötigt       
def teilUmspeichern(trennung, path0, path1, path2):
    '''
    speichert die Daten aus path0 nach einer Trennung in path1 und path2 um

    Inputs:     trennung - vektor mit Namen der Dokumente, die in path1 sollen
                path0 - ursprünglicher Dateipfad
                path1 - gewünschter Dateipfad der ausgewählten Dokumente
                path2 - gewünschter Dateipfad der nicht ausgewählten Dokumente
            
    Outputs:    keine, speichern in den Ordnern
    '''
    for i in trennung:
        pdfUmspeichern(i, path0, path1)
      
    ges = []
    for i in os.walk(path0):
        ges.append(i)
    
    for i in ges[0][2]:
        if i not in trennung:
            pdfUmspeichern(i, path0, path2)

        
# =============================================================================
# umwandeln der pdfs zu bildern 
# =============================================================================
        
def getBild(datei, pdfordner):
    '''
    speichert ungeschützte pdf um als ppm Datei

    Inputs:     datei - ungeschützte pdf datei 
                pdfordner - Ordner, in dem datei gespeichert ist
    
    Outputs:    kein output in python, ppm Datei wird in pdfordner gespeichert
                
    '''
    os.chdir(pdfordner)
    im = convert_from_path(datei, 
                        poppler_path = r'C:\ProgramData\Anaconda3\pkgs\poppler-22.01.0-h24fffdf_2\Library\bin')
    return im
    
 
def getBilder(pdfordner):
    '''
    wendet getBild auf jede Datei im pdfordner an
    
    Inputs:      pdfordner - Ordner mit den Dateien, die umgewandelt werden sollen
    
    Outputs:    die Bilder
    '''
    bilder = []
    for i in next(os.walk(pdfordner))[2]:
        bilder.append(getBild(i, pdfordner))
    return bilder   

# =============================================================================
# den text aus den bildern auslesen 
# =============================================================================


def bildAuslesen(s):
    '''
    liest für jedes bild in s den Text aus
    
    Inputs:     s - Liste mit Bildern
    
    Outputs:     s_text - Liste mit aus den Bildern ausgelesenen Texten
    '''
    s_text = []
    for j in range(len(s)):
        text = []
        for i in s[j]:
            text.append(py.image_to_string(i, lang = 'deu'))
        s_text.append(text)
    return s_text


def getText(pdfordner):
    '''
    list Texte aus Dateien aus (Hintereinanderschaltung von getBilder und
                                bildAuslesen)

    Inputs:     pdfordner - Ordner mit den ungeschützten pdf Dateien
    
    Outputs:    txt - Liste mit aus den Dateien ausgelesenen Texten

    '''
    bilder = getBilder(pdfordner)
    txt = bildAuslesen(bilder)
    return txt

# wird möglicherweise nicht mehr benötigt   
# das ist eigentlich die getText, nur um die dauer erweitert (vielleicht
# eher die getText weg und diese behalten, da diese in den weitern files
# benutzt wird)
def ordnerEinlesen(pdfordner):
    '''
    liest die Texte eines Ordners ein und gibt an, wie lange es dauert
    
    Inputs:     pdfordner - Ordner mit den ungeschützten pdf Dateien
    
    Outputs:    txt - Liste mit aus den Dateien ausgelesenen Texten
                dauer - Zeit, die die Ausführung dieser Funktion dauert

    '''
    start = datetime.now()
    txt = getText(pdfordner)
    ende = datetime.now()
    dauer = ende - start
    return txt, dauer

def seitenZusammen(texte):
    '''
    fügt mehrere Seiten zusammen
    
    Inputs:      teste -     Liste mit mehreren Listen, jede Liste enthält ein 
                            Dokument, können auch mehrseiteige Elemente sein
                    
    Outputs:     neu_text -  Liste mit den einzelnen Dokumenten, bei mehreren 
                            Seiten nicht mehr auf Seiten aufgeteilt
    '''
    neu_text = []
    for i in range(len(texte)):
        neu_text.append(' '.join(texte[i]))
    return neu_text

# =============================================================================
# Duplikate finden
# =============================================================================

def findDuplicates(vektor):
    '''
    diese Funktion gibt für jedes Element des Vektors vor, wie oft es vorkommt
    
    Inputs:     vektor - interessierender Vektor
    
    Outputs:    anzahlen - Häufigkeiten des vorkommens jedes Eintrages 
                           (bspw wäre vektor: (1, 2, 1, 3, 3), dann wäre
                            anzahlen: (2, 1, 2, 2, 2))
    '''
    anzahlen = [0] * len(vektor)
    for i in range(len(vektor)):
        anzahlen[i] = vektor.count(vektor[i])
    return(anzahlen)

# TODO
# Beschreibung schreiben
def createDuplDict(txt):
    '''
    
    
    Inputs:     txt
    
    Outputs:    vorkommen

    '''
    dupl_vekt = findDuplicates(txt)
    dupl_anzahl = []
    i = 1
    while sum(dupl_anzahl) < len(txt):
        dupl_anzahl.append(len(myGleich(dupl_vekt, i)))
        i += 1
    vorkommen = dict(zip(range(1, i + 1), dupl_anzahl))
    return(vorkommen)

# =============================================================================
# =============================================================================
# # Textvorverarbeitung (für 3_Tabellenerstellung)
# =============================================================================
# =============================================================================

# Levenshtein Funktion
# Die soll dafür sorgen, dass beim Vergleich zweier Namen oÄ die
# Levenshtein Distanz berechnet wird, also dass man zB sagen kann, dass
# das bei ner LD von 1 oder sowas wahrscheinlich das gleiche Wort ist, nur
# dass da beim einlesen was schief gegangen ist, was auf jeden Fall passieren 
# wird. Dann kann man zB sagen man nimmt immer den ersten als richtigen, wenn 
# die LD so klein ist, dass man sagt es ist das gleiche, weil das vllt
# am wslsten richtig geschrieben ist, beim zweiten vllt auch oft beim ersten
# abgeschrieben, per Hand vllt am anfang am ordentlichsten etc.

# TODO  überall zum Vergleich nutzen, überlegen wie

# =============================================================================
# Infos aus Dokument auslesen
# =============================================================================

# Namen, VNR, Geschäftsvorgang, Coronaind

# unterschied zu vektor.vergleich(): gibt auch mehrere aus!
def myGleich(vektor, vergleich):
    # überprüft
    '''
    überprüft ob das Wort vergleich in vektor vorkommt und wo
    
    Inputs:  vektor 
            vergleich - Wert, mit dem verglichen werden soll
            
    Ouput:  erg - Indizes, an denen das Wort steht bzw "1" wenn Wort nicht 
                    in dem Vektor vorkommt
    '''
    erg = [i for i in range(len(vektor)) if vektor[i] == vergleich]
    return erg 


def myVektorGleich(vektor1, vektor2):
    '''
    überprüft für welche i vektor1[i] = vektor2[i]
    
    Inputs:     vektor1
                vektor2
                
    Outputs:    erg - Indizes für die vektor1[i] = vektor2[i]
    '''
    erg = [i for i in range(len(vektor1)) if vektor1[i] == vektor2[i]]
    return erg    

    
def searchIndizes(schreiben, token):
    # überprüft
    '''
    überprüft ob das gesuchte token in dem dokument vorkommt und gibt die
    indizes an, in denen es steht, bspw um Wörter wie von:, grüße, etc. zu 
    finden
    
    Inputs:     schreiben - das NICHT TOKENISIERTE Dokument
                token - das Token, das gefunden werden soll
            
    Ouputs:     erg - Indizes, an denen das Wort steht bzw "1" wenn Wort nicht 
                      in dem Vektor vorkommt
    '''
    vektor = schreiben.split()
    erg = myGleich(vektor, token)
    return(erg)


def exWort(indizes):   
    '''
    gibt TRUE aus, wenn das Wort vorkommt und FALSE wenn nicht
    
    Inputs:     indizes - Ausgabe von searchIndizes
    
    Outputs:    TRUE/ FALSE
    '''
    return bool(type(indizes) is list)
    


def naechsteZeichen(schreiben, tok, n):
   '''
   findet die naechsten n token, hinter einem vorgegebenen token
    
   Inputs:      schreiben - das NICHT TOKENISIERTE Dokument
                tok - token (WORT)
                n - Anzahl weiterer Token 
              
   Outputs:     A -  Matrix mit den entsprechenden Token
                   Sonderfall: Dokument zuende: []
   '''
   schreiben_tok = schreiben.split()
   tokenstarter = searchIndizes(schreiben, tok)
   # erstelle ein A
   A = []
   for i in range(len(tokenstarter)): 
       A.append([])
       for j in range(n):
           A[i].append('')
   # fülle das A
   for i in range(len(tokenstarter)):
        for j in range(n):
            if len(schreiben_tok) > tokenstarter[i] + j + 1:
                A[i][j] = schreiben_tok[int(tokenstarter[i]) + j + 1]
            else:
                A[i][j] = []
   return A


def findeFunc(schreiben, tokenvektor, n):
    '''
    findet die nächsten n token nach den token aus tokenvektor
    
    Inputs:     schreiben - das NICHT TOKENISIERTE Dokument 
                tokenvektor - Vektor mit mgl token 
                n - Anzahl Token hinter dem angegebenen, die betrachtet werden
              
    Outputs:    mgl_token - die entsprechenden Token
    '''
    count = -1
    namen = []
    mgl_token = []
    
    # wenn nur nach einem token gesucht wird
    if isinstance(tokenvektor, str):
        tk = searchIndizes(schreiben, tokenvektor)
        if exWort(tk):
            erg = naechsteZeichen(schreiben, tokenvektor, n)
            mgl_token += erg
            
    # wenn nach einem Vektor gesucht wird        
    else:
        for i in range(len(tokenvektor)):
            count += 1
            token = tokenvektor[i]
            tk = searchIndizes(schreiben, token)
            if exWort(tk):
               locals()['token_%s' % count] = naechsteZeichen(schreiben, token, n)
               namen += [count]
        for i in namen:
            mgl_token += locals()['token_%s' % i]
    return mgl_token



def levenMatrix(vektor):
    '''
    berechnet für eine Liste die Levenshtein Distanz zwischen allen
    Listenelementen
    
    Inputs:     vektor (die Liste)
    
    Outpust:    LM - die Matrix

    '''
    # wenn der form [['vor' 'zu'], ['vor2' 'zu2'], ['vor3' 'zu3']]
    # (was so sein sollte, zumindest bei Namenfinden)
    # umwandeln in ['vor zu', 'vor2 zu2', 'vor3 zu3']
    if len(vektor[0][0]) > 1:
        neuvek = []
        for i in range(len(vektor)):
           neuvek.append(' '.join(vektor[i]))
        vektor = neuvek
    n = len(vektor)
    # setzt schon die Nullen auf der Diagonalen fest
    LM = [0] * n
    for x in range(n):
        LM[x] = [0] * n
    for i in range(n):
        for j in range(n):
            if j > i:
                LM[i][j] = lev(vektor[i], vektor[j])
            else: # quasi gespiegelte Matrix
                LM[i][j] = LM[j][i]
                
    LM = pd.DataFrame(LM)
    LM.columns = vektor
    return(LM)

def findEinzelToken(mglkeiten):
    '''
    sucht die Token einzeln (nicht als ganzer 'Wortblock' nach Duplikaten ab)
    
    Inputs:     mglkeiten - Vektor mit den möglichen Namen, VNRs oÄ
    
    Outputs:    Erg - Matrix mit der Anzahl Wörter, die in jeder Kombination
                      der Wortblöcke häufiger vorkommt

    '''
    # geht erst ab zweielementigen
    z = len(mglkeiten)
    n = len(mglkeiten[0])
    Erg = pd.DataFrame(z * [z * [0]])
    for i in range(z): # ueber alle Wortblöcke
        for j in range(n): # hole das einzelene Token raus
            vglwort = mglkeiten[i][j]
            for k in range(z):
                count = 0
                for l in range(n):
                    if (vglwort == mglkeiten[k][l]):
                        count += 1
                Erg[i][k] += count
    return(Erg)


# diese funktion soll die ganzen übereinstimmungen, die nicht auf der
# mittellinie liegen, und alle teilweisen übereinstimmungen mit den
# jeweils übereinstimmenden Wörtern ausgeben

# bzw stattdessesn

# def whichMehrfach(mglkeiten, n = 2):
#     erg = findEinzelToken(mglkeiten)
#     # betrachte dazu nur die obere dreiecksmatrix
#     erg

def countToken(mglkeiten):
    '''
    zählt für jedes unterschiedliche Token, wie oft es im Vektor mglkeiten
    vorkommt
    
    Inputs:     mglkeiten - Vektor mit möglichen Namen, VNRs etc, deren
                            Häufigkeiten bestimmt werden sollen
                            
    Outputs:    counter -   Objekt, das beinhaltet, wie oft jedes Token
                            vorkommt, hier erstmal ohne Levenshtein-Distanz
                            zu beachten, sehr ähnliche werden also als zwei 
                            einzelne aufgeführt
    '''
    # TODO so ähnlich -> ausgabe anpassen!
    if ([mglkeiten[0]] == mglkeiten):
        return(cl.Counter(mglkeiten[0]))
    else:
        alle = []
        for i in range(len(mglkeiten)):
            for j in range(len(mglkeiten[i])):
                alle.append(mglkeiten[i][j])
                
        counter = cl.Counter(alle)
        
        return(counter)

### VNR =======================================================================

# Das wird als erstes gemacht, weil hier die Zeichen noch wichtig sind. 
# Hiernach werden sie entfernt (für Namen, GeVo usw.)

def vnrFinden(schreiben):
    '''
    durchsucht die typischen Plätze für Versicherungsnummern
    
    Inputs:     schreiben
    
    Outputs:    mögliche VNRs
    '''
    vnr_token = ['Versicherungsnummer', 'VersicherungsNr.', 'Nr.', 'Nr:', 
                 'Nr.:', 'Nr', 'VNR', 'VNR:', 'Versicherungsnummer:', 
                 'VSNR:', 'VSNR', 'VSNR.', 'VSNR,','VSNR.:']
    
    return findeFunc(schreiben, vnr_token, 2)


def vnrBauen(schreiben):
    '''
    versucht die VNR zusammenzubauen, und zu stoppen, wenn sie vorbei ist
    
    Inputs:     schreiben
    
    Outputs:    mögliche VNRs
    '''
    mglkeiten = vnrFinden(schreiben)
    vgl_vektor = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    neu_mgl = mglkeiten # der gleichen Form wie mglkeiten machen
    # loopen über die verschiedenen male, dass eine VNR auftritt
    for k in range(len(mglkeiten)):
        # loopen über die verschiedenen token die ausgelesen werden
        for j in range(len(mglkeiten[k])): # das ist auch immer das gleiche
            # ev loopen über die ersten beiden elemente um zu gucken, dass
            # es wirklich eine Zahl ist?
            a_mgl = neu_mgl[k][j]
            if len(a_mgl) > 0:
                for i in range(len(a_mgl)):
                    # checken ob was vorkommt, was keine Ziffer oÄ ist
                    if a_mgl[i] not in vgl_vektor:
                            a_mgl = a_mgl.replace(a_mgl[i], 'X')
            neu_mgl[k][j] = a_mgl
    # lösche, ab wenn es mit 2 X irgendwo losgeht & wenn es leer ist
    str_zsm = [None] * len(mglkeiten)
    for k in range(len(mglkeiten)):
        for j in range(len(mglkeiten[k])): 
                if 'X' in neu_mgl[k][j] or neu_mgl[k][j] == []:
                    neu_mgl[k][j] = ''
        # füge zusammen, was dann für jede VNR übrig bleibt
        str_zsm[k] = "".join(neu_mgl[k])       
    return str_zsm



def searchZahlenzeichenketten(schreiben):
    '''
    findet Zahlenzeichenketten (mit den Zeichen und Zahlen aus vgl_vektor)
    in einem Dokument
    
    Inputs:      schreiben - nicht tokenisiertes Dokument
    
    Outputs:    zahlentoken - die Zahlentoken, die in dem Dokument 
                                vorkommen
                numbind - die Indizes dieser Zahlentoken

    '''
    vgl_vektor = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    schreibsplit = schreiben.split()
    schreibsplit
    isdig = []
    for i in range(len(schreibsplit)):
        # isdig.append(schreibsplit[i].isdigit())
        token = schreibsplit[i]
        for j in range(len(token)):
            if (token[j] not in vgl_vektor):
                isdig.append(False)
                break;
            elif (j == (len(token) - 1)):
                isdig.append(True)
    numbind = myGleich(isdig, True)
    zahlentoken = np.array(schreibsplit)[numbind]
    return(zahlentoken, numbind)


# TODO
# hier nochmal checken was passiert
def searchVNR(schreiben):
    '''
    findet VNRs indem es die Zahlentoken aus 
    searchZahlenzeichenketten zusammenfügt
    
    Inputs:     schreiben - nicht tokensiertes Dokument
    
    Outputs:    zt - 

    '''
    zahlentoken, numbind = searchZahlenzeichenketten(schreiben)
    isnext = []
    for i in range(len(numbind) - 1):
        if (numbind[i] + 1 == numbind[i + 1]):
            isnext.append(True)
        else:
            isnext.append(False)
    # kombiniere die, die direkt hintereinanderstehen
    zahlentoken = list(zahlentoken)
    zt = zahlentoken
    for i in range(len(numbind) - 1):
        if (isnext[i] and not isnext[i + 1]):
            # TODO 
            # hier nochmal ändern, dass eine indexverschiebung durh
            # ver#nderung von zt in einer schleifeniteration in den 
            # nächsten beachtet wird, das ist aktuell noch nicht der fall
            zt[i:(i + 2)] = [''.join(zahlentoken[i:(i + 2)])]
    return(zt)

    
# TODO - Ergänzungen notwendig: searchVNR rein, levenshtein rein, häufigkeit rein
# vergleicht die VNRs
def vnrVergleich(schreiben):
    # überprüft
    '''
    vergleicht die VNRs 
    
    Inputs:     schreiben - nicht tokensiertes Dokument
    
    Outputs:    im Optimalfall: VNR, sonst aktuell (ÄNDERN!!) 'unklare VNR' und
                'keine VNR gefunden' (printet das, speichert für die hinteren
                                  beiden fälle None)
    '''
    try: 
        mgl_vnr = vnrBauen(schreiben)
        vnr = mgl_vnr[0]
        gleich = [None] * len(mgl_vnr)
        for i in range(len(mgl_vnr)):
            gleich[i] = bool(vnr == mgl_vnr[i])
        if False not in gleich:
            return vnr
        else:
            return ('ev ' + vnr)
            # TODO 
            # jetzt gebe ich hier einfach die erste mit einem 'EV' davor
            # aus, das ist jetzt nicht unbedingt die Optimallösung
    except:
        # TODO
        # nach VNR hinter dem Namen suchen wenn keine zu finden im Text (Kopf)
        # vielleicht einfach lange Zahlen suchen?
        return 'keine gefunden'

# TODO
# hier noch nach langen ketten von Zeichen suchen zb "!!!!!!!!"

### Punctuation entfernen =====================================================

def delPunct(schreiben):
    # überprüft
    '''
    löscht punctuation aus dem Schreiben
    
    Inputs:     schreiben - NICHT tokenisiertes schreiben
    
    Outputs:    schreiben ohne punctuation

    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»‚‚')
    token_punctless =[token for token in schreiben if token not in punctuation]
    token_punctless = ''.join(token_punctless)
    return token_punctless

def delPuncts(dok_list):
    # überprüft
    '''
    löscht die Punctuation aus allen Schreiben in der dok_list
    
    Inputs:     dok_list - Liste nicht-tokenisierter Dokumente
    
    Outputs:    dok_list ohne punctuation
    '''
    new_list = []
    for i in dok_list:
        new_list.append(delPunct(i))
    return new_list

### alle Buchstaben klein =====================================================

def machLowercase(dok_list):
    # überprüft
    '''
    macht alle buchstaben zu Kleinbuchstaben
    
    Inputs:     dok_list - Liste nicht-tokenisierter Dokumente
    
    Outputs:    new_list - Liste mit nicht-tokenisierten Dokumenten, komplett 
                           in Kleinbuchstaben
    '''
    new_list = []
    for i in dok_list:
        new_list.append(i.lower())
    return new_list

### Name ======================================================================

def namenFinden(schreiben):
    '''
    durchsucht mit namenFinden die typischen Plätze für Namen
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    mgl_namen - mögliche Namen
    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»')
    numbers = list('0123456789')
    schreiben_lower = delPunct(schreiben).lower()
    namen_token = ['grüßen', 'grüße', 'gruß', 'von', 'from']
    schreiben_tok = schreiben_lower.split()
    # da der Name oft als erstes im Briefkopf steht:
    stelle12 = schreiben_tok[0:2]
    m = 2
    mgl_namen = findeFunc(schreiben_lower, namen_token, m)
    mgl_namen.append(stelle12)
    todel = []
    # loop über alle mgl namen um email adressen und zahlen rauszuwerfen
    nein_words = ['von', 'signal', 'iduna', 'signaliduna', 'kundenservice',
                  'kundenberater', 'kundenberaterin', 'ihnen', 'der', 
                  'die', 'das', 'dem']
    for i in range(len(mgl_namen)):
        for j in range(m): 
            if mgl_namen[i][j] in nein_words:
                todel.append(i)
                break            
        # sonst die beiden wortteile getrennt betrachten und nur die einzelnen
        # buchstaben betrachten
        else: 
            for j in range(m):
                part = list(mgl_namen[i][j])
                if any(map(lambda v: v in punctuation, part)):
                    todel.append(i)
                    break
                elif any(map(lambda v: v in numbers, part)):
                    todel.append(i)
                    break
                # damit email adressen mit signaliduna rausgehen
                elif all(map(lambda v: v in part, list('signaliduna'))):
                    todel.append(i)
                    break
    j = 0
    for i in range(len(todel)):
        del(mgl_namen[todel[i] - j])
        j = j + 1
    return mgl_namen

# TODO
# hier cleverere Auswahl treffen, levenshtein oÄ rein (das reicht aber noch 
# nicht glaube ich)
def nameVergleich(schreiben):
    '''
    JETZT WIRD EINER DER NAMEN DIE AM HÄUFIGSTEN VORKOMMEN GENOMMEN (DER ERSTE)
    PROBLEME: ANDERS GESCHRIEBEN; NAME KOMMT NUR EINMAL VOR
    Inputs: schreiben (nicht tokenisiert)
    
    Outputs: ein möglicher Name
    '''
    try:
        namen_vek = namenFinden(schreiben)
        counter = []
        for i in range(len(namen_vek)):
            counter.append(namen_vek.count(namen_vek[i]))
        max_count = max(counter)
        erg = myGleich(counter, max_count)
        if len(namen_vek[erg[0]][0]) > 0:
            ein_max = namen_vek[erg[0]]
        else:
            ein_max = namen_vek[erg[1]]
    except:
        ein_max = ['N', 'Ö']
    return ein_max

### GeVo ======================================================================

def KFindenSimple(schreiben, k_words):
    '''
    sucht nach den Wörtern, die eine Kündigung beschreiben und sortiert so in
    Kündigung und keine Kündigung
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    dok_Gevo - 'K' oder 'N', je nachdem ob Kündigung oder nicht
    '''
    schreiben_lower = delPunct(schreiben).lower()
    dok = schreiben_lower.split()
    dok_GeVo = []
    if any(map(lambda v: v in k_words, dok)):
         dok_GeVo = 'K'
    else:
        dok_GeVo = 'N'
    return(dok_GeVo)


def gevoFinden(schreiben, k_words, nk_words, pos = 'K', neg = 'N'):
    '''
    sucht nach den Wörtern, die eine Kündigung beschreiben und sortiert so in
    Kündigung und keine Kündigung
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    dok_Gevo - 'K' oder 'N', je nachdem ob Kündigung oder nicht
    '''
    schreiben_lower = delPunct(schreiben).lower()
    dok = schreiben_lower.split()
    dok_bigram = [' '.join(b) for l in [' '.join(dok)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    dok_GeVo = []

    if nk_words == []:
        if any(map(lambda v: v in k_words, dok)):
            dok_GeVo = pos
        elif any(map(lambda v: v in k_words, dok_bigram)):
            dok_GeVo = pos
        else:
            dok_GeVo = neg
    else: 
        if any(map(lambda v: v in nk_words, dok)):
            dok_GeVo = neg
        elif any(map(lambda v: v in nk_words, dok_bigram)):
            dok_GeVo = neg
        elif any(map(lambda v: v in k_words, dok)):
             dok_GeVo = pos
        elif any(map(lambda v: v in k_words, dok_bigram)):
            dok_GeVo = pos
        else:
            dok_GeVo = neg
            
    return(dok_GeVo)


### Grund =====================================================================


#TODO
# insb die Wörter noch sehr stark überarbeiten
# B: Berufswechse/ Abgang
# F: finanziell
# R: Rente/ Berufsausstieg
# T: Todesfall
# S: Sonstiges/ Keiner

def grundFinden(schreiben):
    '''
    sucht nach den Wörtern, die einen Grund beschreiben und ordnet
    entsprechende Gründe zu
    
    Inputs:     schreiben - nicht tokenisiertes Schreiben
    
    Outputs:    dok_Grund - Vektor mit Buchstaben, die die Gründe beschreiben
    '''
    schreiben_lower = delPunct(schreiben).lower()
    dok = schreiben_lower.split()
    dok_Grund = []
    b_words = ['arbeitnehmerwechsel', 'neue arbeit', 'abgang']
    f_words = ['finanziell', 'geld', 'insolvenz', 'finanziellen']
    r_words = ['rente', 'rentenbeginn', 'rentenzahlung', 'nicht arbeitsfähig', 
               'nicht mehr arbeitsfähig']
    t_words = ['todesfall', 'gestorben', 'todes', 'tod', 'verstorben', 
               'verstorbene']
    if any(map(lambda v: v in b_words, dok)):
        dok_Grund += 'B'
    if any(map(lambda v: v in f_words, dok)):
        dok_Grund += 'F'
    if any(map(lambda v: v in r_words, dok)):
        dok_Grund += 'R'
    if any(map(lambda v: v in t_words, dok)):
        dok_Grund += 'T'

    return dok_Grund


def grundVergleich(schreiben):
    '''
    gibt aus einem Dokument aus, zu welchem Grund die meisten Wörter vorkommen
    und wie viele es zu jedem Grund gibt
    
    Inputs:     schreiben
    
    Outputs:    Grund - am häufigsten vorkommender Grund
    
                falls dieser nicht eindeutig:
                erg_tab - Liste, wie oft jeder Grund vorkommt
    '''
    schreiben_lower = delPunct(schreiben).lower()
    mgl_gruende = grundFinden(schreiben_lower)
    res = defaultdict(int)
    for i in mgl_gruende:
        res[i] += 1
    erg_tab = (dict(res))
    B_freq = erg_tab.get('B', 0)
    F_freq = erg_tab.get('F', 0)
    R_freq = erg_tab.get('R', 0)
    T_freq = erg_tab.get('T', 0)
    freq = ['B', 'F', 'R', 'T']
    freqs = [B_freq, F_freq, R_freq, T_freq]
    maxGruende = max(freqs)
    
    if maxGruende > 0:
        Grund = freq[myGleich(freqs, maxGruende)[0]]
        if sum(freqs) == maxGruende:
            return Grund
        else:
            return Grund, erg_tab
    else:
        return 'S' 

### Corona ====================================================================

def covidFinden(schreiben):
    # überprüft
    '''
    durchsucht ob die Wörter Corona, covid, oÄ vorkommen
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    erg - Vektor mit Plätzen, an denen Coronawörter stehen (oÄ)
    '''
    covid_token = ['corona', 'covid', 'covid19', 'pandemie', 'coronabedingt',
                   'pandemisch', 'pandemische', 'pandemischen', 'kurzarbeit',
                   'coronakrise', 'lockdown', 'coronaregeln']
    schreiben_lower = delPunct(schreiben).lower()
    # TODO 
    # irgendwie startswith einbringen?
    erg = []
    for i in range(len(covid_token)):
        erg += searchIndizes(schreiben_lower, covid_token[i]) 
    return erg


def covidVergleich(schreiben):
    
    '''
    gibt TRUE aus, wenn ein Coronawort vorkommt (oder mehrere) und FALSE, 
    wenn nicht
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    TRUE/FALSE
    '''
    schreiben_lower = delPunct(schreiben).lower()
    erg = covidFinden(schreiben_lower)
    if len(erg) > 0:
        return True
    else:
        return False

# =============================================================================
# Nur den Textkörper rausholen (für die Klassifikation)
# =============================================================================


def getHauptteil(schreiben):
    '''
    holt den Textkörper aus einem schreiben
    DAS MACHT SIE NICHT MEHR WEIL DAS KEINEN SINN ERGIBT 
    NUR NOCH DRIN UM NICHT ALLES ÄNDERN ZU MÜSSEN
    
    Inputs:     schreiben - das nicht-tokenisierte Dokument
    
    Outputs:    erg - schreiben ohne 'Einleitung' und stand jetzt alles in
                      Kleinbuchstaben und ohne Punctuation
    '''
    schreiben_pl = delPunct(schreiben)
    schreiben_lower = schreiben_pl.lower()
    erg = schreiben_lower
    # try:
    #     erg = schreiben_lower[schreiben_lower.index('sehr geehrte'):]
    # except:
    #     try:
    #         erg = schreiben_lower[schreiben_lower.index('guten tag'):]
    #     except:
    #         try:
    #             erg = schreiben_lower[schreiben_lower.index('liebe'):]
    #         except:
    #             erg = schreiben_lower
    return erg

def getHauptteile(dok_list):
    '''
    getHauptteil für dok_list
    '''
    new_list = []
    for i in dok_list:
        new_list.append(getHauptteil(i))
    return new_list

# TODO
# eventuell alles nach den Grüßen löschen

# =============================================================================
# Alles in Excel Tabelle schreiben
# =============================================================================

### Tabelle erstellen ========================================================
 

def getInfos(schreiben):
    '''
    gibt für ein schreiben den Namen, die VNR, den GeVo und Corona aus
    
    Inputs:     schreiben - nicht tokenisiertes Dokument
    
    Outputs:    Infos - mit:
                name
                vnr
                gevo
                Corona
    '''
    name = nameVergleich(schreiben)
    vnr = vnrVergleich(schreiben)
    gevo = gevoFinden(schreiben, ['kündigung', 'kündige', 'kündigen'],
                      ['beitragsfreistellung', 'beitragspause', 'erhöhung'])
    covid = covidVergleich(schreiben)
    textteil = ' '.join(getHauptteil(schreiben).split())
    Infos = [name, vnr, gevo, covid, textteil]
    return Infos


def createTable(dok_list):
    '''
    erstellt eine Tabelle aus getInfos
    
    Inputs:     dok_list
    
    Outputs:    Tabelle

    '''
    Tabelle = []
    for i in range(len(dok_list)):
        Tabelle += getInfos(dok_list[i])
    return(Tabelle)
     
### Tabelle in Excel schreiben ================================================

def tabelleSchreiben(dok_list, filename):
    '''
    erstellt eine Excel Tabelle mit den gegebenen Infos
    
    Inputs:     dok_list - Liste mit den Dokumenten
    
                filename - der Name, unter dem das Excel file gespeichert werden 
                           soll
                           
    Outputs:    keine Outputs in python, excel Tabelle wird im Ordner 
                Schriftstücke\Outputs unter filename gespeichert
    '''
    tab = createTable(dok_list)
    namen = tab[0::5]
    vnrs = tab[1::5]
    gevos = tab[2::5]
    covids = tab[3::5]
    texts = tab[4::5]
    
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output')
    outWorkbook = xlsxwriter.Workbook(filename)
    outSheet = outWorkbook.add_worksheet()
    outSheet.write('A1', 'Vorname') 
    outSheet.write('B1', 'Nachname')
    outSheet.write('C1', 'VNR')
    outSheet.write('D1', 'GeVos')
    outSheet.write('E1', 'Corona?')
    outSheet.write('F1', 'Text')

    for i in range(len(vnrs)):
        try:
            outSheet.write(i + 1, 0, namen[i][0])
        except:
            outSheet.write(i + 1, 0, '')
        try:
            outSheet.write(i + 1, 1, namen[i][1])
        except:
            outSheet.write(i + 1, 1, '')
        try:
            outSheet.write(i + 1, 2, vnrs[i])
        except:
            outSheet.write(i + 1, 2, '')
        try:
            outSheet.write(i + 1, 3, gevos[i][0])
        except:
            outSheet.write(i + 1, 3, '')
        try:
            outSheet.write(i + 1, 4, covids[i])
        except:
            outSheet.write(i + 1, 4, '')
        try:
            outSheet.write(i + 1, 5, texts[i])
        except:
            outSheet.write(i + 1, 5, '')
            
    outWorkbook.close()
    

# =============================================================================
# =============================================================================
# # Textmining (für 4_Datenvorbereitung)
# =============================================================================
# =============================================================================

def setColRowNames(dataframe, colnames, rownames):
    '''
    setzt für eine pd.DataFrame colnames und rownames
    
    Inputs:     dataframe
                colnames
                rownames
                
    Outputs:    dataframe mit col- und rownames
    '''
    
    dataframe.columns = colnames
    dataframe2 = dataframe.T
    dataframe2.columns = rownames
    dataframe = dataframe2.T
    return(dataframe)

# =============================================================================
# Stemming und Co.
# =============================================================================

def getStem(texte):
    '''
    stemmt eine Liste von Texten
    
    Inputs:     texte - dok_list quasi????
    
    Outputs:    txt_stem - Liste mit gestemmten Texten

    '''
    stemmer = snowballstemmer.stemmer('german')
    txt_stem = []
    for i in range(len(texte)):
        try:
            stem_teil = stemmer.stemWords(texte[i].split())
            txt_stem.append(' '.join(stem_teil))
        except:
            txt_stem.append('nan')
    return(txt_stem)




def delWortvektor(schreiben, wortvektor):
    '''
    löscht alle Wörter die in einem Vektor vorkommen aus einem schreiben
    
    Inputs:     schreiben
                wortvektor
        
    Outputs:    schreiben ohne wörter aus wortvektor

    '''
    schreib = [token for token in schreiben.split() if token not in wortvektor]
    schreib2 = ' '.join(schreib)
    return(schreib2)





stop_words = get_stop_words('de')
stop_words_stem = getStem(stop_words)

# del_words = getStem(['betreff', 'bitte', 'dame', 'dortmund', 'freundliche',
#        'geehrt', 'grüßen', 'herr', 'hiermit', 'iduna', 'seit',
#        'signal', 'signaliduna', 'wegen', 'wurde', 'möchte',  'service-fax',
#        'fax', 'ich' , 'web', 'wohl', 'kurz', 'ab', 'fur', 'lasse',
#        'ev', 'steh', 'selbstverstand', 'zentral', 'gut', 'tag', 'bereit',
#        'cc', 'wurd', 'infosignalidunad', 'hamburg'])


# my_del_words = stop_words + del_words 

# ges_del = list(map((lambda x: x + ','), my_del_words)) + my_del_words


# löscht Stoppwörter und "del_words" aus einem Schreiben
# def delWords(schreiben):
#     # überprüft
#     '''
#     löscht Stoppwörter und "del_words" aus einem Schreiben
    
#     Inputs:     schreiben (mit Stopp- und del_words)
        
#     Outputs:    schreiben ohne Stoppwörter und del_words

#     '''
#     schreib2 = delWortvektor(schreiben, ges_del)
#     # schreib = [token for token in schreiben.split() if token not in my_del_words]
#     # schreib2 = ' '.join(schreib)
#     return(schreib2)    
    


def delNumbers(schreiben):
    '''
    löscht alle token, in denen Zahlen vorkommen
    
    Inputs:     schreiben - nicht tokenisiertes schreiben
    
    Outputs:    erg - nicht tokenisiertes schreiben ohne zahlen (als str)

    '''
    numbers = list('0123456789')  
    main = getHauptteil(schreiben)
    n = len(main.split())
    ausgabe = []
    for i in range(n):
        if not any(map(lambda v: v in numbers, main.split()[i])):
            ausgabe.append(main.split()[i])
    erg = ' '.join(ausgabe)
    return(erg)



### DTM Matrizen berechenn

def getDTM(vect, txt, cols = True):
    '''
    erstellt die Document Term Matrix (oben die Dokumente und seitlich die 
                                       Wörter)
    
    Inputs:     vect - CountVectorizer
                txt - Texte ohne Stoppwörter
    
    Outputs:    

    '''
    vects = vect.fit_transform(txt)
    anzahl = len(txt) 
    td = pd.DataFrame(vects.todense())
    td.columns = vect.get_feature_names()
    tdM = td.T
    tdM.columns = ['Doc '+ str(i) for i in range(1, anzahl + 1)]
    # tdM['total_count'] = tdM.sum(axis=1)
    # tdM.drop(columns=['total_count'])
    if cols != True:
        td2 = tdM.T
        td2 = td2[cols]
        tdM = td2.T
    return(tdM)

def getTokenCount(vect, X_train, name):
    vect.fit_transform(X_train)
    token = vect.get_feature_names()
    dtm = getDTM(vect, X_train)
    count = countFuncAll(dtm, token)
    count_DF = pd.DataFrame(data = [token, count], index = ['token', name]).T
    return count_DF

def delSelteneWoerter(vect, X_train_K, X_train_NK, p, K_len_train, NK_len_train):
    
    count_DF_K = getTokenCount(vect, X_train_K, 'count K')
    count_DF_NK = getTokenCount(vect, X_train_NK, 'count NK')

    gem_df = pd.merge(count_DF_K, count_DF_NK, on = 'token', how = 'outer')
    gem_df = gem_df.fillna(0)
    
    keep_K = np.where(gem_df['count K'] > (p * K_len_train))
    keep_NK = np.where(gem_df['count NK'] > (p * NK_len_train))
    
    keep_both = np.unique((np.append(keep_K, keep_NK)))
    keep_token = list(gem_df['token'][keep_both])
    
    new_df = pd.concat([gem_df['token'][keep_both], gem_df['count K'][keep_both], gem_df['count NK'][keep_both]], axis = 1)
    new_df = setColRowNames(new_df, new_df.columns, new_df['token'])
    
    return(keep_token, new_df)

def delSelteneWoerter2(vect, X_train_B, X_train_F, X_train_K, X_train_R, X_train_T, p, y_train):
    

    count_DF_B = getTokenCount(vect, X_train_B, 'count B')
    count_DF_F = getTokenCount(vect, X_train_F, 'count F')
    count_DF_K = getTokenCount(vect, X_train_K, 'count K')
    count_DF_R = getTokenCount(vect, X_train_R, 'count R')
    count_DF_T = getTokenCount(vect, X_train_T, 'count T')
    
    gem_1 = pd.merge(count_DF_B, count_DF_F, how = 'outer', on = 'token')
    gem_2 = pd.merge(gem_1, count_DF_K, how = 'outer', on = 'token')
    gem_3 = pd.merge(gem_2, count_DF_R, how = 'outer', on = 'token')
    gem_4 = pd.merge(gem_3, count_DF_T, how = 'outer', on = 'token')
#    gem_df = pd.concat([count_DF_B, count_DF_F, count_DF_K, count_DF_R, count_DF_T])#, on = 'token', how = 'outer')
    gem_df = gem_4    
    gem_df = gem_df.fillna(0)
    
    B_len = len(myGleich(y_train, 'B'))
    F_len = len(myGleich(y_train, 'F'))
    K_len = len(myGleich(y_train, 'K'))
    R_len = len(myGleich(y_train, 'R'))
    T_len = len(myGleich(y_train, 'T'))
    
    keep_B = gem_df['token'][gem_df['count B'] > (p * B_len)]
    keep_F = gem_df['token'][gem_df['count F'] > (p * F_len)]
    keep_K = gem_df['token'][gem_df['count K'] > (p * K_len)]
    keep_R = gem_df['token'][gem_df['count R'] > (p * R_len)]
    keep_T = gem_df['token'][gem_df['count T'] > (p * T_len)]
   
    keep_all = list(keep_B) + list(keep_F) + list(keep_K) + list(keep_R) + list(keep_T)    
    keep_token = np.unique(keep_all)
    
    gem_dfT = gem_df.T
    gem_dfT.columns = gem_df['token'] 
    new_df = gem_dfT[keep_token].T

    return(keep_token, new_df)


# def delSelteneWoerter(vect, X_train_K, X_train_NK, p, K_len_train, NK_len_train):
    
#     vect_K = vect
#     vect_K.fit_transform(X_train_K)
#     token_K = vect_K.get_feature_names() 
#     dtm_K = getDTM(vect_K, X_train_K)
    
#     count_K = countFuncAll(dtm_K, token_K)
#     count_K_df = pd.DataFrame(data = [token_K, count_K], index = ['token', 'count K']).T
     
#     vect_NK = vect
#     vect_NK.fit_transform(X_train_NK)
#     token_NK = vect_NK.get_feature_names() 
#     dtm_NK = getDTM(vect_NK, X_train_NK)
    
#     count_NK = countFuncAll(dtm_NK, token_NK)
#     count_NK_df = pd.DataFrame(data = [token_NK, count_NK], index = ['token', 'count NK']).T
    
#     gem_df = pd.merge(count_K_df, count_NK_df, on = 'token', how = 'outer')
#     gem_df = gem_df.fillna(0)
    
#     # Lösche alle Token, die in weniger als 1% der Kündigungen und in weniger
#     # als p% der Nicht-Kündigungen vorkommen (13.57; 126.28)
    
#     keep_K = np.where(gem_df['count K'] > (p * K_len_train))
#     keep_NK = np.where(gem_df['count NK'] > (p * NK_len_train))
    
#     keep_both = np.unique((np.append(keep_K, keep_NK)))
#     keep_token = list(gem_df['token'][keep_both])
    
#     new_df = pd.concat([gem_df['token'][keep_both], gem_df['count K'][keep_both], gem_df['count NK'][keep_both]], axis = 1)
#     new_df = setColRowNames(new_df, new_df.columns, new_df['token'])
    
#     return(keep_token, new_df)

# TODO
# einfangen wenn df() = 0 ist (geht das?)


### lösche seltene Wörter ================


def countFuncAll(dtm, token_vec):
    '''
    cF     zaehlt in wievielen der einzeltexte aus texte ein token vorkommt, ACHTUNG:
        zaehlt nicht wie oft es insgesamt vorkommt, sondern, in wie vielen der
        texten, wenn es also in einem text häufiger als einmal vorkommt, zählt
        das nur als ein Vorkommen
    wendet countFunc auf einen Vektor von token an

    Inputs:     dtm - 
                token_vec
    
    Outputs:     countvec
    '''
    countvec = []
    for i in range(len(token_vec)):
        countvec.append(len(np.where(dtm.loc[token_vec[i]] > 0)[0]))
    return countvec

def myNotIn(vektor, teilvektor):
    '''
    gebe vektor ohne die elemente aus teilvektor zurück
    
    Inputs:     vektor - langer Vektor 
                teilvektor - Vektor mit Elementen, die aus vektor gelöscht
                                werden sollen
        
    Outputs:    neuvek - neuer Vektor ohne die Elemente aus teilvektor
    '''
    neuvek = []
    for i in range(len(vektor)):
        if vektor[i] not in teilvektor:
            neuvek.append(vektor[i])
    return(neuvek)

def delWenige(anzahlvektor, anzahl, token_vek):
    '''
    löscht aus einem Vektor anzahlvektor alle weniger oder gleich anzahl
    mal vorkommenden wörter und gibt aus, welche wörter aus token_vek die sind,
    die übrig bleiben, also häufiger als anzahl mal vorkommen
    funktioniert auch für n-gramme
    
    Inputs:     anzahlvektor
                anzahl
                token_vek
                
    Outputs:    keep_words

    '''
    dellist = []
    for i in range(anzahl):
        dellist +=myGleich(anzahlvektor, i + 1)
    
    keep = myNotIn(range(len(anzahlvektor)), dellist)
    
    keep_words = np.array(token_vek)[keep]
    
    return keep_words

def getTFIDF(tf):
    '''
    berechnet die Tfidf Transformation einer Matrix
    
    Inputs:     tf - Matrix
    
    Outputs:    tf_idf - mit tf-idf transformierte Matrix

    '''
    tf_array = np.array(tf)
    df = []
    # J: Anzahl Dokumente
    J = len(tf.columns)
    
    for j in range(J):
    # in wie vielen dokumenten ist das nicht enthalten
        nichtent = len(myGleich(tf_array[:,j],0))
        df.append(J - nichtent)
    
    idf = []
    for j in range(J):
        idf.append(np.log(J/df[j]))
    
    tf_idf = tf * idf
    return(tf_idf)

# =============================================================================
# Gini Index
# =============================================================================


def giniCalc(token, dokvek, auftvek):
    '''
    berechnet den normalisierten Gini Koeffizienten eines Tokens
    
    Inputs:
                token
                dokvek - enthält "anzahl dokumente klasse 1 mit token", ..., 
                                "anzahl dokumente klasse n mit token"
                auftvek - enthält "anzahl dokumente klasse1", ...,
                                    "anzahl dokumente klasse n"
    
            
    Output:     gini - Gini Koeffizient des Token
    '''
    try: 
        n = len(auftvek)
        p = []
        for i in range(n):
            p.append(dokvek[i]/np.sum(dokvek))
        P = auftvek
        p_tilde = []
        for i in range(n):
            p_tilde.append(p[i]/P[i])
        p_tilde_ges = np.sum(p_tilde)
        gini = 0
        for i in range(n):
            gini += (p_tilde[i]/p_tilde_ges)**2
    except:
        gini = 0
    return gini

def getGini(df, auftvek):
    gini_ind = []
    token = list(df['token'])
    for t in range(len(token)):
        K_count = list(df['count K'])[t]
        NK_count = list(df['count NK'])[t]
        gini_ind.append(giniCalc(token[t], [K_count, NK_count], auftvek))
    return gini_ind

def getGini2(df, auftvek):
    gini_ind = []
    token = list(df['token'])
    for t in range(len(token)):
        B_count = list(df['count B'])[t]
        F_count = list(df['count F'])[t]
        K_count = list(df['count K'])[t]
        R_count = list(df['count R'])[t]
        T_count = list(df['count T'])[t]
        gini_ind.append(giniCalc(token[t], [B_count, F_count, K_count, R_count, T_count], auftvek))
    return gini_ind


### ===========================================================================
### Klassifikation (für 5_Klassifikation)
### ===========================================================================

def createWC(txt, sw, name, mw = 200):
    '''
     Inputs: txt - 
             sw - zu verwendende Stoppwörter
             name - filename
    
    Output: png Bild mit Wordcloud
    
    

    '''
    cloud = WordCloud(background_color = "white", stopwords = sw, max_words = mw)
    cloud.generate(txt)
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Wordclouds')
    cloud.to_file(name)

def getHäufigeWörter(txt, nummer = 200):
         
    word_cloud_dict = Counter(' '.join(txt).split())
    bigrams = [' '.join(b) for l in [' '.join(txt)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    word_cloud_dict_bi = Counter(bigrams)

    haufig = word_cloud_dict.most_common(nummer)
    haufig_bi = word_cloud_dict_bi.most_common(nummer)
    
    mcwords = []
    mcwords_bi = []
    for i in range(nummer):
        mcwords.append(haufig[i][0])
        mcwords_bi.append(haufig_bi[i][0])
        
        haufig_sl = set(mcwords).difference(stop_words_stem)
    
    return(haufig_sl, mcwords_bi)

def getHäufigeWörtermitSW(txt, nummerw, nummerbi):
    word_cloud_dict = Counter(' '.join(txt).split())
    bigrams = [' '.join(b) for l in [' '.join(txt)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    word_cloud_dict_bi = Counter(bigrams)

    haufig = word_cloud_dict.most_common(nummerw)
    haufig_bi = word_cloud_dict_bi.most_common(nummerbi)
    return(haufig, haufig_bi)
    
    
def getAnzahlen(vglwortvek, Klasse1, Klasse2):
    inKl1 = []
    for i in vglwortvek:
        count = 0
        for j in range(len(Klasse1)):
            if i in Klasse1[j].split():
                count += 1
            elif i in [' '.join(b) for l in [' '.join(Klasse1[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]:
                count += 1
        inKl1.append(count)
    
    inKl2 = []
    for i in vglwortvek:
        count = 0
        for j in range(len(Klasse2)):
            if i in Klasse2[j].split():
                count += 1
            elif i in [' '.join(b) for l in [' '.join(Klasse2[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]:
                count += 1
        inKl2.append(count)
        
    summeKl1 = 0
    for j in range(len(Klasse1)):
        if any(map(lambda v: v in vglwortvek, Klasse1[j].split())):
            summeKl1 += 1
        elif any(map(lambda v: v in vglwortvek, [' '.join(b) for l in [' '.join(Klasse1[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])])):
            summeKl1 += 1
    
    summeKl2 = 0
    for j in range(len(Klasse2)):
        if any(map(lambda v: v in vglwortvek, Klasse2[j].split())):
            summeKl2 += 1
        elif any(map(lambda v: v in vglwortvek, [' '.join(b) for l in [' '.join(Klasse2[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])])):
            summeKl2 += 1
            
    return(inKl1, summeKl1, inKl2, summeKl2)



def getWerte(k_words, nk_words, Xtr, ytr, positiv = 'K', negativ = 'N'):
    ypred = []
    for i in range(len(Xtr)):
        ypred.append(gevoFinden(Xtr[i], k_words, nk_words, pos = positiv, neg = negativ))
    
    if positiv == 1:
        cm_falsch = confusion_matrix(ytr, ypred)
        cm = np.array([[1, 1], [1, 1]])
        cm[0][0] = cm_falsch[1][1]
        cm[1][1] = cm_falsch[0][0]
        cm[0][1] = cm_falsch[1][0]
        cm[1][0] = cm_falsch[0][1]
        # cm = cm_new
    else:
        cm = confusion_matrix(ytr, ypred)
    sensi = cm[0][0]/(sum(cm[0]))
    spezi = cm[1][1]/(sum(cm[1]))
    richtigkl = (cm[0][0] + cm[1][1])/len(ypred)
    return(cm, sensi, spezi, richtigkl, ypred)

def getWerteKlassi(ytr, ypred):
    '''
    für klassen 0 und 1, wobei 0 die POSITIVE KLASSE ist
    
    '''
    cm = confusion_matrix(ytr, ypred)
    cm_new = np.array([[1, 1], [1, 1]])
    cm_new[0][0] = cm[1][1]
    cm_new[1][1] = cm[0][0]
    cm_new[0][1] = cm[1][0]
    cm_new[1][0] = cm[0][1]
    cm = cm_new
    sensi = cm[0][0]/(sum(cm[0]))
    spezi = cm[1][1]/(sum(cm[1]))
    richtigkl = (cm[0][0] + cm[1][1])/len(ypred)
    return(cm, sensi, spezi, richtigkl)


def getBestCombinations(words, Xtr, ytr, gevo1, gevo2 = 'K'):
    '''
    ich will hier mit k vergleichen und so wie confusion matrix geschrieben
    ist muss k gevo2 sein, wenn der erste Buchstabe des grundes VOR K
    im alphabet kommt und sonst gevo1

    '''
    numbers = list(range(len(words)))
    combinations = []
    for r in range(len(numbers)+1):
        for combination in itertools.combinations(numbers, r):
            combinations.append(combination)

    cms = []
    frichtig = []
    kombis = []
    for i in range(len(combinations)):
        kombis.append(list(np.array(words)[list(list(combinations)[i])]))
        if gevo2 == 'R':
            [cm, a, b, c, d] = getWerte(kombis[i], [], Xtr, ytr, gevo1, gevo2)
            cm_new = np.array([[1, 1], [1, 1]])
            cm_new[0][0] = cm[1][0]
            cm_new[1][1] = cm[0][1]
            cm_new[0][1] = cm[1][1]
            cm_new[1][0] = cm[0][0]
            cm_fin = cm_new
        elif gevo2 == 'T':
            [cm, a, b, c, d] = getWerte(kombis[i], [], Xtr, ytr, gevo1, gevo2)
            cm_new = np.array([[1, 1], [1, 1]])
            cm_new[0][0] = cm[0][1]
            cm_new[1][1] = cm[1][0]
            cm_new[0][1] = cm[0][0]
            cm_new[1][0] = cm[1][1]
            cm_fin = cm_new
        else:
            [cm_fin, a, b, c, d] = getWerte(kombis[i], [], Xtr, ytr, gevo1, gevo2)
        cms.append(cm_fin)
        frichtig.append(cms[i][0][0])
        
    best_kombis = myGleich(frichtig, max(frichtig))

    np.array(kombis)[best_kombis]

    krichtig = []
    for i in best_kombis:
        krichtig.append(cms[i][1][1])
        
    best_kombis_2 = myGleich(krichtig, max(krichtig))

    erg = np.array(kombis)[list(np.array(best_kombis)[best_kombis_2])]
    return erg
# =============================================================================
# Random Forest
# =============================================================================
# TOOD
# aktuell noch nur auf K NK

# def getRF(dtm_train, dtm_test, y_train, y_test):
    
#     rfc =  RandomForestClassifier(n_estimators=100, verbose=True)

#     rf_fit = rfc.fit(dtm_train.T, y_train)

#     vorhersagen = rf_fit.predict_proba(dtm_test.T)
#     wkeiten_k = list(pd.DataFrame(vorhersagen)[0])

#     y_pred = []
#     for i in range(len(y_test)):
#         if wkeiten_k[i] < 0.5:
#             y_pred.append('N')
#         else:
#             y_pred.append('K')

#     cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
#     # cm = setColRowNames(cm, ['als NK erkannt', 'als K erkannt'], ['NK', 'K'])
#     return(cm, wkeiten_k, y_pred)

def getRF(dtm_train, dtm_test, y_train, y_test, neval, mfval):
    if mfval: # max feat auf deafault
        random.seed(1802)
        rfc = RandomForestClassifier(n_estimators= neval, verbose=True)
    else:
        random.seed(1802)
        rfc = RandomForestClassifier(n_estimators= neval, max_features = mfval, verbose=True)
    rf_fit = rfc.fit(dtm_train, y_train)

    vorhersagen = rf_fit.predict(dtm_test)
    vorhersagen_wk = rf_fit.predict_proba(dtm_test)

    return(vorhersagen_wk, vorhersagen)

def RFAll(dtm_train, dtm_test, y_train, y_test, X_test_ind, neval = 100, 
          mfval = True, threshold_RF = False):
    '''
    y_train und y_test in 1 0 kodierung!!!

    '''
    
    [RF_wkeiten, RF_pred] = getRF(dtm_train, dtm_test, y_train, y_test, neval, mfval)

    mydf = pd.DataFrame([X_test_ind, y_test, RF_wkeiten[:,1]]).T
    mydf.columns = ['Dokument', 'wahre Klasse', 'Wkeit pos']
    mydf = mydf.sort_values('Dokument')
    if threshold_RF == False:
        [cm, sensi, spezi, richtigkl] = getWerteKlassi(y_test, RF_pred)
        y_pred = RF_pred
    else:
        y_pred = getPred(RF_wkeiten[:,1], threshold_RF)
        [cm, sensi, spezi, richtigkl] = getWerteKlassi(y_test, y_pred)
    
    return(cm, sensi, spezi, richtigkl, y_pred, mydf)



def ergtable(y_test, y_pred, true_wkeiten):
    '''
    gibt eine übersichtliche Ergebnistafel für den RF
    
    Inputs:     y_test
                y_pred
                true_wkeiten (teilw aus rfBerechnen)
    
    Outputs:    erg - 

    '''
    erg = pd.DataFrame()
    erg['wahrer GeVo'] = y_test
    # erg['Wkeit dafür'] = true_wkeiten
    erg['prog. GeVo'] = y_pred
    erg['richtig'] = [0] * len(y_test)
    erg['richtig'][myVektorGleich(y_test,y_pred)] = 1
    return(erg)

def getPred(wkeiten, threshold):
    y_pred = []
    for i in range(len(wkeiten)):
        if wkeiten[i] >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return(y_pred)

# =============================================================================
# SVM
# =============================================================================
def getSVM(dtm_train, dtm_test, y_train, y_test):
    '''
    
    Inputs:     dtm_train
                dtm_test
                y_train (1 0 kodiert)
                y_test (1 0 kodiert)
                
    Outputs:    result
                cm

    '''
    warnings.filterwarnings('ignore')
    random.seed(1802)
    model = svm.SVC(C=1, kernel='rbf', gamma='scale', probability = True)
  
    model.fit(dtm_train, y_train)
    
    # Make prediction
    prediction = model.predict(dtm_test)
    prediction_proba = model.predict_proba(dtm_test)
    pred = prediction.tolist()
    pred_proba = pd.DataFrame(prediction_proba.tolist())

    
    return(pred, pred_proba)

def SVMAll(dtm_train, dtm_test, y_train, y_test, X_test_ind, threshold_SVM = False):
    '''
    y_train und y_test in 1 0 kodierung!!!

    '''
    
    [SVM_pred, SVM_wkeiten] = getSVM(dtm_train, dtm_test, y_train, y_test)

    mydf = pd.DataFrame([X_test_ind, y_test, SVM_wkeiten[1]]).T
    mydf.columns = ['Dokument', 'wahre Klasse', 'Wkeit pos']
    mydf = mydf.sort_values('Dokument')

    [cm, sensi, spezi, richtigkl] = getWerteKlassi(y_test, SVM_pred)
    if threshold_SVM == False:
        [cm, sensi, spezi, richtigkl] = getWerteKlassi(y_test, SVM_pred)
        y_pred = SVM_pred
    else:
        y_pred = getPred(SVM_wkeiten[1], threshold_SVM)
        [cm, sensi, spezi, richtigkl] = getWerteKlassi(y_test, y_pred)

    return(cm, sensi, spezi, richtigkl, y_pred, mydf)

# def getSVM(dtm_train, dtm_test, y_train, y_test):
#     '''
    
#     Inputs:     dtm_train
#                 dtm_test
#                 y_train (1 0 kodiert)
#                 y_test (1 0 kodiert)
                
#     Outputs:    result
#                 cm

#     '''
#     warnings.filterwarnings('ignore')
#     model = svm.SVC(C=1, kernel='rbf', gamma='scale', probability = True)
  
#     model.fit(dtm_train, y_train)
    
#     # Make prediction
#     prediction = model.predict(dtm_test)
#     prediction_proba = model.predict_proba(dtm_test)
#     pred = prediction.tolist()
#     pred_proba = pd.DataFrame(prediction_proba.tolist())

#     cm = confusion_matrix(y_test, pred)
#     # cm = setColRowNames(cm, ['als K erkannt', 'als NK erkannt'], ['K', 'NK'])

#     return(cm, pred, pred_proba)




# def getSVM(dtm_train, dtm_test, y_train, y_test):
#     '''
    
#     Inputs:     dtm_train
#                 dtm_test
#                 y_train (1 0 kodiert)
#                 y_test (1 0 kodiert)
                
#     Outputs:    result
#                 cm

#     '''
#     warnings.filterwarnings('ignore')
#     model = svm.SVC(C=1, kernel='rbf', gamma='scale', probability = True)
  
#     model.fit(dtm_train, y_train)
    
#     # Make prediction
#     prediction = model.predict(dtm_test)
#     prediction_proba = model.predict_proba(dtm_test)
#     pred = prediction.tolist()
#     pred_proba = pd.DataFrame(prediction_proba.tolist())

#     cm = confusion_matrix(y_test, pred)
#     # cm = setColRowNames(cm, ['als K erkannt', 'als NK erkannt'], ['K', 'NK'])

#     return(cm, pred, pred_proba)

# def getSVM(X_train, X_test, y_train, y_test):
#     '''
    
#     Inputs:     X_train
#                 X_test
#                 y_train
#                 y_test
                
#     Outputs:    result
#                 cm

#     '''
#     warnings.filterwarnings('ignore')
#     model = svm.SVC(C=1, kernel='rbf', gamma=1)
#     model.fit(X_train, y_train)
    
#     # Make prediction
#     prediction = model.predict(X_test)
    
#     # Get results
#     result = X_test
#     result['contraceptive'] = y_test
#     result['prediction'] = prediction.tolist()
#     result.head()
    
#     cm = confusion_matrix(result['contraceptive'], result['prediction'])

#     return(result, cm)


# =============================================================================
# Extratrees
# =============================================================================

def getET(X_train, X_test, y_train, y_test):
    '''
    
    Inputs:     X_train
                X_test
                y_train
                y_test
                
    Outputs:    ypred
                cm

    '''
    
    clf = ExtraTreesClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)

    # cv_scores = cross_val_score(clf, X_train, y_train, cv=5 )
    # print("CV average score: %.2f" % cv_scores.mean())

    y_pred = clf.predict(X_test)

    cm_ET = confusion_matrix(y_test, y_pred)
    return(y_pred, cm_ET)

# =============================================================================
# Plotfunktionen
# =============================================================================


def ROCFunc(y_test, y_pred, name, xlab = "1 - Spezifität", ylab = "Sensitivität"):
    '''
    berechnet ROC Kurve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                Grafik

    '''
    auc = metrics.roc_auc_score(y_test, y_pred)
    false_positive_rate, true_positive_rate,\
        thresolds = metrics.roc_curve(y_test, y_pred)
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(4, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, 
                      facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=10, 
              weight='bold', color='blue')
    
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()
    return(auc)


def HistFuncK(daten, name, xlab = "Wahrscheinlichkeit für eine Kündigung", ylab = "Anzahl an Kündigungen"):
    '''
    berechnet ROC Kurve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                Grafik

    '''
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    plt.xlim([0, 1])
    plt.ylim([0, 4000])
    
    plt.hist(daten, color = 'lightgreen')
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()

def HistFuncNK(daten, name, xlab = "Wahrscheinlichkeit für eine Kündigung", ylab = "Anzahl an anderen Dokumenten"):
    '''
    berechnet ROC Kurve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                Grafik

    '''
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    plt.xlim([0, 1])
    plt.ylim([0, 8000])
    
    plt.hist(daten, color = 'lightblue')
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()

def GiniFunc(daten, name, xlab = "Token, sortiert nach Gini - Index", ylab = "Gini - Index"):
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    #plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.plot(np.sort(daten))#, color = 'lightgreen')
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()


def AnteileFunc(daten, name, xlab = "Token, sortiert nach Differenzen", ylab = "Differenz"):
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    #plt.xlim([0, 1])
    plt.ylim([-1, 1])
    
    plt.plot(np.sort(daten))#, color = 'lightgreen')
    os.chdir(r'W:\Sonder\lva-93300\Schriftstücke\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()



# =============================================================================
# komplettfunc für RF und SVM (muss nur noch ein random.seed() vor)
# =============================================================================
# klgr TRUE: alle daten
# klgr FALSE: kleine matrix
def RFcomplete(grund, dtm_train, dtm_test, y_train, y_test, X_train_ind, X_test_ind, klgr = True):
    [cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
     mydf_train] = RFAll(dtm_train, dtm_train, y_train, y_train, X_train_ind)
    if klgr == True:
        name1 = grund + 'hist_K_RF_train.png'   
        name2 = grund + 'hist_NK_RF_train.png'  
        name3 = grund + 'roc_RF_train.png'
        name4 = grund + 'hist_K_RF_test.png'   
        name5 = grund + 'hist_NK_RF_test.png' 
        name6 = grund + 'roc_RF_test.png'
    else:
        name1 = grund + 'hist_K_RF_train_kl.png'   
        name2 = grund + 'hist_NK_RF_train_kl.png'  
        name3 = grund + 'roc_RF_train_kl.png'
        name4 = grund + 'hist_K_RF_test_kl.png'   
        name5 = grund + 'hist_NK_RF_test_kl.png' 
        name6 = grund + 'roc_RF_test_kl.png'
              
    wkeiten_pos = mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 1]
    HistFuncK(wkeiten_pos, name1)
    wkeiten_neg = np.sort(mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 0])
    HistFuncNK(wkeiten_neg, name2)
    
    ROCFunc(y_train, mydf_train['Wkeit pos'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['Wkeit pos'])).T
    th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
    th['spezi'] = 1 - th['false_positive rate']
    th['sensi + spezi'] = th['spezi'] + th['true_positive_rate']
    
    ts = list(th['threshold'][th['sensi + spezi'] == max(th['sensi + spezi'])])[0]                  

    if min(wkeiten_pos) > max(wkeiten_neg):
        random.seed(1802)
        ts = random.uniform(max(wkeiten_neg), min(wkeiten_pos))
    
    y_pred_train_new = getPred(mydf_train['Wkeit pos'], ts)
    [cm_train_new, sensi_train_new, spezi_train_new,
     richtigkl_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test, 
     mydf_test] = RFAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_RF = ts)
                           
    wkeiten_pos = mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 1]
    HistFuncK(wkeiten_pos, name4)
    wkeiten_neg = np.sort(mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 0])
    HistFuncNK(wkeiten_neg, name5)
    
    ROCFunc(y_test, mydf_test['Wkeit pos'], name6) 

    return(cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
           mydf_train, ts,
           cm_train_new, sensi_train_new, spezi_train_new, richtigkl_train_new,
           y_pred_train_new,
           cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test,
           mydf_test)  

def SVMcomplete(grund, dtm_train, dtm_test, y_train, y_test, X_train_ind, X_test_ind, klgr = True):
    [cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
     mydf_train] = SVMAll(dtm_train, dtm_train, y_train, y_train, X_train_ind)
    if klgr == True:
        name1 = grund + '_hist_K_SVM_train.png'   
        name2 = grund + '_hist_NK_SVM_train.png'  
        name3 = grund + '_roc_SVM_train.png'
        name4 = grund + '_hist_K_SVM_test.png'   
        name5 = grund + '_hist_NK_SVM_test.png' 
        name6 = grund + '_roc_SVM_test.png'
    else:
        name1 = grund + '_hist_K_SVM_train_kl.png'   
        name2 = grund + '_hist_NK_SVM_train_kl.png'  
        name3 = grund + '_roc_SVM_train_kl.png'
        name4 = grund + '_hist_K_SVM_test_kl.png'   
        name5 = grund + '_hist_NK_SVM_test_kl.png' 
        name6 = grund + '_roc_SVM_test_kl.png'
              
    wkeiten_pos = mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 1]
    HistFuncK(wkeiten_pos, name1)
    wkeiten_neg = np.sort(mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 0])
    HistFuncNK(wkeiten_neg, name2)
    
    ROCFunc(y_train, mydf_train['Wkeit pos'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['Wkeit pos'])).T
    th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
    th['spezi'] = 1 - th['false_positive rate']
    th['sensi + spezi'] = th['spezi'] + th['true_positive_rate']
    
    ts = list(th['threshold'][th['sensi + spezi'] == max(th['sensi + spezi'])])[0]                  

    if min(wkeiten_pos) > max(wkeiten_neg):
        random.seed(1802)
        ts = random.uniform(max(wkeiten_neg), min(wkeiten_pos))
    
    y_pred_train_new = getPred(mydf_train['Wkeit pos'], ts)
    [cm_train_new, sensi_train_new, spezi_train_new,
     richtigkl_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test, 
     mydf_test] = SVMAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_SVM = ts)
                           
    wkeiten_pos = mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 1]
    HistFuncK(wkeiten_pos, name4)
    wkeiten_neg = np.sort(mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 0])
    HistFuncNK(wkeiten_neg, name5)
    
    ROCFunc(y_test, mydf_test['Wkeit pos'], name6) 

    return(cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
           mydf_train, ts,
           cm_train_new, sensi_train_new, spezi_train_new, richtigkl_train_new,
           y_pred_train_new,
           cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test,
           mydf_test)  

# def SVMcomplete(dtm_train, dtm_test, y_train, y_test, X_train_ind, X_test_ind):
#     [cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
#      mydf_train] = SVMAll(dtm_train, dtm_train, y_train, y_train, X_train_ind)
                         
#     wkeiten_pos = mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 1]
#     hist_k_train = plt.hist(wkeiten_pos)
#     wkeiten_neg = np.sort(mydf_train['Wkeit pos'][mydf_train['wahre Klasse'] == 0])
#     hist_nk_train = plt.hist(wkeiten_neg)
    
#     roc_train = ROCFunc(y_train, mydf_train['Wkeit pos'])                   

#     th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['Wkeit pos'])).T
#     th.columns = ['false_positive rate', 'true_positive_rate', 'threshold']
#     th['spezi'] = 1 - th['false_positive rate']
#     th['sensi + spezi'] = th['spezi'] + th['true_positive_rate']
    
#     ts = list(th['threshold'][th['sensi + spezi'] == max(th['sensi + spezi'])])[0]
    
#     if min(wkeiten_pos) > max(wkeiten_neg):
#         random.seed(1802)
#         ts = random.uniform(max(wkeiten_neg), min(wkeiten_pos))
    
    
#     y_pred_train_new = getPred(mydf_train['Wkeit pos'], ts)
#     [cm_train_new, sensi_train_new, spezi_train_new,
#      richtigkl_train_new] = getWerteKlassi(y_train, y_pred_train_new)

#     [cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test, 
#      mydf_test] = SVMAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
#                            threshold_SVM = ts)
                          
#     hist_k_test = plt.hist(mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 1])
#     hist_nk_test = plt.hist(np.sort(mydf_test['Wkeit pos'][mydf_test['wahre Klasse'] == 0]))
     
#     roc_test = ROCFunc(y_test, mydf_test['Wkeit pos']) 

#     return(cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
#            mydf_train, hist_k_train, hist_nk_train, roc_train, ts,
#            cm_train_new, sensi_train_new, spezi_train_new, richtigkl_train_new,
#            y_pred_train_new,
#            cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test,
#            mydf_test, hist_k_test, hist_nk_test, roc_test)                  
    
    
def getEineKlass(x_true, fpr, bpr, rpr, tpr, kpr):
    f_x = list(set(fpr).intersection(x_true))
    b_x = list(set(bpr).intersection(x_true))
    r_x = list(set(rpr).intersection(x_true))
    t_x = list(set(tpr).intersection(x_true))
    k_x = list(set(kpr).intersection(x_true))
    all_x = [len(f_x), len(b_x), len(r_x), len(t_x), len(k_x)]
    return(all_x)

def getErgebnisse(y_pred_F, y_pred_B, y_pred_R, y_pred_T, X_test, y_test):

    f_pred = myGleich(y_pred_F, 1) + myGleich(y_pred_F, 'F')
    b_pred = myGleich(y_pred_B, 1) + myGleich(y_pred_B, 'B')
    r_pred = myGleich(y_pred_R, 1) + myGleich(y_pred_R, 'R')
    t_pred = myGleich(y_pred_T, 1) + myGleich(y_pred_T, 'T')
    k_pred = list(set(range(len(X_test))).difference(f_pred + b_pred + r_pred + t_pred))   
    f_true = myGleich(y_test, 'F') 
    b_true = myGleich(y_test, 'B') 
    r_true = myGleich(y_test, 'R') 
    t_true = myGleich(y_test, 'T') 
    k_true = myGleich(y_test, 'K') + myGleich(y_test, 'S')
    
    ergdf = pd.DataFrame([[0] * 5] * 5)
    ergdf = setColRowNames(ergdf, 
                            ['F wahr', 'B wahr', 'R wahr', 'T wahr', 'K wahr'],
                            ['F klass', 'B klass', 'R klass', 'T klass', 'K klass'])
    
    ergdf['F wahr'] = getEineKlass(f_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['B wahr'] = getEineKlass(b_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['R wahr'] = getEineKlass(r_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['T wahr'] = getEineKlass(t_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['K wahr'] = getEineKlass(k_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    
    ergdf = ergdf.T
    return(ergdf)
    
# =============================================================================
# damit dill funktioniert
# =============================================================================

def delStoppw():
    return 0

def delWords():
    return 0
def delNumbersNames():
    return 0

def delSeltene():
    return 0

def countFunc():
    return 0

def giniCalcAll():
    return 0

def rfBerechnen():
    return 0

def gevoVergleich():
    return 0

def kriegGevo():
    return 0

def getWC():
    return 0

def getAllWC():
    return 0
