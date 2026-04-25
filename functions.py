# =============================================================================
# Loading packages
# =============================================================================

import pandas as pd # e.g. for the tf-idf matrix
import numpy as np
import random
import os # for the working directories
import pytesseract as py # for OCR
py.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pikepdf # for saving PDF files as unprotected files
from pdf2image import convert_from_path #, convert_from_bytes
# import dill
from datetime import datetime
from Levenshtein import distance as lev
from collections import defaultdict # for the frequencies of the GeVos
import collections as cl # also for frequencies
import xlsxwriter # to write into Excel tables
from wordcloud import WordCloud # for word clouds
import itertools
from collections import Counter
import snowballstemmer # for stemming
from stop_words import get_stop_words # for German stop words
from sklearn.ensemble import RandomForestClassifier # for Random Forest
from sklearn.metrics import confusion_matrix
from sklearn import svm # for SVM
import warnings # for SVM 
from sklearn.ensemble import ExtraTreesClassifier # ET
from sklearn import metrics # ROC 
import matplotlib.pyplot as plt # ROC
import matplotlib

# =============================================================================
# set random.seed, in case it is needed
# =============================================================================

random.seed(1921)

# =============================================================================
# =============================================================================
# # read PDFs + OCR (for 1_convert_PDFs and 2_read_PDFs)
# =============================================================================
# =============================================================================

# =============================================================================
# saving protected PDFs as unprotected
# =============================================================================

def pdfUmspeichern(file, datafolder, pdffolder):
    '''
    saves PDF "file" as an unprotected PDF
    
    Inputs:     file - protected PDF file
                datafolder - folder where the protected PDF is located
                pdffolder - target folder for the unprotected PDF
    
    Outputs:    no output in Python, unprotected PDF is saved in pdffolder
    '''
    os.chdir(datafolder)
    pdf = pikepdf.open(file, allow_overwriting_input=True)
    os.chdir(pdffolder)
    pdf.save(file)
    

def vielUmspeichern(datafolder, pdffolder):
    '''
    saves all protected PDFs from datafolder as unprotected PDFs in 
    pdffolder
    
    Inputs:     datafolder - folder containing the protected PDFs
                pdffolder - target folder for the unprotected PDFs
    
    Outputs:    no output in Python, unprotected PDFs are saved in pdffolder
    '''
    for i in next(os.walk(datafolder))[2]:
        pdfUmspeichern(i, datafolder, pdffolder)
             
def teilUmspeichern(separation, path0, path1, path2):
    '''
    saves the data from path0 into path1 and path2 based on a split

    Inputs:     separation - vector with names of documents that should go to path1
                path0 - original file path
                path1 - target path for selected documents
                path2 - target path for non-selected documents
            
    Outputs:    none, files are saved in the respective folders
    '''
    for i in separation:
        pdfUmspeichern(i, path0, path1)
      
    ges = []
    for i in os.walk(path0):
        ges.append(i)
    
    for i in ges[0][2]:
        if i not in separation:
            pdfUmspeichern(i, path0, path2)

        
# =============================================================================
# converting PDFs to images 
# =============================================================================
        
def getBild(file, pdffolder):
    '''
    saves an unprotected PDF as a ppm file

    Inputs:     file - unprotected PDF file 
                pdffolder - folder where the file is stored
    
    Outputs:    no output in Python, ppm file is saved in pdffolder
                
    '''
    os.chdir(pdffolder)
    im = convert_from_path(file, 
                        poppler_path = r'C:\ProgramData\Anaconda3\pkgs\poppler-22.01.0-h24fffdf_2\Library\bin')
    return im
    
 
def getBilder(pdffolder):
    '''
    applies getBild to each file in pdffolder
    
    Inputs:      pdffolder - folder with files to be converted
    
    Outputs:    the images
    '''
    pictures = []
    for i in next(os.walk(pdffolder))[2]:
        pictures.append(getBild(i, pdffolder))
    return pictures   

# =============================================================================
# extracting text from the images 
# =============================================================================


def bildAuslesen(s):
    '''
    extracts text from each image in s
    
    Inputs:     s - list of images
    
    Outputs:     s_text - list of extracted texts from the images
    '''
    s_text = []
    for j in range(len(s)):
        text = []
        for i in s[j]:
            text.append(py.image_to_string(i, lang = 'deu'))
        s_text.append(text)
    return s_text


def getText(pdffolder):
    '''
    extracts text from files (combination of getBilder and
                                bildAuslesen)

    Inputs:     pdffolder - folder with unprotected PDF files
    
    Outputs:    txt - list of extracted texts from the files

    '''
    pictures = getBilder(pdffolder)
    txt = bildAuslesen(bilder)
    return txt


def ordnerEinlesen(pdffolder):
    '''
    reads the texts of a folder and returns how long it takes
    
    Inputs:     pdffolder - folder with unprotected PDF files
    
    Outputs:    txt - list of extracted texts from the files
                dauer - time taken to execute this function

    '''
    start = datetime.now()
    txt = getText(pdffolder)
    end = datetime.now()
    duration = end - start
    return txt, duration

def seitenZusammen(texts):
    '''
    merges multiple pages
    
    Inputs:      texts -     list of lists, each list contains a document,
                            can also include multi-page elements
                    
    Outputs:     new_text -  list of individual documents, no longer split
                            into pages
    '''
    new_text = []
    for i in range(len(texts)):
        new_text.append(' '.join(texts[i]))
    return new_text

# =============================================================================
# find duplicates
# =============================================================================

def findDuplicates(vector):
    '''
    this function returns how often each element of the vector occurs
    
    Inputs:     vector - vector of interest
    
    Outputs:    amounts - frequencies of occurrence of each entry 
                           (e.g. if vector: (1, 2, 1, 3, 3), then
                            amounts: (2, 1, 2, 2, 2))
    '''
    amounts = [0] * len(vector)
    for i in range(len(vector)):
        amounts[i] = vector.count(vector[i])
    return(amounts)


def createDuplDict(txt):
    '''
    
    
    Inputs:     txt
    
    Outputs:    occurs

    '''
    dupl_vect = findDuplicates(txt)
    dupl_amount = []
    i = 1
    while sum(dupl_amount) < len(txt):
        dupl_amount.append(len(myGleich(dupl_vect, i)))
        i += 1
    occurs = dict(zip(range(1, i + 1), dupl_amount))
    return(occurs)

# =============================================================================
# =============================================================================
# # Text preprocessing (for 3_table_creation)
# =============================================================================
# =============================================================================

# Levenshtein function
# This is intended to ensure that when comparing two names or similar,
# the Levenshtein distance is calculated, so that one can say, for example,
# that with an LD of 1 or so it is probably the same word, just that
# something went wrong during reading, which will definitely happen.
# Then, for example, one could always take the first as the correct one if
# the LD is small enough to assume they are the same, because that one is
# probably written correctly, the second might often be copied from the first,
# written by hand, maybe the first one is the neatest, etc.


# =============================================================================
# extract information from document
# =============================================================================

# names, VNR, business transaction, corona indicator

# difference to vector.vergleich(): also returns multiple results!
def myGleich(vector, comparison):
    # checks
    '''
    checks whether the word "comparison" occurs in "vector" and where
    
    Inputs:  vector 
            comparison - value to compare against
            
    Ouput:  res - indices where the word occurs or "1" if the word does not 
                    occur in the vector
    '''
    res = [i for i in range(len(vector)) if vector[i] == comparison]
    return res 


def myvectorGleich(vector1, vector2):
    '''
    checks for which i vector1[i] = vector2[i]
    
    Inputs:     vector1
                vector2
                
    Outputs:    res - indices where vector1[i] = vector2[i]
    '''
    res = [i for i in range(len(vector1)) if vector1[i] == vector2[i]]
    return res    

    
def searchIndizes(document, token):
    # checks
    '''
    checks whether the searched token occurs in the document and returns the
    indices where it appears, e.g. to find words like "von:", "grüße", etc.
    
    Inputs:     document - the NON-tokenized document
                token - the token to search for
            
    Ouputs:     res - indices where the word occurs or "1" if the word does not 
                      occur in the vector
    '''
    vector = document.split()
    res = myGleich(vector, token)
    return(res)


def exWort(indices):   
    '''
    returns TRUE if the word occurs and FALSE otherwise
    
    Inputs:     indices - output of searchIndizes
    
    Outputs:    TRUE/ FALSE
    '''
    return bool(type(indices) is list)
    


def naechsteZeichen(document, tok, n):
   '''
   finds the next n tokens after a given token
    
   Inputs:      document - the NON-tokenized document
                tok - token (WORD)
                n - number of subsequent tokens 
              
   Outputs:     A - matrix with the corresponding tokens
                   special case: end of document: []
   '''
   document_tok = document.split()
   tokenstarter = searchIndizes(document, tok)
   # create A
   A = []
   for i in range(len(tokenstarter)): 
       A.append([])
       for j in range(n):
           A[i].append('')
   # fill A
   for i in range(len(tokenstarter)):
        for j in range(n):
            if len(document_tok) > tokenstarter[i] + j + 1:
                A[i][j] = document_tok[int(tokenstarter[i]) + j + 1]
            else:
                A[i][j] = []
   return A


def findeFunc(document, tokenvector, n):
    '''
    finds the next n tokens after the tokens in tokenvector
    
    Inputs:     document - the NON-tokenized document 
                tokenvector - vector with possible tokens 
                n - number of tokens after the specified one to consider
              
    Outputs:    poss_token - the corresponding tokens
    '''
    count = -1
    names = []
    poss_token = []
    
    # if only one token is searched
    if isinstance(tokenvector, str):
        tk = searchIndizes(document, tokenvector)
        if exWort(tk):
            res = naechsteZeichen(document, tokenvector, n)
            poss_token += res
            
    # if a vector is searched        
    else:
        for i in range(len(tokenvector)):
            count += 1
            token = tokenvector[i]
            tk = searchIndizes(document, token)
            if exWort(tk):
               locals()['token_%s' % count] = naechsteZeichen(document, token, n)
               names += [count]
        for i in names:
            poss_token += locals()['token_%s' % i]
    return poss_token

def levenMatrix(vector):
    '''
    computes the Levenshtein distance between all elements
    of a list
    
    Inputs:     vector (the list)
    
    Outputs:    LM - the matrix

    '''
    # if of the form [['vor' 'zu'], ['vor2' 'zu2'], ['vor3' 'zu3']]
    # (which should be the case, at least for name extraction)
    # convert to ['vor zu', 'vor2 zu2', 'vor3 zu3']
    if len(vector[0][0]) > 1:
        newvec = []
        for i in range(len(vector)):
           neuvek.append(' '.join(vector[i]))
        vector = newvec
    n = len(vector)
    # initializes zeros on the diagonal
    LM = [0] * n
    for x in range(n):
        LM[x] = [0] * n
    for i in range(n):
        for j in range(n):
            if j > i:
                LM[i][j] = lev(vector[i], vector[j])
            else: # mirrored matrix
                LM[i][j] = LM[j][i]
                
    LM = pd.DataFrame(LM)
    LM.columns = vector
    return(LM)

def findEinzelToken(possibilities):
    '''
    searches tokens individually (not as full "word blocks") for duplicates
    
    Inputs:     possibilities - vector with possible names, VNRs, etc.
    
    Outputs:    Res - matrix with the number of words that occur more often
                      in each combination of word blocks

    '''
    # works only for elements with at least two entries
    z = len(possibilities)
    n = len(possibilities[0])
    Res = pd.DataFrame(z * [z * [0]])
    for i in range(z): # over all word blocks
        for j in range(n): # extract individual token
            compword = possibilities[i][j]
            for k in range(z):
                count = 0
                for l in range(n):
                    if (compword == possibilities[k][l]):
                        count += 1
                Res[i][k] += count
    return(Res)


### this function should output all matches that are not on the
### main diagonal, and all partial matches with the corresponding
### matching words

def countToken(possibilities):
    '''
    counts how often each distinct token occurs in the vector mglkeiten
    
    Inputs:     possibilities - vector with possible names, VNRs, etc., whose
                            frequencies should be determined
                            
    Outputs:    counter -   object containing how often each token occurs,
                            initially without considering Levenshtein distance,
                            so very similar ones are counted separately
    '''
    if ([possibilities[0]] == possibilities):
        return(cl.Counter(possibilities[0]))
    else:
        all = []
        for i in range(len(possibilities)):
            for j in range(len(possibilities[i])):
                all.append(possibilities[i][j])
                
        counter = cl.Counter(all)
        
        return(counter)

### VNR =======================================================================

# This is done first, because the characters are still important here.
# After this, they are removed (for names, GeVo, etc.)

def vnrFinden(schreiben):
    '''
    searches typical positions for insurance numbers
    
    Inputs:     schreiben
    
    Outputs:    possible VNRs
    '''
    vnr_token = ['Versicherungsnummer', 'VersicherungsNr.', 'Nr.', 'Nr:', 
                 'Nr.:', 'Nr', 'VNR', 'VNR:', 'Versicherungsnummer:', 
                 'VSNR:', 'VSNR', 'VSNR.', 'VSNR,','VSNR.:']
    
    return findeFunc(schreiben, vnr_token, 2)


def vnrBauen(schreiben):
    '''
    tries to construct the VNR and stops when it ends
    
    Inputs:     schreiben
    
    Outputs:    possible VNRs
    '''
    mglkeiten = vnrFinden(schreiben)
    vgl_vector = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    neu_mgl = mglkeiten # keep the same structure as mglkeiten
    # loop over the different occurrences of a VNR
    for k in range(len(mglkeiten)):
        # loop over the different tokens that are extracted
        for j in range(len(mglkeiten[k])): # always the same
            # possibly loop over the first two elements to check whether
            # it is actually a number?
            a_mgl = neu_mgl[k][j]
            if len(a_mgl) > 0:
                for i in range(len(a_mgl)):
                    # check if something occurs that is not a digit or similar
                    if a_mgl[i] not in vgl_vector:
                            a_mgl = a_mgl.replace(a_mgl[i], 'X')
            neu_mgl[k][j] = a_mgl
    # delete from where it starts with 2 X somewhere & if it is empty
    str_zsm = [None] * len(mglkeiten)
    for k in range(len(mglkeiten)):
        for j in range(len(mglkeiten[k])): 
                if 'X' in neu_mgl[k][j] or neu_mgl[k][j] == []:
                    neu_mgl[k][j] = ''
        # concatenate what remains for each VNR
        str_zsm[k] = "".join(neu_mgl[k])       
    return str_zsm



def searchZahlenzeichenketten(schreiben):
    '''
    finds numeric character strings (with the characters and digits from vgl_vector)
    in a document
    
    Inputs:      schreiben - non-tokenized document
    
    Outputs:    zahlentoken - the numeric tokens that occur in the document
                numbind - the indices of these numeric tokens

    '''
    vgl_vector = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    schreibsplit = schreiben.split()
    schreibsplit
    isdig = []
    for i in range(len(schreibsplit)):
        # isdig.append(schreibsplit[i].isdigit())
        token = schreibsplit[i]
        for j in range(len(token)):
            if (token[j] not in vgl_vector):
                isdig.append(False)
                break;
            elif (j == (len(token) - 1)):
                isdig.append(True)
    numbind = myGleich(isdig, True)
    zahlentoken = np.array(schreibsplit)[numbind]
    return(zahlentoken, numbind)


# TODO
# check again what happens here
def searchVNR(schreiben):
    '''
    finds VNRs by combining numeric tokens from 
    searchZahlenzeichenketten
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    zt - 

    '''
    zahlentoken, numbind = searchZahlenzeichenketten(schreiben)
    isnext = []
    for i in range(len(numbind) - 1):
        if (numbind[i] + 1 == numbind[i + 1]):
            isnext.append(True)
        else:
            isnext.append(False)
    # combine those that are directly consecutive
    zahlentoken = list(zahlentoken)
    zt = zahlentoken
    for i in range(len(numbind) - 1):
        if (isnext[i] and not isnext[i + 1]):
            # TODO 
            # modify here so that index shifts caused by
            # changes to zt within one loop iteration are
            # considered in the next; this is currently not the case
            zt[i:(i + 2)] = [''.join(zahlentoken[i:(i + 2)])]
    return(zt)

    
# TODO - additions needed: include searchVNR, Levenshtein, frequency
# compares the VNRs
def vnrVergleich(schreiben):
    # checks
    '''
    compares the VNRs 
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    ideally: VNR, otherwise currently (TO CHANGE!!) 'unclear VNR' and
                'no VNR found' (prints that, stores None for the latter cases)
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
            # currently just return the first one with 'ev' in front,
            # which is not necessarily the optimal solution
    except:
        # TODO
        # search for VNR after the name if none found in the text (header)
        # maybe simply search for long numbers?
        return 'none found'


### remove punctuation =====================================================

def delPunct(schreiben):
    # checks
    '''
    removes punctuation from the document
    
    Inputs:     schreiben - NON-tokenized document
    
    Outputs:    document without punctuation

    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»‚‚')
    token_punctless =[token for token in schreiben if token not in punctuation]
    token_punctless = ''.join(token_punctless)
    return token_punctless

def delPuncts(dok_list):
    # checks
    '''
    removes punctuation from all documents in dok_list
    
    Inputs:     dok_list - list of non-tokenized documents
    
    Outputs:    dok_list without punctuation
    '''
    new_list = []
    for i in dok_list:
        new_list.append(delPunct(i))
    return new_list

### convert all letters to lowercase =========================================

def machLowercase(dok_list):
    # checks
    '''
    converts all letters to lowercase
    
    Inputs:     dok_list - list of non-tokenized documents
    
    Outputs:    new_list - list of non-tokenized documents, completely
                           in lowercase
    '''
    new_list = []
    for i in dok_list:
        new_list.append(i.lower())
    return new_list

### Name ======================================================================

def namenFinden(schreiben):
    '''
    searches typical positions for names
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    mgl_namen - possible names
    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»')
    numbers = list('0123456789')
    schreiben_lower = delPunct(schreiben).lower()
    namen_token = ['grüßen', 'grüße', 'gruß', 'von', 'from']
    schreiben_tok = schreiben_lower.split()
    # since the name is often at the beginning of the letterhead:
    stelle12 = schreiben_tok[0:2]
    m = 2
    mgl_namen = findeFunc(schreiben_lower, namen_token, m)
    mgl_namen.append(stelle12)
    todel = []
    # loop over all possible names to remove email addresses and numbers
    nein_words = ['von', 'signal', 'iduna', 'signaliduna', 'kundenservice',
                  'kundenberater', 'kundenberaterin', 'ihnen', 'der', 
                  'die', 'das', 'dem']
    for i in range(len(mgl_namen)):
        for j in range(m): 
            if mgl_namen[i][j] in nein_words:
                todel.append(i)
                break            
        # otherwise consider both parts separately and check individual characters
        else: 
            for j in range(m):
                part = list(mgl_namen[i][j])
                if any(map(lambda v: v in punctuation, part)):
                    todel.append(i)
                    break
                elif any(map(lambda v: v in numbers, part)):
                    todel.append(i)
                    break
                # to filter out email addresses containing "signaliduna"
                elif all(map(lambda v: v in part, list('signaliduna'))):
                    todel.append(i)
                    break
    j = 0
    for i in range(len(todel)):
        del(mgl_namen[todel[i] - j])
        j = j + 1
    return mgl_namen

# TODO
# make a smarter selection here, include Levenshtein etc. (probably not enough yet)
def nameVergleich(schreiben):
    '''
    CURRENTLY ONE OF THE MOST FREQUENT NAMES IS SELECTED (THE FIRST ONE)
    PROBLEMS: DIFFERENT SPELLINGS; NAME OCCURS ONLY ONCE
    
    Inputs: schreiben (non-tokenized)
    
    Outputs: a possible name
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
    searches for words that indicate a cancellation and classifies into
    cancellation or non-cancellation
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    dok_Gevo - 'K' or 'N', depending on cancellation or not
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
    searches for words that indicate a cancellation and classifies into
    cancellation or non-cancellation
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    dok_Gevo - 'K' or 'N', depending on cancellation or not
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

### Reason =====================================================================

def grundFinden(schreiben):
    '''
    searches for words that describe a reason and assigns
    corresponding categories
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    dok_Grund - vector with letters representing the reasons
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
    determines which reason appears most frequently in a document
    and how often each reason occurs
    
    Inputs:     schreiben
    
    Outputs:    Grund - most frequent reason
    
                if not unique:
                erg_tab - dictionary with frequencies of each reason
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
    # checks
    '''
    checks whether words like Corona, covid, etc. occur
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    erg - vector with positions where COVID-related words occur
    '''
    covid_token = ['corona', 'covid', 'covid19', 'pandemie', 'coronabedingt',
                   'pandemisch', 'pandemische', 'pandemischen', 'kurzarbeit',
                   'coronakrise', 'lockdown', 'coronaregeln']
    schreiben_lower = delPunct(schreiben).lower()
    # TODO 
    # maybe include startswith somehow?
    erg = []
    for i in range(len(covid_token)):
        erg += searchIndizes(schreiben_lower, covid_token[i]) 
    return erg


def covidVergleich(schreiben):
    
    '''
    returns TRUE if a COVID-related word occurs (or multiple), and FALSE
    otherwise
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    TRUE/FALSE
    '''
    schreiben_lower = delPunct(schreiben).lower()
    erg = covidFinden(schreiben_lower)
    if len(erg) > 0:
        return True
    else:
        return False

# =============================================================================
# Extract only the main body text (for classification)
# =============================================================================


def getHauptteil(schreiben):
    '''
    extracts the main body of a document
    THIS NO LONGER DOES THAT BECAUSE IT DOES NOT MAKE SENSE
    ONLY KEPT TO AVOID CHANGING EVERYTHING
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    erg - document without "introduction" and currently all in
                      lowercase and without punctuation
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
    applies getHauptteil to dok_list
    '''
    new_list = []
    for i in dok_list:
        new_list.append(getHauptteil(i))
    return new_list

# TODO
# possibly remove everything after the greetings

# =============================================================================
# Write everything into an Excel table
# =============================================================================

### Create table ========================================================
 

def getInfos(schreiben):
    '''
    returns name, VNR, GeVo and COVID indicator for a document
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    Infos - containing:
                name
                vnr
                gevo
                covid
                text body
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
    creates a table using getInfos
    
    Inputs:     dok_list
    
    Outputs:    table

    '''
    Tabelle = []
    for i in range(len(dok_list)):
        Tabelle += getInfos(dok_list[i])
    return(Tabelle)
     
### write table to Excel ======================================================

def tabelleSchreiben(dok_list, filename):
    '''
    creates an Excel table with the given information
    
    Inputs:     dok_list - list of documents
    
                filename - the name under which the Excel file should be saved
                           
    Outputs:    no outputs in Python, Excel table is saved in the folder 
                Schriftstücke\Output under filename
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
    outSheet.write('A1', 'First name') 
    outSheet.write('B1', 'Last name')
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
# # Text mining (for 4_data_preparation)
# =============================================================================
# =============================================================================

def setColRowNames(dataframe, colnames, rownames):
    '''
    sets column and row names for a pandas DataFrame
    
    Inputs:     dataframe
                colnames
                rownames
                
    Outputs:    dataframe with column and row names
    '''
    
    dataframe.columns = colnames
    dataframe2 = dataframe.T
    dataframe2.columns = rownames
    dataframe = dataframe2.T
    return(dataframe)

# =============================================================================
# Stemming etc.
# =============================================================================

def getStem(texte):
    '''
    stems a list of texts
    
    Inputs:     texte - basically dok_list?
    
    Outputs:    txt_stem - list of stemmed texts

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




def delWortvector(schreiben, wortvector):
    '''
    removes all words contained in a given vector from a document
    
    Inputs:     schreiben
                wortvector
        
    Outputs:    document without words from wortvector

    '''
    schreib = [token for token in schreiben.split() if token not in wortvector]
    schreib2 = ' '.join(schreib)
    return(schreib2)





stop_words = get_stop_words('de')
stop_words_stem = getStem(stop_words)    


def delNumbers(schreiben):
    '''
    removes all tokens that contain numbers
    
    Inputs:     schreiben - non-tokenized document
    
    Outputs:    erg - non-tokenized document without numbers (as string)

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



### compute DTM matrices

def getDTM(vect, txt, cols = True):
    '''
    creates the Document-Term Matrix (documents as columns and 
                                      words as rows)
    
    Inputs:     vect - CountVectorizer
                txt - texts without stopwords
    
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


### remove rare words ================


def countFuncAll(dtm, token_vec):
    '''
    counts in how many individual documents from the texts a token appears, NOTE:
        it does not count total occurrences, but in how many documents it appears.
        If it appears multiple times in one document, it is counted only once.
    applies countFunc to a vector of tokens

    Inputs:     dtm - 
                token_vec
    
    Outputs:    countvec
    '''
    countvec = []
    for i in range(len(token_vec)):
        countvec.append(len(np.where(dtm.loc[token_vec[i]] > 0)[0]))
    return countvec


def myNotIn(vector, teilvector):
    '''
    returns vector without the elements contained in teilvector
    
    Inputs:     vector - long vector 
                teilvector - vector with elements to be removed from vector
        
    Outputs:    neuvek - new vector without elements from teilvector
    '''
    neuvek = []
    for i in range(len(vector)):
        if vector[i] not in teilvector:
            neuvek.append(vector[i])
    return(neuvek)


def delWenige(anzahlvector, anzahl, token_vek):
    '''
    removes from a vector all words that occur less than or equal to "anzahl"
    times and returns which words from token_vek remain, i.e. those that occur
    more frequently than "anzahl"
    also works for n-grams
    
    Inputs:     anzahlvector
                anzahl
                token_vek
                
    Outputs:    keep_words

    '''
    dellist = []
    for i in range(anzahl):
        dellist += myGleich(anzahlvector, i + 1)
    
    keep = myNotIn(range(len(anzahlvector)), dellist)
    
    keep_words = np.array(token_vek)[keep]
    
    return keep_words


def getTFIDF(tf):
    '''
    computes the tf-idf transformation of a matrix
    
    Inputs:     tf - matrix
    
    Outputs:    tf_idf - matrix transformed with tf-idf

    '''
    tf_array = np.array(tf)
    df = []
    # J: number of documents
    J = len(tf.columns)
    
    for j in range(J):
        # number of documents where the term is NOT present
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
    computes the normalized Gini coefficient of a token
    
    Inputs:
                token
                dokvek - contains "number of documents in class 1 with token", ..., 
                         "number of documents in class n with token"
                auftvek - contains "number of documents in class 1", ...,
                          "number of documents in class n"
    
            
    Output:     gini - Gini coefficient of the token
    '''
    try: 
        n = len(auftvek)
        p = []
        for i in range(n):
            p.append(dokvek[i] / np.sum(dokvek))
        P = auftvek
        p_tilde = []
        for i in range(n):
            p_tilde.append(p[i] / P[i])
        p_tilde_ges = np.sum(p_tilde)
        gini = 0
        for i in range(n):
            gini += (p_tilde[i] / p_tilde_ges) ** 2
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
        gini_ind.append(
            giniCalc(
                token[t],
                [B_count, F_count, K_count, R_count, T_count],
                auftvek
            )
        )
    return gini_ind

### ===========================================================================
### Classification (for 5_Classification)
### ===========================================================================

def createWC(txt, sw, name, mw = 200):
    '''
     Inputs: txt - 
             sw - stopwords to be used
             name - filename
    
    Output: png image with word cloud
    
    

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
    for classes 0 and 1, where 0 is the POSITIVE CLASS
    
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
    here I want to compare with K, and depending on how the confusion matrix
    is defined, K must be gevo2 if the first letter of the class
    comes BEFORE K in the alphabet, otherwise gevo1

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
### =============================================================================
### Random Forest
### =============================================================================



def getRF(dtm_train, dtm_test, y_train, y_test, neval, mfval):
    if mfval: # max features to default
        random.seed(1802)
        rfc = RandomForestClassifier(n_estimators= neval, verbose=True)
    else:
        random.seed(1802)
        rfc = RandomForestClassifier(n_estimators= neval, max_features = mfval, verbose=True)
    rf_fit = rfc.fit(dtm_train, y_train)

    predictions = rf_fit.predict(dtm_test)
    predictions_proba = rf_fit.predict_proba(dtm_test)

    return(predictions_proba, predictions)

def RFAll(dtm_train, dtm_test, y_train, y_test, X_test_ind, neval = 100, 
          mfval = True, threshold_RF = False):
    '''
    y_train and y_test in binary (0/1) encoding!!!

    '''
    
    [RF_probs, RF_pred] = getRF(dtm_train, dtm_test, y_train, y_test, neval, mfval)

    mydf = pd.DataFrame([X_test_ind, y_test, RF_probs[:,1]]).T
    mydf.columns = ['Document', 'true class', 'probability positive']
    mydf = mydf.sort_values('Document')
    if threshold_RF == False:
        [cm, sensi, speci, accuracy] = getWerteKlassi(y_test, RF_pred)
        y_pred = RF_pred
    else:
        y_pred = getPred(RF_probs[:,1], threshold_RF)
        [cm, sensi, speci, accuracy] = getWerteKlassi(y_test, y_pred)
    
    return(cm, sensi, speci, accuracy, y_pred, mydf)



def ergtable(y_test, y_pred, true_probs):
    '''
    returns a clear results table for the RF
    
    Inputs:     y_test
                y_pred
                true_probs (partly from rfBerechnen)
    
    Outputs:    erg - 

    '''
    erg = pd.DataFrame()
    erg['true label'] = y_test
    # erg['probability'] = true_probs
    erg['predicted label'] = y_pred
    erg['correct'] = [0] * len(y_test)
    erg['correct'][myvectorGleich(y_test,y_pred)] = 1
    return(erg)

def getPred(probs, threshold):
    y_pred = []
    for i in range(len(probs)):
        if probs[i] >= threshold:
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
                y_train (binary encoded 0/1)
                y_test (binary encoded 0/1)
                
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
    y_train and y_test in binary (0/1) encoding!!!

    '''
    
    [SVM_pred, SVM_probs] = getSVM(dtm_train, dtm_test, y_train, y_test)

    mydf = pd.DataFrame([X_test_ind, y_test, SVM_probs[1]]).T
    mydf.columns = ['Document', 'true class', 'probability positive']
    mydf = mydf.sort_values('Document')

    [cm, sensi, speci, accuracy] = getWerteKlassi(y_test, SVM_pred)
    if threshold_SVM == False:
        [cm, sensi, speci, accuracy] = getWerteKlassi(y_test, SVM_pred)
        y_pred = SVM_pred
    else:
        y_pred = getPred(SVM_probs[1], threshold_SVM)
        [cm, sensi, speci, accuracy] = getWerteKlassi(y_test, y_pred)

    return(cm, sensi, speci, accuracy, y_pred, mydf)


# =============================================================================
# Extra Trees
# =============================================================================

def getET(X_train, X_test, y_train, y_test):
    '''
    
    Inputs:     X_train
                X_test
                y_train
                y_test
                
    Outputs:    y_pred
                cm (confusion matrix)

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
# Plot functions
# =============================================================================


def ROCFunc(y_test, y_pred, name, xlab = "1 - Specificity", ylab = "Sensitivity"):
    '''
    computes ROC curve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                plot

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


def HistFuncK(daten, name, xlab = "Probability of termination", ylab = "Number of terminations"):
    '''
    computes ROC curve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                plot

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

def HistFuncNK(daten, name, xlab = "Probability of termination", ylab = "Number of other documents"):
    '''
    computes ROC curve
    
    Inputs:     y_test
                y_pred
                
    Outputs:    auc 
                plot

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

def GiniFunc(daten, name, xlab = "Token sorted by Gini index", ylab = "Gini index"):
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


def AnteileFunc(daten, name, xlab = "Token sorted by differences", ylab = "Difference"):
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
# complete function for RF and SVM (only random.seed() needs to be set beforehand)
# =============================================================================
# klgr TRUE: all data
# klgr FALSE: small matrix
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
              
    wkeiten_pos = mydf_train['probability positive'][mydf_train['true class'] == 1]
    HistFuncK(wkeiten_pos, name1)
    wkeiten_neg = np.sort(mydf_train['probability positive'][mydf_train['true class'] == 0])
    HistFuncNK(wkeiten_neg, name2)
    
    ROCFunc(y_train, mydf_train['probability positive'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['probability positive'])).T
    th.columns = ['false_positive rate', 'true_positive rate', 'threshold']
    th['specificity'] = 1 - th['false_positive rate']
    th['sensitivity + specificity'] = th['specificity'] + th['true_positive rate']
    
    ts = list(th['threshold'][th['sensitivity + specificity'] == max(th['sensitivity + specificity'])])[0]                  

    if min(wkeiten_pos) > max(wkeiten_neg):
        random.seed(1802)
        ts = random.uniform(max(wkeiten_neg), min(wkeiten_pos))
    
    y_pred_train_new = getPred(mydf_train['probability positive'], ts)
    [cm_train_new, sensi_train_new, spezi_train_new,
     richtigkl_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test, 
     mydf_test] = RFAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_RF = ts)
                           
    wkeiten_pos = mydf_test['probability positive'][mydf_test['true class'] == 1]
    HistFuncK(wkeiten_pos, name4)
    wkeiten_neg = np.sort(mydf_test['probability positive'][mydf_test['true class'] == 0])
    HistFuncNK(wkeiten_neg, name5)
    
    ROCFunc(y_test, mydf_test['probability positive'], name6) 

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
              
    wkeiten_pos = mydf_train['probability positive'][mydf_train['true class'] == 1]
    HistFuncK(wkeiten_pos, name1)
    wkeiten_neg = np.sort(mydf_train['probability positive'][mydf_train['true class'] == 0])
    HistFuncNK(wkeiten_neg, name2)
    
    ROCFunc(y_train, mydf_train['probability positive'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['probability positive'])).T
    th.columns = ['false_positive rate', 'true_positive rate', 'threshold']
    th['specificity'] = 1 - th['false_positive rate']
    th['sensitivity + specificity'] = th['specificity'] + th['true_positive rate']
    
    ts = list(th['threshold'][th['sensitivity + specificity'] == max(th['sensitivity + specificity'])])[0]                  

    if min(wkeiten_pos) > max(wkeiten_neg):
        random.seed(1802)
        ts = random.uniform(max(wkeiten_neg), min(wkeiten_pos))
    
    y_pred_train_new = getPred(mydf_train['probability positive'], ts)
    [cm_train_new, sensi_train_new, spezi_train_new,
     richtigkl_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test, 
     mydf_test] = SVMAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_SVM = ts)
                           
    wkeiten_pos = mydf_test['probability positive'][mydf_test['true class'] == 1]
    HistFuncK(wkeiten_pos, name4)
    wkeiten_neg = np.sort(mydf_test['probability positive'][mydf_test['true class'] == 0])
    HistFuncNK(wkeiten_neg, name5)
    
    ROCFunc(y_test, mydf_test['probability positive'], name6) 

    return(cm_train, sensi_train, spezi_train, richtigkl_train, y_pred_train, 
           mydf_train, ts,
           cm_train_new, sensi_train_new, spezi_train_new, richtigkl_train_new,
           y_pred_train_new,
           cm_test, sensi_test, spezi_test, richtigkl_test, y_pred_test,
           mydf_test)

                 
    
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
                            ['F true', 'B true', 'R true', 'T true', 'K true'],
                            ['F predicted', 'B predicted', 'R predicted', 'T predicted', 'K predicted'])
    
    ergdf['F true'] = getEineKlass(f_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['B true'] = getEineKlass(b_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['R true'] = getEineKlass(r_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['T true'] = getEineKlass(t_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    ergdf['K true'] = getEineKlass(k_true, f_pred, b_pred, r_pred, t_pred, k_pred)
    
    ergdf = ergdf.T
    return(ergdf)
    
# =============================================================================
# required for dill to work
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
