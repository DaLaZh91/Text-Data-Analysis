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
    counts how often each distinct token occurs in the vector possibilities
    
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

def vnrFinden(document):
    '''
    searches typical positions for insurance numbers
    
    Inputs:     document
    
    Outputs:    possible VNRs
    '''
    vnr_token = ['Versicherungsnummer', 'VersicherungsNr.', 'Nr.', 'Nr:', 
                 'Nr.:', 'Nr', 'VNR', 'VNR:', 'Versicherungsnummer:', 
                 'VSNR:', 'VSNR', 'VSNR.', 'VSNR,','VSNR.:']
    
    return findeFunc(document, vnr_token, 2)


def vnrBauen(document):
    '''
    tries to construct the VNR and stops when it ends
    
    Inputs:     document
    
    Outputs:    possible VNRs
    '''
    possibilities = vnrFinden(document)
    comp_vector = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    new_poss = possibilities # keep the same structure as possibilities
    # loop over the different occurrences of a VNR
    for k in range(len(possibilities)):
        # loop over the different tokens that are extracted
        for j in range(len(possibilities[k])): # always the same
            # possibly loop over the first two elements to check whether
            # it is actually a number?
            a_poss = new_poss[k][j]
            if len(a_poss) > 0:
                for i in range(len(a_poss)):
                    # check if something occurs that is not a digit or similar
                    if a_poss[i] not in comp_vector:
                            a_poss = a_poss.replace(a_poss[i], 'X')
            new_poss[k][j] = a_poss
    # delete from where it starts with 2 X somewhere & if it is empty
    str_tog = [None] * len(possibilities)
    for k in range(len(possibilities)):
        for j in range(len(possibilities[k])): 
                if 'X' in new_poss[k][j] or new_poss[k][j] == []:
                    new_poss[k][j] = ''
        # concatenate what remains for each VNR
        str_tog[k] = "".join(new_poss[k])       
    return str_tog



def searchZahlenzeichenketten(document):
    '''
    finds numeric character strings (with the characters and digits from comp_vector)
    in a document
    
    Inputs:      document - non-tokenized document
    
    Outputs:    numbertoken - the numeric tokens that occur in the document
                numbind - the indices of these numeric tokens

    '''
    comp_vector = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '.',
                  ',', '-', '|']
    docsplit = document.split()
    docsplit
    isdig = []
    for i in range(len(docsplit)):
        # isdig.append(docsplit[i].isdigit())
        token = docsplit[i]
        for j in range(len(token)):
            if (token[j] not in comp_vector):
                isdig.append(False)
                break;
            elif (j == (len(token) - 1)):
                isdig.append(True)
    numbind = myGleich(isdig, True)
    numbertoken = np.array(docsplit)[numbind]
    return(numbertoken, numbind)

def searchVNR(document):
    '''
    finds VNRs by combining numeric tokens from 
    searchZahlenzeichenketten
    
    Inputs:     document - non-tokenized document
    
    Outputs:    nt - 

    '''
    numbertoken, numbind = searchZahlenzeichenketten(document)
    isnext = []
    for i in range(len(numbind) - 1):
        if (numbind[i] + 1 == numbind[i + 1]):
            isnext.append(True)
        else:
            isnext.append(False)
    # combine those that are directly consecutive
    numbertoken = list(numbertoken)
    nt = numbertoken
    for i in range(len(numbind) - 1):
        if (isnext[i] and not isnext[i + 1]):
            nt[i:(i + 2)] = [''.join(numbertoken[i:(i + 2)])]
    return(nt)

# compares the VNRs
def vnrVergleich(document):
    # checks
    '''
    compares the VNRs 
    
    Inputs:     document - non-tokenized document
    
    Outputs:    ideally: VNR, otherwise currently 'unclear VNR' and
                'no VNR found' (prints that, stores None for the latter cases)
    '''
    try: 
        poss_vnr = vnrBauen(document)
        vnr = poss_vnr[0]
        equal = [None] * len(poss_vnr)
        for i in range(len(poss_vnr)):
            equal[i] = bool(vnr == poss_vnr[i])
        if False not in equal:
            return vnr
        else:
            return ('ev ' + vnr)
    except:
        return 'none found'


### remove punctuation =====================================================

def delPunct(document):
    # checks
    '''
    removes punctuation from the document
    
    Inputs:     document - NON-tokenized document
    
    Outputs:    document without punctuation

    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»‚‚')
    token_punctless =[token for token in document if token not in punctuation]
    token_punctless = ''.join(token_punctless)
    return token_punctless

def delPuncts(doc_list):
    # checks
    '''
    removes punctuation from all documents in dok_list
    
    Inputs:     doc_list - list of non-tokenized documents
    
    Outputs:    doc_list without punctuation
    '''
    new_list = []
    for i in doc_list:
        new_list.append(delPunct(i))
    return new_list

### convert all letters to lowercase =========================================

def machLowercase(doc_list):
    # checks
    '''
    converts all letters to lowercase
    
    Inputs:     doc_list - list of non-tokenized documents
    
    Outputs:    new_list - list of non-tokenized documents, completely
                           in lowercase
    '''
    new_list = []
    for i in doc_list:
        new_list.append(i.lower())
    return new_list

### Name ======================================================================

def namenFinden(document):
    '''
    searches typical positions for names
    
    Inputs:     document - non-tokenized document
    
    Outputs:    poss_names - possible names
    '''
    punctuation = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€°’‘“—«”£„‘©®»')
    numbers = list('0123456789')
    document_lower = delPunct(document).lower()
    names_token = ['grüßen', 'grüße', 'gruß', 'von', 'from']
    document_tok = document_lower.split()
    # since the name is often at the beginning of the letterhead:
    pos12 = document_tok[0:2]
    m = 2
    poss_names = findeFunc(document_lower, names_token, m)
    poss_names.append(pos12)
    todel = []
    # loop over all possible names to remove email addresses and numbers
    no_words = ['von', 'company_name', 'kundenservice',
                  'kundenberater', 'kundenberaterin', 'ihnen', 'der', 
                  'die', 'das', 'dem']
    for i in range(len(poss_names)):
        for j in range(m): 
            if poss_names[i][j] in no_words:
                todel.append(i)
                break            
        # otherwise consider both parts separately and check individual characters
        else: 
            for j in range(m):
                part = list(poss_names[i][j])
                if any(map(lambda v: v in punctuation, part)):
                    todel.append(i)
                    break
                elif any(map(lambda v: v in numbers, part)):
                    todel.append(i)
                    break
                # to filter out email addresses containing "company_name"
                elif all(map(lambda v: v in part, list('company_name'))):
                    todel.append(i)
                    break
    j = 0
    for i in range(len(todel)):
        del(poss_names[todel[i] - j])
        j = j + 1
    return poss_names

def nameVergleich(document):
    '''
    CURRENTLY ONE OF THE MOST FREQUENT NAMES IS SELECTED (THE FIRST ONE)
    PROBLEMS: DIFFERENT SPELLINGS; NAME OCCURS ONLY ONCE
    
    Inputs: document (non-tokenized)
    
    Outputs: a possible name
    '''
    try:
        names_vec = namenFinden(document)
        counter = []
        for i in range(len(names_vec)):
            counter.append(names_vec.count(names_vec[i]))
        max_count = max(counter)
        res = myGleich(counter, max_count)
        if len(names_vec[res[0]][0]) > 0:
            a_max = names_vec[res[0]]
        else:
            a_max = names_vec[res[1]]
    except:
        a_max = ['N', 'Ö']
    return a_max

### GeVo ======================================================================

def KFindenSimple(document, c_words):
    '''
    searches for words that indicate a cancellation and classifies into
    cancellation or non-cancellation
    
    Inputs:     document - non-tokenized document
    
    Outputs:    doc_Gevo - 'C' or 'N', depending on cancellation or not
    '''
    document_lower = delPunct(document).lower()
    doc = document_lower.split()
    doc_GeVo = []
    if any(map(lambda v: v in c_words, doc)):
         doc_GeVo = 'C'
    else:
        doc_GeVo = 'N'
    return(doc_GeVo)


def gevoFinden(document, c_words, nc_words, pos = 'C', neg = 'N'):
    '''
    searches for words that indicate a cancellation and classifies into
    cancellation or non-cancellation
    
    Inputs:     document - non-tokenized document
    
    Outputs:    doc_Gevo - 'C' or 'N', depending on cancellation or not
    '''
    document_lower = delPunct(document).lower()
    doc = document_lower.split()
    doc_bigram = [' '.join(b) for l in [' '.join(dok)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    doc_GeVo = []

    if nc_words == []:
        if any(map(lambda v: v in c_words, doc)):
            doc_GeVo = pos
        elif any(map(lambda v: v in c_words, doc_bigram)):
            doc_GeVo = pos
        else:
            doc_GeVo = neg
    else: 
        if any(map(lambda v: v in nc_words, doc)):
            doc_GeVo = neg
        elif any(map(lambda v: v in nc_words, doc_bigram)):
            doc_GeVo = neg
        elif any(map(lambda v: v in c_words, doc)):
             doc_GeVo = pos
        elif any(map(lambda v: v in c_words, doc_bigram)):
            doc_GeVo = pos
        else:
            doc_GeVo = neg
            
    return(doc_GeVo)

### Reason =====================================================================

def grundFinden(document):
    '''
    searches for words that describe a reason and assigns
    corresponding categories
    
    Inputs:     document - non-tokenized document
    
    Outputs:    doc_reason - vector with letters representing the reasons
    '''
    document_lower = delPunct(document).lower()
    doc = document_lower.split()
    doc_reason = []
    j_words = ['arbeitnehmerwechsel', 'neue arbeit', 'abgang']
    f_words = ['finanziell', 'geld', 'insolvenz', 'finanziellen']
    r_words = ['rente', 'rentenbeginn', 'rentenzahlung', 'nicht arbeitsfähig', 
               'nicht mehr arbeitsfähig']
    d_words = ['todesfall', 'gestorben', 'todes', 'tod', 'verstorben', 
               'verstorbene']
    if any(map(lambda v: v in j_words, doc)):
        doc_reason += 'J'
    if any(map(lambda v: v in f_words, doc)):
        doc_reason += 'F'
    if any(map(lambda v: v in r_words, doc)):
        doc_reason += 'R'
    if any(map(lambda v: v in d_words, doc)):
        doc_reason += 'D'

    return doc_reason


def grundVergleich(document):
    '''
    determines which reason appears most frequently in a document
    and how often each reason occurs
    
    Inputs:     document
    
    Outputs:    reason - most frequent reason
    
                if not unique:
                res_tab - dictionary with frequencies of each reason
    '''
    document_lower = delPunct(document).lower()
    poss_reasons = grundFinden(document_lower)
    res = defaultdict(int)
    for i in poss_reasons:
        res[i] += 1
    res_tab = (dict(res))
    J_freq = res_tab.get('J', 0)
    F_freq = res_tab.get('F', 0)
    R_freq = res_tab.get('R', 0)
    D_freq = res_tab.get('D', 0)
    freq = ['J', 'F', 'R', 'D']
    freqs = [J_freq, F_freq, R_freq, D_freq]
    poss_reasons = max(freqs)
    
    if poss_reasons > 0:
        reason = freq[myGleich(freqs, poss_reasons)[0]]
        if sum(freqs) == poss_reasons:
            return reason
        else:
            return reason, res_tab
    else:
        return 'O' 

### Corona ====================================================================

def covidFinden(document):
    # checks
    '''
    checks whether words like Corona, covid, etc. occur
    
    Inputs:     document - non-tokenized document
    
    Outputs:    res - vector with positions where COVID-related words occur
    '''
    covid_token = ['corona', 'covid', 'covid19', 'pandemie', 'coronabedingt',
                   'pandemisch', 'pandemische', 'pandemischen', 'kurzarbeit',
                   'coronakrise', 'lockdown', 'coronaregeln']
    document_lower = delPunct(document).lower()
    res = []
    for i in range(len(covid_token)):
        res += searchIndizes(document_lower, covid_token[i]) 
    return res


def covidVergleich(document):
    
    '''
    returns TRUE if a COVID-related word occurs (or multiple), and FALSE
    otherwise
    
    Inputs:     document - non-tokenized document
    
    Outputs:    TRUE/FALSE
    '''
    document_lower = delPunct(document).lower()
    res = covidFinden(document_lower)
    if len(res) > 0:
        return True
    else:
        return False

# =============================================================================
# Extract only the main body text (for classification)
# =============================================================================


def getHauptteil(document):
    '''
    extracts the main body of a document
    THIS NO LONGER DOES THAT BECAUSE IT DOES NOT MAKE SENSE
    ONLY KEPT TO AVOID CHANGING EVERYTHING
    
    Inputs:     document - non-tokenized document
    
    Outputs:    res - document without "introduction" and currently all in
                      lowercase and without punctuation
    '''
    document_pl = delPunct(document)
    document_lower = document_pl.lower()
    res = document_lower
    return res

def getHauptteile(dok_list):
    '''
    applies getHauptteil to doc_list
    '''
    new_list = []
    for i in doc_list:
        new_list.append(getHauptteil(i))
    return new_list

# =============================================================================
# Write everything into an Excel table
# =============================================================================

### Create table ========================================================
 

def getInfos(document):
    '''
    returns name, VNR, GeVo and COVID indicator for a document
    
    Inputs:     document - non-tokenized document
    
    Outputs:    Infos - containing:
                name
                vnr
                gevo
                covid
                text body
    '''
    name = nameVergleich(document)
    vnr = vnrVergleich(document)
    gevo = gevoFinden(document, ['kündigung', 'kündige', 'kündigen'],
                      ['beitragsfreistellung', 'beitragspause', 'erhöhung'])
    covid = covidVergleich(document)
    textpart = ' '.join(getHauptteil(document).split())
    Infos = [name, vnr, gevo, covid, textpart]
    return Infos

def createTable(doc_list):
    '''
    creates a table using getInfos
    
    Inputs:     doc_list
    
    Outputs:    table

    '''
    table = []
    for i in range(len(doc_list)):
        table += getInfos(doc_list[i])
    return(table)
     
### write table to Excel ======================================================

def tabelledocument(doc_list, filename):
    '''
    creates an Excel table with the given information
    
    Inputs:     doc_list - list of documents
    
                filename - the name under which the Excel file should be saved
                           
    Outputs:    no outputs in Python, Excel table is saved in the folder 
                \Output under filename
    '''
    tab = createTable(doc_list)
    names = tab[0::5]
    vnrs = tab[1::5]
    gevos = tab[2::5]
    covids = tab[3::5]
    texts = tab[4::5]
    
    os.chdir(r'W:\your_folder\Output')
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
            outSheet.write(i + 1, 0, names[i][0])
        except:
            outSheet.write(i + 1, 0, '')
        try:
            outSheet.write(i + 1, 1, names[i][1])
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

def getStem(texts):
    '''
    stems a list of texts
    
    Inputs:     texts - basically doc_list
    
    Outputs:    txt_stem - list of stemmed texts

    '''
    stemmer = snowballstemmer.stemmer('german')
    txt_stem = []
    for i in range(len(texts)):
        try:
            stem_teil = stemmer.stemWords(texts[i].split())
            txt_stem.append(' '.join(stem_teil))
        except:
            txt_stem.append('nan')
    return(txt_stem)

def delWortvector(document, wordvector):
    '''
    removes all words contained in a given vector from a document
    
    Inputs:     document
                wordvector
        
    Outputs:    document without words from wordvector

    '''
    doc = [token for token in document.split() if token not in wordvector]
    doc2 = ' '.join(doc)
    return(doc2)

stop_words = get_stop_words('de')
stop_words_stem = getStem(stop_words)    


def delNumbers(document):
    '''
    removes all tokens that contain numbers
    
    Inputs:     document - non-tokenized document
    
    Outputs:    res - non-tokenized document without numbers (as string)

    '''
    numbers = list('0123456789')  
    main = getHauptteil(document)
    n = len(main.split())
    output = []
    for i in range(n):
        if not any(map(lambda v: v in numbers, main.split()[i])):
            output.append(main.split()[i])
    res = ' '.join(output)
    return(res)

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
    amount = len(txt) 
    td = pd.DataFrame(vects.todense())
    td.columns = vect.get_feature_names()
    tdM = td.T
    tdM.columns = ['Doc '+ str(i) for i in range(1, amount + 1)]
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

def delSelteneWoerter(vect, X_train_C, X_train_NC, p, C_len_train, NC_len_train):
    
    count_DF_C = getTokenCount(vect, X_train_C, 'count C')
    count_DF_NC = getTokenCount(vect, X_train_NC, 'count NC')

    gem_df = pd.merge(count_DF_C, count_DF_NC, on = 'token', how = 'outer')
    gem_df = gem_df.fillna(0)
    
    keep_C = np.where(gem_df['count C'] > (p * C_len_train))
    keep_NC = np.where(gem_df['count NC'] > (p * NC_len_train))
    
    keep_both = np.unique((np.append(keep_C, keep_NC)))
    keep_token = list(gem_df['token'][keep_both])
    
    new_df = pd.concat([gem_df['token'][keep_both], gem_df['count C'][keep_both], gem_df['count NC'][keep_both]], axis = 1)
    new_df = setColRowNames(new_df, new_df.columns, new_df['token'])
    
    return(keep_token, new_df)

def delSelteneWoerter2(vect, X_train_J, X_train_F, X_train_C, X_train_R, X_train_D, p, y_train):
    

    count_DF_J = getTokenCount(vect, X_train_J, 'count J')
    count_DF_F = getTokenCount(vect, X_train_F, 'count F')
    count_DF_C = getTokenCount(vect, X_train_C, 'count C')
    count_DF_R = getTokenCount(vect, X_train_R, 'count R')
    count_DF_D = getTokenCount(vect, X_train_T, 'count D')
    
    gem_1 = pd.merge(count_DF_J, count_DF_F, how = 'outer', on = 'token')
    gem_2 = pd.merge(gem_1, count_DF_C, how = 'outer', on = 'token')
    gem_3 = pd.merge(gem_2, count_DF_R, how = 'outer', on = 'token')
    gem_4 = pd.merge(gem_3, count_DF_D, how = 'outer', on = 'token')
    gem_df = gem_4    
    gem_df = gem_df.fillna(0)
    
    J_len = len(myGleich(y_train, 'J'))
    F_len = len(myGleich(y_train, 'F'))
    C_len = len(myGleich(y_train, 'C'))
    R_len = len(myGleich(y_train, 'R'))
    D_len = len(myGleich(y_train, 'D'))
    
    keep_J = gem_df['token'][gem_df['count J'] > (p * J_len)]
    keep_F = gem_df['token'][gem_df['count F'] > (p * F_len)]
    keep_C = gem_df['token'][gem_df['count C'] > (p * C_len)]
    keep_R = gem_df['token'][gem_df['count R'] > (p * R_len)]
    keep_D = gem_df['token'][gem_df['count D'] > (p * D_len)]
   
    keep_all = list(keep_B) + list(keep_F) + list(keep_C) + list(keep_R) + list(keep_D)    
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


def myNotIn(vector, partvector):
    '''
    returns vector without the elements contained in teilvector
    
    Inputs:     vector - long vector 
                partvector - vector with elements to be removed from vector
        
    Outputs:    newvec - new vector without elements from partvector
    '''
    newvec = []
    for i in range(len(vector)):
        if vector[i] not in partvector:
            newvec.append(vector[i])
    return(newvec)


def delWenige(countvector, count, token_vec):
    '''
    removes from a vector all words that occur less than or equal to "anzahl"
    times and returns which words from token_vek remain, i.e. those that occur
    more frequently than "anzahl"
    also works for n-grams
    
    Inputs:     countvector
                count
                token_vec
                
    Outputs:    keep_words

    '''
    dellist = []
    for i in range(count):
        dellist += myGleich(countvector, i + 1)
    
    keep = myNotIn(range(len(countvector)), dellist)
    
    keep_words = np.array(token_vec)[keep]
    
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
        notpres = len(myGleich(tf_array[:,j],0))
        df.append(J - notpres)
    
    idf = []
    for j in range(J):
        idf.append(np.log(J/df[j]))
    
    tf_idf = tf * idf
    return(tf_idf)

# =============================================================================
# Gini Index
# =============================================================================


def giniCalc(token, docvec, occvec):
    '''
    computes the normalized Gini coefficient of a token
    
    Inputs:
                token
                docvec - contains "number of documents in class 1 with token", ..., 
                         "number of documents in class n with token"
                occvec - contains "number of documents in class 1", ...,
                          "number of documents in class n"
    
            
    Output:     gini - Gini coefficient of the token
    '''
    try: 
        n = len(occvec)
        p = []
        for i in range(n):
            p.append(docvec[i] / np.sum(docvec))
        P = occvec
        p_tilde = []
        for i in range(n):
            p_tilde.append(p[i] / P[i])
        p_tilde_all = np.sum(p_tilde)
        gini = 0
        for i in range(n):
            gini += (p_tilde[i] / p_tilde_all) ** 2
    except:
        gini = 0
    return gini


def getGini(df, occvec):
    gini_ind = []
    token = list(df['token'])
    for t in range(len(token)):
        C_count = list(df['count C'])[t]
        NC_count = list(df['count NC'])[t]
        gini_ind.append(giniCalc(token[t], [C_count, NC_count], occvec))
    return gini_ind


def getGini2(df, occvec):
    gini_ind = []
    token = list(df['token'])
    for t in range(len(token)):
        J_count = list(df['count J'])[t]
        F_count = list(df['count F'])[t]
        C_count = list(df['count C'])[t]
        R_count = list(df['count R'])[t]
        D_count = list(df['count D'])[t]
        gini_ind.append(
            giniCalc(
                token[t],
                [J_count, F_count, C_count, R_count, D_count],
                occvec
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

def getHäufigeWörter(txt, number = 200):
         
    word_cloud_dict = Counter(' '.join(txt).split())
    bigrams = [' '.join(b) for l in [' '.join(txt)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    word_cloud_dict_bi = Counter(bigrams)

    freq = word_cloud_dict.most_common(number)
    freq_bi = word_cloud_dict_bi.most_common(number)
    
    mcwords = []
    mcwords_bi = []
    for i in range(number):
        mcwords.append(freq[i][0])
        mcwords_bi.append(freq_bi[i][0])
        
        freq_sl = set(mcwords).difference(stop_words_stem)
    
    return(freq_sl, mcwords_bi)

def getHäufigeWörtermitSW(txt, numberw, numberbi):
    word_cloud_dict = Counter(' '.join(txt).split())
    bigrams = [' '.join(b) for l in [' '.join(txt)] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    word_cloud_dict_bi = Counter(bigrams)

    freq = word_cloud_dict.most_common(numberw)
    freq_bi = word_cloud_dict_bi.most_common(numberbi)
    return(freq, freq_bi)
    
    
def getAnzahlen(compwordvec, class1, class2):
    incl1 = []
    for i in compwordvec:
        count = 0
        for j in range(len(class1)):
            if i in class1[j].split():
                count += 1
            elif i in [' '.join(b) for l in [' '.join(class1[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]:
                count += 1
        incl1.append(count)
    
    incl2 = []
    for i in compwordvec:
        count = 0
        for j in range(len(class2)):
            if i in class2[j].split():
                count += 1
            elif i in [' '.join(b) for l in [' '.join(class2[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]:
                count += 1
        inKl2.append(count)
        
    sumcl1 = 0
    for j in range(len(class1)):
        if any(map(lambda v: v in compwordvec, class1[j].split())):
            sumcl1 += 1
        elif any(map(lambda v: v in compwordvec, [' '.join(b) for l in [' '.join(class1[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])])):
            sumcl1 += 1
    
    sumcl2 = 0
    for j in range(len(class2)):
        if any(map(lambda v: v in compwordvec, class2[j].split())):
            sumcl2 += 1
        elif any(map(lambda v: v in compwordvec, [' '.join(b) for l in [' '.join(class2[j].split())] for b in zip(l.split(' ')[:-1], l.split(' ')[1:])])):
            sumcl2 += 1
            
    return(incl1, sumcl1, incl2, sumcl2)



def getWerte(c_words, nc_words, Xtr, ytr, positive = 'C', negative = 'N'):
    ypred = []
    for i in range(len(Xtr)):
        ypred.append(gevoFinden(Xtr[i], c_words, nc_words, pos = positive, neg = negative))
    
    if positive == 1:
        cm_wrong = confusion_matrix(ytr, ypred)
        cm = np.array([[1, 1], [1, 1]])
        cm[0][0] = cm_wrong[1][1]
        cm[1][1] = cm_wrong[0][0]
        cm[0][1] = cm_wrong[1][0]
        cm[1][0] = cm_wrong[0][1]
        # cm = cm_new
    else:
        cm = confusion_matrix(ytr, ypred)
    sensi = cm[0][0]/(sum(cm[0]))
    speci = cm[1][1]/(sum(cm[1]))
    accuracy = (cm[0][0] + cm[1][1])/len(ypred)
    return(cm, sensi, speci, accuracy, ypred)

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
    speci = cm[1][1]/(sum(cm[1]))
    accuracy = (cm[0][0] + cm[1][1])/len(ypred)
    return(cm, sensi, speci, accuracy)


def getBestCombinations(words, Xtr, ytr, gevo1, gevo2 = 'C'):
    '''
    here I want to compare with C, and depending on how the confusion matrix
    is defined, C must be gevo2 if the first letter of the class
    comes BEFORE C in the alphabet, otherwise gevo1

    '''
    numbers = list(range(len(words)))
    combinations = []
    for r in range(len(numbers)+1):
        for combination in itertools.combinations(numbers, r):
            combinations.append(combination)

    cms = []
    fright = []
    combis = []
    for i in range(len(combinations)):
        combis.append(list(np.array(words)[list(list(combinations)[i])]))
        if gevo2 == 'R':
            [cm, a, b, c, d] = getWerte(combis[i], [], Xtr, ytr, gevo1, gevo2)
            cm_new = np.array([[1, 1], [1, 1]])
            cm_new[0][0] = cm[1][0]
            cm_new[1][1] = cm[0][1]
            cm_new[0][1] = cm[1][1]
            cm_new[1][0] = cm[0][0]
            cm_fin = cm_new
        else:
            [cm_fin, a, b, c, d] = getWerte(combis[i], [], Xtr, ytr, gevo1, gevo2)
        cms.append(cm_fin)
        frichtig.append(cms[i][0][0])
        
    best_combis = myGleich(fright, max(fright))

    np.array(kombis)[best_combis]

    kright = []
    for i in best_combis:
        kright.append(cms[i][1][1])
        
    best_combis_2 = myGleich(kright, max(kright))

    res = np.array(combis)[list(np.array(best_combis)[best_combis_2])]
    return res
    
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
    
    Outputs:    res - 

    '''
    res = pd.DataFrame()
    res['true label'] = y_test
    # res['probability'] = true_probs
    res['predicted label'] = y_pred
    res['correct'] = [0] * len(y_test)
    res['correct'][myvectorGleich(y_test,y_pred)] = 1
    return(res)

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
    
    os.chdir(r'W:\your_folder\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()
    return(auc)


def HistFuncK(data, name, xlab = "Probability of termination", ylab = "Number of terminations"):
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
    
    plt.hist(data, color = 'lightgreen')
    os.chdir(r'W:\your_folder\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()

def HistFuncNK(data, name, xlab = "Probability of termination", ylab = "Number of other documents"):
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
    
    plt.hist(data, color = 'lightblue')
    os.chdir(r'W:\your_folder\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()

def GiniFunc(data, name, xlab = "Token sorted by Gini index", ylab = "Gini index"):
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    #plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.plot(np.sort(data))#, color = 'lightgreen')
    os.chdir(r'W:\your_folder\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()


def AnteileFunc(data, name, xlab = "Token sorted by differences", ylab = "Difference"):
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.style'] = 'normal'
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel(xlab, fontsize = 11)
    plt.ylabel(ylab, fontsize = 11)
    
    #plt.xlim([0, 1])
    plt.ylim([-1, 1])
    
    plt.plot(np.sort(data))#, color = 'lightgreen')
    os.chdir(r'W:\your_folder\Output\Plots')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    return()



# =============================================================================
# complete function for RF and SVM (only random.seed() needs to be set beforehand)
# =============================================================================
# klgr TRUE: all data
# klgr FALSE: small matrix
def RFcomplete(reason, dtm_train, dtm_test, y_train, y_test, X_train_ind, X_test_ind, klgr = True):
    [cm_train, sensi_train, speci_train, accuracy_train, y_pred_train, 
     mydf_train] = RFAll(dtm_train, dtm_train, y_train, y_train, X_train_ind)
    if klgr == True:
        name1 = reason + 'hist_C_RF_train.png'   
        name2 = reason + 'hist_NC_RF_train.png'  
        name3 = reason + 'roc_RF_train.png'
        name4 = reason + 'hist_C_RF_test.png'   
        name5 = reason + 'hist_NC_RF_test.png' 
        name6 = reason + 'roc_RF_test.png'
    else:
        name1 = reason + 'hist_C_RF_train_kl.png'   
        name2 = reason + 'hist_NC_RF_train_kl.png'  
        name3 = reason + 'roc_RF_train_kl.png'
        name4 = reason + 'hist_C_RF_test_kl.png'   
        name5 = reason + 'hist_NC_RF_test_kl.png' 
        name6 = reason + 'roc_RF_test_kl.png'
              
    probs_pos = mydf_train['probability positive'][mydf_train['true class'] == 1]
    HistFuncK(probs_pos, name1)
    probs_neg = np.sort(mydf_train['probability positive'][mydf_train['true class'] == 0])
    HistFuncNK(probs_neg, name2)
    
    ROCFunc(y_train, mydf_train['probability positive'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['probability positive'])).T
    th.columns = ['false_positive rate', 'true_positive rate', 'threshold']
    th['specificity'] = 1 - th['false_positive rate']
    th['sensitivity + specificity'] = th['specificity'] + th['true_positive rate']
    
    ts = list(th['threshold'][th['sensitivity + specificity'] == max(th['sensitivity + specificity'])])[0]                  

    if min(probs_pos) > max(probs_neg):
        random.seed(1802)
        ts = random.uniform(max(probs_neg), min(probs_pos))
    
    y_pred_train_new = getPred(mydf_train['probability positive'], ts)
    [cm_train_new, sensi_train_new, speci_train_new,
     accuracy_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, speci_test, accuracy_test, y_pred_test, 
     mydf_test] = RFAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_RF = ts)
                           
    probs_pos = mydf_test['probability positive'][mydf_test['true class'] == 1]
    HistFuncK(probs_pos, name4)
    probs_neg = np.sort(mydf_test['probability positive'][mydf_test['true class'] == 0])
    HistFuncNK(probs_neg, name5)
    
    ROCFunc(y_test, mydf_test['probability positive'], name6) 

    return(cm_train, sensi_train, speci_train, accuracy_train, y_pred_train, 
           mydf_train, ts,
           cm_train_new, sensi_train_new, speci_train_new, accuracy_train_new,
           y_pred_train_new,
           cm_test, sensi_test, speci_test, accuracy_test, y_pred_test,
           mydf_test)  

def SVMcomplete(reason, dtm_train, dtm_test, y_train, y_test, X_train_ind, X_test_ind, klgr = True):
    [cm_train, sensi_train, speci_train, accuracy_train, y_pred_train, 
     mydf_train] = SVMAll(dtm_train, dtm_train, y_train, y_train, X_train_ind)
    if klgr == True:
        name1 = reason + '_hist_K_SVM_train.png'   
        name2 = reason + '_hist_NK_SVM_train.png'  
        name3 = reason + '_roc_SVM_train.png'
        name4 = reason + '_hist_K_SVM_test.png'   
        name5 = reason + '_hist_NK_SVM_test.png' 
        name6 = reason + '_roc_SVM_test.png'
    else:
        name1 = reason + '_hist_K_SVM_train_kl.png'   
        name2 = reason + '_hist_NK_SVM_train_kl.png'  
        name3 = reason + '_roc_SVM_train_kl.png'
        name4 = reason + '_hist_K_SVM_test_kl.png'   
        name5 = reason + '_hist_NK_SVM_test_kl.png' 
        name6 = reason + '_roc_SVM_test_kl.png'
              
    probs_pos = mydf_train['probability positive'][mydf_train['true class'] == 1]
    HistFuncK(probs_pos, name1)
    probs_neg = np.sort(mydf_train['probability positive'][mydf_train['true class'] == 0])
    HistFuncNK(probs_neg, name2)
    
    ROCFunc(y_train, mydf_train['probability positive'], name3) 

    th = pd.DataFrame(metrics.roc_curve(y_train, mydf_train['probability positive'])).T
    th.columns = ['false_positive rate', 'true_positive rate', 'threshold']
    th['specificity'] = 1 - th['false_positive rate']
    th['sensitivity + specificity'] = th['specificity'] + th['true_positive rate']
    
    ts = list(th['threshold'][th['sensitivity + specificity'] == max(th['sensitivity + specificity'])])[0]                  

    if min(probs_pos) > max(probs_neg):
        random.seed(1802)
        ts = random.uniform(max(probs_neg), min(probs_pos))
    
    y_pred_train_new = getPred(mydf_train['probability positive'], ts)
    [cm_train_new, sensi_train_new, speci_train_new,
     accuracy_train_new] = getWerteKlassi(y_train, y_pred_train_new)

    [cm_test, sensi_test, speci_test, accuracy_test, y_pred_test, 
     mydf_test] = SVMAll(dtm_train, dtm_test, y_train, y_test, X_test_ind,
                           threshold_SVM = ts)
                           
    probs_pos = mydf_test['probability positive'][mydf_test['true class'] == 1]
    HistFuncK(probs_pos, name4)
    probs_neg = np.sort(mydf_test['probability positive'][mydf_test['true class'] == 0])
    HistFuncNK(probs_neg, name5)
    
    ROCFunc(y_test, mydf_test['probability positive'], name6) 

    return(cm_train, sensi_train, speci_train, accuracy_train, y_pred_train, 
           mydf_train, ts,
           cm_train_new, sensi_train_new, speci_train_new, accuracy_train_new,
           y_pred_train_new,
           cm_test, sensi_test, speci_test, accuracy_test, y_pred_test,
           mydf_test)

                 
    
def getEineKlass(x_true, fpr, jpr, rpr, dpr, cpr):
    f_x = list(set(fpr).intersection(x_true))
    j_x = list(set(jpr).intersection(x_true))
    r_x = list(set(rpr).intersection(x_true))
    d_x = list(set(dpr).intersection(x_true))
    c_x = list(set(cpr).intersection(x_true))
    all_x = [len(f_x), len(j_x), len(r_x), len(d_x), len(c_x)]
    return(all_x)

def getErgebnisse(y_pred_F, y_pred_J, y_pred_R, y_pred_D, X_test, y_test):

    f_pred = myGleich(y_pred_F, 1) + myGleich(y_pred_F, 'F')
    j_pred = myGleich(y_pred_J, 1) + myGleich(y_pred_J, 'J')
    r_pred = myGleich(y_pred_R, 1) + myGleich(y_pred_R, 'R')
    d_pred = myGleich(y_pred_D, 1) + myGleich(y_pred_D, 'D')
    c_pred = list(set(range(len(X_test))).difference(f_pred + j_pred + r_pred + d_pred))   
    f_true = myGleich(y_test, 'F') 
    j_true = myGleich(y_test, 'J') 
    r_true = myGleich(y_test, 'R') 
    d_true = myGleich(y_test, 'D') 
    c_true = myGleich(y_test, 'C') + myGleich(y_test, 'O')
    
    resdf = pd.DataFrame([[0] * 5] * 5)
    resdf = setColRowNames(resdf, 
                            ['F true', 'J true', 'R true', 'D true', 'C true'],
                            ['F predicted', 'J predicted', 'R predicted', 'D predicted', 'C predicted'])
    
    resdf['F true'] = getEineKlass(f_true, f_pred, j_pred, r_pred, d_pred, c_pred)
    resdf['J true'] = getEineKlass(j_true, f_pred, j_pred, r_pred, d_pred, c_pred)
    resdf['R true'] = getEineKlass(r_true, f_pred, j_pred, r_pred, d_pred, c_pred)
    resdf['D true'] = getEineKlass(d_true, f_pred, j_pred, r_pred, d_pred, c_pred)
    resdf['C true'] = getEineKlass(c_true, f_pred, j_pred, r_pred, d_pred, c_pred)
    
    resdf = resdf.T
    return(resdf)
    
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
