#UTILS : 
import spacy 
from tqdm import tqdm
import ast
from collections import defaultdict
from itertools import combinations

def coocurrence(*inputs):
    """_summary_

    Returns:
        _type_: _description_
    """
    com = defaultdict(int)
    
    for named_entities in inputs:
        # Build co-occurrence matrix
        for w1, w2 in combinations(sorted(named_entities), 2):
            com[w1, w2] += 1
            com[w2, w1] += 1  #Including both directions

    result = defaultdict(dict)
    for (w1, w2), count in com.items():
        if w1 != w2:
            result[w1][w2] = {'weight': count}
    return result

def apply_ner(df,column, l_types = ["LOC", "GPE"]):
    """_summary_

    Args:
        df (_type_): _description_
        column (_type_): _description_
        l_types (list, optional): _description_. Defaults to ["LOC", "GPE"].

    Returns:
        _type_: _description_
    """

    list_ = []
    ner = spacy.load("en_core_web_sm")
    i=0
    for text in tqdm(df[column].astype(str)):
        i += 1
        doc = ner(text)
        list_.append([ent.text for ent in doc.ents if ent.label_ in l_types])
    return list_

def regroup_hash(lst) : 

  liste=[]
  new_k=[]
  for item in lst: 
        if item in ['tokyo2020']:
            liste.append('Tokyo2020')
        if item in ['olympics','olympicGames']:
          liste.append('Olympics')
        if item in ['tokyoolympics','tokyoolympics2021']:
          liste.append('TokyoOlympics')
        if item in ['teamindia','cheer4india']:
          liste.append('TeamIndia')  
        if item in ['ind','india']:
          liste.append('India')  
        if item in ['mirabaichanu','mirabachanu','mirabai','mirabai_chanu','mirabaichanu', 'mirabhaichanu', 'mirabi_chanu', 'mirrabaichanu', 'merabaichanu']:
          liste.append('MirabaiChanu')
        if item == 'swevaus':
          liste.append('SWEvAUS')
        if item == 'uswnt':
          liste.append('USWNT')
        if item == 'weightlifting':
          liste.append('Weightlifting')
        if item == 'swimming':
          liste.append('Swimming')
        if item == 'football':
          liste.append('Football')
        if item == 'teamgb':
          liste.append('TeamGB')
        if item == 'badminton':
          liste.append('Badminton')        
        if item == 'hockey':
          liste.append('Hockey')
        if item == 'silver':
          liste.append('Silver')
  for elem in liste:
      if elem not in new_k:
        new_k.append(elem)
  liste = new_k
  return liste


def find_common_hash(x, lst_tag):
    x=ast.literal_eval(x)
    lst=[]
    for elem in x : 
      if elem in lst_tag :
        lst.append(elem)
    return lst