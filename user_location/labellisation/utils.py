#UTILS : 
import spacy 
from tqdm import tqdm

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
