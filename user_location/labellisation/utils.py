#UTILS : 
import spacy 
from tqdm import tqdm

from collections import defaultdict
from itertools import combinations
import logging

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


# def filter_ents_by_min_weight(edges, min_weight):
#     """_summary_

#     Args:
#         edges (_type_): _description_
#         min_weight (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     coocur_edges_filtered = defaultdict()
#     for k1, e in edges.items():
#         ents_over_x_weight = {k2: v for k2, v in e.items() if v['weight'] > min_weight}
#         if ents_over_x_weight:  # ie. Not empty
#             coocur_edges_filtered[k1] = ents_over_x_weight
#     return coocur_edges_filtered


# def create_coocur_edges(df,column, l_types = ["LOC", "GPE"]):
#     """_summary_

#     Args:
#         df (_type_): _description_
#         column (_type_): _description_
#         l_types (list, optional): _description_. Defaults to ["LOC", "GPE"].

#     Returns:
#         _type_: _description_
#     """
#     logging.info("Apply NER on {}".format(column))
#     ner = apply_ner(df, column,l_types)
#     return coocurrence(*ner)
