from NLP_project.user_location.labellisation.dic_label import d
from NLP_project.user_location.labellisation.utils import coocurrence, apply_ner
from NLP_project.user_location.config import args
from NLP_project.user_location.preprocess.preprocess import DataPreprocessor

import pandas as pd
import logging

logging.basicConfig(filename='labelizer.log', level=logging.DEBUG)

class Labelizer():
    def __init__(self, df:pd.DataFrame, column:str, langage = False):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column (str): _description_
        """
        self.column = column
        try: 
            df[column]
            self.df = df
        except:  
            dp = DataPreprocessor(df,column.split('cleaned_')[-1])
            self.df = dp.get_df(detect_language_=langage)
    @property
    def apply_ner_(self):
      """

      """
      return apply_ner(self.df, self.column)

    def create_coocur_edges_(self,ner):
      return coocurrence(*ner)

    def extend_d(self,d,ccr):
        for lab in list(d.keys()):
            d[lab].extend(list(ccr[lab].keys()))
        return d

    def drop_unexistent_locations(self, ner):
        df = self.df.copy()
        df["ner"] = ner
        shape = df.shape[0]
        df = df[df["ner"].apply(lambda x :len(x)) >0]

        logging.info("{} % of false locations **** Final size of df {}".format(df.shape[0]/shape*100, df.shape[0]))
        return df

    def label_data(self,d,ccr_edge,string_,threshold=3):
        """_summary_

        Args:
            d (_type_): _description_
            ccr_edge (_type_): _description_
            string_ (_type_): _description_
            threshold (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        

        #if the name of the region is one of the tokens, it can be considered as the label 
        l_regions = [item for sublist in [d[r] for r in list(d.keys())] for item in sublist]
        for region in l_regions:
            if string_ == region:
                return [k for k,v in d.items() if region in v][0]
        for label in list(d.keys()):
            if string_ == label:
                return label


        ccr = ccr_edge[string_]
        #ccr is a dict that has often associated words as keys and the weight as value

        l_ccr2 = []
        for k1, e in ccr.items():
          l_ccr2.append({k1:w for w in e.values() if w > threshold})

        l_linked = [item for sublist in [list(a.keys()) for a in l_ccr2] for item in sublist] 

        l_final = []
        for region in list(d.keys()):
        #region is our labels
        #we check if there is an intersection between our hand-made regions associated to the label. 
            if len(list(set(d[region]) & set(l_linked))) > 0:
                l_final.append(region)
        #we can therefore have two labels, if two contradictory information were found in the localization, for example "London, Ontario"
        
        if len(l_final) ==1: 
            return l_final[0]
    
    def get_df(self, d = d,threshold=3, to_csv=False):
        """

        Args:
            d (_type_, optional): _description_. Defaults to d.
            threshold (int, optional): _description_. Defaults to 3.
            to_csv (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        logging.info("creating coocurences edges")

        #Process df with a NER
        ner = self.apply_ner_
        df = self.drop_unexistent_locations(ner)

        ccr = self.create_coocur_edges_(ner)
        d = self.extend_d(d,ccr)

        logging.info("LABELLISATION")
        df["label"] = df["ner"].progress_apply(lambda x : self.label_data(d,ccr,x[-1], threshold=threshold))
        df = df.drop(["ner"], axis=1)

        shape = df.shape[0]
        df = df.dropna(subset = ["label"])
        logging.info("{} % of unlabelled data **** Final size of df {}".format((1 - (df.shape[0]/shape))*100, df.shape[0]))  
        if to_csv: 
            df.to_csv(args["output_path"] + "preprocessed_df.csv", index=False)
        return df
