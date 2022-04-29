from NLP_project.user_location.labellisation.dic_label import d
from NLP_project.user_location.labellisation.utils import coocurrence, apply_ner
from NLP_project.user_location.config import structure_dict
from NLP_project.preprocess.preprocess import DataPreprocessor
import os
import json
import pandas as pd
import logging
from nltk.tokenize import TweetTokenizer
from geopy.geocoders import Nominatim



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

    def label_data(self,string_,d,history):
        """_summary_

        Args:
            d (_type_): _description_
            ccr_edge (_type_): _description_
            string_ (_type_): _description_
            threshold (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """

        try : 
          label = history[string_] 
          return label
        except:
          geolocator = Nominatim(user_agent="MyApp")
          try : 
            location = geolocator.geocode(string_)
          except : 
            location = None
          
          
          tok = TweetTokenizer()

          if location == None: 
            return "Unknown"
          else:
            location = location[-2].split(',')[-1].lower()
            location = tok.tokenize(location)[0]
            
            final = [label for label, region in d.items() if location in region]
            if len(final)== 1: 
              history[string_] = final[0]
              return final[0]
            elif len(final) == 0:
              return "Location Not In Dic"
            else : 
              return "Multiple Location"
       
    
    def get_df(self, to_csv=True,d = d):
        """

        Args:
            d (_type_, optional): _description_. Defaults to d.
            threshold (int, optional): _description_. Defaults to 3.
            to_csv (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        logging.basicConfig(filename='labelizer.log', level=logging.DEBUG)
        #Process df with a NER
        ner = self.apply_ner_
        df = self.drop_unexistent_locations(ner)

        path = structure_dict["output_path"]
        if os.path.exists(path): 
            with open(f"{path}history.json") as f:
                history = json.load(f)
        else : 
            history = {}            

        logging.info("LABELLISATION")
        df["label"] = df["ner"].progress_apply(lambda x : self.label_data(x[-1], d, history))
        df = df.drop(["ner"], axis=1)

        shape = df.shape[0]
        df = df[df["label"].isin(list(d.keys()))]

        logging.info("{} % of unlabelled data **** Final size of df {}".format((1 - (df.shape[0]/shape))*100, df.shape[0]))  
        if to_csv: 
            df.to_csv(structure_dict["path_to_csv"] + "labellized_df.csv", index=False)
        return df
