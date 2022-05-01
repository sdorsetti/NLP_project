from NLP_project.labellisation.utils import *
from NLP_project.labellisation.dic_label import d
from NLP_project.config import structure_dict
from nltk.tokenize import TweetTokenizer
from geopy.geocoders import Nominatim
from collections import Counter
from shapely.geometry import Point, Polygon
import logging 
import pandas as pd
import ast
import os
import json

class Labelizer():
    def __init__(self, df:pd.DataFrame, column:str, langage = False):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column (str): _description_
        """
        self.column = column
        self.df= df
        self.__lsttag = None
            

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

    def label_location(self,string_,shp,history):
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


          if location == None: 
            return "Unknown"
          else:          
            location = Point(location.longitude, location.latitude)

            label = [continent for continent,poly in zip(shp.continent,shp.geometry) if location.within(poly)]

            if len(label)>0:
              history[string_] = label[0] 
              path = structure_dict["path_to_csv"]
              with open(f"{path}history.json", 'w') as fp:
                json.dump(history, fp)

              return label[0]
            elif len(label) == 0:
              return "Location not in labels"
           
           
    @property
    def count_hashtags(self):
        if self.__lsttag == None:
            lst=[]
            for i in range(self.df.shape[0]):
                    lst.append(ast.literal_eval(self.df[self.column].iloc[i]))
        
            lst_hash =[item.lower() for sublist in lst for item in sublist]
            
            count= Counter(lst_hash)
            nb_occ= count.most_common(20)
            self.__lsttag = [i[0] for i in nb_occ]
        return self.__lsttag
                        
    def get_df(self, to_csv=True, shp=None):
        """

        Args:
            d (_type_, optional): _description_. Defaults to d.
            threshold (int, optional): _description_. Defaults to 3.
            to_csv (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        logging.basicConfig(filename='labelizer.log', level=logging.DEBUG)
        if self.column == "user_location":

            #Process df with a NER
            ner = self.apply_ner_
            df = self.drop_unexistent_locations(ner)

            path = structure_dict["path_to_csv"]
            if os.path.exists(f"{path}history.json"): 
                with open(f"{path}history.json") as f:
                    history = json.load(f)
            else : 
                history = {}             

            logging.info("LABELLISATION")
            df["label"] = df["ner"].progress_apply(lambda x : self.label_location(x[-1], shp, history))
            df = df.drop(["ner"], axis=1)

            shape = df.shape[0]
            df = df[df["label"].isin(list(shp.continent))]

            logging.info("{} % of unlabelled data **** Final size of df {}".format((1 - (df.shape[0]/shape))*100, df.shape[0]))  
            if to_csv: 
                df.to_csv(f"{structure_dict['path_to_csv']}labellized_{self.column}_df.csv", index=False)
            self.df = df
            return self.df
        elif self.column == "hashtags":
            df = self.df.copy()
            df['label'] = df['hashtags'].apply(lambda x : find_common_hash(x.lower(), self.count_hashtags))
            df["label"] = df["label"].apply(lambda x : regroup_hash(x))
            df["label"] = df['label'].apply(lambda x : sorted(x))
            if to_csv: 
                df.to_csv(f"{structure_dict['path_to_csv']}labellized_{self.column}_df.csv", index=False)
            return df

            

