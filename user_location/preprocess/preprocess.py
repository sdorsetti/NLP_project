from NLP_project.user_location.preprocess.utils import *
from NLP_project.user_location.config import args
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import logging

logging.basicConfig(filename='preprocessor.log', level=logging.DEBUG)

class DataPreprocessor():
    def __init__(self, data:pd.DataFrame, column: str):
        self.column = column
        self.data = data
        self.corpus = self.data[column].astype(str).array
    @property
    def preprocess(self, f_hastags = remove_hashtags, threshold = 50, url = True, html = True, hashtags=True, mentions = True):
        """_summary_

        Args:
            url (bool, optional): _description_. Defaults to True.
            html (bool, optional): _description_. Defaults to True.
            hashtags (bool, optional): _description_. Defaults to True.
            mentions (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        tokenizer = TweetTokenizer()
        tokenized_sentences = []
        logging.info("tokenizing")
        for sentence in tqdm(self.corpus):
            tokens = tokenizer.tokenize(sentence)
            if url :
                tokens = remove_url(tokens)
            if html: 
                tokens = remove_html(tokens)
            if hashtags: 
                tokens = f_hastags(tokens)
            if mentions: 
                tokens = remove_mentions(tokens)
            tokens = list(map(lambda x: x.lower(), tokens))
            tokenized_sentences.append(tokens)
        logging.info("Phrasing")
        phrases = Phrases(tokenized_sentences, threshold=threshold)
        phraser = Phraser(phrases)
        clean_corpus = []
        for sentence in tokenized_sentences:
            clean_corpus.append(phraser[sentence])
        return [" ".join(i) for i in clean_corpus]
    
    def get_df(self, language='en', detect_language_=True, to_csv=False):
        """_summary_

        Args:
            language (str, optional): _description_. Defaults to 'en'.
            detect_language_ (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        
        logging.info("****PREPROCESSING COLUMN {} OF TWEETS DF".format(self.column))
        df = self.data.copy()
        df["cleaned_{}".format(self.column)] = self.preprocess
        if detect_language_:
            logging.info("language detection")
            tqdm.pandas()
            df["language"] = df["cleaned_{}".format(self.column)].progress_apply(lambda x : detect_language(x))
            df = df[df['language'] == language]
            shape = df.shape[0]
            df = df.drop(["language"],axis=1)
            logging.info("{} % of tweets in {}".format(df.shape[0]/shape*100, language))
        if to_csv: 
            df.to_csv(args["output_path"] + "preprocessed_df.csv", index=False)
        return df