from nltk.stem.snowball import SnowballStemmer
import spacy
from spacymoji import Emoji

nlp = spacy.load("en_core_web_sm")
emoji = Emoji(nlp)
nlp.add_pipe("emoji", first=True)
stemmer = SnowballStemmer("english")

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def extract_emojies(x):
  doc = nlp(x['text']) #with emojis
  emojis = [token.text for token in doc if token._.is_emoji]
  return emojis