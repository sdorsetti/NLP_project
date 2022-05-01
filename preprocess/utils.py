from langdetect import detect
import re 

def remove_hashtags(tokens):
    tokens = filter(lambda x: "#" not in x, tokens)
    return list(tokens)

def remove_hastags_sign(tokens):
    tokens = map(lambda x: x.replace('#',""), tokens)
    return list(tokens)

def remove_url(tokens):
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)

def remove_html(tokens):
    tokens = filter(lambda x: x[0]+x[-1] != '<>', tokens)
    return list(tokens)

def remove_mentions(tokens):
    tokens = filter(lambda x: "@" not in x, tokens)
    return list(tokens)

def detect_language(text):
  try: 
    return detect(text)
  except: 
    print("unreconnized character")

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

