from langdetect import detect

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
