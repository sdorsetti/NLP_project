import numpy as np 
from NLP_project.user_location.modelisation.torch_dataset import TweetDataset
from NLP_project.user_location.modelisation.config import args
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import TweetTokenizer

def tokenize_pad_numericalize(entry, vocab_stoi, max_length=40):
  text = [vocab_stoi[token] if token in vocab_stoi else vocab_stoi['<unk>'] for token in entry]
  padded_text = None
  if len(text) < max_length:   padded_text = text + [ vocab_stoi['<pad>'] for i in range(len(text), max_length) ] 
  elif len(text) > max_length: padded_text = text[:max_length]
  else:                        padded_text = text
  return padded_text

def create_dataset(df,column, vocab_stoi,over_sampling=True, test_size=0.3):
    tok = TweetTokenizer()
    max_length = df[column].apply(lambda x : len(tok.tokenize(x))).max()
    
    X = np.vstack(np.array(df[column].apply(lambda x : tokenize_pad_numericalize(x, vocab_stoi,max_length))))
    label = list(df["label"].unique())
    dico_label = {i:j for j,i in enumerate(label)}
    y = np.array(df['label'].apply(lambda x : dico_label[x]))
    
    if over_sampling: 
        ros = RandomOverSampler(random_state=0)
        X, y = ros.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size = test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    stratify=y_train, 
                                                    test_size = test_size)

    tweets_loc = {"train":[{"text": X_train[idx,:], "label": y_train[idx]} for idx in range(len(X_train))],
    "test": [{"text": X_test[idx,:], "label": y_test[idx]} for idx in range(len(X_test))],
    "val": [{"text": X_val[idx,:], "label": y_val[idx]} for idx in range(len(y_val))]
    }
    

    train_loader = DataLoader(TweetDataset(tweets_loc['train'], args), batch_size=args['bsize'], num_workers=1, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TweetDataset(tweets_loc['val'], args), batch_size=args['bsize'], num_workers=1, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TweetDataset(tweets_loc['test'], args), batch_size=args['bsize'], num_workers=1, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader


