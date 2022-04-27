import numpy as np 
from NLP_project.user_location.modelisation.torch_dataset import TweetDataset
from NLP_project.user_location.config import  params_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import TweetTokenizer

def tokenize_pad_numericalize(entry, vocab_stoi, max_length=40):
    """_summary_

    Args:
        entry (_type_): _description_
        vocab_stoi (_type_): _description_
        max_length (int, optional): _description_. Defaults to 40.

    Returns:
        _type_: _description_
    """
    text = [vocab_stoi[token] if token in vocab_stoi else vocab_stoi['<unk>'] for token in entry]
    padded_text = None
    if len(text) < max_length:   padded_text = text + [ vocab_stoi['<pad>'] for i in range(len(text), max_length) ] 
    elif len(text) > max_length: padded_text = text[:max_length]
    else:                        padded_text = text
    return padded_text

def create_dataset(df,column, vocab_stoi,over_sampling=True, test_size=0.3):
    """_summary_

    Args:
        df (_type_): _description_
        column (_type_): _description_
        path_to_pretrained_vectors (_type_): _description_
        over_sampling (bool, optional): _description_. Defaults to True.
        test_size (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    tok = TweetTokenizer()
    max_length = df[column].apply(lambda x : len(tok.tokenize(x))).max()
    #   _,vocab_stoi = open_pretrained_vectors(path_to_pretrained_vectors)
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

    num_workers = params_model["num_workers"]
    train_loader = DataLoader(TweetDataset(tweets_loc['train']), batch_size=params_model['bsize'], num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TweetDataset(tweets_loc['val']), batch_size=params_model['bsize'], num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TweetDataset(tweets_loc['test']), batch_size=params_model['bsize'], num_workers=num_workers, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader



