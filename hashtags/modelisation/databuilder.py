import pandas as pd
from sklearn.model_selection import train_test_split
import ast

def create_dataset(df, label_column):
    df_dummies = df[label_column].apply(lambda x : ast.literal_eval(x)).str.join('|').str.get_dummies()
    list_columns =df_dummies.columns.tolist()
    df_expl = df.copy()
    df= pd.concat([df_expl, df_dummies], axis=1 )

    for elem in list_columns : 
        df[elem].loc[df[elem].isnull()] = df[elem].loc[df[elem].isnull()].apply(lambda x: 0)
        df[elem].loc[:]=df[elem].loc[:].apply(lambda x: int(x))
    df = df.loc[:,~df.columns.duplicated()]
    LABEL_COLUMNS = df_dummies.columns.tolist()[:]

    train_df, val_df = train_test_split(df, test_size=0.05)

    train_with = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
    train_without= train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]

    train_df = pd.concat(
        [train_with.sample(600),
        train_without]
        )

    return train_df, val_df