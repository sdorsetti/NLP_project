import sys
sys.path.insert(0,"C:/Users/Stanislasd’Orsetti/NLP_project/")

from hashtags.modelisation.databuilder import create_dataset
from config import structure_dict
from hashtags.modelisation.model import LABEL_COLUMNS, HashtagTweetTagger
from hashtags.modelisation.torch_dataset import HashTagTweetDataModule
from hashtags.modelisation.utils import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from tqdm import tqdm


column = 'label'
csv_path = "C:/Users/Stanislasd’Orsetti/NLP_project/data/labellized_hashtags_df.csv"
df = pd.read_csv(csv_path)

#BUILD TRAINING DF
train_df, val_df = create_dataset(df,column)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

#TORCHDATA
data_module = HashTagTweetDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

#MODEL
model = HashtagTweetTagger(
  n_classes=len(LABEL_COLUMNS),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps
)

#TRAINING
logger = TensorBoardLogger("lightning_logs", name="hashtags_tweets")
checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
  accelerator="cpu",
  logger=logger,
  checkpoint_callback=checkpoint_callback,
  callbacks=[early_stopping_callback],
  max_epochs=N_EPOCHS,
  gpus=1,
  progress_bar_refresh_rate=30,
)

trainer.fit(model, data_module)
trainer.test()

torch.save(model.state_dict(), "hashtags/model.pt")

