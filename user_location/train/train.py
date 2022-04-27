import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter

from typer import Exit

from modelisation.config import structure_dict, params_model
from modelisation.databuilder import *
from modelisation.model import TweetModel

from NLP_project.user_location.modelisation.utils import open_pretrained_vectors, plot_losses

import logging
import os

def train(model, train_loader,optimizer, ep, params, device):

    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        ep (_type_): _description_
        args (_type_): _description_
    """
    model.train()
    loss_it, acc_it = [],[]

    for batch in tqdm(enumerate(train_loader), desc="Epoch %s:" % (ep), total=train_loader.__len__()):
        
        batch = {'text': batch['text'].to(device), 'label': batch['label'].to(device)}
        optimizer.zero_grad()
        logits = model(batch['text'])

        b_counter = Counter(batch['label'].detach().cpu().tolist())
        b_weights = torch.tensor( [ sum(batch['label'].detach().cpu().tolist()) / b_counter[label] 
        if b_counter[label] > 0 else 0 for label in list(range(params['num_class'])) ] )
        b_weights = b_weights.to(device)

        loss_function = nn.CrossEntropyLoss(weight=b_weights)
        loss = loss_function(logits, batch['label'])

        loss.backward()

        optimizer.step()

        loss_it.append(loss.item())
        _, tag_seq  = torch.max(logits, 1)

        correct = (tag_seq.flatten() == batch['label'].flatten()).float().sum()
        acc = correct / batch['label'].flatten().size(0)
        acc_it.append(acc.item())

    logging.info("Epoch %s : %s : (%s %s) (%s %s)" % (
    str(ep), 
    'Training', 
    'loss', 
    sum(loss_it)/len(loss_it), 
    'acc', 
    sum(acc_it) / len(acc_it))
    )
    return sum(loss_it)/len(loss_it)

def inference(str,val_loader, model, device):

    """_summary_

    Args:
        loader (_type_): _description_
        model (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    model.eval()
    loss_it, acc_it = list(), list()
    preds, trues = list(), list()

    for batch in tqdm(enumerate(val_loader),total=val_loader.__len__()):
        with torch.no_grad():

            batch = {'text': batch['text'].to(device), 'label': batch['label'].to(device)}
            logits = model(batch['text'])
            
            b_counter = Counter(batch['label'].detach().cpu().tolist())
            b_weights = torch.tensor( [ sum(batch['label'].detach().cpu().tolist()) / b_counter[label] if b_counter[label] > 0 else 0 for label in list(range(20)) ] )
            b_weights = b_weights.to(device)

            loss_function = nn.CrossEntropyLoss(weight=b_weights)
            loss = loss_function(logits, batch['label'])
            loss_it.append(loss.item())
            _, tag_seq  = torch.max(logits, 1)

            correct = (tag_seq.flatten() == batch['label'].flatten()).float().sum()
            acc = correct / batch['label'].flatten().size(0)
            acc_it.append(acc.item())
        
            preds.extend(tag_seq.cpu().detach().tolist())
            trues.extend(batch['label'].cpu().detach().tolist())

    loss_it_avg = sum(loss_it)/len(loss_it)
    acc_it_avg = sum(acc_it)/len(acc_it)

    logging.info("%s : (%s %s) (%s %s)" % (
        str,
        'loss', 
        sum(loss_it)/len(loss_it), 
        'acc', 
        sum(acc_it) / len(acc_it))
        )
    
    return trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it

if __name__ == "__main__":
    output_path = structure_dict["output_path"]

    logging.basicConfig(filename=f'{output_path}TRAINING.log', level=logging.DEBUG)

    pretrained_vectors_path = structure_dict["pretrained_vectors_path"]
    column = structure_dict["column"]
    labelled_df_path = structure_dict["path_to_csv"] + "labellized_df.csv"
    logging.info("********1 IMPORT DATA ***********")
    if not os.path.exists(labelled_df_path): 
        logging.warning("No df processed ! Labellization of df")
        
    else: 
        df = pd.read_csv(labelled_df_path)

    device = params_model["device"]
    epochs = params_model["num_epochs"]
    optimizer = params_model["optim"]
    lr = params_model["learning_rate"]
    momentum = params_model["momentum"]
    model_name = params_model["model_name"]
    test_size = params_model["test_split"]


    pretrained_vectors, vocab_stoi = open_pretrained_vectors(pretrained_vectors_path,drop_vectors=False)

    #loaddataset
    logging.info("*****************2 : CREATE DATASET ********************")
    train_loader, val_loader, test_loader= create_dataset(df,column, vocab_stoi,over_sampling=True, test_size=0.3)

    #model
    if params_model["architecture"] == "arch1":
        model = TweetModel(pretrained_vectors)
    else:
        model = None
    del pretrained_vectors, vocab_stoi
    
    model = model.to(device)

    if optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)

    val_ep_losses = list()
    train_ep_losses = list()
    #TRAIN
    logging.info("**************** 3 :  TRAINING**********")
    for ep in range(epochs):
 
        train_loss_it = train(model, train_loader,optimizer, ep, params_model, device)
        train_ep_losses.append(train_loss_it)
        trues, preds, val_loss_it_avg, val_acc_it_avg, val_loss_it, val_acc_it = inference("val",val_loader, model, device)
        val_ep_losses.append(val_loss_it_avg)
        plot_losses(train_ep_losses,val_ep_losses)

    logging.info("END OF TRAINING **** SAVING RESULTS *****")

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, output_path + f"{model_name}.pt")
    
    #PREDICT
    logging.info("**********4 : TESTING RESULTS*********")
    trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it = inference("test",test_loader, model)
    with open(output_path + "pred.json","w") as fp:

        json.dump({
        "trues":trues, 
        "preds":preds, 
        "loss_it_avg":loss_it_avg, 
        "acc_it_avg":acc_it_avg, 
        "loss_it":loss_it, 
        "acc_it":acc_it
        }, fp, indent=4)