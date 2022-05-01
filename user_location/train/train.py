from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter
from NLP_project.config import structure_dict
from NLP_project.user_location.modelisation.databuilder import *
from NLP_project.user_location.modelisation.model import TweetModel,MergedModel
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

    for i,batch in enumerate(train_loader):

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

    for i,batch in enumerate(val_loader):
        with torch.no_grad():

            batch = {'text': batch['text'].to(device), 'label': batch['label'].to(device)}
            logits = model(batch['text'])
            
            b_counter = Counter(batch['label'].detach().cpu().tolist())
            b_weights = torch.tensor( [ sum(batch['label'].detach().cpu().tolist()) / b_counter[label] 
                                       if b_counter[label] > 0 else 0 
                                       for label in list(range(params_model["num_class"])) ] )
            
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
    path_to_config = structure_dict["path_to_config"]

    with open(path_to_config) as f:
        params_model = json.load(f)
        f.close()

    logging.basicConfig(filename=f'{output_path}TRAINING.log', level=logging.DEBUG)

    pretrained_vectors_path = structure_dict["pretrained_vectors_path"]
    labelled_df_path = structure_dict["path_to_csv"] + "labellized_user_location_df.csv"
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
    optimiz = params_model["optim"]
    column = params_model["column"]
    ner = params_model["ner"]

    if ner == "ner":
        ner=True
    else : 
        ner=False


    #loaddataset
    logging.info("*****************2 : CREATE DATASET ********************")
    pretrained_vectors, vocab_stoi = open_pretrained_vectors(pretrained_vectors_path,drop_vectors=False)

    train_loader, val_loader, test_loader= create_dataset(df,column, vocab_stoi,over_sampling=True, test_size=0.3, ner=ner)

    #model
    if params_model["architecture"] == "arch1":
        model = TweetModel(params_model,pretrained_vectors.vectors)
    elif params_model["architecture"]=="merged":
        modelA = TweetModel(params_model,pretrained_vectors.vectors)
        modelB = TweetModel(params_model,pretrained_vectors.vectors)

        modelA.load_state_dict(torch.load(output_path + f"ner_text.pt"))
        modelB.load_state_dict(torch.load(output_path + f"ner_user_decription.pt"))

        model = MergedModel(modelA, modelB)

    del pretrained_vectors, vocab_stoi

    model = model.to(device)

    if optimiz == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif optimiz == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)

    val_ep_losses = list()
    train_ep_losses = list()
    #TRAIN
    logging.info("**************** 3 :  TRAINING**********")
    for ep in tqdm(range(epochs)):

        train_loss_it = train(model, train_loader,optimizer, ep, params_model, device)
        train_ep_losses.append(train_loss_it)
        trues, preds, val_loss_it_avg, val_acc_it_avg, val_loss_it, val_acc_it = inference("val",val_loader, model, device)
        val_ep_losses.append(val_loss_it_avg)
        plot_losses(train_ep_losses,val_ep_losses, params_model)

    logging.info("END OF TRAINING **** SAVING RESULTS *****")

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, output_path + f"{model_name}.pt")
    #PREDICT
    logging.info("**********4 : TESTING RESULTS*********")
    trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it = inference("test",test_loader, model, device)
    with open(f"{output_path}pred_{model_name}.json","w") as fp:

        json.dump({
        "trues":trues, 
        "preds":preds, 
        "loss_it_avg":loss_it_avg, 
        "acc_it_avg":acc_it_avg, 
        "loss_it":loss_it, 
        "acc_it":acc_it
        }, fp, indent=4)