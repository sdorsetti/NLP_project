import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter

from modelisation.config import args
from modelisation.databuilder import *
from modelisation.model import TweetModel

def train(model, train_loader,optimizer, ep, args, device):

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
        b_weights = torch.tensor( [ sum(batch['label'].detach().cpu().tolist()) / b_counter[label] if b_counter[label] > 0 else 0 for label in list(range(args['num_class'])) ] )
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

def inference(val_loader, model, device):

    """_summary_

    Args:
        loader (_type_): _description_
        model (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    model.eval()
    loss_it, acc_it, f1_it = list(), list(), list()
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
    
    return trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it

if __name__ == "__main__":
    #params:
    output_path = args["output_path"]
    device = args["device"]
    epochs = args["num_epochs"]
    optimizer = args["optim"]
    pretrained_vectors = args["pretrained_vectors_path"]
    column = args["column"]
    df = pd.read_csv(args["path_to_csv"])
    vocab_stoi = ""
    #loaddataset
    train_loader, val_loader, test_loader= create_dataset(df,column, vocab_stoi,over_sampling=True, test_size=0.3)

    #model
    model = TweetModel(args)

    if optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = args['lr'])
    val_ep_losses = list()
    #TRAIN
    for ep in range(epochs):
        train(model, train_loader,optimizer, ep, args, device)
        trues, preds, val_loss_it_avg, val_acc_it_avg, val_loss_it, val_acc_it = inference(val_loader, model, device)
        val_ep_losses.append(val_loss_it_avg)
        
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, output_path + "model.pt")
    
    #PREDICT
    trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it = inference(test_loader, model)
    with open(output_path + "pred.json","w") as fp:

        json.dump({
            "trues":trues, 
        "preds":preds, 
        "loss_it_avg":loss_it_avg, 
        "acc_it_avg":acc_it_avg, 
        "loss_it":loss_it, 
        "acc_it":acc_it
        }, fp, indent=4)