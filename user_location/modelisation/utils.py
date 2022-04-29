from NLP_project.user_location.config import structure_dict
import torch
from torchtext.vocab import vocab
import matplotlib.pyplot as plt
import json


def open_pretrained_vectors(path, drop_vectors = True):
    """_summary_

    Args:
        path (_type_): _description_
        drop_vectors (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    f = torch.load(path)
    vocab_stoi = vocab(f.stoi)
    unk_index = 0
    pad_index = 1
    vocab_stoi.insert_token("<unk>",unk_index)
    vocab_stoi.insert_token("<pad>", pad_index)
    vocab_stoi.set_default_index(unk_index)
    if drop_vectors:
        f = None
    return f,vocab_stoi

def plot_losses(loss_train, loss_val, args):
    fig, ax = plt.subplots(figsize=(15,15))
    ax.plot(range(len(loss_train)),loss_train)
    ax.plot(range(len(loss_train)),loss_val)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cross Entropy")
    a,b = 0.93,0.99
    for u,v in args.items():
        plt.text(a,b, f"{str(u)} : {str(v)}",horizontalalignment='left',
        verticalalignment='top', transform = ax.transAxes, fontsize=8, color='r')
        b = b-0.02
    ax.set_title(f"LOSS AT EP {len(loss_train)}", fontsize=15)
    ax.legend(["Train","Val"])
    fig.savefig(f'{structure_dict["output_path"]}LOSSES_{args["model_name"]}.png', dpi=fig.dpi)


