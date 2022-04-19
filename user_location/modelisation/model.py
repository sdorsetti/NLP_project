import torch
import torch.nn as nn
def get_pretrained_vectors(path:str):
    return None

class TweetModel(nn.Module):
    def __init__(self,args):
        """_summary_

        Args:
            input_dim (_type_): _description_
            hidden_dim (_type_): _description_
            output_dim (_type_): _description_
            pretrained_vectors (_type_, optional): _description_. Defaults to None.
        """
        super(TweetModel, self).__init__()
        pretrained_vectors=  get_pretrained_vectors(args["pretrained_vectors_path"])
        self.ebd = torch.nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.hidden_linear_layer = torch.nn.Linear(args["hidden_dim"], args["hidden_dim"], bias=True)
        self.hidden_linear_layer2 = torch.nn.Linear(args["hidden_dim"], 1000, bias=True)
        self.hidden_linear_layer3 = torch.nn.Linear(1000, 1000, bias=True)
        self.classification_layer = torch.nn.Linear(1000, args["output_dim"], bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x  = self.ebd(x)
        x  = x.mean(1)
        h  = torch.relu(self.hidden_linear_layer( x ))
        h  = torch.relu(self.hidden_linear_layer2( h ))
        h  = torch.relu(self.hidden_linear_layer3( h ))
        # h  = self.dropout(h)
        h  = self.classification_layer(h)
        logits = self.softmax(h)
        return logits