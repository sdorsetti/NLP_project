loss_train = {
    "output_path": "/content/gdrive/MyDrive/cours/ensae/NLP/output/",
    "pretrained_vectors_path":"/content/gdrive/MyDrive/cours/ensae/NLP/pretrained_vectors.pth",
    "column": "cleaned_text",
    "path_to_csv": "/content/gdrive/MyDrive/cours/ensae/NLP/",
    
}
params_model  = {
    "optim": "Adam",
    "device": "cpu",
    "num_workers":2,
    'bsize': 16,
    "test_split" :0.3,
    "num_epochs": 2,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "architecture": "arch1",
    "model_name" : "baseline",
    "num_class": 4

}