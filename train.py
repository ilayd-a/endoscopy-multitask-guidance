import torch
from transformers import SegformerForSemanticSegmentation


def train(model, train_loader, optimizer, classes_to_train):
    print("Training model")
    model.train()


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0').to(device)
    print(model)
