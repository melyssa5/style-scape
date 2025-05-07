# Train a model
import argparse

import torch
from models.alexnet import AlexNet
from models.dinoV2 import DinoV2
from utils.train import train_one_epoch, evaluate_epoch


def parse_args():
    ''' Extract command line arguements'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        required=True,
        choices=['alexnet', 'bagnet33', 'dinoV2', 'restnet50', 'simclr']
    )
    parser.add_argument('--data_dir', default='data')

    return parser.parse_args()


def get_model(name):
    if name == 'alexnet':
        return AlexNet()
    if name == 'dinoV2':
        return DinoV2()
    
def main():
    '''Train a model'''
    arguments = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 15
    batch_size = 32
    epochs = 10
    # Check model name
    # Intialize model
    # Prepare data, put necessary transforms
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    #for epoch in range(epochs):
        #     loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        #     print(f"[{model_name}] Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f}")

        # acc, report, cm = evaluate_epoch(model, test_loader, device)
        # print(f"\nFinal Test Accuracy: {acc:.2%}")
        # print(report)
        # print(cm)

