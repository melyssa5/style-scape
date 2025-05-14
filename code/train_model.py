# Train a model
import argparse
import torch
from models.alexnet import AlexNet
from models.dinoV2 import DinoV2
from train import train_epoch, evaluate

def train_loop(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=50, save_path="best_model.pth"):
    best_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss
            print(f"New best model saved to {save_path} (Loss: {best_loss:.4f})")

        if test_loader:
            acc = evaluate(model, test_loader, device)
            print(f"Test Accuracy: {acc:.2%}")


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


