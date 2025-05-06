from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, processor, batch_size=32):
    transform = lambda x: processor(x, return_tensors="pt")["pixel_values"][0]
    
    train_set = ImageFolder(f"{data_dir}/train", transform=transform)
    test_set = ImageFolder(f"{data_dir}/stylized", transform=transform)
    
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
             DataLoader(test_set, batch_size=batch_size))