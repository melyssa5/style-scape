from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, test_loader, device, print_results=True):
    """Returns and prints full evaluation metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Generate reports
    clf_report = classification_report(
        all_targets, 
        all_preds,
        target_names=test_loader.dataset.classes,
        digits=4
    )
    
    conf_matrix = pd.DataFrame(
        confusion_matrix(all_targets, all_preds),
        index=test_loader.dataset.classes,
        columns=test_loader.dataset.classes
    )
    
    # Print formatted results
    if print_results:
        print("\n" + "="*50)
        print(f"{'EVALUATION METRICS':^50}")
        print("="*50)
        print("\nCLASSIFICATION REPORT:")
        print(clf_report)
        
        print("\nCONFUSION MATRIX (Top 5 Classes):")
        print(conf_matrix.iloc[:5, :5])  # Show subset for readability
        
        acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean()
        print(f"\nOVERALL ACCURACY: {acc.item():.2%}")
        print("="*50 + "\n")
    
    return {
        "accuracy": acc.item(),
        "report": clf_report,
        "confusion_matrix": conf_matrix
    }