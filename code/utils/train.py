import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


# Sources used
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html 

def train_one_epoch(model, dataloader, optimizer, loss_fn, device='cuda'):
    """Train model for one epoch without progress bars."""
    model.train()
    total_loss = 0
    
    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_epoch(model, test_loader, device='cuda', print_results=True):
   
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays for sklearn
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Generate metrics
    results = {
        'accuracy': (all_preds == all_targets).mean(),
        'report': classification_report(
            all_targets, all_preds,
            target_names=test_loader.dataset.classes,
            digits=4
        ),
        'confusion_matrix': pd.DataFrame(
            confusion_matrix(all_targets, all_preds),
            index=test_loader.dataset.classes,
            columns=test_loader.dataset.classes
        )
    }
    
    if print_results:
        print("\n" + "="*50)
        print(f"{'EVALUATION METRICS':^50}")
        print("="*50)
        print("\nCLASSIFICATION REPORT:")
        print(results['report'])
        
        print("\nCONFUSION MATRIX (Top 5 Classes):")
        print(results['confusion_matrix'].iloc[:5, :5])
        print(f"\nOVERALL ACCURACY: {results['accuracy']:.2%}")
        print("="*50)
    
    return results