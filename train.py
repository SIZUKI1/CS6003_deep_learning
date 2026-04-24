import numpy as np
import os
import argparse
import pickle
from data_utils import get_dataloaders, DataLoader
from model import MLP
from loss import CrossEntropyLoss
from optimizer import SGD, StepLR

def calculate_accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)

def train(args):
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    train_loader = DataLoader(X_train, y_train, args.batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    output_dim = len(classes)
    
    # Initialize model
    model = MLP(input_dim, args.hidden_dim, output_dim, activation=args.activation)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_params = None
    
    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_accs = []
        
        for X_batch, y_batch in train_loader:
            # Forward
            logits = model.forward(X_batch)
            loss = criterion(logits, y_batch)
            acc = calculate_accuracy(logits, y_batch)
            
            # Backward
            grad = criterion.backward()
            model.backward(grad)
            
            # Update
            optimizer.step()
            
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            
        scheduler.step()
        
        # Validation
        val_logits = model.forward(X_val)
        val_loss = criterion(val_logits, y_val)
        val_acc = calculate_accuracy(val_logits, y_val)
        
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accs)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best parameters
            best_params = model.get_params()
            with open(args.save_path, 'wb') as f:
                pickle.dump({'params': best_params, 'history': history, 'args': args}, f)
            print(f"--> Saved best model with Val Acc: {val_acc:.4f}")
            
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='EuroSAT_RGB')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--lr_step', type=int, default=20)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='best_model.pkl')
    
    args = parser.parse_args()
    train(args)
