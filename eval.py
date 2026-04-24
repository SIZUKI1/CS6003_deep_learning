import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_utils import get_dataloaders
from model import MLP
import os

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Saved learning_curves.png")

from scipy.ndimage import gaussian_filter

def visualize_weights(params, hidden_dim):
    # W1 is the first layer weights: (InputDim, HiddenDim)
    W1 = params[0]['W']
    
    # Pick top 25 neurons
    n_visualize = 25
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    print("Processing weights for visualization...")
    for i in range(n_visualize):
        ax = axes[i // cols, i % cols]
        # Reshape weight vector of one neuron back to (64, 64, 3)
        w_img = W1[:, i].reshape(64, 64, 3)
        
        # Apply slight smoothing to highlight color trends and reduce high-frequency noise
        w_img_smooth = gaussian_filter(w_img, sigma=1.0)
        
        # Robust normalization: clip extreme values (outliers) then scale to [0, 1]
        vmin, vmax = np.percentile(w_img_smooth, [2, 98])
        w_img_rescaled = np.clip((w_img_smooth - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        ax.imshow(w_img_rescaled)
        ax.axis('off')
        ax.set_title(f'Neuron {i}')
        
    plt.tight_layout()
    plt.savefig('weight_visualization.png', dpi=150)
    print("Saved improved weight_visualization.png")

def evaluate(model_pkl_path):
    with open(model_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    history = data['history']
    args = data['args']
    
    # Reconstruct model and load params
    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = get_dataloaders(args.data_dir)
    
    input_dim = X_train.shape[1]
    output_dim = len(classes)
    
    model = MLP(input_dim, args.hidden_dim, output_dim, activation=args.activation)
    
    # Manually load parameters
    model_params = model.get_params()
    for i in range(len(params)):
        model_params[i]['W'][:] = params[i]['W']
        model_params[i]['b'][:] = params[i]['b']
        
    # Evaluate on test set
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45, ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # Plot history
    plot_history(history)
    
    # Visualize weights
    visualize_weights(params, args.hidden_dim)
    
    # Error Analysis: Save some bad cases
    bad_indices = np.where(preds != y_test)[0]
    os.makedirs('bad_cases', exist_ok=True)
    for i in range(min(5, len(bad_indices))):
        idx = bad_indices[i]
        # X_test[idx] is flattened, normalized. Let's find the original image if possible, 
        # or just reshape the normalized one.
        img_data = X_test[idx].reshape(64, 64, 3)
        # Denormalize roughly for visualization (since we used (X-mean)/std)
        # This is hard without exact mean/std, but let's try a simple [0,1] clip
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
        
        plt.figure()
        plt.imshow(img_data)
        plt.title(f"True: {classes[y_test[idx]]}, Pred: {classes[preds[idx]]}")
        plt.savefig(f"bad_cases/bad_{i}.png")
        plt.close()
    print("Saved some bad cases to bad_cases/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pkl')
    args = parser.parse_args()
    evaluate(args.model)
