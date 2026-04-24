import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class EuroSATDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        
        print(f"Found {len(self.image_paths)} images in {len(self.classes)} classes.")
        
    def load_data(self):
        images = []
        print("Loading images...")
        for i, img_path in enumerate(self.image_paths):
            if i % 5000 == 0:
                print(f"Loaded {i}/{len(self.image_paths)} images")
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array.flatten())
        
        return np.array(images), np.array(self.labels)

def get_dataloaders(root_dir, batch_size=32, seed=42):
    dataset = EuroSATDataset(root_dir)
    X, y = dataset.load_data()
    
    # Split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    
    # Simple normalization (zero mean, unit variance based on train set)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), dataset.classes

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_indices = indices[start:end]
            yield self.X[batch_indices], self.y[batch_indices]
            
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
