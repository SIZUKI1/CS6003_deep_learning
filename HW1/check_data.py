from data_utils import get_dataloaders
(X_train, y_train), (X_val, y_val), (X_test, y_test), classes = get_dataloaders('EuroSAT_RGB')
print(f"X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
print(f"X_train max: {X_train.max():.4f}, min: {X_train.min():.4f}")
print(f"Shape: {X_train.shape}")
