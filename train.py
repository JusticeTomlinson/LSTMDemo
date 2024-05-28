import torch
import numpy as np

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'train_mae': [],
        'val_rmse': [],
        'val_mae': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_rmse, total_mae, train_batches = 0, 0, 0, 0

        # Training loop
        for inputs, targets in train_loader:
            inputs = inputs.squeeze(2).to(device)  # squeeze out placeholder dimension
            targets = targets.unsqueeze(1)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            total_train_loss += loss.item()
            total_rmse += np.sqrt(criterion(outputs, targets).item())
            total_mae += torch.nn.functional.l1_loss(outputs, targets, reduction='sum').item()
            train_batches += 1

        # Average training metrics
        history['train_loss'].append(total_train_loss / train_batches)
        history['train_rmse'].append(total_rmse / train_batches)
        history['train_mae'].append(total_mae / train_batches)

        # Validation loop
        model.eval()
        total_val_loss, val_rmse, val_mae, val_batches = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.squeeze(2).to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1)

                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
                val_rmse += np.sqrt(val_loss.item())
                val_mae += torch.nn.functional.l1_loss(outputs, targets, reduction='sum').item()
                val_batches += 1

        # Average validation metrics
        history['val_loss'].append(total_val_loss / val_batches)
        history['val_rmse'].append(val_rmse / val_batches)
        history['val_mae'].append(val_mae / val_batches)

        # Output current epoch metrics
        print(f'Epoch [{epoch + 1}/{num_epochs}]: '
              f'Train Loss: {history["train_loss"][-1]:.4f}, Train RMSE: {history["train_rmse"][-1]:.4f}, Train MAE: {history["train_mae"][-1]:.4f}, '
              f'Val Loss: {history["val_loss"][-1]:.4f}, Val RMSE: {history["val_rmse"][-1]:.4f}, Val MAE: {history["val_mae"][-1]:.4f}')

    return history