import torch
import matplotlib.pyplot as plt


def plot_metrics(train_history, title='Training and Validation Performance Metrics'):
    """Plots combined graphs for training and validation metrics over epochs.
    
    Args:
        train_history (dict): A dictionary containing lists of metrics per epoch for both training and validation.
        title (str): Overall title for the plots.
    """
    metrics = ['loss', 'rmse', 'mae']
    num_metrics = len(metrics)
    plt.figure(figsize=(12, 4 * num_metrics))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(num_metrics, 1, i) 
        train_metric = train_history[f'train_{metric}']
        val_metric = train_history[f'val_{metric}']

        if isinstance(train_metric[0], torch.Tensor):
            train_metric = [x.cpu().detach().numpy() if x.is_cuda else x.numpy() for x in train_metric]
            val_metric = [x.cpu().detach().numpy() if x.is_cuda else x.numpy() for x in val_metric]

        plt.plot(train_metric, label=f'Train {metric.capitalize()}')
        plt.plot(val_metric, label=f'Val {metric.capitalize()}')

        plt.title(f'{metric.upper()} Over Time')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()

    plt.tight_layout(pad=2.0)
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()
