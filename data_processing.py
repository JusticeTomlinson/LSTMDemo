import numpy as np

def create_sequences(data, window_size):
    """
    Create sequences and corresponding labels from selected features in a multivariate time series dataset.
    
    Args:
        data (np.array): Input array of shape [num_samples, num_features].
        window_size (int): Number of timesteps in each sequence.
        feature_indices (list of int): Indices of the features to include in the sequences and labels.
    
    Returns:
        sequences (np.array): Array of sequences with shape [num_samples - window_size, window_size, len(feature_indices)].
        labels (np.array): Array of labels with shape [num_samples - window_size, len(feature_indices)].
    """
    sequences = []
    labels = []
    num_samples = len(data)
    for i in range(num_samples - window_size):
        sequences.append(np.array(data.loc[i:(i + window_size) - 1]))  # Select columns by feature_indices
        labels.append(np.array(data["pollution"][i + window_size]))
        
    return np.array(sequences), np.array(labels)