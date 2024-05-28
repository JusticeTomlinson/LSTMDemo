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
        sequences.append(np.array(data.loc[i:(i + window_size) - 1]))
        labels.append(np.array(data["pollution"][i + window_size]))
        
    return np.array(sequences), np.array(labels)




def get_final_results(training_dataframe):
    final_vals = [training_dataframe["train_loss"][-1], training_dataframe["val_loss"][-1],\
    training_dataframe["train_rmse"][-1], training_dataframe["val_rmse"][-1],
    training_dataframe["train_mae"][-1], training_dataframe["val_mae"][-1]]
    return final_vals



def sort_results(r1, r2, r3, r4, r5, r6):
    sorted_results = {}
    
    uv_rnn_res = get_final_results(r1)
    mv_rnn_res = get_final_results(r2)
    uv_gru_res = get_final_results(r3)
    mv_gru_res = get_final_results(r4)
    uv_lstm_res = get_final_results(r5)
    mv_lstm_res = get_final_results(r6)
    
    sorted_results["train_losses"] = [uv_rnn_res[0], mv_rnn_res[0], uv_gru_res[0],\
                                    mv_gru_res[0], uv_lstm_res[0], mv_lstm_res[0]] 
    sorted_results["val_losses"] = [uv_rnn_res[1], mv_rnn_res[1], uv_gru_res[1],\
                                    mv_gru_res[1], uv_lstm_res[1], mv_lstm_res[1]] 
    sorted_results["train_rmses"] = [uv_rnn_res[2], mv_rnn_res[2], uv_gru_res[2],\
                                mv_gru_res[2], uv_lstm_res[2], mv_lstm_res[2]] 
    sorted_results["val_rmses"] = [uv_rnn_res[3], mv_rnn_res[3], uv_gru_res[3],\
                                mv_gru_res[3], uv_lstm_res[3], mv_lstm_res[3]] 
    sorted_results["train_maes"] = [uv_rnn_res[4], mv_rnn_res[4], uv_gru_res[4],\
                                mv_gru_res[4], uv_lstm_res[4], mv_lstm_res[4]] 
    sorted_results["val_maes"] = [uv_rnn_res[5], mv_rnn_res[5], uv_gru_res[5],\
                                mv_gru_res[5], uv_lstm_res[5], mv_lstm_res[5]]
    
    return sorted_results