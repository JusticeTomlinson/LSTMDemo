import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers) -> None:
        super(RNN, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x -> batch_size, seq, input_size
        self.fc = nn.Linear(hidden_size, output_size) # may need to change these params


    def forward(self, x, future_num=0):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # process the last output

        # Future predictions if needed
        outputs = [out]
        for _ in range(future_num):
            out, hn = self.rnn(out.unsqueeze(1), hn)  # Feed the last output back into the RNN
            out = self.fc(out[:, -1, :])
            outputs.append(out)

        return torch.stack(outputs, dim=1) if future_num > 0 else outputs[0]
         
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers) -> None:
        super(GRU, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, future_num=0):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])

        outputs = [out]
        for _ in range(future_num):
            out, hn = self.gru(out.unsqueeze(1), hn)
            out = self.fc(out[:, -1, :])
            outputs.append(out)

        return torch.stack(outputs, dim=1) if future_num > 0 else outputs[0]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output one value per sequence

    def forward(self, x, future_num=0):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Last time step

        # Future predictions
        outputs = [out]
        for _ in range(future_num):
            out, (hn, cn) = self.lstm(out.unsqueeze(1), (hn, cn))
            out = self.fc(out[:, -1, :])
            outputs.append(out)

        return torch.stack(outputs, dim=1) if future_num > 0 else outputs[0]
