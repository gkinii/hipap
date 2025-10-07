import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMNet(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size,
                 output_size,
                 num_layers=2):

        super().__init__()
        self.pred_len = params.pred_len
        self.lstm = nn.LSTM(input_size=hidden_size,
                           hidden_size=output_size,
                           num_layers=num_layers,
                           batch_first=True)
        
        
        self.lstm_ln = nn.LayerNorm(output_size)
        
        self.ff_layer1 = nn.Linear(output_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, output_size)

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(output_size)
        
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        output, (hidden, _) = self.lstm(x)
        
        h=output.shape[-1]
        lstm_output=output.view(b, a, t, h)
        # lstm_output= self.lstm_ln(lstm_output+input_batch)

        # out=self.ff_layer1(lstm_output)
        # out = F.relu(out)
        # out = self.ff_layer2(out)
        # out=self.ff_dropout(out)
        # out = self.ff_ln(out + lstm_output)
        
        return lstm_output

class LSTMDecode(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size,
                 output_size,
                 num_layers=2):

        super().__init__()
        self.pred_len = params.pred_len
        print("PRED LEN",self.pred_len)
        self.lstm = nn.LSTM(input_size=hidden_size,
                           hidden_size=output_size,
                           num_layers=num_layers,
                           batch_first=True)
        
        
        self.lstm_ln = nn.LayerNorm(output_size)
        
        self.ff_layer1 = nn.Linear(output_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, output_size)

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(output_size)
        
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        output, (hidden, _) = self.lstm(x)
        
        h=output.shape[-1]
        lstm_output=output.view(b, a, t, h)
        lstm_output=lstm_output.repeat(1,1,self.pred_len,1)

        
        return lstm_output


class MLPDecode(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size,
                 output_size,
                 num_layers=2):

        super().__init__()
        self.pred_len = params.pred_len
        self.linear_layer = nn.Linear(hidden_size, output_size)
        
        self.linear_ln = nn.LayerNorm(output_size)
        
        self.ff_layer1 = nn.Linear(output_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, output_size)

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(output_size)
        
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        output= self.linear_layer(x)
        
        h=output.shape[-1]
        linear_output=output.view(b, a, t, h)
        linear_output= self.linear_ln(linear_output+input_batch)

        out=self.ff_layer1(linear_output)
        out = F.relu(out)
        out = self.ff_layer2(out)
        out=self.ff_dropout(out)
        out = self.ff_ln(out + linear_output)
        out=out.repeat(1,1,self.pred_len,1)
        return out
