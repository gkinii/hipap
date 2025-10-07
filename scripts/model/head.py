import torch
import torch.nn as nn
import torch.nn.functional as F



class LstmHead(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size,
                 num_layers=10):

        super().__init__()
        self.pred_len = params.pred_len
        self.coord_scale=params.grid_coord_scale
        self.lstm = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.mlp = nn.Linear(hidden_size, 2)  # Output all predictions at once
        self.lnorm=nn.LayerNorm(hidden_size)
        
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        output, (hidden, _) = self.lstm(x)  # Only use final hidden state
        out = self.lnorm(output+x)
        # print(out.shape)
        out = self.mlp(out)  # (b*a, 2*pred_len)

        preds = out.view(b, a, self.pred_len, 2)
        
        return preds*self.coord_scale



class PredictionHead(nn.Module):

    def __init__(self, params, hidden_size):
        super().__init__()

        self.coord_scale=params.grid_coord_scale
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(params.drop_prob)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
                
    def forward(self, x):
        """
        Args:
            x: [b, a, pred_len, h]
        Returns:
            out: [b, a, pred_len, 2]
        """
        # First layer with residual
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual  # Residual connection
        
        # Output projection
        out = self.norm2(out)
        out = self.fc2(out)
        
        return out*self.coord_scale



class LSTMPredictionHead(nn.Module):
    """
    LSTM-based prediction head that processes the prediction sequence.
    """
    def __init__(self,params, hidden_size):
        super().__init__()
        self.lstm_hidden = hidden_size
        self.coord_scale=params.grid_coord_scale
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.projection = nn.Linear(self.lstm_hidden, 2)
                
    def forward(self, x):
        """
        Args:
            x: [b, a, pred_len, h]
        Returns:
            out: [b, a, pred_len, 2]
        """
        b, a, pred_len, h = x.shape
        
        # Reshape to process each 'a' separately: [b*a, pred_len, h]
        x = x.reshape(b * a, pred_len, h)
        
        # LSTM processes the sequence
        lstm_out, _ = self.lstm(x)  # [b*a, pred_len, lstm_hidden]
        
        # Project to output dimension
        out = self.projection(lstm_out)  # [b*a, pred_len, 2]
        
        # Reshape back
        out = out.reshape(b, a, pred_len, 2)
        
        return out*self.coord_scale



# class PredictionHeadLayer(nn.Module):

#     def __init__(self, hidden_size: int, hidden_units=None):
#         super().__init__()

#         layers = []
#         in_features = hidden_size

#         # Hidden MLP layers (with ReLU)
#         if hidden_units:
#             for units in hidden_units:
#                 layers.append(nn.Linear(in_features, units))
#                 layers.append(nn.ReLU())
#                 in_features = units

#         layers.append(nn.Linear(in_features, 2))

#         self.mlp = nn.Sequential(*layers)

#     # ------------------------------------------------------------------ #
#     def forward(self, input_batch):

#         preds = self.mlp(input_batch)

#         return preds


# class PredictionHeadLayer(nn.Module):

#     def __init__(self, hidden_size: int, hidden_units=None):
#         super().__init__()

#         layers = []
#         in_features = hidden_size

#         # Hidden MLP layers (with ReLU)
#         if hidden_units:
#             for units in hidden_units:
#                 layers.append(nn.Linear(in_features, units))
#                 layers.append(nn.ReLU())
#                 in_features = units

#         # Final 11-unit output layer (no activation)
#         layers.append(nn.Linear(in_features, 11))

#         self.mlp = nn.Sequential(*layers)

#     # ------------------------------------------------------------------ #
#     def forward(self, input_batch: dict):

#         x = input_batch["hidden_vecs"]            # (..., h)
#         logits = self.mlp(x)                      # (..., 11)

#         preds = {
#             "agents/position":                    logits[..., 0:3],   # (x,y,z)
#             "agents/orientation":                 logits[..., 3:4],   # θ
#             "agents/position/raw_scale_tril":     logits[..., 4:10],  # 6 params
#             "agents/orientation/raw_concentration": logits[..., 10:11],
#         }

#         if "mixture_logits" in input_batch:
#             preds["mixture_logits"] = input_batch["mixture_logits"]

#         return preds