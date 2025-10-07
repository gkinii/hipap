import torch.nn as nn
from .model_params import ModelParams
import torch.nn as nn
from . import agent_encoder, attention, scene_encoder,scene_attn, head, preprocess, resnet, lstm, encoder
import torch

class HumanRobotInteractionDecoder(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.hidden = 32
        attn_layers=[attention.AgentSelfAlignmentLayer, attention.SelfAttnTransformerLayer]
        self.factor=2**len(attn_layers)
        
        # Scene processing
        # self.scene_preprocess_layer = preprocess.GridPreprocessLayer()
        # self.scene_encoder_layer = scene_encoder.ConvOccupancyGridEncoderLayer(
        #     params,
        #     hidden_size=self.hidden//self.factor)
        # self.scene_cross_attn_layer = scene_attn.SceneCrossAttnTransformerLayer(
        #     params,
        #     hidden_size=self.hidden//self.factor)
        
        self.lstm_decoder=lstm.MLPDecode(params, self.hidden,self.hidden)

        self.atnn_up_block=encoder.AttentionBlock(params=params, attn_layers=attn_layers, mode='up', hidden_size=self.hidden)
        self.head=nn.Linear(self.hidden*self.factor,2)

    def forward(self, input_batch):
        
        out=self.lstm_decoder(input_batch)
        out= self.atnn_up_block(out)
        out=self.head(out)
        
        return out
    

class LearnedPredictionDecoder(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size):
        super().__init__()
        self.pred_len = params.pred_len
        
        self.learned_embedding = nn.Parameter(
            torch.empty(1, 1, self.pred_len, hidden_size)
        )
        nn.init.xavier_uniform_(self.learned_embedding)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )
        
        self.attn_lnorm = nn.LayerNorm(hidden_size)
                
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        
        queries = self.learned_embedding.expand(b, a, -1, -1)  # [b, a, pred_len, h]
        
        queries = queries.reshape(b * a, self.pred_len, h)
        keys_values = input_batch.reshape(b * a, t, h)
        
        mask = torch.triu(
            torch.ones(self.pred_len, t, dtype=torch.bool, device=input_batch.device),
            diagonal=1
        )

        out, _ = self.cross_attn(
            query=queries,
            key=keys_values,
            value=keys_values,
            attn_mask=mask
        )  
        
        out = self.attn_lnorm(out)
        out = out.reshape(b, a, self.pred_len, h)
        
        return out  # [b, a, pred_len, h]