import torch.nn as nn
from .model_params import ModelParams
import torch.nn as nn
from . import agent_encoder, attention, scene_encoder,scene_attn, head, preprocess, resnet, lstm

class AttentionBlock(nn.Module):

    def __init__(
        self,
        params: ModelParams,
        attn_layers: list,
        mode: str='down',
        hidden_size: int=128
    ):
        super().__init__()

        layers=nn.ModuleList()
        if mode=='down':

            for attn_layer in attn_layers:
                layers.append(attn_layer(params=params,hidden_size=int(hidden_size)))
                layers.append(lstm.LSTMNet(params=params, hidden_size=hidden_size, output_size=hidden_size//2))
                hidden_size=hidden_size//2
        
        else:
            for attn_layer in attn_layers:
                layers.append(attn_layer(params=params,hidden_size=int(hidden_size)))
                layers.append(lstm.LSTMNet(params=params, hidden_size=hidden_size, output_size=hidden_size*2))
                hidden_size=hidden_size*2


        self.seq_layers = nn.Sequential(*layers)


    def forward(self, input):
        out = self.seq_layers(input)
        return out

class HumanRobotInteractionEncoder(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.hidden = 256
        attn_layers=[attention.AgentSelfAlignmentLayer, attention.SelfAttnTransformerLayer]
        self.factor=2**len(attn_layers)

        # Agent encoders
        self.human_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer(
            agent_type='human',
            hidden_size=512,
            output_size=self.hidden,
            params=params)
        self.robot_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer(
            agent_type='robot',
            hidden_size=512,
            output_size=self.hidden,
            params=params)
        
        # Scene processing
        self.scene_preprocess_layer = preprocess.GridPreprocessLayer()
        self.scene_encoder_layer = scene_encoder.ConvOccupancyGridEncoderLayer(
            params,
            hidden_size=self.hidden//self.factor)
        self.scene_cross_attn_layer = scene_attn.SceneCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden//self.factor)
        
        self.atnn_down_block=AttentionBlock(params=params, attn_layers=attn_layers, mode='down', hidden_size=self.hidden)

        self.agent_type_cross_attn_layer= attention.AgentTypeCrossAttentionLayer(
            params,
            hidden_size=self.hidden//self.factor)

    
    def forward(self, input_batch):
        
        scene_enc = self.scene_preprocess_layer(input_batch)
        scene_enc = self.scene_encoder_layer(scene_enc)

        # Agent encoding
        human_emb = self.human_agent_encoding_layer(input_batch)
        robot_emb = self.robot_agent_encoding_layer(input_batch)
        
        human_emb= self.atnn_down_block(human_emb)
        robot_emb= self.atnn_down_block(robot_emb)
        
        human_emb=self.agent_type_cross_attn_layer(human_emb,robot_emb)

        out = self.scene_cross_attn_layer(human_emb, scene_enc)

        return out[:,:,-1:,:]