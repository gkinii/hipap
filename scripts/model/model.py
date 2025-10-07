import torch.nn as nn
from .model_params import ModelParams
import torch.nn as nn
from . import agent_encoder, attention, scene_encoder,scene_attn, head, preprocess, resnet, lstm, attention_lstm, decoder

class HumanRobotInteractionTransformer(nn.Module):

    def __init__(
        self,
        params: ModelParams,
    ):
        super().__init__()

        self.hidden_size = 512
        self.human_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer(
            agent_type='human',
            hidden_size=512,
            output_size=self.hidden_size,
            params=params)
        # self.robot_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer(
        #     agent_type='robot',
        #     hidden_size=512,
        #     output_size=self.hidden_size,
        #     params=params)
        
        self.human_self_alignment_layer = attention.AgentSelfAlignmentLayer(
            params,
            hidden_size=self.hidden_size,
            downsample_seq=1)
        


        self.agent_type_cross_attn_layer= attention.AgentTypeCrossAttentionLayer(
            params,
            hidden_size=self.hidden_size)
        self.human_self_attn_layer= attention.SelfAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size)
        


        self.scene_preprocess_layer=preprocess.GridPreprocessLayer(params)
        self.scene_encoder_layer= scene_encoder.ConvOccupancyGridEncoderLayer(
            params,
            hidden_size=self.hidden_size)
        self.scene_cross_attn_layer=scene_attn.SceneCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size)

        self.prediction_head_layer=head.LstmHead(
            params,
            hidden_size=self.hidden_size,
            num_layers=2)
        self.mlp=nn.Linear(2,self.hidden_size)
        

    def forward(self, input_batch):
        
        scene_enc=self.scene_preprocess_layer(input_batch)
        scene_enc = self.scene_encoder_layer(scene_enc)

        human_emb = self.human_agent_encoding_layer(input_batch)
        # robot_emb = self.robot_agent_encoding_layer(input_batch)
        
        human_emb= self.human_self_alignment_layer(human_emb)
        # robot_emb= self.human_self_alignment_layer(robot_emb)

        # out=self.agent_type_cross_attn_layer(human_emb,robot_emb)
        out=self.human_self_attn_layer(human_emb)

        out = self.scene_cross_attn_layer(out,scene_enc)
        out=self.human_self_attn_layer(out)
        out = self.scene_cross_attn_layer(out,scene_enc)
        out=self.human_self_attn_layer(out)

        out = self.prediction_head_layer(out)



        return out
    



class DummyModel(nn.Module):

    def __init__(
        self,
        params: ModelParams,
    ):
        super().__init__()

        self.hidden_size = 256
        self.human_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer(
            agent_type='human',
            hidden_size=512,
            output_size=self.hidden_size,
            params=params)
        
        self.human_self_alignment_layer = attention.AgentSelfAlignmentLayer(
            params,
            hidden_size=self.hidden_size,
            downsample_seq=1)
        self.human_self_attn_layer= attention.SelfAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size)

        self.scene_preprocess_layer=preprocess.GridPreprocessLayer(params,scale_coords=False)
        self.vision_transformer=scene_encoder.TransformerOccupancyGridEncoderLayer(params,prefilter=False,hidden_size=self.hidden_size)
        self.scene_cross_attn_layer=scene_attn.PatchCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size)

        # self.prediction_head_layer=head.LstmHead(
        #     params,
        #     hidden_size=self.hidden_size,
        #     num_layers=2)
        
        self.learned_prediction_layer=decoder.LearnedPredictionDecoder(
            params,
            hidden_size=self.hidden_size
        )

        self.head=head.LSTMPredictionHead(params,self.hidden_size)

    def forward(self, input_batch):
        
        raw_gird=input_batch['map_image']
        scene_grid, scene_coord=self.scene_preprocess_layer(input_batch)
        scene_enc = self.vision_transformer(scene_grid, scene_coord)

        # print(scene_grid)
        human_emb = self.human_agent_encoding_layer(input_batch)
        # robot_emb = self.robot_agent_encoding_layer(input_batch)
        
        human_emb= self.human_self_alignment_layer(human_emb)
        # robot_emb= self.human_self_alignment_layer(robot_emb)

        # out=self.agent_type_cross_attn_layer(human_emb,robot_emb)
        # out=self.human_self_attn_layer(human_emb)

        # out = self.scene_cross_attn_layer(out,scene_enc)
        # out=self.human_self_attn_layer(out)
        human_emb = self.scene_cross_attn_layer(human_emb,scene_enc)
        
        out = self.learned_prediction_layer(human_emb)
        out = self.scene_cross_attn_layer(out,scene_enc)
        # out = self.human_self_alignment_layer(out)
        # out = self.scene_cross_attn_layer(out,scene_enc)
        # out=self.human_self_attn_layer(out)

        # out = self.prediction_head_layer(out)
        out = self.head(out)


        return out, raw_gird, scene_coord
    