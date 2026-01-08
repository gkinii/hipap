from .embed import AgentTemporalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class FeatureConcatAgentEncoderLayer(nn.Module):
    
    def __init__(self,
                agent_type, 
                params,
                hidden_size,
                output_size
                ):
        super().__init__()
        
        agents_feature_config= getattr(params, f"{agent_type}_agents_feature_config")
        num_encoders = len(agents_feature_config) + 2
        input_size = num_encoders * (hidden_size)
        
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Feature Embeddings
        for key, layer in agents_feature_config.items():
            self.agent_feature_embedding_layers.append(
                layer(key, params, hidden_size)
            )
        
        # Temporal Embedding
        self.agent_feature_embedding_layers.append(
            AgentTemporalEncoder(
                list(agents_feature_config.keys())[0],
                params,
                hidden_size
            )
        )

        self.ff_layer = nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True
        )

        self.lnorm=nn.LayerNorm(output_size)
        self.ff_dropout = nn.Dropout(params.drop_prob)

    def forward(self, input_batch: Dict[str, torch.Tensor]):
        
        layer_embeddings = []
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch)
            
            # flatten last two dimensions
            original_shape = layer_embedding.shape
            new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)
            layer_embedding = layer_embedding.reshape(new_shape)

            layer_embeddings.append(layer_embedding)

        embedding = torch.cat(layer_embeddings, dim=-1)
        # Apply final feedforward layer
        out = self.ff_layer(embedding)
        out= self.lnorm(out)
        out = self.ff_dropout(out)
        
        return out



class FeatureAddAgentEncoderLayer(nn.Module):
    """
    MLP that connects all features
    """
    
    def __init__(self,
                 agent_type, 
                 params,
                 hidden_size,
                 output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        agents_feature_config = getattr(params, f"{agent_type}_agents_feature_config")
        print(agents_feature_config)
        
        self.num_encoders = len(agents_feature_config) + 1
        self.input_size = self.num_encoders * self.hidden_size
        
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Feature Embeddings
        for key, layer in agents_feature_config.items():
            self.agent_feature_embedding_layers.append(
                layer(key, params, hidden_size)
            )
        
        # Temporal Embedding
        self.agent_temporal_embedding_layer = AgentTemporalEncoder(
            list(agents_feature_config.keys())[0],
            params,
            hidden_size
        )
        
        self.lnorm = nn.LayerNorm(self.input_size)

        self.ff_layer = nn.Sequential(
            nn.Linear(self.input_size, output_size),
            # nn.ReLU(),
            # nn.Linear(output_size*4, output_size*2),
            # nn.ReLU(),
            # nn.Linear(output_size*2, output_size),
        )

        self.ff_dropout = nn.Dropout(params.drop_prob)

    def _flatten_embedding(self, layer_embedding):
        original_shape = layer_embedding.shape
        new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)
        flat_layer_embedding = layer_embedding.reshape(new_shape)
        return flat_layer_embedding

    def forward(self, input_batch: Dict[str, torch.Tensor]):
        layer_embeddings = []
        
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch)
            
            # flatten last two dimensions
            layer_embedding = self._flatten_embedding(layer_embedding)
            layer_embeddings.append(layer_embedding)
            
            # print(layer)
            # print(layer_embedding.shape)

        # [B, A, T, num_enc * H]
        features = torch.cat(layer_embeddings, dim=-1)
        
        temporal_embedding = self.agent_temporal_embedding_layer(input_batch)
        temporal_embedding = temporal_embedding.repeat(
            1, 1, 1, self.num_encoders
        )  # [B, A, T, num_enc * H]

        x = features + temporal_embedding
    
        out = self.ff_layer(self.lnorm(x))
        
        return out

class FeatureAttnAgentEncoderLearnedLayer(nn.Module):
    """Independently encodes features and attends to them.
    
    Agent features are cross-attended with a learned query or hidden_vecs instead
    of MLP.
    """
    
    def __init__(self, params):
        super().__init__()
        
        # Set name attribute for compatibility
        self.name = getattr(params, 'name', self.__class__.__name__)
        
        # Multi-head attention layer
        self.num_heads = params.num_heads
        self.hidden_size = params.hidden_size
        
        # PyTorch MultiheadAttention expects embed_dim to be divisible by num_heads
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=params.hidden_size,
            num_heads=params.num_heads,
            batch_first=True,  # Important for handling batch dimension first
            dropout=0.0  # Add dropout if needed in params
        )
        
        # Layer normalization
        self.attn_ln = nn.LayerNorm(params.hidden_size, eps=params.ln_eps)
        
        # Feedforward layers
        self.ff_layer1 = nn.Linear(
            in_features=params.hidden_size,
            out_features=params.transformer_ff_dim,
            bias=True
        )
        
        self.ff_layer2 = nn.Linear(
            in_features=params.transformer_ff_dim,
            out_features=params.hidden_size,
            bias=True
        )
        
        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(params.hidden_size, eps=params.ln_eps)
        
        # Agent feature embedding layers
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Position Feature
        position_layer_class = params.agents_feature_config[params.agents_position_key]
        self.agent_feature_embedding_layers.append(
            position_layer_class(
                params.agents_position_key, 
                params.hidden_size - 8, 
                params
            )
        )
        
        # Other Feature Embeddings
        for key, layer_class in params.agents_feature_config.items():
            if key == params.agents_position_key:
                continue
            self.agent_feature_embedding_layers.append(
                layer_class(key, params.hidden_size - 8, params)
            )
        
        # Temporal Embedding
        first_key = list(params.agents_feature_config.keys())[0]
        self.agent_feature_embedding_layers.append(
            AgentTemporalEncoder(
                first_key,
                params.hidden_size - 8,
                params
            )
        )
        
        # Learned query vector - initialized with uniform distribution
        self.learned_query_vec = nn.Parameter(
            torch.empty(1, 1, 1, 1, params.hidden_size).uniform_(-1.0, 1.0)
        )
    
    def _build_learned_query(self, input_batch):
        """Converts self.learned_query_vec into a learned query vector."""
        # Get dimensions from the input
        b = input_batch['agents/position'].shape[0]
        num_agents = input_batch['agents/position'].shape[1]
        num_steps = input_batch['agents/position'].shape[2]
        
        # Tile the learned query vector
        # [b, num_agents, num_steps, 1, h]
        return self.learned_query_vec.repeat(b, num_agents, num_steps, 1, 1)
    
    def forward(self, input_batch: Dict[str, torch.Tensor], 
                training: Optional[bool] = None):
        input_batch = input_batch.copy()
        
        # Collect embeddings and masks from all layers
        layer_embeddings = []
        layer_masks = []
        
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch)
            layer_embeddings.append(layer_embedding)
            # layer_masks.append(layer_mask)
        
        # Concatenate all embeddings along feature dimension (axis 3)
        embedding = torch.cat(layer_embeddings, dim=3)
        
        # Get dimensions
        b = embedding.shape[0]
        a = embedding.shape[1]
        t = embedding.shape[2]
        n = embedding.shape[3]
        
        # Create one-hot encoding for feature IDs
        # [1, 1, 1, N, 8]
        one_hot = F.one_hot(torch.arange(0, n, device=embedding.device), num_classes=8).float()
        one_hot = one_hot.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # [b, a, t, N, 8]
        one_hot_id = one_hot.repeat(b, a, t, 1, 1)
        
        # Concatenate one-hot IDs to embeddings
        embedding = torch.cat([embedding, one_hot_id], dim=-1)
        
        # Concatenate masks
        attention_mask = torch.cat(layer_masks, dim=-1)
        
        # Build learned query
        learned_query = self._build_learned_query(input_batch)
        
        # Reshape for multi-head attention
        # PyTorch MultiheadAttention expects: (batch * agents * time, seq_len, hidden_size)
        bat = b * a * t
        
        # Reshape tensors
        learned_query_reshaped = learned_query.view(bat, 1, self.hidden_size)

        embedding_reshaped = embedding.view(bat, n, -1)

        
        # Create attention mask for PyTorch MultiheadAttention
        # PyTorch expects shape: (seq_len, seq_len) or (batch * num_heads, seq_len, seq_len)
        # Since we're doing cross-attention, we need (1, n) mask
        attn_mask_reshaped = attention_mask.view(bat, n)
        
        # Convert boolean mask to float mask (True -> 0, False -> -inf)
        attn_mask_float = torch.where(
            attn_mask_reshaped.unsqueeze(1),  # (bat, 1, n)
            torch.zeros_like(attn_mask_reshaped.unsqueeze(1), dtype=torch.float32),
            torch.full_like(attn_mask_reshaped.unsqueeze(1), float('-inf'), dtype=torch.float32)
        )
        
        # Apply multi-head attention
        attn_out, attn_weights = self.attn_layer(
            query=learned_query_reshaped,
            key=embedding_reshaped,
            value=embedding_reshaped,
            key_padding_mask=~attn_mask_reshaped,  # PyTorch uses True for padded positions
            need_weights=True,
            average_attn_weights=True  # Average across heads
        )
        
        # Reshape attention output back to original dimensions
        # attn_out shape: (bat, 1, hidden_size) -> (b, a, t, hidden_size)
        attn_out = attn_out.view(b, a, t, self.hidden_size)
        
        # Reshape attention weights if needed
        # attn_weights shape: (bat, 1, n) -> (b, a, t, num_heads, 1, n)
        attn_score = None
        if attn_weights is not None:
            # Note: PyTorch returns averaged weights, so we need to unsqueeze for compatibility
            attn_score = attn_weights.view(b, a, t, 1, 1, n)
        
        # Apply layer normalization to attention output
        attn_out_normalized = self.attn_ln(attn_out)
        
        # Feedforward network
        out = self.ff_layer1(attn_out_normalized)
        out = F.relu(out)
        out = self.ff_layer2(out)
        out = self.ff_dropout(out)
        
        # Residual connection and final layer norm
        out = self.ff_ln(out + attn_out_normalized)
        # Update input batch
        input_batch['hidden_vecs'] = out
        if attn_score is not None:
            input_batch[f'attn_scores/{self.name}'] = attn_score
        
        return input_batch

