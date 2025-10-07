import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentSelfAlignmentLayer(nn.Module):
    """
    Enables agent to become aware of its temporal identity.
    Agent features are cross-attended with a learned query in the temporal dimension,
    then refined by a temporal LSTM block (replacing the MLP FFN).
    """
    def __init__(self, params, hidden_size, downsample_seq=1):
        super().__init__()

        self.downsample_seq = downsample_seq
        self.hidden_size = hidden_size
        self.ff_dim = params.ff_dim

        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
            )

        # Cross-attention: query is learned (length = out_seq), keys/values are the full time steps
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )
        self.attn_ln = nn.LayerNorm(hidden_size)

        # === LSTM "FFN" block: bottleneck to ff_dim, then back to hidden_size ===
        self.ff_lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.ff_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.ff_lstm2 = nn.LSTM(
            input_size=self.ff_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size)

        # Learned query vector [1, 1, 1, h] -> expanded to [b, a, out_seq, h]
        self.learned_query_vec = nn.Parameter(
            torch.empty(1, 1, 1, hidden_size).uniform_(-1.0, 1.0)
        )

    def forward(self, input_batch):
        # input_batch: [b, a, t, h]
        b, a, t, h = input_batch.shape
        assert h == self.hidden_size

        out_seq = t // self.downsample_seq
        if out_seq < 1:
            raise ValueError("downsample_seq is too large for the given t.")

        # Build learned query: [b, a, out_seq, h]
        learned_query = self.learned_query_vec.repeat(b, a, out_seq, 1)

        # Reshape for MHA (batch_first=True expects [N, S, E])
        # Query length = out_seq, Key/Value length = t
        q = learned_query.view(b * a, out_seq, h)  # [b*a, out_seq, h]
        kv = input_batch.view(b * a, t, h)         # [b*a, t, h]

        attn_out, _ = self.attn_layer(query=q, key=kv, value=kv, attn_mask=None)
        attn_out = attn_out.view(b, a, out_seq, h)

        # Residual to the *query* (same shape), then norm
        attn_out = self.attn_ln(attn_out + learned_query)

        # --- LSTM "FFN" over time per agent (sequence = out_seq) ---
        y = attn_out.contiguous().view(b * a, out_seq, h)  # [b*a, out_seq, h]
        y, _ = self.ff_lstm1(y)                            # [b*a, out_seq, ff_dim]
        y, _ = self.ff_lstm2(y)                            # [b*a, out_seq, h]
        y = y.view(b, a, out_seq, h)

        y = self.ff_dropout(y)
        y = self.ff_ln(y + attn_out)  # residual to block input

        return y


class SelfAttnTransformerLayer(nn.Module):
    """Performs full self-attention across the agent and time dimensions."""
    
    def __init__(
        self,
        params,
        hidden_size,
    ):
        super().__init__()
        

        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        
        self.attn_ln = nn.LayerNorm(hidden_size)
        
        self.ff_layer1 = nn.Linear(hidden_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, hidden_size)
        
        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size)
    
    def forward(self, input_batch):
        
        # [b, a, t, h]
        b, a, t, h = input_batch.shape
        
        input_batch_reshaped = input_batch.reshape(b, a * t, h)
        
        attn_out, _ = self.attn_layer(
            input_batch_reshaped,
            input_batch_reshaped,
            input_batch_reshaped,
            attn_mask=None,
        )

        attn_out = attn_out.reshape(b, a, t, h)
        attn_out = self.attn_ln(attn_out + input_batch)
        
        out = self.ff_layer1(attn_out)
        out = F.relu(out)
        out = self.ff_layer2(out)
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        return out
    


class AgentTypeCrossAttentionLayer(nn.Module):
    def __init__(self, params, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.ff_dim = params.ff_dim

        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(hidden_size)

        # ---- LSTM "FFN" block (bottleneck to ff_dim, back to hidden_size) ----
        self.ff_lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.ff_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.ff_lstm2 = nn.LSTM(
            input_size=self.ff_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size)


    def forward(self, human_embedding, robot_embedding):
        # human_embedding: [b, ha, ht, h]
        # robot_embedding: [b, ra, rt, h]
        b, ha, ht, h = human_embedding.shape
        _, ra, rt, rh = robot_embedding.shape
        assert h == self.hidden_size == rh, "Hidden sizes must match."

        # Flatten agents×time for cross-attention
        q = human_embedding.reshape(b, ha * ht, h)       # [b, ha*ht, h]
        kv = robot_embedding.reshape(b, ra * rt, rh)     # [b, ra*rt, h]

        attn_out, _ = self.attn_layer(
            query=q,
            key=kv,
            value=kv,
            attn_mask=None
        )

        # Residual + norm (unflatten back to [b, ha, ht, h])
        attn_out = attn_out.view(b, ha, ht, h)
        human_embedding = human_embedding  # for clarity
        attn_out = self.attn_ln(human_embedding + attn_out)

        # ---- LSTM "FFN" over time per human agent ----
        # Merge batch and agent dims; sequence is ht
        y = attn_out.contiguous().view(b * ha, ht, h)  # [b*ha, ht, h]

        y, _ = self.ff_lstm1(y)                        # [b*ha, ht, ff_dim]
        y, _ = self.ff_lstm2(y)                        # [b*ha, ht, h]

        y = y.view(b, ha, ht, h)
        y = self.ff_dropout(y)
        y = self.ff_ln(y + attn_out)                   # residual to block input

        return y
