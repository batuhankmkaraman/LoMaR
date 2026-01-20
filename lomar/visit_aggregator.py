import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VisitAggregator(nn.Module):
    def __init__(self, config):        
        super(VisitAggregator, self).__init__()
        
        # Fixed (non-trainable) sinusoidal positional encodings for temporal order of visits.
        # Shape: [T, D_model]. We register it as a parameter with requires_grad=False so it moves with the module.
        self.temporal_encoding = nn.Parameter(
            self.generate_positional_encodings(
                config['n_past_visits'],
                config['model_embedding_dim']
            ),
            requires_grad=False
        ).to(config['torch_device'])

        # Project raw input embeddings (D_in) into the model/transformer embedding space (D_model).
        self.projection = nn.Linear(
            config['input_embedding_dim'],
            config['model_embedding_dim']
        ).to(config['torch_device'])

        # Single transformer encoder layer to contextualize visit embeddings across time.
        # NOTE: TransformerEncoderLayer expects inputs as [seq_len, batch, d_model] by default.
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config['model_embedding_dim'],
            nhead=config['n_heads'],
            dropout=config['global_do_rate']
        ).to(config['torch_device'])

        # Dropout applied right after projection (standard regularization for embeddings).
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

    def forward(self, visit_embeddings, mask):
        # 1) Map visit embeddings into the transformer hidden dimension.
        # Input shape (typical): [B, T, D_in] -> after projection: [B, T, D_model]
        visit_embeddings = self.projection(visit_embeddings)
        visit_embeddings = self.dropout(visit_embeddings)
        
        # 2) Inject temporal order information (positional encoding).
        # Broadcasting adds [T, D_model] to [B, T, D_model].
        visit_embeddings = visit_embeddings + self.temporal_encoding
        
        # 3) Prepare the padding mask for the transformer.
        # src_key_padding_mask expects shape [B, T] where True indicates positions to ignore (padding).
        transformer_mask = mask.bool() if mask is not None else None
        
        # 4) Transformer expects [T, B, D_model] so we transpose time and batch dims.
        visit_embeddings = visit_embeddings.permute(1, 0, 2)  # [B, T, D] -> [T, B, D]
        pooled = self.transformer(visit_embeddings, src_key_padding_mask=transformer_mask)  # [T, B, D]
        
        # 5) Pool across the time dimension to get a single vector per patient/sequence.
        # If we have a padding mask, compute a masked mean; otherwise, plain mean.
        if transformer_mask is not None:
            # Invert padding mask so True = "valid token to include".
            valid_mask = ~transformer_mask  # [B, T]

            # Expand for broadcasting over embedding dimension and match [T, B, 1] layout.
            expanded_mask = valid_mask.unsqueeze(-1).transpose(0, 1)  # [B, T, 1] -> [T, B, 1]

            # Zero-out padded positions before summation.
            pooled = pooled * expanded_mask  # [T, B, D] * [T, B, 1]

            # Sum over time and divide by number of valid tokens (masked mean).
            pooled_sum = pooled.sum(dim=0)             # [B, D]
            counts = expanded_mask.sum(dim=0)          # [B, 1]
            pooled = pooled_sum / counts.clamp(min=1)  # avoid divide-by-zero
        else:
            # No padding: simple mean over time.
            pooled = pooled.mean(dim=0)  # [B, D]
        
        # pooled is already [B, D]; squeeze is harmless if an extra singleton dim exists.
        return pooled.squeeze(1)
    
    @staticmethod
    def generate_positional_encodings(length, model_embedding_dim):
        # Standard sinusoidal positional encoding used in the original Transformer paper.
        # Returns tensor of shape [length, model_embedding_dim].
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_embedding_dim, 2) * -(math.log(10000.0) / model_embedding_dim))
        positional_encodings = torch.zeros(length, model_embedding_dim)
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        return positional_encodings

