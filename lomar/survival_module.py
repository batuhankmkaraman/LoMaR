import torch
import torch.nn as nn
import torch.nn.functional as F

class SurvivalModule(nn.Module):
    def __init__(self, config):
        super(SurvivalModule, self).__init__()

        # Predicts a non-negative hazard value for each follow-up time step (e.g., years 1..T).
        # Input:  [B, D_model]  -> Output: [B, T]
        self.hazard_fc = nn.Linear(config['model_embedding_dim'], config['n_future_dx']).to(config['torch_device'])

        # Optional baseline term (per sample) added to the cumulative risk/logit.
        # Input:  [B, D_model]  -> Output: [B, 1]
        self.base_hazard_fc = nn.Linear(config['model_embedding_dim'], 1).to(config['torch_device'])

        # Enforce non-negativity of hazards.
        self.relu = nn.ReLU(inplace=True)

        # Build a fixed upper-triangular mask used to compute cumulative sums over time.
        # Steps:
        #   1) ones(T,T) -> lower-triangular (including diagonal)
        #   2) transpose -> upper-triangular (including diagonal)
        # This lets each time t aggregate hazards from time 1..t.
        mask = torch.ones([config['n_future_dx'], config['n_future_dx']])
        mask = torch.tril(mask, diagonal=0)  # lower-triangular
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False).to(config['torch_device'])  # upper-triangular
        self.upper_triagular_mask = mask


    def hazards(self, x):
        # Raw hazards can be any real value; apply ReLU to ensure they are >= 0.
        raw_hazard = self.hazard_fc(x)   # [B, T]
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard                # [B, T]

    def forward(self, x):
        # Compute per-time hazards.
        hazards = self.hazards(x)        # [B, T]

        # Expand hazards into a [B, T, T] tensor so we can apply a triangular mask and
        # compute a cumulative (time-dependent) quantity for each horizon.
        B, T = hazards.size()            # hazards is [B, T]
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # [B, T, 1] -> [B, T, T]

        # Apply the fixed upper-triangular mask.
        # Interpretation: for each horizon t (last dimension), keep hazards up to that time
        # and zero-out hazards beyond it.
        masked_hazards = expanded_hazards * self.upper_triagular_mask  # [B, T, T]

        # Aggregate hazards across the "time-to-sum" dimension to get a cumulative value per horizon.
        # Result: [B, T]. Add a per-sample baseline term broadcasted over time.
        cum_logit = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)  # [B, T] + [B, 1]
        return cum_logit
