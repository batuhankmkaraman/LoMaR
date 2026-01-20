import torch
import torch.nn as nn
import torch.nn.functional as F

from lomar.visit_aggregator import *
from lomar.survival_module import *

class LoMaR(nn.Module):
    def __init__(self, config):
        super(LoMaR, self).__init__()

        # VisitAggregator:
        #   - Takes a longitudinal sequence of visit embeddings
        #   - Adds temporal/positional signal
        #   - Uses a transformer encoder to contextualize visits
        #   - Pools over time to produce a single patient/history representation
        self.visit_transformer = VisitAggregator(config)

        # SurvivalModule:
        #   - Maps the pooled patient embedding to time-dependent risk / hazard outputs
        #   - Produces one score per follow-up horizon (e.g., 1..T years)
        self.classifier = SurvivalModule(config)

        # Dropout applied to the pooled history embedding before the survival head.
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

        # Cache device for consistent tensor placement.
        self.torch_device = config['torch_device']

    def forward(self, batch):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        # Move inputs to device and ensure floating type for projections/transformer ops.
        visit_embeddings = batch['visit_embeddings'].to(self.torch_device).float()
        mask = batch['visit_mask'].to(self.torch_device).float() # Mask is cast to bool inside the visit aggregator.

        # 1) Pool the longitudinal visit sequence into a single history embedding per sample.
        # Output: [B, D_model]
        complete_history_embedding = self.visit_transformer(visit_embeddings, mask)

        # 2) Apply dropout + survival head to obtain time-dependent risk scores/logits.
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(complete_history_embedding))
        return risk_prediction_logits
