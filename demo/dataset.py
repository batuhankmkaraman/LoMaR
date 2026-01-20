import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import copy

class BreastDataset(Dataset):
    def __init__(self, config, history_masking_id=None):
        # Store config.
        self.config = config

        # Load the master CSV that contains per-patient info:
        # - paths to per-visit embedding .npy files (e.g., npy_-4, ..., npy_0)
        # - future diagnosis labels (e.g., dx_1, ..., dx_K)
        # - split assignment (trainvaltest column)
        past_visits = pd.read_csv(config['path_to_csv'], low_memory=False)

        # Determine history length (number of past visits) from config.
        # We represent visits with relative year codes:
        #   n_past_visits=5  -> [-4, -3, -2, -1,  0]
        n_past_visits = int(config['n_past_visits'])
        history_years = list(range(-(n_past_visits - 1), 1))

        # Optional: synthetic "history masking" to ablate certain visit slots.
        # If enabled, we will treat some visits as missing even if they exist in the CSV.
        # Convention: pattern value 1 => mask (hide) this visit, 0 => keep it if present.
        history_masking_id_dict = None
        if history_masking_id is not None:
            patterns = {
                0: [1, 1, 1, 1, 0],
                1: [1, 1, 1, 0, 0],
                2: [1, 1, 0, 0, 0],
                3: [1, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 0],
                5: [0, 1, 0, 1, 0],
                6: [1, 1, 0, 1, 0],
            }
            # Map each year code (e.g., -4..0) to a {0,1} masking decision.
            keys = history_years
            history_masking_id_dict = dict(zip(keys, patterns[history_masking_id]))

        # Save dataset state.
        self.history_masking_id = history_masking_id
        self.history_masking_id_dict = history_masking_id_dict

        # Keep a copy for indexing and record dataset size.
        self.past_visits = past_visits.copy()
        self.len_data = len(self.past_visits)

        # Map string labels in the CSV to numeric values.
        # Note: Unknown is mapped to -1; training code typically needs to ignore/handle these separately.
        self.label_dict = {
            "Not Malignant": 0,
            "Malignant": 1,
            "Unknown": -1,
        }

    def __len__(self):
        # Number of patient rows in this split.
        return int(self.len_data)

    def __getitem__(self, idx):
        sample = {}

        # Recompute history year codes from config to keep __getitem__ self-contained.
        n_past_visits = int(self.config['n_past_visits'])
        history_years = list(range(-(n_past_visits - 1), 1))
        viscodes = np.array(history_years)

        # Get the CSV row for this patient/sample.
        patient_data = self.past_visits.iloc[idx]

        # We will build:
        # - embeddings: list of per-visit embedding vectors (or zero vectors if missing)
        # - visit_mask: indicates "missing after masking" (0=present, 1=missing/masked)
        # - original_visit_mask: indicates "truly missing in data" (0=present, 1=missing)
        embeddings = []
        visit_mask = []
        original_visit_mask = []

        if self.history_masking_id_dict is not None:
            # Masking enabled: some existing visits are artificially treated as missing.
            for history_year in history_years:
                has_file = (pd.isna(patient_data['npy_' + str(history_year)]) == False)
                is_masked = (self.history_masking_id_dict[history_year] == 1)

                if has_file and (not is_masked):
                    # Load precomputed embedding from disk.
                    visit_embeddings = np.load(patient_data['npy_' + str(history_year)])
                    embeddings.append(visit_embeddings)
                    visit_mask.append(0)
                else:
                    # If missing or masked out, use a zero vector placeholder.
                    # (512 is assumed embedding dim; should match how embeddings were saved.)
                    embeddings.append(np.zeros((512)))
                    visit_mask.append(1)

                # Track the true missingness (ignoring synthetic masking).
                original_visit_mask.append(0 if has_file else 1)
        else:
            # No masking: missingness is purely determined by whether the file path exists in CSV.
            for history_year in history_years:
                if pd.isna(patient_data['npy_' + str(history_year)]) == False:
                    visit_embeddings = np.load(patient_data['npy_' + str(history_year)])
                    embeddings.append(visit_embeddings)
                    visit_mask.append(0)
                    original_visit_mask.append(0)
                else:
                    embeddings.append(np.zeros((512)))
                    visit_mask.append(1)
                    original_visit_mask.append(1)

        # Stack into arrays:
        # - visit_embeddings: [T, D]
        # - visit_mask: [T]
        # - original_visit_mask: [T]
        embeddings = np.stack(embeddings)
        visit_mask = np.array(visit_mask)
        original_visit_mask = np.array(original_visit_mask)

        # Future survival labels across horizons 1..K where K = config['n_future_dx'].
        # The CSV is expected to have columns: dx_1, dx_2, ..., dx_K.
        n_future_dx = int(self.config['n_future_dx'])
        surv_cols = ['dx_' + str(z) for z in range(1, n_future_dx + 1)]

        # Map string labels to numeric; missing values are treated as "Unknown" (-1).
        label = np.array([float(self.label_dict[z]) for z in patient_data[surv_cols].fillna('Unknown')])

        # Package sample dict (NumPy arrays). Often converted to torch tensors in a collate_fn or training loop.
        sample['visit_embeddings'] = embeddings
        sample['visit_mask'] = visit_mask
        sample['original_visit_mask'] = np.array(original_visit_mask)
        sample['viscodes'] = viscodes
        sample['label'] = label

        return sample
