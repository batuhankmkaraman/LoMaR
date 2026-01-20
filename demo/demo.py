import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *

import time

import torch
import copy 
import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score

from lomar.model import *
from demo.dataset import *

def compute_metrics(outputs, labels):
    """
    Compute ROC-AUC metrics for a multi-horizon prediction setting.

    Args:
        outputs: numpy array of shape [B, K] with predicted scores/logits per horizon.
        labels:  numpy array of shape [B, K] with binary labels {0,1} or -1 for unknown.

    Returns:
        List: [avg_rocauc, rocauc_year1, rocauc_year2, ...]
              avg_rocauc ignores NaNs (years where ROC-AUC is undefined).
    """
    rocauc_scores = []

    for i in range(labels.shape[1]):
        # Slice labels/predictions for horizon i (e.g., year i+1).
        year_labels = labels[:, i]
        year_predictions = outputs[:, i]

        # Only evaluate on valid labels (here: -1 means Unknown).
        mask = (year_labels != -1)

        # ROC-AUC is defined only if both classes {0,1} are present in the filtered labels.
        if len(np.unique(year_labels[mask])) == 2:
            rocauc_score = roc_auc_score(year_labels[mask], year_predictions[mask])
            rocauc_scores.append(rocauc_score)
        else:
            rocauc_scores.append(float('nan'))

    # Average ROC-AUC across horizons, ignoring undefined horizons (NaNs).
    avg_rocauc = np.nanmean(rocauc_scores)
    
    # Return average first, then per-horizon scores.
    return [avg_rocauc] + rocauc_scores

def evalute(model, test_dataset, test_loader, config):
        """
        Run a single forward pass over the entire test_loader (assumed full-batch),
        then compute pseudo-group ROC-AUC metrics based on dataset-provided boolean
        columns (pseudo_0, pseudo_1, ...).

        Note:
            - This function currently takes only the *first batch* from test_loader.
              In the demo() setup, batch_size=len(test_dataset), so this corresponds
              to evaluating the full dataset in one pass.
        """
        model.eval()
        with torch.no_grad():
            # Fetch a single batch (in this demo: the full dataset).
            batch = next(iter(test_loader))

            # Labels: [B, K] where K = number of future horizons (e.g., 5 years).
            labels = batch['label'].to(config['torch_device'])

            # Model outputs: expected [B, K].
            outputs = model(batch)

        # Collect results per pseudo split/group.
        pseudo_results = pd.DataFrame()
        for i_pseudo in range(config['n_pseudo']):
            # Ensure there is a boolean column pseudo_i in the dataset dataframe.
            # If absent, default to True for all rows (i.e., use all samples).
            if "pseudo_"+str(i_pseudo) in test_dataset.past_visits.columns:
                pass 
            else: 
                test_dataset.past_visits["pseudo_"+str(i_pseudo)] = True

            # Boolean indexing mask over rows/samples.
            indices = test_dataset.past_visits["pseudo_"+str(i_pseudo)]

            # Filter predictions/labels for this pseudo group and compute ROC-AUCs.
            pseudo_preds = outputs[indices].detach().cpu().numpy()
            pseudo_labels = labels[indices].detach().cpu().numpy()
            pseudo_rocauc = compute_metrics(pseudo_preds, pseudo_labels)

            # Store results in a dataframe for easy averaging/printing.
            pseudo_results.loc[i_pseudo, 'history_masking_id'] = test_dataset.history_masking_id
            cols = ['average', '1_year', '2_year', '3_year', '4_year', '5_year']
            pseudo_results.loc[i_pseudo, cols] = pseudo_rocauc[:6]

        # Average across pseudo groups; indexed by history_masking_id for convenience.
        average_pseudo_results = pd.DataFrame(columns=pseudo_results.columns)
        average_pseudo_results.loc[test_dataset.history_masking_id] = pseudo_results.mean()

        # Pack outputs for downstream analysis / saving.
        res = {}
        res['labels'] = labels.detach().cpu().numpy()
        res['outputs'] = outputs.detach().cpu().numpy()
        res['average_pseudo_results'] = average_pseudo_results
        return res

def demo(config):
    """
    Demo inference script:
      - Loads a trained LoMaR checkpoint
      - Evaluates across multiple history masking settings (0..6)
      - Computes pseudo-group ROC-AUC metrics
      - Saves all outputs into an .npz log file
    """

    the_log = {}
    the_log['config'] = config 

    # Pick device automatically (GPU if available).
    config['torch_device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("config:")
    print(config)
  
    # Create model and load pretrained weights.
    model = LoMaR(config)
    model.load_state_dict(torch.load(config['model_weight_dir']), strict=False)
    model.eval()
    print("Loaded the model, moving to inference.")
    
    inference_log = {}

    # Evaluate multiple synthetic history masking settings (ablation / robustness check).
    for history_masking_id in range(7):
     
        # Dataset is expected to apply the requested history masking internally.
        test_dataset =  BreastDataset(config, history_masking_id)

        # Full-batch loader: one batch contains all samples.
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        print("Starting test for history_masking_id:", history_masking_id)
        start_time = time.time()
        test_evaluation_results = evalute(model, test_dataset, test_loader, config)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        # Store results under the masking id key.
        inference_log[test_dataset.history_masking_id] = test_evaluation_results
        
    # Save the full inference log to disk.
    the_log['inference'] = inference_log
    log_dir = config['results_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    np.savez(log_dir+'log_inference.npz', **the_log)       
    print("Saved results to:")
    print(log_dir+'log_inference.npz')
    print('Done')
