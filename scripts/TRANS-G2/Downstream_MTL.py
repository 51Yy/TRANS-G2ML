import os
import yaml
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaModel,
    RobertaConfig,
    RobertaTokenizer,
)

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

from torchmetrics import R2Score, SpearmanCorrCoef, MeanAbsoluteError

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Downstream_Dataset, DataAugmentation


class PeriodicSubsamplingDataLoader:

    def __init__(self, dataset, batch_size, subsample_size=None, subsample_ratio=None, 
                 random_seed=42, num_workers=0, shuffle=True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.base_seed = random_seed
        self.epoch = 0
        
        if subsample_size is not None:
            self.subsample_size = min(subsample_size, len(dataset))
        elif subsample_ratio is not None:
            self.subsample_size = max(1, int(len(dataset) * subsample_ratio))
        else:
            self.subsample_size = len(dataset)
            
        print(f"Created PeriodicSubsamplingDataLoader with {len(dataset)} total samples, "
              f"using {self.subsample_size} samples per epoch")
        
        self.dataloader = self._create_dataloader()
        
    def _create_dataloader(self):

        current_seed = self.base_seed + self.epoch
        
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        indices = torch.randperm(len(self.dataset))[:self.subsample_size].tolist()
        
        subset = torch.utils.data.Subset(self.dataset, indices)
        
        return DataLoader(
            subset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def update_epoch(self, epoch=None):

        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
            
        self.dataloader = self._create_dataloader()
        return self
    
    def __iter__(self):

        return iter(self.dataloader)
    
    def __len__(self):

        return len(self.dataloader)


"""Layer-wise learning rate decay"""

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)

"""Model"""

"""MultiTask Model with Uncertainty Weighting"""
class MultiTaskModelWithUncertainty(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(MultiTaskModelWithUncertainty, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        
        # task1
        self.Task1Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

        # task2
        self.Task2Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

        self.log_sigma1 = nn.Parameter(torch.zeros(1)) 
        self.log_sigma2 = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        task1_output = self.Task1Regressor(logits)
        task2_output = self.Task2Regressor(logits)
        return task1_output, task2_output, self.log_sigma1, self.log_sigma2

"""Uncertainty Weighted Loss"""
def compute_uncertainty_weighted_loss(loss_task1, loss_task2, log_sigma1, log_sigma2):
    # Precision terms (1 / sigma^2)
    precision1 = torch.exp(-log_sigma1)
    precision2 = torch.exp(-log_sigma2)

    weighted_loss_task1 = precision1 * loss_task1
    weighted_loss_task2 = precision2 * loss_task2

    regularization = log_sigma1 + log_sigma2

    total_loss = weighted_loss_task1 + weighted_loss_task2 + regularization

    return total_loss

"""Train"""
def train_multitask_with_separate_data(model, optimizer, scheduler, loss_fn, task1_dataloader, task2_dataloader, device):
    """
    Using uncertainty weighting or average loss to train the multi-task model with separate data for each task.
    """
    
    model.train()
    
    task1_len = len(task1_dataloader)
    task2_len = len(task2_dataloader)
    
    max_steps = max(task1_len, task2_len)
    
    task1_iter = iter(task1_dataloader)
    task2_iter = iter(task2_dataloader)
    
    for step in range(max_steps):

        if step < task1_len:
            task1_batch = next(task1_iter)
        else:
            task1_iter = iter(task1_dataloader)
            task1_batch = next(task1_iter)
        
        if step < task2_len:
            task2_batch = next(task2_iter)
        else:
            task2_iter = iter(task2_dataloader)
            task2_batch = next(task2_iter)
        
        task1_input_ids = task1_batch["input_ids"].to(device)
        task1_attention_mask = task1_batch["attention_mask"].to(device)
        task1_labels = task1_batch["prop"].to(device)

        task2_input_ids = task2_batch["input_ids"].to(device)
        task2_attention_mask = task2_batch["attention_mask"].to(device)
        task2_labels = task2_batch["prop"].to(device)

        optimizer.zero_grad()

        task1_outputs, _, log_sigma1, _ = model(task1_input_ids, task1_attention_mask)
        _, task2_outputs, _, log_sigma2 = model(task2_input_ids, task2_attention_mask)

        loss_task1 = loss_fn(task1_outputs.squeeze(), task1_labels.squeeze())
        loss_task2 = loss_fn(task2_outputs.squeeze(), task2_labels.squeeze())

        total_loss = compute_uncertainty_weighted_loss(loss_task1, loss_task2, log_sigma1, log_sigma2)
        
        # # average loss
        # total_loss = (loss_task1 + loss_task2) / 2

        total_loss.backward()
        optimizer.step()
        scheduler.step()
    
    return None

def test_multitask_with_separate_data(model, loss_fn, train_task1_dataloader, train_task2_dataloader, test_task1_dataloader, test_task2_dataloader, 
                                      device, scaler_task1, scaler_task2):

    train_loss = 0
    test_loss = 0
    
    # Initialize variables for both tasks
    metrics = {
        "task1": {"train_loss": 0, "test_loss": 0, "train_pred": torch.tensor([]), "train_true": torch.tensor([]), 
                  "test_pred": torch.tensor([]), "test_true": torch.tensor([])},
        "task2": {"train_loss": 0, "test_loss": 0, "train_pred": torch.tensor([]), "train_true": torch.tensor([]), 
                  "test_pred": torch.tensor([]), "test_true": torch.tensor([])}
    }

    model.eval()

    # Evaluate Task 1
    with torch.no_grad():
        for step, batch in enumerate(train_task1_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device)

            task1_output, _, _, _ = model(input_ids, attention_mask)
            outputs = task1_output.float()
            
            outputs_cpu = outputs.cpu().detach()
            prop_cpu = prop.cpu().detach()
            
            outputs_np = scaler_task1.inverse_transform(outputs_cpu.reshape(-1, 1))
            prop_np = scaler_task1.inverse_transform(prop_cpu.reshape(-1, 1))
            
            outputs_tensor = torch.from_numpy(outputs_np.flatten()).float() ###
            prop_tensor = torch.from_numpy(prop_np.flatten()).float()

            loss = loss_fn(outputs_tensor, prop_tensor)
            metrics["task1"]["train_loss"] += loss.item() * len(prop_tensor)

            metrics["task1"]["train_pred"] = torch.cat([metrics["task1"]["train_pred"], outputs_tensor])
            metrics["task1"]["train_true"] = torch.cat([metrics["task1"]["train_true"], prop_tensor])


        for step, batch in enumerate(test_task1_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device)

            task1_output, _, _, _ = model(input_ids, attention_mask)
            outputs = task1_output.float()
            
            outputs_cpu = outputs.cpu().detach()
            prop_cpu = prop.cpu().detach()
            
            outputs_np = scaler_task1.inverse_transform(outputs_cpu.reshape(-1, 1))
            prop_np = scaler_task1.inverse_transform(prop_cpu.reshape(-1, 1))
            
            outputs_tensor = torch.from_numpy(outputs_np.flatten()).float()
            prop_tensor = torch.from_numpy(prop_np.flatten()).float()

            loss = loss_fn(outputs_tensor, prop_tensor)
            metrics["task1"]["test_loss"] += loss.item() * len(prop_tensor)

            metrics["task1"]["test_pred"] = torch.cat([metrics["task1"]["test_pred"], outputs_tensor])
            metrics["task1"]["test_true"] = torch.cat([metrics["task1"]["test_true"], prop_tensor])
        
        metrics["task1"]["train_loss"] /= len(metrics["task1"]["train_pred"].flatten())
        metrics["task1"]["test_loss"] /= len(metrics["task1"]["test_pred"].flatten())

        metrics["task1"]["train_r2"] = R2Score()(
            metrics["task1"]["train_pred"].flatten(),
            metrics["task1"]["train_true"].flatten()
        ).item()

        metrics["task1"]["test_r2"] = R2Score()(
            metrics["task1"]["test_pred"].flatten(),
            metrics["task1"]["test_true"].flatten()
        ).item()

        metrics["task1"]["train_spearman"] = SpearmanCorrCoef(compute_on_step=False)(
            metrics["task1"]["train_pred"].flatten(),
            metrics["task1"]["train_true"].flatten()
        ).item()

        metrics["task1"]["test_spearman"] = SpearmanCorrCoef(compute_on_step=False)(
            metrics["task1"]["test_pred"].flatten(),
            metrics["task1"]["test_true"].flatten()
        ).item()

        metrics["task1"]["train_mae"] = MeanAbsoluteError()(
            metrics["task1"]["train_pred"].flatten(),
            metrics["task1"]["train_true"].flatten()
        ).item()

        metrics["task1"]["test_mae"] = MeanAbsoluteError()(
            metrics["task1"]["test_pred"].flatten(),
            metrics["task1"]["test_true"].flatten()
        ).item()

        print(f"Task 1 - Train MSE: {metrics['task1']['train_loss']:.4f}, Test MSE: {metrics['task1']['test_loss']:.4f}")
        print(f"Task 1 - Train R^2: {metrics['task1']['train_r2']:.4f}, Test R^2: {metrics['task1']['test_r2']:.4f}")
        print(f"Task 1 - Train Spearman: {metrics['task1']['train_spearman']:.4f}, Test Spearman: {metrics['task1']['test_spearman']:.4f}")
        print(f"Task 1 - Train MAE: {metrics['task1']['train_mae']:.4f}, Test MAE: {metrics['task1']['test_mae']:.4f}")

        # Evaluate Task 2 (similar to Task 1)
        for step, batch in enumerate(train_task2_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device)

            _, task2_output, _, _ = model(input_ids, attention_mask)
            outputs = task2_output.float()
            
            outputs_cpu = outputs.cpu().detach()
            prop_cpu = prop.cpu().detach()
            
            outputs_np = scaler_task2.inverse_transform(outputs_cpu.reshape(-1, 1))
            prop_np = scaler_task2.inverse_transform(prop_cpu.reshape(-1, 1))

            outputs_tensor = torch.from_numpy(outputs_np.flatten()).float()
            prop_tensor = torch.from_numpy(prop_np.flatten()).float()

            loss = loss_fn(outputs_tensor, prop_tensor)
            metrics["task2"]["train_loss"] += loss.item() * len(prop_tensor)

            metrics["task2"]["train_pred"] = torch.cat([metrics["task2"]["train_pred"], outputs_tensor])
            metrics["task2"]["train_true"] = torch.cat([metrics["task2"]["train_true"], prop_tensor])

        for step, batch in enumerate(test_task2_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device)

            _, task2_output, _, _ = model(input_ids, attention_mask)
            outputs = task2_output.float()
            
            outputs_cpu = outputs.cpu().detach()
            prop_cpu = prop.cpu().detach()

            outputs_np = scaler_task2.inverse_transform(outputs_cpu.reshape(-1, 1))
            prop_np = scaler_task2.inverse_transform(prop_cpu.reshape(-1, 1))

            outputs_tensor = torch.from_numpy(outputs_np.flatten()).float()
            prop_tensor = torch.from_numpy(prop_np.flatten()).float()

            loss = loss_fn(outputs_tensor, prop_tensor)
            metrics["task2"]["test_loss"] += loss.item() * len(prop_tensor)

            metrics["task2"]["test_pred"] = torch.cat([metrics["task2"]["test_pred"], outputs_tensor])
            metrics["task2"]["test_true"] = torch.cat([metrics["task2"]["test_true"], prop_tensor])


        # Finalize metrics for Task 2
        metrics["task2"]["train_loss"] /= len(metrics["task2"]["train_pred"].flatten())
        metrics["task2"]["test_loss"] /= len(metrics["task2"]["test_pred"].flatten())

        metrics["task2"]["train_r2"] = R2Score()(
            metrics["task2"]["train_pred"].flatten(),
            metrics["task2"]["train_true"].flatten()
        ).item()

        metrics["task2"]["test_r2"] = R2Score()(
            metrics["task2"]["test_pred"].flatten(),
            metrics["task2"]["test_true"].flatten()
        ).item()

        metrics["task2"]["train_spearman"] = SpearmanCorrCoef(compute_on_step=False)(
            metrics["task2"]["train_pred"].flatten(),
            metrics["task2"]["train_true"].flatten()
        ).item()

        metrics["task2"]["test_spearman"] = SpearmanCorrCoef(compute_on_step=False)(
            metrics["task2"]["test_pred"].flatten(),
            metrics["task2"]["test_true"].flatten()
        ).item()

        metrics["task2"]["train_mae"] = MeanAbsoluteError()(
            metrics["task2"]["train_pred"].flatten(),
            metrics["task2"]["train_true"].flatten()
        ).item()

        metrics["task2"]["test_mae"] = MeanAbsoluteError()(
            metrics["task2"]["test_pred"].flatten(),
            metrics["task2"]["test_true"].flatten()
        ).item()

        print(f"Task 2 - Train MSE: {metrics['task2']['train_loss']:.4f}, Test MSE: {metrics['task2']['test_loss']:.4f}")
        print(f"Task 2 - Train R^2: {metrics['task2']['train_r2']:.4f}, Test R^2: {metrics['task2']['test_r2']:.4f}")
        print(f"Task 2 - Train Spearman: {metrics['task2']['train_spearman']:.4f}, Test Spearman: {metrics['task2']['test_spearman']:.4f}")
        print(f"Task 2 - Train MAE: {metrics['task2']['train_mae']:.4f}, Test MAE: {metrics['task2']['test_mae']:.4f}")

    return metrics

def main(finetune_config):

    """Tokenizer"""
    if finetune_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(finetune_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    """Data"""
    print("Start Cross Validation")
        
    task1_data = pd.read_csv(finetune_config['task1_train_file'])
    task1_data = task1_data.loc[task1_data['Paper_ID']==100,:].reset_index(drop=True)  # data of Paper_ID 100 is used for task 1
    task2_data = pd.read_csv(finetune_config['task2_train_file'])
    task2_data = task2_data.loc[task2_data['Paper_ID']==finetune_config['Paper_ID'],:].reset_index(drop=True)
    columns = task1_data.columns.tolist()
    for column in columns[2:-1]:
        if task1_data[column].dtype == 'float64':
            task1_data[column] = task1_data[column].apply(lambda x: '{:g}'.format(x)) ### important
    for column in columns[2:-1]:
        if task2_data[column].dtype == 'float64':
            task2_data[column] = task2_data[column].apply(lambda x: '{:g}'.format(x)) ### important

    """K-fold """
    print("K-Fold Cross Validation")
    splits_task1 = KFold(n_splits=finetune_config['k'], shuffle=True, random_state=finetune_config["split_seed"])
    splits_task2 = KFold(n_splits=finetune_config['k'], shuffle=True, random_state=finetune_config["split_seed"])
    """K-fold"""

    # monitor the best metrics in each fold
    train_loss_avg, val_loss_avg, test_loss_avg, train_r2_avg, val_r2_avg, test_r2_avg, train_sprm_avg, val_sprm_avg, test_sprm_avg, train_mae_avg, val_mae_avg, test_mae_avg = [], [], [], [], [], [], [], [], [], [], [], []
    
    train_loss_task1_list, val_loss_task1_list, test_loss_task1_list, train_r2_task1_list, val_r2_task1_list, test_r2_task1_list, train_sprm_task1_list, val_sprm_task1_list, test_sprm_task1_list, train_mae_task1_list, val_mae_task1_list, test_mae_task1_list = [], [], [], [], [], [], [], [], [], [], [], []

    train_loss_task2_list, val_loss_task2_list, test_loss_task2_list, train_r2_task2_list, val_r2_task2_list, test_r2_task2_list, train_sprm_task2_list, val_sprm_task2_list, test_sprm_task2_list, train_mae_task2_list, val_mae_task2_list, test_mae_task2_list = [], [], [], [], [], [], [], [], [], [], [], []

    all_predictions_task1 = []
    all_predictions_task2 = []

    for fold, ((train_idx_task1, val_idx_task1), (train_idx_task2, val_idx_task2)) in enumerate(zip(splits_task1.split(np.arange(task1_data.shape[0])), splits_task2.split(np.arange(task2_data.shape[0])))):
        print('Fold {}'.format(fold + 1))

        fold_train_data_task1 = task1_data.loc[train_idx_task1, :].reset_index(drop=True)
        fold_test_data_task1 = task1_data.loc[val_idx_task1, :].reset_index(drop=True)

        fold_train_data_task2 = task2_data.loc[train_idx_task2, :].reset_index(drop=True)
        fold_test_data_task2 = task2_data.loc[val_idx_task2, :].reset_index(drop=True)

        # nested validation
        if finetune_config['use_val_set']:
            fold_train_data_task1, fold_val_data_task1 = train_test_split(
                fold_train_data_task1,
                test_size=0.125, # 0.125 x 0.8 = 0.1
                random_state=finetune_config["split_seed"],
                shuffle=True
            )
            fold_train_data_task2, fold_val_data_task2 = train_test_split(
                fold_train_data_task2,
                test_size=0.125, # 0.125 x 0.8 = 0.1
                random_state=finetune_config["split_seed"],
                shuffle=True
            )

            print(f"Fold {fold+1} - Task1: Validation samples: {len(fold_val_data_task1)}")
            print(f"Fold {fold+1} - Task2: Validation samples: {len(fold_val_data_task2)}")

        print(f"Fold {fold+1} - Task 1: Training samples: {len(fold_train_data_task1)}, Test samples: {len(fold_test_data_task1)}")
        print(f"Fold {fold+1} - Task 2: Training samples: {len(fold_train_data_task2)}, Test samples: {len(fold_test_data_task2)}")  

        train_data_task1 = fold_train_data_task1.iloc[:, 2:].reset_index(drop=True)  
        test_data_task1 = fold_test_data_task1.iloc[:, 2:].reset_index(drop=True) 
        if finetune_config['use_val_set']:
            val_data_task1 = fold_val_data_task1.iloc[:, 2:].reset_index(drop=True) 
        
        train_data_task2 = fold_train_data_task2.iloc[:, 2:].reset_index(drop=True)
        test_data_task2 = fold_test_data_task2.iloc[:, 2:].reset_index(drop=True)
        if finetune_config['use_val_set']:
            val_data_task2 = fold_val_data_task2.iloc[:, 2:].reset_index(drop=True)

        original_test_smiles_task1 = fold_test_data_task1.iloc[:, 2].tolist() 
        original_test_smiles_task2 = fold_test_data_task2.iloc[:, 2].tolist()

        DataAug_task1 = DataAugmentation(finetune_config['aug_indicator'])
        DataAug_task2 = DataAugmentation(finetune_config['aug_indicator'])
        if finetune_config['aug_flag']:
            print("SMILES Data Augmentation, fold =", finetune_config['aug_indicator'])

            train_data_task1 = DataAug_task1.smiles_augmentation_2(train_data_task1)
            train_data_task2 = DataAug_task2.smiles_augmentation_2(train_data_task2)

            val_data_task1 = DataAug_task1.smiles_augmentation_2(val_data_task1) if finetune_config['use_val_set'] else None
            val_data_task2 = DataAug_task2.smiles_augmentation_2(val_data_task2) if finetune_config['use_val_set'] else None
            
        train_data_task1 = DataAug_task1.combine_smiles(train_data_task1)
        test_data_task1 = DataAug_task1.combine_smiles(test_data_task1)
        val_data_task1 = DataAug_task1.combine_smiles(val_data_task1) if finetune_config['use_val_set'] else None

        train_data_task2 = DataAug_task2.combine_smiles(train_data_task2)
        test_data_task2 = DataAug_task2.combine_smiles(test_data_task2)
        val_data_task2 = DataAug_task2.combine_smiles(val_data_task2) if finetune_config['use_val_set'] else None

        if finetune_config['aug_special_flag']:
            print("Combine descriptors")

            train_data_task1 = DataAug_task1.combine_columns(train_data_task1)
            test_data_task1 = DataAug_task1.combine_columns(test_data_task1)
            val_data_task1 = DataAug_task1.combine_columns(val_data_task1) if finetune_config['use_val_set'] else None

            train_data_task2 = DataAug_task2.combine_columns(train_data_task2)
            test_data_task2 = DataAug_task2.combine_columns(test_data_task2)
            val_data_task2 = DataAug_task2.combine_columns(val_data_task2) if finetune_config['use_val_set'] else None

        print("Augmented Train Data Sample for task1:")
        print(train_data_task1.iloc[0,0], train_data_task1.iloc[0,1])
        print("Augmented Test Data Sample for task1:")
        print(test_data_task1.iloc[0,0], test_data_task1.iloc[0,1])
        if finetune_config['use_val_set']:
            print("Augmented Validation Data Sample for task1:")
            print(val_data_task1.iloc[0,0], val_data_task1.iloc[0,1])
        
        print("Augmented Train Data Sample for task2:")
        print(train_data_task2.iloc[0,0], train_data_task2.iloc[0,1])
        print("Augmented Test Data Sample for task2:")
        print(test_data_task2.iloc[0,0], test_data_task2.iloc[0,1])
        if finetune_config['use_val_set']:
            print("Augmented Validation Data Sample for task2:")
            print(val_data_task2.iloc[0,0], val_data_task2.iloc[0,1])

        scaler_task1 = StandardScaler()
        train_data_task1.iloc[:, 1] = scaler_task1.fit_transform(train_data_task1.iloc[:, 1].values.reshape(-1, 1))
        test_data_task1.iloc[:, 1] = scaler_task1.transform(test_data_task1.iloc[:, 1].values.reshape(-1, 1))
        if finetune_config['use_val_set']:
            val_data_task1.iloc[:, 1] = scaler_task1.transform(val_data_task1.iloc[:, 1].values.reshape(-1, 1))

        scaler_task2 = StandardScaler()
        train_data_task2.iloc[:, 1] = scaler_task2.fit_transform(train_data_task2.iloc[:, 1].values.reshape(-1, 1))
        test_data_task2.iloc[:, 1] = scaler_task2.transform(test_data_task2.iloc[:, 1].values.reshape(-1, 1))
        if finetune_config['use_val_set']:
            val_data_task2.iloc[:, 1] = scaler_task2.transform(val_data_task2.iloc[:, 1].values.reshape(-1, 1))

        scaler1_filename = f"_paper{finetune_config['Paper_ID']}_seed{finetune_config['split_seed']}_fold{fold+1}.pkl"
        scaler1_path = finetune_config['scaler_task1_path'].replace('.pkl', scaler1_filename)

        scaler2_filename = f"_paper{finetune_config['Paper_ID']}_seed{finetune_config['split_seed']}_fold{fold+1}.pkl"
        scaler2_path = finetune_config['scaler_task2_path'].replace('.pkl', scaler2_filename)

        joblib.dump(scaler_task1, scaler1_path)
        joblib.dump(scaler_task2, scaler2_path)

        task1_train_dataset = Downstream_Dataset(train_data_task1, tokenizer, finetune_config['blocksize'])
        task2_train_dataset = Downstream_Dataset(train_data_task2, tokenizer, finetune_config['blocksize'])
        task1_test_dataset = Downstream_Dataset(test_data_task1, tokenizer, finetune_config['blocksize'])
        task2_test_dataset = Downstream_Dataset(test_data_task2, tokenizer, finetune_config['blocksize'])

        task1_size = len(task1_train_dataset)
        task2_size = len(task2_train_dataset)
        print(f"Task 1 dataset size: {task1_size}, Task 2 dataset size: {task2_size}")
        
        # do_subsample = task1_size != task2_size
        # target_size = min(task1_size, task2_size)
        do_subsample = False
        
        task1_test_dataloader = DataLoader(task1_test_dataset, finetune_config['batch_size'], 
                                        shuffle=False, num_workers=finetune_config["num_workers"])
        task2_test_dataloader = DataLoader(task2_test_dataset, finetune_config['batch_size'], 
                                        shuffle=False, num_workers=finetune_config["num_workers"])

        if do_subsample:

            if task1_size > task2_size:
                print(f"Using periodic subsampling for Task 1 (from {task1_size} to {target_size})")
                task1_train_dataloader = PeriodicSubsamplingDataLoader(
                    task1_train_dataset,
                    batch_size=finetune_config['batch_size'],
                    subsample_size=target_size,
                    random_seed=finetune_config["split_seed"] + fold,
                    num_workers=finetune_config["num_workers"]
                )
  
                task2_train_dataloader = DataLoader(
                    task2_train_dataset,
                    batch_size=finetune_config['batch_size'],
                    shuffle=True,
                    num_workers=finetune_config["num_workers"]
                )
            else:

                task1_train_dataloader = DataLoader(
                    task1_train_dataset,
                    batch_size=finetune_config['batch_size'],
                    shuffle=True,
                    num_workers=finetune_config["num_workers"]
                )
                print(f"Using periodic subsampling for Task 2 (from {task2_size} to {target_size})")
                task2_train_dataloader = PeriodicSubsamplingDataLoader(
                    task2_train_dataset,
                    batch_size=finetune_config['batch_size'],
                    subsample_size=target_size,
                    random_seed=finetune_config["split_seed"] + fold,
                    num_workers=finetune_config["num_workers"]
                )
        else:

            task1_train_dataloader = DataLoader(
                task1_train_dataset,
                batch_size=finetune_config['batch_size'],
                shuffle=True,
                num_workers=finetune_config["num_workers"]
            )
            task2_train_dataloader = DataLoader(
                task2_train_dataset,
                batch_size=finetune_config['batch_size'],
                shuffle=True,
                num_workers=finetune_config["num_workers"]
            )

        if finetune_config['use_val_set']:
            task1_val_dataset = Downstream_Dataset(val_data_task1, tokenizer, finetune_config['blocksize'])
            task2_val_dataset = Downstream_Dataset(val_data_task2, tokenizer, finetune_config['blocksize'])
            task1_val_dataloader = DataLoader(task1_val_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])
            task2_val_dataloader = DataLoader(task2_val_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

        """Parameters for scheduler"""
        steps_per_epoch = max(len(task1_train_dataloader), len(task2_train_dataloader))
        training_steps = max(1, steps_per_epoch * finetune_config['num_epochs'])
        warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

        """Train the model"""
        model = MultiTaskModelWithUncertainty(drop_rate=finetune_config['drop_rate']).to(device)
        model = model.double()  # Convert model to double precision, aiming for better numerical stability
        loss_fn = nn.MSELoss()

        if finetune_config['LLRD_flag']:
            optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                        "weight_decay": 0.0},
                    {"params": model.Task1Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                        "weight_decay": finetune_config['weight_decay']},
                    {"params": model.Task2Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                        "weight_decay": finetune_config['weight_decay']},
                    {"params": [model.log_sigma1, model.log_sigma2], "lr": finetune_config['lr_rate_reg'],
                        "weight_decay": 0.0}
                ]
            )

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        torch.cuda.empty_cache()
        # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
        best_train_loss, best_train_task1_loss, best_train_task2_loss = float('inf'), float('inf'), float('inf')
        best_val_loss, best_val_task1_loss, best_val_task2_loss = float('inf'), float('inf'), float('inf')
        best_train_r2, best_train_task1_r2, best_train_task2_r2 = 0, 0, 0
        best_val_r2, best_val_task1_r2, best_val_task2_r2 = 0, 0, 0
        best_train_mae, best_train_task1_mae, best_train_task2_mae = float('inf'), float('inf'), float('inf')
        best_val_mae, best_val_task1_mae, best_val_task2_mae = float('inf'), float('inf'), float('inf')
        best_train_sprm, best_train_task1_sprm, best_train_task2_sprm = 0, 0, 0
        best_val_sprm, best_val_task1_sprm, best_val_task2_sprm = 0, 0, 0

        count = 0     # Keep track of how many successive non-improvement epochs

        fold_train_losses = []
        fold_val_losses = []
        fold_train_task1_losses = []
        fold_val_task1_losses = []
        fold_train_task2_losses = []
        fold_val_task2_losses = []

        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))

            if do_subsample:
                if task1_size > task2_size:
                    task1_train_dataloader.update_epoch(epoch)
                else:
                    task2_train_dataloader.update_epoch(epoch)

            train_multitask_with_separate_data(model, optimizer, scheduler, loss_fn, task1_train_dataloader, task2_train_dataloader, device)
      
            monitor_dataloader_task1 = task1_val_dataloader
            monitor_dataloader_task2 = task2_val_dataloader
            
            metrics = test_multitask_with_separate_data(model, loss_fn, task1_train_dataloader, task2_train_dataloader, monitor_dataloader_task1, monitor_dataloader_task2, device, scaler_task1, scaler_task2)
            
            fold_train_losses.append(metrics["task1"]["train_loss"] + metrics["task2"]["train_loss"])
            fold_val_losses.append(metrics["task1"]["test_loss"] + metrics["task2"]["test_loss"])
            fold_train_task1_losses.append(metrics["task1"]["train_loss"])
            fold_val_task1_losses.append(metrics["task1"]["test_loss"])
            fold_train_task2_losses.append(metrics["task2"]["train_loss"])
            fold_val_task2_losses.append(metrics["task2"]["test_loss"])

            monitor_name = "Validation"

            print(f"Epoch {epoch+1} - Task 1 Train Loss: {metrics['task1']['train_loss']:.4f}, Task 2 Train Loss: {metrics['task2']['train_loss']:.4f}")
            print(f"Epoch {epoch+1} - Task 1 {monitor_name} Loss: {metrics['task1']['test_loss']:.4f}, Task 2 {monitor_name} Loss: {metrics['task2']['test_loss']:.4f}")
        
            if metrics["task1"]["test_loss"] + metrics["task2"]["test_loss"] < best_val_loss * 2:

                best_train_loss = (metrics["task1"]["train_loss"] + metrics["task2"]["train_loss"]) /2
                best_val_loss = (metrics["task1"]["test_loss"] + metrics["task2"]["test_loss"]) /2
                best_train_r2 = (metrics["task1"]["train_r2"] + metrics["task2"]["train_r2"]) / 2
                best_val_r2 = (metrics["task1"]["test_r2"] + metrics["task2"]["test_r2"]) / 2
                best_train_mae = (metrics["task1"]["train_mae"] + metrics["task2"]["train_mae"]) / 2
                best_val_mae = (metrics["task1"]["test_mae"] + metrics["task2"]["test_mae"]) / 2
                best_train_sprm = (metrics["task1"]["train_spearman"] + metrics["task2"]["train_spearman"]) / 2
                best_val_sprm = (metrics["task1"]["test_spearman"] + metrics["task2"]["test_spearman"]) / 2

                best_train_task1_loss = metrics["task1"]["train_loss"]
                best_val_task1_loss = metrics["task1"]["test_loss"]
                best_train_task1_r2 = metrics["task1"]["train_r2"]
                best_val_task1_r2 = metrics["task1"]["test_r2"]
                best_train_task1_sprm = metrics["task1"]["train_spearman"]
                best_val_task1_sprm = metrics["task1"]["test_spearman"]
                best_train_task1_mae = metrics["task1"]["train_mae"]
                best_val_task1_mae = metrics["task1"]["test_mae"]

                best_train_task2_loss = metrics["task2"]["train_loss"]
                best_val_task2_loss = metrics["task2"]["test_loss"]
                best_train_task2_r2 = metrics["task2"]["train_r2"]
                best_val_task2_r2 = metrics["task2"]["test_r2"]
                best_train_task2_sprm = metrics["task2"]["train_spearman"]
                best_val_task2_sprm = metrics["task2"]["test_spearman"]
                best_train_task2_mae = metrics["task2"]["train_mae"]    
                best_val_task2_mae = metrics["task2"]["test_mae"]

                count = 0

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold': fold}
                torch.save(state, finetune_config['best_model_path'].replace('.pt', f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}_fold{fold+1}.pt'))
                print(f"Best model for fold {fold+1} saved at epoch {epoch+1}")

            else:
                count += 1

            if count >= finetune_config['tolerance']:
                print(f"Early stop at epoch {epoch+1} (no improvement for {count} epochs)")

                break
        
        current_fold_model_path = finetune_config['best_model_path'].replace('.pt', f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}_fold{fold+1}.pt')

        if os.path.exists(current_fold_model_path):
            model_load = MultiTaskModelWithUncertainty(drop_rate=finetune_config['drop_rate']).to(device).double()
            checkpoint = torch.load(current_fold_model_path, map_location=device)
            model_load.load_state_dict(checkpoint['model'])
            print(f"Loading best model for fold {fold+1} from {current_fold_model_path}")
        else:
            model_load = model
            print(f"Used model is not the best")

        model_load.eval()
        with torch.no_grad():

            test_predictions_task1 = []
            test_true_values_task1 = []
            test_predictions_task2 = []
            test_true_values_task2 = []
                            
            for batch in task1_test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['prop'].to(device)
                
                task1_output, _, _, _ = model_load(input_ids, attention_mask)
                predictions = task1_output.cpu().detach().numpy()  
                true_values = labels.cpu().detach().numpy()       
                
                predictions = scaler_task1.inverse_transform(predictions.reshape(-1, 1)).flatten()
                true_values = scaler_task1.inverse_transform(true_values.reshape(-1, 1)).flatten()
                
                test_predictions_task1.extend(predictions)
                test_true_values_task1.extend(true_values)
            
            for batch in task2_test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['prop'].to(device)
                
                _, task2_output, _, _ = model_load(input_ids, attention_mask)
                predictions = task2_output.cpu().detach().numpy()  
                true_values = labels.cpu().detach().numpy()       
                
                predictions = scaler_task2.inverse_transform(predictions.reshape(-1, 1)).flatten()
                true_values = scaler_task2.inverse_transform(true_values.reshape(-1, 1)).flatten()
                
                test_predictions_task2.extend(predictions)
                test_true_values_task2.extend(true_values)
    
        for i in range(len(fold_test_data_task1)):
            all_predictions_task1.append({
                'Fold': fold + 1,
                'Paper_ID': fold_test_data_task1.iloc[i, 0],  
                'ID': fold_test_data_task1.iloc[i, 1],        
                'SMILES': original_test_smiles_task1[i], 
                'True_Value': test_true_values_task1[i],
                'Predicted_Value': test_predictions_task1[i],
                'Dataset': 'Test'
            })
        for i in range(len(fold_test_data_task2)):
            all_predictions_task2.append({
                'Fold': fold + 1,
                'Paper_ID': fold_test_data_task2.iloc[i, 0],  
                'ID': fold_test_data_task2.iloc[i, 1],        
                'SMILES': original_test_smiles_task2[i],     
                'True_Value': test_true_values_task2[i],
                'Predicted_Value': test_predictions_task2[i],
                'Dataset': 'Test'
            })
        
        loss_curve_data = {
            'Epoch': list(range(1, len(fold_train_losses) + 1)),
            'Train_Loss': fold_train_losses,
            'Train_Task1_Loss': fold_train_task1_losses,
            'Train_Task2_Loss': fold_train_task2_losses,
            'Val_Loss': fold_val_losses,
            'Val_Task1_Loss': fold_val_task1_losses,
            'Val_Task2_Loss': fold_val_task2_losses
        }
        loss_curve_df = pd.DataFrame(loss_curve_data)
        loss_curve_file = finetune_config['result_file'].replace('.csv', f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}_fold{fold+1}_loss_curve.csv')
        loss_curve_df.to_csv(loss_curve_file, index=False)
        print(f"Loss curve for fold {fold+1} saved to {loss_curve_file}")

        ### record metrics
        test_predictions_array_task1 = np.array(test_predictions_task1)
        test_true_values_array_task1 = np.array(test_true_values_task1)

        test_predictions_array_task2 = np.array(test_predictions_task2)
        test_true_values_array_task2 = np.array(test_true_values_task2)

        test_mse_task1 = mean_squared_error(test_true_values_array_task1, test_predictions_array_task1)
        test_mse_task2 = mean_squared_error(test_true_values_array_task2, test_predictions_array_task2)
        test_mse = (test_mse_task1 + test_mse_task2) / 2

        try:
            test_r2_task1 = r2_score(test_true_values_array_task1, test_predictions_array_task1)
            test_r2_task2 = r2_score(test_true_values_array_task2, test_predictions_array_task2)
            test_r2 = (test_r2_task1 + test_r2_task2) / 2 
        except Exception as e:
            print(f"Error calculating R^2: {e}")
            test_r2_task1 = float('nan')
            test_r2_task2 = float('nan')
            test_r2 = float('nan')
        test_mae_task1 = mean_absolute_error(test_true_values_array_task1, test_predictions_array_task1)
        test_mae_task2 = mean_absolute_error(test_true_values_array_task2, test_predictions_array_task2)
        test_mae = (test_mae_task1 + test_mae_task2) / 2 
        test_sprm_task1, _ = spearmanr(test_true_values_array_task1, test_predictions_array_task1)
        test_sprm_task2, _ = spearmanr(test_true_values_array_task2, test_predictions_array_task2)
        test_sprm = (test_sprm_task1 + test_sprm_task2) / 2

        print(f"Fold {fold+1} Test MSE for Task 1: {test_mse_task1:.4f}, Task 2: {test_mse_task2:.4f}, Average: {test_mse:.4f}")
        print(f"Fold {fold+1} Test R^2 for Task 1: {test_r2_task1:.4f}, Task 2: {test_r2_task2:.4f}, Average: {test_r2:.4f}")
        print(f"Fold {fold+1} Test MAE for Task 1: {test_mae_task1:.4f}, Task 2: {test_mae_task2:.4f}, Average: {test_mae:.4f}")
        print(f"Fold {fold+1} Test Spearman for Task 1: {test_sprm_task1:.4f}, Task 2: {test_sprm_task2:.4f}, Average: {test_sprm:.4f}")

        train_loss_avg.append(best_train_loss) #MSE
        val_loss_avg.append(best_val_loss)
        test_loss_avg.append(test_mse)
        train_r2_avg.append(best_train_r2)
        val_r2_avg.append(best_val_r2)
        test_r2_avg.append(test_r2)
        train_sprm_avg.append(best_train_sprm)
        val_sprm_avg.append(best_val_sprm)
        test_sprm_avg.append(test_sprm)
        train_mae_avg.append(best_train_mae)
        val_mae_avg.append(best_val_mae)
        test_mae_avg.append(test_mae)

        # Task 1 metrics
        train_loss_task1_list.append(best_train_task1_loss)
        val_loss_task1_list.append(best_val_task1_loss)
        test_loss_task1_list.append(test_mse_task1)
        train_r2_task1_list.append(best_train_task1_r2)
        val_r2_task1_list.append(best_val_task1_r2)
        test_r2_task1_list.append(test_r2_task1)
        train_sprm_task1_list.append(best_train_task1_sprm)
        val_sprm_task1_list.append(best_val_task1_sprm)
        test_sprm_task1_list.append(test_sprm_task1)
        train_mae_task1_list.append(best_train_task1_mae)
        val_mae_task1_list.append(best_val_task1_mae)
        test_mae_task1_list.append(test_mae_task1)
        # Task 2 metrics
        train_loss_task2_list.append(best_train_task2_loss)
        val_loss_task2_list.append(best_val_task2_loss)
        test_loss_task2_list.append(test_mse_task2)
        train_r2_task2_list.append(best_train_task2_r2)
        val_r2_task2_list.append(best_val_task2_r2)
        test_r2_task2_list.append(test_r2_task2)
        train_sprm_task2_list.append(best_train_task2_sprm)
        val_sprm_task2_list.append(best_val_task2_sprm)
        test_sprm_task2_list.append(test_sprm_task2)
        train_mae_task2_list.append(best_train_task2_mae)
        val_mae_task2_list.append(best_val_task2_mae)
        test_mae_task2_list.append(test_mae_task2)

        def cleanup_gpu_memory():
            
            torch.cuda.empty_cache()

            import gc
            gc.collect()
        try:

            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            if 'scheduler' in locals():
                del scheduler
            if 'task1_train_dataloader' in locals():
                del task1_train_dataloader
            if 'task1_test_dataloader' in locals():
                del task1_test_dataloader
            if 'task2_train_dataloader' in locals():
                del task2_train_dataloader
            if 'task2_test_dataloader' in locals():
                del task2_test_dataloader
            if 'task1_val_dataloader' in locals():
                del task1_val_dataloader
            if 'task2_val_dataloader' in locals():
                del task2_val_dataloader
            if 'model_load' in locals() and ('model' not in locals() or model_load is not model):
                del model_load
            if 'checkpoint' in locals():
                del checkpoint
                
            cleanup_gpu_memory()
            print(f"Fold {fold+1} completed, memory thoroughly cleared")
        except Exception as e:
            print(f"Warning during memory cleanup: {e}")

            torch.cuda.empty_cache()
    

    """Average of metrics over all folds"""
    train_mse_task1 = np.mean(np.array(train_loss_task1_list))
    val_mse_task1 = np.mean(np.array(val_loss_task1_list))
    test_mse_task1 = np.mean(np.array(test_loss_task1_list))
    train_r2_task1 = np.mean(np.array(train_r2_task1_list))
    val_r2_task1 = np.mean(np.array(val_r2_task1_list))
    test_r2_task1 = np.mean(np.array(test_r2_task1_list))
    train_sprm_task1 = np.mean(np.array(train_sprm_task1_list))
    val_sprm_task1 = np.mean(np.array(val_sprm_task1_list))
    test_sprm_task1 = np.mean(np.array(test_sprm_task1_list))
    train_mae_task1 = np.mean(np.array(train_mae_task1_list))
    val_mae_task1 = np.mean(np.array(val_mae_task1_list))
    test_mae_task1 = np.mean(np.array(test_mae_task1_list))
    std_train_mse_task1 = np.std(np.array(train_loss_task1_list))
    std_train_r2_task1 = np.std(np.array(train_r2_task1_list))
    std_train_sprm_task1 = np.std(np.array(train_sprm_task1_list))
    std_train_mae_task1 = np.std(np.array(train_mae_task1_list))
    std_val_mse_task1 = np.std(np.array(val_loss_task1_list))
    std_val_r2_task1 = np.std(np.array(val_r2_task1_list))
    std_val_sprm_task1 = np.std(np.array(val_sprm_task1_list))
    std_val_mae_task1 = np.std(np.array(val_mae_task1_list))
    std_test_mse_task1 = np.std(np.array(test_loss_task1_list))
    std_test_r2_task1 = np.std(np.array(test_r2_task1_list))
    std_test_sprm_task1 = np.std(np.array(test_sprm_task1_list))
    std_test_mae_task1 = np.std(np.array(test_mae_task1_list))
    
    train_mse_task2 = np.mean(np.array(train_loss_task2_list))
    val_mse_task2 = np.mean(np.array(val_loss_task2_list))
    test_mse_task2 = np.mean(np.array(test_loss_task2_list))
    train_r2_task2 = np.mean(np.array(train_r2_task2_list))
    val_r2_task2 = np.mean(np.array(val_r2_task2_list))
    test_r2_task2 = np.mean(np.array(test_r2_task2_list))    
    train_sprm_task2 = np.mean(np.array(train_sprm_task2_list))
    val_sprm_task2 = np.mean(np.array(val_sprm_task2_list))
    test_sprm_task2 = np.mean(np.array(test_sprm_task2_list))
    train_mae_task2 = np.mean(np.array(train_mae_task2_list))
    val_mae_task2 = np.mean(np.array(val_mae_task2_list))
    test_mae_task2 = np.mean(np.array(test_mae_task2_list))
    std_train_mse_task2 = np.std(np.array(train_loss_task2_list))
    std_train_r2_task2 = np.std(np.array(train_r2_task2_list))
    std_train_sprm_task2 = np.std(np.array(train_sprm_task2_list))
    std_train_mae_task2 = np.std(np.array(train_mae_task2_list))
    std_val_mse_task2 = np.std(np.array(val_loss_task2_list))
    std_val_r2_task2 = np.std(np.array(val_r2_task2_list))
    std_val_sprm_task2 = np.std(np.array(val_sprm_task2_list))
    std_val_mae_task2 = np.std(np.array(val_mae_task2_list))
    std_test_mse_task2 = np.std(np.array(test_loss_task2_list))
    std_test_r2_task2 = np.std(np.array(test_r2_task2_list))
    std_test_sprm_task2 = np.std(np.array(test_sprm_task2_list))
    std_test_mae_task2 = np.std(np.array(test_mae_task2_list))
    
    results_data = {

        'Train_MSE_Task1_Mean': [train_mse_task1],
        'Train_R2_Task1_Mean': [train_r2_task1],
        'Train_MAE_Task1_Mean': [train_mae_task1],
        'Train_Spearman_Task1_Mean': [train_sprm_task1],
        'Train_MSE_Task1_Std': [std_train_mse_task1],
        'Train_R2_Task1_Std': [std_train_r2_task1],
        'Train_MAE_Task1_Std': [std_train_mae_task1],
        'Train_Spearman_Task1_Std': [std_train_sprm_task1],
        'Val_MSE_Task1_Mean': [val_mse_task1],
        'Val_R2_Task1_Mean': [val_r2_task1],
        'Val_MAE_Task1_Mean': [val_mae_task1],
        'Val_Spearman_Task1_Mean': [val_sprm_task1],
        'Val_MSE_Task1_Std': [std_val_mse_task1],
        'Val_R2_Task1_Std': [std_val_r2_task1],
        'Val_MAE_Task1_Std': [std_val_mae_task1],
        'Val_Spearman_Task1_Std': [std_val_sprm_task1],
        'Test_MSE_Task1_Mean': [test_mse_task1],
        'Test_R2_Task1_Mean': [test_r2_task1],
        'Test_MAE_Task1_Mean': [test_mae_task1],
        'Test_Spearman_Task1_Mean': [test_sprm_task1],
        'Test_MSE_Task1_Std': [std_test_mse_task1],
        'Test_R2_Task1_Std': [std_test_r2_task1],
        'Test_MAE_Task1_Std': [std_test_mae_task1],
        'Test_Spearman_Task1_Std': [std_test_sprm_task1],

        'Train_MSE_Task2_Mean': [train_mse_task2],
        'Train_R2_Task2_Mean': [train_r2_task2],
        'Train_MAE_Task2_Mean': [train_mae_task2],
        'Train_Spearman_Task2_Mean': [train_sprm_task2],
        'Train_MSE_Task2_Std': [std_train_mse_task2],
        'Train_R2_Task2_Std': [std_train_r2_task2],
        'Train_MAE_Task2_Std': [std_train_mae_task2],
        'Train_Spearman_Task2_Std': [std_train_sprm_task2],
        'Val_MSE_Task2_Mean': [val_mse_task2],
        'Val_R2_Task2_Mean': [val_r2_task2],
        'Val_MAE_Task2_Mean': [val_mae_task2],
        'Val_Spearman_Task2_Mean': [val_sprm_task2],
        'Val_MSE_Task2_Std': [std_val_mse_task2],
        'Val_R2_Task2_Std': [std_val_r2_task2],
        'Val_MAE_Task2_Std': [std_val_mae_task2],
        'Val_Spearman_Task2_Std': [std_val_sprm_task2],
        'Test_MSE_Task2_Mean': [test_mse_task2],
        'Test_R2_Task2_Mean': [test_r2_task2],
        'Test_MAE_Task2_Mean': [test_mae_task2],
        'Test_Spearman_Task2_Mean': [test_sprm_task2],
        'Test_MSE_Task2_Std': [std_test_mse_task2],
        'Test_R2_Task2_Std': [std_test_r2_task2],
        'Test_MAE_Task2_Std': [std_test_mae_task2],
        'Test_Spearman_Task2_Std': [std_test_sprm_task2],
        
        'Train_MSE_Task1_All_Folds': [str(train_loss_task1_list)],
        'Val_MSE_Task1_All_Folds': [str(val_loss_task1_list)],
        'Test_MSE_Task1_All_Folds': [str(test_loss_task1_list)],
        'Train_R2_Task1_All_Folds': [str(train_r2_task1_list)],
        'Val_R2_Task1_All_Folds': [str(val_r2_task1_list)],
        'Test_R2_Task1_All_Folds': [str(test_r2_task1_list)],
        'Train_MAE_Task1_All_Folds': [str(train_mae_task1_list)],
        'Val_MAE_Task1_All_Folds': [str(val_mae_task1_list)],
        'Test_MAE_Task1_All_Folds': [str(test_mae_task1_list)],
        'Train_Spearman_Task1_All_Folds': [str(train_sprm_task1_list)],
        'Val_Spearman_Task1_All_Folds': [str(val_sprm_task1_list)],
        'Test_Spearman_Task1_All_Folds': [str(test_sprm_task1_list)],

        'Train_MSE_Task2_All_Folds': [str(train_loss_task2_list)],
        'Val_MSE_Task2_All_Folds': [str(val_loss_task2_list)],
        'Test_MSE_Task2_All_Folds': [str(test_loss_task2_list)],
        'Train_R2_Task2_All_Folds': [str(train_r2_task2_list)],
        'Val_R2_Task2_All_Folds': [str(val_r2_task2_list)],
        'Test_R2_Task2_All_Folds': [str(test_r2_task2_list)],
        'Train_MAE_Task2_All_Folds': [str(train_mae_task2_list)],
        'Val_MAE_Task2_All_Folds': [str(val_mae_task2_list)],
        'Test_MAE_Task2_All_Folds': [str(test_mae_task2_list)],
        'Train_Spearman_Task2_All_Folds': [str(train_sprm_task2_list)],
        'Val_Spearman_Task2_All_Folds': [str(val_sprm_task2_list)],
        'Test_Spearman_Task2_All_Folds': [str(test_sprm_task2_list)],
    }

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(finetune_config['result_file'].replace('.csv',f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}.csv'), index=False)

    predictions_df_task1 = pd.DataFrame(all_predictions_task1)
    predictions_file_task1 = finetune_config['result_file'].replace('.csv', f'_predictions_task1_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}.csv')
    predictions_df_task1.to_csv(predictions_file_task1, index=False)
    print(f"All predictions for task1 saved to {predictions_file_task1}")

    predictions_df_task2 = pd.DataFrame(all_predictions_task2)
    predictions_file_task2 = finetune_config['result_file'].replace('.csv', f'_predictions_task2_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}.csv')
    predictions_df_task2.to_csv(predictions_file_task2, index=False)
    print(f"All predictions for task2 saved to {predictions_file_task2}")


if __name__ == "__main__":

    for paper_id in [101]: 

        print(f"********** Run with Paper {paper_id} **********")

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        finetune_config = yaml.load(open("config_finetune_MTL.yaml", "r"), Loader=yaml.FullLoader)
        finetune_config["split_seed"] = seed # fix the random seed for each paper
        finetune_config["Paper_ID"] = paper_id
        print(finetune_config)

        """Device"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if finetune_config['model_indicator'] == 'pretrain':
            print("Use the pretrained model")
            PretrainedModel = RobertaModel.from_pretrained(finetune_config['model_path'])
            tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
            PretrainedModel.config.hidden_dropout_prob = finetune_config['hidden_dropout_prob']
            PretrainedModel.config.attention_probs_dropout_prob = finetune_config['attention_probs_dropout_prob']
        else:
            print("No Pretrain")
            config = RobertaConfig(
                vocab_size=50265,
                max_position_embeddings=514,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            PretrainedModel = RobertaModel(config=config)
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
        max_token_len = finetune_config['blocksize']

        """Run the main function"""
        main(finetune_config)