import os
import yaml
from copy import deepcopy

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

class DownstreamRegression(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output

"""Train"""

def train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, device):
    model.train()
    train_step_losses = []
    val_step_losses = []
    
    for step, batch in enumerate(train_dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prop = batch["prop"].to(device).float()
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).float()
        
        loss = loss_fn(outputs.squeeze(), prop.squeeze())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_step_losses.append({
            'Step': step + 1,
            'Loss': loss.item(),
            'Batch_Size': len(prop),
            'Dataset': 'Train',
        })
        
        model.eval()
        val_total_loss = 0.0
        val_total_samples = 0
        
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_prop = val_batch["prop"].to(device).float()
                
                val_outputs = model(val_input_ids, val_attention_mask).float()
                
                val_batch_loss = loss_fn(val_outputs.squeeze(), val_prop.squeeze())
                
                val_total_loss += val_batch_loss.item() * len(val_prop)
                val_total_samples += len(val_prop)
            
            avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else 0
            
            val_step_losses.append({
                'Step': step + 1,  
                'Loss': avg_val_loss,
                'Batch_Size': val_total_samples,
                'Dataset': 'Val',
            })
        
        model.train()
    
    return train_step_losses, val_step_losses

def test(model, loss_fn, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch):

    train_loss = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():
        train_pred, train_true, test_pred, test_true = torch.tensor([]), torch.tensor([]), torch.tensor(
            []), torch.tensor([])

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            train_loss += loss.item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), outputs.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = R2Score()(
            train_pred.flatten().to("cpu"),
            train_true.flatten().to("cpu")
        ).item()

        sprm_train = SpearmanCorrCoef()(
            train_pred.flatten().to("cpu"),
            train_true.flatten().to("cpu")
        ).item()

        mae_train = MeanAbsoluteError()(
            train_pred.flatten().to("cpu"),
            train_true.flatten().to("cpu")
        ).item()

        print("train MSE = ", train_loss)
        print("train MAE = ", mae_train)
        print("train r^2 = ", r2_train)
        print("train spearman = ", sprm_train)

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            test_loss += loss.item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), outputs.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])


        test_loss = test_loss / len(test_pred.flatten())
        r2_test = R2Score()(
            test_pred.flatten().to("cpu"),
            test_true.flatten().to("cpu")
        ).item()

        sprm_test = SpearmanCorrCoef()(
            test_pred.flatten().to("cpu"),
            test_true.flatten().to("cpu")
        ).item()

        mae_test = MeanAbsoluteError()(
            test_pred.flatten().to("cpu"),
            test_true.flatten().to("cpu")
        ).item()


        print("test MSE = ", test_loss)
        print("test MAE = ", mae_test)
        print("test r^2 = ", r2_test)
        print("test spearman = ", sprm_test)

    return train_loss, test_loss, r2_train, r2_test, sprm_train, sprm_test, mae_train, mae_test

def main(finetune_config):

    """Tokenizer"""
    if finetune_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(finetune_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    """Data"""
    print("Start Cross Validation")
    data = pd.read_csv(finetune_config['train_file'])
    data = data.loc[data['Paper_ID'] == finetune_config['Paper_ID'], :].reset_index(drop=True)
    columns = data.columns.tolist()
    ### important
    for column in columns[2:-1]:
        if data[column].dtype == 'float64':
            data[column] = data[column].apply(lambda x: '{:g}'.format(x)) 

    """K-fold """
    print("K-Fold Cross Validation")
    splits = KFold(n_splits=finetune_config['k'], shuffle=True, random_state=finetune_config['split_seed'])
    """K-fold"""

    train_loss_avg, val_loss_avg, test_loss_avg, train_r2_avg, val_r2_avg, test_r2_avg, train_sprm_avg, val_sprm_avg, test_sprm_avg, train_mae_avg, val_mae_avg, test_mae_avg = [], [], [], [], [], [], [], [], [], [], [], []    

    all_predictions = []

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data.shape[0]))):
        print('Fold {}'.format(fold + 1))

        fold_train_data = data.loc[train_idx, :].reset_index(drop=True)
        fold_test_data = data.loc[val_idx, :].reset_index(drop=True)

        if finetune_config['use_val_set']:
            fold_train_data, fold_val_data = train_test_split(
                fold_train_data,
                test_size=0.125, # 0.125 x 0.8 = 0.1 of the total data
                random_state=finetune_config['split_seed'],
                shuffle=True

            )
            fold_train_data = fold_train_data.reset_index(drop=True)
            fold_val_data = fold_val_data.reset_index(drop=True)
            print(f"Fold {fold+1} - Training samples: {len(fold_train_data)}, Validation samples: {len(fold_val_data)}, Test samples: {len(fold_test_data)}")   

        train_data = fold_train_data.iloc[:, 2:].reset_index(drop=True) 
        test_data = fold_test_data.iloc[:, 2:].reset_index(drop=True)   
        if finetune_config['use_val_set']:
            val_data = fold_val_data.iloc[:, 2:].reset_index(drop=True) 

        original_test_smiles = fold_test_data.iloc[:, 2].tolist()

        DataAug = DataAugmentation(finetune_config['aug_indicator'])
        if finetune_config['aug_flag']:
            print("SMILES Data Augmentation, fold =", finetune_config['aug_indicator'])

            train_data = DataAug.smiles_augmentation_2(train_data)
            val_data = DataAug.smiles_augmentation_2(val_data) if finetune_config['use_val_set'] else None

        train_data = DataAug.combine_smiles(train_data)
        val_data = DataAug.combine_smiles(val_data) if finetune_config['use_val_set'] else None
        test_data = DataAug.combine_smiles(test_data)

        if finetune_config['aug_special_flag']:
            print("Combine descriptors")
            train_data = DataAug.combine_columns(train_data)
            test_data = DataAug.combine_columns(test_data)
            val_data = DataAug.combine_columns(val_data) if finetune_config['use_val_set'] else None
            
        print(f"Data Augmentation Completed, {finetune_config['aug_indicator']} fold applied.")
        print("Augmented Train Data Sample:", train_data.shape)
        print(train_data.iloc[0,0], train_data.iloc[0,1])
        if finetune_config['use_val_set']:
            print("Augmented Validation Data Sample:", val_data.shape)
            print(val_data.iloc[0,0], val_data.iloc[0,1])
        print("Augmented Test Data Sample:", test_data.shape)
        print(test_data.iloc[0,0], test_data.iloc[0,1])

        scaler = StandardScaler()
        train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
        test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))
        if finetune_config['use_val_set']:
            val_data.iloc[:, 1] = scaler.transform(val_data.iloc[:, 1].values.reshape(-1, 1))


        train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
        test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'])
        train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
        test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])
        if finetune_config['use_val_set']:
            val_dataset = Downstream_Dataset(val_data, tokenizer, finetune_config['blocksize'])
            val_dataloader = DataLoader(val_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

        """Parameters for scheduler"""
        steps_per_epoch = len(train_dataloader)
        training_steps = max(1, steps_per_epoch * finetune_config['num_epochs'])
        warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

        """Train the model"""
        model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
        model = model.double()
        loss_fn = nn.MSELoss()

        if finetune_config['LLRD_flag']:
            optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                        "weight_decay": 0.0},
                    {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                        "weight_decay": finetune_config['weight_decay']},
                ]
            )

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        torch.cuda.empty_cache()
        best_train_sprm, best_val_sprm, best_train_r2, best_val_r2 = 0.0, 0.0, 0.0, 0.0
        train_loss_best, val_loss_best, best_train_mae, best_val_mae = float('inf'), float('inf'), float('inf'), float('inf')
        count = 0 

        fold_train_losses = []
        fold_val_losses = []

        fold_train_step_losses = []
        fold_val_step_losses = []

        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))
            train_step_losses, val_step_losses = train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, device)

            monitor_dataloader = val_dataloader
            
            train_loss, val_loss, r2_train, r2_val, sprm_train, sprm_val, mae_train, mae_val = test(model, loss_fn, train_dataloader,monitor_dataloader, device, scaler, optimizer, scheduler, epoch)

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)

            for step_data in train_step_losses:
                step_data['Epoch'] = epoch + 1
                step_data['Fold'] = fold + 1
                fold_train_step_losses.append(step_data)
                
            for step_data in val_step_losses:
                step_data['Epoch'] = epoch + 1
                step_data['Fold'] = fold + 1
                fold_val_step_losses.append(step_data)

            if val_loss < val_loss_best:
                best_train_r2 = r2_train
                best_val_r2 = r2_val
                train_loss_best = train_loss
                val_loss_best = val_loss
                best_train_mae = mae_train
                best_val_mae = mae_val
                best_train_sprm = sprm_train
                best_val_sprm = sprm_val
                count = 0

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold': fold}
                torch.save(state, finetune_config['best_model_path'].replace('.pt', f'_paper{finetune_config["Paper_ID"]}_fold{fold+1}_seed{finetune_config["split_seed"]}.pt'))
                print(f"Best model for fold {fold+1} saved at epoch {epoch+1}")

            else:
                count += 1

            if count >= finetune_config['tolerance']:
                print(f"Early stop at epoch {epoch+1} (no improvement for {count} epochs)")

                print(f"Fold {fold+1} Best Train MSE: {train_loss_best}")
                print(f"Fold {fold+1} Best Val MSE: {val_loss_best}")
                print(f"Fold {fold+1} Best Train MAE: {best_train_mae}")
                print(f"Fold {fold+1} Best Val MAE: {best_val_mae}")
                print(f"Fold {fold+1} Best Train R^2: {best_train_r2}")
                print(f"Fold {fold+1} Best Val R^2: {best_val_r2}")
                print(f"Fold {fold+1} Best Train Spearman: {best_train_sprm}")
                print(f"Fold {fold+1} Best Val Spearman: {best_val_sprm}")

                break
        
        current_fold_model_path = finetune_config['best_model_path'].replace('.pt', f'_paper{finetune_config["Paper_ID"]}_fold{fold+1}_seed{finetune_config["split_seed"]}.pt')

        if os.path.exists(current_fold_model_path):
            model_load = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device).double()
            checkpoint = torch.load(current_fold_model_path, map_location=device)
            model_load.load_state_dict(checkpoint['model'])
            print(f"Loading best model for fold {fold+1} from {current_fold_model_path}")
        else:
            model_load = model
            print(f"Used model is not the best")

        model_load.eval()
        with torch.no_grad():

            test_predictions = []
            test_true_values = []
            
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['prop'].to(device)
                
                outputs = model_load(input_ids, attention_mask)
                predictions = outputs.cpu().numpy()
                true_values = labels.cpu().numpy()
                
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                true_values = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
                
                test_predictions.extend(predictions)
                test_true_values.extend(true_values)

        for i in range(len(fold_test_data)):
            all_predictions.append({
                'Fold': fold + 1,
                'Paper_ID': fold_test_data.iloc[i, 0],  
                'ID': fold_test_data.iloc[i, 1],        
                'SMILES': original_test_smiles[i], 
                'True_Value': test_true_values[i],
                'Predicted_Value': test_predictions[i],
                'Dataset': 'Test'
            })
        
        loss_curve_data = {
            'Epoch': list(range(1, len(fold_train_losses) + 1)),
            'Train_Loss': fold_train_losses,
            'Val_Loss': fold_val_losses
        }
        loss_curve_df = pd.DataFrame(loss_curve_data)
        loss_curve_file = finetune_config['result_file'].replace('.csv', f'_paper{finetune_config["Paper_ID"]}_fold{fold+1}_seed{finetune_config["split_seed"]}_loss_curve.csv')
        loss_curve_df.to_csv(loss_curve_file, index=False)
        print(f"Loss curve for fold {fold+1} saved to {loss_curve_file}")

        step_loss_df = pd.DataFrame(fold_train_step_losses + fold_val_step_losses)
        step_loss_file = finetune_config['result_file'].replace('.csv', f'_paper{finetune_config["Paper_ID"]}_fold{fold+1}_seed{finetune_config["split_seed"]}_step_losses.csv')
        step_loss_df.to_csv(step_loss_file, index=False)
        print(f"Step losses for fold {fold+1} saved to {step_loss_file}")

        test_predictions_array = np.array(test_predictions)
        test_true_values_array = np.array(test_true_values)

        test_mse = mean_squared_error(test_true_values_array, test_predictions_array)

        try:
            test_r2 = r2_score(test_true_values_array, test_predictions_array)
        except Exception as e:
            print(f"Error calculating R^2: {e}")
            test_r2 = float('nan') 
        test_mae = mean_absolute_error(test_true_values_array, test_predictions_array)
        test_sprm, _ = spearmanr(test_true_values_array, test_predictions_array)
        print(f"Fold {fold+1} Test MSE: {test_mse}")
        print(f"Fold {fold+1} Test R^2: {test_r2}")
        print(f"Fold {fold+1} Test MAE: {test_mae}")
        print(f"Fold {fold+1} Test Spearman: {test_sprm}")

        train_loss_avg.append(train_loss_best) #MSE
        val_loss_avg.append(val_loss_best)
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
            if 'train_dataloader' in locals():
                del train_dataloader
            if 'test_dataloader' in locals():
                del test_dataloader
            if 'train_dataset' in locals():
                del train_dataset
            if 'test_dataset' in locals():
                del test_dataset
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
    train_mse = np.mean(np.array(train_loss_avg))
    val_mse = np.mean(np.array(val_loss_avg))
    test_mse = np.mean(np.array(test_loss_avg))
    train_r2 = np.mean(np.array(train_r2_avg))
    val_r2 = np.mean(np.array(val_r2_avg))
    test_r2 = np.mean(np.array(test_r2_avg))
    train_sprm = np.mean(np.array(train_sprm_avg))
    val_sprm = np.mean(np.array(val_sprm_avg))
    test_sprm = np.mean(np.array(test_sprm_avg))
    train_mae = np.mean(np.array(train_mae_avg))
    val_mae = np.mean(np.array(val_mae_avg))
    test_mae = np.mean(np.array(test_mae_avg))
    std_test_mse = np.std(np.array(test_loss_avg))
    std_test_r2 = np.std(np.array(test_r2_avg))
    std_test_sprm = np.std(np.array(test_sprm_avg))
    std_test_mae = np.std(np.array(test_mae_avg))

    print("Train MSE =", train_mse)
    print("Test MSE =", test_mse)
    print("Train MAE =", train_mae)
    print("Test MAE =", test_mae)
    print("Train R^2 =", train_r2)
    print("Test R^2 =", test_r2)
    print("Train Spearman =", train_sprm)
    print("Test Spearman =", test_sprm)

    print("Standard Deviation of Test MSE =", std_test_mse)
    print("Standard Deviation of Test R^2 =", std_test_r2)
    print("Standard Deviation of Test MAE =", std_test_mae)
    print("Standard Deviation of Test Spearman =", std_test_sprm)


    results_data = {

        'Train_MSE_Mean': [train_mse],
        'Train_R2_Mean': [train_r2],
        'Train_MAE_Mean': [train_mae],
        'Train_Spearman_Mean': [train_sprm],

        'Val_MSE_Mean': [val_mse],
        'Val_R2_Mean': [val_r2],
        'Val_MAE_Mean': [val_mae],
        'Val_Spearman_Mean': [val_sprm],

        'Test_MSE_Mean': [test_mse],
        'Test_R2_Mean': [test_r2],
        'Test_MAE_Mean': [test_mae],
        'Test_Spearman_Mean': [test_sprm],
        'Test_MSE_Std': [std_test_mse],
        'Test_R2_Std': [std_test_r2],
        'Test_MAE_Std': [std_test_mae],
        'Test_Spearman_Std': [std_test_sprm],
        

        'Train_MSE_All_Folds': [str(train_loss_avg)],
        'Val_MSE_All_Folds': [str(val_loss_avg)],
        'Test_MSE_All_Folds': [str(test_loss_avg)],
        'Train_R2_All_Folds': [str(train_r2_avg)],
        'Val_R2_All_Folds': [str(val_r2_avg)],
        'Test_R2_All_Folds': [str(test_r2_avg)],
        'Train_MAE_All_Folds': [str(train_mae_avg)],
        'Val_MAE_All_Folds': [str(val_mae_avg)],
        'Test_MAE_All_Folds': [str(test_mae_avg)],
        'Train_Spearman_All_Folds': [str(train_sprm_avg)],
        'Val_Spearman_All_Folds': [str(val_sprm_avg)],
        'Test_Spearman_All_Folds': [str(test_sprm_avg)]
    }

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(finetune_config['result_file'].replace('.csv', f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}.csv'), index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_file = finetune_config['result_file'].replace('.csv', f'_paper{finetune_config["Paper_ID"]}_seed{finetune_config["split_seed"]}_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"All predictions saved to {predictions_file}")


if __name__ == "__main__":

    for paper_id in [100]:
        
        print(f"********** Run with Paper {paper_id} **********")

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        """Load config"""

        finetune_config = yaml.load(open("config_finetune_STL.yaml", "r"), Loader=yaml.FullLoader)
        finetune_config['Paper_ID'] = paper_id
        finetune_config['split_seed'] = seed
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