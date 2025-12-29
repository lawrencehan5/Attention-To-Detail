import os
import optuna
import gc
import time
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

from config import MAIN_SEED, NUM_TRIAL_STANDARD, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker
from data.datasets import load_datasets
from training.standard_training import train_standard

torch.set_float32_matmul_precision('high')


def main():
    # set up datasets
    _, (train_dataset_224, valid_dataset_224, test_dataset_224) = load_datasets()

    # load pre-trained model
    _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # standard resnet hyperparameter optimization 
    def objective(trial):
        # set seed
        set_seed(MAIN_SEED + trial.number)

        # define hyperparameter search space
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3])

        # create folder to save results
        os.makedirs("transfer_resnet_results", exist_ok=True)

        # for reproducibility
        g = torch.Generator()
        g.manual_seed(MAIN_SEED + trial.number)

        # fixed generator for validation
        g_fixed = torch.Generator()
        g_fixed.manual_seed(MAIN_SEED)

        # new dataloaders of current batch size
        train_loader_224 = DataLoader(
            dataset=train_dataset_224,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        valid_loader_224 = DataLoader(
            dataset=valid_dataset_224,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=g_fixed
        )

        # new model instance
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, 2)
        )

        # train and obtain final validation accuracy
        result = train_standard(
            model=model,
            train_loader=train_loader_224,
            valid_loader=valid_loader_224,
            num_epochs=25,
            lr=lr,
            weight_decay=weight_decay,
            eta_min=ETA_MIN,
            label_smoothing=LABEL_SMOOTHING,
            trial=trial,
            plot=False
        )

        torch.save(result, f"transfer_resnet_results/transfer_resnet_trial_{trial.number}.pt")
        
        if result['pruned']:
            raise optuna.TrialPruned()

        del model
        del train_loader_224
        del valid_loader_224
        gc.collect()
        torch.cuda.empty_cache()

        return result['max_valid_acc']

    # search for hyperparamters
    storage = "sqlite:///./transfer_resnet.db"
    study_name = "transfer_resnet_hyperparameters_optimization"

    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=25, reduction_factor=3)

    sampler = optuna.samplers.TPESampler(seed=MAIN_SEED)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
        sampler=sampler
    )

    # load progress, if any
    completed_trials = [t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]]
    print(f"Study started with {len(completed_trials)} completed trials.")
    start_time = time.time()

    # start optimization
    study.optimize(objective, n_trials=NUM_TRIAL_STANDARD-len(completed_trials), gc_after_trial=True)

    elapsed_time = time.time() - start_time

    print(f"Optimization completed. Time taken: {elapsed_time / 60:.2f} minutes")

    best_trial = study.best_trial
    print("Best trial:", best_trial.number)
    print(f"Validation accuracy: {best_trial.value:.7f}")
    print("Best hyperparameters:", best_trial.params)


if __name__ == '__main__':
    main()
