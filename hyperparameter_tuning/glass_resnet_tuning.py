import os
import optuna
import gc
import time
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

from config import MAIN_SEED, NUM_TRIAL_GLASS, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker
from data.datasets import load_datasets, MultiViewDataset
from models.glass_models import GlobalLocalAttentionModel
from training.glass_training import train_glass

torch.set_float32_matmul_precision('high')


def main():
    # set up datasets
    (train_dataset_orig, valid_dataset_orig, test_dataset_orig), _ = load_datasets()

    # load pre-trained model
    _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # glass resnet hyperparameter optimization 
    def objective(trial):
        # set seed
        set_seed(MAIN_SEED + trial.number)

        # define hyperparameter search space
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3])

        # unique hyperparameters to glass
        lr_global = trial.suggest_float("lr_global", 1e-6, 1e-4, log=True)
        lr_local = trial.suggest_float("lr_local", 1e-6, 1e-4, log=True)
        lr_atten_class = trial.suggest_float("lr_atten_class", 5e-6, 1e-3, log=True)
        wd_global = trial.suggest_float("wd_global", 1e-7, 1e-3, log=True)
        wd_local = trial.suggest_float("wd_local", 1e-7, 1e-3, log=True)
        num_crops = trial.suggest_categorical("num_crops", [2, 4, 6, 8, 10])

        # create folder to save results
        os.makedirs("gl_resnet_results", exist_ok=True)

        # new dataset of current number of local crops
        train_dataset_gl = MultiViewDataset(
            train_dataset_orig,
            num_crops=num_crops,
            patch_size=224
        )

        valid_dataset_gl = MultiViewDataset(
            valid_dataset_orig,
            num_crops=num_crops,
            patch_size=224
        )

        # for reproducibility
        g = torch.Generator()
        g.manual_seed(MAIN_SEED + trial.number)

        # fixed generator for validation
        g_fixed = torch.Generator()
        g_fixed.manual_seed(MAIN_SEED)

        # new dataloaders of current dataset and batch size
        train_loader = DataLoader(
            dataset=train_dataset_gl,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        valid_loader = DataLoader(
            dataset=valid_dataset_gl,
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
        model = GlobalLocalAttentionModel(dropout_rate=dropout_rate, model_type="resnet")

        # train and obtain result
        result = train_glass(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=25,
            all_lr={'lr_global': lr_global, 'lr_local': lr_local, 'lr_atten_class': lr_atten_class},
            all_wd={'wd_global': wd_global, 'wd_local': wd_local},
            eta_min=ETA_MIN,
            label_smoothing=LABEL_SMOOTHING,
            trial=trial,
            plot=False
        )

        torch.save(result, f"gl_resnet_results/gl_resnet_trial_{trial.number}.pt")

        if result['pruned']:
            raise optuna.TrialPruned()

        # clean up
        del model
        del train_loader
        del valid_loader
        del train_dataset_gl
        del valid_dataset_gl
        gc.collect()
        torch.cuda.empty_cache()

        return result['max_valid_acc']

    # search for hyperparamters
    storage = "sqlite:///./gl_resnet.db"
    study_name = "gl_resnet_hyperparameters_optimization"

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
    study.optimize(objective, n_trials=NUM_TRIAL_GLASS-len(completed_trials), gc_after_trial=True)

    elapsed_time = time.time() - start_time

    print(f"Optimization completed. Time taken: {elapsed_time / 60:.2f} minutes")

    best_trial = study.best_trial
    print("Best trial:", best_trial.number)
    print(f"Validation accuracy: {best_trial.value:.7f}")
    print("Best hyperparameters:", best_trial.params)


if __name__ == '__main__':
    main()
