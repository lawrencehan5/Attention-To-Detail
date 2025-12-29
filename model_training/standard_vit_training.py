import gc
import logging
import warnings
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

from config import MAIN_SEED, FINAL_VALID_PERCENTAGE, MAX_IMAGE_PIXELS, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker, evaluate_model
from data.datasets import load_datasets
from training.standard_training import train_standard

torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def main():
    # set up datasets
    _, (train_dataset_224, valid_dataset_224, test_dataset_224) = load_datasets()

    # combine training and validation dataset and create a small validation split
    set_seed(MAIN_SEED)

    full_train_dataset = ConcatDataset([train_dataset_224, valid_dataset_224])

    valid_size = int(len(full_train_dataset) * FINAL_VALID_PERCENTAGE)
    train_size = len(full_train_dataset) - valid_size

    generator = torch.Generator().manual_seed(MAIN_SEED)

    final_train_224, final_valid_224 = random_split(full_train_dataset, [train_size, valid_size], generator=generator)

    print("Final training size:", len(final_train_224))
    print("Final validation size:", len(final_valid_224))

    # load pre-trained model
    _ = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # final standard vit model training
    all_metrics = []

    best_param = {
        'lr': 7.480854951931871e-05,
        'weight_decay': 0.0004369279810718895,
        'batch_size': 128,
        'dropout_rate': 0.1
    }

    for seed in (100, 200, 300, 400, 500):
        set_seed(seed)

        train_g = torch.Generator()
        train_g.manual_seed(seed)

        valid_g = torch.Generator()
        valid_g.manual_seed(MAIN_SEED)

        test_g = torch.Generator()
        test_g.manual_seed(MAIN_SEED)

        train_loader_224 = DataLoader(
            final_train_224,
            batch_size=best_param['batch_size'],
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=train_g
        )

        valid_loader_224 = DataLoader(
            final_valid_224,
            batch_size=best_param['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=valid_g
        )

        test_loader_224 = DataLoader(
            test_dataset_224,
            batch_size=best_param['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=test_g
        )

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Sequential(
            nn.Dropout(best_param['dropout_rate']),
            nn.Linear(model.heads.head.in_features, 2)
        )

        result = train_standard(
            model=model,
            train_loader=train_loader_224,
            valid_loader=valid_loader_224,
            num_epochs=50,
            lr=best_param['lr'],
            weight_decay=best_param['weight_decay'],
            eta_min=ETA_MIN,
            label_smoothing=LABEL_SMOOTHING,
            verbose=True,
            plot = True,
            save = "transfer_vit"
        )

        torch.save(result, f"transfer_vit_seed{seed}.pt")

        # load best epoch
        model.load_state_dict(torch.load(f"transfer_vit.pt"))

        metrics = evaluate_model(model, test_loader_224, global_local=False)
        all_metrics.append(metrics)
        print(metrics)

        del model
        del train_loader_224
        del valid_loader_224
        del test_loader_224
        torch.cuda.empty_cache()
        gc.collect()

    # final result calculation
    metric_arrays = {key: np.array([m[key] for m in all_metrics]) for key in all_metrics[0]}

    for metric_name, values in metric_arrays.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        print(f"{metric_name.replace('_', ' ').title():<25}: {mean * 100:.2f}% ± {std * 100:.2f}%")


if __name__ == '__main__':
    main()
