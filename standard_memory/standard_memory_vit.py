import gc
import torch
import logging
import warnings
from PIL import Image
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader

from config import MAIN_SEED, MAX_IMAGE_PIXELS, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker
from data.datasets import load_datasets
from training.standard_training import train_standard

torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def main():
    # set up datasets
    _, (train_dataset_224, valid_dataset_224, test_dataset_224) = load_datasets()
    
    # load pre-trained model
    _ = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # standard vit model training
    best_param = {
        'lr': 7.480854951931871e-05,
        'weight_decay': 0.0004369279810718895,
        'batch_size': 32,
        'dropout_rate': 0.1
    }

    set_seed(MAIN_SEED)

    train_g = torch.Generator()
    train_g.manual_seed(MAIN_SEED)

    valid_g = torch.Generator()
    valid_g.manual_seed(MAIN_SEED)

    train_loader_224 = DataLoader(
        dataset=train_dataset_224,
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
        dataset=valid_dataset_224,
        batch_size=best_param['batch_size'],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=valid_g
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
        num_epochs=25,
        lr=best_param['lr'],
        weight_decay=best_param['weight_decay'],
        eta_min=ETA_MIN,
        label_smoothing=LABEL_SMOOTHING,
        verbose=False,
        plot=False
    )
    
    mem_peak_MB = max(res["mem_peak_MB"] for res in result["gpu_memory"])
    all_epoch_times = [res["epoch_time_sec"] for res in result["gpu_memory"]]
    avg_epoch_time = sum(all_epoch_times) / len(all_epoch_times)
    max_valid_acc = result['max_valid_acc']
    
    print(f"max peak memory={mem_peak_MB}, mean epoch time={avg_epoch_time}, max valid accuracy={max_valid_acc}")

    del model
    del train_loader_224
    del valid_loader_224
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
