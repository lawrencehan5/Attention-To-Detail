import gc
import torch
import logging
import warnings
import numpy as np
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from config import MAIN_SEED, MAX_IMAGE_PIXELS, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker
from data.datasets import load_datasets, MultiViewDataset
from models.glass_models import GlobalLocalAttentionModel
from training.glass_training import train_glass

torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def main():
    all_num_crops = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # set up datasets
    (train_dataset_orig, valid_dataset_orig, test_dataset_orig), _ = load_datasets()
    
    # load pre-trained model
    _ = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # glass convnext model training
    all_memory, all_time = [], []

    best_param = {
        'batch_size': 32,
        'dropout_rate': 0.2,
        'lr_global': 6.549128272517001e-06,
        'lr_local': 9.999570997037118e-05,
        'lr_atten_class': 1.679984935907822e-05,
        'wd_global': 2.852588208730554e-06,
        'wd_local': 3.8782167347490953e-07,
        'num_crops': 6
    }

    for num_crops in all_num_crops:
        set_seed(MAIN_SEED)

        train_g = torch.Generator()
        train_g.manual_seed(MAIN_SEED)

        valid_g = torch.Generator()
        valid_g.manual_seed(MAIN_SEED)

        test_g = torch.Generator()
        test_g.manual_seed(MAIN_SEED)

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

        train_loader = DataLoader(
            dataset=train_dataset_gl,
            batch_size=best_param['batch_size'],
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=train_g
        )

        valid_loader = DataLoader(
            dataset=valid_dataset_gl,
            batch_size=best_param['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=valid_g
        )

        model = GlobalLocalAttentionModel(dropout_rate=best_param['dropout_rate'], model_type="convnext")

        result = train_glass(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=25,
            all_lr={'lr_global': best_param['lr_global'], 'lr_local': best_param['lr_local'], 'lr_atten_class': best_param['lr_atten_class']},
            all_wd={'wd_global': best_param['wd_global'], 'wd_local': best_param['wd_local']},
            eta_min=ETA_MIN,
            label_smoothing=LABEL_SMOOTHING,
            verbose=False,
            plot=False
        )
        
        mem_peak_MB = max(res["mem_peak_MB"] for res in result["gpu_memory"])
        all_epoch_times = [res["epoch_time_sec"] for res in result["gpu_memory"]]
        avg_epoch_time = sum(all_epoch_times) / len(all_epoch_times)
        max_valid_acc = result['max_valid_acc']

        all_memory.append(mem_peak_MB)
        all_time.append(avg_epoch_time)
        
        print(f"num_crops={num_crops}, max peak memory={mem_peak_MB}, mean epoch time={avg_epoch_time}, max valid accuracy={max_valid_acc}")

        del model
        del train_loader
        del valid_loader
        del train_dataset_gl
        del valid_dataset_gl
        gc.collect()
        torch.cuda.empty_cache()

    # memory profiling
    x = np.array(all_num_crops)
    y = np.array(all_memory)

    slope, intercept = np.polyfit(x, y, 1)
    print(f"Static Memory (Intercept): {intercept:.2f} MB")
    print(f"Dynamic Memory (Slope): {slope:.2f} MB per crop")
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    print(f"Linearity R2: {r2:.5f}")

    # time profiling
    x = np.array(all_num_crops)
    y = np.array(all_time)

    slope, intercept = np.polyfit(x, y, 1)
    print(f"Static Time (Intercept): {intercept:.2f} sec/epoch")
    print(f"Dynamic Time (Slope): {slope:.2f} sec/epoch per crop")
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    print(f"Linearity R2: {r2:.5f}")

if __name__ == '__main__':
    main()
