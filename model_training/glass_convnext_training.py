import gc
import torch
import logging
import warnings
import numpy as np
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

from config import MAIN_SEED, FINAL_VALID_PERCENTAGE, MAX_IMAGE_PIXELS, ETA_MIN, LABEL_SMOOTHING
from utils import set_seed, seed_worker, evaluate_model
from data.datasets import load_datasets, MultiViewDataset
from models.glass_models import GlobalLocalAttentionModel
from training.glass_training import train_glass

torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def main():
    # set up datasets
    (train_dataset_orig, valid_dataset_orig, test_dataset_orig), _ = load_datasets()

    # combine training and validation dataset and create a small validation split
    set_seed(MAIN_SEED)

    full_train_dataset = ConcatDataset([train_dataset_orig, valid_dataset_orig])

    valid_size = int(len(full_train_dataset) * FINAL_VALID_PERCENTAGE)
    train_size = len(full_train_dataset) - valid_size

    generator = torch.Generator().manual_seed(MAIN_SEED)

    final_train_orig, final_valid_orig = random_split(full_train_dataset, [train_size, valid_size], generator=generator)

    print("Final training size:", len(final_train_orig))
    print("Final validation size:", len(final_valid_orig))
    
    # load pre-trained model
    _ = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # final glass convnext model training
    all_metrics = []

    best_param = {
        'batch_size': 32,
        'dropout_rate': 0.2,
        'lr_global': 6.549128272517001e-06,
        'lr_local': 9.999570997037118e-05,
        'lr_atten_class': 1.679984935907822e-05,
        'wd_global': 2.852588208730554e-06,
        'wd_local': 3.8782167347490953e-07,
        'num_crops': 4
    }

    for seed in (100, 200, 300, 400, 500):
        set_seed(seed)

        train_g = torch.Generator()
        train_g.manual_seed(seed)

        valid_g = torch.Generator()
        valid_g.manual_seed(MAIN_SEED)

        test_g = torch.Generator()
        test_g.manual_seed(MAIN_SEED)

        train_dataset_gl = MultiViewDataset(
            final_train_orig,
            num_crops=best_param['num_crops'],
            patch_size=224
        )

        valid_dataset_gl = MultiViewDataset(
            final_valid_orig,
            num_crops=best_param['num_crops'],
            patch_size=224
        )

        test_dataset_gl = MultiViewDataset(
            test_dataset_orig,
            num_crops=best_param['num_crops'],
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

        test_loader = DataLoader(
            dataset=test_dataset_gl,
            batch_size=best_param['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
            generator=test_g
        )

        model = GlobalLocalAttentionModel(dropout_rate=best_param['dropout_rate'], model_type="convnext")

        result = train_glass(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=50,
            all_lr={'lr_global': best_param['lr_global'], 'lr_local': best_param['lr_local'], 'lr_atten_class': best_param['lr_atten_class']},
            all_wd={'wd_global': best_param['wd_global'], 'wd_local': best_param['wd_local']},
            eta_min=ETA_MIN,
            label_smoothing=LABEL_SMOOTHING,
            verbose=False,
            plot=True,
            save="gl_convnext"
        )

        # save results
        torch.save(result, f"gl_convnext_seed{seed}.pt")

        # load best epoch
        model.load_state_dict(torch.load(f"gl_convnext.pt"))

        metrics = evaluate_model(model, test_loader, global_local=True)
        all_metrics.append(metrics)
        print(metrics)

        del model
        del train_loader
        del valid_loader
        del test_loader
        del train_dataset_gl
        del valid_dataset_gl
        del test_dataset_gl
        gc.collect()
        torch.cuda.empty_cache()

    # final result calculation
    metric_arrays = {key: np.array([m[key] for m in all_metrics]) for key in all_metrics[0]}

    for metric_name, values in metric_arrays.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        print(f"{metric_name.replace('_', ' ').title():<25}: {mean * 100:.2f}% ± {std * 100:.2f}%")


if __name__ == '__main__':
    main()
