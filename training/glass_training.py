import time
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import ETA_MIN, LABEL_SMOOTHING
from utils import get_accuracy_loss, plot_training_curve, get_cuda_memory_stats

#-------------------------------------
# training function for glass models |
#-------------------------------------
def train_glass(
        model,
        train_loader,
        valid_loader,
        num_epochs=10,
        all_lr={"lr_global": 0.00005, "lr_local": 0.00005, "lr_atten_class": 0.00005},
        all_wd={"wd_global": 0.0001, "wd_local": 0.0001},
        eta_min=ETA_MIN,
        label_smoothing=LABEL_SMOOTHING,
        device=None,
        verbose=False,
        trial=None,
        plot=False,
        save=None
    ):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.compile(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW([
        {"params": model.global_model.parameters(), "lr": all_lr["lr_global"], "weight_decay": all_wd["wd_global"]},
        {"params": model.local_model.parameters(), "lr": all_lr["lr_local"], "weight_decay": all_wd["wd_local"]},
        {"params": model.attention.parameters(), "lr": all_lr["lr_atten_class"], "weight_decay": 0},
        {"params": model.classifier.parameters(), "lr": all_lr["lr_atten_class"], "weight_decay": 0}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    scaler = GradScaler('cuda')

    train_loss, valid_loss, valid_acc = [], [], []
    gpu_memory = []
    best_valid_acc = -1

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        total_loss = torch.zeros(1, device=device)
        
        for global_views, local_crops, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs-1}", unit="batch", leave=False):
            global_views = global_views.to(device, non_blocking=True)
            local_crops = local_crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            batch_size = global_views.shape[0]
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                
                logits = model(global_views, local_crops)
                
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach() * batch_size
        
        # update scheduler
        scheduler.step()

        # training loss
        epoch_loss = (total_loss / len(train_loader.dataset)).item()
        train_loss.append(epoch_loss)

        # validation accuracy and loss
        val_acc, val_loss = get_accuracy_loss(model, valid_loader, "glass")
        valid_acc.append(val_acc)
        valid_loss.append(val_loss)

        # optuna pruning
        if trial is not None:

            # report epoch-level val_acc to optuna
            trial.report(val_acc, epoch)

            # early pruning
            if trial.should_prune():

                result = {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                    "max_valid_acc": max(valid_acc),
                    "best_epoch": valid_acc.index(max(valid_acc)),
                    "pruned": True
                    }
                return result

        # gpu memory status
        mem_stats = get_cuda_memory_stats()
        mem_stats["epoch_time_sec"] = time.time() - epoch_start
        gpu_memory.append(mem_stats)

        if verbose:
            print(f"Epoch {epoch}: Train Loss={epoch_loss:.7f}, Valid Loss={val_loss:.7f}, Valid Acc={val_acc:.7f}")

        # save model
        if save is not None and val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model._orig_mod.state_dict(), save + ".pt")

    # plot training curve
    if plot:
        plot_training_curve(num_epochs, train_loss, valid_acc)

    result = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "max_valid_acc": max(valid_acc),
        "best_epoch": valid_acc.index(max(valid_acc)),
        "pruned": False,
        "gpu_memory": gpu_memory
        }

    return result
