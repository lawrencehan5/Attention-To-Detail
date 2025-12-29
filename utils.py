import random
import torch
import os
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.amp import autocast
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

from config import MAIN_SEED

#--------------------------------
# functions for reproducibility |
#--------------------------------
def set_seed(seed=MAIN_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'generator'):
            dataset.generator.manual_seed(worker_seed)


#------------------------------------------------------------------
# get accuracy and average loss of current model on a data loader |
#------------------------------------------------------------------
def get_accuracy_loss(model, data_loader, architecture):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = torch.zeros(1, device=device)

    if architecture == "glass":
        with torch.no_grad():
            for global_views, local_crops, labels in tqdm(data_loader, desc="Computing accuracy", unit="batch", leave=False):
                global_views = global_views.to(device, non_blocking=True)
                local_crops = local_crops.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                batch_size = labels.shape[0]

                with autocast(device_type='cuda'):
                    logits = model(global_views, local_crops)
                    loss = criterion(logits, labels)

                total_loss += loss.detach() * batch_size

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size
    else:
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Computing accuracy", unit="batch", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                batch_size = labels.shape[0]

                with autocast(device_type='cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)

                total_loss += loss.detach() * batch_size

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]

    accuracy = correct / total
    avg_loss = (total_loss / total).item()

    return accuracy, avg_loss


#-------------------------------------------
# function for plotting the training curve |
#-------------------------------------------
def plot_training_curve(num_epochs, train_loss, valid_acc):
    epochs = range(1, num_epochs+1)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # training loss
    ax1.set_xlabel('Epochs', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Training Loss', color='black', fontsize=14, fontweight='normal')
    ax1.plot(epochs, train_loss, color='darkblue', label='Training Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax1.set_xticks(epochs)

    # validation accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='black', fontsize=14, fontweight='normal')
    ax2.plot(epochs, valid_acc, color='darkgreen', label='Validation Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

    # add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12, frameon=True)

    plt.title('Training Progress: Loss and Accuracy', fontsize=16, fontweight='normal')
    fig.tight_layout()

    plt.show()


#---------------------------------------------
# function for getting the gpu memory status |
#---------------------------------------------
def get_cuda_memory_stats():
    return {
        "mem_allocated_MB": torch.cuda.memory_allocated() / 1024**2,
        "mem_reserved_MB": torch.cuda.memory_reserved() / 1024**2,
        "mem_peak_MB": torch.cuda.max_memory_allocated() / 1024**2,
        "mem_reserved_peak_MB": torch.cuda.max_memory_reserved() / 1024**2,
    }


#------------------------------------------------------------
# function for evaluating a model and return metric results |
#------------------------------------------------------------
def evaluate_model(model, test_loader, global_local):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    all_preds, all_labels, all_probs, all_confidences = [], [], [], []

    if global_local == False:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    logits = model(images)

                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.shape[0]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(conf.cpu().numpy())
    else:
        with torch.no_grad():
            for global_views, local_crops, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
                global_views = global_views.to(device, non_blocking=True)
                local_crops = local_crops.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    logits = model(global_views, local_crops)

                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.shape[0]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(conf.cpu().numpy())

    # accuracy
    acc = correct / total

    # classification report
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    weighted_f1 = report['weighted avg']['f1-score']
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']

    # auc
    auc_score = roc_auc_score(all_labels, np.array(all_probs)[:, 1])

    # confidence analysis
    confidences = np.array(all_confidences)
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    correct_mask = preds == labels

    avg_conf_correct = confidences[correct_mask].mean()
    avg_conf_incorrect = confidences[~correct_mask].mean()
    overall_conf = confidences.mean()

    # obtain the expected calibration error
    ece = ece_score(all_confidences, all_preds, all_labels, n_bins=15)

    # return all metrics
    return {
        'accuracy': acc,
        'auc_score': auc_score,
        'f1_score': weighted_f1,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'conf_correct': avg_conf_correct,
        'conf_incorrect': avg_conf_incorrect,
        'overall_conf': overall_conf,
        'ece': ece
    }


#--------------------------------------------------------
# function for obtaining the expected calibration error |
#--------------------------------------------------------
def ece_score(confidences, predictions, labels, n_bins=10):
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracies = (predictions == labels).astype(float)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    N = len(confidences)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_bounds[i]) & (confidences < bin_bounds[i+1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            ece += (bin_size / N) * abs(acc - conf)

    return ece
