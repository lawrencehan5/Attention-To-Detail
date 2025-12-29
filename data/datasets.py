import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from config import DATA_ROOT, MAIN_SEED, VALID_SIZE, TEST_SIZE, MEAN, STD
from utils import set_seed

#------------------------------------------
# load original size and 224x224 datasets |
#------------------------------------------
def load_datasets():
    transform_orig = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset_orig = datasets.ImageFolder(DATA_ROOT, transform=transform_orig)

    labels = [label for _, label in full_dataset_orig.samples]

    set_seed(MAIN_SEED)

    # split train, with valid and test
    train_indices, valid_test_indices = train_test_split(
        range(len(full_dataset_orig)),
        test_size=(VALID_SIZE + TEST_SIZE) / len(full_dataset_orig),
        shuffle=True,
        stratify=labels,
        random_state=MAIN_SEED
    )

    # split valid and test
    valid_indices, test_indices = train_test_split(
        valid_test_indices,
        test_size=TEST_SIZE / (VALID_SIZE + TEST_SIZE),
        shuffle=True,
        stratify=[labels[i] for i in valid_test_indices],
        random_state=MAIN_SEED
    )

    # create subset
    train_dataset_orig = Subset(full_dataset_orig, train_indices)
    valid_dataset_orig = Subset(full_dataset_orig, valid_indices)
    test_dataset_orig = Subset(full_dataset_orig, test_indices)

    # resize to 224 x 224 for standard transfer learning
    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    full_dataset_224 = datasets.ImageFolder(DATA_ROOT, transform=transform_224)

    train_dataset_224 = Subset(full_dataset_224, train_indices)
    valid_dataset_224 = Subset(full_dataset_224, valid_indices)
    test_dataset_224  = Subset(full_dataset_224, test_indices)

    return [(train_dataset_orig, valid_dataset_orig, test_dataset_orig), 
            (train_dataset_224, valid_dataset_224, test_dataset_224)]


#----------------------------------
# set up multiview custom dataset |
#----------------------------------
class MultiViewDataset(Dataset):
    def __init__(self, origal_dataset, num_crops=4, patch_size=224):
        self.dataset = origal_dataset
        self.num_crops = num_crops
        self.patch_size = patch_size
        self.generator = torch.Generator()

        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        global_view, local_crops = generate_views(
            image,
            num_crops=self.num_crops,
            patch_size=self.patch_size,
            g=self.generator
        )

        global_view = (global_view - self.mean) / self.std

        local_crops = (local_crops - self.mean) / self.std

        return global_view, local_crops, label
    

#--------------------------------------------------
# generate stratified global view and local crops |
#--------------------------------------------------
def generate_views(image, num_crops=4, patch_size=224, g=None):
    _, h, w = image.shape

    # global 224x224 view
    global_view = TF.resize(image, [patch_size, patch_size])

    # determine adaptive grid size
    min_dim = min(h, w)
    G = min(min_dim // patch_size, 8)

    # if there are less grids than number of local crops
    if G * G < num_crops:
        G = 0 # random crop on the entire image

    crops = []

    if G > 0:
        cell_h = h // G
        cell_w = w // G

        # list all grid cells
        cells = [(i, j) for i in range(G) for j in range(G)]

        # sample cells without replacement
        # num_crops < G*G since we have max(num_crops) <= 10
        indices = torch.randperm(len(cells), generator=g)[:num_crops]
        sampled_cells = [cells[i] for i in indices.tolist()]

        for (ci, cj) in sampled_cells:

            # cell boundaries
            y0 = ci * cell_h
            x0 = cj * cell_w

            # safe boundaries for patch sampling
            max_dy = max(0, cell_h - patch_size)
            max_dx = max(0, cell_w - patch_size)

            # random offset inside this cell
            dy = int(torch.randint(0, max_dy+1, (1,), generator=g))
            dx = int(torch.randint(0, max_dx+1, (1,), generator=g))

            top = y0 + dy
            left = x0 + dx

            # clamp to ensure full patch fits
            top = min(top, max(0, h - patch_size))
            left = min(left, max(0, w - patch_size))

            crop = image[:, top:top + patch_size, left:left + patch_size]

            # pad if crop is smaller than patch_size
            if crop.shape[1] != patch_size or crop.shape[2] != patch_size:
                pad_h = patch_size - crop.shape[1]
                pad_w = patch_size - crop.shape[2]
                crop = F.pad(crop, (0, pad_w, 0, pad_h), value=0)

            crops.append(crop)
    else:
        # random crops on entire image
        for _ in range(num_crops):
            max_y = max(0, h - patch_size)
            max_x = max(0, w - patch_size)

            top = int(torch.randint(0, max_y+1, (1,), generator=g))
            left = int(torch.randint(0, max_x+1, (1,), generator=g))
            crop = image[:, top:top+patch_size, left:left+patch_size]

            # pad if needed
            if crop.shape[1] != patch_size or crop.shape[2] != patch_size:
                pad_h = patch_size - crop.shape[1]
                pad_w = patch_size - crop.shape[2]
                crop = F.pad(crop, (0, pad_w, 0, pad_h), value=0)

            crops.append(crop)

    local_crops = torch.stack(crops)
    return global_view, local_crops
