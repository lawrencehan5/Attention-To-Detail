#--------------------------------------------------------
# data preprocessing and calculating average image size |
#--------------------------------------------------------
import os
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from config import DATA_ROOT, MAX_IMAGE_PIXELS

# max image pixels
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# load truncated images
Image.LOAD_TRUNCATED_IMAGES = True

corrupted_count = 0

fake_width, fake_height, fake_count = 0, 0, 0
real_width, real_height, real_count = 0, 0, 0

# counters for mode changes
changed_fake_images = 0
changed_real_images = 0

# dictionary of original modes
mode_changes = {
    "fake": {},
    "real": {}
}

# for mean and standard deviation calculation
pixel_sum = torch.zeros(3, dtype=torch.float64)
pixel_sum_square = torch.zeros(3, dtype=torch.float64)
total_pixels = 0

for class_name in ("fake", "real"):
    class_dir = os.path.join(DATA_ROOT, class_name)

    with os.scandir(class_dir) as files:
        images = [f for f in files
                  if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image in tqdm(images, desc=f"Processing {class_name} images", unit="images", leave=False):
        img_path = image.path

        try:
            with Image.open(img_path) as pil_img:
                pil_img.load()

                original_mode = pil_img.mode

                # convert to rgb if image not in rgb
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                    # overwrite original file
                    pil_img.save(img_path)

                    # track mode change
                    if class_name == 'fake':
                        changed_fake_images += 1
                    else:
                        changed_real_images += 1

                    # store original mode information
                    if original_mode not in mode_changes[class_name]:
                        mode_changes[class_name][original_mode] = 0
                    mode_changes[class_name][original_mode] += 1

                # collect image dimensions
                width, height = pil_img.size

                # fake class
                if class_name == 'fake':
                    fake_width += width
                    fake_height += height
                    fake_count += 1

                # real class
                elif class_name == 'real':
                    real_width += width
                    real_height += height
                    real_count += 1

                # mean and std
                img_tensor = TF.to_tensor(pil_img).to(dtype=torch.float64)

                pixel_sum += img_tensor.sum(dim=(1, 2))
                pixel_sum_square += (img_tensor ** 2).sum(dim=(1, 2))
                total_pixels += width * height

        # corrupted images
        except Exception as e:
            corrupted_count += 1
            print(f"Corrupted image detected: {img_path} - {str(e)}")

# calculate average width and height for each class
fake_avg_width = fake_width / fake_count if fake_count > 0 else 0
fake_avg_height = fake_height / fake_count if fake_count > 0 else 0
real_avg_width = real_width / real_count if real_count > 0 else 0
real_avg_height = real_height / real_count if real_count > 0 else 0

# print statistics
print("\nNumber of images per class")
print(f"Fake class: {fake_count} images")
print(f"Real class: {real_count} images")
print(f"Corrupted: {corrupted_count} images")

print("\nAverage width and height")
print(f"Fake class: ({fake_avg_width}, {fake_avg_height}) pixels")
print(f"Real class: ({real_avg_width}, {real_avg_height}) pixels")

print("\nSummary of Mode Changes")
print(f"Total images changed: {changed_fake_images + changed_real_images}")
print(f"Fake class changed: {changed_fake_images}")
print(f"Real class changed: {changed_real_images}")

# mode change for each class
print("\nMode changes per class")
for class_name in ('fake', 'real'):
    if mode_changes[class_name]:
        print(f"{class_name} class:")
        for mode, count in mode_changes[class_name].items():
            print(f"mode {mode}: {count} images converted")
    else:
        print(f"No mode changes for {class_name} class.")

# calculate mean and std
mean = pixel_sum / total_pixels
std = torch.sqrt((pixel_sum_square / total_pixels) - (mean ** 2))
print("\nDataset mean and standard deviation")
print("Dataset mean (R, G, B) =", mean)
print("Dataset std  (R, G, B) =", std)
