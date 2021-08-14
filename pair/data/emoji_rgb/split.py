import os
import numpy as np
import random
valite_number = 20

# determined
np.random.seed(42)
random.seed(42)
train_path = "train/0"
validate_path = "validate/0"

filenames = sorted(os.listdir(os.path.join(os.getcwd(), train_path)))
image_list = []
for i in filenames:
    if i.endswith('.png'):
        image_list.append(i)

print(f"Find {len(image_list)} images.")
random.shuffle(image_list)
image_list = image_list[0:valite_number]
for image_name in image_list:
    source = os.path.join(os.getcwd(), train_path, image_name)
    destination = os.path.join(os.getcwd(), validate_path, image_name)
    os.rename(source, destination)
    print(f"moving {image_name}")




