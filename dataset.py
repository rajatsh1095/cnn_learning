import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset:
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = []

        class_names = sorted(os.listdir(root_dir))

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)

            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)

                label = self.class_to_idx[class_name]
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size))

        image = np.array(image).astype(np.float32)

        image /= 255.0

        image = np.transpose(image, [2, 0, 1])

        return image

    def get_batch(self, batch_size=4):

        indices = np.random.choice(len(self.samples), batch_size)

        images = []
        labels = []

        for idx in indices:

            image_path, label = self.samples[idx]
            image = self.load_image(image_path)

            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y