import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

correction_factor = 0.2
DEFAULT_ROOT = 'data/sampledata/'

def flipimg(image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped

def resize(image):
    return cv2.resize(image, None, fx = 0.5, fy = 1)

class DataPipeline():
    def __init__(self, data_root = DEFAULT_ROOT, seed = None, limit = -1):
        self.lines = []
        self.seed = seed
        self.data_root = data_root
        ## Read csv lines
        with open(data_root + 'driving_log.csv') as f:
            reader = csv.reader(f)
            # skip name row
            next(reader)
            for line in reader:
                self.lines.append(line)

        self.train_samples, self.validation_samples = train_test_split(self.lines[:limit], test_size = 0.2, random_state = self.seed)

    # account for augmentation
    def num_training(self):
        return len(self.train_samples) * 4

    # account for augmentation
    def num_validation(self):
        return len(self.validation_samples) * 4

    def get_img(self, source_path):
        filename = source_path.split('/')[-1]
        # shrink width by half
        image = resize(cv2.imread(self.data_root + 'IMG/' + filename))
        # convert to bgr
        return image[:, :, ::-1]


    # Augmentation
    def augment_sample(self, sample):
        center_img = self.get_img(sample[0])
        left_img = self.get_img(sample[1])
        right_img = self.get_img(sample[2])
        center_val = float(sample[3])
        flip_img, flip_val = flipimg(center_img, center_val)
        left_val, right_val = (center_val + correction_factor, center_val - correction_factor)
        return (center_img, left_img, right_img, flip_img), (center_val, left_val, right_val, flip_val)

    def generate(self, batch_size = 32, validation = False):
        if validation:
            samples = self.validation_samples
        else:
            samples = self.train_samples

        num_samples = len(samples)

        while 1:
            shuffle(samples, random_state = self.seed)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset: offset + batch_size]
                images, measurements = [], []

                for batch_sample in batch_samples:
                    augmented_img, augmented_val = self.augment_sample(batch_sample)
                    images.extend(augmented_img)
                    measurements.extend(augmented_val)
                x, y = np.array(images), np.array(measurements)
                yield shuffle(x, y, random_state = self.seed)
