"""
Creates augmented and size-normalized versions of the datasets
(baubles, christmas trees, snowy landscapes).
"""
from __future__ import print_function, division
import os
import random
import re
import numpy as np
from scipy import misc
from scipy import ndimage
from ImageAugmenter import create_aug_matrices
from skimage import transform as tf

random.seed(42)
np.random.seed(42)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
READ_MAIN_DIR = os.path.join(FILE_DIR, "downloaded/")
WRITE_MAIN_DIR = os.path.join(FILE_DIR, "preprocessed/")
DATASETS = ["baubles", "christmas-trees", "snowy-landscapes"]
SCALES = {
    "baubles": (64, 64),
    "christmas-trees": (64, 64),
    "snowy-landscapes": (64, 64+32)
}
AUGMENTATIONS = {
    "baubles": {
        "n": 8, "hflip": True, "vflip": False,
        "scale_to_percent": (0.9, 1.1), "scale_axis_equally": True,
        "rotation_deg": 10, "shear_deg": 0,
        "translation_x_px": 8, "translation_y_px": 8,
        "brightness_change": 0.1, "noise_mean": 0.0, "noise_std": 0.025
    },
    "christmas-trees": {
        "n": 4, "hflip": True, "vflip": False,
        "scale_to_percent": (0.9, 1.1), "scale_axis_equally": True,
        "rotation_deg": 5, "shear_deg": 0,
        "translation_x_px": 8, "translation_y_px": 4,
        "brightness_change": 0.1, "noise_mean": 0.0, "noise_std": 0.025
    },
    "snowy-landscapes": {
        "n": 5, "hflip": True, "vflip": False,
        "scale_to_percent": (0.98, 1.02), "scale_axis_equally": True,
        "rotation_deg": 4, "shear_deg": 0,
        "translation_x_px": 8, "translation_y_px": 4,
        "brightness_change": 0.1, "noise_mean": 0.0, "noise_std": 0.00
    }
}

def main():
    """Main method that reads the images, augments them, normalizes
    their size and then saves them."""
    nb_processed = 0
    for dataset_name in DATASETS:
        print("-----------------")
        print("Dataset: '%s'" % (dataset_name,))
        print("-----------------")

        dataset_dir = os.path.join(WRITE_MAIN_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        dataset = Dataset([os.path.join(READ_MAIN_DIR, dataset_name)])
        print("Found %d images total." % (len(dataset.fps),))

        errors = []

        scale_height, scale_width = SCALES[dataset_name]
        target_aspect_ratio = scale_width / scale_height

        # iterate over every image in the current dataset,
        # augment that image N times, add cols/rows until target aspect ratio
        # is reached, resize it (e.g. 64x64), save it
        for img_idx, (image_filepath, image) in enumerate(zip(dataset.fps, dataset.get_images())):
            print("[%s] Image %d of %d (%.2f%%)..." \
                  % (dataset_name, img_idx+1, len(dataset.fps),
                     100*(img_idx+1)/len(dataset.fps)))

            # IOErrors during loading of images result here in a None value
            if image is None:
                print("Error / None")
                errors.append((
                    image_filepath,
                    "Failed to load image '%s' (idx %d for dataset %s)" \
                    % (image_filepath, img_idx, dataset_name)
                ))
            else:
                # resize too big images to smaller ones before any augmentation
                # (for performance reasons)
                height = image.shape[0]
                width = image.shape[1]
                aspect_ratio = width / height
                if width > 1000 or height > 1000:
                    image = misc.imresize(image, (1000, int(1000 * aspect_ratio)))

                # augment image
                # converts augmented versions automatically to float32, 0-1
                augmentations = augment(image, **AUGMENTATIONS[dataset_name])

                # create list of original image + augmented versions
                images_aug = [image / 255.0]
                images_aug.extend(augmentations)

                # for each augmented version of the images:
                # resize it to target aspect ratio (e.g. same width and height),
                # save it
                for aug_idx, image_aug in enumerate(images_aug):
                    image_aug = to_aspect_ratio_add(image_aug, target_aspect_ratio)
                    filename = "{:0>6}_{:0>3}.jpg".format(img_idx, aug_idx)
                    img_scaled = misc.imresize(image_aug, (scale_height, scale_width))
                    misc.imsave(os.path.join(dataset_dir, filename), img_scaled)

            nb_processed += 1

    print("Processed %d images with %d errors." % (nb_processed, len(errors)))
    for (fp, err) in errors:
        print("File %s error:" % (fp,))
        print(err)
    print("Finished.")

def to_aspect_ratio_add(image, target_ratio):
    """Add black cols/rows to an image so that it matches an aspect ratio.
    Args:
        image           The image to change.
        target_ratio    Intended final aspect ratio (width/height).
    Returns:
        Modified image
    """
    height = image.shape[0]
    width = image.shape[1]
    ratio = width / height

    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    # loops here are inefficient, but easy to read
    i = 0
    if ratio < target_ratio:
        # vertical image, height > width
        while ratio < target_ratio:
            if i % 2 == 1:
                pad_right += 1
                width += 1
            else: # i % 4 == 3
                pad_left += 1
                width += 1
            ratio = width / height
            i += 1
    elif ratio > target_ratio:
        # horizontal image, width > height
        while ratio > target_ratio:
            if i % 2 == 1:
                pad_top += 1
                height += 1
            else: # i % 4 == 3
                pad_bottom += 1
                height += 1
            ratio = width / height
            i += 1

    # add black cols/rows
    if any([val > 0 for val in [pad_top, pad_bottom, pad_left, pad_right]]):
        image = np.pad(image, ((pad_top, pad_bottom), \
                               (pad_left, pad_right), \
                               (0, 0)), \
                              mode="constant")

    return image

# currently not used in this script, only the _add method above is used
def to_aspect_ratio_add_and_remove(image, target_ratio):
    """Add and remove cols/rows from an image so that it matches an aspect ratio.
    Args:
        image           The image to change.
        target_ratio    Intended final aspect ratio (width/height).
    Returns:
        Modified image
    """
    height = image.shape[0]
    width = image.shape[1]
    ratio = width / height

    remove_top = 0
    remove_right = 0
    remove_bottom = 0
    remove_left = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    # loops here are inefficient, but easy to read
    i = 0
    if ratio < target_ratio:
        # vertical image, height > width
        while ratio < target_ratio:
            if i % 4 == 0:
                remove_top += 1
                height -= 1
            elif i % 4 == 2:
                remove_bottom += 1
                height -= 1
            elif i % 4 == 1:
                pad_right += 1
                width += 1
            else: # i % 4 == 3
                pad_left += 1
                width += 1
            ratio = width / height
            i += 1
    elif ratio > target_ratio:
        # horizontal image, width > height
        while ratio > target_ratio:
            if i % 4 == 0:
                remove_right += 1
                width -= 1
            elif i % 4 == 2:
                remove_left += 1
                width -= 1
            elif i % 4 == 1:
                pad_top += 1
                height += 1
            else: # i % 4 == 3
                pad_bottom += 1
                height += 1
            ratio = width / height
            i += 1

    # remove cols/rows
    if any([val > 0 for val in [remove_top, remove_right, remove_bottom, remove_left]]):
        image = image[remove_top:(height - remove_bottom), remove_left:(width - remove_right), ...]

    # add cols/rows (black)
    if any([val > 0 for val in [pad_top, pad_bottom, pad_left, pad_right]]):
        image = np.pad(image, ((pad_top, pad_bottom), \
                               (pad_left, pad_right), \
                               (0, 0)), \
                              mode="constant")

    return image


def augment(image, n,
            hflip=False, vflip=False, scale_to_percent=1.0, scale_axis_equally=True,
            rotation_deg=0, shear_deg=0, translation_x_px=0, translation_y_px=0,
            brightness_change=0.0, noise_mean=0.0, noise_std=0.0):
    """Augment an image n times.
    Args:
            n                   Number of augmentations to generate.
            hflip               Allow horizontal flipping (yes/no).
            vflip               Allow vertical flipping (yes/no)
            scale_to_percent    How much scaling/zooming to allow. Values are around 1.0.
                                E.g. 1.1 is -10% to +10%
                                E.g. (0.7, 1.05) is -30% to 5%.
            scale_axis_equally  Whether to enforce equal scaling of x and y axis.
            rotation_deg        How much rotation to allow. E.g. 5 is -5 degrees to +5 degrees.
            shear_deg           How much shearing to allow.
            translation_x_px    How many pixels of translation along the x axis to allow.
            translation_y_px    How many pixels of translation along the y axis to allow.
            brightness_change   How much change in brightness to allow. Values are around 0.0.
                                E.g. 0.2 is -20% to +20%.
            noise_mean          Mean value of gaussian noise to add.
            noise_std           Standard deviation of gaussian noise to add.
    Returns:
        List of numpy arrays
    """
    assert n >= 0
    result = []
    if n == 0:
        return result

    width = image.shape[0]
    height = image.shape[1]
    matrices = create_aug_matrices(n, img_width_px=width, img_height_px=height,
                                   scale_to_percent=scale_to_percent,
                                   scale_axis_equally=scale_axis_equally,
                                   rotation_deg=rotation_deg,
                                   shear_deg=shear_deg,
                                   translation_x_px=translation_x_px,
                                   translation_y_px=translation_y_px)
    for i in range(n):
        img = np.copy(image)
        matrix = matrices[i]

        # random horizontal / vertical flip
        if hflip and i % 2 == 0:
            img = np.fliplr(img)
        if vflip and random.random() > 0.5:
            img = np.flipud(img)

        # random brightness adjustment
        by_percent = random.uniform(1.0 - brightness_change, 1.0 + brightness_change)
        img = img * by_percent

        # gaussian noise
        # numpy requires a std above 0
        if noise_std > 0:
            img = img + (255 * np.random.normal(noise_mean, noise_std, (img.shape)))

        # clip to 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)

        arr = tf.warp(img, matrix, mode="nearest") # projects to float 0-1
        img = np.array(arr * 255, dtype=np.uint8)
        result.append(img)

    return result

class Dataset(object):
    """Helper class to handle the loading of the LFW dataset dataset."""
    def __init__(self, dirs):
        """Instantiate a dataset object.
        Args:
            dirs    List of filepaths to directories. Direct subdirectories will be read.
        """
        self.dirs = dirs
        self.fps = self.get_filepaths()

    def get_filepaths(self):
        """Create a list of all filepaths in the dataset's directories.
        Returns:
            List of filepaths.
        """
        image_filepaths = set()
        for one_dir in self.dirs:
            for root, dirnames, filenames in os.walk(one_dir):
                for filename in filenames:
                    if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename):
                        image_filepaths.add(os.path.join(root, filename))
        image_filepaths = sorted(list(image_filepaths))
        return image_filepaths

    def get_images(self, start_at=None, count=None):
        """Returns a generator of images.
        Args:
            start_at    Index of first image to return or None.
            count       Maximum number of images to return or None.
        Returns:
            Generator of images (numpy arrays).
        """
        start_at = 0 if start_at is None else start_at
        end_at = len(self.fps) if count is None else start_at+count
        for fp in self.fps[start_at:end_at]:
            try:
                image = ndimage.imread(fp, mode="RGB")
            except IOError as exc:
                image = None
            yield image

if __name__ == "__main__":
    main()
