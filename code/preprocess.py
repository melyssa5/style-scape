"""
Homework 5 - CNNs
CSCI1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and stddev for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.stddev = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_stddev()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), task == '3', True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), task == '3', False, False)
        self.stylized_data = self.get_data(
            os.path.join(self.data_path, "stylized/"), task in ['4', '5'], False, False) \
            if os.path.exists(os.path.join(self.data_path, "stylized/")) else None
        
        
        

    def calc_mean_and_stddev(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        # TASK 1
        # TODO: Calculate the mean and standard deviation
        #       of the samples in data_sample and store them in
        #       self.mean and self.stddev respectively.
        #
        #       Note: This is _not_ a mean over all pixels;
        #             it is a mean image (the mean input data point).
        #       
        #             For example, the mean of the two images:
        #
        #             [[[0, 0, 100], [0, 0, 100]],      [[[100, 0, 0], [100, 0, 0]],
        #              [[0, 100, 0], [0, 100, 0]],  and  [[0, 100, 0], [0, 100, 0]],
        #              [[100, 0, 0], [100, 0, 0]]]       [[0, 0, 100], [0, 0, 100]]]
        #
        #             would be
        #
        #             [[[50, 0, 50], [50, 0, 50]],
        #              [[0, 100, 0], [0, 100, 0]],
        #              [[50, 0, 50], [50, 0, 50]]]
        #
        # ==========================================================

        self.mean = np.mean(data_sample, axis=0)
        self.stddev = np.std(data_sample, axis=0) + 1e-8

        # ==========================================================

        print("Dataset mean shape: [{0}, {1}, {2}]".format(
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[2]))

        print("Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0,0,0], self.mean[0,0,1], self.mean[0,0,2]))

        print("Dataset stddev shape: [{0}, {1}, {2}]".format(
            self.stddev.shape[0], self.stddev.shape[1], self.stddev.shape[2]))

        print("Dataset stddev top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.stddev[0,0,0], self.stddev[0,0,1], self.stddev[0,0,2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        # TASK 1
        # TODO: Standardize the input image. Use self.mean and self.stddev
        #       that were calculated in calc_mean_and_stddev() to perform
        #       the standardization.
        # =============================================================

        img = (img - self.mean) / self.stddev 

        # =============================================================

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        if self.task == '3':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            img = self.standardize(img)
        return img

    def custom_preprocess_fn(self, img):
        """ Custom preprocess function for ImageDataGenerator. """

        if self.task == '3':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            img = self.standardize(img)

        # EXTRA CREDIT: 
        # Write your own custom data augmentation procedure, creating
        # an effect that cannot be achieved using the arguments of
        # ImageDataGenerator. This can potentially boost your accuracy
        # in the validation set. Note that this augmentation should
        # only be applied to some input images, so make use of the
        # 'random' module to make sure this happens. Also, make sure
        # that ImageDataGenerator uses *this* function for preprocessing
        # on augmented data.

        if random.random() < 0.3:
            img = img + tf.random.uniform(
                (hp.img_size, hp.img_size, 1),
                minval=-0.1,
                maxval=0.1)
        # Randomly change contrast
        if random.random() < 0.3:
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        # Randomly crop images
        if random.random() < 0.5:
            crop_size = tf.cast(tf.shape(img)[:2] * 0.8, tf.int32)  
            img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
            img = tf.image.resize(img, [hp.img_size, hp.img_size])

        return img

    def get_data(self, path, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
            # TODO: Use the arguments of ImageDataGenerator()
            #       to augment the data. Leave the
            #       preprocessing_function argument as is unless
            #       you have written your own custom preprocessing
            #       function (see custom_preprocess_fn()).
            #
            # Documentation for ImageDataGenerator: https://bit.ly/2wN2EmK
            #
            # ============================================================

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                rotation_range=15,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range = [.07, 1.2],
                zoom_range=0.1,
                shear_range=0.1)

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 224 if is_vgg else hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
