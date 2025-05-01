import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
import hyperparameters as hp


class ResNet50(tf.keras.Model):
       def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()

        # Load the ResNet50 base model
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet' if pretrained else None,  # ← Toggle pretrained vs. scratch
            input_shape=(224, 224, 3),
            pooling='avg'
        )

        # Freeze the base model if pretrained
        self.base_model.trainable = not pretrained  # If pretrained, freeze; else train all

        # Classification head for 15-scene classification
        self.head = tf.keras.Sequential([
            Dense(256, activation='relu', name='dense1'),
            Dropout(0.3, name='dropout1'),
            Dense(128, activation='relu', name='dense2'),
            Dropout(0.3, name='dropout2'),
            Dense(units=hp.num_classes, activation='softmax', name='output_layer')
        ])

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

       def call(self, x):
              x = tf.keras.applications.resnet.preprocess_input(x * 255.0)  # Match ResNet preprocessing
              x = self.base_model(x)
              x = self.head(x)
              return x

       @staticmethod
       def loss_fn(labels, predictions):
              cce = tf.keras.losses.SparseCategoricalCrossentropy()
              return cce(labels, predictions)

        

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TASK 3
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        # TASK 3
        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.
        for layer in self.vgg16:
              layer.trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [
          Flatten(name='flatten'),
          Dense(256, activation='relu', name='dense1'),        
          Dropout(0.3, name='dropout1'),
          Dense(128, activation='relu', name='dense2'),
          Dropout(0.3, name='dropout2'),
          Dense(96, activation='relu', name='dense3'),
          Dense(units=hp.num_classes, activation='softmax', name='output_layer')
        ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TASK 3
        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        #       Read the documentation carefully, some might not work with our 
        #       model!

        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        return cce(labels, predictions)
