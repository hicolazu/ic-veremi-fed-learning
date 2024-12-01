import os
from abc import ABC
from metrics import *
from dataset import load_veremi
from config import Config, Colors

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel('ERROR')


class VeremiBase(ABC):
    def __init__(self, data_file: str, model_type: str, label: str, feature: str, activation: str = "softmax"):
        """ The Veremi Client Constructor
            :param model_type: Keras Model Type ('mlp' or 'lstm'
            :param label: Model label type ('binary', 'multiclass', 'atk_1', 'atk_2', 'atk_4', 'atk_8', 'atk_16')
            :param feature: Feature to evaluate ('feat1', 'feat2', 'feat3')
        """
        self.lb = None
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.model = None
        self.data_file = data_file
        self.label = label
        self.feature = feature
        self.model_type = model_type
        self.activation = activation

        self.load_veremi()
        self.create_model()

    def create_model(self):
        layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, output = \
            None, None, None, None, None, None, None, None, None, None
        if self.model_type == 'mlp':
            layer1 = keras.layers.Input(shape=(self.train_data.shape[1],))
            layer2 = keras.layers.Dense(256, activation="relu")(layer1)
            layer3 = keras.layers.Dense(256, activation="relu")(layer2)
            layer4 = keras.layers.Dropout(0.5)(layer3)
            output = keras.layers.Dense(self.train_labels.shape[1], activation=self.activation)(layer4)
        else:
            pass

        # ML Model
        name = self.label + "-" + self.model_type + "-" + self.feature + ".keras"
        self.model = keras.Model(inputs=layer1, outputs=output, name=name)
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=Config.learning_rate),
            metrics=[f1]
        )
        self.model.summary()

    def load_veremi(self):
        print("Loading dataset in " + self.__class__.__name__ + "...")
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.lb, self.dataset = load_veremi(
            self.data_file,
            feature=self.feature,
            label=self.label
        )
