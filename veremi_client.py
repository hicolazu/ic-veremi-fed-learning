import logging

import flwr as fl
import tensorflow as tf
from config import Config
from tensorflow import keras
from veremi_base import VeremiBase
from flwr.common.logger import FLOWER_LOGGER

tf.get_logger().setLevel('ERROR')

FLOWER_LOGGER.setLevel(logging.WARNING)


# VeReMi Client Class
class VeremiClient(VeremiBase, fl.client.NumPyClient):
    def __init__(self, data_file: str, model_type: str, label: str, feature: str, activation: str = "softmax"):
        VeremiBase.__init__(self, data_file, model_type, label, feature, activation)
        self.history = None

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=Config.early_stop_monitor,
            patience=Config.early_stop_patience,
            min_delta=Config.early_stop_min_delta,
            restore_best_weights=Config.early_stop_restore_best_weights
        )
        self.model.set_weights(parameters)
        self.history = self.model.fit(
            self.train_data,
            self.train_labels,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=[early_stopping],
            validation_data=(self.test_data, self.test_labels),
            verbose=1,
        )
        result = {
            "f1_score:": float(self.history.history['f1'][-1]),
            "f1_val": float(self.history.history['val_f1'][-1]),
        }
        return self.model.get_weights(), len(self.train_data), result

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    in_filename = int(input("Escolha o arquivo de treino (0 - VeReMi_1.csv, 1 - VeReMi_2.csv, 2 - VeReMi_3.csv): "))

    filename = ''

    if in_filename == 0:
        filename = 'VeReMi_1.csv'
    elif in_filename == 1:
        filename = 'VeReMi_2.csv'
    elif in_filename == 2:
        filename = 'VeReMi_3.csv'

    # Start Client
    fl.client.start_numpy_client(
        server_address="[::]:8080",
        client=VeremiClient(Config.csv + filename, Config.model_type, Config.label, Config.feature, Config.output_activation)
    )