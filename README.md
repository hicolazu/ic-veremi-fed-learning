## Misbehavior Detection in Vehicular Networks with Federated Learning

### Overview

This Jupyter notebook implements a federated learning approach for detecting misbehavior in vehicular networks. The project aims to leverage distributed machine learning techniques to improve the detection of malicious activities within a vehicular environment while preserving data privacy.

### Content

1. **Imports**: Includes necessary Python libraries for data handling, machine learning, and visualization.
2. **Data Preprocessing**: Functions to load, clean, and prepare the data for training and testing, including feature selection and normalization.
3. **Model Training**: Code for training machine learning models using a federated learning framework.
4. **Evaluation**: Methods for evaluating the performance of the models using metrics like precision, recall, and F1 score.

### Components

#### Client and Server

- **VeremiClient**: Represents a client-side component in a federated learning setup. It processes local data, trains a machine learning model, and sends updates to the central server without sharing the raw data, thus preserving data privacy.
- **VeremiServer**: The central coordinating server in federated learning. It aggregates updates from multiple clients, updates the global model, and redistributes it back to the clients for further training.

#### Keras MLP Model
The Keras MLP (Multilayer Perceptron) model in this notebook is a type of feedforward artificial neural network. It consists of multiple layers of neurons, where each neuron in a layer is connected to every neuron in the subsequent layer. This architecture allows the model to learn complex patterns and relationships in the data.

**Layers and Properties**
**Input Layer**:

The input layer size matches the number of features in the dataset. For example, if the dataset has 10 features, the input layer will have 10 neurons.
**Hidden Layers**:

These layers are between the input and output layers and consist of neurons with activation functions. Common choices include ReLU (Rectified Linear Unit), which helps in introducing non-linearity to the model. The number of hidden layers and neurons per layer can vary based on the complexity of the problem and the size of the dataset.
Example configuration:
First hidden layer: 128 neurons, ReLU activation.
Second hidden layer: 64 neurons, ReLU activation.

**Output Layer**:

This layer's size depends on the classification task. For binary classification, a single neuron with a sigmoid activation function is typical, providing output values between 0 and 1. For multiclass classification, the output layer can have multiple neurons, with a softmax activation function to output a probability distribution over the classes.
Properties:

- Loss Function: Depending on the task, different loss functions are used. For binary classification, binary_crossentropy is common. For multiclass tasks, categorical_crossentropy is often used.
- Optimizer: Algorithms like Adam or SGD (Stochastic Gradient Descent) optimize the model's weights during training.
- Metrics: Accuracy, precision, recall, and F1 score are common metrics to evaluate model performance.

#### Federated Learning with Flower (flwr)
In the context of federated learning, the Flower library facilitates the communication between clients (such as veremiClient) and a central server (veremiServer). The main steps in this process include:

- Model Training on Clients:

Each client trains a local model on its private data. This is done independently, without sharing the data with other clients or the central server.
- Sending Weights to the Server:

After local training, each client sends the updated model weights (parameters) to the federated server. In Flower, this is typically done using the Client class, which includes a method like get_weights that extracts the weights from the Keras model and sends them to the server.
- Server Aggregation:

The server aggregates the weights received from multiple clients. This aggregation could be a simple averaging of weights or a more complex method considering client contributions.
- Model Update and Redistribution:

The aggregated weights are used to update the global model, which is then redistributed back to the clients. This updated model incorporates the learnings from all clients and is used as the starting point for the next round of training.

#### Config

- The configuration (`Config`) includes parameters that control data processing, model architecture, training settings, and federated learning-specific settings. These parameters are crucial for customizing the workflow to suit the specific requirements of the problem.

#### Metrics

- **Accuracy, Precision, Recall, F1 Score**: Metrics used to evaluate the model's performance, providing insights into its accuracy, precision, and ability to detect positive instances.
- **Confusion Matrix**: A detailed matrix showing the counts of true positive, true negative, false positive, and false negative predictions, useful for understanding the model's classification behavior.

### Dataset

- The dataset includes features like RSSI (Received Signal Strength Indicator), AoA (Angle of Arrival), distance, and conformity measures, with a target variable `attackerType` for classification. The data preprocessing steps involve normalization and label binarization, essential for preparing the data for effective model training.

### CSV
- To generate de csv file, run scripts/download_files.py to download all the simulations from https://github.com/VeReMi-dataset/VeReMi/releases
- Then, run scripts/extract_v2.py to extract the log files and create csv
- At the end, run scripts/split.py to divide the dataset into 3 cvs, 1 for the server and 2 for the clients trainning

### Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flower (flwr) - Federated Learning library

### Installation

To run the notebook, ensure you have Python installed along with the required libraries. You can install the dependencies using pip:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn flwr
```

### Usage

1. Clone the repository or download the notebook file.
2. Open the notebook using Jupyter Notebook or any compatible environment.
3. Run the cells sequentially to preprocess the data, train the models, and evaluate their performance.

### Acknowledgements

This project utilizes federated learning to enhance privacy and security in vehicular networks. We acknowledge the open-source libraries and frameworks used in this project.