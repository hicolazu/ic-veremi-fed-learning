import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config import Config


def load_veremi(csv_file: str, feature: str, label: str, delimiter=','):
    # Import VeReMi Dataset
    data = pd.read_csv(csv_file, delimiter=delimiter)

    data_normal = data.loc[data['attackerType'] == 0]
    data_atk = data.loc[data['attackerType'] != 0]
    atk_size = int(data_atk.shape[0])
    data = pd.concat([data_normal.sample(atk_size), data_atk])
    data = shuffle(data)

    dataset = data
    target = data[data.columns[-1:]]
    data = data[data.columns[0:-1]]

    # normalize data
    data = (data - data.mean()) / data.std()

    # label binarize one-hot style
    lb = preprocessing.LabelBinarizer()
    lb.fit(target)
    if label == 'multiclass':
        target = lb.transform(target)
    else:
        target = lb.transform(target)
        target = MultiLabelBinarizer().fit_transform(target)

    # Create training and test data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        target,
        train_size=Config.data_train_size,
        test_size=Config.data_test_size,
        # random_state=42
    )

    return train_data, test_data, train_labels, test_labels, lb, dataset