import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def normalize(dataset):
    """
        Normalize the features in the dataset and encode the labels.

        This function separates the features and the label ('emotion_label'), normalizes the features using Min-Max scaling,
        and encodes the labels using label encoding. It scales feature values to be between 0 and 1, which is often beneficial
        for machine learning algorithms.

        Args:
            dataset (pandas.DataFrame): The dataset containing features and an 'emotion_label' column.

        Returns:
            tuple:
                - X (pandas.DataFrame): The normalized features.
                - y (pandas.DataFrame): The encoded labels.
                - label_encoder (LabelEncoder): The fitted LabelEncoder instance.
        """
    y = dataset['emotion_label']
    X = dataset.loc[:, dataset.columns != 'emotion_label']

    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    label_encoder = LabelEncoder()

    X = pd.DataFrame(np_scaled, columns=cols)
    y = pd.DataFrame(label_encoder.fit_transform(y))
    return X, y, label_encoder


def apply_smote(X, y):
    """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to the dataset.

        This function oversamples the minority class in the dataset using SMOTE, which helps to handle class imbalance
        by creating synthetic samples.

        Args:
            X (pandas.DataFrame or numpy.ndarray): The feature set.
            y (pandas.DataFrame or numpy.ndarray): The target labels.

        Returns:
            tuple: The resampled feature set and target labels.
        """
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    return X, y


def train_test_val_split(X, y, test_size, val_size):
    """
       Split the dataset into training, testing, and validation sets.

       The dataset is first split into training and a temporary test set. The temporary test set is then split
       into actual testing and validation sets.

       Args:
           X (pandas.DataFrame or numpy.ndarray): The feature set.
           y (pandas.DataFrame or numpy.ndarray): The target labels.
           test_size (float): The proportion of the dataset to include in the test split.
           val_size (float): The proportion of the dataset to include in the validation split.

       Returns:
           tuple: The training, testing, and validation sets (X_train, X_test, X_val, y_train, y_test, y_val).
       """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    return X_train, X_test, X_val, y_train, y_test, y_val


def reshape(X_train, X_test, X_val):
    """
        Reshape the feature sets for model input.

        This function reshapes the training, testing, and validation feature sets to include an additional dimension,
        which might be necessary for certain types of models like Convolutional Neural Networks.

        Args:
            X_train (pandas.DataFrame or numpy.ndarray): Training feature set.
            X_test (pandas.DataFrame or numpy.ndarray): Testing feature set.
            X_val (pandas.DataFrame or numpy.ndarray): Validation feature set.

        Returns:
            tuple: The reshaped feature sets.
        """
    X_train = X_train.to_numpy()[..., np.newaxis]
    X_test = X_test.to_numpy()[..., np.newaxis]
    X_val = X_val.to_numpy()[..., np.newaxis]
    return X_train, X_test, X_val
