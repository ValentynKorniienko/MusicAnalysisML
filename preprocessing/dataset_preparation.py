import pandas as pd


def preprocess_emotify_dataset(file_path):
    """
    Preprocess the Emotify dataset by reading a CSV file, trimming and dropping columns,
    transforming to average values, applying one-hot encoding, and adjusting track IDs.
    Args:
        file_path (str): The path to the CSV file of the Emotify dataset.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    df = _read_csv(file_path)
    df = _trim_and_drop_columns(df)
    df = _transform_to_average_values(df)
    df = _one_hot_encoding(df)
    df = _adjust_track_ids(df)
    return df


def save_preprocessed_dataset(preprocessed_dataset, path_to_save):
    """
    Save the preprocessed dataset to a CSV file.
    This function takes a preprocessed pandas DataFrame and saves it as a CSV file
    at the specified location. The index of the DataFrame is not included in the CSV file.
    Args:
        preprocessed_dataset (pandas.DataFrame): The preprocessed dataset to be saved.
        path_to_save (str): The file path where the CSV file will be saved.
    Returns:
        None: This function does not return anything. It saves the dataset as a CSV file.
    """
    preprocessed_dataset.to_csv(path_to_save, index=False)


def _read_csv(file_path):
    """
    Read a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)


def _trim_and_drop_columns(df, columns_to_remove=None):
    """
    Trim whitespace from column names and drop specified columns.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
        columns_to_remove (list, optional): List of column names to remove.
            Defaults to a preset list of columns if None.
    Returns:
        pandas.DataFrame: The DataFrame with trimmed column names and specified columns removed.
    """
    if columns_to_remove is None:
        columns_to_remove = ['mood', 'liked', 'disliked', 'age', 'gender', 'mother tongue']

    # Trim whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop specified columns
    df = df.drop(columns=columns_to_remove, axis=1)

    return df


def _transform_to_average_values(df):
    """
    Transform the DataFrame to calculate the average of emotionality features
    grouped by 'track id' and 'genre'.
    Args:
        df (pandas.DataFrame): The DataFrame to transform.
    Returns:
        pandas.DataFrame: The transformed DataFrame with average values of emotionality features.
    """
    # Define the columns that are not part of emotionality features
    emotionality_features = df.columns.difference(['track id', 'genre'])

    # Group by 'track id' and 'genre', then calculate the mean of the emotionality features
    return df.groupby(['track id', 'genre'], as_index=False)[emotionality_features].mean()


def _one_hot_encoding(df, emotionality_features=None):
    """
    Apply one-hot encoding to emotionality features based on the maximum value in each row.
    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        emotionality_features (list, optional): List of emotionality feature columns.
            Defaults to a preset list of features if None.
    Returns:
        pandas.DataFrame: The DataFrame with one-hot encoded features.
    """
    if emotionality_features is None:
        emotionality_features = ['amazement', 'calmness', 'joyful_activation', 'nostalgia', 'power', 'sadness',
                                 'solemnity', 'tenderness', 'tension']

    def _one_hot_encode_max(row):
        """
        One-hot encode a row based on the maximum value among specified features.

        Args:
            row (pandas.Series): A row of the DataFrame.

        Returns:
            list: A list representing the one-hot encoded features.
        """
        max_value = row[emotionality_features].max()
        return [1 if value == max_value else 0 for value in row[emotionality_features]]

    # Apply the one-hot encoding function to each row
    grouped_data_encoded = df.copy()
    grouped_data_encoded[emotionality_features] = grouped_data_encoded.apply(_one_hot_encode_max, axis=1,
                                                                             result_type='expand')
    return grouped_data_encoded


def _adjust_track_ids(df, track_id_column='track id'):
    """
    Adjust 'track id' in the DataFrame such that each set of hundred starts from 1.
    Values like 100, 200, etc., are adjusted to 100 instead of 0.
    Args:
        df (pandas.DataFrame): The DataFrame containing the 'track id'.
        track_id_column (str): The name of the column containing the 'track id'.
    Returns:
        pandas.DataFrame: The DataFrame with adjusted 'track id'.
    """
    df[track_id_column] = df[track_id_column] % 100
    df[track_id_column] = df[track_id_column].replace(0, 100)
    return df