def merge_datasets(preprocessed_dataset, feature_dataset):
    """
        Merge preprocessed dataset with feature dataset based on 'track id' and 'genre',
        transform to single label and drop unused rows.

        This function merges two datasets: one containing preprocessed data and the other containing extracted features.
        The merge is performed on 'track id' and 'genre' columns. It then transforms the merged dataset to have a single
        emotion label per track and drops unused rows.

        Args:
            preprocessed_dataset (pandas.DataFrame): A DataFrame containing the preprocessed data.
            feature_dataset (pandas.DataFrame): A DataFrame containing extracted features.

        Returns:
            pandas.DataFrame: The merged DataFrame with a single emotion label per track and unused rows dropped.
        """
    full_dataset = preprocessed_dataset.merge(feature_dataset, on=['track id', 'genre'], how='inner')
    full_dataset = __transform_to_single_label(full_dataset)
    return __drop_unused_rows(full_dataset)


def __transform_to_single_label(full_dataset):
    """
        Transform the dataset to have a single emotion label for each row.

        This function applies the '__determine_emotion_label' function to each row of the dataset to extract
        the dominant emotion based on the emotionality features.

        Args:
            full_dataset (pandas.DataFrame): The merged dataset.

        Returns:
            pandas.DataFrame: The dataset with an additional 'emotion_label' column.
        """
    full_dataset['emotion_label'] = full_dataset.apply(__determine_emotion_label, axis=1)
    return full_dataset


def __determine_emotion_label(row):
    """
        Determine the dominant emotion label for a given row.

        This function iterates over a predefined list of emotionality features.
        It returns the first emotion with a value of 1 in the row, indicating the dominant emotion.

        Args:
            row (pandas.Series): A row from the DataFrame.

        Returns:
            str: The dominant emotion label for the row. Returns None if no emotion is marked.
        """
    emotionality_features = ['amazement', 'calmness', 'joyful_activation', 'nostalgia', 'power', 'sadness', 'solemnity',
                             'tenderness', 'tension']

    for emotion in emotionality_features:
        if row[emotion] == 1:
            return emotion
    return None  # or a default label if no emotion is marked


def __drop_unused_rows(full_dataset):
    """
        Drop unused columns from the dataset.

        This function removes specific columns that are no longer needed after merging and transforming the dataset.
        These columns include 'track id', 'genre', various emotionality features, and 'segment'.

        Args:
            full_dataset (pandas.DataFrame): The dataset from which to drop columns.

        Returns:
            pandas.DataFrame: The dataset with specified columns removed.
        """
    full_dataset.drop(
        columns=['track id', 'genre', 'amazement', 'calmness', 'joyful_activation', 'nostalgia', 'power', 'sadness',
                 'solemnity', 'tenderness', 'tension', 'segment'])
    return full_dataset
