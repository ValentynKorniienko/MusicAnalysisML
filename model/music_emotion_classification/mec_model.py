from keras import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense
from keras.src.optimizers import RMSprop


def compile_model(model):
    """
        Compile the machine learning model with appropriate optimizer, loss function, and metrics.

        This function compiles the model passed to it, using the RMSprop optimizer and sparse categorical
        crossentropy as the loss function. It's designed for multi-class classification problems. The model's
        summary is also printed out to give an overview of the model's architecture.

        Args:
            model (keras.Model): The uncompiled Keras model to be compiled.
            X_train (numpy.ndarray): The training data, used to determine the input shape for the model.

        Returns:
            keras.Model: The compiled Keras model.
        """
    model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
        Train the compiled model with the provided training and validation data.

        Training includes early stopping and learning rate reduction on plateau to prevent overfitting
        and to optimize training. Early stopping halts training when validation loss stops improving, and
        learning rate reduction reduces the learning rate when validation loss stops decreasing.

        Args:
            model (keras.Model): The compiled Keras model to be trained.
            X_train (numpy.ndarray): Training feature data.
            y_train (numpy.ndarray): Training target labels.
            X_val (numpy.ndarray): Validation feature data.
            y_val (numpy.ndarray): Validation target labels.

        Returns:
            keras.callbacks.History: The history object containing training and validation loss and accuracy
                                     for each epoch.
        """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')

    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=32,
                     callbacks=[reduce_lr, early_stopping])


def create_cnn_lstm_model(X_train):
    """
        Create a CNN-LSTM hybrid model.

        This function constructs a neural network that combines convolutional neural network (CNN) layers with
        Long Short-Term Memory (LSTM) layers. This type of model is beneficial for sequence data that requires
        both spatial feature extraction (handled by CNN layers) and sequence learning (handled by LSTM layers),
        such as time-series data or in this case, audio features.

        Args:
            input_shape (tuple): The shape of the input data. Typically, this is a 2D shape for sequences,
                                 where dimensions are (time_steps, features).

        Returns:
            keras.Model: The constructed CNN-LSTM Keras model.

        The model architecture includes the following layers:
        - Conv1D layers for convolutional operations, extracting spatial features within the input sequences.
        - BatchNormalization layers for normalizing activations in the network, improving stability.
        - MaxPooling1D layers for downsampling the feature maps, reducing the dimensions.
        - Dropout layers for regularization to prevent overfitting.
        - LSTM layers for learning dependencies in sequence data.
        - Dense (fully connected) layers for classification.
        - The final output layer uses softmax activation for multi-class classification.
        """
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = Sequential()

    model.add(Conv1D(128, 4, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(128, 4, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Conv1D(256, 4, activation='relu', padding='same'))
    model.add(Conv1D(256, 4, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Conv1D(512, 4, activation='relu', padding='same'))
    model.add(Conv1D(512, 4, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    # RNN layer (LSTM in this example)
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(LSTM(128))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Fully connected layer for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    return model
