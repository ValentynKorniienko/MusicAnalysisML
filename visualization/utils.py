from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_history(history):
    """
        Plot the training and validation accuracy and loss over epochs.

        This function plots the history of training, including accuracy and loss for both training and
        validation sets over each epoch. This is useful to visualize the model's learning process and
        to identify overfitting or underfitting.

        Args:
            history (keras.callbacks.History): A history object returned by the fit method of models.

        Returns:
            None: This function does not return anything. It generates and displays plots.
        """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def plot_confusion_matrix(model, label_encoder, X_val, y_val):
    """
       Plot a confusion matrix for the validation set.

       This function uses the model to predict labels on the validation set and then plots the confusion matrix
       to visualize how well the model is predicting each class. Additionally, it prints a classification report
       for more detailed performance analysis.

       Args:
           model (keras.Model): The trained model.
           label_encoder (LabelEncoder): The LabelEncoder used to encode the labels.
           X_val (numpy.ndarray): Validation feature data.
           y_val (numpy.ndarray): Validation target labels.

       Returns:
           numpy.ndarray: The confusion matrix.
       """
    y_pred = model.predict(X_val)
    y_pred_classes = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    __print_classification_report(y_val, y_pred_classes, label_encoder)
    return cm


def __print_classification_report(y_val, y_pred_classes, label_encoder):
    """
        Print a classification report.

        This private function prints a classification report for the validation set, including precision,
        recall, and f1-score for each class.

        Args:
            y_val (numpy.ndarray): Validation target labels.
            y_pred_classes (numpy.ndarray): Predicted labels for the validation set.
            label_encoder (LabelEncoder): The LabelEncoder used to encode the labels.

        Returns:
            None: This function does not return anything. It prints the classification report.
        """
    report = classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_)
    print(report)


def plot_accuracy_by_emotion(cm, label_encoder, xlabel, ylabel, title):
    """
        Plot accuracy for each emotion (class) based on the confusion matrix.

        This function calculates and prints the accuracy for each class and then plots these accuracies.
        It helps to identify which classes the model is performing well on and which ones are challenging.

        Args:
            cm (numpy.ndarray): The confusion matrix.
            label_encoder (LabelEncoder): The LabelEncoder used to encode the labels.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.

        Returns:
            None: This function does not return anything. It prints accuracies and displays a plot.
        """
    diagonal = cm.diagonal()

    total_samples_per_genre = cm.sum(axis=1)

    accuracies = diagonal / total_samples_per_genre

    for label, acc in zip(label_encoder.classes_, accuracies):
        print(f"Accuracy for {label}: {acc * 100:.2f}%")

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_encoder.classes_, y=accuracies)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
