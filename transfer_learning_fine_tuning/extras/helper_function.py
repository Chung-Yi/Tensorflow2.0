import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import zipfile
import os
import datetime
from sklearn.metrics import confusion_matrix

def load_and_prep_image(filename, img_shape=224, scale=True):
    # read image
    img = tf.io.read_file(filename)

    # decode image to tensor
    img = tf.image.decode_jpeg(img)

    # resize the image
    img = tf.image.resize(img, [img_shape, img_shape])

    if scale:
        return img / 255.

    else:
        return img


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predict label", 
           ylabel="True label", 
           xticks=np.arange(n_classes), 
           yticks=np.arange(n_classes),
           xticklabels=labels, # axes will labeled with class names(if they exits) or ints
           yticklabels=labels,
           )

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                       horizontalalignment="center",
                       color="white" if cm[i, j] > threshold else "black",
                       size=text_size)

        else:
            plt.text(j, i, f"{cm[i, j]}", 
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)

    # save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

def unzip_data(filename):
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dir(dir_path):
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    print(f"Saving TensorBoard log files to: {log_dir}")

    return tensorboard_callback

def create_checkpoint_callback(checkpoint_file):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, 
                                                            save_weights_only=True,
                                                            save_best_only=False,
                                                            save_freq="epoch",
                                                            verbose=1)
    return checkpoint_callback

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()


def squeeze_dimension():
    """
    The tf.keras.layers.GlobalAveragePooling2D() layer transforms a 4D tensor into a 2D tensor by averaging the values across the inner-axes.

    The previous sentence is a bit of a mouthful, so let's see an example. 
    """
    # Define the input shape
    input_shape = (1, 4, 4, 3)

    # Create a random tensor
    tf.random.set_seed(42)
    input_tensor = tf.random.normal(input_shape)
    print(f"Random input tensor: \n {input_shape}")

    # pass the random tensor through a global average pooling 2D layer
    global_average_pooled_tensor = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    print(f"2D global average pooled random tensor:\n {global_average_pooled_tensor}")

    print(f"shape of input_tensor: {input_tensor.shape}")
    print(f"shape of global_average_pooled_tensor: {global_average_pooled_tensor.shape}")

    # use tf.reduce can also get the same result
    result = tf.reduce_mean(input_tensor, axis=[1,2])
    print(f"tensorflow reduce_mean: {result}")


if __name__ == "__main__":
    squeeze_dimension()
