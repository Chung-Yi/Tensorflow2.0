import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import itertools
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model

class_names = {
    "names": [
            "T - shirt / top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
    ],
}

num_classes = 10
epoch = 30
figure_size = (15, 15)
img_size = 28


def load_data():
    # the data has already been sorted into training and testing set
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    return (train_data, train_labels), (test_data, test_labels)

def plot_multiple_image(train_data, train_labels):
    plt.figure(figsize=(7, 7))
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        rand_index = random.choice(range(len(train_data)))
        plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
        plt.title(class_names["names"][train_labels[rand_index]])
        # plt.axis(False)

def create_model(train_data, train_labels, test_data, test_labels):
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    
    # compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy']
    )

    # fit the model
    non_norm_history = model.fit(train_data, train_labels, epochs=epoch, validation_data=(test_data, test_labels))

    return non_norm_history

def plot_curve(history):
    pd.DataFrame(history.history).plot()

def create_model_and_find_optimal_lr(train_data, train_labels, test_data, test_labels):
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

    # create the learning rate callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))

    # fit the model
    find_lr_history = model.fit(train_data, 
                                train_labels,
                                epochs=epoch,
                                validation_data=(test_data, test_labels), 
                                callbacks=[lr_scheduler])
    return find_lr_history

def create_model_with_optimal_lr(train_data, train_labels, test_data, test_labels, lr_optimal):
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(lr_optimal),
    metrics=['accuracy'])

    model.fit(train_data, 
             train_labels,
             epochs=epoch,
             validation_data=(test_data, test_labels)
            )

    return model


def check_lr_by_plot(history):
    lrs = 1e-3 * 10 ** (tf.range(epoch) / 20)
    plt.semilogx(lrs, history.history['loss'])
    plt.xlabel('Learning rate')
    plt.ylabel('loss')
    # plt.xticks(lrs)
    plt.title('Finding the ideal learning rate')

def create_confusion_metrix_plot(y_test, y_pred, classes, figure_size, text_size=20):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)
    n_classes = len(classes)

    # prettify 
    fig, ax = plt.subplots(figsize=figure_size)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    ax.set(title="confusion matrix",
           xlabel="prediction label",
           ylabel="true label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels
           )
    
    # set x-axis labels to the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    # adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%', 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                #  size=15
                )

def plot_random_image(model, images, true_labels, classes):
    i = random.randint(0, len(images))
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, img_size, img_size))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    plt.imshow(target_image)

    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(f"pred: {pred_label} {100*tf.reduce_max(pred_probs):.2f}%, True: {true_label}", color=color)

def get_the_training_pattern(model):
    weights, biases = model.layers[1].get_weights()
    print(biases.shape)
    print(weights.shape)



def main():
    (train_data, train_labels), (test_data, test_labels) = load_data()

    index_of_choice = 17
    plt.imshow(train_data[index_of_choice], cmap=plt.cm.binary)
    plt.title(class_names["names"][train_labels[index_of_choice]])

    plot_multiple_image(train_data, train_labels)

    # normalize
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    train_labels = tf.one_hot(train_labels, depth=num_classes)
    test_labels = tf.one_hot(test_labels, depth=num_classes)

    # history = create_model(train_data, train_labels, test_data, test_labels)
    # plot_curve(history)

    # find_lr_history = create_model_and_find_optimal_lr(train_data, train_labels, test_data, test_labels)
    # check_lr_by_plot(find_lr_history)

    model = create_model_with_optimal_lr(train_data, train_labels, test_data, test_labels, lr_optimal=0.002)

    y_pred = model.predict(test_data)
    y_pred = y_pred.argmax(axis=1)

    create_confusion_metrix_plot(test_labels, y_pred, class_names['names'], figure_size)

    plot_random_image(model, test_data, test_labels, class_names['names'])

    get_the_training_pattern(model)

    plot_model(model, show_shapes=True)

if __name__ == "__main__":
    main()