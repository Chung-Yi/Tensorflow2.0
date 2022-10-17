from cProfile import label
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix

def main():
    tf.random.set_seed(42)
    X, y = create_make_circles_data()
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

    # regression_example()
    
    # create_model_1(X, y)

    # create_model_2(X, y)

    # model_3 = create_model_3(X, y)

    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # model_7 = create_model_7(X_train, y_train, X_test, y_test)
    # create_model_and_find_optimal_lr(X_train, y_train, X_test, y_test)

    model = create_model_with_optimal_lr(X_train, y_train, X_test, y_test, 0.02)
    y_preds = model.predict(X_test)
    confusion_matrix(y_test, tf.round(y_preds))

    create_confusion_metrix_plot(y_test, y_preds)

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title("train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1,2,2)
    plt.title("test")
    plot_decision_boundary(model, X_test, y_test)

    create_confusion_metrix_plot(y_test, y_preds)

    

def create_make_circles_data():
    X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

    return X, y

def create_model_1(X, y):
    # create the model 
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["accuracy"])
    
    # fit the model
    model_1.fit(X, y, epochs=200)

    # evaluate the model
    model_1.evaluate(X, y)

    return model_1


def create_model_2(X, y):
    # create the model 
    model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["accuracy"])
    
    # fit the model
    model_2.fit(X, y, epochs=200)

    # evaluate the model
    print(model_2.evaluate(X, y))

    return model_2

def create_model_3(X, y):
    # create the model 
    model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
    
    # fit the model
    model_3.fit(X, y, epochs=100)

    # evaluate the model
    print(model_3.evaluate(X, y))

    return model_3

def create_model_4(X, y):
    tf.random.set_seed(42)
    # create the model 
    model_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ])

    # compile the model
    model_4.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    metrics=["accuracy"])
    
    # fit the model
    model_4.fit(X, y, epochs=100)

    # evaluate the model
    print(model_4.evaluate(X, y))

    return model_4

def create_model_5(X, y):
    tf.random.set_seed(42)
    # create the model 
    model_5 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model_5.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    metrics=["accuracy"])
    
    # fit the model
    model_5.fit(X, y, epochs=100)

    # evaluate the model
    print(model_5.evaluate(X, y))

    return model_5

def create_model_6(X, y):
    tf.random.set_seed(42)
    # create the model 
    model_6 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile the model
    model_6.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    metrics=["accuracy"])
    
    # fit the model
    model_6.fit(X, y, epochs=100)

    # evaluate the model
    print(model_6.evaluate(X, y))

    return model_6

def create_model_7(X_train, y_train, X_test, y_test):
    # set random seed
    tf.random.set_seed(42)

    # 1. create the model 
    model_7 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 2. compile the model
    model_7.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=0.01),
                    metrics=["accuracy"]
    )

    # 3. fit the model
    history = model_7.fit(X_train, y_train, epochs=100)

    print(history.history)

    # 4. evaluate the model
    print(model_7.evaluate(X_test, y_test))
    pd.DataFrame(history.history).plot()

    return model_7

def create_model_and_find_optimal_lr(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])
    
    # create a learning rate callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))

    history = model.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])
    print(model.evaluate(X_test, y_test))
    pd.DataFrame(history.history).plot()

    check_lr_by_plot(history)

    return model

def create_model_with_optimal_lr(X_train, y_train, X_test, y_test, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                   optimizer=tf.keras.optimizers.Adam(lr),
                   metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=100)

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"model loss on test data: {loss}")
    print(f"model accuracy on test data: {accuracy * 100}%")

    return model


def plot_decision_boundary(model, X, y):
    
    # define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() , X[:, 1].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                        np.linspace(y_min, y_max, 100))

    # print(xx)
    # print("===============================")
    # print(yy)

    # create X value
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # print(x_in)

    # os._exit(0)

    # make prediction
    y_pred = model.predict(x_in)

    if len(y_pred[0]) > 1:
        print("multiple classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    
    else:
        print("binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def regression_example():
    tf.random.set_seed(42)

    X = tf.range(0, 1000, 5)
    y = tf.range(100, 1100, 5) # y = X + 100

    X_train = X[:150]
    X_test = X[150:]

    y_train = y[:150]
    y_test = y[150:]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["mae"])

    model.fit(X_train, y_train, epochs=100)

    y_pred = model.predict(X_test)

    # plt.figure(figure=(10, 7))
    plt.scatter(X_train, y_train, c='b', label="training data")
    plt.scatter(X_test, y_test, c='g', label="testing data")
    plt.scatter(X_test, y_pred, c='r', label="predictions")

def check_lr_by_plot(history):
    lrs = 1e-4 * (10 ** (tf.range(100) / 20))
    print(f"learning rate: {lrs}")

    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])

    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("learning rate vs loss")

def create_confusion_metrix_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, tf.round(y_pred))
    cm_norm = cm.astype("float") / cm.sum(axis=1)
    n_classes = cm.shape[0]

    # prettify it 
    fig, ax = plt.subplots(figsize=(10, 10))

    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # create classes
    classes = False

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # label the axes
    ax.set(title="confusion metrix", 
           xlabel="predicted label",
           ylabel="true label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    
    # set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)
    
    # set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)", 
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=15
        )

if __name__ == "__main__":
    main()