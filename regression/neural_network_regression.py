
from distutils.archive_util import make_archive
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

tf.random.set_seed(42)

# create feature
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
X = tf.cast(tf.constant(X), dtype=tf.float32)

# create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
y = tf.cast(tf.constant(y), dtype=tf.float32)

def main():
    # basic_regression_test()

    # model = model_train(X, y)
    # prediction = model.predict([17.0])
    # print(prediction)

    X_train, y_train, X_test, y_test = split_data()
    model = model_train(X_train, y_train)
    plot_model(model=model, show_shapes=True)

    y_pred = model.predict(X_test)

    plot_predictions(X_train, y_train, X_test, y_test, y_pred)

    # evaluate the model on the test data
    print(model.evaluate(X_test, y_test))

    # calculate mae by tf.metrics.mean_absolute_error
    mae_err = mae(y_test, y_pred)
    print(mae_err)

    mse_err = mse(y_test, y_pred)
    print(mse_err)

    ############### Running experiments to improve our model ###############
    # 1. get more data
    # 2. make your model larger
    # 3. train for longer
    
    ########################### model_1 ###########################
    # Set the random_seed
    tf.random.set_seed(42)

    # 1. create the model
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # 2. compile the model
    model_1.compile(loss=tf.keras.losses.mae, 
             optimizer=tf.keras.optimizers.SGD(),
             metrics=["mae"])
    
    # 3. fit the model
    model_1.fit(X_train, y_train, epochs=100)

    # make and plot prediction 
    y_pred_1 = model_1.predict(X_test)
    plot_predictions(X_train, y_train, X_test, y_test, y_pred_1)

    # calculate model_1 evaluation metrics
    mae_1 = mae(y_test, tf.squeeze(y_pred_1))
    print(mae_1)

    mse_1 = mse(y_test, tf.squeeze(y_pred_1))
    print(mse_1)

    ########################### model_2 ###########################
    # Set the random_seed
    tf.random.set_seed(42)

    # 1. create the model
    model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # 2. compile the model
    model_2.compile(loss=tf.keras.losses.mae, 
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["mse"])
    
    # 3. fit the model
    model_2.fit(X_train, y_train, epochs=100)

    # make and plot prediction 
    y_pred_2 = model_2.predict(X_test)
    plot_predictions(X_train, y_train, X_test, y_test, y_pred_2)

    # calculate model_2 evaluation metrics
    mae_2 = mae(y_test, tf.squeeze(y_pred_2))
    print(mae_2)

    mse_2 = mse(y_test, tf.squeeze(y_pred_2))
    print(mse_2)

    ########################### model_3 ###########################
    # Set the random_seed
    tf.random.set_seed(42)

    # 1. create the model
    model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # 2. compile the model
    model_3.compile(loss=tf.keras.losses.mae, 
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["mae"])
    
    # 3. fit the model
    model_3.fit(X_train, y_train, epochs=500)

    # make and plot prediction 
    y_pred_3 = model_3.predict(X_test)
    plot_predictions(X_train, y_train, X_test, y_test, y_pred_3)

    # calculate model_3 evaluation metrics
    mae_3 = mae(y_test, tf.squeeze(y_pred_3))
    print(mae_3)

    mse_3 = mse(y_test, tf.squeeze(y_pred_3))
    print(mse_3)

    # compare the results
    all_results = generate_dataframe_of_result(mae_1, mse_1, mae_2, mse_2, mae_3, mse_3)
    print(all_results)

    ############### save the best model ###############
    # 1. save model format
    model_2.save("best_model_SaveModel_format")

    # save hd5 format
    model_2.save("best_model_HD5_format.h5")

    ############ load in SaveModel format ############
    loaded_SaveModel_format = tf.keras.models.load_model("best_model_SaveModel_format")
    loaded_SaveModel_format.summary()
    model_2.summary()

    ############ compare model_2 predictions with SaveModel format model predictions ############
    model_2_preds = model_2.predict(X_test)
    loaded_SaveModel_format_preds = loaded_SaveModel_format.predict(X_test)

    print(model_2_preds)
    print(loaded_SaveModel_format_preds)
    print(model_2_preds == loaded_SaveModel_format_preds)

    ############ load in the hd5 format ############
    loaded_h5_model = tf.keras.models.load_model("best_model_HD5_format.h5")
    loaded_h5_model.summary()

    ############ compare model_2 predictions with h5 format model predictions ############
    loaded_h5_model_pred = loaded_h5_model.predict(X_test)
    print(model_2_preds == loaded_h5_model_pred)
    

    ############ medicine cost data ############
    insurance = read_medicine_cost_data("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    insurance_one_hot = pd.get_dummies(insurance)

    # create X, y value
    X = insurance_one_hot.drop("charges", axis=1)
    y = insurance_one_hot["charges"]

    # create training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ################################## build a neural network(sort like model_2) ##################################
    tf.random.set_seed(42)

    # 1. create a model
    insurance_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(1)
    ])

    # 2. compile the model
    insurance_model.compile(loss=tf.keras.losses.mae, 
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=["mae"])
    
    # 3. fit the model
    history = insurance_model.fit(X_train, y_train, epochs=100)

    # evaluate the model on the test data
    insurance_model.evaluate(X_test, y_test)

    # plot histroty(loss curve)
    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")


    ###################################################################
    X_train_normal, X_test_normal, y_train, y_test = preprocess_data(insurance)
    insurance_model.fit(X_train_normal, y_train, epochs=100)

    # evaluate the model on the test data
    insurance_model.evaluate(X_test_normal, y_test)




def read_medicine_cost_data(datasource):
    insurance = pd.read_csv(datasource)
    return insurance

def preprocess_data(insurance):
    # create a column transformer
    ct = make_column_transformer(
        (MinMaxScaler(), ["age", "bmi", "children"]),
        (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )

    # create X, y 
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    # create training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit the column transformer to our training data
    ct.fit(X_train)

    # trandform training and testing data with normalize
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)

    return X_train_normal, X_test_normal, y_train, y_test


def generate_dataframe_of_result(mae_1, mse_1, mae_2, mse_2, mae_3, mse_3):
    model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
                     ["model_2", mae_2.numpy(), mse_2.numpy()],
                     ["model_3", mae_3.numpy(), mse_3.numpy()]
                    ]
    
    all_results = pd.DataFrame(model_results, columns=['model', 'mae', 'mse'])

    return all_results

def model_train(X, y):

    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(50, input_shape=[1], name="input_shape"),
        tf.keras.layers.Dense(1, name="ouput_layer")
    ], name='sequential')

    model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])

    model.fit(X, y, epochs=100)

    model.summary()

    return model


def basic_regression_test():
    plt.scatter(X, y)
    plt.show()

def split_data():
    X_large = tf.range(-100, 100, 4)
    y_large = X_large + 10

    X_train = X_large[:40]
    y_train = y_large[:40]

    X_test = X_large[40:]
    y_test = y_large[40:]

    # plt.figure(figsize=(10, 7))
    # plt.scatter(X_train, y_train, c='b', label='training data')
    # plt.scatter(X_test, y_test, c='r', label='testing data')
    # plt.legend()
    # plt.show()

    return X_train, y_train, X_test, y_test

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", label="training data")
    plt.scatter(test_data, test_labels, c="g", label="testing data")
    plt.scatter(test_data, predictions, c="r", label="predictions")
    plt.legend()

def mae(y_test, y_pred):
    return tf.metrics.mean_absolute_error(y_test, tf.squeeze(y_pred))

def mse(y_test, y_pred):
    return tf.metrics.mean_squared_error(y_test, tf.squeeze(y_pred))

if __name__ == "__main__":
    main()