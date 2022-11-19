import os
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from os import listdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation
# train_dir = "data/pizza_steak/train"
# test_dir = "data/pizza_steak/test"

train_dir = "data/10_food_classes_all_data/train"
test_dir = "data/10_food_classes_all_data/test"

test_image = 'data/pizza_steak/test_images/pizza1.jpg'

checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32
img_size = 224

# class_names = ['pizza', 'steak']
class_names = np.array(sorted([item for item in listdir(train_dir)]))

def load_and_prep_image(filename):
    img = tf.io.read_file(filename)

    img = tf.image.decode_image(img)

    img = tf.image.resize(img, size=[img_size, img_size])

    img /= 255.

    return img

def view_random_image(target_dir, target_class):
    target_folder = os.path.join(target_dir, target_class)
    random_image = random.sample(os.listdir(target_folder), 1)

    print(random_image)

    img = mpimg.imread(target_folder + '/' + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f'image shape: {img.shape}')

    # return img

def generate_batch_data(class_names):
    class_mode = "binary"
    if len(class_names) > 2: 
        class_mode = "categorical"
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   batch_size=batch_size,
                                                   target_size=(img_size, img_size),
                                                   class_mode=class_mode, # class_mode will be categorical
                                                   seed=42)

    valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                                   batch_size=batch_size,
                                                   target_size=(img_size, img_size),
                                                   class_mode=class_mode,
                                                   seed=42)

    return train_data, valid_data

def generate_batch_with_augmentation(class_names):

    class_mode = "binary"
    if len(class_names) > 2: 
        class_mode = "categorical"

    def show_origin_and_augmented_images():
        random_number = random.randint(0, 32)
        print(f"show image number: {random_number}")

        plt.imshow(images[random_number])
        plt.title(f"origin image")
        plt.axis(False)

        plt.figure()
        plt.imshow(augmented_images[random_number])
        plt.title(f"augmented image")
        plt.axis(False)


    train_data_augmented_gen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=0.2,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.3,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = ImageDataGenerator(rescale=1/255)

    test_data = ImageDataGenerator(rescale=1/255)

    
    # create augmented data from training directory
    train_data_augmented = train_data_augmented_gen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )

    # create non-augmented data from directory
    train_data = train_data.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    # create non-augmented test data from directory
    test_data = test_data.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode
    )

    images, labels = train_data.next()
    augmented_images, augmented_labels = train_data_augmented.next()

    show_origin_and_augmented_images()


    return train_data_augmented, test_data


def plot_training_curve(history):
    pd.DataFrame(history.history).plot(figsize=(10, 7))


def create_conv_model(class_num):

    if class_num > 2:
        loss = "categorical_crossentropy"
        activation_output = "softmax"
    else:
        loss = "binary_crossentropy"
        activation_output = "sigmoid"

    tf.random.set_seed(42)

    # build a CNN model(same as the Tiny VGG)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10, 
                               kernel_size=3, 
                               activation="relu", 
                               input_shape=(img_size, img_size, 3)
                               ),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, padding="valid", name="max_pool_1"),

        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2, name="max_pool_2"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(class_num, activation=activation_output)

    ]) 

    # compile the model
    model.compile(loss=loss,
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])


    return model

def create_model_avoid_overfitting(class_num):

    if class_num > 2:
        loss = "categorical_crossentropy"
        activation_output = "softmax"
    else:
        loss = "binary_crossentropy"
        activation_output = "sigmoid"

    print(loss, activation_output)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, 3, activation='relu', 
                            input_shape=(img_size, img_size, 3)),
        # tf.keras.layers.MaxPool2D(pool_size=2),
        # tf.keras.layers.Conv2D(10, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation='relu'),
        # tf.keras.layers.Conv2D(10, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(class_num, activation=activation_output)
    ])

    # compile the model
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    return model


def train(model, train_data, valid_data):

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
    #                                                 save_weights_only=True,
    #                                                 verbose=1,
    #                                                 save_freq=2*batch_size)


    # fit the model
    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        # callbacks=[cp_callback]
                    )

    return history


def create_non_CNN_model(train_data, valid_data):
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    print(model.summary())

    history = model.fit(train_data, 
                        epochs=10,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))


    return history

def main():
    view_random_image(train_dir, target_class=random.choice(class_names))
    train_data, valid_data = generate_batch_data(class_names)

    # generate augmented image
    train_data_augmented, valid_data = generate_batch_with_augmentation(class_names)
    

    # create model
    # model = create_conv_model(len(class_names))

    model = create_model_avoid_overfitting(len(class_names))

    print(model.summary())

    history = train(model, train_data_augmented, valid_data)

    plot_training_curve(history)

    # history_non_cnn = create_non_CNN_model(train_data, valid_data)

    loss, acc = model.evaluate(valid_data)

    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # predict
    test_img = load_and_prep_image(test_image)
    pred = model.predict(tf.expand_dims(test_img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[int(tf.argmax(pred[0]))]
    else:
        pred_class = class_names[int(tf.round(pred))]



    

    #############################################################
    # pizza_img = "data/pizza_steak/test/pizza/11297.jpg"
    # img = mpimg.imread(pizza_img)
    # # img = cv2.resize(img, (img_size, img_size))
    # img = tf.image.resize(img, [img_size, img_size])
    # # plt.imshow(img)
    # # plt.show()
    # img =  img / 255
    # img = tf.expand_dims(img, axis=0)

    # pred = model.predict(img)

    # view random images
    # plt.figure()
    # plt.subplot(1,2,1)
    # view_random_image('data/pizza_steak/train', 'steak')
    # plt.subplot(1,2,2)
    # view_random_image('data/pizza_steak/train', 'pizza')

if __name__ == '__main__':
    main()