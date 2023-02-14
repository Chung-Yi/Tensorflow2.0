
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
import cv2
from extras.helper_function import walk_through_dir, create_tensorboard_callback, plot_loss_curves
from tensorflow.keras.layers.experimental import preprocessing

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

train_1_percent_dir = "10_food_classes_1_percent/train"
test_1_percent_dir = "10_food_classes_1_percent/test"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def augmentation_params():
    return tf.keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.Rescaling(1/255.)
    ], name="data_augmentation")

def show_image(image_path, target_class):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(f"original random image from class {target_class}")
    plt.axis(False)


def base_model(train_data, test_data, trainable=False, include_top=False):

    # 1. create a model with tf.keras.applications
    # you don't need to normalize if using EfficientNet
    base_model = tf.keras.applications.EfficientNetB0(include_top=include_top)

    # 2. freeze the base model
    base_model.trainable = trainable

    # 3. create inputs into our model
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")


    # 4. if using ResNet50V2 you will need to normalize input data
    # base_model = tf.keras.applications.ResNet50V2(include_top=include_top)
    # x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255.)(inputs)

    # 5. pass the inputs to the base_model
    x = base_model(inputs)
    print(f"shape after passing inputs through base model:{x.shape}")

    # 6. Average pool the output of the base model(aggregate all the most important information, reduce number of computations)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"shape after GlobalAveragePooling2D: {x.shape}")

    # 7. create the output activation layer
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

    # 8. combine the inputs with outputs into a model
    model_0 = tf.keras.Model(inputs, outputs)

    # 9. compile for the model
    model_0.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    # 10. Fit the model and save the history
    history = model_0.fit(train_data, 
                epochs=5,
                steps_per_epoch=len(train_data),
                validation_data=test_data,
                validation_steps=int(0.25 * len(test_data)),
                callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_feature_extraction")])
    
    return history, model_0 


def main():
    # walk_through_dir("10_food_classes_1_percent")

    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                batch_size=BATCH_SIZE
                                                                                )
    test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                batch_size=BATCH_SIZE,
                                                                                )
    # model_1: Use feature extraction transfer learning on 1% of the training data with data augmentation
    train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_1_percent_dir, 
                                                                               image_size=IMG_SIZE,
                                                                               label_mode="categorical",
                                                                               batch_size=BATCH_SIZE)
    
    test_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_1_percent_dir, 
                                                                               image_size=IMG_SIZE,
                                                                               label_mode="categorical",
                                                                               batch_size=BATCH_SIZE)
    

    
    
    # see an example of batch of data
    # for images, labels in train_data_10_percent.take(1):
    #     print(images, labels)

    # create base model
    history, model_base = base_model(train_data_10_percent, test_data_10_percent)

    # evaluate model
    model_base.evaluate(test_data_10_percent)

    # layer name
    for idx, layer in enumerate(model_base.layers):
        print(layer.name)
    
    # model summary
    print(model_base.summary())

    plot_loss_curves(history)
    
    ###########################################
    # show random image
    target_class = random.choice(train_data_1_percent.class_names)
    target_dir = "10_food_classes_1_percent/train/" + target_class
    random_image = random.choice(os.listdir(target_dir)) 
    random_image_path = target_dir + "/" +  random_image
    show_image(random_image_path, target_class)

    # show augmentation image
    img = cv2.imread(random_image_path)
    augmentation_generator = augmentation_params()
    augmentation_img = augmentation_generator(tf.expand_dims(img, axis=0))
    plt.figure()
    plt.title("augmentation image")
    plt.axis(False)
    plt.imshow(tf.squeeze(augmentation_img))



if __name__ == "__main__":
    main()