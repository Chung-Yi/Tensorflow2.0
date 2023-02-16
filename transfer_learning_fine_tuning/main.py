
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
import cv2
from extras.helper_function import walk_through_dir, create_tensorboard_callback, plot_loss_curves, create_checkpoint_callback
from tensorflow.keras.layers.experimental import preprocessing


train_1_percent_dir = "10_food_classes_1_percent/train"
test_1_percent_dir = "10_food_classes_1_percent/test"

train_10_percent_dir = "10_food_classes_10_percent/train"
test_10_percent_dir = "10_food_classes_10_percent/test"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def augmentation_params():
    return tf.keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        # preprocessing.Rescaling(1/255.) keep for ResNet50V2, remove for EfficientNet
    ], name="data_augmentation")

def show_image(image_path, target_class):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(f"original random image from class {target_class}")
    plt.axis(False)


def base_model(train_data, test_data, experiment_name, checkpoint_file, trainable=False, include_top=False, data_augmentation=None):


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
    # if use_augmentation is true, use data augmentation
    if data_augmentation is not None:
        x = data_augmentation(inputs)
    else:
        x = base_model(inputs)
    print(f"shape after passing inputs through base model:{x.shape}")
    
    x = base_model(x, training=False) # pass augmented images to base model but keep it in inference mode, so batchnorm layers don't get updated
    
    # 6. Average pool the output of the base model(aggregate all the most important information, reduce number of computations)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"shape after GlobalAveragePooling2D: {x.shape}")

    # 7. create the output activation layer
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

    # 8. combine the inputs with outputs into a model
    model = tf.keras.Model(inputs, outputs)

    # 9. compile for the model
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    # 10. Fit the model and save the history
    history = model.fit(
                    train_data, 
                    epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=int(0.25 * len(test_data)),
                    callbacks=[create_tensorboard_callback("transfer_learning", experiment_name), create_checkpoint_callback(checkpoint_file)]
                )

    return history, model


def main():
    # walk_through_dir("10_food_classes_1_percent")

    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_10_percent_dir,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                batch_size=BATCH_SIZE
                                                                                )
    test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_10_percent_dir,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                batch_size=BATCH_SIZE,
                                                                                )
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
    
    ##########################################################################################
    # # create base model""
    # history, model_base = base_model(train_data_10_percent, test_data_10_percent, "10_percent_feature_extraction")

    # # evaluate model
    # model_base.evaluate(test_data_10_percent)

    # # layer name
    # for idx, layer in enumerate(model_base.layers):
    #     print(layer.name)
    
    # # model summary
    # print(model_base.summary())

    # plot_loss_curves(history)
    
    # ###########################################
    # # show random image
    # target_class = random.choice(train_data_1_percent.class_names)
    # target_dir = "10_food_classes_1_percent/train/" + target_class
    # random_image = random.choice(os.listdir(target_dir)) 
    # random_image_path = target_dir + "/" +  random_image
    # show_image(random_image_path, target_class)

    # # show augmentation image
    # img = cv2.imread(random_image_path)
    augmentation_generator = augmentation_params()
    # augmentation_img = augmentation_generator(tf.expand_dims(img, axis=0))
    # plt.figure()
    # plt.title("augmentation image")
    # plt.axis(False)
    # plt.imshow(tf.squeeze(augmentation_img))

   
    ##########################################################################################

    # model_1: Feature extraction transfer learning on 1% of the training data with data augmentation
    # history1, model1 = base_model(train_data_1_percent, test_data_1_percent, "1_percent_data_aug", "1_percent_model_checkpoint_weights/checkpoint.ckpt", data_augmentation=augmentation_generator)
    
    # model1.evaluate(test_data_1_percent)

    # plot_loss_curves(history1)

    # model_2: Feature extraction transfer learning model with 10% data and data augmentation
    history2,  model2 = base_model(train_data_10_percent, test_data_10_percent, "10_percent_data_aug", "10_percent_model_checkpoint_weights/checkpoint.ckpt", data_augmentation=augmentation_generator)
    print(model2.summary())
    plot_loss_curves(history2)
    print(model2.evaluate(test_data_10_percent))


   

if __name__ == "__main__":
    main()