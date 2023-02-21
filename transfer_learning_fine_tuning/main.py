
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
import cv2
import numpy as np
from extras.helper_function import walk_through_dir, create_tensorboard_callback, plot_loss_curves, create_checkpoint_callback
from tensorflow.keras.layers.experimental import preprocessing


train_1_percent_dir = "10_food_classes_1_percent/train"
test_1_percent_dir = "10_food_classes_1_percent/test"

train_10_percent_dir = "10_food_classes_10_percent/train"
test_10_percent_dir = "10_food_classes_10_percent/test"

train_all_dir = "10_food_classes_all_data/train"
test_all_dir = "10_food_classes_all_data/test"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INIT_EPOCH = 5

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


def get_base_model(trainable=False, include_top=False, data_augmentation=None):


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
    # history = model.fit(
    #                 train_data, 
    #                 epochs=5,
    #                 steps_per_epoch=len(train_data),
    #                 validation_data=test_data,
    #                 validation_steps=int(0.25 * len(test_data)),
    #                 callbacks=[create_tensorboard_callback("transfer_learning", experiment_name), create_checkpoint_callback(checkpoint_file)]
    #             )

    return model

def fine_tune_model(model):
    model.layers[2].trainable = True
    for layer in model.layers[2].layers[:-10]:
        layer.trainable = False
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])
    return model

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))


    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()



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
    
    train_data_all = tf.keras.preprocessing.image_dataset_from_directory(directory=train_all_dir, 
                                                                               image_size=IMG_SIZE,
                                                                               label_mode="categorical",
                                                                               batch_size=BATCH_SIZE)

    test_data_all = tf.keras.preprocessing.image_dataset_from_directory(directory=test_all_dir, 
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
    model2 = get_base_model(data_augmentation=augmentation_generator)
    history2 = model2.fit(
                    train_data_10_percent, 
                    epochs=INIT_EPOCH,
                    steps_per_epoch=len(train_data_10_percent),
                    validation_data=test_data_10_percent,
                    validation_steps=int(0.25 * len(test_data_10_percent)),
                    callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_data_aug"), create_checkpoint_callback("10_percent_model_checkpoint_weights/checkpoint.ckpt")]
                )
    print(model2.summary())
    plot_loss_curves(history2)
    evaluate_result_model2 = model2.evaluate(test_data_10_percent)
    print(evaluate_result_model2)

    # load model_2 weights
    model2.load_weights("10_percent_model_checkpoint_weights/checkpoint.ckpt")
    evaluate_result_model2_load_weights = model2.evaluate(test_data_10_percent)
    print(evaluate_result_model2_load_weights)
    print(np.isclose(evaluate_result_model2, evaluate_result_model2_load_weights))

    #####################################################################################

    model_fine_tune = fine_tune_model(model2)

    # model_fine_tune.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])
    print(len(model_fine_tune.trainable_variables))
    # os._exit(0)

    history_fine_10_percent_data_aug = model_fine_tune.fit(train_data_10_percent, 
                        epochs=INIT_EPOCH + 5,
                        validation_data=test_data_10_percent,
                        validation_steps=int(0.25*len(test_data_10_percent)),
                        initial_epoch=history2.epoch[-1], # start from the last epoch
                        callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_fine_tune_last_10")]
                        )

    plot_loss_curves(history_fine_10_percent_data_aug)
    results_fine_tune_10_percent = model_fine_tune.evaluate(test_data_10_percent)
    print(results_fine_tune_10_percent)

    compare_historys(history2, history_fine_10_percent_data_aug)

    ################################################################################
    
    # fine tune model with all data
    model2.load_weights("10_percent_model_checkpoint_weights/checkpoint.ckpt")
    print(model2.evaluate(test_data_all))

    model2.compile(loss="categorical_crossentropy", 
                   optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                   metrics=["accuracy"])
    history_fine_tune_data_all = model2.fit(
        train_data_all, 
        epochs=INIT_EPOCH + 5,
        validation_data=test_data_all,
        validation_steps=int(0.25*len(test_data_all)),
        initial_epoch=history2.epoch[-1],
        callbacks=[create_tensorboard_callback("transfer_learning", "full_10_classes_fine_tune_last_10")]
    )
    result_fine_tune_all_data = model2.evaluate(test_data_all)
    print(result_fine_tune_all_data)

    compare_historys(history2, history_fine_tune_data_all)




if __name__ == "__main__":
    main()