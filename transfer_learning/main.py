import datetime
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from utils.helper import plot_loss_curves

IMG_SHAPE = (224, 224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("training images: ")
train_data = train_datagen.flow_from_directory(train_dir, target_size=IMG_SHAPE, class_mode="categorical")

print("testing images: ")
test_data = test_datagen.flow_from_directory(test_dir, target_size=IMG_SHAPE, class_mode="categorical")

# setting up callback

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback



def create_model(model_url, num_classes=10):
    feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, 
                                       name="feature_extractor_layer", 
                                       input_shape=IMG_SHAPE+(3,)) # freeze the already learned patterns
    # create the model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model

# create model using tensorflow hub
# compare the following two models

# create resnetv2_50
resnetv2_50_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
resnet_model = create_model(resnetv2_50_url, train_data.num_classes)

# compile the resnet model
resnet_model.compile(loss="categorical_crossentropy", 
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"]
                    )

# train the resnet model
resnet_history = resnet_model.fit(
    train_data, epochs=10, steps_per_epoch=len(train_data),
    validation_data=test_data, validation_steps=len(test_data),
    callbacks=[create_tensorboard_callback(dir_name="tensorboard_hub", 
                                           experiment_name="resnetv2"
                                          )]
)

plot_loss_curves(resnet_history)

# create efficientnet_b0
efficientnet_b0_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
efficientnet_model = create_model(efficientnet_b0_url, train_data.num_classes)

efficientnet_model.compile(loss="categorical_crossentropy", 
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

efficientnet_history = efficientnet_model.fit(
    train_data, epochs=10, steps_per_epoch=len(train_data), 
    validation_data=test_data, validation_steps=len(test_data),
    callbacks=[create_tensorboard_callback(dir_name="tensorboard_hub", 
                                         experiment_name="efficientb0"
                                         )]
)

plot_loss_curves(efficientnet_history)