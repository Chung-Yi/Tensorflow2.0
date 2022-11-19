import imghdr
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from pizza_and_steak import create_conv_model

checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
test_image = 'data/pizza_steak/test_images/steak1.jpg'
img_size = 224

pizza_img = "data/pizza_steak/test/pizza/11297.jpg"

class_num = 10

def predict_image(model, filename):
    img = image.load_img(filename, target_size=(img_size, img_size))
    plt.imshow(img)

    Y = image.img_to_array(img)
    X = np.expand_dims(Y,axis=0)

    X /= 255

    val = model.predict(X)
    print(val)


def main():
    # create model
    model = create_conv_model(class_num)

    # get the latest cp
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model.load_weights(latest)

    img = mpimg.imread(pizza_img)
    # img = cv2.resize(img, (img_size, img_size))
    img = tf.image.resize(img, [img_size, img_size])
    # plt.imshow(img)
    # plt.show()
    img /= 255
    img = tf.expand_dims(img, axis=0)

    pred = model.predict(img)

    predict_image(model, pizza_img)






if __name__ == '__main__':
    main()