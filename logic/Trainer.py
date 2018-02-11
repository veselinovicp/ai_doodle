from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization, UpSampling2D, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import cv2
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from skimage import img_as_ubyte
from scipy.optimize import fmin_l_bfgs_b


class Trainer:

    def __init__(self, train_size=100, batch_size=1, epochs=10):
        print("Trainer created")
        self.train_size = train_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.__load_data()
        self.__create_model()

    def train(self):
        print("Start training")
        model_checkpoint = ModelCheckpoint("../output/weights.hdf5", monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='min')  # weights.{epoch:02d}.hdf5
        tensor_board = TensorBoard(log_dir='../output/', histogram_freq=0,
                                   write_graph=True, write_images=False)  # log_dir=self.folder,

        self.model.fit(self.input_images, self.output_images, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       callbacks=[model_checkpoint, tensor_board], shuffle=True, validation_split=0.1)

    # def predict(self):
    def predict_test_value(self, weights_path, image_path="../data/input_image.png"):

        self.model.load_weights(weights_path)
        input = cv2.imread(image_path)
        input = cv2.resize(input, (200, 200))
        gray_scale = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        array = img_to_array(gray_scale)
        array = array.reshape((1,) + array.shape)
        predicted_output = self.model.predict(array)
        print("shape ", predicted_output.shape)
        predicted_image = array_to_img(predicted_output[0])
        plt.imsave('../data/predicted_image.png', predicted_image, cmap=plt.cm.gray)
        # return predicted_output, self.test_output[test_number:test_number+32]

    def __load_data(self):
        print("Start loading data")

        generator = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

        train = generator.flow_from_directory('../data/michelangelo',
                                              target_size=(674, 514),
                                              batch_size=32,
                                              class_mode='input')

        gray_scales = []
        edgeses = []
        for i in range(self.train_size):
            print((i * 100. / self.train_size), " %")
            img, _ = train.next()
            gray_scale = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
            gray_scale = cv2.resize(gray_scale, (200, 200))

            edges = feature.canny(gray_scale, sigma=3.)
            # edges = cv2.resize(edges, (200, 200))
            if i == 0:
                print("output min", img_to_array(gray_scale).min())
                print("output max", img_to_array(gray_scale).max())

            gray_scales.append(img_to_array(gray_scale))
            edgeses.append(img_to_array(edges))

            if i == 0:
                print("input min", img_to_array(edges).min())
                print("output max", img_to_array(gray_scale).max())
            # plt.imsave('../data/michelangelo/input_image'+str(i)+'.png', edges, cmap=plt.cm.gray)
            # plt.imsave('../data/michelangelo/output_image' + str(i) + '.png', gray_scale, cmap=plt.cm.gray)

        self.input_images = np.stack(edgeses)
        print("input image shape ", self.input_images.shape)
        self.output_images = np.stack(gray_scales)
        print("output image shape ", self.output_images.shape)

        # input_image = load_img('../data/input_image.png', grayscale=True)  # this is a PIL image
        # self.input_image = img_to_array(input_image)
        # self.input_image = self.input_image.reshape((1,) + self.input_image.shape)
        # print("input image shape ", self.input_image.shape)
        #
        # output_image = load_img('../data/output_image.png', grayscale=True)  # this is a PIL image
        # self.output_image = img_to_array(output_image)
        # self.output_image = self.output_image.reshape((1,) + self.output_image.shape)
        #
        # print("output image shape ", self.output_image.shape)

    def __create_model(self):

        # raise NotImplementedError("Subclass must implement abstract method")

        K.clear_session()
        self.model = Sequential()

        self.model.add(Conv2D(128, (3, 3), input_shape=(200, 200, 1), padding='same',
                              activation='relu'))  # input_shape=(674, 514, 1)
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # self.model.add(BatchNormalization())

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        # self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # self.model.add(Conv2D(1, (3, 3), padding='same', activation='linear'))

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(200 * 200 * 1, activation='linear'))
        self.model.add(Reshape((200, 200, 1)))

        # self.model.add(Dense(10, activation='relu', input_shape=(200, 200, 1)))
        #
        # self.model.add(Flatten())
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(100, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(200 * 200 * 1, activation='linear'))
        # self.model.add(Reshape((200, 200, 1)))

        self.model.compile(loss='mean_squared_error', optimizer='adam',
                           metrics=['accuracy'])

        print(self.model.summary())


class VGGTrainer:

    def __init__(self, train_size=100, batch_size=1, epochs=10):
        print("Trainer created")
        self.train_size = train_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.__load_data()
        self.__create_model()

    def train(self):
        print("Start training")
        model_checkpoint = ModelCheckpoint("../output/weights.hdf5", monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='min')  # weights.{epoch:02d}.hdf5
        tensor_board = TensorBoard(log_dir='../output/', histogram_freq=0,
                                   write_graph=True, write_images=False)  # log_dir=self.folder,

        self.model.fit(self.input_images, self.output_images, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       callbacks=[model_checkpoint, tensor_board], shuffle=True, validation_split=0.1)

    # def predict(self):
    def predict_test_value(self, weights_path, image_path="../data/input_image.png"):

        self.model.load_weights(weights_path)
        input = cv2.imread(image_path)
        input = cv2.resize(input, (224, 224))
        # gray_scale = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        array = img_to_array(input)
        array = array.reshape((1,) + array.shape)
        array = preprocess_input(array)
        predicted_output = self.model.predict(array)
        print("shape ", predicted_output.shape)
        predicted_image = array_to_img(predicted_output[0])
        plt.imsave('../data/predicted_image.png', predicted_image, cmap=plt.cm.gray)
        # return predicted_output, self.test_output[test_number:test_number+32]

    def __load_data(self):
        print("Start loading data")

        generator = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

        train = generator.flow_from_directory('../data/michelangelo',
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='input')

        gray_scales = []
        edgeses = []
        for i in range(self.train_size):
            print((i * 100. / self.train_size), " %")
            img, _ = train.next()
            gray_scale = cv2.cvtColor(img[0], cv2.COLOR_RGB2GRAY)#COLOR_BGR2RGB  COLOR_BGR2GRAY

            # edges = feature.canny(gray_scale, sigma=3.)
            # img_cv = np.array(img[0])
            # # Convert RGB to BGR
            # img_cv = img_cv[:, :, ::-1].copy()
            # slice1Copy = np.uint8(slice1)

            # opencvImage = np.array(img[0])

            opencv_image = cv2.cvtColor(np.array(img[0]), cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            # opencv_image = opencv_image * 255.0

            # Display the OpenCV image using inbuilt methods.
            # cv2.imshow('Demo Image', opencv_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            array = img_to_array(img[0])
            array = array[:, :, ::-1].copy()
            # edges = cv2.Canny(np.uint8(opencv_image), 100, 200)# np.uint8(array)

            edges = feature.canny(gray_scale, sigma=3.)
            # print(edges.dtype)
            # cv2.imshow('image', np.array(edges))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            edges = img_as_ubyte(edges)

            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # edges = cv2.resize(edges, (200, 200))
            if i == 0:
                print("output min", img_to_array(gray_scale).min())
                print("output max", img_to_array(gray_scale).max())

            # gray_scale = np.array(gray_scale, dtype=np.uint8)
            # gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_GRAY2BGR)

            gray_scales.append(img_to_array(gray_scale))
            edgeses.append(img_to_array(edges))

            if i == 0:
                print("input min", img_to_array(edges).min())
                print("input max", img_to_array(edges).max())
            # plt.imsave('../data/michelangelo/input_image'+str(i)+'.png', edges, cmap=plt.cm.gray)
            # plt.imsave('../data/michelangelo/output_image' + str(i) + '.png', gray_scale, cmap=plt.cm.gray)

        self.input_images = np.stack(edgeses)
        self.input_images = preprocess_input(self.input_images)
        print("input image shape ", self.input_images.shape)
        self.output_images = np.stack(gray_scales)
        # self.output_images = preprocess_input(self.output_images)
        print("output image shape ", self.output_images.shape)

        # input_image = load_img('../data/input_image.png', grayscale=True)  # this is a PIL image
        # self.input_image = img_to_array(input_image)
        # self.input_image = self.input_image.reshape((1,) + self.input_image.shape)
        # print("input image shape ", self.input_image.shape)
        #
        # output_image = load_img('../data/output_image.png', grayscale=True)  # this is a PIL image
        # self.output_image = img_to_array(output_image)
        # self.output_image = self.output_image.reshape((1,) + self.output_image.shape)
        #
        # print("output image shape ", self.output_image.shape)

    def __create_model(self):

        # raise NotImplementedError("Subclass must implement abstract method")

        K.clear_session()

        vgg_model = VGG16(include_top=False, input_shape = (224, 224, 3))
        vgg_model.layers

        # layer.trainable = False
        print(vgg_model.summary())
        self.model = Sequential()
        for i, layer in enumerate(vgg_model.layers):
            layer.trainable = False
            self.model.add(layer)

        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.2))
        # self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(224 * 224 * 1, activation='linear'))
        self.model.add(Reshape((224, 224, 1)))



        # self.model.add(Conv2D(128, (3, 3), input_shape=(200, 200, 1), padding='same',
        #                       activation='relu'))  # input_shape=(674, 514, 1)
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        #
        #
        # self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        #
        #
        # self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        #
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        #
        # self.model.add(Flatten())
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(1000, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(100, activation='relu'))
        # self.model.add(Dense(200 * 200 * 1, activation='linear'))
        # self.model.add(Reshape((200, 200, 1)))


        self.model.compile(loss='mean_squared_error', optimizer='adam',
                           metrics=['accuracy'])

        print(self.model.summary())
