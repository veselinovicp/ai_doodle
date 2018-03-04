from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_socketio import emit


class StyleTransfer:

    def __init__(self, width=512, height=512, content_image_path = '../data/hugo.jpg', style_image_path = '../data/wave.jpg',
                 content_image_base64 = None,
                 style_image_base64 =None,
                 iterations = 10, content_weight = 0.025, style_weight = 5.0, total_variation_weight = 1.0,
                 output_path = None, web_socket_channel=None, max_fun = 20 ):
        self.height = height
        self.width = width
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.iterations = iterations

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.content_image_base64 = content_image_base64
        self.style_image_base64 = style_image_base64
        self.output_path = output_path
        self.web_socket_channel = web_socket_channel
        self.max_fun = max_fun


    def transfer(self):


        # content_image_path = '../data/hugo.jpg'
        if self.content_image_base64 is None:
            content_image = Image.open(self.content_image_path)
        else:
            content_image = Image.open(BytesIO(base64.b64decode(self.content_image_base64)))
        content_image = content_image.resize((self.height, self.width))

        # style_image_path = '../data/wave.jpg'
        if self.style_image_base64 is None:
            style_image = Image.open(self.style_image_path)
        else:
            style_image = Image.open(BytesIO(base64.b64decode(self.style_image_base64)))
        style_image = style_image.resize((self.height, self.width))

        content_array = np.asarray(content_image, dtype='float32')
        content_array = np.expand_dims(content_array, axis=0)
        print(content_array.shape)

        style_array = np.asarray(style_image, dtype='float32')
        style_array = np.expand_dims(style_array, axis=0)
        print(style_array.shape)

        dimensions = (1, self.height, self.width, 3)

        content_array[:, :, :, 0] -= 103.939
        content_array[:, :, :, 1] -= 116.779
        content_array[:, :, :, 2] -= 123.68
        if content_array.shape[3] == 4:
            content_array = content_array[:, :, :, 0:3]#::-1  :-1
        # content_array = content_array.reshape(dimensions)

        style_array[:, :, :, 0] -= 103.939
        style_array[:, :, :, 1] -= 116.779
        style_array[:, :, :, 2] -= 123.68
        if style_array.shape[3] == 4:
            style_array = style_array[:, :, :, 0:3]#::-1 :-1
        # style_array = style_array.reshape(dimensions)

        content_image = backend.variable(content_array)
        style_image = backend.variable(style_array)

        combination_image = backend.placeholder(dimensions)

        input_tensor = backend.concatenate([content_image,
                                            style_image,
                                            combination_image], axis=0)
        model = VGG16(input_tensor=input_tensor, weights='imagenet',
                      include_top=False)

        layers = dict([(layer.name, layer.output) for layer in model.layers])



        loss = backend.variable(0.)

        def content_loss(content, combination):
            return backend.sum(backend.square(combination - content))

        layer_features = layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        loss += self.content_weight * content_loss(content_image_features,
                                              combination_features)

        def gram_matrix(x):
            features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
            gram = backend.dot(features, backend.transpose(features))
            return gram

        def style_loss(style, combination):
            S = gram_matrix(style)
            C = gram_matrix(combination)
            channels = 3
            size = self.height * self.width
            return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

        feature_layers = ['block1_conv2', 'block2_conv2',
                          'block3_conv3', 'block4_conv3',
                          'block5_conv3']
        for layer_name in feature_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features)
            loss += (self.style_weight / len(feature_layers)) * sl

        def total_variation_loss(x):
            a = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, 1:, :self.width - 1, :])
            b = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, :self.height - 1, 1:, :])
            return backend.sum(backend.pow(a + b, 1.25))

        loss += self.total_variation_weight * total_variation_loss(combination_image)

        grads = backend.gradients(loss, combination_image)

        outputs = [loss]
        outputs += grads
        f_outputs = backend.function([combination_image], outputs)

        def eval_loss_and_grads(x):
            x = x.reshape(dimensions)
            outs = f_outputs([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            return loss_value, grad_values

        class Evaluator(object):

            def __init__(self):
                self.loss_value = None
                self.grads_values = None

            def loss(self, x):
                assert self.loss_value is None
                loss_value, grad_values = eval_loss_and_grads(x)
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        evaluator = Evaluator()

        x = np.random.uniform(0, 255, dimensions) - 128.



        for i in range(self.iterations):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=self.max_fun)
            print('Current loss value:', min_val)
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
            if self.web_socket_channel is not None:
                self.__post_result_via_web_socket(x)

        if self.web_socket_channel is None:
            x = x.reshape((self.height, self.width, 3))
            x = x[:, :, ::-1]
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = np.clip(x, 0, 255).astype('uint8')

            result = Image.fromarray(x)

            if self.output_path is not None:
                imsave(self.output_path, result)
            else:
                buffered = BytesIO()
                result.save(buffered, format="JPEG")
                # return base64.b64encode(buffered.getvalue())
                return buffered.getvalue()

    def __post_result_via_web_socket(self, input):
        x = np.copy(input)
        x = x.reshape((self.height, self.width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = np.clip(x, 0, 255).astype('uint8')
        result = Image.fromarray(x)
        buffered = BytesIO()
        result.save(buffered, format="JPEG")
        result =  base64.b64encode(buffered.getvalue()).decode("utf-8")
        emit(self.web_socket_channel, result)
        buffered.close()




