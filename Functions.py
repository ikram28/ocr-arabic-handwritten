import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import string
from shutil import copyfile, rmtree
import re
import cv2
from PIL import Image, ImageDraw
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda
import tensorflow.keras.backend as K


racine_gt = "C:/Users/ASUS ROG STRIX/Desktop/OCR/enit_ifn database/ifnenit-database-demo/data/set_a/tru"

def get_Word(name):
    file_name = name.split("/")[-1].split(".")[0]
    path_gt = racine_gt + "/" + file_name + ".tru"
    load_profile = open(path_gt, "r")
    label = load_profile.read().splitlines()[6]
    word = label.split(":")[-3].split(";")[0].split("|")[:-1]
    return word


def evaluate_word(name):
    word = get_Word(name)
    for i, car in enumerate(word):
        if car[-1] == "1" or car[-1] == "2":
            word[i] = "-"
    return word

def get_lexicon_2(names):
    arabic_labels = []
    for name in names:
        arabic_labels = arabic_labels + evaluate_word(name)
    return list(dict.fromkeys(arabic_labels))

def get_lengths(names):
    d = {}
    for name in names:
        file_name = name.split("/")[-1].split(".")[0]
        word = get_Word(name)
        d[file_name] = len(word)
    return d

def open_image(name, img_size=[100, 300]):
    img = cv2.imread(name, 0)
    img = cv2.resize(img, (img_size[1], img_size[0]), Image.LANCZOS)
    img = cv2.threshold(img, 255 // 2, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(img)
    word = get_Word(name)
    return img, word

class Readf:
    def __init__(self, img_size=(100, 300), max_len=17, normed=False, batch_size=64, classes={}, mean=118.2423, std=36.72):
        self.batch_size = batch_size
        self.img_size = img_size
        self.normed = normed
        self.classes = classes
        self.max_len = max_len
        self.mean = mean
        self.std = std
        self.voc = list(self.classes.keys())

        if type(classes) == dict:
            self.blank = classes["-"]

    def make_target(self, text):
        return np.array([self.classes[char] if char in self.voc else self.classes['-'] for char in text])

    def get_labels(self, names):
        Y_data = np.full([len(names), self.max_len], self.blank)
        for i, name in enumerate(names):
            img, word = open_image(name, self.img_size)
            word = self.make_target(word)
            Y_data[i, 0:len(word)] = word
        return Y_data

    def get_blank_matrices(self):
        shape = (self.batch_size,) + self.img_size
        X_data = np.empty(shape)
        Y_data = np.full([self.batch_size, self.max_len], self.blank)
        input_length = np.ones((self.batch_size, 1))
        label_length = np.zeros((self.batch_size, 1))
        return X_data, Y_data, input_length, label_length

    def run_generator(self, names, downsample_factor=2):
        n_instances = len(names)
        N = n_instances // self.batch_size
        rem = n_instances % self.batch_size
    
        while True:
            X_data, Y_data, input_length, label_length = self.get_blank_matrices()
    
            i, n = 0, 0
    
            for name in names:
                img, word = open_image(name, self.img_size)
                word = self.make_target(word)
    
                # Skip if word length is zero
                if len(word) == 0:
                    continue
    
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = (self.img_size[0] + 4) // downsample_factor - 2
    
                X_data[i] = img[np.newaxis, :, :]
                i += 1
    
                if i == self.batch_size:
                    n += 1
                    inputs = {
                        'the_input': X_data,
                        'the_labels': Y_data,
                        'input_length': input_length,
                        'label_length': label_length,
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}
                    yield (inputs, outputs)
    
                    # Reset everything
                    X_data, Y_data, input_length, label_length = self.get_blank_matrices()
                    i = 0
    
            # Handle remaining instances
            if rem > 0:
                inputs = {
                    'the_input': X_data[:rem],
                    'the_labels': Y_data[:rem],
                    'input_length': input_length[:rem],
                    'label_length': label_length[:rem],
                }
                outputs = {'ctc': np.zeros([rem])}
                yield (inputs, outputs)


    


class CRNN:
    def __init__(self, img_w, img_h, output_size, max_len):
        self.img_w = img_w
        self.img_h = img_h
        self.output_size = output_size
        self.max_len = max_len

        # Network parameters
        self.conv_filters = 16
        self.kernel_size = (3, 3)
        self.pool_size = 2
        self.time_dense_size = 32
        self.rnn_size = 512

        self.model = self.build_model()

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        #y_pred = y_pred[:, 2:, :]

        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(self):
        # Input layer
        input_data = Input(name='the_input', shape=(self.img_w, self.img_h), dtype='float32')
        
        # Expand dimensions to include channel dimension
        expanded_input = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_data)
        
        # Convolutional layers
        conv_1 = Conv2D(self.conv_filters, self.kernel_size, padding='same', activation='relu', name='conv1')(expanded_input)
        pool_1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='pool1')(conv_1)
        
        conv_2 = Conv2D(self.conv_filters, self.kernel_size, padding='same', activation='relu', name='conv2')(pool_1)
        pool_2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='pool2')(conv_2)
        
        # Reshape layer to prepare for RNN
        
        #Reshaping the outputs from a CNN to be the inputs of an RNN enables the integration of spatially encoded features with temporal dependencies for sequential data analysis
        conv_to_rnn_dims = (self.img_w // (self.pool_size * 2), self.img_h // (self.pool_size * 2) * self.conv_filters)
        reshaped = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(pool_2)
        
        # Dense layer
        dense = Dense(self.time_dense_size, activation='relu', name='dense')(reshaped)
        
        # RNN layers
        rnn = Bidirectional(LSTM(self.rnn_size, return_sequences=True), name='biLSTM')(dense)
        
        # Output layer
        y_pred = Dense(self.output_size, activation='softmax', name='softmax')(rnn)
        
        
       
        
        labels = Input(name='the_labels', shape=[self.max_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        
        ctc_loss = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[ctc_loss, y_pred])
        model.summary()



        return model
    
    
    
