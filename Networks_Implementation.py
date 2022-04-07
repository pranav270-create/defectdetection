import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import os
import matplotlib
import copy
import tensorflow as tf
from torch.autograd import Variable
filename = "/home/allen/data/defect/frozen_inference_graph.pb"

class Detector_RCNN(nn.Module):
    def __init__(self, filename, width = 0.01, norm = 'batch'):
        super(Detector_RCNN, self).__init__()

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=detection_graph)


    # Input tensor is the image
    self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          
    def inference(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                                feed_dict={image_tensor: image_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        return boxes, classes, scores, num


class Classifier_CNN(nn.Module):
    
    def conv_block(self, in_channels, out_channels, kernel, stride = 1, padding = 'same', bias = True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride= stride, padding = padding, bias=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride= stride, padding = padding, bias=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride= stride, padding = padding, bias=True)
        )

    def norm_act(self, p, size, alpha):
        return nn.Sequential(
            nn.Dropout2d(p = p),
            nn.BatchNorm2d(size, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope = alpha)
            )


    def __init__(self, width = 0.01):
        super(Classifier_CNN, self).__init__()

        # define network topology
        self.c1 = conv_block(self, 3, 32, 3)
        self.n1 = norm_act(self, 0.3, 32, 0.05)

        self.c2 = conv_block(self, 32, 64, 3)
        self.n2 = norm_act(self, 0.3, 64, 0.05)

        self.c3 = conv_block(self, 64, 128, 3)
        self.n3 = norm_act(self, 0.3, 128, 0.05)

        self.linear = nn.Linear(15488,1, bias=True)

        self.maxpool = nn.MaxPool2d(2)
        
        # initialize weights
        for layer in self.layers.keys():
            if('conv' == layer[:-1]):
                # from pytorch init documentation https://pytorch.org/docs/stable/nn.init.html
                nn.init.normal_(self.layers[layer].weight, mean=0, std=width)

    def forward(self, x):
        
        output1 = self.n1(self.c1(x))
        x = self.maxpool(output1)

        output2 = self.n2(self.c2(x))
        x = self.maxpool(output2)

        output3 = self.n3(self.c3(x))
        x = self.maxpool(output3)

        # flatten x before the fully connected layer via pytorch's view function
        x = x.view(x.size(0), -1)
        x = nn.Sigmoid(self.linear(x))

        return x

    def loss(self):
        return nn.CrossEntropyLoss()

