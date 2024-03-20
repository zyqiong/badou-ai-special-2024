import tensorflow as tf
from tensorflow.keras import Sequential,layers


class Basic_block(layers.Layer):

    def __init__(self,filte_num,stride=1):
        super(self,Basic_block).__init__()

        self.conv1=layers.Conv2D(filte_num,(3,3),strides=1,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')





