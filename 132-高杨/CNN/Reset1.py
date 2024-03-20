import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import layers,Sequential


#Reset最基本的的单元
class Basic_block(layers.Layer):  #Layer要大写继承类
                #filter_num输入的通道数
    def __init__(self,filter_num,stride=1):
        #super(Basic_block, self).__init__()这一语句中，调用了父类layers.Layer的__init__方法，以确保正确地初始化父类。
        super(Basic_block,self).__init__()

        self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        #短接
        if stride!=1:
            #如果不步长不等于1，那么需要转化步长和最初的convlution一致
         self.downsample=Sequential()
         self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
             self.downsample= lambda x:x



    # layers.Layer.call()可以查看参数
    def call(self,inputs,training=None):
        #前向传播逻辑这里写的
        out=self.conv1(inputs)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)


        #残差就是输出后的函数（f（x））-最开始传进来的变量x   output=f（x）
        #identity是尽管变化（依据传入数据和最后卷积过后的数据而定，用步长改变）
        identity=self.downsample(inputs)


        #最初的数据直接和卷积后的层相加，relu可以多次使用是因为没有参数，只是作为方法使用
        output=layers.add([out,identity])
        output=self.relu(output)  #等价于 tf，nn，relu
        return output

# Res Block 是由多个basic block堆叠在一起




class ResNet(keras.Model):  #keras。Model和 keras。layer差不多

    #例如layers_dim= 【2，2，2，2】   每个basicblock有两层， 一共有四个组成一个Resnet
    def __init__(self,layers_dim,num_classer=100):
        super(ResNet,self).__init__()

        #最开始输入数据的层
        self.stem = Sequential([layers.Conv2D(64,(3,3),strides=(1,1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
                                ])
        self.layers1=self.bulid_resblock(64,layers_dim[0])
        #strides= 2 是因为随着数据往后进行，dim会越来越小，约定俗称
        self.layers2 = self.bulid_resblock(128, layers_dim[1],stride=2)
        self.layers3 = self.bulid_resblock(256, layers_dim[2],stride=2)
        self.layers4 = self.bulid_resblock(512, layers_dim[3],stride=2)


        #全连接层
        # output [b,512,h,w], 不知道长和宽的情况下，可以用以下语句计算
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(num_classer)



    def call(self,inputs,training=None):

        x = self.stem(inputs)

        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)

        # 通过avgpool已经变成了一个 [b,c]
        x = self.avgpool(x)
        x= self.fc(x)

        return  x

    def bulid_resblock(self,filter_num,blocks,stride=1):
        res_blocks=Sequential()
            #只在第一步的时候做下采样，之后的保持一致。
        res_blocks.add(Basic_block(filter_num,stride))

        for _ in range(1,blocks):
            # 只在第一步的时候做下采样，之后的保持一致。
            res_blocks.add(Basic_block(filter_num,stride=1))

        return  res_blocks

def resnet18():
    #18层意思是 每个basic_block都有2个小单元（convlution） 一个resnet一个有两个asic_block
    #而resnet一共有四个所有一共用16个 +输入层+输出层
    return  ResNet([2,2,2,2])

def resnet34():

    return  ResNet([3,4,6,3])








