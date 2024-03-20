import tensorflow as tf
from tensorflow.keras import layers
#4决定了一下你要输出的通道数,输入的通道数取决与x
x=tf.range(12)
# layer=layers.Conv2D(4,kernel_size=5,strides=1,padding='valid')
# out=layer(x)

print(x.shape)

# layer=layers.Conv2D(4,kernel_size=5,strides=1,padding='same')
# out=layer(x)
#用same 就是输出大小和输入的大小一致  strides=2 输出的shape会有影响减半 32 -》 16

# layer=layers.Conv2D(4,kernel_size=5,strides=1,padding='same')
# out=layer(x)
#可以用layer。kernel来实现查看weight的存储内容         bias-》layer.bias



#池化操作语法
#x [1,14,14,4]
pool=layers.MaxPool2D(2,strides=2)
out=pool(x)
# out [1,6,6,4] 不改变Channel大小
# out=tf.nn.max_pooling(x,2,strides=2,padd) 用这种方式写更灵活，需要接触底层的时候用

#上采样操作 ，将低维的都变成高维
#x [1,7,7,4]
layer=layers.UpSampling2D(size=3)
out=layer(x)
# out [1,21,21,4]

#Relu函数可以把feture map 负值去掉去掉
x=tf.random.normal([2,3])
tf.nn.relu(x)  #把负值给去掉了

layers.ReLU()(x)  #另一种写法


