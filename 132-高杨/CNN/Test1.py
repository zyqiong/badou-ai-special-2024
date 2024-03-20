import tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets,Sequential
import os

import warnings
warnings.filterwarnings('ignore')

print(tf.test.is_gpu_available())

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)
conv_layers = [
    # five units of conv +max pooling
    # 一般把channel扩大一倍作用是得到最后的像素变小，但是像素中的信息量会慢慢变大，慢慢转化为高层的概念
    # 例如刚开始卷积的是一个车子的外形像素点级别的的特征，后面高级特征例如车子的车门天窗等特征
    # 安装软件包如果出错了用Python的 PYPI软件包下载
    # 安装不了包的话可以用下方的终端进行安装


    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

]
def preprocess(x,y):
    #[0 -   1]
    x=tf.cast(x,dtype=tf.float32)/ 255. #注意要加上。表示是浮点型数据
    y=tf.cast(y,dtype=tf.int32)  #注意这里是整型数据
    return x,y


(x,y),(x_test,y_test)=datasets.cifar100.load_data()
print(x.shape,y.shape,x_test.shape,y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]) )
def main():
    conv_net = Sequential(conv_layers)
    #conv_net.build(input_shape=[None,32,32,3])
    # x=tf.random.normal([4,32,32,3])
    # out=conv_net(x)
    # print(out.shape)
    fc_net= Sequential([
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(100, activation=None)

    ])
    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None,512])

    #优化器
    optimizer=optimizers.Adam(lr=1e-4)



    #python中两个列表想加会合并
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                #[b,32,32,3] ->  [b,1,1,512]
                out = conv_net(x)
                #squeeze
                #-1得意思打平成一维，512列
                out=tf.reshape(out,[-1,512])
                #全链接层   [b,512] -> [b,100]
                logits=fc_net(out)
                #[b] ->[b,100]
                #要以管理员的方式运行软件
                y_onehot= tf.one_hot(y,depth=100)
                #计算多分类任务得交叉熵
                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                #这一语句计算了模型的损失函数。y_onehot表示真实标签的独热编码形式，logits表示模型的输出。categorical_crossentropy是一种常用的交叉熵损失函数，
                # 用于多分类问题。from_logits=True表示logits是未经过Softmax激活函数的输出，需要在损失函数内部进行Softmax计算。
                loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,variables)  #这一语句使用tape.gradient方法计算了损失函数对于模型参数variables的梯度
            #tape是由tf.GradientTape()创建的上下文管理器，用于记录计算图中的操作并自动计算梯度。loss表示要计算梯度的目标张量
            # ，variables表示被求导的张量（即模型的参数）
            optimizer.apply_gradients(zip(grads,variables))
            #这一语句将梯度应用到模型的参数上，使用优化器optimizer对模型参数进行更新。
            # zip(grads,variables)将梯度和参数一一对应打包成元组，
            # 然后通过apply_gradients方法将它们应用到对应的参数上。

            if step % 100 ==0:
                print(epoch,step,'loss:',float(loss))
        #测试一下是否真的收敛
        total_num = 0
        total_correct = 0
        for x,y in test_db:
            out= conv_net(x)
            out=tf.reshape(out, [-1,512])
            logits=fc_net(out)
            prob=tf.nn.softmax(logits,axis=1)
            pred=tf.argmax(prob,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            correct =tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num+= x.shape[0]
            total_correct+=int(correct)
        acc = total_correct / total_num
        print(epoch,'acc: ',acc)

if __name__ == '__main__':
    main()









