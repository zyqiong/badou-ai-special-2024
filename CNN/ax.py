import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers,optimizers,datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(2345)

conv_layers=[
    #unit1
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'),
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

]

fc_layers=[

    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(100,activation='relu')

]

#漏写
(x,y),(x_test,y_test)=datasets.cifar100.load_data()

def progress(x,y):
    x=tf.cast(x,dtype=tf.float32) / 255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(progress).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(progress).batch(64)

next(iter(train_db))


optimizer=optimizers.Adam(lr=1e-4)



def main():

    conv_net=Sequential(conv_layers)
    fc_net=Sequential(fc_layers)
    #漏写

    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None,512])

    variables=conv_net.trainable_variables+fc_net.trainable_variables



    for epoch in range(50):
        for step,x,y in enumerate(train_db):
            with tf.GradientTape as tape:
                out=conv_net(x)
                out=tf.reshape(out,[-1,512])
                logits=fc_net(out)

                y_onehot=tf.one_hot(y,depth=100)

                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))

        if step %100==0:
            print(step,epoch,'loss:',loss)







