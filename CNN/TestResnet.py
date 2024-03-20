import tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets,Sequential
import os

from Reset1 import resnet18


import warnings
warnings.filterwarnings('ignore')



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

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
    model=resnet18()
    model.build(input_shape=(None,32,32,3))
    #优化器
    optimizer=optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:

                logits=model(x)
                #[b] ->[b,100]
                #要以管理员的方式运行软件
                y_onehot= tf.one_hot(y,depth=100)
                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            if step % 100 ==0:
                print(epoch,step,'loss:',float(loss))
        #测试一下是否真的收敛
        total_num = 0
        total_correct = 0
        for x,y in test_db:

            logits=model(x)
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









