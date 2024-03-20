import tensorflow as tf
from tensorflow.keras import datasets,layers,Sequential,models
import matplotlib.pyplot as plt



(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#归一化
train_images,test_images = train_images / 255.0 , test_images / 255.0

plt.figure(figsize=(20,10))

for i in range(20):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()


model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPool2D((2,2)),


    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)


])
print(model.summary())

#设置优化器，损失函数以及metrics
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

#训练模型
#这里设置输入训练数据集（图片及标签）、验证数据集（图片及标签）以及迭代次数epochs
history = model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))
