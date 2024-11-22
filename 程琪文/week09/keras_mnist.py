from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据归一化
train_images = train_images.reshape(train_images.shape[0], 28 * 28)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape(test_images.shape[0], 28 * 28)
test_images = test_images.astype('float32') / 255.0

# one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建模型
network = Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(784,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译
# binary_crossentropy、categorical_crossentropy、sparse_categorical_crossentropy
network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))

res = network.predict(digit.reshape((1, 28 * 28)))
print(res)

# for i in range(res[1].shape[0]):
#     if (res[1][i] == 1):
#         print("the number for the picture is : ", i)
#         break
