import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

dataset = load_digits()

dataset.target

dataset.data.shape

dataset.target.shape

dataset.images.shape

dataset.images

df2 = dataset.images.reshape(1,-1)
df2.shape

len(dataset.images)

# Check the nth image in dataset
n=2

plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]

X = dataset.images.reshape((len(dataset.images), -1))
X.shape

Y =  dataset.target
Y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
print(y_train.shape)
print(X_test.shape)

from sklearn import svm
model = svm.SVC(gamma=0.001)
model.fit(X_train,y_train)

n = 1
result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n],cmap=plt.cm.gray_r,interpolation='nearest')
print(result)

print("\n")

plt.axis("off")
plt.title('%i' %result)
plt.show()

