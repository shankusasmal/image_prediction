import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
(xtr,ytr),(xte,yte)=keras.datasets.mnist.load_data()
len(xtr)
xtr.shape
ytr.shape
print(xte.shape)
xt=xtr/255
xtes=xte/255
x=xt.reshape(len(xt),28*28)
xtest=xte.reshape(10000,28*28)
a=keras.Sequential([keras.layers.Dense(100,input_shape=(784,),activation="relu"),
                    keras.layers.Dense(10,activation="sigmoid")
                   ])
a.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
a.fit(x,ytr,epochs=5)
a.evaluate(xtest,yte)
b=a.predict(xtest)
b.shape
c=[np.argmax(i) for i in b]
print((b[1201]),yte[1201])
import math
data = tf.math.confusion_matrix(yte,c)
sns.heatmap(data,annot=True,fmt='d')
plt.xlabel("predictions")
plt.ylabel("yte")
plt.show()
