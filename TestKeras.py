import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

class Model():
    def __init__(self):
        self.model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        self.model.add(Dense(64, activation='relu', input_dim=20))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.sgd,
                           metrics=['accuracy'])
    def fit(self,x_train, y_train):
        self.model.fit(x_train, y_train,epochs=20,batch_size=128)
    def predict(self,x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

model = Model()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = np.sum(y_test==y_pred)/len(y_test)
print(acc)
