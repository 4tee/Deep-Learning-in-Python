# Last steps in classification models #

You'll now create a classification model using the titanic dataset, which has been pre-loaded into a DataFrame called `df`. You'll take information about the passengers and predict which ones survived.

The predictive variables are stored in a NumPy array `predictors`. The target to predict is in `df.survived`, though you'll have to manipulate it for keras. The number of predictive features is stored in `n_cols`.

Here, you'll use the `'sgd'` optimizer, which stands for [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). You'll learn more about this in the next chapter!

## Instructions ##

* Convert `df.survived` to a categorical variable using the `to_categorical()` function.
* Specify a `Sequential` model called `model`.
* Add a `Dense` layer with `32` nodes. Use `'relu'` as the `activation` and `(n_cols,)` as the `input_shape`.
* Add the `Dense` output layer. Because there are two outcomes, it should have 2 units, and because it is a classification model, the `activation` should be `'softmax'`.
* Compile the model, using `'sgd'` as the `optimizer`, `'categorical_crossentropy'` as the loss function, and `metrics=['accuracy']` to see the accuracy (what fraction of predictions were correct) at the end of each epoch.
* Fit the model using the `predictors` and the `target`.

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)
```

```
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 0s - loss: 6.9471 - acc: 0.5625
736/891 [=======================>......] - ETA: 0s - loss: 2.9928 - acc: 0.5720
891/891 [==============================] - 0s - loss: 2.7525 - acc: 0.5701     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.9492 - acc: 0.5625
672/891 [=====================>........] - ETA: 0s - loss: 1.3231 - acc: 0.5997
891/891 [==============================] - 0s - loss: 1.1409 - acc: 0.6308     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.4814 - acc: 0.8125
736/891 [=======================>......] - ETA: 0s - loss: 0.9095 - acc: 0.6332
891/891 [==============================] - 0s - loss: 0.8555 - acc: 0.6510     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.3219 - acc: 0.8438
736/891 [=======================>......] - ETA: 0s - loss: 0.6720 - acc: 0.6807
891/891 [==============================] - 0s - loss: 0.6722 - acc: 0.6689     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.5781 - acc: 0.6250
736/891 [=======================>......] - ETA: 0s - loss: 0.6107 - acc: 0.6739
891/891 [==============================] - 0s - loss: 0.6103 - acc: 0.6723     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.4816 - acc: 0.7812
768/891 [========================>.....] - ETA: 0s - loss: 0.5901 - acc: 0.7057
891/891 [==============================] - 0s - loss: 0.5888 - acc: 0.7048     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.9318 - acc: 0.4375
736/891 [=======================>......] - ETA: 0s - loss: 0.6132 - acc: 0.6957
891/891 [==============================] - 0s - loss: 0.6098 - acc: 0.6925     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.4446 - acc: 0.8438
736/891 [=======================>......] - ETA: 0s - loss: 0.6039 - acc: 0.6970
891/891 [==============================] - 0s - loss: 0.5979 - acc: 0.6970     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.5866 - acc: 0.7188
736/891 [=======================>......] - ETA: 0s - loss: 0.5596 - acc: 0.7201
891/891 [==============================] - 0s - loss: 0.5744 - acc: 0.7082     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.5388 - acc: 0.7812
736/891 [=======================>......] - ETA: 0s - loss: 0.5871 - acc: 0.6943
891/891 [==============================] - 0s - loss: 0.5837 - acc: 0.6958
```