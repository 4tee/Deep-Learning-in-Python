# Fitting the model #

You're at the most fun part. You'll now fit the model. Recall that the data to be used as predictive features is loaded in a NumPy matrix called `predictors` and the data to be predicted is stored in a NumPy matrix called `target`. Your `model` is pre-written and it has been compiled with the code from the previous exercise.

## Instructions ##

* Fit the `model`. Remember that the first argument is the predictive features (`predictors`), and the data to be predicted (`target`) is the second argument.

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)
```

```
    Epoch 1/10
    
 32/534 [>.............................] - ETA: 0s - loss: 125.9789
534/534 [==============================] - 0s - loss: 53.8696      
    Epoch 2/10
    
 32/534 [>.............................] - ETA: 0s - loss: 22.8705
534/534 [==============================] - 0s - loss: 26.0802     
    Epoch 3/10
    
 32/534 [>.............................] - ETA: 0s - loss: 15.8384
534/534 [==============================] - 0s - loss: 22.1124     
    Epoch 4/10
    
 32/534 [>.............................] - ETA: 0s - loss: 22.5975
534/534 [==============================] - 0s - loss: 21.9797     
    Epoch 5/10
    
 32/534 [>.............................] - ETA: 0s - loss: 10.3009
534/534 [==============================] - 0s - loss: 20.9924     
    Epoch 6/10
    
 32/534 [>.............................] - ETA: 0s - loss: 11.0671
534/534 [==============================] - 0s - loss: 21.1918     
    Epoch 7/10
    
 32/534 [>.............................] - ETA: 0s - loss: 10.9625
534/534 [==============================] - 0s - loss: 21.2215     
    Epoch 8/10
    
 32/534 [>.............................] - ETA: 0s - loss: 18.0066
534/534 [==============================] - 0s - loss: 21.1707     
    Epoch 9/10
    
 32/534 [>.............................] - ETA: 0s - loss: 20.1527
534/534 [==============================] - 0s - loss: 20.6593     
    Epoch 10/10
    
 32/534 [>.............................] - ETA: 0s - loss: 20.0096
534/534 [==============================] - 0s - loss: 20.5022
```