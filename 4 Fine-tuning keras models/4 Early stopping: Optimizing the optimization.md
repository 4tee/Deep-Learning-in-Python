# Early stopping: Optimizing the optimization #

Now that you know how to monitor your model performance throughout optimization, you can use early stopping to stop optimization when it isn't helping any more. Since the optimization stops automatically when it isn't helping, you can also set a high value for `epochs` in your call to `.fit()`, as Dan showed in the video.

The model you'll optimize has been specified as `model`. As before, the data is pre-loaded as `predictors` and `target`.

## Instructions ##

* Import `EarlyStopping` from `keras.callbacks`.
* Compile the model, once again using `'adam'` as the `optimizer`, `'categorical_crossentropy'` as the loss function, and `metrics=['accuracy']` to see the accuracy at each epoch.
* Create an `EarlyStopping` object called `early_stopping_monitor`. Stop optimization when the validation loss hasn't improved for 2 epochs by specifying the `patience` parameter of `EarlyStopping()` to be `2`.
* Fit the model using the `predictors` and `target`. Specify the number of `epochs` to be `30` and use a validation split of `0.3`. In addition, pass `[early_stopping_monitor]` to the `callbacks` parameter.

```python
# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])
```

```
    Train on 623 samples, validate on 268 samples
    Epoch 1/30
    
 32/623 [>.............................] - ETA: 0s - loss: 5.6563 - acc: 0.4688
623/623 [==============================] - 0s - loss: 1.6352 - acc: 0.5666 - val_loss: 1.0811 - val_acc: 0.6530
    Epoch 2/30
    
 32/623 [>.............................] - ETA: 0s - loss: 1.8312 - acc: 0.4688
623/623 [==============================] - 0s - loss: 0.8320 - acc: 0.6067 - val_loss: 0.5691 - val_acc: 0.7313
    Epoch 3/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.8391 - acc: 0.6562
623/623 [==============================] - 0s - loss: 0.7152 - acc: 0.6501 - val_loss: 0.5296 - val_acc: 0.7575
    Epoch 4/30
    
 32/623 [>.............................] - ETA: 0s - loss: 1.0122 - acc: 0.6250
623/623 [==============================] - 0s - loss: 0.6786 - acc: 0.6693 - val_loss: 0.5248 - val_acc: 0.7276
    Epoch 5/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5494 - acc: 0.7188
608/623 [============================>.] - ETA: 0s - loss: 0.6740 - acc: 0.6464
623/623 [==============================] - 0s - loss: 0.6816 - acc: 0.6453 - val_loss: 0.6560 - val_acc: 0.6978
    Epoch 6/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.4547 - acc: 0.8438
384/623 [=================>............] - ETA: 0s - loss: 0.6312 - acc: 0.7214
623/623 [==============================] - 0s - loss: 0.6270 - acc: 0.7095 - val_loss: 0.5818 - val_acc: 0.7015
    Epoch 7/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6545 - acc: 0.6562
623/623 [==============================] - 0s - loss: 0.6546 - acc: 0.7014 - val_loss: 0.6737 - val_acc: 0.6679
```

> Because optimization will automatically stop when it is no longer helpful, it is okay to specify the maximum number of epochs as 30 rather than using the default of 10 that you've used so far. Here, it seems like the optimization stopped after 7 epochs.