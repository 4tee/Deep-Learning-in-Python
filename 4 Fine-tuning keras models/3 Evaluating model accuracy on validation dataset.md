# Evaluating model accuracy on validation dataset #

Now it's your turn to monitor model accuracy with a validation data set. A model definition has been provided as `model`. Your job is to add the code to compile it and then fit it. You'll check the validation score in each epoch.

## Instructions ##

* Compile your model using `'adam'` as the `optimizer` and `'categorical_crossentropy'` for the `loss`. To see what fraction of predictions are correct (the `accuracy`) in each epoch, specify the additional keyword argument `metrics=['accuracy']` in `model.compile()`.
* Fit the model using the `predictors` and `target`. Create a validation split of 30% (or `0.3`). This will be reported in each epoch.

```python
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

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)
```

```
    Train on 623 samples, validate on 268 samples
    Epoch 1/10
    
 32/623 [>.............................] - ETA: 0s - loss: 3.3028 - acc: 0.4062
623/623 [==============================] - 0s - loss: 1.3112 - acc: 0.6003 - val_loss: 0.6817 - val_acc: 0.7201
    Epoch 2/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6875 - acc: 0.7188
623/623 [==============================] - 0s - loss: 0.8805 - acc: 0.5714 - val_loss: 1.1021 - val_acc: 0.6418
    Epoch 3/10
    
 32/623 [>.............................] - ETA: 0s - loss: 1.0322 - acc: 0.5938
623/623 [==============================] - 0s - loss: 0.7983 - acc: 0.6276 - val_loss: 0.8588 - val_acc: 0.6343
    Epoch 4/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6472 - acc: 0.6875
623/623 [==============================] - 0s - loss: 0.7531 - acc: 0.6517 - val_loss: 0.6877 - val_acc: 0.7015
    Epoch 5/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6888 - acc: 0.6250
623/623 [==============================] - 0s - loss: 0.6756 - acc: 0.6421 - val_loss: 0.5783 - val_acc: 0.7164
    Epoch 6/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5663 - acc: 0.6875
623/623 [==============================] - 0s - loss: 0.6562 - acc: 0.6485 - val_loss: 0.5250 - val_acc: 0.7500
    Epoch 7/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5533 - acc: 0.7500
623/623 [==============================] - 0s - loss: 0.5994 - acc: 0.6806 - val_loss: 0.5103 - val_acc: 0.7201
    Epoch 8/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6073 - acc: 0.7500
623/623 [==============================] - 0s - loss: 0.5909 - acc: 0.6902 - val_loss: 0.5229 - val_acc: 0.7463
    Epoch 9/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5572 - acc: 0.7188
623/623 [==============================] - 0s - loss: 0.6621 - acc: 0.6613 - val_loss: 0.5711 - val_acc: 0.7015
    Epoch 10/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.4853 - acc: 0.7812
623/623 [==============================] - 0s - loss: 0.6123 - acc: 0.6886 - val_loss: 0.5309 - val_acc: 0.7351
```