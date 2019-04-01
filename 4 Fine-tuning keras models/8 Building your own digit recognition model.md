# Building your own digit recognition model #

You've reached the final exercise of the course - you now know everything you need to build an accurate model to recognize handwritten digits!

We've already done the basic manipulation of the MNIST dataset shown in the video, so you have `X` and `y` loaded and ready to model with. `Sequential` and `Dense` from keras are also pre-imported.

To add an extra challenge, we've loaded only 2500 images, rather than 60000 which you will see in some published results. Deep learning models perform better with more data, however, they also take longer to train, especially when they start becoming more complex.

If you have a computer with a CUDA compatible GPU, you can take advantage of it to improve computation time. If you don't have a GPU, no problem! You can set up a deep learning environment in the cloud that can run your models on a GPU. Here is a [blog post](https://www.datacamp.com/community/tutorials/deep-learning-jupyter-aws) by Dan that explains how to do this - check it out after completing this exercise! It is a great next step as you continue your deep learning journey.

Ready to take your deep learning to the next level? Check out [Advanced Deep Learning with Keras in Python](https://www.datacamp.com/courses/advanced-deep-learning-with-keras-in-python) to see how the Keras functional API lets you build domain knowledge to solve new types of problems. Once you know how to use the functional API, take a look at ["Convolutional Neural Networks for Image Processing"](https://www.datacamp.com/courses/convolutional-neural-networks-for-image-processing) to learn image-specific applications of Keras.

## Instructions ##

* Create a `Sequential` object to start your model. Call this `model`.
* Add the first `Dense` hidden layer of `50` units to your model with `'relu'` activation. For this data, the `input_shape` is `(784,)`.
* Add a second `Dense` hidden layer with `50` units and a `'relu'` activation function.
* Add the output layer. Your activation function should be `'softmax'`, and the number of nodes in this layer should be the same as the number of possible outputs in this case: `10`.
* Compile `model` as you have done with previous models: Using `'adam'` as the `optimizer`, `'categorical_crossentropy'` for the `loss`, and `metrics=['accuracy']`.
* Fit the model using `X` and `y` using a `validation_split` of `0.3`.

```python
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X,y,validation_split=0.3)
```

```
    Train on 1750 samples, validate on 750 samples
    Epoch 1/10
    
  32/1750 [..............................] - ETA: 2s - loss: 2.3488 - acc: 0.0312
 352/1750 [=====>........................] - ETA: 0s - loss: 2.1773 - acc: 0.2472
 800/1750 [============>.................] - ETA: 0s - loss: 1.9990 - acc: 0.3600
1280/1750 [====================>.........] - ETA: 0s - loss: 1.7839 - acc: 0.4602
1750/1750 [==============================] - 0s - loss: 1.5936 - acc: 0.5269 - val_loss: 0.8763 - val_acc: 0.7653
    Epoch 2/10
    
  32/1750 [..............................] - ETA: 0s - loss: 1.0659 - acc: 0.6250
 544/1750 [========>.....................] - ETA: 0s - loss: 0.7339 - acc: 0.8088
1056/1750 [=================>............] - ETA: 0s - loss: 0.6747 - acc: 0.8210
1568/1750 [=========================>....] - ETA: 0s - loss: 0.6426 - acc: 0.8214
1750/1750 [==============================] - 0s - loss: 0.6267 - acc: 0.8263 - val_loss: 0.5099 - val_acc: 0.8693
    Epoch 3/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.4156 - acc: 0.8750
 512/1750 [=======>......................] - ETA: 0s - loss: 0.4015 - acc: 0.8848
1024/1750 [================>.............] - ETA: 0s - loss: 0.4195 - acc: 0.8760
1536/1750 [=========================>....] - ETA: 0s - loss: 0.4120 - acc: 0.8783
1750/1750 [==============================] - 0s - loss: 0.4120 - acc: 0.8794 - val_loss: 0.4162 - val_acc: 0.8773
    Epoch 4/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.2290 - acc: 0.9375
 512/1750 [=======>......................] - ETA: 0s - loss: 0.3512 - acc: 0.9023
 992/1750 [================>.............] - ETA: 0s - loss: 0.3304 - acc: 0.9042
1504/1750 [========================>.....] - ETA: 0s - loss: 0.3183 - acc: 0.9109
1750/1750 [==============================] - 0s - loss: 0.3186 - acc: 0.9074 - val_loss: 0.3951 - val_acc: 0.8840
    Epoch 5/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.0926 - acc: 1.0000
 544/1750 [========>.....................] - ETA: 0s - loss: 0.2701 - acc: 0.9246
1056/1750 [=================>............] - ETA: 0s - loss: 0.2574 - acc: 0.9252
1568/1750 [=========================>....] - ETA: 0s - loss: 0.2713 - acc: 0.9177
1750/1750 [==============================] - 0s - loss: 0.2715 - acc: 0.9177 - val_loss: 0.3761 - val_acc: 0.8787
    Epoch 6/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.3231 - acc: 0.9375
 512/1750 [=======>......................] - ETA: 0s - loss: 0.2296 - acc: 0.9336
1024/1750 [================>.............] - ETA: 0s - loss: 0.2236 - acc: 0.9414
1536/1750 [=========================>....] - ETA: 0s - loss: 0.2180 - acc: 0.9382
1750/1750 [==============================] - 0s - loss: 0.2176 - acc: 0.9406 - val_loss: 0.3555 - val_acc: 0.8987
    Epoch 7/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.1352 - acc: 0.9688
 544/1750 [========>.....................] - ETA: 0s - loss: 0.1575 - acc: 0.9669
1056/1750 [=================>............] - ETA: 0s - loss: 0.1683 - acc: 0.9602
1568/1750 [=========================>....] - ETA: 0s - loss: 0.1779 - acc: 0.9547
1750/1750 [==============================] - 0s - loss: 0.1833 - acc: 0.9543 - val_loss: 0.3776 - val_acc: 0.8933
    Epoch 8/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.1476 - acc: 0.9688
 544/1750 [========>.....................] - ETA: 0s - loss: 0.1590 - acc: 0.9651
1056/1750 [=================>............] - ETA: 0s - loss: 0.1584 - acc: 0.9650
1568/1750 [=========================>....] - ETA: 0s - loss: 0.1566 - acc: 0.9649
1750/1750 [==============================] - 0s - loss: 0.1559 - acc: 0.9646 - val_loss: 0.3676 - val_acc: 0.8893
    Epoch 9/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.1720 - acc: 0.9688
 544/1750 [========>.....................] - ETA: 0s - loss: 0.1334 - acc: 0.9743
1056/1750 [=================>............] - ETA: 0s - loss: 0.1239 - acc: 0.9744
1568/1750 [=========================>....] - ETA: 0s - loss: 0.1322 - acc: 0.9719
1750/1750 [==============================] - 0s - loss: 0.1286 - acc: 0.9737 - val_loss: 0.3488 - val_acc: 0.8880
    Epoch 10/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.0830 - acc: 1.0000
 544/1750 [========>.....................] - ETA: 0s - loss: 0.1020 - acc: 0.9816
1056/1750 [=================>............] - ETA: 0s - loss: 0.1019 - acc: 0.9811
1568/1750 [=========================>....] - ETA: 0s - loss: 0.1061 - acc: 0.9802
1750/1750 [==============================] - 0s - loss: 0.1077 - acc: 0.9806 - val_loss: 0.3669 - val_acc: 0.8973
```

> You've done something pretty amazing. You should see better than 90% accuracy recognizing handwritten digits, even while using a small training set of only 1750 images!