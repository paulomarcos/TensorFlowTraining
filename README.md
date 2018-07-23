# TensorFlow Training
Files used for helping me understand how to implement different deep learning classifiers using TensorFlow.

### Iris Dataset
Used Keras inside TensorFlow to predict the labels for the IRIS dataset. After running *iris_eager.py* for 2000 epochs, I've got the following results:
```
Epoch 000: Loss: 1.095, Accuracy: 35.000%
Epoch 050: Loss: 0.465, Accuracy: 84.167%
Epoch 100: Loss: 0.273, Accuracy: 96.667%
Epoch 150: Loss: 0.189, Accuracy: 97.500%
Epoch 200: Loss: 0.136, Accuracy: 97.500%
Epoch 250: Loss: 0.106, Accuracy: 97.500%
...
Epoch 1850: Loss: 0.045, Accuracy: 98.333%
Epoch 1900: Loss: 0.048, Accuracy: 97.500%
Epoch 1950: Loss: 0.048, Accuracy: 99.167%
Epoch 2000: Loss: 0.046, Accuracy: 99.167%
Test set accuracy: 93.333%
Example 0 prediction: Iris setosa
Example 1 prediction: Iris versicolor
Example 2 prediction: Iris virginica
```

### Conv Nets with MNIST
The model is the following: INTPUT LAYER > CONV > POOL > CONV > POOL > DENSE > LOGITS > OUTPUT.
The results I have got using the convolutional neural networks on the MNIST dataset for the *cnn_mnist.py* file is displayed below:
```
INFO:tensorflow:Saving checkpoints for 34000 into /tmp/mnist_convnet_model/model.ckpt.
INFO:tensorflow:Loss for final step: 0.096133135.
INFO:tensorflow:Saving dict for global step 34000: accuracy = 0.9784, global_step = 34000, loss = 0.06860989
{'accuracy': 0.9784, 'loss': 0.06860989, 'global_step': 34000}
```

### ResNet50 in TensorFlow
Using Keras and TensorFlow based on deeplearning.ai's approach, we used it to train the Keio Cup Dataset (KCD) on the task of classifying a cup/glass to be filled with liquid, empty or opaque. The results on both Testing set and Training set are the following:
```
Epoch 60/64
3953/3953 [==============================] - 1258s 318ms/step - loss: 1.2368 - acc: 0.5932
Epoch 61/64
3953/3953 [==============================] - 1260s 319ms/step - loss: 0.9196 - acc: 0.6613
Epoch 62/64
3953/3953 [==============================] - 1261s 319ms/step - loss: 0.8096 - acc: 0.6969
Epoch 63/64
3953/3953 [==============================] - 1263s 319ms/step - loss: 0.7739 - acc: 0.7149
Epoch 64/64
3953/3953 [==============================] - 1264s 320ms/step - loss: 0.7375 - acc: 0.7435
```
And
```
1318/1318 [==============================] - 111s 85ms/step
Loss = 1.0352251507986296
Test Accuracy = 0.642640364278611
```
