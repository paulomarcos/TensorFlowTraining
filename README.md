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
