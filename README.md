# Chicken-Breeds-Image-Classification
# The Trained Model is saved in Pickle, you can download and load it to use in your own script
Chicken Breeds Image Classification Project using CNN, data taken from kaggle 
link to file : https://www.kaggle.com/abdalnassir/chicken-breeds
* The dataset can be fetch using kaggle library on colab 
* 1000+ samples
* 80:20 pre train test split

## Library Used
* Tensorflow
* Keras
* matplotlib
* Numpy
* Pandas
* Kaggle



### Display random images to examine

![](/img/sample.png)


### Rescaling Image
### Image Augmenting (train set only)
* Rotation
* Horizontal flip
* Zoom range
* Shear
* Fill mode nearest
* Data generating
* resizing image to 60x60
* default batch size 
* class mode categorical

### Build Convnet
* 1 hidden layer (perceptron 128 units)
* output layer 'Softmax'
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (128,128,3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64,(3,3), activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128,(3,3), activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation= 'relu'),
    layers.Dense(5, activation= 'softmax') 
])


### Adding Loss Function and Optimizer
* loss : categorical cross entropy
* optimizer adam
* metrics accuracy

### Adding Custom Early stop function using Callback from tensorflow 
* Model will stop training once val accuracy reaches atleast 85%


### Training the model
* epoch : 40
* verbose : 1
* callbacks
 
## Results
* Best Validation Loss: 0.41
* Best Validation Accuracy: 0.85

* Loss

![](/img/loss.png)

* Accuracy

![](/img/accuracy.png)

