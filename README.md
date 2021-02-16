# Flower Species Recognition
## Dataset Overview
The dataset consist of 5 species of flowers. There are about 1000 pictures of each species.

[Dataset Overview](https://imgur.com/Tq4jKrc)

## Defining CNN Models 
Different CNN models were used to compare. 

Single convolutional layer with 32 filters followed by Max Pooling layer was used for the first model. ReLu was used as activation function. The model was trained for 20 epochs.
```python
def define_model_one_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(5,activation='softmax'))
    
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```
After the model was trained, we had 56.9% accuracy score. And it seemed that there was overfitting.
[The line plot of 1 block model](https://imgur.com/ktgQGFA) (Blue lines are train, orange lines are test on line plot.)

The other model has 2 convolutional layers with 32 and 64 filters and Max Pooling layers.
```python
def define_model_two_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(5,activation='softmax'))
    
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```
We had 59.8% accuracy score at the end. There was a little bit increase but still overfitting.
[The line plot of 2 block model](https://imgur.com/Zn05Daq)

The other model has 3 convolutional layers with 32,64 and 128 filters.
```python
def define_model_three_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(5,activation='softmax'))
    
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```
We had 62.2% accuracy score, when the third model finished.
[The line plot of 3 block model](https://imgur.com/D81Jmn4)

We can say that convolutional layers, which detects the features of pictures, improves model performance. But we should fix the overfitting problem. We can say here that the reason of overfitting is small dataset. 
## Data Augmentation
Data augmentation increases the amount of data with applying small changes to pictures as Rotation,Flipping,Scaling.

Data Augmentation was applied in run_model function.
```python
    datagen = ImageDataGenerator(rescale=1.0/255.0)
```
Datagen was seperated as training and testing datagen. Small changes as flipping and shifting were applied to training datagen. 
```python
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
              width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
```
The last trained model with 3 convolutional layers were trained again for 50 epochs.

After training, we got 70.2% accuracy score. 
[The line plot of the model](https://imgur.com/WKhZhaT)
70.2% isn't enough. To improve performance, transfer learning was used in the project.
## Transfer Learning
Transfer learning is a machine learning method that uses pre-trained models. We can have faster and better models with less dataset with transfer learning. Because the pre-trained models that we use were trained with large datasets and the weights of neural networks were saved. 

VGG16 model, which was the winner of ImageNet Classification Contest in 2014, was used as transfer learning model. 

The model is comprised of two main parts, the feature extractor part of the model that is made up of VGG blocks, and the classifier part of the model that is made up of fully connected layers and the output layer. Feature extractor part was used. 
[Parts of VGG16](https://imgur.com/UZr5yJp)
The model was trained for 10 epochs because the weights were certain.
 ```python
def define_model_vgg_16():
    # load model
    model = VGG16(include_top=False,input_shape=(224,224,3))
    # mark loaded layers as not trainable
    for layer in model.layers:
      layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128,activation='relu',kernel_initializer='he_uniform')(flat1)
    output = Dense(5,activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs = output)
    #compile model
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```
At the end, we had 82% accuracy score. That was the better than other models. So the last model was saved and converted to tflite file to run on Flutter.
[The line plot of VGG16 model](https://imgur.com/zNJNO19)

## Flutter Application
Application has a screen with floating action button that picks image from gallery. Model is run in background with runModelOnImage function from Tensorflow Lite library.

### Running Model on Application
[Rose, ](https://imgur.com/twY4nKv)
[Daisy, ](https://imgur.com/i5zF2n2)
[Tulips, ](https://imgur.com/CttYgKy)
[Dandelion, ](https://imgur.com/Iux6FsQ)
[Sunflowers ](https://imgur.com/yHhIle1)
