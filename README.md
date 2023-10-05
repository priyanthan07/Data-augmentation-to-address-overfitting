# Data-augmentation-to-address-overfitting
This project mainly focused on ensuring how data augmentation reduces overfitting

## Required libraries
     tensorflow
     keras
     matplotlib
     numpy
     cv2
     PIL
     os
     pathlib
     
## Data preprocessing
     Data set's URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

     => resize
     => scaling
     => Augmentation
     
## Model without data augmentation

    num_classes  =5 
    model = models.Sequential([
          layers.Conv2D(16, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
      
          layers.Flatten(),
          layers.Dense(128, activation= 'relu'),
          layers.Dense(num_classes),
      ])

      model.compile( optimizer = "adam", 
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
                    metrics = ['accuracy'])
      model.fit(x_train_scaled,y_train , epochs = 3)

### Training
      loss: 0.7226 
      accuracy: 0.7309
      
### Evaluation
      loss: 0.9783 
      accuracy: 0.6199

Due to this overfitting, there is a variation in the values
## Data augmentation
      data_augmentation = models.Sequential([
          layers.experimental.preprocessing.RandomZoom(0.3),
          layers.experimental.preprocessing.RandomRotation(0.5),
          layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180,180,3)),
      ])

## Model with data augmentation
      num_classes  =5 
      model1 = models.Sequential([
          data_augmentation,
          layers.Conv2D(16, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3 , activation = 'relu', padding = 'same'),
          layers.MaxPooling2D(),
          layers.Dropout(0.2),
          layers.Flatten(),
          layers.Dense(128, activation= 'relu'),
          layers.Dense(num_classes),
      ])

      model1.compile( optimizer = "adam", 
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
                    metrics = ['accuracy'])
      model1.fit(x_train_scaled,y_train , epochs = 10)

### Training

    loss: 0.7757 
    accuracy: 0.6979
### Evaluation

    loss: 0.8950 
    accuracy: 0.6512

Using augmentation and dropout methods, the overfitting issue is minimized.
