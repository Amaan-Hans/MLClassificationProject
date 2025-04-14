import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import median_filter

# Load data
trainDataFile = 'traindata.txt'
labelsFile = 'trainlabels.txt'
testDataFile = 'testdata.txt'
targetLabelsFile = 'targetlabels.txt'

# Read training data
with open(trainDataFile, 'r') as file:
    lines = file.readlines()
    matrix = [list(map(float, line.strip().split(','))) for line in lines]

# Read labels
with open(labelsFile, 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]

# Convert to numpy arrays
trainingData = np.array(matrix)
trainLabels = np.array(values)

# Remove specified features
features_to_remove = [88, 92, 98, 209, 447, 454, 545, 677, 722, 743, 763, 790, 804, 817, 893, 1014]
trainingData = np.delete(trainingData, features_to_remove, axis=1)

trainingData[:, :-1][trainingData[:, :-1] > 255] = 255

# Normalize the data (excluding the last column)
trainingData[:, :-1] = trainingData[:, :-1] / 255.0

# Correct the orientation
def correct_orientation(data):
    corrected_data = []
    for row in data:
        orientation = int(row[-1])
        image = row[:1024].reshape(32, 32)
        if orientation == 1:
            image = np.rot90(image, 3)  # 90 degrees clockwise
        elif orientation == 2:
            image = np.rot90(image, 2)  # 180 degrees
        elif orientation == 3:
            image = np.rot90(image, 1)  # 90 degrees counterclockwise
        corrected_row = np.concatenate((image.flatten(), row[1024:-1]))
        corrected_data.append(corrected_row)
    return np.array(corrected_data)

# Apply median filter to reduce noise
def apply_median_filter(data, size=3):
    filtered_data = []
    for row in data:
        image = row[:1024].reshape(32, 32)
        filtered_image = median_filter(image, size=size)
        filtered_row = np.concatenate((filtered_image.flatten(), row[1024:]))
        filtered_data.append(filtered_row)
    return np.array(filtered_data)

# Correct orientation and reduce noise
trainingData = correct_orientation(trainingData)
trainingData = apply_median_filter(trainingData)

# Reshape the data for CNN
X = trainingData[:, :1024].reshape(-1, 32, 32, 1)
y = trainLabels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
num_classes = 21  # Assuming labels range from 0 to 20
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Learning rate scheduler and early stopping
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train_categorical, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, y_test_categorical),
                    callbacks=[lr_reduction, early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Read test data
with open(testDataFile, 'r') as file:
    lines = file.readlines()
    matrix = [list(map(float, line.strip().split(','))) for line in lines]

# Read target labels
with open(targetLabelsFile, 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]

# Convert to numpy arrays
testData = np.array(matrix)
targetLabels = np.array(values)

# Remove specified features from test data
testData = np.delete(testData, features_to_remove, axis=1)

testData[:, :-1][testData[:, :-1] > 255] = 255

# Normalize the test data (excluding the last column)
testData[:, :-1] = testData[:, :-1] / 255.0

# Correct the orientation for test data
testData = correct_orientation(testData)
testData= apply_median_filter(testData)

# Reshape the test data for CNN
X_test_additional = testData[:, :1024].reshape(-1, 32, 32, 1)
y_test_additional = targetLabels

# Convert target labels to categorical (one-hot encoding)
y_test_additional_categorical = to_categorical(y_test_additional, num_classes)

# Evaluate the model on additional test data
loss, accuracy = model.evaluate(X_test_additional, y_test_additional_categorical)
print(f"Additional Test Data Accuracy: {accuracy * 100:.2f}%")

# Display the first 5 images with the trainlabel 5
def display_images(data, labels, label_value, num_images=5):
    images = data[labels == label_value][:num_images]
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, img in enumerate(images):
        img = img[:1024].reshape(32, 32)  # reshape to 32x32
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.show()

display_images(trainingData, trainLabels, label_value=5, num_images=5)
model.save('trained_model.keras')