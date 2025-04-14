import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('trained_model.h5')

# Remove specified features
features_to_remove = [88, 92, 98, 209, 447, 454, 545, 677, 722, 743, 763, 790, 804, 817, 893, 1014]

# Correct the orientation of the data
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

def preprocess_data(data):
    data = np.delete(data, features_to_remove, axis=1)
    data[:, :-1][data[:, :-1] > 255] = 255
    data[:, :-1] = data[:, :-1] / 255.0
    data = correct_orientation(data)
    data = apply_median_filter(data)
    return data

def main():
    # Read test data
    test_data = pd.read_csv("testdata.txt", header=None).values

    # Preprocess test data
    test_data = preprocess_data(test_data)
    
    # Reshape the data for CNN
    X_test = test_data[:, :1024].reshape(-1, 32, 32, 1)

    # Predict labels
    infer_labels = model.predict(X_test)
    infer_labels = np.argmax(infer_labels, axis=1)
    
    infer_labels = pd.DataFrame(infer_labels)
    
    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (test_data.shape[0], 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(test_data.shape[0], 1)}"
    
    infer_labels.to_csv("predlabels.txt", index=False, header=False)

if __name__ == "__main__":
    main()
