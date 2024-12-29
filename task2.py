# MLP
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2
import os


# Load and preprocess the dataset  
def load_data(data_dir):  
    images = []  
    labels = []  
    shape_labels = os.listdir(data_dir)  # Get shape type folders  
    for label in shape_labels:  
        shape_folder = os.path.join(data_dir, label)  
        if os.path.isdir(shape_folder):  # Check if it's a directory  
            for img_file in os.listdir(shape_folder):  
                img_path = os.path.join(shape_folder, img_file)  
                img = cv2.imread(img_path)  
                img = cv2.resize(img, (100, 100))  # Resize to 64x64  
                images.append(img)  
                labels.append(label)  
    return np.array(images), np.array(labels)  

# Load dataset  
data_dir = 'C:/Users/Surface/Desktop/lessons/artificial_inteligence/tasks/task2/train'  
X, y = load_data(data_dir)  

# Normalize the images
X = X.astype('float32') / 255.0

# Encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Flatten the images for MLP
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Build the MLP model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(lb.classes_), activation='softmax')  # Output layer for number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# # Save the model
# model.save('shape_recognition_model.h5')


# Function to predict the shape of a single image  
def predict_shape(image_path, model, label_binarizer):  
    img = cv2.imread(image_path)  
    img = cv2.resize(img, (100, 100))  # Resize to match training  
    img = img.astype('float32') / 255.0  # Normalize  
    img = img.reshape(1, -1)  # Flatten the image  
    prediction = model.predict(img)  
    predicted_class = label_binarizer.inverse_transform(prediction)  
    return predicted_class[0]  

# Example usage  
image_to_predict = 'C:/Users/Surface/Desktop/lessons/artificial_inteligence/tasks/task2/star.png'  # Update with the path to the image you want to predict  
predicted_shape = predict_shape(image_to_predict, model, lb)  
print(f'The predicted shape is: {predicted_shape}')  
