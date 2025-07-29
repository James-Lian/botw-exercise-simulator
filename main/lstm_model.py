import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from keras import layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# maintain a consistent random seed for initializing ML model
tf.random.set_seed(8)
random.seed(8)

'''
BOTW Actions List

(Menu Navigation)
- pause (+)
- minus (-)
- L and R
- D-Pad
- camera poses?
Some of these might be hard-coded in actually... 


- interact (A)
--- create sub-forms (?)

(Navigation)
- walking
- sprinting
- whistle-sprinting
- crouch (hard-coded)
- scope (hard-coded)
- swimming
- jumping (hard-coded)

(Combat)
- lock-on/shielding (pose, hard-coded)
--- shield jump
--- shield surf
- dodging (flurry rush)
- parry
- attacking (melee)
--- jump attack (jumping is hard coded)
- special attack (charged, melee)
- aiming (bow)
- release (bow)
'''

# Build and train ML LSTM model
class ActionRecognitionModel:
    # input_shape=(10 time steps, 33 * 3 landmarks xyz coords + 1 timestamp)
    def __init__(self, num_classes=1, input_shape=(10,100), filename="", epochs=100, patience=-1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.model = self._build_model()
        self.history = None
        self.is_trained = False
        self.filename = filename
        self.epochs = epochs
        self.patience = patience

    def _build_model(self):
        model = keras.Sequential([
            # for each time step, Masking layer checks if ALL features = 0, if so, it will skip
            layers.Masking(mask_value=0.0, input_shape=self.input_shape),
            layers.LSTM(256, return_sequences=True), 
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64, return_sequences=True, dropout=0.1),
            layers.LSTM(32, dropout=0.1),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=['accuracy']
        )
        return model

    def prepare_training_data(self, raw_sequences, raw_labels):
        '''
        Args:
            raw_sequences: np array (sequences, timesteps, features)
            raw_labels: np array (sequences, classes)
        '''

        encoded_labels = self.label_encoder.fit_transform(raw_labels)

        # convert to categorical for softmax output
        # .to_categorical() acts like onehotencoder, where only 1 category is 'hot' and all others are 'cold'
        categorical_labels = to_categorical(encoded_labels, num_classes=self.num_classes)

        X = np.stack(raw_sequences)
        y = categorical_labels

        return X, y, encoded_labels

    def train(self, sequences, labels, validation_split=0.2, epochs=None, batch_size=32):
        '''
        Train the action recognition model.
        '''
        if epochs == None:
            epochs = self.epochs

        X, y, encoded_labels = self.prepare_training_data(sequences, labels)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=True, random_state=42
        )

        checkpoint = callbacks.ModelCheckpoint(
            filepath=self.filename + '_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        if self.patience > 0:
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=20, 
                restore_best_weights=True
            )

            self.history = self.model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stop, checkpoint]
            )
        else: 
            self.history = self.model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[checkpoint]
            )

        self.is_trained = True

        return self.history, X_val, y_val

    def predict(self, X):
        '''
        Make predictions on input data.
        '''
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions.")
        
        probabilities = self.model.predict(X, verbose=0)
        predicted_indices = np.argmax(probabilities, axis=1)
        # maximma of a function, or in this case, the category with the highest probability
        predicted_classes = self.label_encoder.inverse_transform(predicted_indices)
        
        return predicted_classes, probabilities

    def evaluate(self, X_test, y_test):
        '''
        Evaluate model performance.
        '''
        if not self.is_trained:
            raise Exception("Model must be trained before evaluation.")
        
        predicted_classes, probabilities = self.predict(X_test)
        for i in range(len(predicted_classes)):
            print("Predictions + Probabilities!")
            print(predicted_classes[i], probabilities[i])
            print()

        # convert one-hot encoded labels back to class indices
        true_indices = np.argmax(y_test, axis=1)
        true_classes = self.label_encoder.inverse_transform(true_indices)

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        report = classification_report(true_classes, predicted_classes, 
                                       target_names=self.label_encoder.classes_, 
                                       output_dict=True)

        cm = confusion_matrix(true_classes, predicted_classes)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def plot_training_history(self):
        if self.history is None:
            raise Exception("No training history available.")
        
        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, evaluation_results):
        cm = evaluation_results['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_model(self, filepath=""):
        if not self.is_trained:
            raise Exception("Model must be trained before saving.")
        
        if filepath == "":
            filepath = self.filename + ".keras"
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, label_filepath=None):
        self.model = keras.models.load_model(filepath)
        if label_filepath != None:
            with open(label_filepath, "rb") as f:
                self.label_encoder = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
