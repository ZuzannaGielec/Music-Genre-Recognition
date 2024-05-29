import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import seaborn as sns

# Function to load audio files from a directory and trim/pad to 27 seconds
def load_audio_files(directory, target_duration=27, sr=22050):
    audio_data = []
    labels = []
    for genre in os.listdir(directory):
        genre_path = os.path.join(directory, genre)
        for filename in os.listdir(genre_path):
            filepath = os.path.join(genre_path, filename)
            try:
                audio, _ = librosa.load(filepath, sr=sr, duration=target_duration)
                audio_data.append(audio)
                labels.append(genre)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return np.array(audio_data), np.array(labels)

# Function to split data into training and testing sets
def split_train_test_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Function to divide audio into 3-second segments
def divide_audio(audio, segment_length=3, sr=22050):
    segments = []
    total_length = len(audio)
    num_segments = int(total_length / (segment_length * sr))
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        segments.append(segment)
    return segments

# Function to create spectrograms from audio segments
def create_spectrograms(audio_data, labels, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, (audio, genre) in enumerate(zip(audio_data, labels)):
        segments = divide_audio(audio)
        for j, segment in enumerate(segments):
            plt.figure(figsize=(2, 2))
            S = librosa.feature.melspectrogram(y=segment, sr=22050)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            plt.axis('off')
            spectrogram_filename = f'spectrogram_{i}_{j}_{genre}.png'
            plt.savefig(os.path.join(save_path, spectrogram_filename), bbox_inches='tight', pad_inches=0)
            plt.close()

# Function to build a CNN model for music genre classification
def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load spectrograms as numpy arrays
def load_spectrograms(spectrogram_directory, target_size=(200, 200)):
    spectrograms = []
    labels = []
    for filename in os.listdir(spectrogram_directory):
        if filename.endswith('.png'):
            img_path = os.path.join(spectrogram_directory, filename)
            img = load_img(img_path, target_size=target_size, color_mode='grayscale')
            img_array = img_to_array(img)
            spectrograms.append(img_array)
            label = filename.split('_')[-1].split('.')[0]  # Extracting genre from filename
            labels.append(label)
    return np.array(spectrograms), np.array(labels)

# Function to plot accuracy and validation accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Main function for music genre classification
def music_genre_classification():
    # Load original GTZAN dataset
    data_directory = 'C:/Users/czerw/Documents/Machine Learning/genres_original'
    audio_data, labels = load_audio_files(data_directory)
    
    print(f"Total audio files: {len(audio_data)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test_data(audio_data, labels)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # Create spectrograms and save to folders
    spectrogram_save_path_train = 'spectrograms3sec_train'
    spectrogram_save_path_test = 'spectrograms3_test'
    create_spectrograms(X_train, y_train, spectrogram_save_path_train)
    create_spectrograms(X_test, y_test, spectrogram_save_path_test)

    # Load spectrograms back as numpy arrays
    target_size = (200, 200)  # Ensure this matches the size of your spectrogram images
    X_train_spectrograms, y_train_spectrograms = load_spectrograms(spectrogram_save_path_train, target_size=target_size)
    X_test_spectrograms, y_test_spectrograms = load_spectrograms(spectrogram_save_path_test, target_size=target_size)
    
    print(f"Training spectrograms: {X_train_spectrograms.shape}")
    print(f"Testing spectrograms: {X_test_spectrograms.shape}")
    print(f"Training labels: {len(y_train_spectrograms)}")
    print(f"Testing labels: {len(y_test_spectrograms)}")

    # Ensure consistent shape for input data
    X_train_spectrograms = X_train_spectrograms.reshape(X_train_spectrograms.shape[0], target_size[0], target_size[1], 1)
    X_test_spectrograms = X_test_spectrograms.reshape(X_test_spectrograms.shape[0], target_size[0], target_size[1], 1)

    # Convert labels to categorical format
    unique_labels = np.unique(y_train_spectrograms)
    label_map = {label: index for index, label in enumerate(unique_labels)}
    y_train_indices = np.array([label_map[label] for label in y_train_spectrograms])
    y_test_indices = np.array([label_map[label] for label in y_test_spectrograms])

    num_classes = len(unique_labels)
    y_train_categorical = to_categorical(y_train_indices, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_indices, num_classes=num_classes)

    # Build CNN model
    input_shape = (target_size[0], target_size[1], 1)  # Adjust dimensions based on spectrogram size
    model = build_cnn_model(input_shape, num_classes)

    # Train the model
    history = model.fit(X_train_spectrograms, y_train_categorical, epochs=30, batch_size=32, validation_split=0.2)

    # Plot accuracy and validation accuracy
    plot_accuracy(history)

    # Evaluate the model on the test set
    y_pred = np.argmax(model.predict(X_test_spectrograms), axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)

    # Print accuracy and confusion matrix
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, unique_labels)

# Run the music genre classification
music_genre_classification()

