import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import joblib

# Load the trained model
model = joblib.load("voice_classifier_model.pkl")

# Function to predict gender from an audio file


def predict_gender():
    # Open a file dialog to select an audio file
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav")])

    if file_path:
        try:
            # Extract audio features from the selected file
            audio_features = extract_features(file_path)

            # Predict the gender using the loaded model
            gender = model.predict([audio_features])[0]

            # Display the result
            result_label.config(
                text=f"Predicted Gender: {'Male' if gender == 0 else 'Female'}")
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")
    else:
        result_label.config(text="No file selected.")

# Function to extract audio features from a file


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr)
    # Calculate the mean along columns
    mean_mfcc = np.mean(mfcc_features, axis=1)
    return mean_mfcc


# Create the main window
root = tk.Tk()
root.title("Gender Identification from Audio")

# Create and configure widgets
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

select_button = tk.Button(
    frame, text="Select Audio File", command=predict_gender)
select_button.pack()

result_label = tk.Label(frame, text="", font=("Helvetica", 14))
result_label.pack()

# Run the main GUI event loop
root.mainloop()
