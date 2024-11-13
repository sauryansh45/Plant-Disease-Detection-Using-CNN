import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("trainedmodel.h5")

# Define class labels
class_labels = {
    0: "Tomato_Septoria_leaf_spot",
    1: "Tomato_Tomato_mosaic_virus",
    2: "Tomato_Late_blight",
    3: "Tomato_Spider_mites_Two_spotted_spider_mite",
    4: "Tomato_Tomato_YellowLeaf__Curl_Virus",
    5: "Tomato_healthy",
    6: "Pepper__bell___Bacterial_spot",
    7: "Potato___healthy",
    8: "Tomato__Target_Spot",
    9: "Tomato_Leaf_Mold",
    10: "Tomato_Bacterial_spot",
    11: "Tomato_Early_blight",
    12: "Potato___Late_blight",
    13: "Potato___Early_blight",
    14: "Pepper__bell___healthy"
}

# Function to preprocess the image for prediction
def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to handle the prediction
def predict_disease(file_path):
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to open file dialog and get the selected file path
def open_file_dialog():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        update_image(file_path)

# Function to update the displayed image
def update_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))
    photo = ImageTk.PhotoImage(img)
    panel.config(image=photo)
    panel.image = photo
    prediction = predict_disease(file_path)
    predicted_class_label = class_labels.get(prediction, "Unknown")
    result_label.config(text=f"Predicted Plant: {predicted_class_label}")

# Create the main window
root = tk.Tk()
root.title("Plant Disease Prediction")

# Create and set up GUI elements
open_button = tk.Button(root, text="Open Image", command=open_file_dialog)
open_button.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Predicted Plant: None")
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
