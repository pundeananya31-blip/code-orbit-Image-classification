import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
import numpy as np
import os


# =========================================================
# LOAD PRETRAINED MODEL
# =========================================================

try:
    model = MobileNetV2(weights="imagenet")
except Exception as e:
    print("Model loading error:", e)
    model = None


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()
root.title("AI Image Classification")
root.geometry("900x700")
root.configure(bg="#F4F7FA")
root.resizable(False, False)


# =========================================================
# VARIABLES
# =========================================================

selected_image_path = None
image_on_screen = None


# =========================================================
# TITLE
# =========================================================

title = tk.Label(
    root,
    text="AI IMAGE CLASSIFICATION",
    font=("Arial", 26, "bold"),
    bg="#F4F7FA",
    fg="#173B57"
)
title.pack(pady=(25, 5))


subtitle = tk.Label(
    root,
    text="Using Pretrained MobileNetV2 Model",
    font=("Arial", 13),
    bg="#F4F7FA",
    fg="#607D8B"
)
subtitle.pack(pady=(0, 20))


# =========================================================
# IMAGE PREVIEW FRAME
# =========================================================

preview_frame = tk.Frame(
    root,
    bg="white",
    width=500,
    height=320,
    highlightbackground="#D5DDE5",
    highlightthickness=2
)

preview_frame.pack(pady=10)
preview_frame.pack_propagate(False)


preview_label = tk.Label(
    preview_frame,
    text="No Image Selected\n\nChoose an image to start",
    font=("Arial", 15),
    bg="white",
    fg="#78909C",
    justify="center"
)

preview_label.pack(expand=True)


# =========================================================
# RESULT FRAME
# =========================================================

result_frame = tk.Frame(
    root,
    bg="white",
    width=700,
    height=100,
    highlightbackground="#D5DDE5",
    highlightthickness=1
)

result_frame.pack(pady=15)
result_frame.pack_propagate(False)


prediction_label = tk.Label(
    result_frame,
    text="Prediction: --",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#173B57"
)

prediction_label.pack(pady=(12, 3))


confidence_label = tk.Label(
    result_frame,
    text="Confidence: --",
    font=("Arial", 13),
    bg="white",
    fg="#455A64"
)

confidence_label.pack()


# =========================================================
# PROGRESS BAR
# =========================================================

progress = ttk.Progressbar(
    root,
    orient="horizontal",
    length=500,
    mode="determinate"
)

progress.pack(pady=8)

progress["value"] = 0


# =========================================================
# BUTTON FUNCTIONS
# =========================================================

def choose_image():
    global selected_image_path
    global image_on_screen

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.webp"),
            ("JPG Files", "*.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_image_path = file_path

    try:
        image = Image.open(file_path)
        image.thumbnail((460, 280))

        image_on_screen = ImageTk.PhotoImage(image)

        preview_label.configure(
            image=image_on_screen,
            text=""
        )

        prediction_label.config(
            text="Prediction: Ready to Predict"
        )

        confidence_label.config(
            text="Confidence: --"
        )

        progress["value"] = 0

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Unable to open image.\n\n{e}"
        )


def predict_image():
    global selected_image_path

    if selected_image_path is None:
        messagebox.showwarning(
            "No Image",
            "Please choose an image first."
        )
        return

    if model is None:
        messagebox.showerror(
            "Model Error",
            "MobileNetV2 model could not be loaded."
        )
        return

    try:
        # Start progress
        progress["value"] = 20
        root.update_idletasks()

        # Load image
        image = Image.open(selected_image_path).convert("RGB")

        # Resize image
        image = image.resize((224, 224))

        progress["value"] = 45
        root.update_idletasks()

        # Convert image to NumPy array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Preprocess image
        image_array = preprocess_input(image_array)

        progress["value"] = 65
        root.update_idletasks()

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        progress["value"] = 85
        root.update_idletasks()

        # Decode prediction
        decoded = decode_predictions(
            predictions,
            top=1
        )[0][0]

        class_id = decoded[0]
        class_name = decoded[1]
        confidence = decoded[2] * 100

        # Format class name
        class_name = class_name.replace("_", " ").title()

        progress["value"] = 100
        root.update_idletasks()

        # Display result
        prediction_label.config(
            text=f"Prediction: {class_name}"
        )

        confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:
        progress["value"] = 0

        messagebox.showerror(
            "Prediction Error",
            f"Something went wrong.\n\n{e}"
        )


def clear_image():
    global selected_image_path
    global image_on_screen

    selected_image_path = None
    image_on_screen = None

    preview_label.config(
        image="",
        text="No Image Selected\n\nChoose an image to start"
    )

    prediction_label.config(
        text="Prediction: --"
    )

    confidence_label.config(
        text="Confidence: --"
    )

    progress["value"] = 0


def exit_app():
    root.destroy()


# =========================================================
# BUTTON FRAME
# =========================================================

button_frame = tk.Frame(
    root,
    bg="#F4F7FA"
)

button_frame.pack(pady=15)


# Choose Image Button
choose_button = tk.Button(
    button_frame,
    text="📁 Choose Image",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#173B57",
    fg="white",
    activebackground="#24597D",
    activeforeground="white",
    width=16,
    height=2,
    relief="flat",
    cursor="hand2"
)

choose_button.grid(row=0, column=0, padx=8)


# Predict Button
predict_button = tk.Button(
    button_frame,
    text="🤖 Predict",
    command=predict_image,
    font=("Arial", 12, "bold"),
    bg="#00897B",
    fg="white",
    activebackground="#00695C",
    activeforeground="white",
    width=16,
    height=2,
    relief="flat",
    cursor="hand2"
)

predict_button.grid(row=0, column=1, padx=8)


# Clear Button
clear_button = tk.Button(
    button_frame,
    text="🔄 Clear",
    command=clear_image,
    font=("Arial", 12, "bold"),
    bg="#607D8B",
    fg="white",
    activebackground="#455A64",
    activeforeground="white",
    width=16,
    height=2,
    relief="flat",
    cursor="hand2"
)

clear_button.grid(row=0, column=2, padx=8)


# Exit Button
exit_button = tk.Button(
    button_frame,
    text="❌ Exit",
    command=exit_app,
    font=("Arial", 12, "bold"),
    bg="#C62828",
    fg="white",
    activebackground="#8E0000",
    activeforeground="white",
    width=16,
    height=2,
    relief="flat",
    cursor="hand2"
)

exit_button.grid(row=0, column=3, padx=8)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Label(
    root,
    text="AI Internship Task 3 • Image Classification",
    font=("Arial", 10),
    bg="#F4F7FA",
    fg="#78909C"
)

footer.pack(pady=5)


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop()