from ultralytics import YOLO
import cv2
import pywhatkit
import time
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = YOLO("best.pt")
print(model.names)


# Define the alert function
def send_alert(dc, image_path):
    message="Something is suspicious"
    phone_number = "+xxxxxxxxx"  # Replace with the recipient's phone number
    if "Fire" in dc and "Smoke" in dc:
        message = "Fire and smoke is detected be alert"
    elif "Fire" in dc:
        message = "Fire is detected be alert"
    elif "Smoke" in dc:
        message = "Smoke is catched once check your sight"
    try:
        pywhatkit.sendwhats_image(phone_number, image_path, message)
        print("Alert with image sent successfully!")
    except Exception as e:
        print(f"Failed to send alert: {e}")


# Function to open file dialog and select an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", ["*.jpg","*.webp"])])
    if file_path:
        process_image(file_path)


# Function to process the selected image
def process_image(image_path):
    results = model.predict(source=image_path,save=True, conf=0.25)



    for result in results:
        img = result.plot()
        detected_classes = list(set([model.names[int(box.cls)] for box in result.boxes]))
        print("Detected Classes:", detected_classes)

        send_alert(detected_classes, image_path)

        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

        cv2.imshow("YOLO Predictions", resized_img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# Create a Tkinter window
root = tk.Tk()
root.title("Fire and Smoke Detection")
root.geometry("300x150")

# Create an Upload Button
btn_upload = tk.Button(root, text="Upload Image", command=upload_image, padx=10, pady=5)
btn_upload.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
