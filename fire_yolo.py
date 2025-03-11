from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# Load a pre-trained YOLO model
model = YOLO('best.pt')

# Function to upload and process video
def upload_video():
    video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4")])
    if video_path:
        process_video(video_path)

# Function to process the selected video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the selected video

    # Get video properties for saving the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the output
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        # Perform object detection on the frame
        results = model(frame)

        # Visualize the results on the frame
        for result in results:
            annotated_frame = result.plot()

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow('YOLO Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Create a Tkinter window
root = tk.Tk()
root.title("Fire and Smoke detection Video Upload")
root.geometry("300x150")


# Create an Upload Button
btn_upload = tk.Button(root, text="Upload Video", command=upload_video, padx=10, pady=5)
btn_upload.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
