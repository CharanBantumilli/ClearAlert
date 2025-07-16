# ClearAlert

ClearAlert is a Python-based application designed for real-time fire and smoke detection, specifically built to enhance security and surveillance camera systems. By leveraging deep learning (YOLO), ClearAlert provides fast, automated alerts for fire and smoke hazards—helping prevent disasters and enable timely response.

## Why ClearAlert?

The motivation behind ClearAlert is to provide a reliable, automated detection system for fire and smoke events in live video feeds, such as those from security or surveillance cameras. Early detection is crucial for safety, property protection, and effective emergency response.

## Features

- **Live Fire/Smoke Detection:** Uses a YOLO deep learning model to detect fire and smoke in images and videos.
- **Designed for Security & Surveillance Cameras:** Integrates easily with live camera feeds for continuous monitoring.
- **Automated Alerts:** Sends WhatsApp notifications with annotated evidence when fire or smoke is detected.
- **Image & Video Support:** Upload and process both images and videos; can be adapted for real-time video stream.
- **Simple GUI:** Easy-to-use Tkinter interface for image/video upload.
- **Visual Confirmation:** Annotated results displayed in-app.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CharanBantumilli/ClearAlert.git
   cd ClearAlert
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python pywhatkit tkinter
   ```

3. **Model File**
   Place your trained YOLO model file (`best.pt`) in the project directory.

## Usage

### Image Detection

```bash
python main.py
```
- Upload an image containing fire/smoke using the GUI.
- Detection results will be displayed and WhatsApp alert sent if a hazard is found.

### Video Detection

```bash
python fire_yolo.py
```
- Upload a video file.
- The app processes each frame for fire/smoke, annotates the results, and saves an output video.

### WhatsApp Alerts

- Update the `phone_number` variable in `main.py` with the recipient's WhatsApp number.
- Ensure WhatsApp Web is set up on your system for `pywhatkit`.

## Example Scenario

- Integrate ClearAlert with a surveillance camera setup.
- When fire or smoke is detected in live feeds, receive instant WhatsApp notifications with annotated evidence.

## License

MIT License. See [LICENSE](LICENSE).

## Author

- Charan Bantumilli - [GitHub](https://github.com/CharanBantumilli)

## Contributing

Pull requests are welcome! For major changes, open an issue to discuss what you’d like to change.

---

ClearAlert: Early detection saves lives.
