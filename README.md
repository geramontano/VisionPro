# VisionPro

VisionPro is a real time computer vision and machine learning system for facial emotion recognition, gaze estimation and approximate user attention analysis using a standard laptop camera.

The system detects facial expressions, estimates the user's gaze direction, approximates the distance between the user and the camera, and logs emotional response associated with the visual region being observed.

## Main features

- Real time facial emotion recognition from webcam.
- Six emotion classes: neutral, happy, sad, angry, afraid and surprised.
- Statistical ensemble model trained with public facial expression datasets.
- Face detection using OpenCV.
- Gaze estimation using eye region analysis and multipoint calibration.
- Approximate camera to face distance estimation.
- CSV logging of emotion, confidence, gaze position, screen zone and distance.
- Post processing script for attention and emotion summaries by screen region.

## Emotion recognition model

The emotion classifier uses a hybrid statistical approach. Instead of relying only on a pretrained black box model, the system extracts handcrafted and geometric features from the detected face and trains a multiclass ensemble classifier.

The model combines Support Vector Machine with RBF kernel, Extra Trees, Random Forest, Logistic Regression and a Multilayer Perceptron classifier.

The feature representation includes HOG descriptors, LBP texture descriptors, geometric facial measurements and proportions related to the mouth, eyebrows and eyes.

The validation accuracy obtained by the trained model was 93.33 percent.

This value represents the performance of the trained model on the validation set, not the instantaneous confidence displayed during live webcam execution.

## Public datasets

The training pipeline was designed to work with public facial expression datasets such as JAFFE, KDEF and RAVDESS.

These datasets are not included directly in the repository because of size and licensing restrictions.

## Gaze estimation

The gaze module estimates the approximate region of the screen observed by the user. It uses eye region analysis, pupil position estimation and multipoint calibration.

This is not a clinical eye tracker. It is an approximate webcam based gaze estimation system intended for classroom level human computer interaction analysis.

## Distance estimation

The distance between the user and the camera is estimated from the apparent face size. The user calibrates the system by placing their face at a known distance from the camera.

## Installation

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Usage

Run the application:

python src/app.py --modelo modelo/emociones_modelo.joblib --camara 0 --distancia-calibracion 60

The value 60 indicates that the distance calibration will use 60 cm as the reference distance.

## Main controls

c  calibrate neutral serious face
d  calibrate camera distance
o  calibrate gaze
l  start or stop CSV logging
v  reset gaze smoothing
r  reset emotion smoothing
q  quit

## Recommended execution flow

1. Start the application.
2. Sit at a known distance from the camera.
3. Press d to calibrate camera distance.
4. Press c to calibrate neutral serious face.
5. Press o to calibrate gaze.
6. Press l to start logging.
7. Interact with the visual content or web page.
8. Press l again or q to stop.

## Output

The system can generate a CSV file with timestamp, predicted emotion, confidence, gaze coordinates, screen zone, approximate distance in centimeters, face area and gaze backend used.

## Project objective

The goal of VisionPro is to explore how facial emotion recognition and approximate gaze estimation can be combined to infer user response to visual content.

For example, when a user observes a web page, the system can estimate which region of the page they were looking at and what emotional response was detected at that moment.

## Limitations

- The gaze estimation is approximate and depends on calibration quality.
- A standard laptop webcam cannot provide the same precision as dedicated eye tracking hardware.
- Lighting, glasses, head movement and camera position can affect performance.
- The emotional prediction is based on visible facial cues and should not be interpreted as a definitive psychological state.

## Author

Gerardo Alejandro Montano Gonzalez
