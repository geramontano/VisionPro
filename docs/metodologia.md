# Methodology

VisionPro follows a modular computer vision pipeline.

## Pipeline

1. Webcam frame acquisition.
2. Face detection.
3. Facial feature extraction.
4. Emotion classification.
5. Temporal smoothing.
6. Gaze estimation.
7. Distance estimation.
8. CSV logging.
9. Post processing and analysis.

## Emotion recognition

The emotion recognition module uses a trained statistical ensemble. The input is not the raw image directly, but a feature vector extracted from the detected face.

The system combines texture descriptors and facial geometry. This allows the model to capture both local visual patterns and structural changes in the face, such as mouth opening, eyebrow position or eye shape.

## Gaze estimation

The gaze module uses calibration. The user looks at known points on the screen. During this process, the system records eye based features and learns a mapping from eye features to screen coordinates.

## Distance estimation

The distance module uses the apparent face size. The user calibrates the system at a known distance. After calibration, the system estimates distance changes using the variation in detected face size.

## Logging

The system stores synchronized information about emotion, confidence, gaze position, screen zone and distance. This allows later analysis of user response to visual content.
