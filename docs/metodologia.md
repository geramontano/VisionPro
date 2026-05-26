# Methodology

This document describes the methodological design of VisionPro, a real time computer vision system for facial emotion recognition, gaze estimation, camera distance approximation and attention based behavioral logging.

VisionPro was designed as a modular machine learning pipeline. Each module performs a specific task: image acquisition, face detection, feature extraction, emotion classification, gaze estimation, distance estimation and data logging.

## 1. General methodological approach

VisionPro follows a hybrid methodology that combines classical computer vision, handcrafted feature extraction, statistical machine learning, real time webcam processing, user based calibration and behavioral data logging.

The system does not rely only on a pretrained black box model. Instead, it uses a trained statistical ensemble over interpretable facial features such as texture descriptors, local gradients and geometric facial proportions.

This design was selected to make the system easier to explain, reproduce and analyze in an academic context.

## 2. System overview

The complete workflow is divided into two main stages:

1. Offline model preparation and training.
2. Online real time inference and interaction logging.

During the offline stage, public facial expression datasets are prepared, normalized and used to train an emotion recognition model.

During the online stage, the webcam stream is processed frame by frame. The system detects the face, extracts features, predicts emotion, estimates gaze, estimates camera distance and optionally stores synchronized data in a CSV file.

## 3. Dataset preparation

The training pipeline was designed to work with public facial expression datasets such as JAFFE, KDEF and RAVDESS.

The datasets are not directly included in the repository because of size and licensing restrictions. Instead, the project provides dataset preparation scripts that organize data into a common folder structure.

The expected prepared dataset structure is:

```text
datasets/preparado/
├── neutro/
├── feliz/
├── triste/
├── enojado/
├── asustado/
└── sorprendido/
```

Each folder contains face images associated with one of the six target emotion classes.

The preparation step performs the following operations:

1. Dataset loading.
2. Label mapping to the target emotion set.
3. Face detection.
4. Face cropping.
5. Image normalization.
6. Storage into the prepared dataset folder.

## 4. Target emotion classes

The system classifies six facial expression categories:

- neutral,
- happy,
- sad,
- angry,
- afraid,
- surprised.

These categories were selected because they are common in facial expression recognition datasets and provide a compact but useful emotion space for classroom level human computer interaction experiments.

## 5. Face detection

At runtime, each webcam frame is passed through a face detection module. The current implementation uses OpenCV as the available face detection backend.

If a face is detected, the system extracts a facial crop and sends it to the feature extraction module. If no face is detected, the system skips emotion prediction for that frame and displays a no face status.

This prevents the model from producing predictions when the input is not a valid facial region.

## 6. Feature extraction

The emotion classifier does not operate directly on raw image pixels. Instead, each detected face is transformed into a numerical feature vector.

The feature vector combines three main feature groups:

1. Histogram of Oriented Gradients features.
2. Local Binary Pattern texture features.
3. Geometric facial measurements.

### 6.1 HOG features

HOG descriptors represent local gradient orientation patterns. They are useful for capturing facial contours, edge structure, mouth shape, eyebrow orientation and wrinkles produced by facial expressions.

### 6.2 LBP features

LBP descriptors represent local texture patterns. They are useful for describing fine changes in facial texture and local intensity relationships.

### 6.3 Geometric features

Geometric features describe proportions related to facial components such as the eyes, eyebrows and mouth. These measurements help represent expression related changes such as mouth opening, eye widening or eyebrow movement.

The final feature vector is built by concatenating all extracted descriptors.

## 7. Emotion recognition model

The emotion recognition module uses an ensemble classifier. The ensemble combines multiple statistical models in order to improve robustness and reduce dependency on a single classifier.

The trained ensemble includes:

- Support Vector Machine with RBF kernel,
- Extra Trees classifier,
- Random Forest classifier,
- Logistic Regression,
- Multilayer Perceptron classifier.

Each classifier contributes to the final emotion probability distribution. The final class is selected from the aggregated probabilities.

This strategy was selected because different models capture different decision patterns. Tree based models are useful for nonlinear feature interactions, SVM can create nonlinear class boundaries, Logistic Regression provides a simpler probabilistic baseline and MLP adds a neural nonlinear component.

## 8. Model validation

After training, the model is evaluated on a validation set. The stored validation accuracy obtained by the model was:

```text
Accuracy validation: 93.33%
```

This metric represents the proportion of correctly classified validation samples.

It is important to distinguish validation accuracy from live webcam confidence. Validation accuracy evaluates the general model on a held out dataset, while live confidence is computed for a specific real time prediction.

The training report is stored in:

```text
reportes/reporte_entrenamiento.txt
```

The report includes precision, recall, F1 score and the confusion matrix for the six emotion classes.

## 9. Real time prediction

During real time execution, the system follows this process for each frame:

1. Capture frame from webcam.
2. Detect face.
3. Crop and normalize face.
4. Extract HOG, LBP and geometric features.
5. Predict emotion probabilities using the trained ensemble.
6. Apply auxiliary facial rules if enabled.
7. Apply temporal smoothing.
8. Display the predicted emotion and confidence.

Temporal smoothing reduces abrupt changes between consecutive frames and makes the live output more stable for demonstrations.

## 10. Neutral face calibration

A practical issue is that a user's neutral face may not look neutral to the general model. Some users have a naturally serious expression that may resemble sadness, anger or fear.

To address this, VisionPro includes a neutral calibration step. During this step, the user maintains a neutral or serious relaxed face. The system records the model's probability pattern for that user and uses it as a personal neutral reference.

This makes the system more adaptable to the specific user without retraining the whole model.

## 11. Distance based emotion profiles

The system also supports distance based profile capture. This allows the user to record how each expression looks when they move farther away from the camera.

The controls are:

```text
n  capture neutral profile at distance
f  capture happy profile at distance
t  capture sad profile at distance
e  capture angry profile at distance
a  capture afraid profile at distance
s  capture surprised profile at distance
x  stop current capture
g  save profiles
```

These profiles are stored locally and are not committed to the repository because they may contain user specific runtime information.

## 12. Gaze estimation methodology

The gaze estimation module estimates the approximate region of the screen observed by the user.

The gaze pipeline includes:

1. Face detection.
2. Eye region extraction.
3. Eye and pupil feature estimation.
4. Multipoint gaze calibration.
5. Regression from eye features to screen coordinates.
6. Screen zone assignment.

The user calibrates gaze by looking at known points on the screen. During this process, the system records eye based features and learns a mapping from those features to screen coordinates.

This webcam based gaze estimation is approximate. It is not intended to replace professional eye tracking hardware.

## 13. Distance estimation methodology

The camera distance module estimates the approximate distance between the user's face and the webcam.

The method is based on the apparent size of the detected face. During calibration, the user places their face at a known distance, for example 60 cm. The system records the corresponding face size in pixels.

During execution, the current face size is compared with the calibrated size. If the face appears smaller, the estimated distance increases. If the face appears larger, the estimated distance decreases.

This method assumes that the camera remains fixed and that the user's head pose does not change dramatically.

## 14. Logging methodology

VisionPro can store synchronized runtime data in a CSV file. The log may include:

- timestamp,
- predicted emotion,
- confidence,
- gaze coordinates,
- relative gaze position,
- screen zone,
- approximate distance,
- face area,
- gaze backend.

The main log file is:

```text
registros/mirada_emociones_v2.csv
```

CSV files are ignored by Git because they may contain personal interaction data.

## 15. Post processing

The script `src/analizar_mirada.py` analyzes the generated CSV logs.

The post processing stage can summarize:

- emotion frequency by screen region,
- average confidence by screen region,
- attention distribution over the screen,
- relationship between gaze zone and predicted emotion.

This allows the system to approximate which regions of a visual interface were associated with positive, negative or neutral facial responses.

## 16. Reproducibility

The repository includes:

- source code,
- training scripts,
- dataset preparation scripts,
- trained model file,
- validation report,
- architecture documentation,
- mathematical formulation,
- methodology documentation.

The original datasets are not included because of licensing and size restrictions. Users should download them from their official sources and follow their license terms.

## 17. Privacy considerations

VisionPro uses webcam input and can generate behavioral logs. For this reason, users should be informed when the camera is active and when logging is enabled.

The repository is configured to avoid committing:

- raw webcam recordings,
- personal face captures,
- gaze logs,
- emotion logs,
- calibration files.

This is important because gaze and emotion data may be sensitive.

## 18. Limitations

The methodology has several limitations:

- Facial expressions are not equivalent to internal emotional states.
- Webcam based gaze estimation is approximate.
- Distance estimation depends on stable face detection.
- Lighting conditions can affect performance.
- Glasses, shadows and head pose can reduce accuracy.
- Public datasets may contain posed expressions rather than spontaneous emotions.
- Model performance may vary across users, cameras and environments.

## 19. Intended use

VisionPro is intended for:

- academic demonstration,
- classroom computer vision projects,
- exploratory human computer interaction analysis,
- prototyping emotion and gaze based interaction systems.

It is not intended for:

- medical diagnosis,
- psychological diagnosis,
- surveillance,
- hiring decisions,
- academic evaluation of individuals,
- clinical emotion recognition.

## 20. Summary

VisionPro follows a modular methodology that combines computer vision, statistical machine learning and real time user interaction analysis. The system first detects the face, then extracts interpretable features, predicts emotion with an ensemble classifier, estimates gaze and distance, and stores synchronized data for later analysis.

The methodological goal is not to claim perfect emotion or gaze recognition, but to build an explainable and reproducible prototype that links facial expression recognition with approximate visual attention.
