# Mathematical formulation

This document summarizes the mathematical structure behind VisionPro.

## Face representation

Let an input video frame be represented as I_t. A face detector extracts a facial region F_t from the frame.

The system computes a feature vector x_t from the detected face. This feature vector includes HOG descriptors, LBP texture descriptors and geometric facial measurements.

## Emotion classification

The emotion set is composed of neutral, happy, sad, angry, afraid and surprised.

The model estimates the probability of each emotion given the extracted feature vector. The final predicted emotion is the class with the highest probability.

The classifier is an ensemble model composed of SVM with RBF kernel, Extra Trees, Random Forest, Logistic Regression and MLP.

## Temporal smoothing

Real time webcam predictions can fluctuate between adjacent frames. To reduce this instability, the system averages recent probability vectors over a temporal window.

The final prediction is selected from the smoothed probability vector.

## Confidence value

The live confidence displayed on screen is not the same as model accuracy. It is an instantaneous confidence score for the current prediction.

The model accuracy was measured on the validation set, while live confidence changes frame by frame.

## Gaze estimation

The system extracts eye based features and maps them to screen coordinates using a calibration process.

During calibration, the user looks at known points on the screen. The system records eye features and learns a mapping from eye features to screen position.

## Distance estimation

The distance is estimated from the apparent size of the face. If the face appears smaller, the user is probably farther away. If the face appears larger, the user is probably closer.

The user calibrates the system at a known distance, for example 60 cm.

## Emotion attention association

For each time step, the system stores predicted emotion, confidence, gaze position, estimated distance and timestamp.

This allows the system to estimate which emotional responses are associated with specific regions of the screen.

## Interpretation

VisionPro does not infer internal psychological states directly. It estimates visible facial expressions and approximate gaze direction.
