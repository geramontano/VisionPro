# Usage guide

This document explains how to install, run and use VisionPro locally.

VisionPro is a real time computer vision system for facial emotion recognition, gaze estimation, approximate camera distance estimation and user attention logging.

## 1. Requirements

The project requires Python and the libraries listed in `requirements.txt`.

The trained model included in this repository was saved with `scikit-learn 1.2.2`. For that reason, the recommended environment is:

```text
Python 3.11
scikit-learn 1.2.2
```

Using a much newer version of scikit-learn may produce model loading errors because Joblib model files depend on the library version used during training.

## 2. Recommended local environment

Create a Conda environment:

```bash
conda create -n visionpro311 python=3.11 -y
```

Activate the environment:

```bash
conda activate visionpro311
```

Install compatible dependencies:

```bash
pip install numpy==1.24.4 scipy==1.10.1 scikit-learn==1.2.2 scikit-image==0.21.0 opencv-python joblib pandas matplotlib
```

Optional installation of MediaPipe:

```bash
pip install mediapipe
```

If MediaPipe does not work on your system, VisionPro can still run using the OpenCV backend.

## 3. Enter the project folder

If the project is already downloaded locally, enter the folder:

```bash
cd /Users/gerardoalejandromontano/Downloads/proyecto_emociones_datasets_publicos
```

If the project was cloned from GitHub, use:

```bash
git clone https://github.com/geramontano/VisionPro.git
cd VisionPro
```

## 4. Run the main application

Run VisionPro with the trained emotion model:

```bash
python src/app.py --modelo modelo/emociones_modelo.joblib --camara 0 --distancia-calibracion 60
```

The argument `--camara 0` selects the default webcam.

The argument `--distancia-calibracion 60` means that distance calibration will use 60 centimeters as the known reference distance.

If the camera does not open, try:

```bash
python src/app.py --modelo modelo/emociones_modelo.joblib --camara 1 --distancia-calibracion 60
```

## 5. Recommended execution order

After the webcam window opens, use the following order:

1. Sit at the reference distance from the camera.
2. Press `d` to calibrate camera distance.
3. Press `c` to calibrate your neutral serious face.
4. Press `o` to calibrate gaze.
5. Press `l` to start logging data.
6. Interact with the visual content or web page.
7. Press `l` again to stop logging.
8. Press `q` to quit the application.

## 6. Main keyboard controls

```text
q  quit application
r  reset emotion smoothing
c  calibrate neutral serious face
d  calibrate camera distance
b  delete distance calibration
o  calibrate gaze
v  reset gaze smoothing
l  start or stop CSV logging
g  save calibration and distance profiles
```

## 7. Distance based emotion profile controls

VisionPro can capture how each facial expression looks when the user is farther away from the camera.

```text
n  capture neutral profile at distance
f  capture happy profile at distance
t  capture sad profile at distance
e  capture angry profile at distance
a  capture afraid profile at distance
s  capture surprised profile at distance
x  stop current distance profile capture
g  save profiles
```

Recommended profile capture flow:

```text
1. Press d to calibrate physical distance.
2. Move to the distance where the system should work.
3. Press n and keep a neutral face for a few seconds.
4. Press x to stop.
5. Press f and keep a happy face for a few seconds.
6. Press x to stop.
7. Repeat with t, e, a and s.
8. Press g to save profiles.
```

Distance profiles are stored locally in:

```text
modelo/perfiles_distancia.npz
modelo/perfiles_distancia.json
```

These files are ignored by Git because they are user specific runtime data.

## 8. Gaze calibration

The gaze module estimates the approximate screen region observed by the user.

To calibrate gaze:

1. Press `o`.
2. Look at each yellow point shown on the screen.
3. Keep your head as still as possible.
4. Move only your eyes when following the points.
5. Wait until the calibration finishes.

The gaze calibration is stored locally in:

```text
modelo/calibracion_mirada.npz
```

If gaze becomes inaccurate, delete the calibration file and recalibrate:

```bash
rm -f modelo/calibracion_mirada.npz
```

Then run the app again and press `o`.

## 9. Distance calibration

The distance module estimates the approximate distance between the user's face and the camera.

To calibrate distance:

1. Sit at the reference distance, for example 60 cm.
2. Run the app with `--distancia-calibracion 60`.
3. Press `d`.
4. Stay still for a few seconds.

The calibration is stored locally in:

```text
modelo/calibracion_distancia.npz
```

If the distance estimate becomes inaccurate, delete the calibration:

```bash
rm -f modelo/calibracion_distancia.npz
```

Then run the app again and press `d`.

## 10. Neutral face calibration

The neutral calibration helps the system adapt to the user's natural neutral or serious face.

To calibrate neutral face:

1. Press `c`.
2. Keep a relaxed neutral face.
3. Do not smile.
4. Do not raise eyebrows.
5. Do not frown.
6. Wait until calibration finishes.

This helps avoid confusing a naturally serious face with angry, sad or afraid expressions.

## 11. Runtime output

During execution, the application displays:

- detected face bounding box,
- predicted emotion,
- live confidence,
- probability bars,
- gaze position,
- screen zone,
- approximate distance,
- calibration status.

The confidence displayed on screen is not the same as validation accuracy. It is the confidence for the current live prediction.

## 12. Logging

Press `l` to start or stop logging.

The main CSV log is saved as:

```text
registros/mirada_emociones_v2.csv
```

The log may include:

- timestamp,
- predicted emotion,
- confidence,
- gaze coordinates,
- relative gaze position,
- screen zone,
- approximate distance,
- face area,
- gaze backend.

CSV logs are ignored by Git because they may contain personal interaction data.

## 13. Analyze logs

After recording data, run:

```bash
python src/analizar_mirada.py --entrada registros/mirada_emociones_v2.csv --salida registros/resumen_mirada.csv
```

The output summary can include:

- emotion frequency by screen region,
- average confidence by region,
- approximate attention distribution,
- relationship between gaze zone and detected emotion.

## 14. Check model accuracy

To print the stored validation accuracy:

```bash
python - <<'PY'
import joblib

paquete = joblib.load("modelo/emociones_modelo.joblib")
accuracy = paquete.get("accuracy_validacion")

print("Accuracy decimal:", accuracy)
print("Accuracy percentage:", round(accuracy * 100, 2), "%")
PY
```

The expected stored validation accuracy is:

```text
93.33%
```

You can also inspect the training report:

```bash
cat reportes/reporte_entrenamiento.txt
```

## 15. Prepare public datasets

The project was designed to work with public facial expression datasets such as JAFFE, KDEF and RAVDESS.

The datasets are not included in the repository because of licensing and size restrictions.

After downloading compatible datasets, the preparation script can be executed as:

```bash
python src/preparar_datasets.py --fuentes jaffe kdef ravdess --ravdess-actores 1 2 3 4 --frames-por-video 4
```

The prepared dataset should be stored in:

```text
datasets/preparado/
```

with one folder per emotion class.

## 16. Train the emotion model

If the prepared dataset is available, train the model with:

```bash
python src/entrenar.py --carpeta datasets/preparado --salida modelo/emociones_modelo.joblib
```

After training, check the report:

```bash
cat reportes/reporte_entrenamiento.txt
```

## 17. Troubleshooting

### The model does not load

If the terminal shows warnings or errors related to `scikit-learn` versions, use the compatible Conda environment:

```bash
conda activate visionpro311
```

Then verify:

```bash
python - <<'PY'
import sklearn
print(sklearn.__version__)
PY
```

The recommended version is:

```text
1.2.2
```

### The camera does not open

Try another camera index:

```bash
python src/app.py --modelo modelo/emociones_modelo.joblib --camara 1 --distancia-calibracion 60
```

### Gaze is inaccurate

Delete the gaze calibration and calibrate again:

```bash
rm -f modelo/calibracion_mirada.npz
```

Then run the app and press `o`.

### Distance is inaccurate

Delete the distance calibration and calibrate again:

```bash
rm -f modelo/calibracion_distancia.npz
```

Then run the app and press `d`.

### Emotions change too quickly

Press `r` to reset smoothing. Make sure lighting is stable and the face is clearly visible.

### Neutral is confused with another emotion

Press `c` and calibrate a relaxed neutral face.

### Expressions at distance are inaccurate

Use distance based emotion profiles:

```text
n, f, t, e, a, s
```

and save with:

```text
g
```

## 18. Recommended demonstration flow

For a classroom demonstration, use the following flow:

```text
1. Start the app.
2. Press d to calibrate distance.
3. Press c to calibrate neutral face.
4. Press o to calibrate gaze.
5. Test the six emotions.
6. Press l to start logging.
7. Look at different regions of a web page.
8. Press l again to stop logging.
9. Run analizar_mirada.py to summarize the data.
```

## 19. Important limitations

VisionPro is an academic prototype. It should not be interpreted as a clinical or psychological system.

Important limitations:

- webcam based gaze estimation is approximate,
- facial expressions are not equivalent to internal emotions,
- lighting affects results,
- head pose affects gaze and distance,
- glasses and shadows may affect eye detection,
- results may vary across users and cameras.

## 20. Privacy note

VisionPro uses webcam input and can create emotion gaze logs. Users should be informed when the camera is active and when logging is enabled.

Do not commit personal webcam recordings, personal face images or private CSV logs to the repository.
