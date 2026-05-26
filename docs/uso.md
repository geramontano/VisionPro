# Usage guide

## Run the system

python src/app.py --modelo modelo/emociones_modelo.joblib --camara 0 --distancia-calibracion 60

## Recommended order

1. Press d to calibrate distance.
2. Press c to calibrate neutral face.
3. Press o to calibrate gaze.
4. Press l to start logging.
5. Press q to quit.

## Analyze logs

python src/analizar_mirada.py --entrada registros/mirada_emociones_v2.csv --salida registros/resumen_mirada.csv

## Notes

For best results, use good frontal lighting, keep the laptop camera fixed, avoid moving the head during gaze calibration and recalibrate if the user changes distance.
