import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cara import DetectorCara
from modelo import crear_modelo
from nombres import EMOCIONES
from rasgos import ExtractorRasgos


parser = argparse.ArgumentParser()
parser.add_argument("--carpeta", default="datasets/preparado")
parser.add_argument("--salida", default="modelo/emociones_modelo.joblib")
parser.add_argument("--reporte", default="reportes/reporte_entrenamiento.txt")
parser.add_argument("--limite-por-clase", type=int, default=0)
args = parser.parse_args()

carpeta = Path(args.carpeta)
imagenes = []
etiquetas = []

for emocion in EMOCIONES:
    rutas = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        rutas.extend((carpeta / emocion).glob(extension))
    rutas = sorted(rutas)
    if args.limite_por_clase > 0:
        rutas = rutas[:args.limite_por_clase]
    for ruta in rutas:
        imagenes.append(ruta)
        etiquetas.append(emocion)

if len(imagenes) < 60:
    raise SystemExit("No hay suficientes imagenes. Primero ejecuta src/preparar_datasets.py")

conteo = {emocion: etiquetas.count(emocion) for emocion in EMOCIONES}
print("Imagenes por emocion:", conteo)

extractor = ExtractorRasgos()
detector = DetectorCara(confianza=0.45)
X = []
y = []

for ruta, etiqueta in tqdm(list(zip(imagenes, etiquetas)), desc="Extrayendo rasgos"):
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        continue
    cara, _ = detector.recortar(imagen)
    if cara is None:
        cara = imagen
    try:
        rasgos = extractor.extraer(cara)
    except Exception:
        continue
    X.append(rasgos)
    y.append(etiqueta)

extractor.cerrar()
detector.cerrar()

X = np.array(X, dtype=np.float32)
y = np.array(y)

if len(set(y)) < 4:
    raise SystemExit("Faltan clases suficientes para entrenar.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=17, stratify=y
)

modelo = crear_modelo()
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)
reporte = classification_report(y_test, pred, digits=4)
matriz = confusion_matrix(y_test, pred, labels=EMOCIONES)

Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
Path(args.reporte).parent.mkdir(parents=True, exist_ok=True)

paquete = {
    "modelo": modelo,
    "clases": list(modelo.classes_),
    "emociones_objetivo": EMOCIONES,
    "accuracy_validacion": float(acc),
    "conteo": conteo,
    "rasgos": int(X.shape[1])
}
joblib.dump(paquete, args.salida)

with open(args.reporte, "w", encoding="utf-8") as archivo:
    archivo.write("Accuracy validacion: " + str(round(acc, 4)) + "\n\n")
    archivo.write("Imagenes por emocion:\n")
    for emocion in EMOCIONES:
        archivo.write(emocion + ": " + str(conteo.get(emocion, 0)) + "\n")
    archivo.write("\nReporte:\n")
    archivo.write(reporte)
    archivo.write("\nMatriz de confusion, orden " + str(EMOCIONES) + ":\n")
    archivo.write(str(matriz))

print("Accuracy validacion:", round(acc, 4))
print(reporte)
print("Modelo guardado en", args.salida)
print("Reporte guardado en", args.reporte)
