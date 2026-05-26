import argparse
import time

import cv2

from cara import DetectorCara


parser = argparse.ArgumentParser()
parser.add_argument("--camara", type=int, default=0)
parser.add_argument("--segundos", type=int, default=45)
args = parser.parse_args()

detector = DetectorCara(confianza=0.40)
cap = cv2.VideoCapture(args.camara)
inicio = time.time()
frames = 0
detectados = 0
areas = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    cara, caja = detector.recortar(frame)
    if cara is not None:
        detectados += 1
        x1, y1, x2, y2 = caja
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
        cv2.putText(frame, "cara detectada", (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
    tasa = detectados / max(frames, 1)
    cv2.putText(frame, "deteccion: " + str(round(tasa * 100, 1)) + "%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    cv2.putText(frame, "colocate a 3 metros | q salir", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    cv2.imshow("Prueba distancia", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if time.time() - inicio >= args.segundos:
        break

cap.release()
cv2.destroyAllWindows()
detector.cerrar()
print("Frames:", frames)
print("Detectados:", detectados)
print("Tasa:", round(detectados / max(frames, 1), 4))
if areas:
    print("Area promedio de cara:", round(sum(areas) / len(areas), 2))
