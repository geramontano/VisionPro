import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


class DetectorCara:
    def __init__(self, confianza=0.55):
        self.confianza = confianza
        self.detector = None
        self.tipo = "opencv"
        self.cascada = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            try:
                self.mp_cara = mp.solutions.face_detection
                self.detector = self.mp_cara.FaceDetection(model_selection=1, min_detection_confidence=confianza)
                self.tipo = "mediapipe"
            except Exception:
                self.detector = None
                self.tipo = "opencv"

    def caja_mediapipe(self, imagen):
        alto, ancho = imagen.shape[:2]
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = self.detector.process(rgb)
        if not resultado.detections:
            return None

        mejor = None
        mejor_area = 0
        for deteccion in resultado.detections:
            caja = deteccion.location_data.relative_bounding_box
            x = int(caja.xmin * ancho)
            y = int(caja.ymin * alto)
            w = int(caja.width * ancho)
            h = int(caja.height * alto)
            area = max(w, 0) * max(h, 0)
            if area > mejor_area:
                mejor_area = area
                mejor = (x, y, w, h)
        return mejor

    def caja_opencv(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        caras = self.cascada.detectMultiScale(
            gris,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(45, 45)
        )
        if len(caras) == 0:
            return None
        caras = sorted(caras, key=lambda c: c[2] * c[3], reverse=True)
        x, y, w, h = caras[0]
        return int(x), int(y), int(w), int(h)

    def recortar(self, imagen, margen=0.28):
        alto, ancho = imagen.shape[:2]

        if self.tipo == "mediapipe" and self.detector is not None:
            mejor = self.caja_mediapipe(imagen)
            if mejor is None:
                mejor = self.caja_opencv(imagen)
        else:
            mejor = self.caja_opencv(imagen)

        if mejor is None:
            return None, None

        x, y, w, h = mejor
        lado = int(max(w, h) * (1 + margen))
        cx = x + w // 2
        cy = y + h // 2
        x1 = max(cx - lado // 2, 0)
        y1 = max(cy - lado // 2, 0)
        x2 = min(cx + lado // 2, ancho)
        y2 = min(cy + lado // 2, alto)

        if x2 <= x1 or y2 <= y1:
            return None, None

        cara = imagen[y1:y2, x1:x2].copy()
        return cara, (x1, y1, x2, y2)

    def cerrar(self):
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()
