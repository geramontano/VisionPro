from pathlib import Path
from datetime import datetime
import csv
import math
import time

import cv2
import numpy as np


def tamano_pantalla():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        ancho = root.winfo_screenwidth()
        alto = root.winfo_screenheight()
        root.destroy()
        return int(ancho), int(alto)
    except Exception:
        return 1440, 900


def limitar(v, a=0.0, b=1.0):
    return float(max(a, min(b, v)))


class SuavizadorPunto:
    def __init__(self, fuerza=0.78):
        self.fuerza = fuerza
        self.punto = None

    def agregar(self, x, y):
        if self.punto is None:
            self.punto = np.array([x, y], dtype=np.float32)
        else:
            nuevo = np.array([x, y], dtype=np.float32)
            distancia = float(np.linalg.norm(nuevo - self.punto))
            fuerza = self.fuerza
            if distancia > 420:
                fuerza = 0.50
            elif distancia > 220:
                fuerza = 0.62
            self.punto = fuerza * self.punto + (1.0 - fuerza) * nuevo
        return float(self.punto[0]), float(self.punto[1])

    def reiniciar(self):
        self.punto = None


class CalibradorMirada:
    def __init__(self, ruta="modelo/calibracion_mirada.npz", ancho=1440, alto=900, lam=0.11):
        self.ruta = Path(ruta)
        self.ancho = int(ancho)
        self.alto = int(alto)
        self.lam = float(lam)
        self.xcoef = None
        self.ycoef = None
        self.media = None
        self.escala = None
        self.feature_len = None
        self.expand_len = None
        self.modo = "sin calibrar"
        self.cargar()

    def cargar(self):
        if not self.ruta.exists():
            return
        try:
            data = np.load(self.ruta, allow_pickle=True)
            self.xcoef = data["xcoef"].astype(np.float32)
            self.ycoef = data["ycoef"].astype(np.float32)
            self.media = data["media"].astype(np.float32) if "media" in data else None
            self.escala = data["escala"].astype(np.float32) if "escala" in data else None
            self.feature_len = int(data["feature_len"]) if "feature_len" in data else (len(self.media) if self.media is not None else None)
            self.expand_len = int(data["expand_len"]) if "expand_len" in data else len(self.xcoef)
            if "ancho" in data:
                self.ancho = int(data["ancho"])
            if "alto" in data:
                self.alto = int(data["alto"])
            if self.xcoef is None or self.ycoef is None or len(self.xcoef) != len(self.ycoef):
                self._limpiar_calibracion()
            else:
                self.modo = "robusta"
        except Exception:
            self._limpiar_calibracion()

    def guardar(self):
        if self.xcoef is None or self.ycoef is None:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.ruta,
            xcoef=self.xcoef,
            ycoef=self.ycoef,
            media=self.media,
            escala=self.escala,
            ancho=np.array([self.ancho], dtype=np.int32),
            alto=np.array([self.alto], dtype=np.int32),
            feature_len=np.array([self.feature_len if self.feature_len is not None else 0], dtype=np.int32),
            expand_len=np.array([self.expand_len if self.expand_len is not None else 0], dtype=np.int32),
        )

    def calibrado(self):
        return self.xcoef is not None and self.ycoef is not None

    def _base(self, f):
        f = np.array(f, dtype=np.float32).reshape(-1)
        if self.feature_len is not None:
            if len(f) < self.feature_len:
                f = np.pad(f, (0, self.feature_len - len(f)), mode="constant")
            elif len(f) > self.feature_len:
                f = f[:self.feature_len]
        return f

    def _expandir(self, f):
        f = self._base(f)
        if self.media is not None and self.escala is not None and len(f) == len(self.media):
            z = (f - self.media) / self.escala
        else:
            z = f
        z = np.clip(z, -5.0, 5.0).astype(np.float32)
        partes = [z]
        partes.append(z * z)
        cruces = []
        limite = min(6, len(z))
        for i in range(limite):
            for j in range(i + 1, limite):
                cruces.append(z[i] * z[j])
        if cruces:
            partes.append(np.array(cruces, dtype=np.float32))
        partes.append(np.array([1.0], dtype=np.float32))
        return np.concatenate(partes).astype(np.float32)

    def ajustar(self, muestras):
        if len(muestras) < 80:
            return False
        feats = []
        yx = []
        yy = []
        for feat, px, py in muestras:
            f = np.array(feat, dtype=np.float32).reshape(-1)
            if len(f) < 2:
                continue
            feats.append(f)
            yx.append(float(px) / max(1, self.ancho))
            yy.append(float(py) / max(1, self.alto))
        if len(feats) < 80:
            return False
        menor = min(len(f) for f in feats)
        mayor = max(len(f) for f in feats)
        self.feature_len = menor if menor == mayor else min(menor, mayor)
        F = np.array([f[:self.feature_len] for f in feats], dtype=np.float32)
        self.media = np.median(F, axis=0).astype(np.float32)
        iqr = (np.percentile(F, 75, axis=0) - np.percentile(F, 25, axis=0)).astype(np.float32)
        std = np.std(F, axis=0).astype(np.float32)
        self.escala = np.where(np.abs(iqr) < 1e-4, std + 1e-3, iqr)
        self.escala = np.where(np.abs(self.escala) < 1e-4, 1.0, self.escala)
        X = np.array([self._expandir(f) for f in F], dtype=np.float32)
        yx = np.array(yx, dtype=np.float32)
        yy = np.array(yy, dtype=np.float32)
        self.expand_len = X.shape[1]
        I = np.eye(X.shape[1], dtype=np.float32)
        I[-1, -1] = 0.0
        lam = self.lam
        if len(muestras) < X.shape[1] * 5:
            lam = max(lam, 0.30)
        A = X.T @ X + lam * I
        self.xcoef = np.linalg.solve(A, X.T @ yx).astype(np.float32)
        self.ycoef = np.linalg.solve(A, X.T @ yy).astype(np.float32)
        self.modo = "robusta"
        self.guardar()
        return True

    def _limpiar_calibracion(self):
        self.xcoef = None
        self.ycoef = None
        self.media = None
        self.escala = None
        self.feature_len = None
        self.expand_len = None
        self.modo = "sin calibrar"

    def predecir(self, feat):
        f = np.array(feat, dtype=np.float32).reshape(-1)
        if len(f) < 2:
            return int(self.ancho * 0.5), int(self.alto * 0.5), 0.5, 0.5

        if self.calibrado():
            expandido = self._expandir(f)
            compatible = (
                self.xcoef is not None
                and self.ycoef is not None
                and len(expandido) == len(self.xcoef)
                and len(expandido) == len(self.ycoef)
            )
            if compatible:
                x = float(np.dot(expandido, self.xcoef))
                y = float(np.dot(expandido, self.ycoef))
            else:
                print("Calibracion de mirada incompatible. Borra modelo/calibracion_mirada.npz y recalibra con o.")
                self._limpiar_calibracion()
                x = 0.5 + (float(f[0]) - 0.5) * 1.35
                y = 0.5 + (float(f[1]) - 0.5) * 1.12
        else:
            x = 0.5 + (float(f[0]) - 0.5) * 1.35
            y = 0.5 + (float(f[1]) - 0.5) * 1.12
        x = limitar(x)
        y = limitar(y)
        return int(x * self.ancho), int(y * self.alto), x, y


class MedidorDistancia:
    def __init__(self, ruta="modelo/calibracion_distancia.npz", ancho_rostro_cm=16.0):
        self.ruta = Path(ruta)
        self.ancho_rostro_cm = float(ancho_rostro_cm)
        self.focal = None
        self.ultima = None
        self.cargar()

    def cargar(self):
        if not self.ruta.exists():
            return
        try:
            data = np.load(self.ruta)
            self.focal = float(np.array(data["focal"]).reshape(-1)[0])
            if "ancho_rostro_cm" in data:
                self.ancho_rostro_cm = float(np.array(data["ancho_rostro_cm"]).reshape(-1)[0])
            if self.focal <= 0 or self.focal > 8000:
                self.focal = None
        except Exception:
            self.focal = None

    def guardar(self):
        if self.focal is None:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.ruta,
            focal=np.array([self.focal], dtype=np.float32),
            ancho_rostro_cm=np.array([self.ancho_rostro_cm], dtype=np.float32),
        )

    def calibrado(self):
        return self.focal is not None

    def reset(self):
        self.focal = None
        self.ultima = None
        try:
            if self.ruta.exists():
                self.ruta.unlink()
        except Exception:
            pass

    def _ancho_caja(self, caja):
        if caja is None:
            return None
        x1, y1, x2, y2 = caja
        ancho_px = max(1.0, float(x2 - x1))
        alto_px = max(1.0, float(y2 - y1))
        if ancho_px < 20 or alto_px < 20:
            return None
        return ancho_px

    def calibrar(self, caja, distancia_cm):
        ancho_px = self._ancho_caja(caja)
        if ancho_px is None:
            return False
        self.focal = (ancho_px * float(distancia_cm)) / max(1e-6, self.ancho_rostro_cm)
        self.ultima = float(distancia_cm)
        self.guardar()
        return True

    def calibrar_con_anchos(self, anchos_px, distancia_cm):
        anchos = [float(a) for a in anchos_px if a is not None and float(a) > 20]
        if len(anchos) < 12:
            return False
        ancho = float(np.median(anchos))
        self.focal = (ancho * float(distancia_cm)) / max(1e-6, self.ancho_rostro_cm)
        self.ultima = float(distancia_cm)
        self.guardar()
        return True

    def estimar(self, caja):
        ancho_px = self._ancho_caja(caja)
        if ancho_px is None:
            return None
        if self.focal is None:
            valor = 12800.0 / ancho_px
        else:
            valor = (self.ancho_rostro_cm * self.focal) / ancho_px
        valor = float(max(20.0, min(320.0, valor)))
        if self.ultima is None:
            self.ultima = valor
        else:
            salto = abs(valor - self.ultima)
            alpha = 0.34 if salto < 25 else 0.18
            self.ultima = (1.0 - alpha) * self.ultima + alpha * valor
        return float(self.ultima)


class SeguidorMirada:
    def __init__(self):
        self.backend = "opencv"
        self.face_mesh = None
        self.suavizador = SuavizadorPunto()
        self.ojo_detector = None
        self._preparar_mediapipe()
        self._preparar_opencv()

    def _preparar_mediapipe(self):
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self.backend = "mediapipe"
        except Exception:
            self.face_mesh = None

    def _preparar_opencv(self):
        try:
            ruta = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            self.ojo_detector = cv2.CascadeClassifier(ruta)
            if self.ojo_detector.empty():
                ruta = cv2.data.haarcascades + "haarcascade_eye.xml"
                self.ojo_detector = cv2.CascadeClassifier(ruta)
        except Exception:
            self.ojo_detector = None

    def cerrar(self):
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass

    def extraer(self, frame, caja=None):
        if self.face_mesh is not None:
            salida = self._extraer_mediapipe(frame)
            if salida is not None:
                return salida
        return self._extraer_opencv(frame, caja)

    def _extraer_mediapipe(self, frame):
        try:
            alto, ancho = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            if len(lm) >= 478:
                izq_iris = np.mean([[lm[i].x, lm[i].y] for i in [468, 469, 470, 471, 472]], axis=0)
                der_iris = np.mean([[lm[i].x, lm[i].y] for i in [473, 474, 475, 476, 477]], axis=0)
            else:
                izq_iris = np.mean([[lm[i].x, lm[i].y] for i in [33, 133, 159, 145]], axis=0)
                der_iris = np.mean([[lm[i].x, lm[i].y] for i in [263, 362, 386, 374]], axis=0)
            izq_esquina_a = np.array([lm[33].x, lm[33].y])
            izq_esquina_b = np.array([lm[133].x, lm[133].y])
            der_esquina_a = np.array([lm[362].x, lm[362].y])
            der_esquina_b = np.array([lm[263].x, lm[263].y])
            izq_arriba = np.array([lm[159].x, lm[159].y])
            izq_abajo = np.array([lm[145].x, lm[145].y])
            der_arriba = np.array([lm[386].x, lm[386].y])
            der_abajo = np.array([lm[374].x, lm[374].y])
            ix = self._relativo(izq_iris[0], izq_esquina_a[0], izq_esquina_b[0])
            dx = self._relativo(der_iris[0], der_esquina_a[0], der_esquina_b[0])
            iy = self._relativo(izq_iris[1], izq_arriba[1], izq_abajo[1])
            dy = self._relativo(der_iris[1], der_arriba[1], der_abajo[1])
            gx = limitar((ix + dx) / 2.0)
            gy = limitar((iy + dy) / 2.0)
            nariz = np.array([lm[1].x, lm[1].y])
            frente = np.array([lm[10].x, lm[10].y])
            menton = np.array([lm[152].x, lm[152].y])
            mejilla_i = np.array([lm[234].x, lm[234].y])
            mejilla_d = np.array([lm[454].x, lm[454].y])
            ojo_i_c = (izq_esquina_a + izq_esquina_b) / 2.0
            ojo_d_c = (der_esquina_a + der_esquina_b) / 2.0
            centro_ojos = (ojo_i_c + ojo_d_c) / 2.0
            cara_ancho = abs(mejilla_d[0] - mejilla_i[0])
            cara_alto = abs(menton[1] - frente[1])
            roll = math.atan2(ojo_d_c[1] - ojo_i_c[1], ojo_d_c[0] - ojo_i_c[0])
            yaw = (nariz[0] - centro_ojos[0]) / max(1e-6, cara_ancho)
            pitch = (nariz[1] - centro_ojos[1]) / max(1e-6, cara_alto)
            separacion_ojos = float(np.linalg.norm(ojo_d_c - ojo_i_c))
            iris_sep = float(np.linalg.norm(der_iris - izq_iris))
            pupila_x_diff = ix - dx
            pupila_y_diff = iy - dy
            feat = np.array([
                gx, gy, ix, dx, iy, dy, pupila_x_diff, pupila_y_diff,
                nariz[0], nariz[1], yaw, pitch, roll, cara_ancho, cara_alto,
                separacion_ojos, iris_sep
            ], dtype=np.float32)
            px = int(((izq_iris[0] + der_iris[0]) / 2.0) * ancho)
            py = int(((izq_iris[1] + der_iris[1]) / 2.0) * alto)
            return {"ok": True, "feat": feat, "ojo": (px, py), "backend": self.backend}
        except Exception:
            return None

    def _extraer_opencv(self, frame, caja=None):
        if caja is None:
            return None
        x1, y1, x2, y2 = caja
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        cara = frame[y1:y2, x1:x2]
        gris = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        alto, ancho = gris.shape[:2]
        zona = gris[int(alto * 0.16):int(alto * 0.62), :]
        ojos = []
        if self.ojo_detector is not None and not self.ojo_detector.empty():
            detectados = self.ojo_detector.detectMultiScale(zona, scaleFactor=1.06, minNeighbors=5, minSize=(18, 12))
            for ex, ey, ew, eh in detectados:
                cx = ex + ew / 2
                if 0.10 * ancho < cx < 0.90 * ancho:
                    ojos.append((ex, ey + int(alto * 0.16), ew, eh))
        if len(ojos) < 2:
            ojos = self._ojos_por_region(gris)
        ojos = sorted(ojos, key=lambda r: r[0])[:2]
        datos = []
        centros = []
        areas = []
        for ex, ey, ew, eh in ojos:
            recorte = gris[max(0, ey):min(alto, ey + eh), max(0, ex):min(ancho, ex + ew)]
            if recorte.size == 0:
                continue
            px, py, area = self._pupila(recorte)
            datos.append((limitar(px), limitar(py)))
            areas.append(area)
            centros.append((x1 + ex + int(px * ew), y1 + ey + int(py * eh)))
        if not datos:
            return None
        gx = float(np.mean([d[0] for d in datos]))
        gy = float(np.mean([d[1] for d in datos]))
        d0 = datos[0]
        d1 = datos[-1]
        centro_ojos = np.mean(np.array(centros, dtype=np.float32), axis=0)
        cx_cara = (x1 + x2) / 2.0 / max(1, w)
        cy_cara = (y1 + y2) / 2.0 / max(1, h)
        area = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
        ancho_rel = (x2 - x1) / max(1, w)
        alto_rel = (y2 - y1) / max(1, h)
        feat = np.array([
            gx, gy, d0[0], d1[0], d0[1], d1[1], d0[0] - d1[0], d0[1] - d1[1],
            cx_cara, cy_cara, ancho_rel, alto_rel, math.sqrt(max(0.0, area)),
            float(np.mean(areas)) if areas else 0.0
        ], dtype=np.float32)
        return {"ok": True, "feat": feat, "ojo": (int(centro_ojos[0]), int(centro_ojos[1])), "backend": self.backend}

    def _ojos_por_region(self, gris):
        alto, ancho = gris.shape[:2]
        y = int(alto * 0.22)
        h = int(alto * 0.28)
        return [
            (int(ancho * 0.12), y, int(ancho * 0.32), h),
            (int(ancho * 0.56), y, int(ancho * 0.32), h),
        ]

    def _pupila(self, ojo):
        ojo = cv2.equalizeHist(ojo)
        ojo = cv2.GaussianBlur(ojo, (5, 5), 0)
        _, th = cv2.threshold(ojo, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return 0.5, 0.5, 0.0
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        h, w = ojo.shape[:2]
        mejor = None
        mejor_puntaje = -1
        for c in contornos[:4]:
            area = cv2.contourArea(c) / max(1.0, w * h)
            if area < 0.015 or area > 0.42:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            forma = min(ww, hh) / max(1, max(ww, hh))
            puntaje = area * 0.65 + forma * 0.35
            if puntaje > mejor_puntaje:
                mejor = c
                mejor_puntaje = puntaje
        if mejor is None:
            mejor = contornos[0]
        m = cv2.moments(mejor)
        if m["m00"] == 0:
            return 0.5, 0.5, 0.0
        x = m["m10"] / m["m00"]
        y = m["m01"] / m["m00"]
        area = cv2.contourArea(mejor) / max(1.0, w * h)
        return limitar(x / max(1, w)), limitar(y / max(1, h)), limitar(area)

    def _relativo(self, v, a, b):
        mn = min(a, b)
        mx = max(a, b)
        return limitar((v - mn) / (mx - mn + 1e-6))


class RegistroMirada:
    def __init__(self, ruta="registros/mirada_emociones_v2.csv", intervalo=0.25):
        self.ruta = Path(ruta)
        self.intervalo = float(intervalo)
        self.ultimo = 0.0
        self.activo = False
        self.campos = [
            "tiempo", "emocion", "confianza", "mirada_x", "mirada_y", "mirada_rel_x", "mirada_rel_y",
            "zona", "backend_mirada", "distancia_cm", "distancia_calibrada", "cara_area", "nota"
        ]
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if not self.ruta.exists():
            with open(self.ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.campos)
                writer.writeheader()

    def zona(self, rx, ry):
        col = "izquierda" if rx < 0.33 else "centro" if rx < 0.66 else "derecha"
        fila = "arriba" if ry < 0.33 else "medio" if ry < 0.66 else "abajo"
        return fila + "_" + col

    def agregar(self, emocion, confianza, x, y, rx, ry, backend, area=None, distancia_cm=None, distancia_calibrada=False, nota=""):
        if not self.activo:
            return
        ahora = time.time()
        if ahora - self.ultimo < self.intervalo:
            return
        self.ultimo = ahora
        fila = {
            "tiempo": datetime.now().isoformat(timespec="milliseconds"),
            "emocion": emocion,
            "confianza": round(float(confianza), 5),
            "mirada_x": int(x),
            "mirada_y": int(y),
            "mirada_rel_x": round(float(rx), 5),
            "mirada_rel_y": round(float(ry), 5),
            "zona": self.zona(rx, ry),
            "backend_mirada": backend,
            "distancia_cm": round(float(distancia_cm), 2) if distancia_cm is not None else "",
            "distancia_calibrada": bool(distancia_calibrada),
            "cara_area": round(float(area), 5) if area is not None else "",
            "nota": nota,
        }
        with open(self.ruta, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.campos)
            writer.writerow(fila)
