import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
from skimage.feature import hog, local_binary_pattern


class ExtractorRasgos:
    def __init__(self, tamano=160):
        self.tamano = tamano
        self.malla = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mp_malla = mp.solutions.face_mesh
                self.malla = self.mp_malla.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except Exception:
                self.malla = None
        self.ultimas_medidas = {}

    def preparar_gris(self, cara):
        cara = cv2.resize(cara, (self.tamano, self.tamano))
        gris = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gris = clahe.apply(gris)
        return cara, gris

    def puntos(self, cara):
        if self.malla is None:
            return None
        rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
        resultado = self.malla.process(rgb)
        if not resultado.multi_face_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in resultado.multi_face_landmarks[0].landmark], dtype=np.float32)

    def distancia(self, puntos, a, b):
        return float(np.linalg.norm(puntos[a, :2] - puntos[b, :2]))

    def valor_seguro(self, numerador, denominador):
        return float(numerador / (denominador + 1e-6))

    def geometria(self, puntos):
        if puntos is None:
            self.ultimas_medidas = {}
            return np.zeros(26, dtype=np.float32)

        ancho_cara = self.distancia(puntos, 234, 454)
        alto_cara = self.distancia(puntos, 10, 152)
        boca_ancho = self.distancia(puntos, 61, 291)
        boca_abierta = self.distancia(puntos, 13, 14)
        ojo_izq = self.distancia(puntos, 159, 145)
        ojo_der = self.distancia(puntos, 386, 374)
        ceja_izq = self.distancia(puntos, 105, 159)
        ceja_der = self.distancia(puntos, 334, 386)
        ceja_centro_izq = self.distancia(puntos, 70, 159)
        ceja_centro_der = self.distancia(puntos, 300, 386)
        comisura_izq_altura = puntos[61, 1] - puntos[0, 1]
        comisura_der_altura = puntos[291, 1] - puntos[0, 1]
        labio_superior = puntos[13, 1] - puntos[0, 1]
        labio_inferior = puntos[14, 1] - puntos[0, 1]
        nariz_boca = self.distancia(puntos, 1, 13)
        barbilla_boca = self.distancia(puntos, 152, 14)

        medidas = {
            "ancho_cara": ancho_cara,
            "alto_cara": alto_cara,
            "boca_ancho": self.valor_seguro(boca_ancho, ancho_cara),
            "boca_abierta": self.valor_seguro(boca_abierta, alto_cara),
            "ojo_izq": self.valor_seguro(ojo_izq, alto_cara),
            "ojo_der": self.valor_seguro(ojo_der, alto_cara),
            "ceja_izq": self.valor_seguro(ceja_izq, alto_cara),
            "ceja_der": self.valor_seguro(ceja_der, alto_cara),
            "ceja_centro_izq": self.valor_seguro(ceja_centro_izq, alto_cara),
            "ceja_centro_der": self.valor_seguro(ceja_centro_der, alto_cara),
            "comisura_izq_altura": float(comisura_izq_altura),
            "comisura_der_altura": float(comisura_der_altura),
            "labio_superior": float(labio_superior),
            "labio_inferior": float(labio_inferior),
            "nariz_boca": self.valor_seguro(nariz_boca, alto_cara),
            "barbilla_boca": self.valor_seguro(barbilla_boca, alto_cara)
        }
        self.ultimas_medidas = medidas

        valores = [
            medidas["boca_ancho"], medidas["boca_abierta"], medidas["ojo_izq"], medidas["ojo_der"],
            medidas["ceja_izq"], medidas["ceja_der"], medidas["ceja_centro_izq"], medidas["ceja_centro_der"],
            medidas["comisura_izq_altura"], medidas["comisura_der_altura"], medidas["labio_superior"],
            medidas["labio_inferior"], medidas["nariz_boca"], medidas["barbilla_boca"]
        ]

        pares = [(61, 291), (78, 308), (13, 14), (159, 145), (386, 374), (70, 300), (105, 334),
                 (1, 152), (10, 152), (33, 263), (0, 17), (199, 200)]
        for a, b in pares:
            valores.append(self.valor_seguro(self.distancia(puntos, a, b), max(ancho_cara, alto_cara)))

        return np.array(valores, dtype=np.float32)

    def textura(self, gris):
        rasgos_hog = hog(
            gris,
            orientations=9,
            pixels_per_cell=(12, 12),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        ).astype(np.float32)

        lbp = local_binary_pattern(gris, P=16, R=2, method="uniform")
        histograma, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 19), range=(0, 18))
        histograma = histograma.astype(np.float32)
        histograma = histograma / (histograma.sum() + 1e-6)

        partes = []
        for fila in range(2):
            for columna in range(2):
                bloque = gris[fila * 80:(fila + 1) * 80, columna * 80:(columna + 1) * 80]
                partes.extend([float(bloque.mean()) / 255.0, float(bloque.std()) / 255.0])

        intensidad = np.array(partes, dtype=np.float32)
        return np.concatenate([rasgos_hog, histograma, intensidad])

    def extraer(self, cara):
        cara, gris = self.preparar_gris(cara)
        puntos = self.puntos(cara)
        geometria = self.geometria(puntos)
        textura = self.textura(gris)
        return np.concatenate([textura, geometria]).astype(np.float32)

    def cerrar(self):
        if self.malla is not None and hasattr(self.malla, "close"):
            self.malla.close()
