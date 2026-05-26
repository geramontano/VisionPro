from pathlib import Path
import json
import numpy as np

from nombres import EMOCIONES


class CalibradorDistancia:
    def __init__(self, ruta="modelo/perfiles_distancia.npz", max_muestras=240):
        self.ruta = Path(ruta)
        self.max_muestras = max_muestras
        self.muestras = {e: [] for e in EMOCIONES}
        self.cargando()

    def cargando(self):
        if not self.ruta.exists():
            return
        try:
            data = np.load(self.ruta, allow_pickle=True)
            clases = list(data["clases"])
            for emocion in EMOCIONES:
                if emocion in clases:
                    idx = clases.index(emocion)
                    arr = data[f"muestras_{idx}"]
                    if arr.ndim == 2 and len(arr) > 0:
                        self.muestras[emocion] = [fila.astype(np.float32) for fila in arr]
        except Exception:
            pass

    def conteos(self):
        return {e: len(self.muestras.get(e, [])) for e in EMOCIONES}

    def agregar(self, emocion, vector):
        if emocion not in self.muestras:
            return
        v = np.array(vector, dtype=np.float32).reshape(-1)
        if len(v) == 0:
            return
        self.muestras[emocion].append(v)
        if len(self.muestras[emocion]) > self.max_muestras:
            self.muestras[emocion] = self.muestras[emocion][-self.max_muestras:]

    def centroides(self):
        salida = {}
        for emocion, lista in self.muestras.items():
            if not lista:
                continue
            arr = np.array(lista, dtype=np.float32)
            centro = np.median(arr, axis=0).astype(np.float32)
            salida[emocion] = centro
        return salida

    def guardar(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        centroides = self.centroides()
        clases = np.array(EMOCIONES)
        payload = {"clases": clases}
        resumen = {}
        for i, emocion in enumerate(EMOCIONES):
            arr = np.array(self.muestras.get(emocion, []), dtype=np.float32)
            if arr.size == 0:
                arr = np.zeros((0, 1), dtype=np.float32)
            payload[f"muestras_{i}"] = arr
            if emocion in centroides:
                resumen[emocion] = int(len(self.muestras[emocion]))
        np.savez_compressed(self.ruta, **payload)
        ruta_json = self.ruta.with_suffix(".json")
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)

    def vacio(self):
        return sum(len(v) for v in self.muestras.values()) == 0


def _normalizar(v):
    v = np.array(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v
    return v / n


def probabilidades_desde_perfiles(vector, clases, calibrador):
    if calibrador is None:
        return None
    centroides = calibrador.centroides()
    if not centroides:
        return None
    x = _normalizar(vector)
    puntajes = []
    for emocion in clases:
        if emocion in centroides:
            c = _normalizar(centroides[emocion])
            sim = float(np.dot(x, c))
            cantidad = len(calibrador.muestras.get(emocion, []))
            bono = min(0.045, cantidad / 4000.0)
            puntajes.append(sim + bono)
        else:
            puntajes.append(-1.0)
    puntajes = np.array(puntajes, dtype=np.float32)
    if np.all(puntajes < -0.5):
        return None
    puntajes = (puntajes + 1.0) / 2.0
    puntajes = np.maximum(puntajes, 1e-6)
    puntajes = np.exp((puntajes - np.max(puntajes)) * 8.0)
    puntajes = puntajes / (puntajes.sum() + 1e-6)
    return puntajes


def mezclar_con_perfiles(probabilidades, clases, vector, calibrador, fuerza=0.20, area_relativa=None):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    perfil = probabilidades_desde_perfiles(vector, clases, calibrador)
    if perfil is None:
        return probs

    fuerza_real = float(fuerza)
    if area_relativa is not None:
        if area_relativa < 0.020:
            fuerza_real *= 1.55
        elif area_relativa < 0.040:
            fuerza_real *= 1.25
        elif area_relativa > 0.090:
            fuerza_real *= 0.70

    fuerza_real = max(0.0, min(0.70, fuerza_real))
    mezcla = (1.0 - fuerza_real) * probs + fuerza_real * perfil
    mezcla = mezcla / (mezcla.sum() + 1e-6)
    return mezcla
