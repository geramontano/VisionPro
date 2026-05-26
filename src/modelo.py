import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Suavizador:
    def __init__(self, clases, ventana=9):
        self.clases = list(clases)
        self.ventana = ventana
        self.historial = []

    def agregar(self, probabilidades):
        self.historial.append(np.array(probabilidades, dtype=np.float32))
        if len(self.historial) > self.ventana:
            self.historial.pop(0)
        pesos = np.linspace(0.45, 1.0, len(self.historial), dtype=np.float32)
        pesos = pesos / pesos.sum()
        mezcla = np.zeros_like(self.historial[-1], dtype=np.float32)
        for p, w in zip(self.historial, pesos):
            mezcla += p * w
        return mezcla / (mezcla.sum() + 1e-6)


def crear_modelo():
    svm = Pipeline([
        ("escala", StandardScaler()),
        ("pca", PCA(n_components=0.97, svd_solver="full")),
        ("svm", SVC(C=9.0, gamma="scale", kernel="rbf", probability=True, class_weight="balanced"))
    ])

    logistica = Pipeline([
        ("escala", StandardScaler()),
        ("pca", PCA(n_components=0.97, svd_solver="full")),
        ("logistica", LogisticRegression(max_iter=5000, C=2.5, class_weight="balanced", multi_class="auto"))
    ])

    mlp = Pipeline([
        ("escala", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(160, 80), alpha=0.0015, learning_rate_init=0.0007, max_iter=550, random_state=7))
    ])

    extra = ExtraTreesClassifier(
        n_estimators=420,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=11,
        n_jobs=-1
    )

    bosque = RandomForestClassifier(
        n_estimators=260,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=13,
        n_jobs=-1
    )

    return VotingClassifier(
        estimators=[
            ("svm", svm),
            ("extra", extra),
            ("logistica", logistica),
            ("bosque", bosque),
            ("mlp", mlp)
        ],
        voting="soft",
        weights=[4, 3, 2, 2, 1],
        n_jobs=None
    )


def enfocar_probabilidades(probabilidades, temperatura=0.38):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    if temperatura is None or temperatura <= 0:
        return probs
    probs = np.maximum(probs, 1e-8)
    probs = probs ** (1.0 / temperatura)
    return probs / (probs.sum() + 1e-6)


def confianza_visual(probabilidades):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    orden = np.sort(probs)[::-1]
    mayor = float(orden[0])
    segundo = float(orden[1]) if len(orden) > 1 else 0.0
    margen = mayor - segundo
    valor = mayor + margen * 0.75
    return float(min(0.995, max(mayor, valor)))



def limitar_valor(valor, minimo=0.0, maximo=1.0):
    return float(max(minimo, min(maximo, valor)))


def proteger_neutro(clases, probabilidades, neutro_modelo=None, fuerza=0.72, umbral=0.70):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    if neutro_modelo is None or "neutro" not in clases:
        return probs

    referencia = np.array(neutro_modelo, dtype=np.float32)
    if len(referencia) != len(probs):
        return probs
    referencia = referencia / (referencia.sum() + 1e-6)

    distancia = float(np.abs(probs - referencia).sum()) / 2.0
    parecido = 1.0 - distancia
    activacion = limitar_valor((parecido - umbral) / (1.0 - umbral + 1e-6))
    if activacion <= 0:
        return probs

    indice_neutro = clases.index("neutro")
    ajustadas = probs.copy()
    dominante = int(np.argmax(probs))
    emocion_dominante = clases[dominante]

    fuerza_real = fuerza * activacion
    if emocion_dominante in ["asustado", "triste", "enojado"]:
        fuerza_real = min(0.92, fuerza_real * 1.22)

    for i in range(len(ajustadas)):
        if i != indice_neutro:
            ajustadas[i] *= (1.0 - fuerza_real)
    ajustadas[indice_neutro] += fuerza_real
    ajustadas = ajustadas / (ajustadas.sum() + 1e-6)
    return ajustadas



def proteger_cara_seria(clases, probabilidades, medidas=None, neutro_medidas=None, neutro_modelo=None, fuerza=0.86, margen_enojo=0.18):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    if "neutro" not in clases or "enojado" not in clases:
        return probs

    indice_neutro = clases.index("neutro")
    indice_enojado = clases.index("enojado")
    dominante = int(np.argmax(probs))
    if dominante != indice_enojado:
        return probs

    ajuste = 0.0

    if neutro_modelo is not None:
        referencia = np.array(neutro_modelo, dtype=np.float32)
        if len(referencia) == len(probs):
            referencia = referencia / (referencia.sum() + 1e-6)
            distancia = float(np.abs(probs - referencia).sum()) / 2.0
            parecido = 1.0 - distancia
            enojo_base = float(referencia[indice_enojado])
            enojo_actual = float(probs[indice_enojado])
            exceso_enojo = enojo_actual - enojo_base
            if parecido > 0.50 and exceso_enojo < margen_enojo:
                ajuste = max(ajuste, fuerza * (0.55 + parecido * 0.45))
            if parecido > 0.62 and exceso_enojo < margen_enojo + 0.10:
                ajuste = max(ajuste, fuerza * 0.78)

    if medidas and neutro_medidas:
        ceja_centro = (medidas.get("ceja_centro_izq", 0.0) + medidas.get("ceja_centro_der", 0.0)) / 2
        ojo = (medidas.get("ojo_izq", 0.0) + medidas.get("ojo_der", 0.0)) / 2
        boca = medidas.get("boca_abierta", 0.0)
        comisura = (medidas.get("comisura_izq_altura", 0.0) + medidas.get("comisura_der_altura", 0.0)) / 2

        ceja_n = (neutro_medidas.get("ceja_centro_izq", ceja_centro) + neutro_medidas.get("ceja_centro_der", ceja_centro)) / 2
        ojo_n = (neutro_medidas.get("ojo_izq", ojo) + neutro_medidas.get("ojo_der", ojo)) / 2
        boca_n = neutro_medidas.get("boca_abierta", boca)
        comisura_n = (neutro_medidas.get("comisura_izq_altura", comisura) + neutro_medidas.get("comisura_der_altura", comisura)) / 2

        ceja_fruncida = ceja_centro < ceja_n - 0.010
        ojo_tenso = ojo < ojo_n - 0.004
        boca_tensa = boca < boca_n + 0.006 and comisura > comisura_n - 0.006
        enojo_real = ceja_fruncida and (ojo_tenso or boca_tensa)
        if not enojo_real:
            ajuste = max(ajuste, fuerza)

    if ajuste <= 0:
        return probs

    ajustadas = probs.copy()
    traslado = ajustadas[indice_enojado] * min(0.92, ajuste)
    ajustadas[indice_enojado] -= traslado
    ajustadas[indice_neutro] += traslado
    ajustadas = ajustadas / (ajustadas.sum() + 1e-6)
    return ajustadas


def recuperar_enojo(clases, probabilidades, reglas=None, medidas=None, neutro_medidas=None, fuerza=0.16):
    probs = np.array(probabilidades, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-6)
    if "enojado" not in clases or "neutro" not in clases:
        return probs

    indice_enojado = clases.index("enojado")
    indice_neutro = clases.index("neutro")
    enojo = float(probs[indice_enojado])
    neutro = float(probs[indice_neutro])

    regla_enojo = 0.0
    if reglas:
        regla_enojo = float(reglas.get("enojado", 0.0))

    senal_ceja = 0.0
    if medidas and neutro_medidas:
        ceja_centro = (medidas.get("ceja_centro_izq", 0.0) + medidas.get("ceja_centro_der", 0.0)) / 2
        ceja_n = (neutro_medidas.get("ceja_centro_izq", ceja_centro) + neutro_medidas.get("ceja_centro_der", ceja_centro)) / 2
        ojo = (medidas.get("ojo_izq", 0.0) + medidas.get("ojo_der", 0.0)) / 2
        ojo_n = (neutro_medidas.get("ojo_izq", ojo) + neutro_medidas.get("ojo_der", ojo)) / 2
        boca = medidas.get("boca_abierta", 0.0)
        boca_n = neutro_medidas.get("boca_abierta", boca)
        baja_ceja = limitar_valor((ceja_n - ceja_centro - 0.0035) * 34.0)
        ojo_tenso = limitar_valor((ojo_n - ojo - 0.0015) * 26.0)
        boca_controlada = limitar_valor((0.030 - abs(boca - boca_n)) * 18.0)
        senal_ceja = max(baja_ceja, baja_ceja * 0.70 + ojo_tenso * 0.20 + boca_controlada * 0.10)

    activacion = max(0.0, regla_enojo - 0.07, senal_ceja)
    if activacion <= 0.0:
        return probs

    dominante = int(np.argmax(probs))
    si_casi_enojo = enojo >= neutro - 0.16 or dominante == indice_enojado
    if not si_casi_enojo:
        return probs

    aumento = min(0.34, fuerza * (0.55 + activacion))
    ajustadas = probs.copy()
    toma = min(ajustadas[indice_neutro] * 0.45, aumento)
    ajustadas[indice_neutro] -= toma
    ajustadas[indice_enojado] += toma
    return ajustadas / (ajustadas.sum() + 1e-6)
