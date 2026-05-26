import numpy as np

from nombres import EMOCIONES


def limitar(x, bajo=0.0, alto=1.0):
    return float(max(bajo, min(alto, x)))


def reglas_desde_medidas(medidas, neutro=None):
    base = {emocion: 0.05 for emocion in EMOCIONES}
    if not medidas:
        base["neutro"] = 0.75
        return base

    boca = medidas.get("boca_abierta", 0.0)
    ancho = medidas.get("boca_ancho", 0.0)
    ojo = (medidas.get("ojo_izq", 0.0) + medidas.get("ojo_der", 0.0)) / 2
    ceja = (medidas.get("ceja_izq", 0.0) + medidas.get("ceja_der", 0.0)) / 2
    ceja_centro = (medidas.get("ceja_centro_izq", 0.0) + medidas.get("ceja_centro_der", 0.0)) / 2
    comisura = (medidas.get("comisura_izq_altura", 0.0) + medidas.get("comisura_der_altura", 0.0)) / 2

    if neutro:
        boca_n = neutro.get("boca_abierta", boca)
        ancho_n = neutro.get("boca_ancho", ancho)
        ojo_n = (neutro.get("ojo_izq", ojo) + neutro.get("ojo_der", ojo)) / 2
        ceja_n = (neutro.get("ceja_izq", ceja) + neutro.get("ceja_der", ceja)) / 2
        ceja_centro_n = (neutro.get("ceja_centro_izq", ceja_centro) + neutro.get("ceja_centro_der", ceja_centro)) / 2
        comisura_n = (neutro.get("comisura_izq_altura", comisura) + neutro.get("comisura_der_altura", comisura)) / 2
    else:
        boca_n = 0.018
        ancho_n = 0.36
        ojo_n = 0.018
        ceja_n = 0.075
        ceja_centro_n = 0.075
        comisura_n = comisura

    cambio_boca = boca - boca_n
    cambio_ancho = ancho - ancho_n
    cambio_ojo = ojo - ojo_n
    cambio_ceja = ceja - ceja_n
    cambio_ceja_centro = ceja_centro - ceja_centro_n
    cambio_comisura = comisura_n - comisura

    feliz = 0.18 + limitar(cambio_ancho * 5.8) * 0.46 + limitar(cambio_comisura * 12.0) * 0.36
    sorprendido = 0.10 + limitar(cambio_boca * 9.0) * 0.45 + limitar(cambio_ojo * 10.0) * 0.23 + limitar(cambio_ceja * 6.5) * 0.22
    asustado = 0.08 + limitar(cambio_ojo * 9.5) * 0.32 + limitar(cambio_ceja * 8.0) * 0.28 + limitar(cambio_boca * 5.0) * 0.22
    ceño_real = limitar((-cambio_ceja_centro - 0.0035) * 24.0)
    ojos_tensos = limitar((-cambio_ojo - 0.0015) * 22.0)
    boca_tensa = limitar((-abs(cambio_boca) + 0.018) * 14.0)
    enojado = 0.055 + ceño_real * 0.66 + (ceño_real * ojos_tensos) * 0.24 + (ceño_real * boca_tensa) * 0.18
    triste = 0.11 + limitar(-cambio_comisura * 11.0) * 0.35 + limitar(-cambio_ancho * 5.5) * 0.22 + limitar(-cambio_ceja * 4.0) * 0.12

    movimiento = abs(cambio_boca) * 7.5 + abs(cambio_ancho) * 3.5 + abs(cambio_ojo) * 7.0 + abs(cambio_ceja) * 3.5 + abs(cambio_comisura) * 7.0
    neutro_valor = limitar(1.08 - movimiento)

    base["feliz"] = feliz
    base["sorprendido"] = sorprendido * 0.80
    base["asustado"] = asustado * 0.88
    base["enojado"] = enojado * 0.82
    base["triste"] = triste
    base["neutro"] = max(neutro_valor, 0.12)

    total = sum(base.values()) + 1e-6
    return {emocion: valor / total for emocion, valor in base.items()}


def mezclar_modelo_y_reglas(clases, probs_modelo, reglas, peso_modelo=0.78):
    vector_reglas = np.array([reglas.get(c, 0.0) for c in clases], dtype=np.float32)
    vector_reglas = vector_reglas / (vector_reglas.sum() + 1e-6)
    if probs_modelo is None:
        return vector_reglas
    probs_modelo = np.array(probs_modelo, dtype=np.float32)
    mezcla = peso_modelo * probs_modelo + (1 - peso_modelo) * vector_reglas
    return mezcla / (mezcla.sum() + 1e-6)
