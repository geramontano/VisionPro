import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from cara import DetectorCara
from distancia import CalibradorDistancia, mezclar_con_perfiles
from mirada import CalibradorMirada, MedidorDistancia, RegistroMirada, SeguidorMirada, tamano_pantalla
from modelo import Suavizador, enfocar_probabilidades, confianza_visual, proteger_neutro, proteger_cara_seria, recuperar_enojo
from nombres import COLORES, EMOCIONES
from rasgos import ExtractorRasgos
from reglas import mezclar_modelo_y_reglas, reglas_desde_medidas


pantalla_ancho, pantalla_alto = tamano_pantalla()

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="modelo/emociones_modelo.joblib")
parser.add_argument("--camara", type=int, default=0)
parser.add_argument("--ancho", type=int, default=1280)
parser.add_argument("--alto", type=int, default=720)
parser.add_argument("--peso-modelo", type=float, default=0.96)
parser.add_argument("--temperatura", type=float, default=0.46)
parser.add_argument("--proteccion-neutro", type=float, default=0.88)
parser.add_argument("--umbral-neutro", type=float, default=0.58)
parser.add_argument("--proteccion-cara-seria", type=float, default=0.68)
parser.add_argument("--margen-enojo", type=float, default=0.11)
parser.add_argument("--recuperacion-enojo", type=float, default=0.18)
parser.add_argument("--perfil-distancia", default="modelo/perfiles_distancia.npz")
parser.add_argument("--peso-distancia", type=float, default=0.0)
parser.add_argument("--calibracion-mirada", default="modelo/calibracion_mirada.npz")
parser.add_argument("--log-mirada", default="registros/mirada_emociones_v2.csv")
parser.add_argument("--intervalo-log", type=float, default=0.25)
parser.add_argument("--pantalla-ancho", type=int, default=pantalla_ancho)
parser.add_argument("--pantalla-alto", type=int, default=pantalla_alto)
parser.add_argument("--calibracion-distancia", default="modelo/calibracion_distancia.npz")
parser.add_argument("--distancia-calibracion", type=float, default=60.0)
args = parser.parse_args()

paquete = None
modelo = None
clases = EMOCIONES
if Path(args.modelo).exists():
    paquete = joblib.load(args.modelo)
    modelo = paquete["modelo"]
    clases = list(paquete.get("clases", modelo.classes_))
    print("Modelo cargado:", args.modelo)
    print("Accuracy validacion guardada:", paquete.get("accuracy_validacion", "sin dato"))
else:
    print("No encontre modelo entrenado. Usare reglas faciales solamente.")

detector = DetectorCara(confianza=0.48)
extractor = ExtractorRasgos()
seguidor_mirada = SeguidorMirada()
calibrador_mirada = CalibradorMirada(ruta=args.calibracion_mirada, ancho=args.pantalla_ancho, alto=args.pantalla_alto)
registro_mirada = RegistroMirada(ruta=args.log_mirada, intervalo=args.intervalo_log)
medidor_distancia = MedidorDistancia(ruta=args.calibracion_distancia)
suavizador = Suavizador(clases, ventana=8)
calibracion_medidas = deque(maxlen=55)
calibracion_modelo = deque(maxlen=55)
neutro_medidas = None
neutro_modelo = None
calibrador_distancia = CalibradorDistancia(ruta=args.perfil_distancia)
modo_distancia = None
capturadas_distancia = 0
captura = cv2.VideoCapture(args.camara)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)

teclas_emocion = {
    ord("n"): "neutro",
    ord("f"): "feliz",
    ord("t"): "triste",
    ord("e"): "enojado",
    ord("a"): "asustado",
    ord("s"): "sorprendido",
}


def dibujar_texto(frame, texto, y, color=(240, 240, 240), escala=0.65):
    cv2.putText(frame, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2)


def punto_en_frame(x_rel, y_rel, frame):
    h, w = frame.shape[:2]
    return int(x_rel * w), int(y_rel * h)


def calibrar_mirada():
    puntos = [
        (0.08, 0.10), (0.30, 0.10), (0.50, 0.10), (0.70, 0.10), (0.92, 0.10),
        (0.08, 0.30), (0.30, 0.30), (0.50, 0.30), (0.70, 0.30), (0.92, 0.30),
        (0.08, 0.50), (0.30, 0.50), (0.50, 0.50), (0.70, 0.50), (0.92, 0.50),
        (0.08, 0.70), (0.30, 0.70), (0.50, 0.70), (0.70, 0.70), (0.92, 0.70),
        (0.08, 0.90), (0.30, 0.90), (0.50, 0.90), (0.70, 0.90), (0.92, 0.90),
    ]
    muestras = []
    print("Calibracion robusta de mirada iniciada.")
    print("Mantente a la misma distancia, no muevas la cabeza y sigue cada punto amarillo.")
    for numero, (rx, ry) in enumerate(puntos, start=1):
        lote = []
        cajas = []
        for paso in range(92):
            ok2, frame2 = captura.read()
            if not ok2:
                continue
            cara2, caja2 = detector.recortar(frame2)
            salida = seguidor_mirada.extraer(frame2, caja2)
            px, py = punto_en_frame(rx, ry, frame2)
            cv2.circle(frame2, (px, py), 19, (0, 255, 255), -1)
            cv2.circle(frame2, (px, py), 37, (0, 170, 255), 3)
            cv2.putText(frame2, f"Punto {numero}/25: mira el circulo amarillo", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 220, 255), 2)
            cv2.putText(frame2, "No sigas el punto con la cabeza, solo con los ojos", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2)
            if caja2 is not None:
                x1, y1, x2, y2 = caja2
                cajas.append((x1, y1, x2, y2))
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (180, 220, 255), 1)
            if salida is not None:
                ox, oy = salida["ojo"]
                cv2.circle(frame2, (ox, oy), 5, (255, 180, 60), -1)
                if paso > 28:
                    lote.append(salida["feat"])
            cv2.imshow("Detector de emociones", frame2)
            cv2.waitKey(1)
        if lote:
            arr = np.array(lote, dtype=np.float32)
            if len(arr) > 8:
                centro = np.median(arr, axis=0)
                dist = np.linalg.norm(arr - centro, axis=1)
                corte = np.percentile(dist, 82)
                arr = arr[dist <= corte]
            for feat in arr:
                muestras.append((feat, rx * args.pantalla_ancho, ry * args.pantalla_alto))
    ok = calibrador_mirada.ajustar(muestras)
    seguidor_mirada.suavizador.reiniciar()
    if ok:
        print("Calibracion de mirada lista:", args.calibracion_mirada)
        print("Muestras usadas:", len(muestras))
    else:
        print("No se pudo calibrar mirada. Acercate mas, mejora la luz o evita lentes con reflejo.")


def calibrar_distancia_en_vivo():
    print("Calibrando distancia a", args.distancia_calibracion, "cm. Quedate quieto frente a la camara.")
    anchos = []
    for paso in range(90):
        ok2, frame2 = captura.read()
        if not ok2:
            continue
        _, caja2 = detector.recortar(frame2)
        if caja2 is not None:
            x1, y1, x2, y2 = caja2
            anchos.append(x2 - x1)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (120, 255, 180), 2)
        cv2.putText(frame2, "Calibrando distancia: no te muevas", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 255, 180), 2)
        cv2.putText(frame2, f"Distancia real usada: {args.distancia_calibracion} cm", (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (120, 255, 180), 2)
        cv2.imshow("Detector de emociones", frame2)
        cv2.waitKey(1)
    if medidor_distancia.calibrar_con_anchos(anchos, args.distancia_calibracion):
        print("Distancia calibrada con", len(anchos), "muestras a", args.distancia_calibracion, "cm")
    else:
        print("No pude calibrar distancia. Acercate, mejora la luz o centra la cara.")


print("Controles: c calibrar neutro serio, r reiniciar suavizado, q salir")
print("Distancia: n neutro | f feliz | t triste | e enojado | a asustado | s sorprendido | x parar | g guardar")
print("Mirada: o calibrar mirada robusta | d calibrar distancia | b borrar distancia | l iniciar/detener registro | v reiniciar punto")
print("Detector de cara:", getattr(detector, "tipo", "sin dato"))
print("Face mesh activo:", extractor.malla is not None)
print("Mirada backend:", seguidor_mirada.backend)
print("Pantalla usada para mirada:", args.pantalla_ancho, "x", args.pantalla_alto)
print("Registro mirada:", args.log_mirada)
print("Distancia calibrada:", medidor_distancia.calibrado())
print("Perfiles a distancia:", calibrador_distancia.conteos())
print("Correccion de emocion por perfiles distancia:", "activa" if args.peso_distancia > 0 else "apagada")

while True:
    ok, frame = captura.read()
    if not ok:
        break

    cara, caja = detector.recortar(frame)
    emocion = "sin cara"
    confianza = 0.0
    probs = None
    reglas = {e: 0.0 for e in EMOCIONES}
    vector_actual = None
    medidas = None
    area_relativa = None
    datos_mirada = None
    punto_mirada = None
    distancia_cm = medidor_distancia.estimar(caja) if caja is not None else None

    if cara is not None:
        try:
            vector_actual = extractor.extraer(cara).reshape(-1)
            rasgos = vector_actual.reshape(1, -1)
            medidas = extractor.ultimas_medidas.copy()
            reglas = reglas_desde_medidas(medidas, neutro=neutro_medidas)
            if modelo is not None:
                probs_modelo = modelo.predict_proba(rasgos)[0]
                probs = mezclar_modelo_y_reglas(clases, probs_modelo, reglas, peso_modelo=args.peso_modelo)
            else:
                probs = np.array([reglas.get(c, 0.0) for c in clases], dtype=np.float32)
                probs = probs / (probs.sum() + 1e-6)

            if caja is not None:
                x1, y1, x2, y2 = caja
                area_relativa = float(max(1, (x2 - x1) * (y2 - y1))) / float(max(1, frame.shape[0] * frame.shape[1]))

            probs = proteger_neutro(clases, probs, neutro_modelo=neutro_modelo, fuerza=args.proteccion_neutro, umbral=args.umbral_neutro)
            probs = proteger_cara_seria(clases, probs, medidas=medidas, neutro_medidas=neutro_medidas, neutro_modelo=neutro_modelo, fuerza=args.proteccion_cara_seria, margen_enojo=args.margen_enojo)
            probs = recuperar_enojo(clases, probs, reglas=reglas, medidas=medidas, neutro_medidas=neutro_medidas, fuerza=args.recuperacion_enojo)
            probs = mezclar_con_perfiles(probs, clases, vector_actual, calibrador_distancia, fuerza=args.peso_distancia, area_relativa=area_relativa)
            probs = enfocar_probabilidades(probs, temperatura=args.temperatura)
            probs = suavizador.agregar(probs)
            probs = proteger_neutro(clases, probs, neutro_modelo=neutro_modelo, fuerza=args.proteccion_neutro * 0.58, umbral=args.umbral_neutro)
            probs = proteger_cara_seria(clases, probs, medidas=medidas, neutro_medidas=neutro_medidas, neutro_modelo=neutro_modelo, fuerza=args.proteccion_cara_seria * 0.42, margen_enojo=args.margen_enojo)
            probs = recuperar_enojo(clases, probs, reglas=reglas, medidas=medidas, neutro_medidas=neutro_medidas, fuerza=args.recuperacion_enojo * 0.55)
            probs = mezclar_con_perfiles(probs, clases, vector_actual, calibrador_distancia, fuerza=args.peso_distancia * 0.65, area_relativa=area_relativa)
            probs = enfocar_probabilidades(probs, temperatura=0.86)
            indice = int(np.argmax(probs))
            emocion = clases[indice]
            confianza = confianza_visual(probs)
        except Exception:
            emocion = "error"
            confianza = 0.0

        x1, y1, x2, y2 = caja
        color = COLORES.get(emocion, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, emocion + " " + str(round(confianza * 100, 1)) + "%", (x1, max(y1 - 12, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if modo_distancia is not None and vector_actual is not None:
            calibrador_distancia.agregar(modo_distancia, vector_actual)
            capturadas_distancia += 1
            if capturadas_distancia % 10 == 0:
                calibrador_distancia.guardar()

    datos_mirada = seguidor_mirada.extraer(frame, caja)
    if datos_mirada is not None:
        mx, my, rx, ry = calibrador_mirada.predecir(datos_mirada["feat"])
        sx, sy = seguidor_mirada.suavizador.agregar(mx, my)
        rx = sx / max(1, args.pantalla_ancho)
        ry = sy / max(1, args.pantalla_alto)
        px = int(rx * frame.shape[1])
        py = int(ry * frame.shape[0])
        punto_mirada = (int(sx), int(sy), rx, ry)
        ox, oy = datos_mirada["ojo"]
        cv2.circle(frame, (ox, oy), 5, (255, 180, 60), -1)
        cv2.circle(frame, (px, py), 13, (50, 255, 255), 2)
        cv2.line(frame, (px - 18, py), (px + 18, py), (50, 255, 255), 2)
        cv2.line(frame, (px, py - 18), (px, py + 18), (50, 255, 255), 2)
        registro_mirada.agregar(emocion, confianza, sx, sy, rx, ry, datos_mirada["backend"], area_relativa, distancia_cm=distancia_cm, distancia_calibrada=medidor_distancia.calibrado())

    y0 = 30
    dibujar_texto(frame, "c neutro | o mirada robusta | d distancia | b reset dist | l log | q salir", y0)
    dibujar_texto(frame, "n/f/t/e/a/s perfiles distancia | x parar | g guardar", y0 + 28, (240, 220, 180), 0.58)

    if neutro_medidas is None and neutro_modelo is None:
        dibujar_texto(frame, "Neutro no calibrado", y0 + 60, (80, 180, 255))
    elif neutro_modelo is not None and neutro_medidas is None:
        dibujar_texto(frame, "Neutro calibrado por modelo", y0 + 60, (120, 255, 120))
    else:
        dibujar_texto(frame, "Neutro calibrado", y0 + 60, (120, 255, 120))

    conteos = calibrador_distancia.conteos()
    total_perfiles = sum(conteos.values())
    estado_distancia = "sin perfiles distancia" if total_perfiles == 0 else f"perfiles distancia {total_perfiles}"
    if modo_distancia is not None:
        estado_distancia = f"capturando {modo_distancia}: {capturadas_distancia}"
    dibujar_texto(frame, estado_distancia, y0 + 88, (140, 255, 255), 0.58)

    estado_mirada = "mirada calibrada " + calibrador_mirada.modo if calibrador_mirada.calibrado() else "mirada no calibrada"
    estado_log = "log ON" if registro_mirada.activo else "log OFF"
    texto_distancia = "distancia --" if distancia_cm is None else "distancia " + str(round(distancia_cm, 1)) + " cm"
    if medidor_distancia.calibrado():
        texto_distancia += " cal"
    dibujar_texto(frame, estado_mirada + " | " + estado_log + " | " + seguidor_mirada.backend, y0 + 114, (120, 220, 255), 0.58)
    dibujar_texto(frame, texto_distancia, y0 + 140, (120, 255, 180), 0.58)
    if punto_mirada is not None:
        mx, my, rx, ry = punto_mirada
        dibujar_texto(frame, f"mirada pantalla: {mx},{my} zona {registro_mirada.zona(rx, ry)}", y0 + 166, (120, 220, 255), 0.55)

    altura_barra = 18
    inicio_y = y0 + 194
    for i, clase in enumerate(clases):
        valor = 0.0
        if probs is not None and len(probs) == len(clases):
            valor = float(probs[i])
        largo = int(260 * valor)
        color = COLORES.get(clase, (220, 220, 220))
        etiqueta = f"{clase} ({conteos.get(clase, 0)})"
        cv2.putText(frame, etiqueta, (20, inicio_y + i * 28 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1)
        cv2.rectangle(frame, (165, inicio_y + i * 28), (165 + largo, inicio_y + i * 28 + altura_barra), color, -1)
        cv2.rectangle(frame, (165, inicio_y + i * 28), (425, inicio_y + i * 28 + altura_barra), (180, 180, 180), 1)

    cv2.imshow("Detector de emociones", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break
    if tecla == ord("r"):
        suavizador = Suavizador(clases, ventana=8)
        seguidor_mirada.suavizador.reiniciar()
    if tecla == ord("v"):
        seguidor_mirada.suavizador.reiniciar()
    if tecla == ord("l"):
        registro_mirada.activo = not registro_mirada.activo
        print("Registro mirada:", "activo" if registro_mirada.activo else "pausado", args.log_mirada)
    if tecla == ord("o"):
        calibrar_mirada()
    if tecla == ord("d"):
        calibrar_distancia_en_vivo()
    if tecla == ord("b"):
        medidor_distancia.reset()
        print("Calibracion de distancia borrada. Presiona d para recalibrar.")
    if tecla == ord("x"):
        modo_distancia = None
        capturadas_distancia = 0
        calibrador_distancia.guardar()
        print("Captura de distancia detenida")
    if tecla == ord("g"):
        calibrador_distancia.guardar()
        calibrador_mirada.guardar()
        medidor_distancia.guardar()
        print("Perfiles de distancia guardados:", calibrador_distancia.conteos())
        print("Calibracion mirada guardada:", args.calibracion_mirada)
        print("Calibracion distancia guardada:", args.calibracion_distancia)
    if tecla in teclas_emocion:
        modo_distancia = teclas_emocion[tecla]
        capturadas_distancia = 0
        print("Capturando distancia para:", modo_distancia)
        print("Alejate y manten esa emocion. Presiona x para parar o cambia a otra emocion.")
        suavizador = Suavizador(clases, ventana=8)
    if tecla == ord("c"):
        calibracion_medidas.clear()
        calibracion_modelo.clear()
        print("Mantente neutro o serio normal unos segundos, sin fruncir el ceno")
        for _ in range(70):
            ok2, frame2 = captura.read()
            if not ok2:
                continue
            cara2, _ = detector.recortar(frame2)
            if cara2 is None:
                cv2.putText(frame2, "No veo la cara para calibrar", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 120, 255), 2)
                cv2.imshow("Detector de emociones", frame2)
                cv2.waitKey(1)
                continue
            try:
                rasgos2 = extractor.extraer(cara2).reshape(1, -1)
                if extractor.ultimas_medidas:
                    calibracion_medidas.append(extractor.ultimas_medidas.copy())
                if modelo is not None:
                    p2 = modelo.predict_proba(rasgos2)[0]
                    p2 = np.array(p2, dtype=np.float32)
                    p2 = p2 / (p2.sum() + 1e-6)
                    calibracion_modelo.append(p2)
            except Exception:
                pass
            cv2.putText(frame2, "Calibrando neutro serio", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 220, 255), 2)
            cv2.imshow("Detector de emociones", frame2)
            cv2.waitKey(1)
        if calibracion_medidas:
            neutro_medidas = {}
            llaves = calibracion_medidas[0].keys()
            for llave in llaves:
                neutro_medidas[llave] = float(np.median([m[llave] for m in calibracion_medidas if llave in m]))
        if calibracion_modelo:
            matriz = np.array(list(calibracion_modelo), dtype=np.float32)
            neutro_modelo = np.median(matriz, axis=0)
            neutro_modelo = neutro_modelo / (neutro_modelo.sum() + 1e-6)
            print("Patron neutro del modelo:")
            for clase, valor in zip(clases, neutro_modelo):
                print(clase, round(float(valor), 4))
        if calibracion_medidas or calibracion_modelo:
            print("Calibracion lista")
        else:
            print("No se pudo calibrar. Acercate mas o mejora la luz")
        suavizador = Suavizador(clases, ventana=8)

calibrador_distancia.guardar()
calibrador_mirada.guardar()
medidor_distancia.guardar()
captura.release()
cv2.destroyAllWindows()
detector.cerrar()
extractor.cerrar()
seguidor_mirada.cerrar()
