import argparse
from collections import deque
from pathlib import Path
import time

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
parser.add_argument("--proteccion-neutro", type=float, default=0.0)
parser.add_argument("--umbral-neutro", type=float, default=0.99)
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

accuracy_referencia = float(paquete.get("accuracy_validacion", 0.0)) if paquete is not None else 0.0

detector = DetectorCara(confianza=0.48)
extractor = ExtractorRasgos()
seguidor_mirada = SeguidorMirada()
calibrador_mirada = CalibradorMirada(ruta=args.calibracion_mirada, ancho=args.pantalla_ancho, alto=args.pantalla_alto)
registro_mirada = RegistroMirada(ruta=args.log_mirada, intervalo=args.intervalo_log)
medidor_distancia = MedidorDistancia(ruta=args.calibracion_distancia)
suavizador = Suavizador(clases, ventana=5)
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


# ===== EMOTION VISION PRO UI =====
BG_CANVAS = (18, 12, 10)          # #0a0c12 aprox en BGR
BG_PANEL = (25, 18, 14)
BG_PANEL_2 = (30, 22, 18)
BORDER = (48, 34, 30)             # #1e2230 aprox
BORDER_SOFT = (58, 44, 38)
TEXT_PRIMARY = (238, 236, 232)
TEXT_SECONDARY = (135, 129, 121)
TEXT_MUTED = (105, 101, 96)
STATUS_GREEN = (88, 156, 104)

def ui_emotion_color(nombre):
    colores = {
        "feliz": (122, 170, 214),         # amber muted
        "triste": (158, 140, 112),        # slate gold
        "enojado": (156, 112, 126),       # muted rose
        "asustado": (168, 134, 122),      # dusty mauve
        "sorprendido": (146, 172, 136),   # muted teal-green
        "neutro": (168, 168, 160),        # warm gray
        "sin cara": (110, 108, 104),
        "error": (120, 120, 170),
    }
    return colores.get(str(nombre).lower(), (150, 150, 150))


def ui_grid_overlay(canvas, spacing=28, color=(26, 22, 18)):
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    canvas[:] = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)


def ui_panel(canvas, x1, y1, x2, y2, fill=BG_PANEL, border=BORDER, alpha=0.96):
    h, w = canvas.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    canvas[:] = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)


def ui_text(canvas, texto, x, y, escala=0.50, color=TEXT_PRIMARY, grosor=1):
    cv2.putText(canvas, texto, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def ui_status_pill(canvas, x1, y1, x2, y2, texto_izq, texto_der=""):
    ui_panel(canvas, x1, y1, x2, y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    cy = int((y1 + y2) / 2)
    cv2.circle(canvas, (x1 + 20, cy - 1), 5, STATUS_GREEN, -1)
    ui_text(canvas, texto_izq, x1 + 34, cy + 5, 0.46, TEXT_SECONDARY, 1)
    if texto_der:
        ui_text(canvas, texto_der, x2 - 70, cy + 5, 0.46, TEXT_SECONDARY, 1)


def ui_metric_card(canvas, x1, y1, x2, y2, titulo, valor):
    ui_panel(canvas, x1, y1, x2, y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    ui_text(canvas, titulo.upper(), x1 + 18, y1 + 26, 0.38, TEXT_MUTED, 1)
    ui_text(canvas, valor, x1 + 18, y1 + 52, 0.72, TEXT_PRIMARY, 1)


def ui_emotion_bar(canvas, x1, y1, x2, label, value, color):
    value = max(0.0, min(1.0, float(value)))
    ui_text(canvas, label, x1, y1, 0.48, TEXT_PRIMARY, 1)
    ui_text(canvas, f"{value*100:.1f}%", x2 - 62, y1, 0.46, TEXT_SECONDARY, 1)

    bar_y = y1 + 10
    bar_h = 4
    cv2.rectangle(canvas, (x1, bar_y), (x2, bar_y + bar_h), BORDER, -1)
    fill = x1 + int((x2 - x1) * value)
    cv2.rectangle(canvas, (x1, bar_y), (fill, bar_y + bar_h), color, -1)


def ui_embed_image(canvas, image, x1, y1, x2, y2):
    if image is None or image.size == 0:
        return

    ph = max(1, y2 - y1)
    pw = max(1, x2 - x1)
    ih, iw = image.shape[:2]

    escala = min(pw / iw, ph / ih)
    nw = max(1, int(iw * escala))
    nh = max(1, int(ih * escala))

    resized = cv2.resize(image, (nw, nh))
    ox = x1 + (pw - nw) // 2
    oy = y1 + (ph - nh) // 2

    cv2.rectangle(canvas, (x1, y1), (x2, y2), BG_CANVAS, -1)
    canvas[oy:oy+nh, ox:ox+nw] = resized


def ui_timeline_chart(canvas, x1, y1, x2, y2, historial, clases):
    ui_panel(canvas, x1, y1, x2, y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    ui_text(canvas, "EMOTION TIMELINE", x1 + 18, y1 + 28, 0.44, TEXT_SECONDARY, 1)
    ui_text(canvas, "Last 30 seconds", x2 - 132, y1 + 28, 0.36, TEXT_MUTED, 1)

    px1 = x1 + 24
    px2 = x2 - 24
    py1 = y1 + 42
    py2 = y2 - 56

    # guias
    for k in range(5):
        yy = py1 + int((py2 - py1) * k / 4)
        cv2.line(canvas, (px1, yy), (px2, yy), BORDER, 1)

    ui_text(canvas, "0%", px1 - 6, py2 + 2, 0.32, TEXT_MUTED, 1)
    ui_text(canvas, "50%", px1 - 10, py1 + int((py2 - py1) * 0.5) + 4, 0.32, TEXT_MUTED, 1)
    ui_text(canvas, "100%", px1 - 12, py1 + 4, 0.32, TEXT_MUTED, 1)

    if len(historial) < 2:
        ui_text(canvas, "Collecting emotion samples...", px1, py1 + 30, 0.42, TEXT_MUTED, 1)
        return

    ult = historial[-1][0]
    inicio = max(0.0, ult - 30.0)
    datos = [item for item in historial if item[0] >= inicio]

    if len(datos) < 2:
        ui_text(canvas, "Collecting emotion samples...", px1, py1 + 30, 0.42, TEXT_MUTED, 1)
        return

    for clase in clases:
        puntos = []
        color = ui_emotion_color(clase)
        for t, probs_dict in datos:
            x = px1 + int((t - inicio) / 30.0 * (px2 - px1))
            valor = float(probs_dict.get(clase, 0.0))
            y = py2 - int(valor * (py2 - py1))
            puntos.append((x, y))
        if len(puntos) >= 2:
            cv2.polylines(canvas, [np.array(puntos, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    # leyenda
    lx = px1
    ly = y2 - 24
    wrap = 0
    for idx, clase in enumerate(clases):
        cx = lx + wrap * 110
        cy = ly + (idx // 4) * 18
        cv2.circle(canvas, (cx, cy - 3), 4, ui_emotion_color(clase), -1)
        ui_text(canvas, str(clase), cx + 10, cy, 0.34, TEXT_MUTED, 1)
        wrap += 1
        if wrap == 4:
            wrap = 0


def ui_radar_chart(canvas, x1, y1, x2, y2, clases, probs):
    ui_panel(canvas, x1, y1, x2, y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    ui_text(canvas, "EMOTION SPECTRUM", x1 + 18, y1 + 28, 0.44, TEXT_SECONDARY, 1)
    ui_text(canvas, f"{len(clases)} emotions", x2 - 96, y1 + 28, 0.36, TEXT_MUTED, 1)

    if probs is None or len(probs) == 0:
        ui_text(canvas, "No probabilities available", x1 + 24, y1 + 60, 0.42, TEXT_MUTED, 1)
        return

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2) + 10
    r = min((x2 - x1), (y2 - y1)) // 3

    n = len(clases)
    if n < 3:
        ui_text(canvas, "Radar chart requires 3+ classes", x1 + 24, y1 + 60, 0.42, TEXT_MUTED, 1)
        return

    # niveles
    for frac in [0.25, 0.50, 0.75, 1.0]:
        pts = []
        for i in range(n):
            ang = -np.pi / 2 + (2 * np.pi * i / n)
            px = cx + int(np.cos(ang) * r * frac)
            py = cy + int(np.sin(ang) * r * frac)
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], True, BORDER, 1)

    # ejes
    for i, clase in enumerate(clases):
        ang = -np.pi / 2 + (2 * np.pi * i / n)
        ex = cx + int(np.cos(ang) * r)
        ey = cy + int(np.sin(ang) * r)
        cv2.line(canvas, (cx, cy), (ex, ey), BORDER, 1)
        lx = cx + int(np.cos(ang) * (r + 18))
        ly = cy + int(np.sin(ang) * (r + 18))
        ui_text(canvas, str(clase), lx - 18, ly, 0.34, TEXT_MUTED, 1)

    vals = []
    for i, clase in enumerate(clases):
        ang = -np.pi / 2 + (2 * np.pi * i / n)
        v = float(probs[i])
        px = cx + int(np.cos(ang) * r * v)
        py = cy + int(np.sin(ang) * r * v)
        vals.append((px, py))

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [np.array(vals, dtype=np.int32)], (120, 120, 120))
    canvas[:] = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
    cv2.polylines(canvas, [np.array(vals, dtype=np.int32)], True, TEXT_PRIMARY, 1, cv2.LINE_AA)


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

inicio_sesion_dashboard = time.time()
ultimo_frame_dashboard = time.time()
fps_dashboard = 0.0
historial_emociones = deque(maxlen=240)
ultimo_historial_dashboard = 0.0
conteo_analisis_dashboard = 0

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

            # BLOQUEO VISUAL SIMPLE DE PERFIL A DISTANCIA
            # Si se presiona n/f/t/e/a/s, la app muestra esa emoción como salida
            # hasta presionar x. No muestra texto de calibración en pantalla.
            if modo_distancia is not None and modo_distancia in clases:
                probs_calibracion = np.zeros_like(probs, dtype=np.float32)
                probs_calibracion[clases.index(modo_distancia)] = 1.0
                probs = probs_calibracion
                emocion = modo_distancia
                confianza = 1.0

            # BLOQUEO VISUAL DURANTE CALIBRACION DE PERFIL A DISTANCIA
            # Si el usuario presiona n/f/t/e/a/s, todas las muestras capturadas
            # se consideran de esa emocion hasta presionar x. Esto no reentrena
            # el modelo en vivo; solamente fija la etiqueta de captura y la salida
            # visual para que el usuario vea claramente que esta calibrando.
            if modo_distancia is not None and modo_distancia in clases:
                probs_calibracion = np.zeros_like(probs, dtype=np.float32)
                probs_calibracion[clases.index(modo_distancia)] = 1.0
                probs = probs_calibracion
                emocion = modo_distancia
                confianza = 1.0
        except Exception:
            emocion = "error"
            confianza = 0.0

        x1, y1, x2, y2 = caja
        color = ui_emotion_color(emocion)

        # CAPTURA ACTIVA DE PERFIL A DISTANCIA
        # Mientras modo_distancia tenga una emocion seleccionada, todos los
        # vectores faciales detectados se agregan al perfil de esa emocion.
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

    now_dashboard = time.time()
    dt_dashboard = max(1e-6, now_dashboard - ultimo_frame_dashboard)
    fps_inst = 1.0 / dt_dashboard
    fps_dashboard = fps_inst if fps_dashboard <= 0 else (0.90 * fps_dashboard + 0.10 * fps_inst)
    ultimo_frame_dashboard = now_dashboard

    if probs is not None and len(probs) == len(clases):
        if now_dashboard - ultimo_historial_dashboard >= 0.5:
            historial_emociones.append(
                (
                    now_dashboard - inicio_sesion_dashboard,
                    {str(clases[i]): float(probs[i]) for i in range(len(clases))}
                )
            )
            ultimo_historial_dashboard = now_dashboard
            conteo_analisis_dashboard += 1

    frame_camara = frame.copy()

    h, w = frame.shape[:2]
    dashboard = np.full_like(frame, BG_CANVAS)
    ui_grid_overlay(dashboard, spacing=34, color=(24, 20, 18))

    margin = 24
    gap = 16

    # Header
    header_y1 = 16
    header_y2 = 62
    ui_text(dashboard, "Emotion Vision Pro", margin + 44, 38, 0.78, TEXT_PRIMARY, 2)
    ui_text(dashboard, "FACE ANALYSIS SYSTEM", margin + 44, 56, 0.36, TEXT_MUTED, 1)
    cv2.rectangle(dashboard, (margin, 16), (margin + 26, 42), BORDER, 1)
    cv2.circle(dashboard, (margin + 13, 29), 9, BORDER, 1)
    ui_status_pill(dashboard, w // 2 - 120, 18, w // 2 + 120, 48, "System Online", "v2.4.1")

    # metric cards
    cards_y1 = 78
    cards_y2 = 150
    total_card_w = w - 2 * margin - 3 * gap
    card_w = total_card_w // 4

    session_elapsed = int(now_dashboard - inicio_sesion_dashboard)
    session_txt = f"{session_elapsed//60:02d}:{session_elapsed%60:02d}"
    analyses_txt = str(conteo_analisis_dashboard)
    distance_txt = "--"
    if distancia_cm is not None:
        distance_txt = f"{distancia_cm:.1f} cm"
    fps_txt = f"{fps_dashboard:.0f}"

    metric_titles = ["Session Time", "Analyses", "Distance", "FPS"]
    metric_values = [session_txt, analyses_txt, distance_txt, fps_txt]

    for idx in range(4):
        x1 = margin + idx * (card_w + gap)
        x2 = x1 + card_w
        ui_metric_card(dashboard, x1, cards_y1, x2, cards_y2, metric_titles[idx], metric_values[idx])

    # main panels
    main_y1 = 176
    charts_y1 = max(480, h - 250)
    camera_x1 = margin
    camera_x2 = margin + int((w - 2 * margin - gap) * 0.64)
    camera_y2 = charts_y1 - gap

    dist_x1 = camera_x2 + gap
    dist_x2 = w - margin
    dist_y1 = main_y1
    dist_y2 = camera_y2

    timeline_x1 = margin
    timeline_x2 = margin + int((w - 2 * margin - gap) * 0.58)
    timeline_y1 = charts_y1
    timeline_y2 = h - 44

    radar_x1 = timeline_x2 + gap
    radar_x2 = w - margin
    radar_y1 = charts_y1
    radar_y2 = h - 44

    # camera panel
    ui_panel(dashboard, camera_x1, main_y1, camera_x2, camera_y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    ui_text(dashboard, "• LIVE" if cara is not None else "• IDLE", camera_x1 + 18, main_y1 + 30, 0.42, STATUS_GREEN if cara is not None else TEXT_MUTED, 1)

    distance_state = "Distance calibrated" if medidor_distancia.calibrado() else "Distance not calibrated"
    distance_value = "--"
    if distancia_cm is not None:
        distance_value = f"{distancia_cm:.1f} cm"
    ui_text(dashboard, distance_state + "  |  " + distance_value, camera_x2 - 270, main_y1 + 30, 0.38, TEXT_SECONDARY, 1)

    cam_pad = 12
    ui_embed_image(dashboard, frame_camara, camera_x1 + cam_pad, main_y1 + 42, camera_x2 - cam_pad, camera_y2 - cam_pad)

    # panel de distribucion
    ui_panel(dashboard, dist_x1, dist_y1, dist_x2, dist_y2, fill=BG_PANEL, border=BORDER, alpha=0.98)
    ui_text(dashboard, "EMOTION ANALYSIS", dist_x1 + 18, dist_y1 + 28, 0.44, TEXT_SECONDARY, 1)
    ui_text(dashboard, "• LIVE" if cara is not None else "• IDLE", dist_x2 - 74, dist_y1 + 28, 0.36, STATUS_GREEN if cara is not None else TEXT_MUTED, 1)

    if probs is not None:
        orden = np.argsort(probs)[::-1]
        base_y = dist_y1 + 66
        for fila, i in enumerate(orden):
            yy = base_y + fila * 42
            if yy + 12 > dist_y2 - 16:
                break
            clase = str(clases[i])
            valor = float(probs[i])
            ui_emotion_bar(dashboard, dist_x1 + 18, yy, dist_x2 - 18, clase.capitalize(), valor, ui_emotion_color(clase))
    else:
        ui_text(dashboard, "No emotion probabilities available", dist_x1 + 18, dist_y1 + 64, 0.42, TEXT_MUTED, 1)

    # charts
    ui_timeline_chart(dashboard, timeline_x1, timeline_y1, timeline_x2, timeline_y2, historial_emociones, list(clases))
    ui_radar_chart(dashboard, radar_x1, radar_y1, radar_x2, radar_y2, list(clases), probs if probs is not None else [])

    # etiqueta de prediccion sobre la camara
    if caja is not None and cara is not None:
        bx1 = camera_x1 + 18
        by1 = main_y1 + 18
        bx2 = bx1 + 220
        by2 = by1 + 34
        ui_panel(dashboard, bx1, by1, bx2, by2, fill=BG_PANEL_2, border=BORDER_SOFT, alpha=0.96)
        ui_text(dashboard, str(emocion).upper(), bx1 + 12, by1 + 22, 0.54, ui_emotion_color(emocion), 1)
        ui_text(dashboard, f"{confianza*100:.1f}%", bx2 - 64, by1 + 22, 0.46, TEXT_PRIMARY, 1)

    # footer
    footer_y = h - 14
    ui_text(dashboard, "Emotion Vision Pro", margin, footer_y - 10, 0.38, TEXT_SECONDARY, 1)
    distance_footer = "Distance calibrated" if medidor_distancia.calibrado() else "Distance not calibrated"
    ui_text(dashboard, "Powered by Machine Learning  |  " + distance_footer, margin, footer_y + 8, 0.32, TEXT_MUTED, 1)

    cv2.circle(dashboard, (w - 220, footer_y - 1), 4, STATUS_GREEN, -1)
    ui_text(dashboard, "API Connected", w - 208, footer_y + 4, 0.34, TEXT_SECONDARY, 1)

    cv2.circle(dashboard, (w - 110, footer_y - 1), 4, TEXT_MUTED, -1)
    ui_text(dashboard, "GPU Accelerated", w - 98, footer_y + 4, 0.34, TEXT_SECONDARY, 1)

    frame[:] = dashboard

    cv2.imshow("Detector de emociones", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break
    if tecla == ord("r"):
        suavizador = Suavizador(clases, ventana=5)
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
        suavizador = Suavizador(clases, ventana=5)
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
        suavizador = Suavizador(clases, ventana=5)

calibrador_distancia.guardar()
calibrador_mirada.guardar()
medidor_distancia.guardar()
captura.release()
cv2.destroyAllWindows()
detector.cerrar()
extractor.cerrar()
seguidor_mirada.cerrar()
