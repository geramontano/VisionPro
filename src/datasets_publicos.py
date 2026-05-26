import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import cv2
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from cara import DetectorCara
from nombres import MAPA_CODIGOS_CARAS, MAPA_RAVDESS


class PreparadorDatasets:
    def __init__(self, raiz="datasets", tamano=160, frames_por_video=4):
        self.raiz = Path(raiz)
        self.descargados = self.raiz / "descargados"
        self.preparado = self.raiz / "preparado"
        self.tamano = tamano
        self.frames_por_video = frames_por_video
        self.detector = DetectorCara(confianza=0.45)
        self.descargados.mkdir(parents=True, exist_ok=True)
        self.preparado.mkdir(parents=True, exist_ok=True)

    def carpeta_emocion(self, emocion):
        carpeta = self.preparado / emocion
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    def descargar(self, url, destino):
        destino = Path(destino)
        if destino.exists() and destino.stat().st_size > 1000:
            return destino
        temporal = destino.with_suffix(destino.suffix + ".tmp")
        with requests.get(url, stream=True, timeout=60) as respuesta:
            respuesta.raise_for_status()
            total = int(respuesta.headers.get("content-length", 0))
            with open(temporal, "wb") as archivo, tqdm(total=total, unit="B", unit_scale=True, desc=destino.name) as barra:
                for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                    if bloque:
                        archivo.write(bloque)
                        barra.update(len(bloque))
        temporal.rename(destino)
        return destino

    def extraer_zip(self, archivo_zip, destino):
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        marcador = destino / ".extraido"
        if marcador.exists():
            return destino
        with zipfile.ZipFile(archivo_zip, "r") as z:
            z.extractall(destino)
        marcador.write_text("ok", encoding="utf-8")
        return destino

    def guardar_cara(self, imagen, emocion, nombre):
        cara, _ = self.detector.recortar(imagen)
        if cara is None:
            cara = imagen
        cara = cv2.resize(cara, (self.tamano, self.tamano))
        destino = self.carpeta_emocion(emocion) / nombre
        cv2.imwrite(str(destino), cara)

    def preparar_jaffe(self):
        url = "https://zenodo.org/records/14974867/files/jaffe.zip?download=1"
        archivo = self.descargados / "jaffe.zip"
        carpeta = self.descargados / "jaffe"
        self.descargar(url, archivo)
        self.extraer_zip(archivo, carpeta)
        contador = 0
        for ruta in carpeta.rglob("*"):
            if ruta.suffix.lower() not in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
                continue
            partes = ruta.name.split(".")
            if len(partes) < 2:
                continue
            codigo = re.sub(r"[^A-Z]", "", partes[1].upper())[:2]
            emocion = MAPA_CODIGOS_CARAS.get(codigo)
            if emocion is None:
                continue
            imagen = cv2.imread(str(ruta))
            if imagen is None:
                continue
            self.guardar_cara(imagen, emocion, f"jaffe_{ruta.stem}.jpg")
            contador += 1
        print("JAFFE preparado:", contador, "imagenes")

    def enlace_kdef(self):
        pagina = "https://www.ugent.be/pp/ekgp/en/research/research-groups/panlab/kdef"
        respuesta = requests.get(pagina, timeout=60)
        respuesta.raise_for_status()
        sopa = BeautifulSoup(respuesta.text, "html.parser")
        for enlace in sopa.find_all("a"):
            texto = enlace.get_text(" ", strip=True).lower()
            href = enlace.get("href", "")
            if "pictures" in texto or href.lower().endswith(".zip"):
                return urljoin(pagina, href)
        raise RuntimeError("No pude encontrar automaticamente el ZIP de KDEF en la pagina oficial.")

    def preparar_kdef(self):
        url = self.enlace_kdef()
        archivo = self.descargados / "kdef.zip"
        carpeta = self.descargados / "kdef"
        self.descargar(url, archivo)
        self.extraer_zip(archivo, carpeta)
        contador = 0
        for ruta in carpeta.rglob("*"):
            if ruta.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            nombre = ruta.stem.upper()
            codigo = None
            vista = "S"
            if len(nombre) >= 7:
                codigo = nombre[4:6]
                vista = nombre[6]
            emocion = MAPA_CODIGOS_CARAS.get(codigo)
            if emocion is None or vista != "S":
                continue
            imagen = cv2.imread(str(ruta))
            if imagen is None:
                continue
            self.guardar_cara(imagen, emocion, f"kdef_{ruta.stem}.jpg")
            contador += 1
        print("KDEF preparado:", contador, "imagenes frontales")

    def archivos_ravdess(self):
        api = "https://zenodo.org/api/records/1188976"
        datos = requests.get(api, timeout=60).json()
        archivos = {}
        for elemento in datos.get("files", []):
            nombre = elemento.get("key", "")
            if nombre.startswith("Video_Speech_Actor_") and nombre.endswith(".zip"):
                archivos[nombre] = elemento["links"]["self"]
        return archivos

    def preparar_ravdess(self, actores=(1, 2), max_videos_por_emocion=80):
        disponibles = self.archivos_ravdess()
        carpeta_base = self.descargados / "ravdess"
        carpeta_base.mkdir(parents=True, exist_ok=True)
        usados = {emocion: 0 for emocion in MAPA_RAVDESS.values()}

        for actor in actores:
            nombre_zip = f"Video_Speech_Actor_{actor:02d}.zip"
            url = disponibles.get(nombre_zip)
            if url is None:
                print("No encontre", nombre_zip)
                continue
            archivo = self.descargados / nombre_zip
            carpeta = carpeta_base / f"Actor_{actor:02d}"
            self.descargar(url, archivo)
            self.extraer_zip(archivo, carpeta)

            for video in carpeta.rglob("*.mp4"):
                partes = video.stem.split("-")
                if len(partes) != 7:
                    continue
                modalidad, canal, emocion_codigo = partes[0], partes[1], partes[2]
                if modalidad != "02" or canal != "01":
                    continue
                emocion = MAPA_RAVDESS.get(emocion_codigo)
                if emocion is None or usados[emocion] >= max_videos_por_emocion:
                    continue
                guardadas = self.frames_video(video, emocion, f"ravdess_{video.stem}")
                if guardadas > 0:
                    usados[emocion] += 1
        print("RAVDESS videos usados:", usados)

    def frames_video(self, ruta_video, emocion, prefijo):
        captura = cv2.VideoCapture(str(ruta_video))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            captura.release()
            return 0
        inicio = int(total * 0.28)
        fin = int(total * 0.78)
        posiciones = []
        if self.frames_por_video <= 1:
            posiciones = [(inicio + fin) // 2]
        else:
            paso = max((fin - inicio) // self.frames_por_video, 1)
            posiciones = [inicio + i * paso for i in range(self.frames_por_video)]
        guardadas = 0
        for numero, posicion in enumerate(posiciones):
            captura.set(cv2.CAP_PROP_POS_FRAMES, posicion)
            ok, frame = captura.read()
            if not ok or frame is None:
                continue
            self.guardar_cara(frame, emocion, f"{prefijo}_{numero:02d}.jpg")
            guardadas += 1
        captura.release()
        return guardadas

    def preparar_cremad_local(self, carpeta_local):
        carpeta_local = Path(carpeta_local)
        contador = 0
        mapa = {
            "NEU": "neutro",
            "HAP": "feliz",
            "SAD": "triste",
            "ANG": "enojado",
            "FEA": "asustado"
        }
        for video in carpeta_local.rglob("*"):
            if video.suffix.lower() not in [".flv", ".mp4", ".mov", ".avi"]:
                continue
            partes = video.stem.split("_")
            if len(partes) < 3:
                continue
            emocion = mapa.get(partes[2].upper())
            if emocion is None:
                continue
            contador += self.frames_video(video, emocion, f"cremad_{video.stem}")
        print("CREMA-D local preparado:", contador, "frames")

    def descargar_cremad(self):
        destino = self.descargados / "CREMA-D"
        if destino.exists():
            return destino
        print("CREMA-D usa Git LFS y puede ocupar varios GB.")
        print("Intentando clonar la fuente oficial. Si falla, instala git-lfs o usa --cremad-local.")
        subprocess.run(["git", "lfs", "install"], check=False)
        subprocess.run(["git", "clone", "https://github.com/CheyneyComputerScience/CREMA-D.git", str(destino)], check=True)
        return destino

    def cerrar(self):
        self.detector.cerrar()
