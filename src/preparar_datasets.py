import argparse

from datasets_publicos import PreparadorDatasets


parser = argparse.ArgumentParser()
parser.add_argument("--fuentes", nargs="+", default=["jaffe", "kdef", "ravdess"])
parser.add_argument("--raiz", default="datasets")
parser.add_argument("--ravdess-actores", nargs="+", type=int, default=[1, 2, 3, 4])
parser.add_argument("--frames-por-video", type=int, default=4)
parser.add_argument("--max-videos-ravdess", type=int, default=90)
parser.add_argument("--cremad-local", default=None)
args = parser.parse_args()

preparador = PreparadorDatasets(raiz=args.raiz, frames_por_video=args.frames_por_video)
try:
    for fuente in args.fuentes:
        fuente = fuente.lower()
        if fuente == "jaffe":
            preparador.preparar_jaffe()
        elif fuente == "kdef":
            preparador.preparar_kdef()
        elif fuente == "ravdess":
            preparador.preparar_ravdess(actores=args.ravdess_actores, max_videos_por_emocion=args.max_videos_ravdess)
        elif fuente == "cremad":
            if args.cremad_local:
                preparador.preparar_cremad_local(args.cremad_local)
            else:
                carpeta = preparador.descargar_cremad()
                preparador.preparar_cremad_local(carpeta)
        else:
            print("Fuente no reconocida:", fuente)
finally:
    preparador.cerrar()
