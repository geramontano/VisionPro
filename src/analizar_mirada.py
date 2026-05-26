from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--entrada", default="registros/mirada_emociones_v2.csv")
parser.add_argument("--salida", default="registros/resumen_mirada.csv")
args = parser.parse_args()

ruta = Path(args.entrada)
if not ruta.exists():
    print("No encontre", ruta)
    raise SystemExit

df = pd.read_csv(ruta)
if df.empty:
    print("El registro esta vacio")
    raise SystemExit

df = df[df["emocion"].notna()]
df = df[df["emocion"] != "sin cara"]
df = df[df["emocion"] != "error"]

agregados = {
    "registros": ("emocion", "size"),
    "confianza_promedio": ("confianza", "mean"),
}
if "distancia_cm" in df.columns:
    df["distancia_cm"] = pd.to_numeric(df["distancia_cm"], errors="coerce")
    agregados["distancia_promedio_cm"] = ("distancia_cm", "mean")

resumen = df.groupby(["zona", "emocion"]).agg(**agregados).reset_index()
total_zona = resumen.groupby("zona")["registros"].transform("sum")
resumen["porcentaje_zona"] = (resumen["registros"] / total_zona * 100).round(2)
resumen["confianza_promedio"] = resumen["confianza_promedio"].round(4)
if "distancia_promedio_cm" in resumen.columns:
    resumen["distancia_promedio_cm"] = resumen["distancia_promedio_cm"].round(2)
resumen = resumen.sort_values(["zona", "registros"], ascending=[True, False])
Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
resumen.to_csv(args.salida, index=False)
print(resumen.to_string(index=False))
print("Guardado en", args.salida)
