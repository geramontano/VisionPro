# Fuentes de datos usadas por el proyecto

## RAVDESS

Ryerson Audio Visual Database of Emotional Speech and Song.

Fuente oficial:
https://zenodo.org/records/1188976

Uso dentro del proyecto:
se descargan videos de habla por actor y se extraen varios frames del tramo central de cada video. El nombre del archivo indica la emocion. Se usan neutro, feliz, triste, enojado, asustado y sorprendido.

## JAFFE

The Japanese Female Facial Expression Dataset.

Fuente oficial:
https://zenodo.org/records/14974867

Uso dentro del proyecto:
el script descarga el ZIP oficial desde Zenodo y convierte las imagenes a carpetas por emocion. No se redistribuyen las imagenes dentro de este ZIP porque los terminos de JAFFE prohiben la redistribucion externa.

## KDEF

Karolinska Directed Emotional Faces.

Fuente oficial:
https://www.ugent.be/pp/ekgp/en/research/research-groups/panlab/kdef

Uso dentro del proyecto:
el script busca el enlace oficial de Pictures ZIP y usa solo imagenes frontales para reducir ruido de perfil.

## CREMA-D opcional

Crowd Sourced Emotional Multimodal Actors Dataset.

Fuente oficial:
https://github.com/CheyneyComputerScience/CREMA-D

Uso dentro del proyecto:
queda como fuente opcional porque ocupa varios GB y requiere Git LFS. Cubre neutro, feliz, triste, enojado y asustado. No cubre sorprendido de forma directa.
