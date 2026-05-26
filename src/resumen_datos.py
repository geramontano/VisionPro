from pathlib import Path

from nombres import EMOCIONES

raiz = Path("datasets/preparado")
for emocion in EMOCIONES:
    total = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        total += len(list((raiz / emocion).glob(ext)))
    print(emocion, total)
