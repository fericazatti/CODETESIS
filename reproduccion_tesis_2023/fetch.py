"""Descarga selectiva de ds004100 (v1.1.3) desde el bucket público de OpenNeuro.

Solo baja lo necesario: participants.tsv, dataset_description.json y, por cada sujeto SEEG
con registros interictales, los .edf/.json/channels.tsv de TODOS sus runs interictales.
Idempotente: si el archivo existe con el tamaño correcto, lo saltea.
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIDS = os.path.join(HERE, "bids", "ds004100")
BASE = "https://s3.amazonaws.com/openneuro.org/"


def fetch(key, size=None, tries=5):
    dst = os.path.join(BIDS, key.split("ds004100/", 1)[1])
    if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size):
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(tries):
        try:
            tmp = f"{dst}.{os.getpid()}.part"  # único por proceso (descargas en paralelo)
            urllib.request.urlretrieve(BASE + key, tmp)
            os.replace(tmp, dst)
            return "ok"
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {i + 1}/{tries} {key}: {exc}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"no se pudo bajar {key}")


def main():
    manifest = json.load(open(os.path.join(HERE, "manifest_interictal.json")))
    subjects = ["sub-" + s.replace("sub-", "") for s in sys.argv[1:]] or sorted(manifest)
    for key in ("ds004100/participants.tsv", "ds004100/participants.json",
                "ds004100/dataset_description.json"):
        fetch(key)
    for sub in subjects:
        for key in manifest[sub]:
            print(sub, os.path.basename(key), fetch(key), flush=True)
    print("FETCH DONE", flush=True)


if __name__ == "__main__":
    main()
