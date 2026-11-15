"""Download and extract the Penn-Fudan Pedestrian database into data/PennFudanPed."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "PennFudanPed.zip"
    if (dest / "PennFudanPed" / "PNGImages").is_dir():
        print("dataset already present")
        return
    try:
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}).\nDownload the zip manually from {URL} and extract it so that "
                         f"{dest}/PennFudanPed/PNGImages and {dest}/PennFudanPed/PedMasks exist.")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    n = len(list((dest / "PennFudanPed" / "PNGImages").glob("*.png")))
    print(f"extracted {n} images to {dest / 'PennFudanPed'}")


if __name__ == "__main__":
    main()
