"""Download a small OPEN Sentinel-1/2 maritime sample from Microsoft Planetary
Computer (anonymous STAC search + signed asset hrefs, no registration).

xView3-SAR is DIU-gated (registration + ~2 TB) and is intentionally NOT used here;
this open sample lets DarkVesselNet run a real Prithvi-2 / SAR-stem forward pass and
the Lee-filter / TGARD components on genuine imagery over a busy shipping lane.

Deps (install once into the project env): pystac-client planetary-computer rioxarray
Usage:
  python scripts/download_sentinel_sample.py --out data/sentinel_sample --max-items 4
  # On MSI: --out /scratch.global/$USER/darkvessel/sentinel_sample (run via srun, not login node)
"""
from __future__ import annotations

import argparse
import os
import sys

# A busy maritime AOI: Singapore Strait (heavy, often-dark vessel traffic).
AOI = {
    "type": "Polygon",
    "coordinates": [[[103.6, 1.05], [104.1, 1.05], [104.1, 1.35], [103.6, 1.35], [103.6, 1.05]]],
}
DATE_RANGE = "2024-01-01/2024-03-31"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sentinel_sample")
    ap.add_argument("--max-items", type=int, default=4)
    ap.add_argument("--collections", default="sentinel-1-grd,sentinel-2-l2a")
    args = ap.parse_args()

    try:
        import planetary_computer as pc
        import pystac_client
    except ImportError:
        sys.stderr.write(
            "Missing deps. Install into the project env:\n"
            "  pip install pystac-client planetary-computer rioxarray\n"
        )
        return 2

    os.makedirs(args.out, exist_ok=True)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    import urllib.request

    for coll in args.collections.split(","):
        coll = coll.strip()
        query = {"eo:cloud_cover": {"lt": 20}} if "sentinel-2" in coll else None
        search = catalog.search(
            collections=[coll], intersects=AOI, datetime=DATE_RANGE,
            query=query, max_items=args.max_items,
        )
        items = list(search.items())
        print(f"[{coll}] {len(items)} items")
        for it in items:
            sub = os.path.join(args.out, coll, it.id)
            os.makedirs(sub, exist_ok=True)
            # Pull a couple of representative bands/polarizations only (keeps it small).
            wanted = ("vv", "vh") if "sentinel-1" in coll else ("B04", "B03", "B02", "B08")
            for key, asset in it.assets.items():
                if key.lower() in [w.lower() for w in wanted]:
                    dst = os.path.join(sub, f"{key}.tif")
                    if os.path.exists(dst):
                        continue
                    print(f"  {it.id} <- {key}")
                    urllib.request.urlretrieve(asset.href, dst)
    print(f"done -> {args.out}")
    print("Point the darkvessel config / GeoBackbone.from_pretrained at this sample "
          "(stub=False) for the real forward pass; xView3 AP stays a labeled target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
