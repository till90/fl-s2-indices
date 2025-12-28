#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API:
- POST /api/search -> Sucht nach Sentinel-2-Szenen
- POST /api/preview -> Erstellt eine Vorschau für eine ausgewählte Szene
- POST /api/timeseries -> Erstellt eine Zeitreihe für eine ausgewählte Szene
- GET /r/<job_id>/overlay.png -> Ruft das Overlay-Bild für einen bestimmten Job ab
- GET /r/<job_id>/index.tif -> Ruft die Index-GeoTIFF-Datei für einen bestimmten Job ab
- GET /r/<job_id>/timeseries.csv -> Ruft die Zeitreihen-CSV-Datei für einen bestimmten Job ab
- GET /r/<job_id>/timeseries.json -> Ruft die Zeitreihen-JSON-Datei für einen bestimmten Job ab
- GET /healthz -> Überprüft den Zustand des Dienstes
"""

import csv
import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from affine import Affine

# -------------------------------
# Config (Cloud Run friendly)
# -------------------------------

APP_TITLE = os.getenv("APP_TITLE", "fl-s2-indices (STAC) – NDVI/NDMI/NDWI/NDRE + CSV")

STAC_API_URL = os.getenv("STAC_API_URL", "https://earth-search.aws.element84.com/v1")
STAC_COLLECTION = os.getenv("STAC_COLLECTION", "sentinel-2-l2a")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

DEFAULT_N_SCENES = int(os.getenv("DEFAULT_N_SCENES", "12"))
DEFAULT_CLOUD_MAX = float(os.getenv("DEFAULT_CLOUD_MAX", "30"))
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "365"))

# Raster limits
MAX_PREVIEW_DIM_PX = int(os.getenv("MAX_PREVIEW_DIM_PX", "1200"))   # PNG overlay max dim
MAX_TS_DIM_PX = int(os.getenv("MAX_TS_DIM_PX", "450"))              # time series sampling max dim

# AOI safety
MAX_AOI_AREA_KM2 = float(os.getenv("MAX_AOI_AREA_KM2", "25.0"))     # adjust as needed
DEFAULT_PAD_M = float(os.getenv("DEFAULT_PAD_M", "0"))              # padding around AOI for preview/timeseries

# Temp cache
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")) / "s2_indices_cache"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "80"))

# RasterIO / GDAL hints for remote COGs
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "25000000")  # 25 MB

# -------------------------------
# Flask
# -------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# -------------------------------
# Helpers
# -------------------------------

@dataclass
class PreviewResult:
    job_id: str
    bounds_wgs84: Tuple[Tuple[float, float], Tuple[float, float]]  # ((south, west), (north, east))
    index_name: str
    item_id: str
    datetime_str: str
    cloud_cover: Optional[float]
    png_path: Path
    tif_path: Path

@dataclass
class TimeseriesResult:
    job_id: str
    index_name: str
    rows: List[Dict[str, Any]]
    csv_path: Path
    json_path: Path


def _cleanup_cache() -> None:
    try:
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)

        now = time.time()
        for mtime, p in items:
            if now - mtime > CACHE_TTL_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)

        if len(items) > MAX_CACHE_ITEMS:
            for _, p in items[MAX_CACHE_ITEMS:]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def _parse_geojson(payload: Any) -> Dict[str, Any]:
    if payload is None:
        raise ValueError("Kein GeoJSON übergeben.")
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError("Leerer GeoJSON-String.")
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("GeoJSON muss ein JSON-Objekt oder String sein.")


def _extract_single_geometry(gj: Dict[str, Any]):
    t = gj.get("type")
    if t == "Feature":
        geom = gj.get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        if len(feats) != 1:
            raise ValueError("FeatureCollection muss genau 1 Feature enthalten.")
        geom = feats[0].get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t in ("Polygon", "MultiPolygon"):
        return shape(gj)
    raise ValueError(f"Nicht unterstützter GeoJSON-Typ: {t}. Erlaubt: Feature, FeatureCollection(1), Polygon, MultiPolygon.")


def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def _geom_to_epsg(geom, src_epsg: int, dst_epsg: int):
    tr = _transformer(src_epsg, dst_epsg)
    return shp_transform(lambda x, y: tr.transform(x, y), geom)


def _bounds_epsg_to_wgs84(minx: float, miny: float, maxx: float, maxy: float, src_epsg: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    tr = _transformer(src_epsg, 4326)
    west, south = tr.transform(minx, miny)
    east, north = tr.transform(maxx, maxy)
    return (south, west), (north, east)


def _compute_scaled_dims(width_m: float, height_m: float, max_dim_px: int) -> Tuple[int, int]:
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Ungültige Ausdehnung (Breite/Höhe <= 0).")
    if width_m >= height_m:
        w = max_dim_px
        h = max(1, int(round(max_dim_px * (height_m / width_m))))
    else:
        h = max_dim_px
        w = max(1, int(round(max_dim_px * (width_m / height_m))))
    return w, h


def _http_request_json(method: str, url: str, **kwargs) -> Dict[str, Any]:
    try:
        r = requests.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
    except Exception as e:
        raise RuntimeError(f"STAC Request fehlgeschlagen: {e}")

    ct = (r.headers.get("Content-Type") or "").lower()
    txt = r.text or ""
    if not r.ok:
        # try json body
        try:
            js = r.json()
        except Exception:
            js = {"raw": txt[:1200]}
        raise RuntimeError(f"STAC Request fehlgeschlagen: Upstream HTTP {r.status_code}: {js}")

    if "application/json" not in ct and "application/geo+json" not in ct and "json" not in ct:
        raise RuntimeError(f"STAC Request lieferte kein JSON (Content-Type={ct}). Auszug: {txt[:300]}")

    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"STAC JSON Parse Error: {e}. Auszug: {txt[:300]}")


def _stac_search(
    intersects_geom: Dict[str, Any],
    limit_n: int,
    cloud_max: Optional[float],
    lookback_days: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ask = max(5, min(200, int(limit_n) * 6))

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max(1, int(lookback_days)))
    dt_range = f"{start.isoformat().replace('+00:00','Z')}/{now.isoformat().replace('+00:00','Z')}"

    base: Dict[str, Any] = {
        "collections": [STAC_COLLECTION],
        "intersects": intersects_geom,
        "datetime": dt_range,
        "limit": ask,
    }
    if cloud_max is not None:
        base["query"] = {"eo:cloud_cover": {"lt": float(cloud_max)}}

    url = f"{STAC_API_URL}/search"

    # Try sort variants (backend differences):
    # - some want properties.datetime, some datetime; sortby as array
    sort_candidates = [
        [{"field": "properties.datetime", "direction": "desc"}],
        [{"field": "datetime", "direction": "desc"}],
        None,
    ]

    js = None
    used_body = None
    last_err = None

    for cand in sort_candidates:
        body_try = dict(base)
        if cand is not None:
            body_try["sortby"] = cand
        try:
            js = _http_request_json("POST", url, json=body_try, headers={"Accept": "application/geo+json"})
            used_body = body_try
            break
        except RuntimeError as e:
            msg = str(e).lower()
            last_err = e
            sort_related = ("sortby" in msg) or ("slice is not a function" in msg) or ("no mapping found" in msg) or ("query_shard_exception" in msg and "sort" in msg)
            if not sort_related:
                raise
            continue

    if js is None or used_body is None:
        raise RuntimeError(f"STAC Request fehlgeschlagen: {last_err}")

    feats = js.get("features") or []

    # final local sort by properties.datetime
    def _dt(f):
        return ((f.get("properties") or {}).get("datetime") or "")

    feats.sort(key=_dt, reverse=True)

    out = []
    for f in feats:
        props = f.get("properties") or {}
        cc = props.get("eo:cloud_cover", None)
        if cloud_max is not None and cc is not None:
            try:
                if float(cc) >= float(cloud_max):
                    continue
            except Exception:
                pass
        out.append(f)
        if len(out) >= limit_n:
            break

    return used_body, out


def _is_jp2_asset(asset_key: str, href: str) -> bool:
    k = (asset_key or "").lower()
    h = (href or "").lower()
    return ("jp2" in k) or h.endswith(".jp2") or ("-jp2" in k)


def _asset_band_meta(asset: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    returns (eo_name, eo_common_name, gsd)
    """
    gsd = asset.get("gsd", None)
    eo = asset.get("eo:bands") or []
    if eo and isinstance(eo, list) and len(eo) >= 1 and isinstance(eo[0], dict):
        return eo[0].get("name"), eo[0].get("common_name"), gsd
    return None, None, gsd


def _pick_asset(item: Dict[str, Any], want: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """
    want: {"common": "..."} and/or {"name": "..."} and optional {"key": "..."}.
    Prefers non-jp2 and image/tiff.
    """
    assets = item.get("assets") or {}
    best = None
    best_score = -1

    want_common = (want.get("common") or "").lower().strip()
    want_name = (want.get("name") or "").lower().strip()
    want_key = (want.get("key") or "").lower().strip()

    for k, a in assets.items():
        if not isinstance(a, dict):
            continue
        href = a.get("href") or ""
        if not href:
            continue
        if _is_jp2_asset(k, href):
            continue

        eo_name, eo_common, gsd = _asset_band_meta(a)
        eo_name_l = (eo_name or "").lower()
        eo_common_l = (eo_common or "").lower()
        k_l = (k or "").lower()

        # match logic
        match = False
        if want_key and (k_l == want_key):
            match = True
        if want_common and (eo_common_l == want_common or k_l == want_common):
            match = True
        if want_name and (eo_name_l == want_name or k_l == want_name):
            match = True

        if not match:
            # small heuristics for rededge variants if eo:bands missing
            if want_name and (want_name in k_l):
                match = True

        if not match:
            continue

        score = 0
        typ = (a.get("type") or "").lower()
        if "image/tiff" in typ or href.lower().endswith(".tif") or href.lower().endswith(".tiff"):
            score += 20
        if want_key and k_l == want_key:
            score += 10
        if want_name and eo_name_l == want_name:
            score += 8
        if want_common and eo_common_l == want_common:
            score += 6
        if gsd is not None:
            # prefer lower gsd slightly for preview quality
            score += max(0, int(10 - float(gsd) / 10.0))

        if score > best_score:
            best_score = score
            best = (k, a)

    if best is None:
        raise ValueError(f"Band-Asset nicht gefunden: {want}. Verfügbare Assets: {', '.join(list(assets.keys())[:25])} ...")

    return best[0], best[1]


def _index_specs() -> Dict[str, Dict[str, Any]]:
    # we keep formulas as (A-B)/(A+B) with band wants
    return {
        "NDVI": {
            "bands": {
                "A": {"common": "nir", "key": "nir"},     # B08 (10m) in Earth Search
                "B": {"common": "red", "key": "red"},     # B04 (10m)
            },
            "label": "NDVI = (NIR - RED) / (NIR + RED)",
        },
        "NDMI": {
            "bands": {
                "A": {"common": "nir", "key": "nir"},        # B08 (10m) -> resampled down
                "B": {"common": "swir16", "key": "swir16"},  # B11 (20m)
            },
            "label": "NDMI = (NIR - SWIR16) / (NIR + SWIR16)",
        },
        "NDWI": {
            "bands": {
                "A": {"common": "green", "key": "green"},  # B03 (10m)
                "B": {"common": "nir", "key": "nir"},      # B08 (10m)
            },
            "label": "NDWI (McFeeters) = (GREEN - NIR) / (GREEN + NIR)",
        },
        "NDRE1": {
            "bands": {
                "A": {"common": "nir08", "name": "nir08", "key": "nir08"},         # B8A (20m)
                "B": {"common": "rededge", "name": "rededge1", "key": "rededge1"}, # B05 (20m)
            },
            "label": "NDRE1 = (B8A - B05) / (B8A + B05)",
        },
        "NDRE2": {
            "bands": {
                "A": {"common": "nir08", "name": "nir08", "key": "nir08"},         # B8A (20m)
                "B": {"common": "rededge", "name": "rededge2", "key": "rededge2"}, # B06 (20m)
            },
            "label": "NDRE2 = (B8A - B06) / (B8A + B06)",
        },
        "NDRE3": {
            "bands": {
                "A": {"common": "nir08", "name": "nir08", "key": "nir08"},         # B8A (20m)
                "B": {"common": "rededge", "name": "rededge3", "key": "rededge3"}, # B07 (20m)
            },
            "label": "NDRE3 = (B8A - B07) / (B8A + B07)",
        },
    }


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-6
    denom = b.copy()
    denom[np.abs(denom) < eps] = np.nan
    return a / denom


def _compute_index(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # (A-B)/(A+B)
    num = (A - B).astype(np.float32)
    den = (A + B).astype(np.float32)
    out = _safe_div(num, den).astype(np.float32)
    return out


def _colorize_index(idx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Simple blue -> white -> green ramp for idx in [-1, 1].
    Returns RGBA uint8.
    """
    h, w = idx.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    v = np.clip((idx + 1.0) / 2.0, 0.0, 1.0)  # 0..1
    v = np.nan_to_num(v, nan=0.0)

    # split at 0.5
    left = v <= 0.5
    right = ~left

    # blue (0,0,255) -> white (255,255,255)
    t = np.zeros_like(v, dtype=np.float32)
    t[left] = (v[left] / 0.5).astype(np.float32)

    rgba[..., 0][left] = (0 + t[left] * 255).astype(np.uint8)
    rgba[..., 1][left] = (0 + t[left] * 255).astype(np.uint8)
    rgba[..., 2][left] = 255

    # white -> green (0,255,0)
    t2 = np.zeros_like(v, dtype=np.float32)
    t2[right] = ((v[right] - 0.5) / 0.5).astype(np.float32)

    rgba[..., 0][right] = (255 - t2[right] * 255).astype(np.uint8)
    rgba[..., 1][right] = 255
    rgba[..., 2][right] = (255 - t2[right] * 255).astype(np.uint8)

    rgba[..., 3] = (valid_mask.astype(np.uint8) * 255)
    return rgba


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def _read_bands_for_item(
    item: Dict[str, Any],
    geom_wgs84,
    pad_m: float,
    out_max_dim: int,
    index_name: str,
) -> Tuple[np.ndarray, np.ndarray, Affine, int, Tuple[float, float, float, float], Dict[str, Any]]:
    """
    Returns: (idx_float32, valid_mask_bool, out_transform, epsg, bounds_in_epsg, debug)
    """
    specs = _index_specs()
    if index_name not in specs:
        raise ValueError(f"Unbekannter Index: {index_name}. Erlaubt: {', '.join(specs.keys())}")

    props = item.get("properties") or {}
    epsg = item.get("properties", {}).get("proj:epsg") or item.get("proj:epsg") or props.get("proj:epsg")
    if epsg is None:
        # try on assets
        assets = item.get("assets") or {}
        for a in assets.values():
            if isinstance(a, dict) and a.get("proj:epsg"):
                epsg = a.get("proj:epsg")
                break
    if epsg is None:
        raise ValueError("Item enthält kein proj:epsg. Kann AOI nicht korrekt reprojizieren.")

    epsg = int(epsg)

    geom = _geom_to_epsg(geom_wgs84, 4326, epsg)

    # area limit
    aoi_area_km2 = float(geom.area) / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    if pad_m and pad_m > 0:
        geom_work = geom.buffer(float(pad_m))
    else:
        geom_work = geom

    minx, miny, maxx, maxy = geom_work.bounds
    bbox = (minx, miny, maxx, maxy)
    width_m = maxx - minx
    height_m = maxy - miny
    out_w, out_h = _compute_scaled_dims(width_m, height_m, out_max_dim)

    # pick assets for A and B
    spec = specs[index_name]
    wantA = spec["bands"]["A"]
    wantB = spec["bands"]["B"]

    keyA, assetA = _pick_asset(item, wantA)
    keyB, assetB = _pick_asset(item, wantB)

    hrefA = assetA.get("href")
    hrefB = assetB.get("href")

    gsdA = assetA.get("gsd", None)
    gsdB = assetB.get("gsd", None)

    # reference = coarser (bigger gsd) to avoid upsampling
    ref_is_A = False
    if gsdA is None and gsdB is None:
        ref_is_A = True
    elif gsdA is None:
        ref_is_A = False
    elif gsdB is None:
        ref_is_A = True
    else:
        ref_is_A = float(gsdA) >= float(gsdB)

    href_ref = hrefA if ref_is_A else hrefB
    href_other = hrefB if ref_is_A else hrefA

    debug = {
        "epsg": epsg,
        "bbox": bbox,
        "out_shape": [out_h, out_w],
        "index": index_name,
        "asset_A": {"key": keyA, "href": hrefA, "gsd": gsdA},
        "asset_B": {"key": keyB, "href": hrefB, "gsd": gsdB},
        "ref": "A" if ref_is_A else "B",
    }

    def _read_one(href: str, bbox_local: Tuple[float, float, float, float], out_hw: Tuple[int, int]) -> Tuple[np.ndarray, Affine, np.ndarray]:
        with rasterio.open(href) as src:
            w = from_bounds(*bbox_local, transform=src.transform)

            # clip window to dataset bounds
            w = w.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            # NEU: wenn nach dem Clip nichts übrig bleibt → AOI außerhalb der Szene
            if w.width <= 0 or w.height <= 0:
                raise ValueError("AOI liegt außerhalb der Szene (Window leer).")

            # transform for window at native resolution
            win_transform = rasterio.windows.transform(w, src.transform)

            out_hh, out_ww = out_hw
            scale_x = (w.width / float(out_ww)) if out_ww > 0 else 1.0
            scale_y = (w.height / float(out_hh)) if out_hh > 0 else 1.0
            out_transform = win_transform * Affine.scale(scale_x, scale_y)

            arr = src.read(
                1,
                window=w,
                out_shape=(out_hh, out_ww),
                resampling=Resampling.bilinear,
                masked=True,
            )

            # NEU: NaN erst nach float32
            msk = np.ma.getmaskarray(arr)
            data = arr.astype(np.float32).filled(np.nan)

            return data, out_transform, msk


    # read ref first to define transform
    data_ref, out_transform, msk_ref = _read_one(href_ref, bbox, (out_h, out_w))
    data_other, out_transform2, msk_other = _read_one(href_other, bbox, (out_h, out_w))

    # AOI mask (not buffered geom_work) -> show/compute only inside original AOI geom
    aoi_mask = rasterize(
        [mapping(geom)],
        out_shape=(out_h, out_w),
        transform=out_transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype=np.uint8,
    ).astype(bool)

    # map ref/other back to A/B
    if ref_is_A:
        A = data_ref
        B = data_other
        mA = msk_ref
        mB = msk_other
    else:
        B = data_ref
        A = data_other
        mB = msk_ref
        mA = msk_other

    idx = _compute_index(A, B)

    # validity: inside AOI AND not masked AND finite
    valid = aoi_mask & (~mA) & (~mB) & np.isfinite(idx)

    idx_out = idx.copy()
    idx_out[~valid] = np.nan

    return idx_out, valid, out_transform, epsg, bbox, debug


def _write_geotiff_float32(path: Path, arr: np.ndarray, transform: Affine, epsg: int, nodata: float = -9999.0) -> None:
    h, w = arr.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "interleave": "band",
    }
    out = np.where(np.isfinite(arr), arr.astype(np.float32), np.float32(nodata))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def _preview_from_geojson(gj: Dict[str, Any], item: Dict[str, Any], index_name: str, pad_m: float) -> Tuple[PreviewResult, Dict[str, Any]]:
    _cleanup_cache()

    geom_wgs84 = _extract_single_geometry(gj)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    idx, valid, out_transform, epsg, bbox, debug = _read_bands_for_item(
        item=item,
        geom_wgs84=geom_wgs84,
        pad_m=float(pad_m),
        out_max_dim=MAX_PREVIEW_DIM_PX,
        index_name=index_name,
    )

    # write outputs
    job_id = uuid.uuid4().hex[:12]
    out_png = TMP_DIR / f"{job_id}.overlay.png"
    out_tif = TMP_DIR / f"{job_id}.index.tif"

    rgba = _colorize_index(idx, valid_mask=valid)
    Image.fromarray(rgba, mode="RGBA").save(out_png)

    _write_geotiff_float32(out_tif, idx, out_transform, epsg, nodata=-9999.0)

    minx, miny, maxx, maxy = bbox
    bounds_wgs84 = _bounds_epsg_to_wgs84(minx, miny, maxx, maxy, epsg)

    props = item.get("properties") or {}
    dt = props.get("datetime") or ""
    cc = props.get("eo:cloud_cover", None)
    item_id = item.get("id") or ""

    rr = PreviewResult(
        job_id=job_id,
        bounds_wgs84=bounds_wgs84,
        index_name=index_name,
        item_id=item_id,
        datetime_str=dt,
        cloud_cover=float(cc) if cc is not None else None,
        png_path=out_png,
        tif_path=out_tif,
    )

    # include stats for UI
    st = _stats(idx)
    return rr, {"stats": st, "debug": debug}


def _timeseries_from_geojson(gj: Dict[str, Any], items: List[Dict[str, Any]], index_name: str, pad_m: float) -> TimeseriesResult:
    _cleanup_cache()

    geom_wgs84 = _extract_single_geometry(gj)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    rows: List[Dict[str, Any]] = []
    for it in items:
        props = it.get("properties") or {}
        dt = props.get("datetime") or ""
        cc = props.get("eo:cloud_cover", None)
        iid = it.get("id") or ""

        try:
            idx, valid, out_transform, epsg, bbox, debug = _read_bands_for_item(
                item=it,
                geom_wgs84=geom_wgs84,
                pad_m=float(pad_m),
                out_max_dim=MAX_TS_DIM_PX,
                index_name=index_name,
            )
            st = _stats(idx)
            row = {
                "datetime": dt,
                "item_id": iid,
                "cloud_cover": float(cc) if cc is not None else None,
                "count": st["count"],
                "mean": st["mean"],
                "median": st["median"],
                "std": st["std"],
                "p10": st["p10"],
                "p90": st["p90"],
            }
            rows.append(row)
        except Exception as e:
            # skip problematic items but keep trace
            rows.append({
                "datetime": dt,
                "item_id": iid,
                "cloud_cover": float(cc) if cc is not None else None,
                "count": 0,
                "mean": None, "median": None, "std": None, "p10": None, "p90": None,
                "error": str(e),
            })

    # sort by datetime ascending for timeseries
    rows.sort(key=lambda r: (r.get("datetime") or ""))

    job_id = uuid.uuid4().hex[:12]
    csv_path = TMP_DIR / f"{job_id}.timeseries.csv"
    json_path = TMP_DIR / f"{job_id}.timeseries.json"

    # write JSON
    json_path.write_text(json.dumps({"index": index_name, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    # write CSV
    fieldnames = ["datetime", "item_id", "cloud_cover", "count", "mean", "median", "std", "p10", "p90", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    return TimeseriesResult(
        job_id=job_id,
        index_name=index_name,
        rows=rows,
        csv_path=csv_path,
        json_path=json_path,
    )


# -------------------------------
# Routes / UI
# -------------------------------

INDEX_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />

  <style>
    :root{
      --bg:#0b0f19;
      --card:#111a2e;
      --text:#e6eaf2;
      --muted:#a8b3cf;
      --border: rgba(255,255,255,.10);
      --primary:#6ea8fe;
      --focus: rgba(110,168,254,.45);
      --radius: 16px;
      --container: 1200px;
      --gap: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    body{ margin:0; font-family: var(--font); background: var(--bg); color: var(--text); }
    .wrap{ max-width: var(--container); margin: 18px auto; padding: 0 14px 24px; display: grid; grid-template-columns: 1.2fr .8fr; gap: var(--gap); }
    header{ max-width: var(--container); margin: 18px auto 0; padding: 0 14px; display:flex; align-items:baseline; justify-content:space-between; gap: 12px; }
    h1{ font-size: 18px; margin:0; letter-spacing: .2px; }
    .hint{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.35; }
    .card{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: 0 18px 60px rgba(0,0,0,.35); overflow: hidden; }
    #map{ height: 70vh; min-height: 520px; }
    .panel{ padding: 12px; display:flex; flex-direction:column; gap: 10px; }
    label{ color: var(--muted); font-size: 12px; }
    textarea{
      width: 100%; min-height: 180px; resize: vertical;
      background: rgba(255,255,255,.04); border: 1px solid var(--border); border-radius: 12px;
      padding: 10px; color: var(--text); font-family: var(--mono); font-size: 12px; outline: none;
    }
    textarea:focus{ border-color: var(--primary); box-shadow: 0 0 0 4px var(--focus); }
    .row{ display:flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button{
      appearance:none; border: 1px solid var(--border); background: rgba(255,255,255,.06);
      color: var(--text); padding: 10px 12px; border-radius: 12px; cursor: pointer; font-weight: 600;
    }
    button.primary{ border-color: rgba(110,168,254,.35); background: rgba(110,168,254,.16); }
    button:disabled{ opacity:.55; cursor:not-allowed; }
    select,input{
      background: rgba(255,255,255,.04); border: 1px solid var(--border); border-radius: 10px;
      padding: 8px 10px; color: var(--text);
    }
    .status{
      color: var(--muted); font-size: 13px; line-height: 1.35;
      padding: 8px 10px; border-radius: 12px; background: rgba(0,0,0,.18); border: 1px solid var(--border);
    }
    .status b{ color: var(--text); }
    .err{ border-color: rgba(255,100,100,.35); background: rgba(255,100,100,.10); color: #ffd1d1; }
    .ok{ border-color: rgba(120,220,160,.35); background: rgba(120,220,160,.08); }
    .small{ font-size: 12px; color: var(--muted); }
    a{ color: var(--primary); text-decoration: none; }
    a:hover{ text-decoration: underline; }
    table{ width:100%; border-collapse: collapse; font-size: 12px; }
    th,td{ padding: 6px 6px; border-bottom: 1px solid rgba(255,255,255,.08); vertical-align: top; }
    th{ color: var(--muted); font-weight: 600; text-align:left; }
    .mono{ font-family: var(--mono); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ title }}</h1>
      <div class="hint">
        Zeichne ein Polygon/Rechteck (immer nur <b>ein</b> Feature). Suche die letzten Szenen über STAC, berechne Indizes (NDVI/NDMI/NDWI/NDRE) und exportiere eine Zeitreihe als CSV.
      </div>
    </div>
    <div class="small">API: <code>/api/search</code> · <code>/api/preview</code> · <code>/api/timeseries</code></div>
  </header>

  <div class="wrap">
    <div class="card"><div id="map"></div></div>

    <div class="card">
      <div class="panel">
        <div class="row">
          <button id="btn-clear">AOI löschen</button>
          <button class="primary" id="btn-search" disabled>Szenen suchen</button>
        </div>

        <div class="row">
          <label>N: <input id="n" type="number" min="1" max="50" value="{{ default_n }}" style="width:90px;"></label>
          <label>Cloud &lt; (%): <input id="cloud" type="number" min="0" max="100" value="{{ default_cloud }}" style="width:110px;"></label>
          <label>Lookback (Tage): <input id="days" type="number" min="1" max="3650" value="{{ default_days }}" style="width:120px;"></label>
          <label>Padding (m): <input id="pad" type="number" min="0" step="10" value="{{ default_pad }}" style="width:110px;"></label>
        </div>

        <div class="row">
          <label>Index:
            <select id="index">
              {% for k in indices %}
                <option value="{{ k }}">{{ k }}</option>
              {% endfor %}
            </select>
          </label>
        </div>

        <div id="status" class="status">Noch keine AOI.</div>

        <div class="row">
          <button id="btn-preview" disabled>Preview Overlay</button>
          <button id="btn-ts" disabled>Zeitreihe berechnen</button>
          <button id="btn-csv" disabled>CSV herunterladen</button>
          <button id="btn-tif" disabled>GeoTIFF herunterladen</button>
        </div>

        <label>GeoJSON (aktuelles Feature, EPSG:4326)</label>
        <textarea id="geojson" spellcheck="false" placeholder="Hier erscheint das GeoJSON…"></textarea>

        <div class="small">
          Limits: AOI ≤ <b>{{ max_area_km2 }} km²</b> · Preview ≤ <b>{{ max_preview }} px</b> · Zeitreihe-Sampling ≤ <b>{{ max_ts }} px</b>.
        </div>

        <div id="sceneBox" style="display:none;">
          <div class="small" style="margin-top:8px;">Gefundene Szenen (neueste oben):</div>
          <select id="scene" style="width:100%; margin-top:6px;"></select>
        </div>

        <div id="tsBox" style="display:none; margin-top:10px;">
          <div class="small">Zeitreihe (AOI-Statistik):</div>
          <div style="max-height: 260px; overflow:auto; margin-top:6px;">
            <table id="tsTable">
              <thead>
                <tr>
                  <th>datetime</th><th>cloud</th><th>mean</th><th>median</th><th>p10</th><th>p90</th><th>std</th><th>count</th><th>item</th><th>error</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <script>
    const map = L.map('map', { preferCanvas: true }).setView([49.87, 8.65], 11);

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const drawn = new L.FeatureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      draw: { polyline:false, circle:false, circlemarker:false, marker:false, polygon:{ allowIntersection:false, showArea:true }, rectangle:true },
      edit: { featureGroup: drawn, edit:true, remove:false }
    });
    map.addControl(drawControl);

    let overlay = null;
    let currentFeature = null;

    let lastSearch = null;
    let lastTimeseriesJob = null;

    const elGeo = document.getElementById('geojson');
    const elStatus = document.getElementById('status');

    const btnClear = document.getElementById('btn-clear');
    const btnSearch = document.getElementById('btn-search');
    const btnPreview = document.getElementById('btn-preview');
    const btnTS = document.getElementById('btn-ts');
    const btnCSV = document.getElementById('btn-csv');
    const btnTIF = document.getElementById('btn-tif');

    const elN = document.getElementById('n');
    const elCloud = document.getElementById('cloud');
    const elDays = document.getElementById('days');
    const elPad = document.getElementById('pad');
    const elIndex = document.getElementById('index');

    const sceneBox = document.getElementById('sceneBox');
    const elScene = document.getElementById('scene');
    const tsBox = document.getElementById('tsBox');
    const tsTableBody = document.querySelector('#tsTable tbody');

    function setStatus(html, cls){
      elStatus.className = 'status' + (cls ? (' ' + cls) : '');
      elStatus.innerHTML = html;
    }

    function featureToGeoJSON(layer){
      return { type: "Feature", properties: { epsg: 4326 }, geometry: layer.toGeoJSON().geometry };
    }

    function setButtons(){
      const hasFeature = !!currentFeature;
      btnSearch.disabled = !hasFeature;

      const hasScenes = !!(lastSearch && lastSearch.items && lastSearch.items.length);
      btnPreview.disabled = !(hasFeature && hasScenes);
      btnTS.disabled = !(hasFeature && hasScenes);

      btnCSV.disabled = !(lastTimeseriesJob && lastTimeseriesJob.csv);
      btnTIF.disabled = !(lastSearch && lastSearch.preview && lastSearch.preview.geotiff);
    }

    function clearAll(){
      drawn.clearLayers();
      currentFeature = null;

      if(overlay){ map.removeLayer(overlay); overlay = null; }

      elGeo.value = '';
      lastSearch = null;
      lastTimeseriesJob = null;

      sceneBox.style.display = 'none';
      tsBox.style.display = 'none';
      tsTableBody.innerHTML = '';

      setButtons();
      setStatus('Noch keine AOI.', '');
    }

    map.on(L.Draw.Event.CREATED, function (e) {
      drawn.clearLayers();
      if(overlay){ map.removeLayer(overlay); overlay = null; }
      lastSearch = null;
      lastTimeseriesJob = null;
      sceneBox.style.display = 'none';
      tsBox.style.display = 'none';
      tsTableBody.innerHTML = '';

      const layer = e.layer;
      drawn.addLayer(layer);
      currentFeature = layer;

      const gj = featureToGeoJSON(layer);
      elGeo.value = JSON.stringify(gj, null, 2);

      setButtons();
      setStatus('AOI gesetzt. Jetzt <b>Szenen suchen</b>.', 'ok');
    });

    map.on('draw:edited', function(){
      const layers = drawn.getLayers();
      if(layers.length < 1) return;
      currentFeature = layers[0];
      const gj = featureToGeoJSON(currentFeature);
      elGeo.value = JSON.stringify(gj, null, 2);

      if(overlay){ map.removeLayer(overlay); overlay = null; }
      lastSearch = null;
      lastTimeseriesJob = null;
      sceneBox.style.display = 'none';
      tsBox.style.display = 'none';
      tsTableBody.innerHTML = '';

      setButtons();
      setStatus('AOI geändert. Bitte <b>Szenen suchen</b> erneut ausführen.', 'ok');
    });

    btnClear.addEventListener('click', clearAll);

    async function apiJson(url, body){
      const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const ct = (res.headers.get('content-type')||'').toLowerCase();
      const raw = await res.text();
      if(!ct.includes('application/json') && !ct.includes('json')){
        throw new Error(`Server lieferte kein JSON (HTTP ${res.status}, Content-Type=${ct}). Antwort-Auszug: ${raw.slice(0,240)}`);
      }
      const js = raw ? JSON.parse(raw) : {};
      if(!res.ok){
        throw new Error(js && js.error ? js.error : (`HTTP ${res.status}`));
      }
      return js;
    }

    function fillSceneSelect(items){
      elScene.innerHTML = '';
      for(const it of items){
        const opt = document.createElement('option');
        opt.value = it.id;
        opt.textContent = `${it.datetime || ''} · cloud=${it.cloud_cover ?? 'n/a'} · ${it.id}`;
        elScene.appendChild(opt);
      }
      sceneBox.style.display = items.length ? 'block' : 'none';
    }

    async function doSearch(){
      if(!currentFeature) return;
      setButtons();
      setStatus('Suche Szenen via STAC…', '');

      let gj;
      try{ gj = JSON.parse(elGeo.value); }catch(e){ setStatus('GeoJSON ist ungültig.', 'err'); return; }

      const n = Number(elN.value || 10);
      const cloud_max = Number(elCloud.value || 100);
      const lookback_days = Number(elDays.value || 365);

      const data = await apiJson('/api/search', { geojson: gj, n, cloud_max, lookback_days });

      lastSearch = data;

      fillSceneSelect(data.items || []);
      setButtons();

      if(!(data.items || []).length){
        setStatus('Keine Szenen gefunden (Filter zu streng?).', 'err');
        return;
      }

      setStatus(`Gefunden: <b>${data.items.length}</b> Szenen. Wähle eine Szene und nutze <b>Preview Overlay</b> oder <b>Zeitreihe berechnen</b>.`, 'ok');
    }

    async function doPreview(){
      if(!lastSearch || !(lastSearch.items||[]).length) return;
      if(!currentFeature) return;

      setStatus('Berechne Index-Overlay…', '');

      const id = elScene.value || lastSearch.items[0].id;
      const index = elIndex.value;
      const pad_m = Number(elPad.value || 0);

      let gj;
      try{ gj = JSON.parse(elGeo.value); }catch(e){ setStatus('GeoJSON ist ungültig.', 'err'); return; }

      const data = await apiJson('/api/preview', { geojson: gj, item_id: id, index, pad_m });

      if(overlay){ map.removeLayer(overlay); overlay = null; }

      const b = data.overlay.bounds;
      overlay = L.imageOverlay(data.overlay.url, b, { opacity: 1.0, interactive:false }).addTo(map);
      map.fitBounds(L.latLngBounds(b).pad(0.15));

      lastSearch.preview = data.download;

      btnTIF.onclick = () => { window.location = data.download.geotiff; };
      setButtons();

      const st = data.stats || {};
      const mean = (st.mean !== null && st.mean !== undefined) ? st.mean.toFixed(4) : 'n/a';
      setStatus(`Preview ok. <b>${index}</b> · mean=${mean} · item=<span class="mono">${data.item_id}</span>`, 'ok');
    }

    function fmt(x){
      if(x === null || x === undefined) return '';
      if(typeof x === 'number'){
        if(Number.isFinite(x)) return x.toFixed(4);
        return '';
      }
      return String(x);
    }

    async function doTimeseries(){
      if(!lastSearch || !(lastSearch.items||[]).length) return;
      if(!currentFeature) return;

      setStatus('Berechne Zeitreihe… (das kann je nach AOI/N etwas dauern)', '');

      let gj;
      try{ gj = JSON.parse(elGeo.value); }catch(e){ setStatus('GeoJSON ist ungültig.', 'err'); return; }

      const index = elIndex.value;
      const pad_m = Number(elPad.value || 0);

      // use current search parameters:
      const n = Number(elN.value || 10);
      const cloud_max = Number(elCloud.value || 100);
      const lookback_days = Number(elDays.value || 365);

      const data = await apiJson('/api/timeseries', { geojson: gj, index, pad_m, n, cloud_max, lookback_days });

      lastTimeseriesJob = data.download;

      tsTableBody.innerHTML = '';
      for(const r of (data.rows||[])){
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">${(r.datetime||'').replace('T',' ').replace('Z','')}</td>
          <td>${r.cloud_cover ?? ''}</td>
          <td>${fmt(r.mean)}</td>
          <td>${fmt(r.median)}</td>
          <td>${fmt(r.p10)}</td>
          <td>${fmt(r.p90)}</td>
          <td>${fmt(r.std)}</td>
          <td>${r.count ?? ''}</td>
          <td class="mono">${(r.item_id||'').slice(0,18)}${(r.item_id||'').length>18?'…':''}</td>
          <td class="mono" title="${r.error||''}">${(r.error||'').slice(0,28)}${(r.error||'').length>28?'…':''}</td>
        `;
        tsTableBody.appendChild(tr);
      }

      tsBox.style.display = 'block';
      btnCSV.onclick = () => { window.location = data.download.csv; };
      setButtons();
      setStatus(`Zeitreihe fertig. Rows: <b>${(data.rows||[]).length}</b> · CSV verfügbar.`, 'ok');
    }

    btnSearch.addEventListener('click', () => doSearch().catch(e => setStatus('Fehler: ' + e.message, 'err')));
    btnPreview.addEventListener('click', () => doPreview().catch(e => setStatus('Fehler: ' + e.message, 'err')));
    btnTS.addEventListener('click', () => doTimeseries().catch(e => setStatus('Fehler: ' + e.message, 'err')));

    clearAll();
  </script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        default_n=DEFAULT_N_SCENES,
        default_cloud=DEFAULT_CLOUD_MAX,
        default_days=DEFAULT_LOOKBACK_DAYS,
        default_pad=int(DEFAULT_PAD_M),
        indices=list(_index_specs().keys()),
        max_area_km2=MAX_AOI_AREA_KM2,
        max_preview=MAX_PREVIEW_DIM_PX,
        max_ts=MAX_TS_DIM_PX,
    )


@app.post("/api/search")
def api_search():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        geom = _extract_single_geometry(gj)

        n = int(body.get("n", DEFAULT_N_SCENES))
        cloud_max = body.get("cloud_max", DEFAULT_CLOUD_MAX)
        cloud_max = float(cloud_max) if cloud_max is not None else None
        lookback_days = int(body.get("lookback_days", DEFAULT_LOOKBACK_DAYS))

        used_body, items = _stac_search(mapping(geom), n, cloud_max, lookback_days)

        # return minimal list for UI
        out_items = []
        for it in items:
            props = it.get("properties") or {}
            out_items.append({
                "id": it.get("id"),
                "datetime": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover"),
            })

        return jsonify({
            "request": used_body,
            "items": out_items,
            "count": len(out_items),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _fetch_item_by_id(item_id: str) -> Dict[str, Any]:
    # direct item endpoint
    url = f"{STAC_API_URL}/collections/{STAC_COLLECTION}/items/{item_id}"
    return _http_request_json("GET", url, headers={"Accept": "application/geo+json"})


@app.post("/api/preview")
def api_preview():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        item_id = (body.get("item_id") or "").strip()
        index_name = (body.get("index") or "").strip()
        pad_m = float(body.get("pad_m", DEFAULT_PAD_M))

        if not item_id:
            raise ValueError("item_id fehlt.")

        item = _fetch_item_by_id(item_id)
        rr, meta = _preview_from_geojson(gj, item, index_name, pad_m)

        sw, ne = rr.bounds_wgs84
        return jsonify({
            "job_id": rr.job_id,
            "index": rr.index_name,
            "item_id": rr.item_id,
            "datetime": rr.datetime_str,
            "cloud_cover": rr.cloud_cover,
            "stats": meta.get("stats"),
            "debug": meta.get("debug"),
            "overlay": {"url": f"/r/{rr.job_id}/overlay.png", "bounds": [[sw[0], sw[1]], [ne[0], ne[1]]]},
            "download": {"geotiff": f"/r/{rr.job_id}/index.tif"},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/timeseries")
def api_timeseries():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        index_name = (body.get("index") or "").strip()
        pad_m = float(body.get("pad_m", DEFAULT_PAD_M))

        n = int(body.get("n", DEFAULT_N_SCENES))
        cloud_max = body.get("cloud_max", DEFAULT_CLOUD_MAX)
        cloud_max = float(cloud_max) if cloud_max is not None else None
        lookback_days = int(body.get("lookback_days", DEFAULT_LOOKBACK_DAYS))

        geom = _extract_single_geometry(gj)
        used_body, feats = _stac_search(mapping(geom), n, cloud_max, lookback_days)

        # We need full items (assets), so fetch each item by id (stable) to avoid partials
        items_full = []
        for f in feats:
            iid = f.get("id")
            if not iid:
                continue
            items_full.append(_fetch_item_by_id(iid))

        ts = _timeseries_from_geojson(gj, items_full, index_name, pad_m)

        return jsonify({
            "job_id": ts.job_id,
            "index": ts.index_name,
            "rows": ts.rows,
            "download": {"csv": f"/r/{ts.job_id}/timeseries.csv", "json": f"/r/{ts.job_id}/timeseries.json"},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/r/<job_id>/overlay.png")
def job_overlay(job_id: str):
    p = TMP_DIR / f"{job_id}.overlay.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=False, conditional=True)


@app.get("/r/<job_id>/index.tif")
def job_index_tif(job_id: str):
    p = TMP_DIR / f"{job_id}.index.tif"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/tiff", as_attachment=True, download_name=f"index_{job_id}.tif", conditional=True)


@app.get("/r/<job_id>/timeseries.csv")
def job_ts_csv(job_id: str):
    p = TMP_DIR / f"{job_id}.timeseries.csv"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="text/csv", as_attachment=True, download_name=f"timeseries_{job_id}.csv", conditional=True)


@app.get("/r/<job_id>/timeseries.json")
def job_ts_json(job_id: str):
    p = TMP_DIR / f"{job_id}.timeseries.json"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="application/json", as_attachment=True, download_name=f"timeseries_{job_id}.json", conditional=True)


@app.get("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
