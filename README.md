AOI zeichnen (Leaflet Draw, 1 Feature)

STAC (Default: Earth Search / sentinel-2-l2a): „letzte N Szenen“ + Cloud-Filter

Indizes: NDVI, NDMI, NDWI, NDRE1/NDRE2/NDRE3 (RedEdge-Indizes)

Preview: farbiges Index-Overlay (PNG) + GeoTIFF (float32, nodata)

Zeitreihe: pro Szene mean/median/p10/p90/std im AOI + CSV Download

Robust gegen STAC-Sort-Inkompatibilitäten (fallback: verschiedene sortby-Varianten, sonst client-seitiges Sortieren)

API Kurzreferenz
POST /api/search
Body: { geojson, n, cloud_max, lookback_days } → Items (id/datetime/cloud)

POST /api/preview
Body: { geojson, item_id, index, pad_m } → Overlay PNG + GeoTIFF Link + Stats

POST /api/timeseries
Body: { geojson, index, pad_m, n, cloud_max, lookback_days } → rows + CSV/JSON Links