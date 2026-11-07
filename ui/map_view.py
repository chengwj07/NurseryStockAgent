"""Leaflet map embedded as an iframe (``srcdoc``) with layer toggles and tiles."""

from __future__ import annotations

import html
import json
from typing import Optional

from agent.state import SessionState
from services.storage import ImageStore

_PAGE: str = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0}#map{background:#f2f2f2}</style></head><body><div id="map"></div>
<script>
const cfg = __CFG__;
const map = L.map('map', {preferCanvas: true, zoomSnap: 0.5, maxZoom: 24}).setView(cfg.init.slice(0, 2), cfg.init[2]);
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 22, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap'});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 22, maxNativeZoom: 19, attribution: 'Esri'});
if (cfg.basemap) { osm.addTo(map); }
const overlays = {};
cfg.rasters.forEach(r => {
  const layer = L.tileLayer(cfg.base + '/tiles/' + r.id + '/{z}/{x}/{y}.png', {
    maxZoom: 24, bounds: [[r.b[1], r.b[0]], [r.b[3], r.b[2]]]});
  overlays['🖼 ' + r.name] = layer; layer.addTo(map);
});
cfg.vectors.forEach(v => {
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const layer = L.geoJSON(null, {style: {color: v.color, weight: 1.5, fillOpacity: 0.35},
    onEachFeature: (f, l) => {
      const p = f.properties || {};
      let tip = '<b>' + esc(p.species || '') + '</b>';
      if (p.price_text) { tip += '<br>价格：' + esc(p.price_text); }
      l.bindTooltip(tip, {sticky: true});
    }});
  overlays['🌳 ' + v.name] = layer; layer.addTo(map);
  fetch(cfg.base + '/layers/' + v.id + '.geojson?v=' + cfg.v).then(r => r.json()).then(g => layer.addData(g));
});
const bases = cfg.basemap ? {'OpenStreetMap': osm, '卫星影像': sat} : {};
L.control.layers(bases, overlays, {collapsed: false}).addTo(map);
function fitToImage() {
  map.invalidateSize();
  if (cfg.fit) { map.fitBounds([[cfg.fit[1], cfg.fit[0]], [cfg.fit[3], cfg.fit[2]]], {animate: false}); }
}
fitToImage();
map.whenReady(fitToImage);
window.addEventListener('load', fitToImage);
setTimeout(fitToImage, 300);
</script></body></html>"""


def build_map_html(
    session: SessionState,
    images: ImageStore,
    base_url: str = "",
    height: int = 560,
    init_view: tuple[float, float, float] = (29.2, 120.1, 7.0),
) -> str:
    """Render the map iframe for a session.

    Args:
        session: Session whose images and layers are shown.
        images: Image store (kept for symmetry / future use).
        base_url: Server URL prefix for tiles/GeoJSON; empty = relative to the page.
        height: Iframe height in pixels.
        init_view: ``(lat, lon, zoom)`` shown before an image is loaded.

    Returns:
        HTML string for ``gr.HTML``.
    """
    del images  # image metadata already lives in the session records
    current = session.images.get(session.current_image_id or "")
    fit: Optional[tuple[float, float, float, float]] = current.bounds if current else None
    cfg = {
        "base": base_url.rstrip("/"),
        "init": list(init_view),
        "v": session.version,
        "rasters": [
            {"id": i.image_id, "name": i.name, "b": list(i.bounds)} for i in session.images.values()
        ],
        "vectors": [
            {"id": l.layer_id, "name": l.name, "color": l.color} for l in session.layers
        ],
        "fit": list(fit) if fit else None,
        # Plain images (PNG/JPG/untagged TIFF) have no real location: no basemap.
        "basemap": bool(current.georeferenced) if current else True,
    }
    payload = json.dumps(cfg, ensure_ascii=False).replace("</", "<\\/")
    page = _PAGE.replace("__CFG__", payload)
    return (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        f'style="width:100%;height:{height}px;border:1px solid #ccc;border-radius:6px"></iframe>'
    )
