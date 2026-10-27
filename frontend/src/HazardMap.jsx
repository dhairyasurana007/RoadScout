import { useCallback, useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./HazardMap.css";

// Northern Virginia / Maryland commute area, shown when there is nothing to fit.
export const DEFAULT_CENTER = [38.9, -77.2];
export const DEFAULT_ZOOM = 9;
const LIST_ZOOM = 15; // zoom used when a list item is clicked
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIBUTION = '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export const STATUSES = ["new", "approved", "rejected", "reported", "possibly_fixed"];
const STATUS_LABELS = {
  new: "New",
  approved: "Approved",
  rejected: "Rejected",
  reported: "Reported",
  possibly_fixed: "Possibly fixed",
};
const SEVERITY_LABELS = { 1: "Low", 2: "Medium", 3: "High" };

const statusText = (status) => STATUS_LABELS[status] ?? String(status);
const severityText = (s) => (SEVERITY_LABELS[s] ? `${s} (${SEVERITY_LABELS[s]})` : String(s));
const hasPosition = (h) => Number.isFinite(h.lat) && Number.isFinite(h.lon);
const locationText = (h) => (hasPosition(h) ? `${h.lat.toFixed(5)}, ${h.lon.toFixed(5)}` : "no location");

function pinIcon(hazard, selected) {
  const html =
    `<span class="rs-pin" data-testid="hazard-pin" data-hazard-id="${Number(hazard.id)}"` +
    ` data-status="${String(hazard.status).replace(/[^a-z_]/g, "")}"` +
    ` data-selected="${selected}">${Number(hazard.severity) || ""}</span>`;
  return L.divIcon({ className: "rs-pin-icon", html, iconSize: [28, 28], iconAnchor: [14, 14] });
}

// Popup content is built from DOM nodes (never an HTML string), so server text cannot inject markup.
function popupNode(hazard) {
  const root = document.createElement("div");
  root.className = "rs-popup";
  root.dataset.testid = "hazard-popup";
  root.dataset.hazardId = String(hazard.id);
  const line = (testid, label, value) => {
    const p = document.createElement("div");
    p.dataset.testid = testid;
    p.textContent = `${label}: ${value}`;
    root.append(p);
  };
  const title = document.createElement("strong");
  title.dataset.testid = "hazard-popup-class";
  title.textContent = hazard.hazard_class;
  root.append(title);
  line("hazard-popup-severity", "Severity", severityText(hazard.severity));
  line("hazard-popup-status", "Status", statusText(hazard.status));
  line("hazard-popup-sightings", "Sightings", hazard.sightings);
  if (hazard.photo_url) {
    const img = document.createElement("img");
    img.dataset.testid = "hazard-popup-photo";
    img.alt = `${hazard.hazard_class} hazard, ${locationText(hazard)}`;
    img.src = hazard.photo_url;
    img.addEventListener("error", () => {
      const note = document.createElement("div");
      note.dataset.testid = "hazard-popup-photo-missing";
      note.textContent = "Photo unavailable";
      img.replaceWith(note);
    });
    root.append(img);
  }
  return root;
}

function Thumb({ hazard }) {
  const [failed, setFailed] = useState(false);
  if (failed || !hazard.photo_url) {
    return (
      <span className="rs-list-missing" data-testid="hazard-list-photo-missing">
        Photo unavailable
      </span>
    );
  }
  return (
    <img
      className="rs-list-thumb"
      data-testid="hazard-list-photo"
      src={hazard.photo_url}
      alt={`${hazard.hazard_class} hazard, ${locationText(hazard)}`}
      onError={() => setFailed(true)}
    />
  );
}

export default function HazardMap() {
  const [data, setData] = useState({ state: "loading", items: [], message: "" });
  const [filter, setFilter] = useState("");
  const [selection, setSelection] = useState({ id: null, source: null, n: 0 });
  const [view, setView] = useState({ lat: DEFAULT_CENTER[0], lon: DEFAULT_CENTER[1], zoom: DEFAULT_ZOOM });
  const [reloadKey, setReloadKey] = useState(0);

  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markers = useRef(new Map()); // hazard id -> L.Marker
  const itemEls = useRef(new Map()); // hazard id -> <li>
  const refitNext = useRef(true); // fit the pins on the next load (first load and filter changes)
  const selectedIdRef = useRef(null); // the selected hazard id, readable inside effects that must not re-run on it

  // The map itself: created once, removed on unmount.
  useEffect(() => {
    const map = L.map(containerRef.current, {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      // No fade or zoom animations: popups and zoom levels change at once, never half-way.
      fadeAnimation: false,
      zoomAnimation: false,
      markerZoomAnimation: false,
    });
    L.tileLayer(TILE_URL, { maxZoom: 19, attribution: ATTRIBUTION }).addTo(map);
    const sync = () => {
      const c = map.getCenter();
      setView({ lat: c.lat, lon: c.lng, zoom: map.getZoom() });
    };
    map.on("moveend zoomend", sync);
    map.on("click", () => setSelection((s) => (s.id === null ? s : { id: null, source: "map", n: s.n + 1 })));
    mapRef.current = map;
    let observer = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => map.invalidateSize());
      observer.observe(containerRef.current);
    }
    return () => {
      observer?.disconnect();
      map.remove();
      mapRef.current = null;
      markers.current.clear();
    };
  }, []);

  // Load the hazards (first load, filter change, refresh); a newer request makes older replies stale.
  useEffect(() => {
    const controller = new AbortController();
    setData((d) => ({ ...d, state: "loading", message: "" }));
    (async () => {
      try {
        const query = filter ? `?status=${encodeURIComponent(filter)}` : "";
        const response = await fetch(`/api/hazards${query}`, { signal: controller.signal });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
          setData({ state: "error", items: [], message: detail || `Could not load hazards (${response.status}).` });
          return;
        }
        const items = await response.json();
        if (!Array.isArray(items)) throw new Error("unexpected response");
        setData({ state: "ready", items, message: "" });
      } catch (err) {
        if (err?.name === "AbortError") return;
        setData({ state: "error", items: [], message: "Could not reach the server." });
      }
    })();
    return () => controller.abort();
  }, [filter, reloadKey]);

  // One pin per hazard that has a position; rebuilt whenever the loaded hazards change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || data.state === "loading") return;
    for (const marker of markers.current.values()) marker.remove();
    markers.current.clear();
    const points = [];
    for (const h of data.items) {
      if (!hasPosition(h)) continue;
      const marker = L.marker([h.lat, h.lon], { icon: pinIcon(h, h.id === selectedIdRef.current), keyboard: false, riseOnHover: true });
      marker.bindPopup(popupNode(h), { autoPanPadding: [20, 20] });
      marker.on("click", () => setSelection((s) => ({ id: h.id, source: "pin", n: s.n + 1 })));
      marker.addTo(map);
      markers.current.set(h.id, marker);
      points.push([h.lat, h.lon]);
    }
    if (refitNext.current) {
      refitNext.current = false;
      if (points.length > 0) map.fitBounds(L.latLngBounds(points), { padding: [40, 40], maxZoom: 16, animate: false });
      else map.setView(DEFAULT_CENTER, DEFAULT_ZOOM, { animate: false });
    }
    // Keep the selection if its hazard is still there, else drop it.
    setSelection((s) => (s.id !== null && !markers.current.has(s.id) ? { id: null, source: "load", n: s.n + 1 } : s));
  }, [data]);

  // Selection: highlight the pin, and either scroll the list (pin click) or pan/zoom and open the popup (list click).
  useEffect(() => {
    for (const [id, marker] of markers.current) {
      const selected = id === selection.id;
      const pin = marker.getElement()?.querySelector("[data-hazard-id]");
      if (pin) pin.dataset.selected = String(selected);
      marker.setZIndexOffset(selected ? 1000 : 0);
    }
    const map = mapRef.current;
    if (selection.id === null) {
      map?.closePopup();
      return;
    }
    if (selection.source === "pin") {
      itemEls.current.get(selection.id)?.scrollIntoView({ block: "nearest" });
    } else if (selection.source === "list") {
      const marker = markers.current.get(selection.id);
      if (map && marker) {
        map.setView(marker.getLatLng(), Math.max(map.getZoom(), LIST_ZOOM), { animate: false });
        marker.openPopup();
      }
    }
  }, [selection]);

  const refresh = useCallback(() => setReloadKey((k) => k + 1), []);
  const onFilter = (event) => {
    refitNext.current = true;
    setFilter(event.target.value);
  };

  selectedIdRef.current = selection.id;
  const loading = data.state === "loading";
  const selectedId = selection.id === null ? "" : String(selection.id);

  return (
    <section aria-label="Hazard map" data-testid="hazard-map-section" data-selected-hazard-id={selectedId}>
      <h2>Hazard map</h2>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap", marginBottom: "0.5rem" }}>
        <label>
          Status{" "}
          <select data-testid="hazard-map-filter" value={filter} onChange={onFilter}>
            <option value="">All statuses</option>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {statusText(s)}
              </option>
            ))}
          </select>
        </label>
        <button type="button" data-testid="hazard-map-refresh" onClick={refresh} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
        {data.state === "ready" && data.items.length > 0 && (
          <span data-testid="hazard-map-count">
            {data.items.length} {data.items.length === 1 ? "hazard" : "hazards"}
          </span>
        )}
      </div>
      {loading && <p data-testid="hazard-map-loading">Loading hazards...</p>}
      {data.state === "error" && (
        <p data-testid="hazard-map-error" role="alert" style={{ color: "#b00020" }}>
          {data.message}
        </p>
      )}
      {data.state === "ready" && data.items.length === 0 && (
        <p data-testid="hazard-map-empty">{filter ? `No hazards with status ${statusText(filter)}` : "No hazards yet"}</p>
      )}

      <div className="rs-map-layout">
        <div
          ref={containerRef}
          className="rs-map"
          data-testid="hazard-map"
          data-map-lat={view.lat.toFixed(5)}
          data-map-lon={view.lon.toFixed(5)}
          data-map-zoom={view.zoom}
        />
        <ul className="rs-map-list" data-testid="hazard-map-list" aria-label="Hazards">
          {data.items.map((h) => {
            const selected = h.id === selection.id;
            return (
              <li
                key={h.id}
                ref={(el) => {
                  if (el) itemEls.current.set(h.id, el);
                  else itemEls.current.delete(h.id);
                }}
                data-testid="hazard-list-item"
                data-hazard-id={h.id}
                data-selected={String(selected)}
              >
                <button
                  type="button"
                  className="rs-list-button"
                  aria-pressed={selected}
                  onClick={() => setSelection((s) => ({ id: h.id, source: "list", n: s.n + 1 }))}
                >
                  <Thumb hazard={h} />
                  <span>
                    <strong data-testid="hazard-list-class">{h.hazard_class}</strong>
                    <br />
                    <span data-testid="hazard-list-severity">Severity {severityText(h.severity)}</span>
                    <br />
                    <span data-testid="hazard-list-status">{statusText(h.status)}</span>
                    <br />
                    <span data-testid="hazard-list-sightings">
                      {h.sightings} {h.sightings === 1 ? "sighting" : "sightings"}
                    </span>
                    <br />
                    <span data-testid="hazard-list-location">{locationText(h)}</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
