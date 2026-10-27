import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
// The classes of config/hazard_prompts.json (the API refuses any other class when a hazard is edited).
export const HAZARD_CLASSES = [
  "pothole",
  "debris in lane",
  "damaged guardrail",
  "fallen or damaged sign",
  "faded lane markings",
];
const SEVERITIES = [1, 2, 3];

const statusText = (status) => STATUS_LABELS[status] ?? String(status);
const severityText = (s) => (SEVERITY_LABELS[s] ? `${s} (${SEVERITY_LABELS[s]})` : String(s));
const hasPosition = (h) => Number.isFinite(h.lat) && Number.isFinite(h.lon);
const locationText = (h) => (hasPosition(h) ? `${h.lat.toFixed(5)}, ${h.lon.toFixed(5)}` : "no location");
const cssToken = (status) => String(status).replace(/[^a-z_]/g, "");

function pinIcon(hazard, selected) {
  const html =
    `<span class="rs-pin" data-testid="hazard-pin" data-hazard-id="${Number(hazard.id)}"` +
    ` data-status="${cssToken(hazard.status)}"` +
    ` data-selected="${selected}">${Number(hazard.severity) || ""}</span>`;
  return L.divIcon({ className: "rs-pin-icon", html, iconSize: [28, 28], iconAnchor: [14, 14] });
}

// Keep an existing pin in step with its hazard without rebuilding it (so an open popup stays open).
function syncPin(marker, hazard, selected) {
  const el = marker.getElement()?.querySelector("[data-hazard-id]");
  if (!el) return;
  el.dataset.status = cssToken(hazard.status);
  el.dataset.selected = String(selected);
  const label = String(Number(hazard.severity) || "");
  if (el.textContent !== label) el.textContent = label;
}

async function errorMessage(response, fallback) {
  const body = await response.json().catch(() => ({}));
  const detail = Array.isArray(body?.detail) ? body.detail.map((d) => d.msg).join("; ") : body?.detail;
  return typeof detail === "string" && detail ? detail : `${fallback} (${response.status}).`;
}

// Review requests in flight, kept outside the component: leaving the map tab and coming back while a request is
// running must not re-enable that hazard's buttons. `epoch` counts finished requests (see the list loader).
const reviewStore = {
  inFlight: new Set(),
  epoch: 0,
  listeners: new Set(),
  start(id) {
    this.inFlight.add(id);
    this.emit();
  },
  finish(id) {
    this.inFlight.delete(id);
    this.epoch += 1;
    this.emit();
  },
  emit() {
    for (const listener of [...this.listeners]) listener();
  },
};

function StatusBadge({ status, testid }) {
  return (
    <span className="rs-badge" data-testid={testid} data-status={cssToken(status)}>
      {statusText(status)}
    </span>
  );
}

function EditedMark({ testid, was }) {
  return (
    <span className="rs-edited" data-testid={testid} title={was == null ? "Edited" : `Edited (was ${was})`}>
      edited
    </span>
  );
}

// Approve / Reject / Reset and the inline class + severity edit of one hazard; used in the list and in the popup.
// ctx = { pending: Set of hazard ids with a request in flight, errors: { id: message }, submit(id, body), clearError(id) }
function ReviewControls({ hazard, ctx }) {
  const id = hazard.id;
  const busy = ctx.pending.has(id);
  const error = ctx.errors[id];
  const [editing, setEditing] = useState(false);
  const [draftClass, setDraftClass] = useState(hazard.hazard_class);
  const [draftSeverity, setDraftSeverity] = useState(String(hazard.severity));
  const classRef = useRef(null);
  const editRef = useRef(null);
  const wasEditing = useRef(false);

  // Keyboard users: focus moves into the form when it opens and back to Edit when it closes.
  useEffect(() => {
    if (editing) classRef.current?.focus();
    else if (wasEditing.current) editRef.current?.focus();
    wasEditing.current = editing;
  }, [editing]);

  const changes = {};
  if (draftClass !== hazard.hazard_class) changes.hazard_class = draftClass;
  if (Number(draftSeverity) !== Number(hazard.severity)) changes.severity = Number(draftSeverity);
  const dirty = Object.keys(changes).length > 0;

  const classOptions = HAZARD_CLASSES.includes(hazard.hazard_class) ? HAZARD_CLASSES : [hazard.hazard_class, ...HAZARD_CLASSES];
  const severityOptions = SEVERITIES.includes(Number(hazard.severity)) ? SEVERITIES : [Number(hazard.severity), ...SEVERITIES];

  const startEdit = () => {
    setDraftClass(hazard.hazard_class);
    setDraftSeverity(String(hazard.severity));
    ctx.clearError(id);
    setEditing(true);
  };
  const cancel = () => {
    ctx.clearError(id);
    setEditing(false);
  };
  const save = async (event) => {
    event.preventDefault();
    if (!dirty || busy) return;
    const result = await ctx.submit(id, changes);
    if (result.ok) setEditing(false);
  };

  return (
    <div className="rs-review" data-testid="hazard-review" data-busy={String(busy)} aria-busy={busy}>
      <div className="rs-review-buttons" role="group" aria-label={`Review hazard ${id}`}>
        <button
          type="button"
          data-testid="hazard-approve"
          aria-label={`Approve hazard ${id}`}
          disabled={busy || hazard.status === "approved"}
          onClick={() => ctx.submit(id, { status: "approved" })}
        >
          Approve
        </button>
        <button
          type="button"
          data-testid="hazard-reject"
          aria-label={`Reject hazard ${id}`}
          disabled={busy || hazard.status === "rejected"}
          onClick={() => ctx.submit(id, { status: "rejected" })}
        >
          Reject
        </button>
        <button
          type="button"
          data-testid="hazard-reset"
          aria-label={`Reset hazard ${id} to new`}
          disabled={busy || hazard.status === "new"}
          onClick={() => ctx.submit(id, { status: "new" })}
        >
          Reset
        </button>
        {!editing && (
          <button
            type="button"
            ref={editRef}
            data-testid="hazard-edit"
            aria-label={`Edit class or severity of hazard ${id}`}
            disabled={busy}
            onClick={startEdit}
          >
            Edit
          </button>
        )}
      </div>
      {editing && (
        <form className="rs-edit" data-testid="hazard-edit-form" aria-label={`Edit hazard ${id}`} onSubmit={save}>
          <label>
            Class{" "}
            <select
              ref={classRef}
              data-testid="hazard-edit-class"
              aria-label={`Class of hazard ${id}`}
              value={draftClass}
              disabled={busy}
              onChange={(e) => setDraftClass(e.target.value)}
            >
              {classOptions.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            Severity{" "}
            <select
              data-testid="hazard-edit-severity"
              aria-label={`Severity of hazard ${id}`}
              value={draftSeverity}
              disabled={busy}
              onChange={(e) => setDraftSeverity(e.target.value)}
            >
              {severityOptions.map((v) => (
                <option key={v} value={String(v)}>
                  {severityText(v)}
                </option>
              ))}
            </select>
          </label>
          <div className="rs-review-buttons">
            <button type="submit" data-testid="hazard-save" aria-label={`Save changes to hazard ${id}`} disabled={busy || !dirty}>
              Save
            </button>
            <button type="button" data-testid="hazard-cancel" aria-label={`Cancel editing hazard ${id}`} disabled={busy} onClick={cancel}>
              Cancel
            </button>
          </div>
          {!dirty && !busy && (
            <span className="rs-hint" data-testid="hazard-edit-hint">
              Change the class or severity to save.
            </span>
          )}
        </form>
      )}
      {busy && (
        <span className="rs-hint" role="status" data-testid="hazard-review-saving">
          Saving...
        </span>
      )}
      {error && (
        <p className="rs-review-error" role="alert" data-testid="hazard-review-error">
          {error}
        </p>
      )}
    </div>
  );
}

// The content of a pin's popup. React renders it into a plain element (a portal) that Leaflet shows as the popup.
function PopupContent({ hazard, ctx }) {
  const [photoFailed, setPhotoFailed] = useState(false);
  return (
    <div className="rs-popup" data-testid="hazard-popup" data-hazard-id={String(hazard.id)}>
      <strong data-testid="hazard-popup-class">{hazard.hazard_class}</strong>
      {hazard.class_edited && <EditedMark testid="hazard-popup-class-edited" was={hazard.original_hazard_class} />}
      <div data-testid="hazard-popup-severity">
        Severity: {severityText(hazard.severity)}
        {hazard.severity_edited && <EditedMark testid="hazard-popup-severity-edited" was={hazard.original_severity} />}
      </div>
      <div data-testid="hazard-popup-status">
        Status: <StatusBadge status={hazard.status} testid="hazard-popup-status-badge" />
      </div>
      <div data-testid="hazard-popup-sightings">Sightings: {hazard.sightings}</div>
      {hazard.photo_url &&
        (photoFailed ? (
          <div data-testid="hazard-popup-photo-missing">Photo unavailable</div>
        ) : (
          <img
            data-testid="hazard-popup-photo"
            alt={`${hazard.hazard_class} hazard, ${locationText(hazard)}`}
            src={hazard.photo_url}
            onError={() => setPhotoFailed(true)}
          />
        ))}
      <ReviewControls hazard={hazard} ctx={ctx} />
    </div>
  );
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
  const [pending, setPending] = useState(() => new Set(reviewStore.inFlight)); // hazard ids with a review request in flight
  const [errors, setErrors] = useState({}); // hazard id -> last review error
  const [notice, setNotice] = useState(""); // page-level message that outlives a hazard (e.g. it was removed)
  const [hostVersion, setHostVersion] = useState(0); // bumped when popup hosts are added or removed

  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markers = useRef(new Map()); // hazard id -> L.Marker
  const hosts = useRef(new Map()); // hazard id -> { el, observer }: the element a pin's popup shows
  const itemEls = useRef(new Map()); // hazard id -> <li>
  const refitNext = useRef(true); // fit the pins on the next load (first load and filter changes)
  const selectedIdRef = useRef(null); // the selected hazard id, readable inside effects that must not re-run on it
  const filterRef = useRef("");
  const staleReload = useRef(false); // a list reply was dropped as stale: reload once the reviews have settled

  filterRef.current = filter;

  const dropMarker = (id) => {
    markers.current.get(id)?.remove();
    markers.current.delete(id);
    hosts.current.get(id)?.observer?.disconnect();
    hosts.current.delete(id);
  };

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
      for (const host of hosts.current.values()) host.observer?.disconnect();
      hosts.current.clear();
      map.remove();
      mapRef.current = null;
      markers.current.clear();
    };
  }, []);

  // Load the hazards (first load, filter change, refresh); a newer request makes older replies stale.
  useEffect(() => {
    const controller = new AbortController();
    setData((d) => ({ ...d, state: "loading", message: "" }));
    // A reply that arrives after a review finished may predate it and would show the old status again, so it is
    // dropped and the list loaded once more. A reply to a request that started during a review is shown, and the
    // list is loaded again when the reviews have settled.
    const startEpoch = reviewStore.epoch;
    const startedBusy = reviewStore.inFlight.size > 0;
    (async () => {
      try {
        const query = filter ? `?status=${encodeURIComponent(filter)}` : "";
        const response = await fetch(`/api/hazards${query}`, { signal: controller.signal });
        if (!response.ok) {
          setData({ state: "error", items: [], message: await errorMessage(response, "Could not load hazards") });
          return;
        }
        const items = await response.json();
        if (!Array.isArray(items)) throw new Error("unexpected response");
        if (controller.signal.aborted) return;
        if (reviewStore.epoch !== startEpoch) {
          // A review finished meanwhile: this reply may predate it. Drop it and load again.
          if (reviewStore.inFlight.size === 0) setReloadKey((k) => k + 1);
          else staleReload.current = true;
          return;
        }
        if (startedBusy) staleReload.current = true; // show it, but load again when the reviews have settled
        setData({ state: "ready", items, message: "" });
      } catch (err) {
        if (err?.name === "AbortError") return;
        setData({ state: "error", items: [], message: "Could not reach the server." });
      }
    })();
    return () => controller.abort();
  }, [filter, reloadKey]);

  // Follow the review requests in flight (also those started before this view was mounted).
  useEffect(() => {
    const listener = () => {
      setPending(new Set(reviewStore.inFlight));
      if (staleReload.current && reviewStore.inFlight.size === 0) {
        staleReload.current = false;
        setReloadKey((k) => k + 1);
      }
    };
    reviewStore.listeners.add(listener);
    listener();
    return () => {
      reviewStore.listeners.delete(listener);
    };
  }, []);

  // One pin per hazard that has a position. Existing pins are updated in place (colour, severity), new ones
  // are added and the ones that left the list are removed, so an open popup survives a review.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || data.state === "loading") return;
    const wanted = new Map(data.items.filter(hasPosition).map((h) => [h.id, h]));
    let hostsChanged = false;
    for (const [id, marker] of [...markers.current]) {
      const h = wanted.get(id);
      const at = marker.getLatLng();
      if (!h || at.lat !== h.lat || at.lng !== h.lon) {
        dropMarker(id);
        hostsChanged = true;
      }
    }
    const points = [];
    for (const h of wanted.values()) {
      let marker = markers.current.get(h.id);
      if (marker) {
        syncPin(marker, h, h.id === selectedIdRef.current);
      } else {
        marker = L.marker([h.lat, h.lon], { icon: pinIcon(h, h.id === selectedIdRef.current), keyboard: false, riseOnHover: true });
        const el = document.createElement("div");
        // A click inside the popup must not reach the map: a control that removes itself while handling the click
        // (Edit, Cancel) would otherwise look like a click on the bare map and close the popup.
        el.addEventListener("click", (e) => e.stopPropagation());
        const record = { el, observer: null };
        if (typeof ResizeObserver !== "undefined") {
          // The popup grows when the edit form opens; Leaflet only re-lays it out when told to.
          record.observer = new ResizeObserver(() => {
            // Next frame: laying the popup out inside the observer callback would loop the observer.
            requestAnimationFrame(() => {
              if (!marker.isPopupOpen()) return;
              // update() re-attaches the content, which drops keyboard focus; put it back.
              const focused = document.activeElement;
              marker.getPopup().update();
              if (focused && el.contains(focused) && document.activeElement !== focused) focused.focus({ preventScroll: true });
            });
          });
          record.observer.observe(el);
        }
        hosts.current.set(h.id, record);
        marker.bindPopup(el, { autoPanPadding: [20, 20] });
        marker.on("click", () => setSelection((s) => ({ id: h.id, source: "pin", n: s.n + 1 })));
        marker.addTo(map);
        markers.current.set(h.id, marker);
        hostsChanged = true;
      }
      points.push([h.lat, h.lon]);
    }
    if (hostsChanged) setHostVersion((v) => v + 1);
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

  const refresh = useCallback(() => {
    setNotice("");
    setReloadKey((k) => k + 1);
  }, []);
  const onFilter = (event) => {
    refitNext.current = true;
    setFilter(event.target.value);
  };

  const clearError = useCallback((id) => {
    setErrors((e) => {
      if (!(id in e)) return e;
      const { [id]: _gone, ...rest } = e;
      return rest;
    });
  }, []);

  // Send one review (PATCH) for one hazard. Resolves { ok } and never throws. One request per hazard at a time;
  // other hazards stay usable. On any failure the hazard on screen is left as it was.
  const submit = useCallback(
    async (id, body) => {
      if (reviewStore.inFlight.has(id)) return { ok: false, skipped: true };
      reviewStore.start(id);
      clearError(id);
      setNotice("");
      let outcome = { ok: false };
      try {
        const response = await fetch(`/api/hazards/${id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (response.ok) {
          const updated = await response.json();
          if (!updated || typeof updated !== "object" || updated.id !== id) throw new Error("unexpected response");
          setData((d) => ({
            ...d,
            // Under a status filter a hazard that no longer matches leaves the list.
            items: d.items.flatMap((h) =>
              h.id !== id ? [h] : filterRef.current && updated.status !== filterRef.current ? [] : [{ ...h, ...updated }],
            ),
          }));
          outcome = { ok: true };
        } else {
          const message = await errorMessage(response, "Could not save the review");
          setErrors((e) => ({ ...e, [id]: message }));
          if (response.status === 404) {
            // The hazard is gone (deleted by a re-run, say): reload so the list matches the server.
            setNotice(`Hazard ${id} was not found, so the list was refreshed.`);
            staleReload.current = true; // reloads as soon as this request has settled (below)
          }
        }
      } catch (err) {
        const message = err?.message === "unexpected response" ? "Unexpected response from the server." : "Could not reach the server.";
        setErrors((e) => ({ ...e, [id]: message }));
      } finally {
        reviewStore.finish(id);
      }
      return outcome;
    },
    [clearError],
  );
  const ctx = { pending, errors, submit, clearError };

  selectedIdRef.current = selection.id;
  const loading = data.state === "loading";
  const selectedId = selection.id === null ? "" : String(selection.id);
  const itemsById = new Map(data.items.map((h) => [h.id, h]));

  return (
    <section aria-label="Hazard map" data-testid="hazard-map-section" data-selected-hazard-id={selectedId} data-host-version={hostVersion}>
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
      {notice && (
        <p data-testid="hazard-review-notice" role="status">
          {notice}
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
                    {h.class_edited && <EditedMark testid="hazard-class-edited" was={h.original_hazard_class} />}
                    <br />
                    <span data-testid="hazard-list-severity">Severity {severityText(h.severity)}</span>
                    {h.severity_edited && <EditedMark testid="hazard-severity-edited" was={h.original_severity} />}
                    <br />
                    <StatusBadge status={h.status} testid="hazard-list-status" />
                    <br />
                    <span data-testid="hazard-list-sightings">
                      {h.sightings} {h.sightings === 1 ? "sighting" : "sightings"}
                    </span>
                    <br />
                    <span data-testid="hazard-list-location">{locationText(h)}</span>
                  </span>
                </button>
                <ReviewControls hazard={h} ctx={ctx} />
              </li>
            );
          })}
        </ul>
      </div>
      {[...hosts.current].map(([id, host]) =>
        itemsById.has(id) ? createPortal(<PopupContent hazard={itemsById.get(id)} ctx={ctx} />, host.el, String(id)) : null,
      )}
    </section>
  );
}
