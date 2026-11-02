import { useEffect, useRef, useState } from "react";
import HazardMap from "./HazardMap.jsx";
import { DEMO_BANNER_TEXT, DemoContext, useDemoStatus } from "./demo.jsx";

const DEFAULT_POLL_MS = 1500;

// The poll interval can be shortened with ?pollMs=50 (used by the Playwright tests).
function pollIntervalMs() {
  const raw = new URLSearchParams(window.location.search).get("pollMs");
  const value = raw === null ? NaN : Number(raw);
  return Number.isFinite(value) && value >= 10 && value <= 60000 ? value : DEFAULT_POLL_MS;
}

const SEVERITY_LABELS = { 1: "Low", 2: "Medium", 3: "High" };

function severityText(severity) {
  const label = SEVERITY_LABELS[severity];
  return label ? `${severity} (${label})` : String(severity);
}

function locationText(hazard) {
  const { lat, lon } = hazard;
  if (typeof lat !== "number" || typeof lon !== "number") return "no location";
  return `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

async function errorDetail(response, fallback) {
  const data = await response.json().catch(() => ({}));
  const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
  return detail || `${fallback} (${response.status}).`;
}

function HazardCard({ hazard }) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const where = locationText(hazard);
  return (
    <li
      data-testid="hazard-item"
      data-candidate-id={hazard.candidate_id}
      className="rs-hazard-card"
    >
      {photoFailed || !hazard.photo_url ? (
        <div data-testid="hazard-photo-missing" className="rs-photo-missing">
          Photo unavailable
        </div>
      ) : (
        <img
          data-testid="hazard-photo"
          src={hazard.photo_url}
          alt={`${hazard.hazard_class} hazard, ${where}`}
          width={240}
          onError={() => setPhotoFailed(true)}
        />
      )}
      <dl>
        <dt>Class</dt>
        <dd data-testid="hazard-class">
          <span className="rs-pill">{hazard.hazard_class}</span>
        </dd>
        <dt>Severity</dt>
        <dd data-testid="hazard-severity">
          <span className="rs-pill rs-sev" data-severity={hazard.severity}>
            {severityText(hazard.severity)}
          </span>
        </dd>
        <dt>Confidence</dt>
        <dd data-testid="hazard-confidence">{Math.round(hazard.confidence * 100)}%</dd>
        <dt>Location</dt>
        <dd data-testid="hazard-location">{where}</dd>
        {hazard.report_description && (
          <>
            <dt>Description</dt>
            <dd data-testid="hazard-description">{hazard.report_description}</dd>
          </>
        )}
      </dl>
    </li>
  );
}

function Hazards({ hazards }) {
  if (hazards.state === "loading") return <p data-testid="hazards-loading" className="rs-hint-text">Loading hazards...</p>;
  if (hazards.state === "error")
    return (
      <p data-testid="hazards-error" role="alert" className="rs-alert">
        {hazards.message}
      </p>
    );
  if (hazards.state !== "ready") return null;
  if (hazards.items.length === 0) return <p data-testid="hazards-empty" className="rs-empty">No verified hazards</p>;
  return (
    <>
      <p data-testid="hazard-count" className="rs-count">
        {hazards.items.length} verified {hazards.items.length === 1 ? "hazard" : "hazards"}
      </p>
      <ul data-testid="hazard-list" className="rs-hazard-list">
        {hazards.items.map((h) => (
          <HazardCard key={h.candidate_id} hazard={h} />
        ))}
      </ul>
    </>
  );
}

// A file field styled as a drop target. The native <input type="file"> still receives the drop; this only adds the
// drag-over highlight.
function DropField({ children }) {
  const [over, setOver] = useState(false);
  return (
    <label
      className="rs-dropzone"
      data-dragover={over}
      onDragEnter={() => setOver(true)}
      onDragOver={() => setOver(true)}
      onDragLeave={() => setOver(false)}
      onDrop={() => setOver(false)}
    >
      {children}
    </label>
  );
}

// Visual only (hidden from assistive technology): the pipeline stages as a row of steps. The stage name itself is in
// the "Stage" text of the drive panel.
const STEP_LABELS = [
  ["preprocessing", "Prepare"],
  ["detecting", "Detect"],
  ["aggregating", "Group"],
  ["verifying", "Verify"],
  ["blurring", "Blur"],
  ["deduplicating", "Merge"],
  ["checking_fixes", "Recheck"],
];

function StageSteps({ drive }) {
  if (!["processing", "done", "failed"].includes(drive.status)) return null;
  const names = STEP_LABELS.map(([name]) => name);
  const active = drive.status === "failed" ? names.indexOf(drive.failed_stage) : names.indexOf(drive.stage);
  return (
    <ol className="rs-steps" aria-hidden="true">
      {STEP_LABELS.map(([name, label], i) => {
        let state = "todo";
        if (drive.status === "done") state = "done";
        else if (active >= 0 && i < active) state = "done";
        else if (active === i) state = drive.status === "failed" ? "failed" : "current";
        else if (active < 0 && drive.status === "processing" && i === 0) state = "current";
        return <li key={name} className="rs-step" data-step={state} data-label={label} />;
      })}
    </ol>
  );
}

// The page has two views: the upload/results page and the hazard map. #map opens the map directly.
function initialTab() {
  return window.location.hash === "#map" ? "map" : "drives";
}

export default function App() {
  const demo = useDemoStatus(); // read-only hosted demo: banner, and no controls that change anything
  const [tab, setTab] = useState(initialTab);
  const [video, setVideo] = useState(null);
  const [gpx, setGpx] = useState(null);
  const [offset, setOffset] = useState("0");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [drive, setDrive] = useState(null);
  const [origin, setOrigin] = useState("created"); // "created" (just uploaded) or "loaded" (by id)
  const [starting, setStarting] = useState(false);
  const [pollError, setPollError] = useState("");
  const [hazards, setHazards] = useState({ state: "idle", items: [], message: "" });
  const [loadId, setLoadId] = useState("");
  const [loading, setLoading] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deletingIds, setDeletingIds] = useState([]); // drives with a delete request in flight
  const [deleteError, setDeleteError] = useState("");
  // Same ids, updated synchronously: blocks a double click before React re-renders the disabled
  // button. Keyed by drive, so a delete in flight for one drive never blocks another.
  const deletes = useRef(new Set());
  const submitting = useRef(false); // blocks a double click before React re-renders the disabled button
  const starts = useRef(false);
  const loads = useRef(false);
  const pollMs = useRef(pollIntervalMs());
  // The id of the drive the page is showing, updated synchronously (state only changes on the next
  // render). Responses for any other drive are stale and are dropped, so a slow poll or /process
  // reply can never overwrite a drive that was opened or uploaded in the meantime.
  const shownId = useRef(null);

  function showDrive(next) {
    shownId.current = next?.id ?? null;
    setDrive(next);
  }

  const driveId = drive?.id;
  const status = drive?.status;

  // The delete confirmation and its error belong to one drive; close them when another is shown, and
  // when a run starts (the button is disabled while processing).
  useEffect(() => {
    setConfirmingDelete(false);
    setDeleteError("");
  }, [driveId]);
  useEffect(() => {
    if (status === "processing") setConfirmingDelete(false);
  }, [status]);

  // Poll the drive while it is processing; stops when it is done or failed, and on unmount.
  useEffect(() => {
    if (!driveId || status !== "processing") return undefined;
    let cancelled = false;
    let timer = null;
    const controller = new AbortController();
    const stale = () => cancelled || shownId.current !== driveId;

    async function tick() {
      try {
        const response = await fetch(`/api/drives/${encodeURIComponent(driveId)}`, {
          signal: controller.signal,
        });
        if (stale()) return;
        if (response.ok) {
          const data = await response.json();
          if (stale() || data.id !== driveId) return;
          setPollError("");
          setDrive(data);
          if (data.status !== "processing") return; // the effect re-runs and stops
        } else if (response.status === 404) {
          setPollError("Drive not found.");
          return;
        } else {
          setPollError(`Could not refresh the status (${response.status}). Retrying...`);
        }
      } catch {
        if (stale()) return;
        setPollError("Could not reach the server. Retrying...");
      }
      timer = setTimeout(tick, pollMs.current);
    }

    timer = setTimeout(tick, pollMs.current);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
    };
  }, [driveId, status]);

  // Fetch the verified hazards once the drive is done; clear them otherwise.
  useEffect(() => {
    if (!driveId || status !== "done") {
      setHazards({ state: "idle", items: [], message: "" });
      return undefined;
    }
    let cancelled = false;
    const stale = () => cancelled || shownId.current !== driveId;
    setHazards({ state: "loading", items: [], message: "" });
    (async () => {
      try {
        const response = await fetch(`/api/drives/${encodeURIComponent(driveId)}/hazards`);
        if (!response.ok) {
          const message = await errorDetail(response, "Could not load hazards");
          if (!stale()) setHazards({ state: "error", items: [], message });
          return;
        }
        const items = await response.json();
        if (!stale()) setHazards({ state: "ready", items, message: "" });
      } catch {
        if (!stale()) setHazards({ state: "error", items: [], message: "Could not reach the server." });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [driveId, status]);

  async function onSubmit(event) {
    event.preventDefault();
    if (submitting.current) return;
    setError("");
    setPollError("");
    showDrive(null);
    if (!video) return setError("Choose a video file.");
    if (!gpx) return setError("Choose a GPX file.");
    const offsetValue = offset.trim() === "" ? 0 : Number(offset);
    if (!Number.isFinite(offsetValue)) return setError("Clock offset must be a number of seconds.");

    const body = new FormData();
    body.append("video", video);
    body.append("gpx", gpx);
    body.append("clock_offset_seconds", String(offsetValue));

    submitting.current = true;
    setBusy(true);
    try {
      const response = await fetch("/api/drives", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((d) => d.msg).join("; ")
          : data.detail;
        setError(detail || `Upload failed (${response.status}).`);
        return;
      }
      setOrigin("created");
      showDrive(data);
    } catch {
      setError("Could not reach the server.");
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }

  async function onStart() {
    if (starts.current || !drive || drive.status === "processing" || drive.video_deleted) return;
    starts.current = true;
    setStarting(true);
    setError("");
    setPollError("");
    const id = drive.id;
    const stale = () => shownId.current !== id; // another drive was opened meanwhile
    try {
      const response = await fetch(`/api/drives/${encodeURIComponent(id)}/process`, { method: "POST" });
      if (stale()) return;
      if (response.status === 409) {
        setError("This drive is already processing.");
        // Pick up the real state so the page follows the run that is already going.
        const current = await fetch(`/api/drives/${encodeURIComponent(id)}`).catch(() => null);
        const data = current?.ok ? await current.json() : null;
        if (data?.video_deleted && !stale()) {
          setError("The raw video was deleted, so this drive cannot be processed again.");
        }
        if (data && !stale()) showDrive(data);
        return;
      }
      if (response.status === 404) {
        setError("Drive not found.");
        return;
      }
      if (!response.ok) {
        const message = await errorDetail(response, "Could not start processing");
        if (!stale()) setError(message);
        return;
      }
      const data = await response.json();
      if (!stale()) showDrive(data);
    } catch {
      if (!stale()) setError("Could not reach the server.");
    } finally {
      starts.current = false;
      setStarting(false);
    }
  }

  async function onDeleteVideo() {
    if (!drive || drive.status === "processing" || drive.video_deleted || starts.current) return;
    const id = drive.id;
    if (deletes.current.has(id)) return;
    deletes.current.add(id);
    setDeletingIds((ids) => [...ids, id]);
    setDeleteError("");
    const stale = () => shownId.current !== id; // another drive was opened meanwhile
    try {
      const response = await fetch(`/api/drives/${encodeURIComponent(id)}/video`, { method: "DELETE" });
      if (stale()) return;
      if (response.status === 404) {
        setDeleteError("Drive not found.");
      } else if (response.status === 409) {
        setDeleteError(await errorDetail(response, "The drive is processing; try again when it finishes"));
        // Pick up the real state (processing, or already deleted).
        const current = await fetch(`/api/drives/${encodeURIComponent(id)}`).catch(() => null);
        const data = current?.ok ? await current.json() : null;
        if (data && !stale()) showDrive(data);
      } else if (!response.ok) {
        setDeleteError(await errorDetail(response, "Could not delete the video"));
      } else {
        const data = await response.json().catch(() => ({}));
        if (stale()) return;
        // The server answers with the updated drive; fall back to the flag if the body is empty.
        showDrive({ ...drive, ...(data && data.id === id ? data : {}), video_deleted: true });
      }
    } catch {
      if (!stale()) setDeleteError("Could not reach the server.");
    } finally {
      deletes.current.delete(id);
      setDeletingIds((ids) => ids.filter((x) => x !== id));
      if (!stale()) setConfirmingDelete(false);
    }
  }

  async function onLoad(event) {
    event.preventDefault();
    if (loads.current) return;
    const id = loadId.trim();
    setError("");
    setPollError("");
    if (!id) return setError("Enter a drive ID.");
    loads.current = true;
    setLoading(true);
    try {
      const response = await fetch(`/api/drives/${encodeURIComponent(id)}`);
      if (response.status === 404) {
        setError("Drive not found.");
        return;
      }
      if (!response.ok) {
        setError(await errorDetail(response, "Could not load the drive"));
        return;
      }
      const data = await response.json();
      setOrigin("loaded");
      showDrive(data);
    } catch {
      setError("Could not reach the server.");
    } finally {
      loads.current = false;
      setLoading(false);
    }
  }

  const processing = drive?.status === "processing";
  const videoDeleted = Boolean(drive?.video_deleted);
  const deleting = deletingIds.includes(drive?.id);

  function showTab(next) {
    setTab(next);
    window.history.replaceState(null, "", next === "map" ? "#map" : window.location.pathname + window.location.search);
  }

  return (
    <DemoContext.Provider value={demo}>
    <main className={`rs-shell${tab === "map" ? " rs-shell--wide" : ""}`}>
      <header className="rs-topbar">
        <div className="rs-brand">
          <h1>RoadScout</h1>
          <p className="rs-tagline">Hello, world. Road hazard scouting starts here.</p>
        </div>
        <nav aria-label="Views" className="rs-tabs">
          <button
            type="button"
            data-testid="tab-drives"
            aria-pressed={tab === "drives"}
            onClick={() => showTab("drives")}
          >
            Upload and results
          </button>
          <button type="button" data-testid="tab-map" aria-pressed={tab === "map"} onClick={() => showTab("map")}>
            Hazard map
          </button>
        </nav>
      </header>
      {demo && (
        <p role="status" data-testid="demo-banner" className="rs-banner">
          {DEMO_BANNER_TEXT}
        </p>
      )}
      <p role="note" data-testid="privacy-note" className="rs-note">
        Photos shown here are privacy-blurred (faces and license plates).{" "}
        {demo
          ? "No raw video is stored on this demo server."
          : "The raw video stays on this machine until you delete it."}
      </p>


      {tab === "map" ? (
        <HazardMap />
      ) : (
        <div className="rs-page">
          {!demo && (
            <div className="rs-card">
          <h2>Upload a drive</h2>
          <form onSubmit={onSubmit} className="rs-form">
            <div className="rs-dropzones">
              <DropField>
                Video file
                <br />
                <input
                  type="file"
                  accept="video/*,.mov,.mp4,.mkv,.avi,.m4v"
                  onChange={(e) => setVideo(e.target.files[0] ?? null)}
                />
              </DropField>
              <DropField>
                GPX file
                <br />
                <input
                  type="file"
                  accept=".gpx,application/gpx+xml,text/xml,application/xml"
                  onChange={(e) => setGpx(e.target.files[0] ?? null)}
                />
              </DropField>
            </div>
            <label className="rs-field-offset">
              Clock offset (seconds)
              <br />
              <input
                type="text"
                inputMode="decimal"
                value={offset}
                onChange={(e) => setOffset(e.target.value)}
              />
              <br />
              <small>
                Added to video time to line it up with the GPX clock. Leave 0 if the clocks agree.
              </small>
            </label>
            <div className="rs-form-actions">
              <button type="submit" disabled={busy}>
                {busy ? "Uploading..." : "Upload drive"}
              </button>
              {busy && <div className="rs-progress" aria-hidden="true" />}
            </div>
          </form>
            </div>
          )}

          <div className="rs-card">
          <h2>Open an existing drive</h2>
          <form onSubmit={onLoad} className="rs-load-form">
            <label>
              Drive ID
              <br />
              <input
                type="text"
                data-testid="load-drive-id"
                value={loadId}
                onChange={(e) => setLoadId(e.target.value)}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Loading..." : "Open drive"}
            </button>
          </form>
          </div>

          {error && (
            <p role="alert" className="rs-alert">
              {error}
            </p>
          )}

          {drive && (
            <section aria-label="Drive status" className="rs-card rs-section-gap">
              <h2>{origin === "created" ? "Drive created" : "Drive loaded"}</h2>
              <dl className="rs-drive-grid">
                <div className="rs-stat">
                  <dt>Drive ID</dt>
                  <dd data-testid="drive-id">{drive.id}</dd>
                </div>
                <div className="rs-stat">
                  <dt>Status</dt>
                  <dd data-testid="drive-status">
                    <span className="rs-state" data-state={drive.status}>
                      {drive.status}
                    </span>
                  </dd>
                </div>
                {drive.stage && drive.status !== "uploaded" && (
                  <div className="rs-stat">
                    <dt>Stage</dt>
                    <dd data-testid="drive-stage">{drive.stage}</dd>
                  </div>
                )}
                <div className="rs-stat">
                  <dt>Video start time</dt>
                  <dd data-testid="drive-started-at">
                    {drive.started_at}
                    {drive.started_at_fell_back && " (not in video metadata; estimated)"}
                  </dd>
                </div>
                <div className="rs-stat">
                  <dt>Clock offset</dt>
                  <dd data-testid="drive-offset">{drive.clock_offset_seconds} s</dd>
                </div>
                <div className="rs-stat">
                  <dt>Raw video</dt>
                  <dd data-testid="drive-video-state">
                    {videoDeleted
                      ? `Deleted${drive.video_deleted_at ? ` at ${drive.video_deleted_at}` : ""}`
                      : "Kept on this machine"}
                  </dd>
                </div>
              </dl>
              <StageSteps drive={drive} />

              {!demo && (
              <div className="rs-drive-actions">
                <button
                  type="button"
                  data-testid="start-processing"
                  onClick={onStart}
                  disabled={starting || processing || videoDeleted}
                >
                  {starting || processing ? "Processing..." : "Start processing"}
                </button>

                {confirmingDelete && !videoDeleted ? (
                  <span data-testid="delete-video-confirm" role="group" aria-label="Confirm deleting the video">
                    <strong>Delete video?</strong>{" "}
                    <button
                      type="button"
                      data-testid="delete-video-yes"
                      onClick={onDeleteVideo}
                      disabled={deleting || processing || starting}
                    >
                      {deleting ? "Deleting..." : "Yes, delete"}
                    </button>{" "}
                    <button
                      type="button"
                      data-testid="delete-video-cancel"
                      onClick={() => setConfirmingDelete(false)}
                      disabled={deleting}
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    data-testid="delete-video"
                    onClick={() => {
                      setDeleteError("");
                      setConfirmingDelete(true);
                    }}
                    disabled={processing || starting || videoDeleted}
                  >
                    {videoDeleted ? "Video deleted" : "Delete video"}
                  </button>
                )}
              </div>
              )}
              {videoDeleted && (
                <p data-testid="video-deleted-hint" className="rs-hint-text">
                  The raw video and unblurred frames were deleted from this machine. The blurred results below
                  are kept, but this drive cannot be processed again.
                </p>
              )}
              {deleteError && (
                <p data-testid="delete-video-error" role="alert" className="rs-alert">
                  {deleteError}
                </p>
              )}

              {pollError && (
                <p data-testid="poll-error" className="rs-warn">
                  {pollError}
                </p>
              )}

              {drive.status === "failed" && (
                <div data-testid="drive-failed" className="rs-failed">
                  <strong>Processing failed</strong>
                  {drive.failed_stage && (
                    <>
                      {" "}
                      at stage <span data-testid="drive-failed-stage">{drive.failed_stage}</span>
                    </>
                  )}
                  <p data-testid="drive-error">{drive.error || "Unknown error."}</p>
                </div>
              )}

              {drive.status === "done" && (
                <>
                  <h2>Verified hazards</h2>
                  <Hazards hazards={hazards} />
                </>
              )}
            </section>
          )}
        </div>
      )}
    </main>
    </DemoContext.Provider>
  );
}
