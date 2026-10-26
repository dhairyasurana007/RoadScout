import { useRef, useState } from "react";

export default function App() {
  const [video, setVideo] = useState(null);
  const [gpx, setGpx] = useState(null);
  const [offset, setOffset] = useState("0");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [drive, setDrive] = useState(null);
  const submitting = useRef(false); // blocks a double click before React re-renders the disabled button

  async function onSubmit(event) {
    event.preventDefault();
    if (submitting.current) return;
    setError("");
    setDrive(null);
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
      setDrive(data);
    } catch {
      setError("Could not reach the server.");
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }

  return (
    <main style={{ fontFamily: "sans-serif", padding: "2rem", maxWidth: 640 }}>
      <h1>RoadScout</h1>
      <p>Hello, world. Road hazard scouting starts here.</p>

      <h2>Upload a drive</h2>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem" }}>
        <label>
          Video file
          <br />
          <input
            type="file"
            accept="video/*,.mov,.mp4,.mkv,.avi,.m4v"
            onChange={(e) => setVideo(e.target.files[0] ?? null)}
          />
        </label>
        <label>
          GPX file
          <br />
          <input
            type="file"
            accept=".gpx,application/gpx+xml,text/xml,application/xml"
            onChange={(e) => setGpx(e.target.files[0] ?? null)}
          />
        </label>
        <label>
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
        <button type="submit" disabled={busy}>
          {busy ? "Uploading..." : "Upload drive"}
        </button>
      </form>

      {error && (
        <p role="alert" style={{ color: "#b00020" }}>
          {error}
        </p>
      )}

      {drive && (
        <section aria-label="Drive status">
          <h2>Drive created</h2>
          <dl>
            <dt>Drive ID</dt>
            <dd data-testid="drive-id">{drive.id}</dd>
            <dt>Status</dt>
            <dd data-testid="drive-status">{drive.status}</dd>
            <dt>Video start time</dt>
            <dd data-testid="drive-started-at">
              {drive.started_at}
              {drive.started_at_fell_back && " (not in video metadata; estimated)"}
            </dd>
            <dt>Clock offset</dt>
            <dd data-testid="drive-offset">{drive.clock_offset_seconds} s</dd>
          </dl>
        </section>
      )}
    </main>
  );
}
