import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useDemo } from "./demo.jsx";
import { Icon } from "./icons.jsx";

// The prefilled agency report of one approved hazard (task 18). Everything shown comes from
// GET /api/hazards/{id}/report. The panel never submits anything anywhere: the only outbound thing is the
// "Open official form" link, which the user clicks themselves, and "Mark as reported" only records (in RoadScout)
// that the user says they submitted the report on the agency's site.

export const NOTHING_SUBMITTED = "Nothing is submitted automatically. You submit this report yourself.";
const AGENCY_CHOICES = [
  { value: "VDOT", label: "VDOT" },
  { value: "MDOT SHA", label: "MDOT SHA" },
  { value: "none", label: "No agency" },
];
const FOCUSABLE =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

async function errorText(response, fallback) {
  const body = await response.json().catch(() => ({}));
  const detail = Array.isArray(body?.detail) ? body.detail.map((d) => d.msg).join("; ") : body?.detail;
  return typeof detail === "string" && detail ? detail : `${fallback} (${response.status}).`;
}

function legacyCopy(text) {
  const previous = document.activeElement;
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.setAttribute("aria-hidden", "true");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  area.remove();
  if (previous && previous.focus) previous.focus({ preventScroll: true });
  return ok;
}

// navigator.clipboard where it exists and is allowed, else the old execCommand route. Resolves true or false.
export async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the fallback
  }
  return legacyCopy(text);
}

const agencyLabel = (agency) => (agency ? agency : "no agency");

function FieldRow({ name, value, info }) {
  const empty = value === null || value === undefined || value === "";
  return (
    <div className="rs-report-field" data-testid="report-field" data-field-key={name} data-empty={String(empty)}>
      <dt>
        {info?.label ?? name}
        {info?.required && (
          <span className="rs-report-required" title="Required on the form">
            {" "}
            (required)
          </span>
        )}
      </dt>
      <dd>
        <span data-testid="report-field-value">{empty ? "Left to you" : value}</span>
        {info?.note && (
          <span className="rs-hint" data-testid="report-field-note">
            {info.note}
          </span>
        )}
      </dd>
    </div>
  );
}

function ReportPhoto({ report }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [report.photo_url]);
  if (!report.photo_url) {
    return <p data-testid="report-photo-missing">No photo for this hazard.</p>;
  }
  if (failed) return <p data-testid="report-photo-missing">Photo unavailable.</p>;
  return (
    <img
      className="rs-report-photo"
      data-testid="report-photo"
      src={report.photo_url}
      alt={`${report.hazard_class} hazard (blurred photo)`}
      onError={() => setFailed(true)}
    />
  );
}

/**
 * Props: hazardId, ctx ({ pending: Set, submit(id, body, opts) }), onClose().
 * The hazard's PATCH requests go through ctx.submit so the list and the pins stay in step and only one request per
 * hazard runs at a time.
 */
export default function ReportPanel({ hazardId, ctx, onClose }) {
  const demo = useDemo(); // read-only demo: the agency is shown but cannot be changed, no "Mark as reported"
  const [state, setState] = useState({ phase: "loading", report: null, message: "", status: 0 });
  const [reloadKey, setReloadKey] = useState(0);
  const [refreshing, setRefreshing] = useState(true);
  const [loads, setLoads] = useState(0); // replies applied so far (a test hook: aborted requests do not count)
  const [agencyError, setAgencyError] = useState("");
  const [copy, setCopy] = useState({ state: "idle" }); // idle | copying | copied | failed
  const [reportedError, setReportedError] = useState("");
  const [focusTick, setFocusTick] = useState(0); // bumped to run the focus effect when no reload follows an action
  const dialogRef = useRef(null);
  const backdropRef = useRef(null);
  const selectRef = useRef(null);
  const markRef = useRef(null);
  const noteRef = useRef(null);
  const copyingRef = useRef(false);
  const loadsRef = useRef(0);
  // Where keyboard focus goes when an action has finished: { want: "select" | "mark", afterLoads } (see below).
  const refocus = useRef(null);
  const busy = ctx.pending.has(hazardId);
  loadsRef.current = loads;

  // Load (and reload) the report; a newer request makes older replies stale.
  useEffect(() => {
    const controller = new AbortController();
    setRefreshing(true);
    (async () => {
      try {
        const response = await fetch(`/api/hazards/${hazardId}/report`, { signal: controller.signal });
        if (!response.ok) {
          const message = await errorText(response, "Could not load the report");
          if (controller.signal.aborted) return;
          setState({ phase: "error", report: null, message, status: response.status });
          setLoads((n) => n + 1);
          return;
        }
        const report = await response.json();
        if (controller.signal.aborted) return;
        if (!report || typeof report !== "object" || report.hazard_id !== hazardId) throw new Error("unexpected response");
        setState({ phase: "ready", report, message: "", status: 200 });
        setLoads((n) => n + 1);
        setCopy({ state: "idle" });
      } catch (err) {
        if (err?.name === "AbortError" || controller.signal.aborted) return;
        const message = err?.message === "unexpected response" ? "Unexpected response from the server." : "Could not reach the server.";
        setState({ phase: "error", report: null, message, status: 0 });
        setLoads((n) => n + 1);
      } finally {
        if (!controller.signal.aborted) setRefreshing(false);
      }
    })();
    return () => controller.abort();
  }, [hazardId, reloadKey]);

  // Keyboard: focus moves into the dialog on open, Escape closes, Tab stays inside.
  useEffect(() => {
    dialogRef.current?.focus();
  }, []);
  // The rest of the page is inert (no clicks, no focus, hidden from assistive technology) while the dialog is open.
  useEffect(() => {
    const changed = [];
    for (const el of document.body.children) {
      if (el === backdropRef.current || el.hasAttribute("inert")) continue;
      el.setAttribute("inert", "");
      changed.push(el);
    }
    return () => changed.forEach((el) => el.removeAttribute("inert"));
  }, []);
  // A control that is disabled while saving (or removed by the reload) drops keyboard focus to <body>. When the
  // action and the reload that follows it are done, focus goes back to the agency select (or, after "Mark as
  // reported", to that button or its confirmation), and to the dialog itself if none of them exists any more.
  // Nothing is moved when the user has already put focus somewhere else inside the dialog.
  useEffect(() => {
    const pending = refocus.current;
    if (!pending || busy || refreshing || loads < pending.afterLoads) return;
    refocus.current = null;
    const root = dialogRef.current;
    if (!root) return;
    const active = document.activeElement;
    if (active && active !== document.body && active !== root && root.contains(active) && !active.disabled) return;
    const usable = (el) => el && el.isConnected && !el.disabled;
    const target = pending.want === "mark" ? [markRef.current, noteRef.current] : [selectRef.current];
    (target.find(usable) ?? root).focus();
  }, [busy, refreshing, loads, focusTick]);
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") {
        // On a focused agency select Escape belongs to the select (it closes its list); leave it alone.
        if (event.target instanceof HTMLSelectElement) return;
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const root = dialogRef.current;
      if (!root) return;
      const items = [...root.querySelectorAll(FOCUSABLE)].filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (items.length === 0) {
        event.preventDefault();
        root.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (!root.contains(active) || active === root) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Run one PATCH, then reload the report when it succeeded or the report is out of date. `want` says where focus
  // goes afterwards (see the effect above).
  const act = async (body, want, fail, onError) => {
    if (busy || refreshing) return;
    onError("");
    refocus.current = { want, afterLoads: Infinity }; // not decided until the PATCH has answered
    const result = await ctx.submit(hazardId, body, { quiet: true });
    const reload = result.ok || (!result.skipped && (result.status === 404 || result.status === 409));
    if (!result.ok && !result.skipped) onError(result.message || fail);
    if (result.skipped) refocus.current = null;
    else if (reload) {
      refocus.current = { want, afterLoads: loadsRef.current + 1 };
      setReloadKey((k) => k + 1);
    } else {
      refocus.current = { want, afterLoads: 0 };
      setFocusTick((t) => t + 1); // nothing to reload: the controls are enabled again, put focus back
    }
  };
  const changeAgency = (value) => act({ agency: value }, "select", "Could not change the agency.", setAgencyError);

  const useAutomatic = () => changeAgency(null);

  const markReported = () => act({ status: "reported" }, "mark", "Could not mark the hazard as reported.", setReportedError);

  const copyReport = useCallback(async () => {
    const text = state.report?.text;
    if (!text || copyingRef.current) return;
    copyingRef.current = true;
    setCopy({ state: "copying" });
    const ok = await copyText(text);
    copyingRef.current = false;
    setCopy({ state: ok ? "copied" : "failed" });
  }, [state.report]);

  const report = state.report;
  const select = (
    <select
      ref={selectRef}
      data-testid="report-agency-select"
      aria-label="Agency"
      value={report ? (report.agency ?? "none") : "none"}
      disabled={demo || !report || busy || refreshing}
      onChange={(e) => changeAgency(e.target.value)}
    >
      {AGENCY_CHOICES.map((c) => (
        <option key={c.value} value={c.value}>
          {c.label}
        </option>
      ))}
    </select>
  );

  const body = (
    <div ref={backdropRef} className="rs-report-backdrop" data-testid="report-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={dialogRef}
        className="rs-report"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rs-report-title"
        data-testid="report-panel"
        data-hazard-id={String(hazardId)}
        data-phase={state.phase}
        data-loads={loads}
        aria-busy={refreshing || busy}
        tabIndex={-1}
      >
        <div className="rs-report-head">
          <h3 id="rs-report-title" data-testid="report-title">
            Prefilled report for hazard {hazardId}
          </h3>
          <button type="button" data-testid="report-close" aria-label="Close report" onClick={onClose}>
            <Icon name="close" size={16} />
            Close
          </button>
        </div>
        <p className="rs-report-banner" data-testid="report-banner" role="note">
          <Icon name="shield" size={20} />
          <span>{NOTHING_SUBMITTED}</span>
        </p>

        {state.phase === "loading" && <p data-testid="report-loading">Loading report...</p>}

        {state.phase === "error" && (
          <div>
            <p role="alert" data-testid="report-error" data-status={String(state.status)}>
              {state.message}
            </p>
            {state.status === 409 && /possibly fixed/i.test(state.message) && (
              <p className="rs-hint" data-testid="report-fixed-hint">
                Close this panel and approve the hazard again to confirm it is still there. Then the report is available.
              </p>
            )}
            {state.status !== 404 && state.status !== 409 && (
              <button type="button" data-testid="report-retry" onClick={() => setReloadKey((k) => k + 1)} disabled={refreshing}>
                {refreshing ? "Loading..." : "Try again"}
              </button>
            )}
          </div>
        )}

        {report && (
          <div data-testid="report-content" data-agency={report.agency ?? "none"} data-agency-source={report.agency_source} data-submitted={String(report.submitted)}>
            <div className="rs-report-agency">
              <label>
                Agency {select}
              </label>
              <span data-testid="report-agency-source" data-source={report.agency_source}>
                {report.agency_source === "user" ? "Chosen by you" : "Automatic (from the hazard's location)"}
              </span>
              {!demo && report.agency_source === "user" && (
                <button type="button" data-testid="report-agency-auto" disabled={busy || refreshing} onClick={useAutomatic}>
                  Use automatic ({agencyLabel(report.automatic_agency)})
                </button>
              )}
              {(busy || refreshing) && (
                <span className="rs-hint" role="status" data-testid="report-updating">
                  Updating...
                </span>
              )}
            </div>
            {agencyError && (
              <p role="alert" className="rs-review-error" data-testid="report-agency-error">
                {agencyError}
              </p>
            )}

            {report.agency ? (
              <>
                <p data-testid="report-category-line">
                  Category: <strong data-testid="report-category">{report.category}</strong>
                </p>
                {report.sub_topic_note && (
                  <p className="rs-hint" data-testid="report-sub-topic-note">
                    {report.sub_topic_note}
                  </p>
                )}
                {!report.verified_fields && (
                  <p className="rs-report-unverified" role="note" data-testid="report-unverified-notice">
                    Unverified field names, check on the live form. {report.verification_note}
                  </p>
                )}
                <dl className="rs-report-fields" data-testid="report-fields" data-verified={String(report.verified_fields)}>
                  {Object.keys(report.fields).map((key) => (
                    <FieldRow key={key} name={key} value={report.fields[key]} info={report.field_info?.[key]} />
                  ))}
                </dl>
              </>
            ) : (
              <p role="status" data-testid="report-no-agency">
                No agency for this hazard. {report.agency_note}
              </p>
            )}
            {report.agency && (
              <p className="rs-hint" data-testid="report-agency-note">
                {report.agency_note}
              </p>
            )}

            <h4>Description</h4>
            <p className="rs-report-description" data-testid="report-description">
              {report.description}
            </p>

            <h4>Location</h4>
            <p data-testid="report-location">
              {report.location?.lat != null ? `${report.location.lat.toFixed(6)}, ${report.location.lon.toFixed(6)}` : "Location unknown"}
              {report.location?.compass ? ` (heading ${report.location.compass})` : ""}
              {report.location?.map_url && (
                <>
                  {" "}
                  <a data-testid="report-map-link" href={report.location.map_url} target="_blank" rel="noopener noreferrer">
                    Open map
                  </a>
                </>
              )}
            </p>

            <h4>Photo (privacy-blurred)</h4>
            <ReportPhoto report={report} />
            <p className="rs-hint" data-testid="report-photo-note">
              {report.photo_note}
            </p>

            <div className="rs-report-actions" role="group" aria-label="Report actions">
              <button type="button" data-testid="report-copy" disabled={copy.state === "copying" || refreshing || busy} onClick={copyReport}>
                <Icon name="copy" size={16} />
                Copy report
              </button>
              <a data-testid="report-download-text" href={report.text_download_url} download>
                <Icon name="download" size={16} />
                Download report
              </a>
              {report.photo_download_url && (
                <a data-testid="report-download-photo" href={report.photo_download_url} download>
                  <Icon name="download" size={16} />
                  Download photo
                </a>
              )}
              {report.form_url && (
                <a data-testid="report-open-form" href={report.form_url} target="_blank" rel="noopener noreferrer">
                  <Icon name="external" size={16} />
                  Open official form
                </a>
              )}
              {!demo && (
                <button
                  type="button"
                  ref={markRef}
                  data-testid="report-mark-reported"
                  disabled={busy || refreshing || report.hazard_status === "reported"}
                  onClick={markReported}
                >
                  {report.hazard_status === "reported" ? "Marked as reported" : "Mark as reported"}
                </button>
              )}
            </div>
            {!demo && (
              <p className="rs-hint">
                "Mark as reported" only records in RoadScout that you submitted this report yourself. It sends nothing.
              </p>
            )}
            <p role="status" data-testid="report-copy-status" data-state={copy.state}>
              {copy.state === "copied" && "Copied"}
              {copy.state === "failed" && "Could not copy automatically. Select the text below and copy it yourself."}
            </p>
            {reportedError && (
              <p role="alert" className="rs-review-error" data-testid="report-reported-error">
                {reportedError}
              </p>
            )}
            {report.hazard_status === "reported" && (
              <p role="status" data-testid="report-reported-note" ref={noteRef} tabIndex={-1}>
                This hazard is marked as reported.
              </p>
            )}

            <label className="rs-report-textlabel">
              Full report text
              <textarea className="rs-report-text" data-testid="report-text" readOnly rows={8} value={report.text} />
            </label>
          </div>
        )}
      </div>
    </div>
  );
  return createPortal(body, document.body);
}
