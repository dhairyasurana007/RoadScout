// Inline SVG icons (no icon library). Every icon is decorative: aria-hidden, no text, so it never changes an
// accessible name or the text a test reads.
const base = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": "true",
  focusable: "false",
};

function Svg({ size, className, children }) {
  return (
    <svg {...base} width={size ?? base.width} height={size ?? base.height} className={className ? `rs-icon ${className}` : "rs-icon"}>
      {children}
    </svg>
  );
}

const PATHS = {
  upload: (
    <>
      <path d="M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 8.5a4 4 0 0 1-.5 9.5H17" />
      <path d="M12 21v-9" />
      <path d="m8.5 15 3.5-3.5 3.5 3.5" />
    </>
  ),
  video: (
    <>
      <rect x="3" y="6" width="13" height="12" rx="2.5" />
      <path d="m16 10.5 5-3v9l-5-3" />
    </>
  ),
  route: (
    <>
      <circle cx="6" cy="18" r="2.2" />
      <circle cx="18" cy="6" r="2.2" />
      <path d="M8.2 18H14a3.5 3.5 0 0 0 0-7h-4a3.5 3.5 0 0 1 0-7h5.8" />
    </>
  ),
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  alert: (
    <>
      <path d="M12 4 2.8 19.5h18.4L12 4Z" />
      <path d="M12 10v4.5" />
      <path d="M12 17.5h.01" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3 5 6v5.5c0 4.3 2.9 7.7 7 9.5 4.1-1.8 7-5.2 7-9.5V6l-7-3Z" />
      <path d="m9 12 2.2 2.2L15.5 10" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
      <circle cx="12" cy="12" r="2.6" />
    </>
  ),
  pin: (
    <>
      <path d="M12 21s6.5-5.6 6.5-11a6.5 6.5 0 0 0-13 0c0 5.4 6.5 11 6.5 11Z" />
      <circle cx="12" cy="10" r="2.3" />
    </>
  ),
  copy: (
    <>
      <rect x="9" y="9" width="11" height="11" rx="2.5" />
      <path d="M5 15V6.5A2.5 2.5 0 0 1 7.5 4H15" />
    </>
  ),
  download: (
    <>
      <path d="M12 4v11" />
      <path d="m7.5 11 4.5 4.5 4.5-4.5" />
      <path d="M5 19.5h14" />
    </>
  ),
  external: (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4 11 13" />
      <path d="M18 14v4.5A1.5 1.5 0 0 1 16.5 20h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10" />
    </>
  ),
  close: (
    <>
      <path d="m6 6 12 12" />
      <path d="M18 6 6 18" />
    </>
  ),
  layers: (
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 12.5 9 5 9-5" />
    </>
  ),
  inbox: (
    <>
      <path d="M3 13.5 6 5h12l3 8.5V19a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5.5Z" />
      <path d="M3 13.5h5l1 2.5h6l1-2.5h5" />
    </>
  ),
  // pipeline stages
  scissors: (
    <>
      <circle cx="6" cy="7" r="2.4" />
      <circle cx="6" cy="17" r="2.4" />
      <path d="M8 8.4 20 16" />
      <path d="M8 15.6 20 8" />
    </>
  ),
  scan: (
    <>
      <path d="M4 8V6a2 2 0 0 1 2-2h2" />
      <path d="M16 4h2a2 2 0 0 1 2 2v2" />
      <path d="M20 16v2a2 2 0 0 1-2 2h-2" />
      <path d="M8 20H6a2 2 0 0 1-2-2v-2" />
      <path d="M8 12h8" />
    </>
  ),
  group: (
    <>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.8" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.8" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.8" />
      <path d="M14.5 17h5M17 14.5v5" />
    </>
  ),
  blur: (
    <>
      <circle cx="12" cy="12" r="3" />
      <circle cx="12" cy="12" r="7" strokeDasharray="2.5 3" />
    </>
  ),
  merge: (
    <>
      <path d="M6 4v5a4 4 0 0 0 4 4h4a4 4 0 0 1 4 4v3" />
      <path d="M18 4v5a4 4 0 0 1-4 4" />
      <path d="m15.5 17.5 2.5 2.5 2.5-2.5" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 11a8 8 0 0 0-14.3-4.5L4 8.5" />
      <path d="M4 4v4.5h4.5" />
      <path d="M4 13a8 8 0 0 0 14.3 4.5L20 15.5" />
      <path d="M20 20v-4.5h-4.5" />
    </>
  ),
  // hazard classes
  pothole: (
    <>
      <path d="M2.5 17.5c2.5 0 3-1.5 5.5-1.5s3 1.5 5.5 1.5 3-1.5 5.5-1.5" />
      <ellipse cx="12" cy="11" rx="6.5" ry="3.2" />
      <path d="M8 11c1.2.9 2.4 1.2 4 1.2s2.8-.3 4-1.2" />
    </>
  ),
  debris: (
    <>
      <path d="m4 18 4.5-9 4 5 2.5-3.5L20 18H4Z" />
      <circle cx="17.5" cy="6.5" r="1.6" />
    </>
  ),
  guardrail: (
    <>
      <path d="M3 9c3-1.6 6 1.6 9 0s6 1.6 9 0" />
      <path d="M3 14c3-1.6 6 1.6 9 0s6 1.6 9 0" />
      <path d="M6 14.5V20M18 14.5V20" />
    </>
  ),
  sign: (
    <>
      <path d="M12 3 21 12l-9 9-9-9 9-9Z" />
      <path d="M12 8v5" />
      <path d="M12 16h.01" />
    </>
  ),
  lanes: (
    <>
      <path d="M8 3 4.5 21" />
      <path d="M16 3l3.5 18" />
      <path d="M12 4v3M12 10.5v3M12 17v3" />
    </>
  ),
};

export function Icon({ name, size, className }) {
  const body = PATHS[name];
  if (!body) return null;
  return (
    <Svg size={size} className={className}>
      {body}
    </Svg>
  );
}

const CLASS_ICON = {
  pothole: "pothole",
  "debris in lane": "debris",
  "damaged guardrail": "guardrail",
  "fallen or damaged sign": "sign",
  "faded lane markings": "lanes",
};

// The icon of a hazard class (a neutral pin for a class we do not know).
export function ClassIcon({ hazardClass, size = 16 }) {
  return <Icon name={CLASS_ICON[hazardClass] ?? "pin"} size={size} className="rs-class-icon" />;
}

// Three bars, the first `level` filled: severity as a meter. Decorative, the label next to it says the level.
export function SeverityMeter({ level }) {
  const n = Math.max(0, Math.min(3, Number(level) || 0));
  return (
    <span className="rs-meter" aria-hidden="true" data-level={n}>
      <i />
      <i />
      <i />
    </span>
  );
}
