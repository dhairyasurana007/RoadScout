import { test, expect } from "@playwright/test";

// The read-only hosted demo (task 23a). GET /api/demo says whether the server is the demo; then the page shows a
// banner and has no control that would change anything. Every API call is mocked with page.route; any request
// that is not a GET/HEAD to /api is recorded (a demo page must send none).

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);
const BANNER = "Read-only demo: nothing you do here is saved.";

const hazard = (id, patch = {}) => ({
  id,
  hazard_class: "pothole",
  lat: 38.84 + id * 0.01,
  lon: -77.43 + id * 0.01,
  direction: 90,
  severity: 2,
  confidence: 0.8,
  sightings: 2,
  candidate_count: 2,
  first_seen: "2026-10-20T12:30:00+00:00",
  last_seen: "2026-10-22T08:10:00+00:00",
  status: "approved",
  reviewed_at: "2026-10-23T10:00:00+00:00",
  class_edited: false,
  severity_edited: false,
  original_hazard_class: null,
  original_severity: null,
  agency: "VDOT",
  agency_source: "user",
  automatic_agency: "VDOT",
  photo_url: `/api/hazards/${id}/photo`,
  crop_url: null,
  ...patch,
});

const HAZARDS = [
  hazard(1, { status: "approved" }),
  hazard(2, { status: "new", reviewed_at: null }),
  hazard(3, { status: "rejected" }),
  hazard(4, { status: "reported" }),
];

function report(h) {
  const description = `Road hazard: ${h.hazard_class}.`;
  return {
    hazard_id: h.id,
    hazard_status: h.status,
    hazard_class: h.hazard_class,
    agency: h.agency,
    agency_name: "VDOT (full name)",
    agency_source: h.agency_source,
    automatic_agency: h.automatic_agency,
    category: "Pothole repair",
    sub_topic_note: null,
    fields: { problem_type: "Pothole repair", description },
    field_info: {
      problem_type: { label: "Kind of repair or problem", required: true, note: "unverified, check on the live form" },
      description: { label: "Description", required: false, note: "unverified, check on the live form" },
    },
    verified_fields: false,
    verification_note: "Unverified.",
    description,
    location: { lat: h.lat, lon: h.lon, compass: "E", heading: 90, map_url: null },
    photo_url: h.photo_url,
    photo_download_url: `${h.photo_url}?download=1`,
    text_download_url: `/api/hazards/${h.id}/report?format=txt`,
    photo_note: "The form has no photo upload.",
    form_url: "https://my.vdot.virginia.gov/",
    submitted: false,
    note: "Nothing was submitted.",
    agency_note: "Change the agency if this is not a road this agency looks after.",
    text: `RoadScout prefilled report for hazard ${h.id} (NOT SUBMITTED)`,
  };
}

const DRIVE = {
  id: "drive-demo",
  status: "done",
  stage: "done",
  started_at: "2026-10-20T12:30:00+00:00",
  started_at_fell_back: false,
  clock_offset_seconds: 0,
  video_deleted: true,
  video_deleted_at: null,
  error: null,
  failed_stage: null,
};

/** Mock everything the pages call. `demo` is the answer of GET /api/demo (a number is an HTTP error status). */
async function mockApi(page, demo) {
  const ctl = { writes: [], demoCalls: 0 };
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/") && !["GET", "HEAD", "OPTIONS"].includes(request.method())) {
      ctl.writes.push(`${request.method()} ${url.pathname}`);
    }
  });
  await page.route(/openstreetmap\.org/, (route) => route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }));
  await page.route(
    (url) => url.pathname === "/api/demo",
    (route) => {
      ctl.demoCalls += 1;
      if (typeof demo === "number") return route.fulfill({ status: demo, json: { detail: "broken" } });
      return route.fulfill({ status: 200, json: { demo } });
    },
  );
  await page.route(
    (url) => url.pathname === "/api/hazards",
    (route) => route.fulfill({ status: 200, json: HAZARDS.map((h) => ({ ...h })) }),
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+$/.test(url.pathname),
    (route) => route.fulfill({ status: demo === true ? 403 : 200, json: { detail: "This is a read-only demo." } }),
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+\/report$/.test(url.pathname),
    (route) => {
      const id = Number(new URL(route.request().url()).pathname.split("/")[3]);
      return route.fulfill({ status: 200, json: report(HAZARDS.find((h) => h.id === id)) });
    },
  );
  await page.route(/\/api\/hazards\/\d+\/photo/, (route) => route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }));
  await page.route(
    (url) => url.pathname === `/api/drives/${DRIVE.id}`,
    (route) => route.fulfill({ status: 200, json: DRIVE }),
  );
  await page.route(
    (url) => url.pathname === `/api/drives/${DRIVE.id}/hazards`,
    (route) => route.fulfill({ status: 200, json: [] }),
  );
  return ctl;
}

const item = (page, id) => page.locator(`[data-testid="hazard-list-item"][data-hazard-id="${id}"]`);

async function openMap(page) {
  await page.goto("/#map");
  await expect(page.getByTestId("hazard-map-section")).toBeVisible();
  await expect(page.getByTestId("hazard-list-item")).toHaveCount(HAZARDS.length);
}

// ----- demo mode ------------------------------------------------------------------------------------------------

test("demo: the banner is shown on both views", async ({ page }) => {
  await mockApi(page, true);
  await page.goto("/");
  await expect(page.getByTestId("demo-banner")).toHaveText(BANNER);
  await page.getByTestId("tab-map").click();
  await expect(page.getByTestId("demo-banner")).toHaveText(BANNER);
});

test("demo: the upload form is gone, opening an existing drive still works and has no Start or Delete", async ({ page }) => {
  const ctl = await mockApi(page, true);
  await page.goto("/");
  await expect(page.getByTestId("demo-banner")).toBeVisible();
  await expect(page.getByRole("button", { name: "Upload drive" })).toHaveCount(0);
  await expect(page.locator('input[type="file"]')).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Upload a drive" })).toHaveCount(0);
  await page.getByTestId("load-drive-id").fill(DRIVE.id);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText(DRIVE.id);
  await expect(page.getByTestId("start-processing")).toHaveCount(0);
  await expect(page.getByTestId("delete-video")).toHaveCount(0);
  expect(ctl.writes).toEqual([]);
});

test("demo: the hazard list has no Approve, Reject, Reset or Edit, but the report button stays", async ({ page }) => {
  const ctl = await mockApi(page, true);
  await openMap(page);
  await expect(page.getByTestId("demo-banner")).toBeVisible();
  for (const testid of ["hazard-approve", "hazard-reject", "hazard-reset", "hazard-edit"]) {
    await expect(page.getByTestId(testid)).toHaveCount(0);
  }
  await expect(item(page, 1).getByTestId("hazard-report-open")).toBeVisible();
  await expect(item(page, 2).getByTestId("hazard-report-open")).toHaveCount(0);
  expect(ctl.writes).toEqual([]);
});

test("demo: the pin popup has no review controls either", async ({ page }) => {
  await mockApi(page, true);
  await openMap(page);
  await page.locator('[data-testid="hazard-pin"][data-hazard-id="1"]').click();
  await expect(page.locator(".leaflet-popup").getByTestId("hazard-popup")).toBeVisible();
  await expect(page.locator(".leaflet-popup").getByTestId("hazard-approve")).toHaveCount(0);
  await expect(page.locator(".leaflet-popup").getByTestId("hazard-edit")).toHaveCount(0);
});

test("demo: the report panel is read-only (no Mark as reported, agency cannot change)", async ({ page }) => {
  const ctl = await mockApi(page, true);
  await openMap(page);
  await item(page, 1).getByTestId("hazard-report-open").click();
  const panel = page.getByTestId("report-panel");
  await expect(panel).toHaveAttribute("data-phase", "ready");
  await expect(page.getByTestId("report-content")).toBeVisible();
  await expect(page.getByTestId("report-mark-reported")).toHaveCount(0);
  await expect(page.getByTestId("report-agency-auto")).toHaveCount(0);
  await expect(page.getByTestId("report-agency-select")).toBeDisabled();
  await expect(page.getByTestId("report-copy")).toBeEnabled();
  await expect(page.getByTestId("report-download-text")).toBeVisible();
  expect(ctl.writes).toEqual([]);
});

// ----- not a demo -----------------------------------------------------------------------------------------------

test("not demo: no banner, the upload form and all controls are there", async ({ page }) => {
  await mockApi(page, false);
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Upload drive" })).toBeVisible();
  await expect(page.locator('input[type="file"]')).toHaveCount(2);
  await expect(page.getByTestId("privacy-note")).toContainText("The raw video stays on this machine");
  await page.getByTestId("load-drive-id").fill(DRIVE.id);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText(DRIVE.id);
  await expect(page.getByTestId("start-processing")).toBeVisible();
  await expect(page.getByTestId("delete-video")).toBeVisible();
  await expect(page.getByTestId("demo-banner")).toHaveCount(0);
});

test("not demo: the map keeps Approve, Reject, Reset, Edit and the report keeps Mark as reported", async ({ page }) => {
  await mockApi(page, false);
  await openMap(page);
  await expect(page.getByTestId("demo-banner")).toHaveCount(0);
  for (const testid of ["hazard-approve", "hazard-reject", "hazard-reset", "hazard-edit"]) {
    await expect(item(page, 1).getByTestId(testid)).toBeVisible();
  }
  await item(page, 1).getByTestId("hazard-report-open").click();
  await expect(page.getByTestId("report-content")).toBeVisible();
  await expect(page.getByTestId("report-mark-reported")).toBeVisible();
  await expect(page.getByTestId("report-agency-select")).toBeEnabled();
  await expect(page.getByTestId("report-agency-auto")).toBeVisible();
});

test("an /api/demo error means not a demo", async ({ page }) => {
  await mockApi(page, 500);
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Upload drive" })).toBeVisible();
  await expect(page.getByTestId("demo-banner")).toHaveCount(0);
});

test("switching views does not ask /api/demo again", async ({ page }) => {
  const ctl = await mockApi(page, true);
  await openMap(page);
  await expect(page.getByTestId("demo-banner")).toBeVisible();
  const before = ctl.demoCalls; // React StrictMode runs the first effect twice in dev, so compare, not count
  await page.getByTestId("tab-drives").click();
  await page.getByTestId("tab-map").click();
  await expect(page.getByTestId("demo-banner")).toBeVisible();
  expect(ctl.demoCalls).toBe(before);
});
