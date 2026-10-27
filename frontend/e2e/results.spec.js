import { test, expect } from "@playwright/test";

// UI behaviour only: every API call is mocked with page.route, so no backend, footage or model is used.
// The poll interval is shortened with ?pollMs=50 so the tests are fast.
const POLL = "/?pollMs=50";
const ID = "d1";

const video = { name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("not-a-real-video") };
const gpx = {
  name: "track.gpx",
  mimeType: "application/gpx+xml",
  buffer: Buffer.from('<gpx><trk><trkseg><trkpt lat="38.8" lon="-77.4"/></trkseg></trk></gpx>'),
};

// A 1x1 PNG, standing in for a best-frame photo.
const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

const drive = (patch = {}) => ({
  id: ID,
  status: "uploaded",
  stage: null,
  error: null,
  failed_stage: null,
  started_at: "2026-10-20T12:30:00+00:00",
  started_at_source: "video_metadata",
  started_at_fell_back: false,
  clock_offset_seconds: 0,
  ...patch,
});

const hazard = (patch = {}) => ({
  id: 11,
  candidate_id: 11,
  drive_id: ID,
  hazard_class: "pothole",
  severity: 3,
  confidence: 0.87,
  lat: 38.845213,
  lon: -77.430118,
  geotag_status: "ok",
  reason: "Deep hole in the lane",
  report_description: "Large pothole in the right lane.",
  best_timestamp: 12.5,
  photo_url: `/api/drives/${ID}/hazards/11/photo`,
  crop_url: null,
  ...patch,
});

/**
 * Mock one drive. `states` are the bodies GET /api/drives/d1 returns in turn (the last one repeats);
 * `process` is the response for POST .../process; `hazards` is the body (or {status, json}) of .../hazards.
 */
async function mockDrive(page, { states = [drive()], process, hazards = [] } = {}) {
  // calls.hold(state) pins what GET returns until the next hold(), so a test can step through
  // the stages deterministically instead of racing the poll interval.
  const calls = { get: 0, process: 0, hazards: 0, held: null, hold: (state) => (calls.held = state) };
  await page.route("**/api/drives", (route) => route.fulfill({ status: 201, json: drive() }));
  await page.route(`**/api/drives/${ID}`, (route) => {
    const body = calls.held ?? states[Math.min(calls.get, states.length - 1)];
    calls.get += 1;
    return route.fulfill({ status: 200, json: body });
  });
  await page.route(`**/api/drives/${ID}/process`, async (route) => {
    calls.process += 1;
    if (route.request().method() !== "POST") return route.fulfill({ status: 405, json: {} });
    if (typeof process === "function") return process(route, calls);
    return route.fulfill({ status: 202, json: process ?? drive({ status: "processing", stage: "queued" }) });
  });
  await page.route(`**/api/drives/${ID}/hazards`, (route) => {
    calls.hazards += 1;
    if (Array.isArray(hazards)) return route.fulfill({ status: 200, json: hazards });
    return route.fulfill(hazards);
  });
  await page.route("**/api/drives/*/hazards/*/photo*", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  return calls;
}

async function uploadDrive(page) {
  await page.getByLabel("Video file").setInputFiles(video);
  await page.getByLabel("GPX file").setInputFiles(gpx);
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");
}

const startButton = (page) => page.getByTestId("start-processing");

test("the unblurred, local-only note is always visible", async ({ page }) => {
  await page.goto(POLL);
  await expect(page.getByRole("heading", { name: "RoadScout" })).toBeVisible();
  const note = page.getByTestId("local-only-note");
  await expect(note).toBeVisible();
  await expect(note).toContainText("unblurred");
  await expect(note).toContainText("Local only");
  await expect(note).toContainText("privacy blur");
});

test("a new drive shows an enabled Start processing button and no results yet", async ({ page }) => {
  await mockDrive(page);
  await page.goto(POLL);
  await expect(startButton(page)).toHaveCount(0);
  await uploadDrive(page);
  await expect(startButton(page)).toBeEnabled();
  await expect(startButton(page)).toHaveText("Start processing");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(page.getByTestId("hazards-empty")).toHaveCount(0);
});

test("Start processing posts to the process route and shows the processing state", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "queued" })], // polls keep returning processing
  });
  await page.goto(POLL);
  await uploadDrive(page);
  const request = page.waitForRequest(
    (r) => r.url().endsWith(`/api/drives/${ID}/process`) && r.method() === "POST",
  );
  await startButton(page).click();
  await request;

  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByTestId("drive-stage")).toHaveText("queued");
  await expect(startButton(page)).toBeDisabled();
  await expect(startButton(page)).toHaveText("Processing...");
  expect(calls.process).toBe(1);
});

test("polling follows the stages from processing to done, then lists the hazards", async ({ page }) => {
  const calls = await mockDrive(page, { hazards: [hazard()] });
  await page.goto(POLL);
  await uploadDrive(page);

  // Each stage is held until the test has seen it on the page, so none can be skipped.
  calls.hold(drive({ status: "processing", stage: "preprocessing" }));
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("preprocessing");
  expect(calls.hazards).toBe(0);
  for (const stage of ["detecting", "verifying"]) {
    calls.hold(drive({ status: "processing", stage }));
    await expect(page.getByTestId("drive-stage")).toHaveText(stage);
    await expect(startButton(page)).toBeDisabled();
  }
  calls.hold(drive({ status: "done", stage: "done" }));
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("drive-stage")).toHaveText("done");
  await expect(page.getByTestId("hazard-list")).toBeVisible();
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(startButton(page)).toBeEnabled();
  expect(calls.hazards).toBe(1);
});

test("polling stops once the drive is done", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "detecting" }), drive({ status: "done", stage: "done" })],
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazards-empty")).toBeVisible();

  const after = calls.get;
  await page.waitForTimeout(400); // about eight poll intervals
  expect(calls.get).toBe(after);
  expect(calls.hazards).toBe(1);
});

test("polling stops once the drive has failed", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [
      drive({ status: "processing", stage: "detecting" }),
      drive({ status: "failed", stage: "failed", failed_stage: "detecting", error: "detecting: boom" }),
    ],
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("failed");

  const after = calls.get;
  await page.waitForTimeout(400);
  expect(calls.get).toBe(after);
  expect(calls.hazards).toBe(0);
});

test("polling stops when the page moves on to another drive", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("detecting");

  // A failed upload clears the drive panel, which unmounts the poller.
  await page.route("**/api/drives", (route) =>
    route.fulfill({ status: 400, json: { detail: "File is not a readable video" } }),
  );
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("File is not a readable video");
  await expect(page.getByTestId("drive-status")).toHaveCount(0);

  await page.waitForTimeout(150); // let any in-flight poll settle
  const after = calls.get;
  await page.waitForTimeout(400);
  expect(calls.get).toBe(after);
});

test("without a valid pollMs the default (slow) interval is used", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.goto("/?pollMs=abc");
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("queued");
  await page.waitForTimeout(600);
  expect(calls.get).toBe(0);
});

test("a failed drive shows the error text and the failed stage, and can be retried", async ({ page }) => {
  const failed = drive({
    status: "failed",
    stage: "failed",
    failed_stage: "detecting",
    error: "detecting: provider timed out",
  });
  let runs = 0;
  const calls = await mockDrive(page, {
    states: [failed],
    process: (route) => {
      runs += 1;
      // The first run fails at once; the retry starts a new run.
      return route.fulfill({
        status: 202,
        json: runs === 1 ? failed : drive({ status: "processing", stage: "queued" }),
      });
    },
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();

  await expect(page.getByTestId("drive-status")).toHaveText("failed");
  await expect(page.getByTestId("drive-failed")).toContainText("Processing failed");
  await expect(page.getByTestId("drive-failed-stage")).toHaveText("detecting");
  await expect(page.getByTestId("drive-error")).toHaveText("detecting: provider timed out");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(startButton(page)).toBeEnabled();

  calls.hold(drive({ status: "processing", stage: "detecting" }));
  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByTestId("drive-failed")).toHaveCount(0);
  await expect(page.getByTestId("drive-error")).toHaveCount(0);
  await expect(startButton(page)).toBeDisabled();
  expect(calls.process).toBe(2);
});

test("a failed drive without an error text still shows a failed state", async ({ page }) => {
  await mockDrive(page, {
    process: drive({ status: "failed", stage: "failed", error: null }),
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-failed")).toBeVisible();
  await expect(page.getByTestId("drive-error")).toHaveText("Unknown error.");
});

test("a done drive with no verified hazards shows the empty state", async ({ page }) => {
  await mockDrive(page, { process: drive({ status: "done", stage: "done" }), hazards: [] });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("hazards-empty")).toHaveText("No verified hazards");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(page.getByTestId("hazard-item")).toHaveCount(0);
});

test("each hazard shows its photo, class, severity, confidence and location", async ({ page }) => {
  await mockDrive(page, {
    process: drive({ status: "done", stage: "done" }),
    hazards: [
      hazard(),
      hazard({
        id: 12,
        candidate_id: 12,
        hazard_class: "debris in lane",
        severity: 2,
        confidence: 0.5,
        lat: 38.8,
        lon: -77.4,
        photo_url: `/api/drives/${ID}/hazards/12/photo`,
      }),
      hazard({
        id: 13,
        candidate_id: 13,
        hazard_class: "faded lane markings",
        severity: 1,
        confidence: 0.64,
        lat: null,
        lon: null,
        geotag_status: "out_of_range",
        photo_url: `/api/drives/${ID}/hazards/13/photo`,
      }),
    ],
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();

  const items = page.getByTestId("hazard-item");
  await expect(items).toHaveCount(3);
  await expect(page.getByTestId("hazard-count")).toHaveText("3 verified hazards");

  const first = items.nth(0);
  await expect(first).toHaveAttribute("data-candidate-id", "11");
  await expect(first.getByTestId("hazard-class")).toHaveText("pothole");
  await expect(first.getByTestId("hazard-severity")).toHaveText("3 (High)");
  await expect(first.getByTestId("hazard-confidence")).toHaveText("87%");
  await expect(first.getByTestId("hazard-location")).toHaveText("38.84521, -77.43012");
  await expect(first.getByTestId("hazard-description")).toHaveText("Large pothole in the right lane.");
  const photo = first.getByTestId("hazard-photo");
  await expect(photo).toHaveAttribute("src", `/api/drives/${ID}/hazards/11/photo`);
  await expect(photo).toHaveAttribute("alt", /pothole/);
  await expect
    .poll(() => photo.evaluate((img) => img.complete && img.naturalWidth > 0))
    .toBe(true);

  const second = items.nth(1);
  await expect(second.getByTestId("hazard-class")).toHaveText("debris in lane");
  await expect(second.getByTestId("hazard-severity")).toHaveText("2 (Medium)");
  await expect(second.getByTestId("hazard-confidence")).toHaveText("50%");
  await expect(second.getByTestId("hazard-location")).toHaveText("38.80000, -77.40000");

  const third = items.nth(2);
  await expect(third.getByTestId("hazard-severity")).toHaveText("1 (Low)");
  await expect(third.getByTestId("hazard-location")).toHaveText("no location");
  await expect(third.getByTestId("hazard-photo")).toHaveAttribute("alt", /no location/);
});

test("a single hazard uses the singular count and a hazard at 0,0 is still located", async ({ page }) => {
  await mockDrive(page, {
    process: drive({ status: "done", stage: "done" }),
    hazards: [hazard({ lat: 0, lon: 0 })],
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("hazard-count")).toHaveText("1 verified hazard");
  await expect(page.getByTestId("hazard-location")).toHaveText("0.00000, 0.00000");
});

test("a photo that fails to load shows a placeholder instead of a broken image", async ({ page }) => {
  await mockDrive(page, { process: drive({ status: "done", stage: "done" }), hazards: [hazard()] });
  await page.route("**/api/drives/*/hazards/*/photo*", (route) =>
    route.fulfill({ status: 404, json: { detail: "Photo not found" } }),
  );
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("hazard-photo-missing")).toHaveText("Photo unavailable");
  await expect(page.getByTestId("hazard-class")).toHaveText("pothole");
});

test("an error while loading hazards is shown", async ({ page }) => {
  await mockDrive(page, {
    process: drive({ status: "done", stage: "done" }),
    hazards: { status: 500, json: { detail: "Could not read results" } },
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("hazards-error")).toHaveText("Could not read results");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
});

test("the button is disabled while starting and a double click sends one request", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "queued" })],
    process: async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 600));
      await route.fulfill({ status: 202, json: drive({ status: "processing", stage: "queued" }) });
    },
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).dblclick();
  await expect(startButton(page)).toBeDisabled();
  await expect(startButton(page)).toHaveText("Processing...");
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await page.waitForTimeout(200);
  expect(calls.process).toBe(1);
  await expect(startButton(page)).toBeDisabled(); // still processing
});

test("409 already processing is reported and the page follows the running job", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "detecting" })], // the refresh after the 409
    process: (route) =>
      route.fulfill({ status: 409, json: { detail: "Drive is already processing" } }),
    hazards: [hazard()],
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();

  await expect(page.getByRole("alert")).toHaveText("This drive is already processing.");
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByTestId("drive-stage")).toHaveText("detecting");
  calls.hold(drive({ status: "done", stage: "done" }));
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
});

test("404 when starting shows drive not found and re-enables the button", async ({ page }) => {
  await mockDrive(page, {
    process: (route) => route.fulfill({ status: 404, json: { detail: "Drive not found" } }),
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByRole("alert")).toHaveText("Drive not found.");
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");
  await expect(startButton(page)).toBeEnabled();
});

test("a server error when starting shows its message", async ({ page }) => {
  await mockDrive(page, {
    process: (route) => route.fulfill({ status: 500, contentType: "text/plain", body: "oops" }),
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByRole("alert")).toHaveText("Could not start processing (500).");
  await expect(startButton(page)).toBeEnabled();
});

test("a network failure when starting shows an error and allows a retry", async ({ page }) => {
  let attempt = 0;
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "detecting" })],
    process: (route) => {
      attempt += 1;
      if (attempt === 1) return route.abort();
      return route.fulfill({ status: 202, json: drive({ status: "processing", stage: "queued" }) });
    },
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByRole("alert")).toHaveText("Could not reach the server.");
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");
  await expect(startButton(page)).toBeEnabled();

  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(calls.process).toBe(2);
});

test("a failed poll is shown, polling continues and recovers", async ({ page }) => {
  let polls = 0;
  await mockDrive(page, { process: drive({ status: "processing", stage: "queued" }), hazards: [] });
  await page.route(`**/api/drives/${ID}`, (route) => {
    polls += 1;
    if (polls <= 2) return route.abort();
    return route.fulfill({ status: 200, json: drive({ status: "done", stage: "done" }) });
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("poll-error")).toContainText("Could not reach the server");
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("poll-error")).toHaveCount(0);
  await expect(page.getByTestId("hazards-empty")).toBeVisible();
});

test("processing a done drive again clears the old hazards until the new run finishes", async ({ page }) => {
  let runs = 0;
  const calls = await mockDrive(page, {
    states: [drive({ status: "done", stage: "done" })],
    process: drive({ status: "processing", stage: "queued" }),
  });
  await page.route(`**/api/drives/${ID}/hazards`, (route) => {
    runs += 1;
    return route.fulfill({
      status: 200,
      json: runs === 1 ? [hazard()] : [hazard({ id: 21, candidate_id: 21, hazard_class: "damaged guardrail" })],
    });
  });
  await page.goto(POLL);
  await page.getByTestId("load-drive-id").fill(ID);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("hazard-class")).toHaveText("pothole");

  calls.hold(drive({ status: "processing", stage: "detecting" })); // the polls after the click
  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(page.getByTestId("hazard-item")).toHaveCount(0);
  calls.hold(drive({ status: "done", stage: "done" }));
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazard-class")).toHaveText("damaged guardrail");
});

test("an existing drive can be loaded by id and shows its results", async ({ page }) => {
  await mockDrive(page, {
    states: [drive({ status: "done", stage: "done", clock_offset_seconds: -1.5 })],
    hazards: [hazard()],
  });
  await page.goto(POLL);
  await page.getByTestId("load-drive-id").fill(`  ${ID}  `);
  await page.getByRole("button", { name: "Open drive" }).click();

  await expect(page.getByRole("heading", { name: "Drive loaded" })).toBeVisible();
  await expect(page.getByTestId("drive-id")).toHaveText(ID);
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("drive-offset")).toHaveText("-1.5 s");
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("loading a processing drive starts polling", async ({ page }) => {
  const calls = await mockDrive(page, {
    states: [drive({ status: "processing", stage: "detecting" })],
  });
  await page.goto(POLL);
  await page.getByTestId("load-drive-id").fill(ID);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(startButton(page)).toBeDisabled();
  calls.hold(drive({ status: "done", stage: "done" }));
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazards-empty")).toBeVisible();
});

test("loading an unknown drive id shows an error, and an empty id sends nothing", async ({ page }) => {
  let requests = 0;
  await page.route("**/api/drives/**", (route) => {
    requests += 1;
    return route.fulfill({ status: 404, json: { detail: "Drive not found" } });
  });
  await page.goto(POLL);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Enter a drive ID.");
  expect(requests).toBe(0);

  await page.getByTestId("load-drive-id").fill("nope");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Drive not found.");
  await expect(page.getByTestId("drive-status")).toHaveCount(0);
});

test("a network failure while loading a drive shows an error", async ({ page }) => {
  await page.route("**/api/drives/**", (route) => route.abort());
  await page.goto(POLL);
  await page.getByTestId("load-drive-id").fill(ID);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Could not reach the server.");
});

test("a non-OK poll shows a retry message, keeps polling and recovers", async ({ page }) => {
  let polls = 0;
  await mockDrive(page, { process: drive({ status: "processing", stage: "queued" }), hazards: [] });
  await page.route(`**/api/drives/${ID}`, (route) => {
    polls += 1;
    if (polls <= 2) return route.fulfill({ status: 500, json: { detail: "boom" } });
    return route.fulfill({ status: 200, json: drive({ status: "done", stage: "done" }) });
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("poll-error")).toHaveText("Could not refresh the status (500). Retrying...");
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("poll-error")).toHaveCount(0);
});

test("a 404 while polling shows drive not found and stops polling", async ({ page }) => {
  let polls = 0;
  await mockDrive(page, { process: drive({ status: "processing", stage: "queued" }) });
  await page.route(`**/api/drives/${ID}`, (route) => {
    polls += 1;
    return route.fulfill({ status: 404, json: { detail: "Drive not found" } });
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("poll-error")).toHaveText("Drive not found.");

  await page.waitForTimeout(150); // let any in-flight poll settle
  const after = polls;
  expect(after).toBe(1);
  await page.waitForTimeout(400);
  expect(polls).toBe(after);
});

test("the hazards loading state is shown until the list arrives", async ({ page }) => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  await mockDrive(page, { process: drive({ status: "done", stage: "done" }) });
  await page.route(`**/api/drives/${ID}/hazards`, async (route) => {
    await gate;
    await route.fulfill({ status: 200, json: [hazard()] });
  });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();

  await expect(page.getByTestId("hazards-loading")).toHaveText("Loading hazards...");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(page.getByTestId("hazards-empty")).toHaveCount(0);
  release();
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(page.getByTestId("hazards-loading")).toHaveCount(0);
});

test("a network failure while loading hazards is shown", async ({ page }) => {
  await mockDrive(page, { process: drive({ status: "done", stage: "done" }) });
  await page.route(`**/api/drives/${ID}/hazards`, (route) => route.abort());
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("hazards-error")).toHaveText("Could not reach the server.");
  await expect(page.getByTestId("hazard-list")).toHaveCount(0);
  await expect(page.getByTestId("hazards-empty")).toHaveCount(0);
});

for (const value of ["0", "5", "9", "60001", "1e9", "-50", ""]) {
  test(`pollMs=${JSON.stringify(value)} is out of range and falls back to the default interval`, async ({ page }) => {
    const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
    await page.goto(`/?pollMs=${value}`);
    await uploadDrive(page);
    await startButton(page).click();
    await expect(page.getByTestId("drive-stage")).toHaveText("queued");
    await page.waitForTimeout(500); // well under the 1500 ms default
    expect(calls.get).toBe(0);
  });
}

test("pollMs=10 (the lower bound) is accepted and polls quickly", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.goto("/?pollMs=10");
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("detecting");
  await expect.poll(() => calls.get).toBeGreaterThan(3);
});

test("pollMs=60000 (the upper bound) is accepted, so no poll happens soon", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.goto("/?pollMs=60000");
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("queued");
  await page.waitForTimeout(500);
  expect(calls.get).toBe(0);
});

test("polling stops when a second successful upload replaces the drive", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("detecting");

  // The second upload creates another (not yet processing) drive; the first drive's poller must be gone.
  await page.route("**/api/drives", (route) => route.fulfill({ status: 201, json: drive({ id: "d2" }) }));
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d2");
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");

  await page.waitForTimeout(150); // let any in-flight poll settle
  const after = calls.get;
  await page.waitForTimeout(400);
  expect(calls.get).toBe(after);
});

test("polling stops when another drive is opened by id", async ({ page }) => {
  const calls = await mockDrive(page, { states: [drive({ status: "processing", stage: "detecting" })] });
  await page.route("**/api/drives/d3", (route) =>
    route.fulfill({ status: 200, json: drive({ id: "d3", status: "done", stage: "done" }) }),
  );
  await page.route("**/api/drives/d3/hazards", (route) => route.fulfill({ status: 200, json: [] }));
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click();
  await expect(page.getByTestId("drive-stage")).toHaveText("detecting");

  await page.getByTestId("load-drive-id").fill("d3");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(page.getByTestId("hazards-empty")).toBeVisible();

  await page.waitForTimeout(150);
  const after = calls.get;
  await page.waitForTimeout(400);
  expect(calls.get).toBe(after);
});

test("a slow process reply does not overwrite a drive opened meanwhile", async ({ page }) => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const calls = await mockDrive(page, {
    process: async (route) => {
      await gate;
      await route.fulfill({ status: 202, json: drive({ status: "processing", stage: "queued" }) });
    },
  });
  await page.route("**/api/drives/d3", (route) =>
    route.fulfill({ status: 200, json: drive({ id: "d3", status: "done", stage: "done" }) }),
  );
  await page.route("**/api/drives/d3/hazards", (route) => route.fulfill({ status: 200, json: [hazard()] }));
  await page.goto(POLL);
  await uploadDrive(page);
  await startButton(page).click(); // the reply for d1 is held back
  await expect.poll(() => calls.process).toBe(1);

  await page.getByTestId("load-drive-id").fill("d3");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);

  release();
  await page.waitForTimeout(300); // time for the late reply (and any poll of d1) to land
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(calls.get).toBe(0); // d1 was never polled: its processing state was never shown
});
