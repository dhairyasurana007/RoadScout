import { test, expect } from "@playwright/test";

// UI behaviour of the Delete video button only: every API call is mocked with page.route, so no
// backend, footage or model is used. Responses that must not race are held back with a gate, so the
// tests step through the states instead of depending on timing.
const POLL = "/?pollMs=50";
const ID = "d1";
const DELETED_AT = "2026-10-27T10:00:00+00:00";

// A 1x1 PNG, standing in for a blurred photo.
const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

const drive = (patch = {}) => ({
  id: ID,
  status: "done",
  stage: "done",
  error: null,
  failed_stage: null,
  started_at: "2026-10-20T12:30:00+00:00",
  started_at_source: "video_metadata",
  started_at_fell_back: false,
  clock_offset_seconds: 0,
  video_deleted: false,
  video_deleted_at: null,
  ...patch,
});

const deletedDrive = (patch = {}) => drive({ video_deleted: true, video_deleted_at: DELETED_AT, ...patch });

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
  blur_status: "ok",
  reason: "Deep hole in the lane",
  report_description: "Large pothole in the right lane.",
  best_timestamp: 12.5,
  photo_url: `/api/drives/${ID}/hazards/11/photo`,
  crop_url: null,
  ...patch,
});

function makeGate() {
  let release;
  const promise = new Promise((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

/**
 * Mock the API for drive d1. `state` is what GET /api/drives/d1 returns (calls.hold(x) changes it);
 * `del` is an optional handler for DELETE .../video, `process` one for POST .../process.
 */
async function mockApi(page, { state = drive(), hazards = [hazard()], del, process } = {}) {
  const calls = { get: 0, del: 0, process: 0, state, hold: (next) => (calls.state = next) };
  await page.route(`**/api/drives/${ID}`, (route) => {
    calls.get += 1;
    return route.fulfill({ status: 200, json: calls.state });
  });
  await page.route(`**/api/drives/${ID}/video`, (route) => {
    calls.method = route.request().method();
    if (calls.method !== "DELETE") return route.fulfill({ status: 405, json: {} });
    calls.del += 1;
    if (typeof del === "function") return del(route, calls);
    return route.fulfill({ status: 200, json: deletedDrive() });
  });
  await page.route(`**/api/drives/${ID}/process`, (route) => {
    calls.process += 1;
    if (typeof process === "function") return process(route, calls);
    return route.fulfill({ status: 202, json: drive({ status: "processing", stage: "queued" }) });
  });
  await page.route(`**/api/drives/${ID}/hazards`, (route) => route.fulfill({ status: 200, json: hazards }));
  await page.route("**/api/drives/*/hazards/*/photo*", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  return calls;
}

async function openDrive(page, id = ID) {
  await page.getByTestId("load-drive-id").fill(id);
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText(id);
}

const deleteButton = (page) => page.getByTestId("delete-video");
const confirmBox = (page) => page.getByTestId("delete-video-confirm");
const yesButton = (page) => page.getByTestId("delete-video-yes");
const cancelButton = (page) => page.getByTestId("delete-video-cancel");
const startButton = (page) => page.getByTestId("start-processing");
const errorBox = (page) => page.getByTestId("delete-video-error");

// Resolves once the browser has received the (held) DELETE reply and the page has had two frames to
// act on it, so "nothing changed" can be asserted without a fixed sleep.
async function lateReplyHandled(page, release) {
  const reply = page.waitForResponse((r) => r.request().method() === "DELETE" && r.url().includes("/video"));
  release();
  await reply;
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
}

async function confirmDelete(page) {
  await deleteButton(page).click();
  await yesButton(page).click();
}

test("there is no Delete video button before a drive is shown", async ({ page }) => {
  await mockApi(page);
  await page.goto(POLL);
  await expect(deleteButton(page)).toHaveCount(0);
});

for (const [status, stage] of [
  ["uploaded", null],
  ["done", "done"],
  ["failed", "failed"],
]) {
  test(`a ${status} drive shows an enabled Delete video button and no confirmation`, async ({ page }) => {
    const calls = await mockApi(page, { state: drive({ status, stage }) });
    await page.goto(POLL);
    await openDrive(page);
    await expect(deleteButton(page)).toBeVisible();
    await expect(deleteButton(page)).toBeEnabled();
    await expect(deleteButton(page)).toHaveText("Delete video");
    await expect(confirmBox(page)).toHaveCount(0);
    await expect(page.getByTestId("drive-video-state")).toHaveText("Kept on this machine");
    expect(calls.del).toBe(0);
  });
}

test("Delete video asks for confirmation first and sends nothing yet", async ({ page }) => {
  const calls = await mockApi(page);
  await page.goto(POLL);
  await openDrive(page);
  await deleteButton(page).click();

  await expect(confirmBox(page)).toBeVisible();
  await expect(confirmBox(page)).toContainText("Delete video?");
  await expect(yesButton(page)).toHaveText("Yes, delete");
  await expect(yesButton(page)).toBeEnabled();
  await expect(cancelButton(page)).toHaveText("Cancel");
  await expect(deleteButton(page)).toHaveCount(0);
  expect(calls.del).toBe(0);
});

test("Cancel closes the confirmation, sends nothing and leaves the drive alone", async ({ page }) => {
  const calls = await mockApi(page);
  await page.goto(POLL);
  await openDrive(page);
  await deleteButton(page).click();
  await cancelButton(page).click();

  await expect(confirmBox(page)).toHaveCount(0);
  await expect(deleteButton(page)).toBeEnabled();
  await expect(deleteButton(page)).toHaveText("Delete video");
  await expect(page.getByTestId("drive-video-state")).toHaveText("Kept on this machine");
  await expect(startButton(page)).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(calls.del).toBe(0);

  // It can be opened again after a cancel.
  await deleteButton(page).click();
  await expect(confirmBox(page)).toBeVisible();
});

test("Yes, delete sends DELETE to the video route, shows progress, then Video deleted", async ({ page }) => {
  const gate = makeGate();
  const calls = await mockApi(page, {
    del: async (route) => {
      await gate.promise;
      await route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);

  const request = page.waitForRequest(
    (r) => r.url().endsWith(`/api/drives/${ID}/video`) && r.method() === "DELETE",
  );
  await deleteButton(page).click();
  await yesButton(page).click();
  await request;

  // The request is held: the confirmation shows progress and cannot be repeated or cancelled.
  await expect(yesButton(page)).toHaveText("Deleting...");
  await expect(yesButton(page)).toBeDisabled();
  await expect(cancelButton(page)).toBeDisabled();
  await expect(page.getByTestId("drive-video-state")).toHaveText("Kept on this machine");

  gate.release();
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(deleteButton(page)).toBeDisabled();
  await expect(confirmBox(page)).toHaveCount(0);
  await expect(startButton(page)).toBeDisabled();
  await expect(page.getByTestId("video-deleted-hint")).toContainText("cannot be processed again");
  await expect(page.getByTestId("drive-video-state")).toContainText(`Deleted at ${DELETED_AT}`);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(calls.del).toBe(1);
  expect(calls.method).toBe("DELETE");

  // The blurred results are still there.
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(page.getByTestId("hazard-photo")).toBeVisible();
});

test("a double click on Yes, delete sends one request", async ({ page }) => {
  const gate = makeGate();
  const calls = await mockApi(page, {
    del: async (route) => {
      await gate.promise;
      await route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await deleteButton(page).click();
  await yesButton(page).dblclick();
  await expect(yesButton(page)).toBeDisabled();
  await expect.poll(() => calls.del).toBe(1);
  gate.release();
  await expect(deleteButton(page)).toHaveText("Video deleted");
  expect(calls.del).toBe(1);
});

test("an empty 204 answer still marks the video as deleted", async ({ page }) => {
  await mockApi(page, { del: (route) => route.fulfill({ status: 204 }) });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(deleteButton(page)).toBeDisabled();
  await expect(startButton(page)).toBeDisabled();
  await expect(page.getByTestId("drive-id")).toHaveText(ID);
});

test("the button is disabled while the drive is processing", async ({ page }) => {
  const calls = await mockApi(page, { state: drive({ status: "processing", stage: "detecting" }) });
  await page.goto(POLL);
  await openDrive(page);
  await expect(deleteButton(page)).toBeDisabled();
  await expect(deleteButton(page)).toHaveText("Delete video");
  await expect(startButton(page)).toBeDisabled();
  expect(calls.del).toBe(0);

  // Once the run ends, the button works again.
  calls.hold(drive({ status: "done", stage: "done" }));
  await expect(page.getByTestId("drive-status")).toHaveText("done");
  await expect(deleteButton(page)).toBeEnabled();
});

test("starting a run while the confirmation is open closes it and disables the button", async ({ page }) => {
  const calls = await mockApi(page);
  await page.goto(POLL);
  await openDrive(page);
  await deleteButton(page).click();
  await expect(confirmBox(page)).toBeVisible();

  calls.hold(drive({ status: "processing", stage: "detecting" })); // what the polls return
  await startButton(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(confirmBox(page)).toHaveCount(0);
  await expect(deleteButton(page)).toBeDisabled();
  expect(calls.del).toBe(0);
});

test("a drive whose video is already deleted shows Video deleted and cannot be processed", async ({ page }) => {
  const calls = await mockApi(page, { state: deletedDrive() });
  await page.goto(POLL);
  await openDrive(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(deleteButton(page)).toBeDisabled();
  await expect(startButton(page)).toBeDisabled();
  await expect(startButton(page)).toHaveText("Start processing");
  await expect(page.getByTestId("video-deleted-hint")).toBeVisible();
  await expect(page.getByTestId("drive-video-state")).toContainText("Deleted");
  // Results stay available.
  await expect(page.getByTestId("hazard-item")).toHaveCount(1);
  await expect(page.getByTestId("hazard-photo")).toHaveAttribute("src", `/api/drives/${ID}/hazards/11/photo`);
  expect(calls.del).toBe(0);
  expect(calls.process).toBe(0);
});

test("a failed drive can have its video deleted", async ({ page }) => {
  const calls = await mockApi(page, {
    state: drive({ status: "failed", stage: "failed", failed_stage: "blurring", error: "blurring: boom" }),
    hazards: [],
    del: (route) =>
      route.fulfill({
        status: 200,
        json: deletedDrive({ status: "failed", stage: "failed", failed_stage: "blurring", error: "blurring: boom" }),
      }),
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(page.getByTestId("drive-status")).toHaveText("failed"); // deleting never changes the status
  await expect(page.getByTestId("drive-error")).toHaveText("blurring: boom");
  expect(calls.del).toBe(1);
});

test("404 shows drive not found and keeps the button usable", async ({ page }) => {
  await mockApi(page, { del: (route) => route.fulfill({ status: 404, json: { detail: "Drive not found" } }) });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(errorBox(page)).toHaveText("Drive not found.");
  await expect(confirmBox(page)).toHaveCount(0);
  await expect(deleteButton(page)).toBeEnabled();
  await expect(deleteButton(page)).toHaveText("Delete video");
});

test("409 shows the server message and follows the drive that is processing", async ({ page }) => {
  const calls = await mockApi(page, {
    del: (route, c) => {
      c.hold(drive({ status: "processing", stage: "detecting" })); // the refresh after the 409
      return route.fulfill({ status: 409, json: { detail: "Drive is processing; wait for it to finish" } });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(errorBox(page)).toHaveText("Drive is processing; wait for it to finish");
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(deleteButton(page)).toBeDisabled();
  expect(calls.del).toBe(1);
});

test("a server error shows its status and a retry succeeds", async ({ page }) => {
  let attempt = 0;
  const calls = await mockApi(page, {
    del: (route) => {
      attempt += 1;
      if (attempt === 1) return route.fulfill({ status: 500, contentType: "text/plain", body: "oops" });
      return route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(errorBox(page)).toHaveText("Could not delete the video (500).");
  await expect(deleteButton(page)).toBeEnabled();
  await expect(page.getByTestId("drive-video-state")).toHaveText("Kept on this machine");

  await confirmDelete(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(errorBox(page)).toHaveCount(0);
  expect(calls.del).toBe(2);
});

test("the server's own error text is shown for other failures", async ({ page }) => {
  await mockApi(page, {
    del: (route) => route.fulfill({ status: 500, json: { detail: "Could not delete everything: video.mp4" } }),
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(errorBox(page)).toHaveText("Could not delete everything: video.mp4");
});

test("a network failure shows an error and allows a retry", async ({ page }) => {
  let attempt = 0;
  await mockApi(page, {
    del: (route) => {
      attempt += 1;
      if (attempt === 1) return route.abort();
      return route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(errorBox(page)).toHaveText("Could not reach the server.");
  await expect(deleteButton(page)).toBeEnabled();

  await confirmDelete(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(errorBox(page)).toHaveCount(0);
});

test("a reply for a drive that is no longer shown does not change the panel", async ({ page }) => {
  const gate = makeGate();
  const calls = await mockApi(page, {
    del: async (route) => {
      await gate.promise;
      await route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  await page.route("**/api/drives/d3", (route) => route.fulfill({ status: 200, json: drive({ id: "d3" }) }));
  await page.route("**/api/drives/d3/hazards", (route) => route.fulfill({ status: 200, json: [] }));
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect.poll(() => calls.del).toBe(1); // the reply for d1 is held back

  await page.getByTestId("load-drive-id").fill("d3");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(deleteButton(page)).toHaveText("Delete video");

  await lateReplyHandled(page, gate.release);
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(deleteButton(page)).toHaveText("Delete video");
  await expect(deleteButton(page)).toBeEnabled();
  await expect(page.getByTestId("drive-video-state")).toHaveText("Kept on this machine");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("a delete in flight for one drive does not block deleting another drive", async ({ page }) => {
  const gate = makeGate();
  const calls = await mockApi(page, {
    del: async (route) => {
      await gate.promise;
      await route.fulfill({ status: 200, json: deletedDrive() });
    },
  });
  let d3Deletes = 0;
  await page.route("**/api/drives/d3", (route) => route.fulfill({ status: 200, json: drive({ id: "d3" }) }));
  await page.route("**/api/drives/d3/hazards", (route) => route.fulfill({ status: 200, json: [] }));
  await page.route("**/api/drives/d3/video", (route) => {
    d3Deletes += 1;
    return route.fulfill({ status: 200, json: deletedDrive({ id: "d3" }) });
  });
  await page.goto(POLL);
  await openDrive(page);
  await confirmDelete(page);
  await expect(yesButton(page)).toHaveText("Deleting...");
  await expect.poll(() => calls.del).toBe(1); // d1's reply is held back

  // Another drive is shown while d1 is still deleting: its button works.
  await page.getByTestId("load-drive-id").fill("d3");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await deleteButton(page).click();
  await expect(yesButton(page)).toBeEnabled();
  await expect(yesButton(page)).toHaveText("Yes, delete");
  await yesButton(page).click();
  await expect(deleteButton(page)).toHaveText("Video deleted");
  expect(d3Deletes).toBe(1);
  expect(calls.del).toBe(1);

  // d1's late reply must not touch d3's panel.
  await lateReplyHandled(page, gate.release);
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(page.getByRole("alert")).toHaveCount(0);

  // Back on d1 (the server now says it is deleted).
  calls.hold(deletedDrive());
  await openDrive(page);
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(deleteButton(page)).toBeDisabled();
});

test("Delete video is disabled while a Start processing request is in flight", async ({ page }) => {
  const gate = makeGate();
  const calls = await mockApi(page, {
    process: async (route) => {
      await gate.promise;
      await route.fulfill({ status: 202, json: drive({ status: "processing", stage: "queued" }) });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await expect(deleteButton(page)).toBeEnabled();

  await startButton(page).click();
  await expect.poll(() => calls.process).toBe(1); // the reply is held back
  await expect(deleteButton(page)).toBeDisabled();
  calls.hold(drive({ status: "processing", stage: "detecting" })); // what the polls return from now on
  gate.release();
  await expect(page.getByTestId("drive-status")).toHaveText("processing");
  await expect(deleteButton(page)).toBeDisabled();
  expect(calls.del).toBe(0);
});

test("opening another drive closes an open confirmation", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/drives/d3", (route) => route.fulfill({ status: 200, json: drive({ id: "d3" }) }));
  await page.route("**/api/drives/d3/hazards", (route) => route.fulfill({ status: 200, json: [] }));
  await page.goto(POLL);
  await openDrive(page);
  await deleteButton(page).click();
  await expect(confirmBox(page)).toBeVisible();

  await page.getByTestId("load-drive-id").fill("d3");
  await page.getByRole("button", { name: "Open drive" }).click();
  await expect(page.getByTestId("drive-id")).toHaveText("d3");
  await expect(confirmBox(page)).toHaveCount(0);
  await expect(deleteButton(page)).toBeEnabled();
});

test("processing a drive whose video was deleted behind the page's back explains why it failed", async ({ page }) => {
  const calls = await mockApi(page, {
    process: (route, c) => {
      c.hold(deletedDrive()); // the refresh after the 409 shows the deleted state
      return route.fulfill({
        status: 409,
        json: { detail: "The raw video of this drive was deleted, so it cannot be processed again" },
      });
    },
  });
  await page.goto(POLL);
  await openDrive(page);
  await expect(startButton(page)).toBeEnabled(); // the page does not know yet
  await startButton(page).click();

  await expect(page.getByRole("alert")).toHaveText(
    "The raw video was deleted, so this drive cannot be processed again.",
  );
  await expect(deleteButton(page)).toHaveText("Video deleted");
  await expect(startButton(page)).toBeDisabled();
  expect(calls.process).toBe(1);
});

test("the privacy note stays visible next to the Delete video button", async ({ page }) => {
  await mockApi(page);
  await page.goto(POLL);
  await openDrive(page);
  await expect(page.getByTestId("privacy-note")).toContainText("privacy-blurred");
  await expect(page.getByTestId("privacy-note")).toContainText("raw video stays on this machine");
});
