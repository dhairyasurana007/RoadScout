import { test, expect } from "@playwright/test";

// UI behaviour of the review actions (task 17). Every API call is mocked with page.route (the PATCH requests are
// recorded so their bodies can be asserted), and so are the OpenStreetMap tiles. Holds use hand-opened gates, never
// fixed sleeps.

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const CLASSES = ["pothole", "debris in lane", "damaged guardrail", "fallen or damaged sign", "faded lane markings"];

const hazard = (id, patch = {}) => ({
  id,
  hazard_class: "pothole",
  lat: 38.84 + id * 0.01,
  lon: -77.43 + id * 0.015,
  direction: 90,
  severity: 2,
  confidence: 0.8,
  sightings: 1,
  candidate_count: 1,
  first_seen: "2026-10-20T12:30:00+00:00",
  last_seen: "2026-10-20T12:30:00+00:00",
  status: "new",
  reviewed_at: null,
  class_edited: false,
  severity_edited: false,
  original_hazard_class: null,
  original_severity: null,
  photo_url: `/api/hazards/${id}/photo`,
  crop_url: null,
  ...patch,
});

const fresh = () => [
  hazard(1, { hazard_class: "pothole", severity: 2, status: "new" }),
  hazard(2, { hazard_class: "debris in lane", severity: 1, status: "approved" }),
  hazard(3, { hazard_class: "damaged guardrail", severity: 3, status: "rejected" }),
  hazard(4, { hazard_class: "faded lane markings", severity: 2, status: "reported" }),
];

function gate() {
  let open;
  const promise = new Promise((resolve) => (open = resolve));
  return { promise, open };
}

/**
 * Mock the hazards API. ctl.items is the server state. PATCH requests are recorded in ctl.patches as
 * { id, body } and applied like the real server does unless ctl.patchRespond (async (route, id, body) => ...) takes over.
 * ctl.queries holds the query string of every GET /api/hazards.
 */
async function mockApi(page, items = fresh()) {
  const ctl = { items, patches: [], queries: [], patchRespond: null, getRespond: null };
  await page.route(/openstreetmap\.org/, (route) => route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }));
  await page.route(
    (url) => url.pathname === "/api/hazards",
    async (route) => {
      const query = new URL(route.request().url()).search;
      ctl.queries.push(query);
      if (ctl.getRespond) return ctl.getRespond(route, query);
      const status = new URLSearchParams(query).get("status");
      return route.fulfill({ status: 200, json: status ? ctl.items.filter((h) => h.status === status) : ctl.items });
    },
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+$/.test(url.pathname),
    async (route) => {
      const request = route.request();
      const id = Number(new URL(request.url()).pathname.split("/").pop());
      if (request.method() !== "PATCH") return route.fulfill({ status: 405, json: { detail: "Method Not Allowed" } });
      const body = request.postDataJSON();
      ctl.patches.push({ id, body, contentType: request.headers()["content-type"] });
      if (ctl.patchRespond) return ctl.patchRespond(route, id, body);
      const target = ctl.items.find((h) => h.id === id);
      if (!target) return route.fulfill({ status: 404, json: { detail: "Hazard not found" } });
      return route.fulfill({ status: 200, json: applyPatch(target, body) });
    },
  );
  await page.route(/^.*\/api\/hazards\/\d+\/photo$/, (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  return ctl;
}

/** What the real server does with a review body (mutates and returns the stored hazard). */
function applyPatch(h, body) {
  if (body.status !== undefined) {
    h.status = body.status;
    h.reviewed_at = body.status === "new" ? null : "2026-10-27T10:00:00+00:00";
  }
  if (body.hazard_class !== undefined) {
    h.original_hazard_class = h.original_hazard_class ?? h.hazard_class;
    h.hazard_class = body.hazard_class;
    h.class_edited = h.hazard_class !== h.original_hazard_class;
    if (!h.class_edited) h.original_hazard_class = null;
  }
  if (body.severity !== undefined) {
    h.original_severity = h.original_severity ?? h.severity;
    h.severity = body.severity;
    h.severity_edited = h.severity !== h.original_severity;
    if (!h.severity_edited) h.original_severity = null;
  }
  return { ...h };
}

const item = (page, id) => page.locator(`[data-testid="hazard-list-item"][data-hazard-id="${id}"]`);
const items = (page) => page.getByTestId("hazard-list-item");
const pin = (page, id) => page.locator(`[data-testid="hazard-pin"][data-hazard-id="${id}"]`);
const pins = (page) => page.getByTestId("hazard-pin");

/**
 * Is the element's box fully inside the map container's box? The page viewport is not used: an open popup makes
 * Leaflet pan the map so the popup fits, which can leave the pin near the bottom of the map, below the page fold
 * at a small viewport.
 */
async function insideMap(page, locator) {
  const outer = await page.getByTestId("hazard-map").boundingBox();
  const inner = await locator.boundingBox();
  if (!outer || !inner) return false;
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}
const popup = (page) => page.locator(".leaflet-popup").getByTestId("hazard-popup");

async function openMap(page, ctl) {
  await page.goto("/#map");
  await expect(page.getByTestId("hazard-map-section")).toBeVisible();
  await expect(items(page)).toHaveCount(ctl.items.length);
  await expect(pins(page)).toHaveCount(ctl.items.length);
}

// ----- controls ------------------------------------------------------------------------------------------------

test("every list item has Approve, Reject, Reset and Edit buttons with labels that name the hazard", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  for (const id of [1, 2, 3, 4]) {
    const row = item(page, id);
    await expect(row.getByTestId("hazard-approve")).toHaveAccessibleName(`Approve hazard ${id}`);
    await expect(row.getByTestId("hazard-reject")).toHaveAccessibleName(`Reject hazard ${id}`);
    await expect(row.getByTestId("hazard-reset")).toHaveAccessibleName(`Reset hazard ${id} to new`);
    await expect(row.getByTestId("hazard-edit")).toHaveAccessibleName(`Edit class or severity of hazard ${id}`);
  }
  await expect(page.getByRole("button", { name: "Approve hazard 3" })).toHaveCount(1);
});

test("the button for the current status is disabled, the others are enabled", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const state = async (id) => ({
    approve: await item(page, id).getByTestId("hazard-approve").isDisabled(),
    reject: await item(page, id).getByTestId("hazard-reject").isDisabled(),
    reset: await item(page, id).getByTestId("hazard-reset").isDisabled(),
  });
  expect(await state(1)).toEqual({ approve: false, reject: false, reset: true }); // new
  expect(await state(2)).toEqual({ approve: true, reject: false, reset: false }); // approved
  expect(await state(3)).toEqual({ approve: false, reject: true, reset: false }); // rejected
  expect(await state(4)).toEqual({ approve: false, reject: false, reset: false }); // reported: all three allowed
});

test("the review controls are outside the item's select button (no nested interactive elements)", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  expect(await item(page, 1).locator("button.rs-list-button button").count()).toBe(0);
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  // pressing a review button does not select the hazard
  await expect(item(page, 1)).toHaveAttribute("data-selected", "false");
});

// ----- approve, reject, reset ----------------------------------------------------------------------------------

test("Approve sends PATCH {status: approved}, shows the badge and recolours the pin at once", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(pin(page, 1)).toHaveAttribute("data-status", "new");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveAttribute("data-status", "new");

  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveAttribute("data-status", "approved");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "approved");
  expect(ctl.patches).toEqual([{ id: 1, body: { status: "approved" }, contentType: "application/json" }]);
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeDisabled();
  await expect(item(page, 1).getByTestId("hazard-reset")).toBeEnabled();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveCount(0);
  // the other hazards are untouched
  await expect(pin(page, 2)).toHaveAttribute("data-status", "approved");
  await expect(pin(page, 3)).toHaveAttribute("data-status", "rejected");
});

test("Reject, then Reset, then Approve walk one hazard through the statuses", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-reject").click();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Rejected");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "rejected");
  await row.getByTestId("hazard-reset").click();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("New");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "new");
  await row.getByTestId("hazard-approve").click();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Approved");
  expect(ctl.patches.map((p) => p.body)).toEqual([{ status: "rejected" }, { status: "new" }, { status: "approved" }]);
  expect(ctl.patches.every((p) => p.id === 1)).toBe(true);
});

test("a reported hazard can be reset to new from the review buttons", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await item(page, 4).getByTestId("hazard-reset").click();
  await expect(item(page, 4).getByTestId("hazard-list-status")).toHaveText("New");
  expect(ctl.patches).toHaveLength(1);
  expect(ctl.patches[0].body).toEqual({ status: "new" });
});

test("a review keeps the selection, the zoom and the count, and does not reload the list", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await item(page, 2).locator("button.rs-list-button").click();
  await expect(popup(page)).toHaveAttribute("data-hazard-id", "2");
  const zoom = await page.getByTestId("hazard-map").getAttribute("data-map-zoom");
  const gets = ctl.queries.length;

  await item(page, 2).getByTestId("hazard-reject").click();
  await expect(item(page, 2).getByTestId("hazard-list-status")).toHaveText("Rejected");
  await expect(item(page, 2)).toHaveAttribute("data-selected", "true");
  await expect(pin(page, 2)).toHaveAttribute("data-selected", "true");
  await expect(popup(page)).toHaveAttribute("data-hazard-id", "2"); // the popup stayed open
  await expect(page.getByTestId("hazard-map-count")).toHaveText("4 hazards");
  await expect(page.getByTestId("hazard-map")).toHaveAttribute("data-map-zoom", zoom); // no re-fit, no zoom change
  await expect.poll(() => insideMap(page, pin(page, 2))).toBe(true); // the pin is still on the map
  expect(ctl.queries.length).toBe(gets);
});

// ----- edit class and severity ---------------------------------------------------------------------------------

test("Edit opens a form with the five classes and severities 1 to 3, preselected, and Save disabled until something changes", async ({
  page,
}) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await expect(row.getByTestId("hazard-edit-form")).toHaveCount(0);
  await row.getByTestId("hazard-edit").click();
  await expect(row.getByTestId("hazard-edit-form")).toBeVisible();
  await expect(row.getByTestId("hazard-edit")).toHaveCount(0);
  const classSelect = row.getByTestId("hazard-edit-class");
  const severitySelect = row.getByTestId("hazard-edit-severity");
  expect(await classSelect.locator("option").evaluateAll((o) => o.map((x) => x.value))).toEqual(CLASSES);
  expect(await severitySelect.locator("option").evaluateAll((o) => o.map((x) => x.value))).toEqual(["1", "2", "3"]);
  await expect(classSelect).toHaveValue("pothole");
  await expect(severitySelect).toHaveValue("2");
  await expect(classSelect).toBeFocused();
  await expect(classSelect).toHaveAccessibleName("Class of hazard 1");
  await expect(severitySelect).toHaveAccessibleName("Severity of hazard 1");
  await expect(row.getByTestId("hazard-save")).toBeDisabled();
  await expect(row.getByTestId("hazard-edit-hint")).toBeVisible();

  await severitySelect.selectOption("3");
  await expect(row.getByTestId("hazard-save")).toBeEnabled();
  await expect(row.getByTestId("hazard-edit-hint")).toHaveCount(0);
  await severitySelect.selectOption("2"); // back to the original: nothing to save again
  await expect(row.getByTestId("hazard-save")).toBeDisabled();
  expect(ctl.patches).toEqual([]);
});

test("saving a new class sends only hazard_class, shows the edited marker and closes the form", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("debris in lane");
  await row.getByTestId("hazard-save").click();

  await expect(row.getByTestId("hazard-list-class")).toHaveText("debris in lane");
  await expect(row.getByTestId("hazard-class-edited")).toHaveText("edited");
  await expect(row.getByTestId("hazard-class-edited")).toHaveAttribute("title", "Edited (was pothole)");
  await expect(row.getByTestId("hazard-severity-edited")).toHaveCount(0);
  await expect(row.getByTestId("hazard-edit-form")).toHaveCount(0);
  await expect(row.getByTestId("hazard-edit")).toBeFocused(); // focus returns to the Edit button
  expect(ctl.patches.map((p) => p.body)).toEqual([{ hazard_class: "debris in lane" }]);
  await expect(row.getByTestId("hazard-list-status")).toHaveText("New"); // an edit is not a review decision
});

test("saving a new severity sends only severity and marks it edited", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-severity").selectOption("3");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-list-severity")).toContainText("3 (High)");
  await expect(row.getByTestId("hazard-severity-edited")).toHaveAttribute("title", "Edited (was 2)");
  await expect(row.getByTestId("hazard-class-edited")).toHaveCount(0);
  await expect(pin(page, 1)).toHaveText("3"); // the pin shows the new severity
  expect(ctl.patches.map((p) => p.body)).toEqual([{ severity: 3 }]);
});

test("changing both sends both in one request", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 3);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("fallen or damaged sign");
  await row.getByTestId("hazard-edit-severity").selectOption("1");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-class-edited")).toBeVisible();
  await expect(row.getByTestId("hazard-severity-edited")).toBeVisible();
  expect(ctl.patches).toHaveLength(1);
  expect(ctl.patches[0].body).toEqual({ hazard_class: "fallen or damaged sign", severity: 1 });
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Rejected"); // status untouched
});

test("Cancel closes the form without a request and the next Edit starts from the real values", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("damaged guardrail");
  await row.getByTestId("hazard-edit-severity").selectOption("1");
  await row.getByTestId("hazard-cancel").click();
  await expect(row.getByTestId("hazard-edit-form")).toHaveCount(0);
  await expect(row.getByTestId("hazard-list-class")).toHaveText("pothole");
  await expect(row.getByTestId("hazard-edit")).toBeFocused();
  expect(ctl.patches).toEqual([]);

  await row.getByTestId("hazard-edit").click();
  await expect(row.getByTestId("hazard-edit-class")).toHaveValue("pothole");
  await expect(row.getByTestId("hazard-edit-severity")).toHaveValue("2");
});

test("an edited hazard shows its markers from the server data, in the list and in the popup", async ({ page }) => {
  const items0 = [
    hazard(1, { hazard_class: "debris in lane", class_edited: true, original_hazard_class: "pothole" }),
    hazard(2, { severity: 3, severity_edited: true, original_severity: 1 }),
    hazard(3),
  ];
  const ctl = await mockApi(page, items0);
  await openMap(page, ctl);
  await expect(item(page, 1).getByTestId("hazard-class-edited")).toBeVisible();
  await expect(item(page, 1).getByTestId("hazard-severity-edited")).toHaveCount(0);
  await expect(item(page, 2).getByTestId("hazard-severity-edited")).toHaveAttribute("title", "Edited (was 1)");
  await expect(item(page, 2).getByTestId("hazard-class-edited")).toHaveCount(0);
  await expect(page.getByTestId("hazard-class-edited")).toHaveCount(1);
  await expect(page.getByTestId("hazard-severity-edited")).toHaveCount(1);

  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-class-edited")).toBeVisible();
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText("debris in lane");
  await pin(page, 2).click();
  await expect(popup(page).getByTestId("hazard-popup-severity-edited")).toBeVisible();
  await expect(popup(page).getByTestId("hazard-popup-severity")).toContainText("Severity: 3 (High)");
  await pin(page, 3).click();
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText("pothole");
  await expect(popup(page).getByTestId("hazard-popup-class-edited")).toHaveCount(0);
});

test("setting the class back to the automatic value clears the edited marker (as the server reports it)", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1, { hazard_class: "debris in lane", class_edited: true, original_hazard_class: "pothole" })]);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("pothole");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-list-class")).toHaveText("pothole");
  await expect(row.getByTestId("hazard-class-edited")).toHaveCount(0);
});

test("a class that is not one of the five is still shown in the edit form so the form never lies", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1, { hazard_class: "sinkhole" })]);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await expect(row.getByTestId("hazard-edit-class")).toHaveValue("sinkhole");
  await expect(row.getByTestId("hazard-save")).toBeDisabled();
  await row.getByTestId("hazard-edit-severity").selectOption("3");
  await row.getByTestId("hazard-save").click();
  expect(ctl.patches[0].body).toEqual({ severity: 3 }); // the unchanged class is not sent
});

// ----- popup ---------------------------------------------------------------------------------------------------

test("the popup has the same review controls; Approve there updates the popup, pin and list and keeps it open", async ({
  page,
}) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await pin(page, 1).click();
  const card = popup(page);
  await expect(card.getByTestId("hazard-popup-status")).toHaveText("Status: New");
  await expect(card.getByTestId("hazard-approve")).toBeVisible();
  await expect(card.getByTestId("hazard-reject")).toBeVisible();
  await expect(card.getByTestId("hazard-reset")).toBeDisabled();
  await expect(card.getByTestId("hazard-edit")).toBeVisible();

  await card.getByTestId("hazard-approve").click();
  await expect(card.getByTestId("hazard-popup-status")).toHaveText("Status: Approved");
  await expect(card.getByTestId("hazard-popup-status-badge")).toHaveAttribute("data-status", "approved");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "approved");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(card).toHaveAttribute("data-hazard-id", "1");
  await expect(card.getByTestId("hazard-approve")).toBeDisabled();
  expect(ctl.patches.map((p) => p.body)).toEqual([{ status: "approved" }]);
});

test("editing in the popup saves, shows the edited marker in popup and list, and keeps the popup open", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await pin(page, 1).click();
  const card = popup(page);
  await card.getByTestId("hazard-edit").click();
  await expect(card.getByTestId("hazard-edit-class")).toBeFocused();
  await card.getByTestId("hazard-edit-class").selectOption("faded lane markings");
  await card.getByTestId("hazard-edit-severity").selectOption("1");
  await card.getByTestId("hazard-save").click();

  await expect(card.getByTestId("hazard-popup-class")).toHaveText("faded lane markings");
  await expect(card.getByTestId("hazard-popup-class-edited")).toBeVisible();
  await expect(card.getByTestId("hazard-popup-severity")).toContainText("1 (Low)");
  await expect(card.getByTestId("hazard-popup-severity-edited")).toBeVisible();
  await expect(card.getByTestId("hazard-edit-form")).toHaveCount(0);
  await expect(item(page, 1).getByTestId("hazard-list-class")).toHaveText("faded lane markings");
  await expect(item(page, 1).getByTestId("hazard-class-edited")).toBeVisible();
  expect(ctl.patches[0].body).toEqual({ hazard_class: "faded lane markings", severity: 1 });
});

test("the edit form fits inside the open popup and the map keeps the popup in view", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await pin(page, 2).click();
  await popup(page).getByTestId("hazard-edit").click();
  const form = popup(page).getByTestId("hazard-edit-form");
  await expect(form).toBeVisible();
  const wrapper = page.locator(".leaflet-popup-content-wrapper");
  await expect
    .poll(async () => {
      const outer = await wrapper.boundingBox();
      const inner = await form.boundingBox();
      // 1 px of slack for sub-pixel rounding; the popup is re-laid out a frame after the form opens, so poll.
      const slack = 1;
      return !!outer && !!inner && inner.x >= outer.x - slack && inner.x + inner.width <= outer.x + outer.width + slack &&
        inner.y >= outer.y - slack && inner.y + inner.height <= outer.y + outer.height + slack;
    })
    .toBe(true);
  await popup(page).getByTestId("hazard-cancel").click();
  await expect(form).toHaveCount(0);
});

// ----- errors --------------------------------------------------------------------------------------------------

test("404 shows 'Hazard not found', refreshes the list and the hazard disappears", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const gets = ctl.queries.length;
  ctl.items = ctl.items.filter((h) => h.id !== 1); // deleted on the server meanwhile
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(page.getByTestId("hazard-review-notice")).toContainText("Hazard 1 was not found");
  await expect(item(page, 1)).toHaveCount(0);
  await expect(pin(page, 1)).toHaveCount(0);
  await expect(items(page)).toHaveCount(3);
  expect(ctl.queries.length).toBeGreaterThan(gets);
  // the notice goes away with the next action
  await item(page, 2).getByTestId("hazard-reject").click();
  await expect(item(page, 2).getByTestId("hazard-list-status")).toHaveText("Rejected");
  await expect(page.getByTestId("hazard-review-notice")).toHaveCount(0);
});

test("the 404 error text is visible on the hazard until the refreshed list arrives", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = (route) => route.fulfill({ status: 404, json: { detail: "Hazard not found" } });
  ctl.getRespond = async (route) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: ctl.items.filter((h) => h.id !== 1) });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("Hazard not found");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New"); // not changed
  hold.open();
  await expect(item(page, 1)).toHaveCount(0);
});

test("422 shows the server message, keeps the previous value, and a retry clears the message", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) =>
    route.fulfill({ status: 422, json: { detail: "status must be one of new, approved, rejected" } });
  const row = item(page, 1);
  await row.getByTestId("hazard-approve").click();
  await expect(row.getByTestId("hazard-review-error")).toHaveText("status must be one of new, approved, rejected");
  await expect(row.getByTestId("hazard-review-error")).toHaveAttribute("role", "alert");
  await expect(row.getByTestId("hazard-list-status")).toHaveText("New");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "new");
  await expect(row.getByTestId("hazard-approve")).toBeEnabled(); // can try again

  ctl.patchRespond = null;
  await row.getByTestId("hazard-approve").click();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(row.getByTestId("hazard-review-error")).toHaveCount(0);
});

test("a 422 with a list of problems joins their messages", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) =>
    route.fulfill({ status: 422, json: { detail: [{ msg: "Input should be a valid integer" }, { msg: "Extra inputs are not permitted" }] } });
  await item(page, 1).getByTestId("hazard-reject").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText(
    "Input should be a valid integer; Extra inputs are not permitted",
  );
});

test("a 500 shows its detail (or the status code) and nothing changes", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) => route.fulfill({ status: 500, json: { detail: "database is locked" } });
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("database is locked");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "new");

  ctl.patchRespond = (route) => route.fulfill({ status: 500, body: "Internal Server Error" });
  await item(page, 1).getByTestId("hazard-reject").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toContainText("500");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New");
});

test("a network error shows 'Could not reach the server.' and keeps the previous value", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) => route.abort();
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("Could not reach the server.");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New");
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeEnabled();
  await expect(page.getByTestId("hazard-map-error")).toHaveCount(0); // the list itself is fine
  await expect(items(page)).toHaveCount(4);
});

test("an unusable success reply is reported as an error and changes nothing", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) => route.fulfill({ status: 200, json: { ok: true } });
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("Unexpected response from the server.");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New");
});

test("a failed edit keeps the form open with the draft, shows the error, and Cancel clears the error", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) => route.fulfill({ status: 422, json: { detail: "hazard_class must be one of pothole, debris in lane" } });
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("damaged guardrail");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-review-error")).toHaveText("hazard_class must be one of pothole, debris in lane");
  await expect(row.getByTestId("hazard-edit-form")).toBeVisible();
  await expect(row.getByTestId("hazard-edit-class")).toHaveValue("damaged guardrail"); // the draft is kept
  await expect(row.getByTestId("hazard-list-class")).toHaveText("pothole"); // the shown value is the old one
  await expect(row.getByTestId("hazard-class-edited")).toHaveCount(0);
  await expect(row.getByTestId("hazard-save")).toBeEnabled();

  await row.getByTestId("hazard-cancel").click();
  await expect(row.getByTestId("hazard-review-error")).toHaveCount(0);
  await expect(row.getByTestId("hazard-edit-form")).toHaveCount(0);
});

test("an error on one hazard does not show on another", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route, id) =>
    id === 1 ? route.fulfill({ status: 500, json: { detail: "boom" } }) : route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), { status: "rejected" }) });
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("boom");
  await item(page, 4).getByTestId("hazard-reject").click();
  await expect(item(page, 4).getByTestId("hazard-list-status")).toHaveText("Rejected");
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveText("boom"); // still there
  await expect(page.getByTestId("hazard-review-error")).toHaveCount(1);
});

test("the popup shows the error too when the review was started there", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  ctl.patchRespond = (route) => route.fulfill({ status: 500, json: { detail: "disk full" } });
  await pin(page, 1).click();
  await popup(page).getByTestId("hazard-approve").click();
  await expect(popup(page).getByTestId("hazard-review-error")).toHaveText("disk full");
  await expect(popup(page).getByTestId("hazard-popup-status")).toHaveText("Status: New");
});

// ----- in flight -----------------------------------------------------------------------------------------------

test("while a review is in flight that hazard's buttons are disabled and 'Saving...' shows; they come back after", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  const row = item(page, 1);
  await row.getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await expect(row.getByTestId("hazard-review-saving")).toBeVisible();
  await expect(row).toHaveAttribute("data-selected", "false");
  await expect(row.getByTestId("hazard-review")).toHaveAttribute("aria-busy", "true");
  for (const t of ["hazard-approve", "hazard-reject", "hazard-reset", "hazard-edit"]) {
    await expect(row.getByTestId(t)).toBeDisabled();
  }
  // the displayed status has not changed yet (no optimistic update)
  await expect(row.getByTestId("hazard-list-status")).toHaveText("New");

  hold.open();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(row.getByTestId("hazard-review-saving")).toHaveCount(0);
  await expect(row.getByTestId("hazard-reject")).toBeEnabled();
  await expect(row.getByTestId("hazard-edit")).toBeEnabled();
});

test("double-clicking Approve sends exactly one request", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await item(page, 1).getByTestId("hazard-approve").dblclick();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeDisabled();
  hold.open();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  expect(ctl.patches).toHaveLength(1);
});

test("clicking Approve then Reject while the first is in flight sends only the first", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await item(page, 1).getByTestId("hazard-reject").click({ force: true }).catch(() => {});
  hold.open();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  expect(ctl.patches.map((p) => p.body)).toEqual([{ status: "approved" }]);
});

test("double-clicking Save sends one request", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-severity").selectOption("3");
  await row.getByTestId("hazard-save").dblclick();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await expect(row.getByTestId("hazard-save")).toBeDisabled();
  await expect(row.getByTestId("hazard-cancel")).toBeDisabled();
  await expect(row.getByTestId("hazard-edit-class")).toBeDisabled();
  await expect(row.getByTestId("hazard-edit-severity")).toBeDisabled();
  hold.open();
  await expect(row.getByTestId("hazard-severity-edited")).toBeVisible();
  expect(ctl.patches).toHaveLength(1);
});

test("a review of one hazard does not block another; replies may arrive in any order", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const holds = { 1: gate(), 3: gate() };
  ctl.patchRespond = async (route, id, body) => {
    await holds[id].promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await item(page, 3).getByTestId("hazard-reset").click();
  await expect.poll(() => ctl.patches.length).toBe(2);
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeDisabled();
  await expect(item(page, 3).getByTestId("hazard-reset")).toBeDisabled();
  await expect(item(page, 2).getByTestId("hazard-reject")).toBeEnabled(); // an unrelated hazard is free
  await expect(item(page, 4).getByTestId("hazard-approve")).toBeEnabled();

  holds[3].open(); // the second one answers first
  await expect(item(page, 3).getByTestId("hazard-list-status")).toHaveText("New");
  await expect(item(page, 3).getByTestId("hazard-reset")).toBeDisabled(); // now new: nothing to reset
  await expect(item(page, 3).getByTestId("hazard-approve")).toBeEnabled();
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeDisabled(); // 1 is still waiting
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New");

  holds[1].open();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(item(page, 1).getByTestId("hazard-reject")).toBeEnabled();
  expect(ctl.patches.map((p) => [p.id, p.body.status]).sort()).toEqual([[1, "approved"], [3, "new"]]);
});

test("an error for one hazard while another is in flight touches only the first", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    if (id === 1) {
      await hold.promise;
      return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === 1), body) });
    }
    return route.fulfill({ status: 500, json: { detail: "nope" } });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await item(page, 4).getByTestId("hazard-reject").click();
  await expect(item(page, 4).getByTestId("hazard-review-error")).toHaveText("nope");
  await expect(item(page, 1).getByTestId("hazard-review-error")).toHaveCount(0);
  await expect(item(page, 1).getByTestId("hazard-review-saving")).toBeVisible();
  hold.open();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(item(page, 4).getByTestId("hazard-list-status")).toHaveText("Reported");
});

test("the popup and the list share the in-flight state of a hazard", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await pin(page, 1).click();
  await popup(page).getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await expect(item(page, 1).getByTestId("hazard-approve")).toBeDisabled(); // the list copy is locked too
  await expect(popup(page).getByTestId("hazard-review-saving")).toBeVisible();
  hold.open();
  await expect(popup(page).getByTestId("hazard-popup-status")).toHaveText("Status: Approved");
  expect(ctl.patches).toHaveLength(1);
});

// ----- status filter -------------------------------------------------------------------------------------------

test("approving under the 'new' filter removes the hazard from the list and the map", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1), hazard(2), hazard(3, { status: "approved" })]);
  await page.goto("/#map");
  await expect(items(page)).toHaveCount(3);
  await page.getByTestId("hazard-map-filter").selectOption("new");
  await expect(items(page)).toHaveCount(2);
  await expect(pins(page)).toHaveCount(2);
  await expect(page.getByTestId("hazard-map-count")).toHaveText("2 hazards");

  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1)).toHaveCount(0);
  await expect(pin(page, 1)).toHaveCount(0);
  await expect(items(page)).toHaveCount(1);
  await expect(page.getByTestId("hazard-map-count")).toHaveText("1 hazard");
  expect(ctl.patches).toHaveLength(1);

  await item(page, 2).getByTestId("hazard-reject").click();
  await expect(items(page)).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-empty")).toHaveText("No hazards with status New");
  await expect(pins(page)).toHaveCount(0);

  // the server really has them now: switching the filter shows them where they belong
  await page.getByTestId("hazard-map-filter").selectOption("approved");
  await expect(items(page)).toHaveCount(2);
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await page.getByTestId("hazard-map-filter").selectOption("rejected");
  await expect(items(page)).toHaveCount(1);
  await expect(item(page, 2)).toBeVisible();
});

test("an edit under a filter keeps the hazard (its status still matches)", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1), hazard(2, { status: "approved" })]);
  await page.goto("/#map");
  await page.getByTestId("hazard-map-filter").selectOption("new");
  await expect(items(page)).toHaveCount(1);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-severity").selectOption("3");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-severity-edited")).toBeVisible();
  await expect(items(page)).toHaveCount(1);
  expect(ctl.patches).toHaveLength(1);
});

test("resetting under the 'approved' filter removes it; under 'All statuses' it stays and changes badge", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1, { status: "approved" }), hazard(2, { status: "approved" })]);
  await page.goto("/#map");
  await expect(items(page)).toHaveCount(2);
  await item(page, 1).getByTestId("hazard-reset").click();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("New");
  await expect(items(page)).toHaveCount(2);

  await page.getByTestId("hazard-map-filter").selectOption("approved");
  await expect(items(page)).toHaveCount(1);
  await item(page, 2).getByTestId("hazard-reset").click();
  await expect(items(page)).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-empty")).toHaveText("No hazards with status Approved");
  expect(ctl.patches.map((p) => p.id)).toEqual([1, 2]);
});

test("a hazard removed from the list by a filter drops its selection and popup", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1), hazard(2)]);
  await page.goto("/#map");
  await page.getByTestId("hazard-map-filter").selectOption("new");
  await expect(items(page)).toHaveCount(2);
  await pin(page, 1).click();
  await expect(popup(page)).toHaveAttribute("data-hazard-id", "1");
  await popup(page).getByTestId("hazard-approve").click();
  await expect(item(page, 1)).toHaveCount(0);
  await expect(page.locator(".leaflet-popup")).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "");
  expect(ctl.patches).toHaveLength(1);
});

test("a list loaded while a review is in flight is shown, then loaded again once the review has settled", async ({ page }) => {
  const ctl = await mockApi(page, [hazard(1), hazard(2, { status: "approved" })]);
  await page.goto("/#map");
  await expect(items(page)).toHaveCount(2);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await page.getByTestId("hazard-map-filter").selectOption("approved"); // the server has not applied the change yet
  await expect(items(page)).toHaveCount(1); // the reply is shown as the server knew it: only hazard 2
  expect(ctl.queries.at(-1)).toBe("?status=approved");
  hold.open();
  await expect(items(page)).toHaveCount(2); // reloaded after the review: both are approved now
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(page.getByTestId("hazard-map-loading")).toHaveCount(0);
});

test("a list reply that was started before a review finished never puts the old status back", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const patchGate = gate();
  const listGate = gate();
  ctl.patchRespond = async (route, id, body) => {
    await patchGate.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  let held = false;
  ctl.getRespond = async (route) => {
    const snapshot = JSON.parse(JSON.stringify(ctl.items)); // what the server knew when this request arrived
    if (!held) {
      held = true;
      await listGate.promise;
    }
    return route.fulfill({ status: 200, json: snapshot }).catch(() => {});
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);
  await page.getByTestId("hazard-map-refresh").click(); // GET #1 starts now and is held, with the OLD status
  await expect.poll(() => held).toBe(true);
  const before = ctl.queries.length;
  patchGate.open(); // the review commits and the page shows it
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  listGate.open(); // the old reply arrives only now
  await expect(page.getByTestId("hazard-map-loading")).toHaveCount(0);
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "approved");
  expect(ctl.queries.length).toBeGreaterThan(before); // a fresh load replaced the dropped one
});

test("leaving the map tab and coming back during a review keeps that hazard's buttons disabled", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const hold = gate();
  ctl.patchRespond = async (route, id, body) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body) });
  };
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect.poll(() => ctl.patches.length).toBe(1);

  await page.getByTestId("tab-drives").click();
  await expect(page.getByTestId("hazard-map-section")).toHaveCount(0);
  await page.getByTestId("tab-map").click();
  await expect(items(page)).toHaveCount(4);
  await expect(pins(page)).toHaveCount(4);
  const row = item(page, 1);
  await expect(row.getByTestId("hazard-review-saving")).toBeVisible();
  for (const t of ["hazard-approve", "hazard-reject", "hazard-reset", "hazard-edit"]) {
    await expect(row.getByTestId(t)).toBeDisabled();
  }
  await expect(item(page, 2).getByTestId("hazard-reject")).toBeEnabled(); // other hazards are free
  await expect(row.getByTestId("hazard-list-status")).toHaveText("New");

  hold.open();
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Approved"); // the reload after the review
  await expect(row.getByTestId("hazard-review-saving")).toHaveCount(0);
  await expect(row.getByTestId("hazard-reject")).toBeEnabled();
  expect(ctl.patches).toHaveLength(1);
});

// ----- keyboard ------------------------------------------------------------------------------------------------

test("the buttons are reachable by role and name and work from the keyboard (Enter and Space)", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const approve = page.getByRole("button", { name: "Approve hazard 1" });
  await approve.focus();
  await expect(approve).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");

  const reject = page.getByRole("button", { name: "Reject hazard 1" });
  await reject.focus();
  await page.keyboard.press("Space");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Rejected");
  expect(ctl.patches.map((p) => p.body.status)).toEqual(["approved", "rejected"]);
});

test("the whole edit works with the keyboard: Edit, change, Tab to Save, Enter", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await page.getByRole("button", { name: "Edit class or severity of hazard 1" }).focus();
  await page.keyboard.press("Enter");
  const classSelect = page.getByRole("combobox", { name: "Class of hazard 1" });
  await expect(classSelect).toBeFocused();
  await classSelect.selectOption("damaged guardrail");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("combobox", { name: "Severity of hazard 1" })).toBeFocused();
  await page.keyboard.press("Tab");
  const save = page.getByRole("button", { name: "Save changes to hazard 1" });
  await expect(save).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(item(page, 1).getByTestId("hazard-list-class")).toHaveText("damaged guardrail");
  await expect(page.getByRole("button", { name: "Edit class or severity of hazard 1" })).toBeFocused();
  expect(ctl.patches[0].body).toEqual({ hazard_class: "damaged guardrail" });
});

test("Cancel is reachable by keyboard after Save", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await page.getByRole("button", { name: "Edit class or severity of hazard 2" }).focus();
  await page.keyboard.press("Enter");
  await page.getByRole("combobox", { name: "Severity of hazard 2" }).focus();
  await page.keyboard.press("Tab"); // Save (disabled, so skipped) -> Cancel
  const cancel = page.getByRole("button", { name: "Cancel editing hazard 2" });
  await expect(cancel).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(item(page, 2).getByTestId("hazard-edit-form")).toHaveCount(0);
  expect(ctl.patches).toEqual([]);
});

test("pressing Enter in the form's Save while nothing changed does nothing", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").press("Enter"); // a select does not submit; nothing changed anyway
  await expect(row.getByTestId("hazard-edit-form")).toBeVisible();
  expect(ctl.patches).toEqual([]);
});

// ----- hygiene -------------------------------------------------------------------------------------------------

test("reviews only ever PATCH /api/hazards/{id} with JSON, and never send anything but the changed fields", async ({ page }) => {
  const ctl = await mockApi(page);
  const writes = [];
  page.on("request", (r) => {
    if (new URL(r.url()).pathname.startsWith("/api/") && r.method() !== "GET") writes.push(`${r.method()} ${new URL(r.url()).pathname}`);
  });
  await openMap(page, ctl);
  await item(page, 1).getByTestId("hazard-approve").click();
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await item(page, 2).getByTestId("hazard-edit").click();
  await item(page, 2).getByTestId("hazard-edit-severity").selectOption("3");
  await item(page, 2).getByTestId("hazard-save").click();
  await expect(item(page, 2).getByTestId("hazard-severity-edited")).toBeVisible();
  expect(writes).toEqual(["PATCH /api/hazards/1", "PATCH /api/hazards/2"]);
  for (const p of ctl.patches) {
    expect(p.contentType).toBe("application/json");
    expect(Object.keys(p.body).every((k) => ["status", "hazard_class", "severity"].includes(k))).toBe(true);
  }
});

test("review controls render text safely for hazard classes that contain markup", async ({ page }) => {
  const evil = "<img src=x onerror='window.__pwned=1'>";
  const ctl = await mockApi(page, [hazard(1, { hazard_class: evil, class_edited: true, original_hazard_class: evil })]);
  await openMap(page, ctl);
  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText(evil);
  await item(page, 1).getByTestId("hazard-edit").click();
  await expect(item(page, 1).getByTestId("hazard-edit-class")).toHaveValue(evil);
  expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
});
