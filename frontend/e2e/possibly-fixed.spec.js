import { test, expect } from "@playwright/test";

// UI behaviour of possibly-fixed hazards (task 19). Every API call is mocked with page.route (PATCH requests are
// recorded so their bodies can be asserted), and so are the OpenStreetMap tiles. No fixed sleeps.

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const FIXED_MESSAGE = "A report is only available for approved hazards, and this hazard is possibly fixed. Approve it first.";

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
  first_seen: "2026-10-01T12:30:00+00:00",
  last_seen: "2026-10-01T12:30:00+00:00",
  status: "new",
  reviewed_at: null,
  misses: 0,
  possibly_fixed_at: null,
  previous_status: null,
  class_edited: false,
  severity_edited: false,
  original_hazard_class: null,
  original_severity: null,
  agency: "VDOT",
  agency_source: "auto",
  automatic_agency: "VDOT",
  photo_url: `/api/hazards/${id}/photo`,
  crop_url: null,
  ...patch,
});

const fresh = () => [
  hazard(1, { status: "possibly_fixed", misses: 3, possibly_fixed_at: "2026-10-25T14:05:00+00:00", previous_status: "approved", reviewed_at: "2026-10-02T10:00:00+00:00" }),
  hazard(2, { status: "approved", misses: 2, reviewed_at: "2026-10-02T10:00:00+00:00" }),
  hazard(3, { status: "possibly_fixed", misses: 1, possibly_fixed_at: "2026-10-26T09:00:00+00:00", previous_status: "new", hazard_class: "damaged guardrail" }),
  hazard(4, { status: "possibly_fixed", misses: 4, possibly_fixed_at: "not a date", previous_status: null, hazard_class: "debris in lane" }),
  hazard(5, { status: "new" }),
];

/**
 * Mock the hazards API. ctl.items is the server state; ctl.patches records { id, body }; ctl.reportGets records the
 * ids of report requests; ctl.reportRespond (async (route, id) => ...) answers them (default: 409 "possibly fixed").
 */
async function mockApi(page, items = fresh()) {
  const ctl = { items, patches: [], queries: [], reportGets: [], reportRespond: null };
  await page.route(/openstreetmap\.org/, (route) => route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }));
  await page.route(
    (url) => url.pathname === "/api/hazards",
    (route) => {
      const query = new URL(route.request().url()).search;
      ctl.queries.push(query);
      const status = new URLSearchParams(query).get("status");
      return route.fulfill({ status: 200, json: status ? ctl.items.filter((h) => h.status === status) : ctl.items });
    },
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+$/.test(url.pathname),
    (route) => {
      const request = route.request();
      const id = Number(new URL(request.url()).pathname.split("/").pop());
      const body = request.postDataJSON();
      ctl.patches.push({ id, body });
      const target = ctl.items.find((h) => h.id === id);
      return route.fulfill({ status: 200, json: applyPatch(target, body) });
    },
  );
  await page.route(/\/api\/hazards\/\d+\/report$/, (route) => {
    const id = Number(new URL(route.request().url()).pathname.split("/")[3]);
    ctl.reportGets.push(id);
    if (ctl.reportRespond) return ctl.reportRespond(route, id);
    return route.fulfill({ status: 409, json: { detail: FIXED_MESSAGE } });
  });
  await page.route(/\/api\/hazards\/\d+\/photo$/, (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  return ctl;
}

/** What the real server does with a review body: answering a possibly fixed hazard dismisses the flag. */
function applyPatch(h, body) {
  if (body.status !== undefined) {
    if (h.status === "possibly_fixed") {
      h.previous_status = null;
      h.possibly_fixed_at = null;
      h.misses = 0;
    }
    h.status = body.status;
    h.reviewed_at = body.status === "new" ? null : "2026-10-27T10:00:00+00:00";
  }
  if (body.hazard_class !== undefined) {
    h.original_hazard_class = h.hazard_class;
    h.hazard_class = body.hazard_class;
    h.class_edited = true;
  }
  return { ...h };
}

const item = (page, id) => page.locator(`[data-testid="hazard-list-item"][data-hazard-id="${id}"]`);
const items = (page) => page.getByTestId("hazard-list-item");
const pin = (page, id) => page.locator(`[data-testid="hazard-pin"][data-hazard-id="${id}"]`);
const popup = (page) => page.locator(".leaflet-popup").getByTestId("hazard-popup");

async function openMap(page, ctl) {
  await page.goto("/#map");
  await expect(page.getByTestId("hazard-map-section")).toBeVisible();
  await expect(items(page)).toHaveCount(ctl.items.length);
  await expect(page.getByTestId("hazard-pin")).toHaveCount(ctl.items.length);
}

test("a possibly fixed hazard has a Possibly fixed badge in the list; others do not", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Possibly fixed");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveAttribute("data-status", "possibly_fixed");
  await expect(item(page, 2).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(item(page, 5).getByTestId("hazard-list-status")).toHaveText("New");
});

test("the note says how many later drives missed it, when it was flagged and what it was before", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const note = item(page, 1).getByTestId("hazard-fix-note");
  await expect(note).toBeVisible();
  await expect(note).toHaveAttribute("data-misses", "3");
  await expect(note.getByTestId("hazard-fix-misses")).toHaveText("Not seen on 3 later drives");
  await expect(note.getByTestId("hazard-fix-flagged-at")).toContainText("Flagged 2026-10-25 14:05 UTC");
  await expect(note.getByTestId("hazard-fix-previous")).toContainText("Was Approved before");
});

test("the note uses the singular for one drive and copes with a bad time and no previous status", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const one = item(page, 3).getByTestId("hazard-fix-note");
  await expect(one.getByTestId("hazard-fix-misses")).toHaveText("Not seen on 1 later drive");
  await expect(one.getByTestId("hazard-fix-previous")).toContainText("Was New before");
  const odd = item(page, 4).getByTestId("hazard-fix-note");
  await expect(odd.getByTestId("hazard-fix-misses")).toHaveText("Not seen on 4 later drives");
  await expect(odd.getByTestId("hazard-fix-flagged-at")).toContainText("an unknown time");
  await expect(odd.getByTestId("hazard-fix-previous")).toHaveCount(0);
});

test("hazards that are not possibly fixed show no note and no hint", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  for (const id of [2, 5]) {
    await expect(item(page, id).getByTestId("hazard-fix-note")).toHaveCount(0);
    await expect(item(page, id).getByTestId("hazard-fix-hint")).toHaveCount(0);
  }
});

test("the pin of a possibly fixed hazard has its own colour and a dashed border", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(pin(page, 1)).toHaveAttribute("data-status", "possibly_fixed");
  await expect(pin(page, 1)).toHaveCSS("background-color", "rgb(239, 108, 0)");
  await expect(pin(page, 1)).toHaveCSS("border-top-style", "dashed");
  await expect(pin(page, 2)).toHaveCSS("border-top-style", "solid");
  await expect(pin(page, 2)).not.toHaveCSS("background-color", "rgb(239, 108, 0)");
});

test("the status filter has a Possibly fixed option that asks for and shows only those hazards", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const filter = page.getByTestId("hazard-map-filter");
  await expect(filter.locator('option[value="possibly_fixed"]')).toHaveText("Possibly fixed");
  await filter.selectOption("possibly_fixed");
  await expect(items(page)).toHaveCount(3);
  expect(ctl.queries.at(-1)).toContain("status=possibly_fixed");
  for (const id of [1, 3, 4]) await expect(item(page, id)).toBeVisible();
});

test("the popup of a possibly fixed pin shows the badge and the note", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await item(page, 1).locator("button.rs-list-button").click();
  await expect(popup(page)).toBeVisible();
  await expect(popup(page).getByTestId("hazard-popup-status-badge")).toHaveText("Possibly fixed");
  await expect(popup(page).getByTestId("hazard-popup-fix-misses")).toHaveText("Not seen on 3 later drives");
  await expect(popup(page).getByTestId("hazard-popup-fix-flagged-at")).toContainText("2026-10-25 14:05 UTC");
  await expect(popup(page).getByTestId("hazard-fix-hint")).toBeVisible();
});

test("a possibly fixed hazard keeps all review controls enabled and explains that they dismiss the flag", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  for (const id of ["hazard-approve", "hazard-reject", "hazard-reset", "hazard-edit"]) {
    await expect(row.getByTestId(id)).toBeEnabled();
  }
  const hint = row.getByTestId("hazard-fix-hint");
  await expect(hint).toBeVisible();
  await expect(hint).toContainText("Approve, Reject or Reset answers this flag");
});

for (const [testid, status, label] of [
  ["hazard-approve", "approved", "Approved"],
  ["hazard-reject", "rejected", "Rejected"],
  ["hazard-reset", "new", "New"],
]) {
  test(`${testid} on a possibly fixed hazard sends status ${status} and removes the flag, note and hint`, async ({ page }) => {
    const ctl = await mockApi(page);
    await openMap(page, ctl);
    await item(page, 1).getByTestId(testid).click();
    await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText(label);
    expect(ctl.patches).toEqual([{ id: 1, body: { status } }]);
    await expect(item(page, 1).getByTestId("hazard-fix-note")).toHaveCount(0);
    await expect(item(page, 1).getByTestId("hazard-fix-hint")).toHaveCount(0);
    await expect(pin(page, 1)).toHaveAttribute("data-status", status);
  });
}

test("under the Possibly fixed filter an answered hazard leaves the list", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await page.getByTestId("hazard-map-filter").selectOption("possibly_fixed");
  await expect(items(page)).toHaveCount(3);
  await item(page, 3).getByTestId("hazard-approve").click();
  await expect(items(page)).toHaveCount(2);
  await expect(item(page, 3)).toHaveCount(0);
});

test("editing the class of a possibly fixed hazard does not change its flag", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const row = item(page, 1);
  await row.getByTestId("hazard-edit").click();
  await row.getByTestId("hazard-edit-class").selectOption("debris in lane");
  await row.getByTestId("hazard-save").click();
  await expect(row.getByTestId("hazard-class-edited")).toBeVisible();
  expect(ctl.patches).toEqual([{ id: 1, body: { hazard_class: "debris in lane" } }]);
  await expect(row.getByTestId("hazard-list-status")).toHaveText("Possibly fixed");
  await expect(row.getByTestId("hazard-fix-misses")).toHaveText("Not seen on 3 later drives");
});

test("a possibly fixed hazard has no Prefilled report button and says why", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(item(page, 1).getByTestId("hazard-report-open")).toHaveCount(0);
  await expect(item(page, 3).getByTestId("hazard-report-open")).toHaveCount(0);
  await expect(item(page, 2).getByTestId("hazard-report-open")).toHaveCount(1);
  await expect(item(page, 1).getByTestId("hazard-fix-report-note")).toContainText("No prefilled report while it is flagged");
  expect(ctl.reportGets).toEqual([]);
});

test("a report that turns out to be refused because the hazard is possibly fixed shows a clear message", async ({ page }) => {
  // The hazard was approved when the list loaded and has been flagged since: the report endpoint answers 409.
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  const button = item(page, 2).getByTestId("hazard-report-open");
  await expect(button).toBeVisible();
  await button.click();
  const panel = page.getByTestId("report-panel");
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("report-error")).toHaveText(FIXED_MESSAGE);
  await expect(panel.getByTestId("report-error")).toHaveAttribute("data-status", "409");
  await expect(panel.getByTestId("report-fixed-hint")).toContainText("approve the hazard again");
  await expect(panel.getByTestId("report-retry")).toHaveCount(0);
  expect(ctl.reportGets.length).toBeGreaterThan(0);
  expect(new Set(ctl.reportGets)).toEqual(new Set([2]));
  await panel.getByTestId("report-close").click();
  await expect(panel).toHaveCount(0);
});

test("a 409 refusal for another reason shows its message without the possibly fixed hint", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route) => route.fulfill({ status: 409, json: { detail: "Something else is wrong." } });
  await openMap(page, ctl);
  await item(page, 2).getByTestId("hazard-report-open").click();
  const panel = page.getByTestId("report-panel");
  await expect(panel.getByTestId("report-error")).toHaveText("Something else is wrong.");
  await expect(panel.getByTestId("report-fixed-hint")).toHaveCount(0);
});
