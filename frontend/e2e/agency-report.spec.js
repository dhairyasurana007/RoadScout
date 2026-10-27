import { test, expect } from "@playwright/test";

// UI behaviour of the prefilled agency report (task 18). Every API call is mocked with page.route (PATCH requests
// and report requests are recorded), and so are the OpenStreetMap tiles. Holds use hand-opened gates, never fixed
// sleeps. The agency hosts are blocked and counted: the UI must never contact them.

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const AGENCY_HOSTS = /vdot\.virginia\.gov|maryland\.gov|salesforce/i;
const NOTHING_SUBMITTED = "Nothing is submitted automatically. You submit this report yourself.";

const MDOT_TOPICS = {
  pothole: "Potholes / Holes / Sinkholes",
  "debris in lane": "Debris, Graffiti, & Litter",
  "damaged guardrail": "Guardrail/Roadway Safety Devices",
  "fallen or damaged sign": "Sign / Digital Message Sign",
  "faded lane markings": "Line Striping / Roadway & Cross Walk Markings",
};
const VDOT_CATEGORIES = {
  pothole: "Pothole repair",
  "debris in lane": "Debris in the road",
  "damaged guardrail": "Damaged guardrail",
  "fallen or damaged sign": "Damaged or missing sign",
  "faded lane markings": "Faded pavement markings",
};

// automatic agency by where the hazard is: Virginia, Maryland or neither (the District of Columbia)
const PLACES = {
  va: { lat: 38.84, lon: -77.43, auto: "VDOT" },
  md: { lat: 39.08, lon: -77.15, auto: "MDOT SHA" },
  dc: { lat: 38.9, lon: -77.03, auto: null },
};

const hazard = (id, place, patch = {}) => ({
  id,
  hazard_class: "pothole",
  lat: PLACES[place].lat + id * 0.01, // distinct spots, so no pin hides another
  lon: PLACES[place].lon + id * 0.01,
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
  agency: PLACES[place].auto,
  agency_source: "auto",
  automatic_agency: PLACES[place].auto,
  photo_url: `/api/hazards/${id}/photo`,
  crop_url: null,
  ...patch,
});

const fresh = () => [
  hazard(1, "va", { hazard_class: "pothole", status: "approved" }),
  hazard(2, "md", { hazard_class: "debris in lane", status: "approved", direction: 200 }),
  hazard(3, "dc", { hazard_class: "pothole", status: "approved" }),
  hazard(4, "va", { hazard_class: "faded lane markings", status: "new", reviewed_at: null }),
  hazard(5, "va", { hazard_class: "damaged guardrail", status: "rejected" }),
  hazard(6, "va", { hazard_class: "fallen or damaged sign", status: "reported" }),
];

function gate() {
  let open;
  const promise = new Promise((resolve) => (open = resolve));
  return { promise, open };
}

/** The effective agency of a stored hazard (override wins over the automatic one). */
function effective(h) {
  if (h.agency_source === "user") return h.agency;
  return h.automatic_agency;
}

/** What GET /api/hazards/{id}/report returns (same keys as the real server). */
function buildReport(h) {
  const agency = h.agency;
  const map = `https://www.openstreetmap.org/?mlat=${h.lat.toFixed(6)}&mlon=${h.lon.toFixed(6)}#map=18/${h.lat.toFixed(6)}/${h.lon.toFixed(6)}`;
  const description = `Road hazard: ${h.hazard_class}. Severity: medium (2 of 3). Seen on ${h.sightings} drives. Location: ${h.lat.toFixed(6)}, ${h.lon.toFixed(6)}. Map: ${map}`;
  let category = null;
  let fields = {};
  let field_info = {};
  let sub_topic_note = null;
  let verified = false;
  let form_url = null;
  if (agency === "MDOT SHA") {
    category = MDOT_TOPICS[h.hazard_class];
    sub_topic_note = "Choose on the form: the required Sub-Topic depends on the Topic and its options are not published.";
    verified = true;
    form_url = "https://mdotsha.my.salesforce-sites.com/customercare";
    fields = { topic: category, sub_topic: null, description, county: null, direction: "E" };
    field_info = {
      topic: { label: "Topic", required: true, note: null },
      sub_topic: { label: "Sub-Topic", required: true, note: sub_topic_note },
      description: { label: "Description", required: false, note: "Includes the coordinates and a map link." },
      county: { label: "County", required: false, note: "Left to you: pick the Maryland county." },
      direction: { label: "Direction", required: false, note: "From the direction of travel." },
    };
  } else if (agency === "VDOT") {
    category = VDOT_CATEGORIES[h.hazard_class];
    sub_topic_note = "Choose on the form: VDOT's guided flow asks follow-up questions.";
    form_url = "https://my.vdot.virginia.gov/";
    fields = { problem_type: category, location: `${h.lat.toFixed(6)}, ${h.lon.toFixed(6)}`, description, blocks_traffic: null };
    field_info = {
      problem_type: { label: "Kind of repair or problem", required: true, note: "unverified, check on the live form" },
      location: { label: "Where it is", required: true, note: "unverified, check on the live form" },
      description: { label: "Description", required: false, note: "unverified, check on the live form" },
      blocks_traffic: { label: "Does it block traffic?", required: false, note: "Left to you" },
    };
  }
  const text = `RoadScout prefilled report for hazard ${h.id} (NOT SUBMITTED)\nAgency: ${agency ?? "none"}\nCategory: ${category}\n${description}\n`;
  return {
    hazard_id: h.id,
    hazard_status: h.status,
    hazard_class: h.hazard_class,
    agency,
    agency_name: agency ? `${agency} (full name)` : null,
    agency_source: h.agency_source,
    automatic_agency: h.automatic_agency,
    category,
    sub_topic_note,
    fields,
    field_info,
    verified_fields: verified,
    verification_note: agency === "VDOT" ? "Unverified: VDOT's report flow could not be checked." : agency ? "Read from the live form." : "No agency form applies.",
    description,
    lat: h.lat,
    lon: h.lon,
    compass: "E",
    map_url: map,
    location: { lat: h.lat, lon: h.lon, compass: "E", heading: 90, map_url: map },
    photo_url: h.photo_url,
    photo_download_url: h.photo_url ? `${h.photo_url}?download=1` : null,
    text_download_url: `/api/hazards/${h.id}/report?format=txt`,
    photo_note: "The form has no photo upload.",
    form_url,
    submitted: false,
    note: "Nothing was submitted.",
    agency_note: agency
      ? "Change the agency if this is not a road this agency looks after."
      : "No agency: this location is not in Virginia or Maryland, so there is no form for it.",
    text,
  };
}

/** What the real server does with a review body (mutates and returns the stored hazard) or a refusal. */
function applyPatch(h, body) {
  if (body.status !== undefined) {
    if (body.status === "reported" && !["approved", "reported"].includes(h.status)) {
      return { error: { status: 409, detail: `only an approved hazard can be marked as reported; this one is ${h.status}` } };
    }
    h.status = body.status;
    if (body.status !== "reported") h.reviewed_at = body.status === "new" ? null : "2026-10-27T10:00:00+00:00";
  }
  if ("agency" in body) {
    const value = body.agency;
    if (value === null || value === "auto") {
      h.agency_source = "auto";
      h.agency = h.automatic_agency;
    } else if (["VDOT", "MDOT SHA", "none"].includes(value)) {
      h.agency_source = "user";
      h.agency = value === "none" ? null : value;
    } else {
      return { error: { status: 422, detail: `agency must be one of VDOT, MDOT SHA, none, auto or null; got ${JSON.stringify(value)}` } };
    }
  }
  return { hazard: { ...h } };
}

/**
 * Mock the hazards API. ctl.items is the server state. ctl.patches / ctl.reportGets record the requests.
 * ctl.patchRespond / ctl.reportRespond (async (route, id, body|n) => ...) take over a response.
 * ctl.agencyHits counts requests to an agency host (the UI must make none).
 */
async function mockApi(page, items = fresh()) {
  const ctl = { items, patches: [], reportGets: [], requests: [], agencyHits: [], patchRespond: null, reportRespond: null, photoStatus: {} };
  page.on("request", (request) => ctl.requests.push(request.url()));
  await page.route(AGENCY_HOSTS, (route) => {
    ctl.agencyHits.push(route.request().url());
    return route.abort();
  });
  await page.route(/openstreetmap\.org/, (route) => route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }));
  await page.route(
    (url) => url.pathname === "/api/hazards",
    (route) => route.fulfill({ status: 200, json: ctl.items.map((h) => ({ ...h })) }),
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+$/.test(url.pathname),
    async (route) => {
      const request = route.request();
      const id = Number(new URL(request.url()).pathname.split("/").pop());
      if (request.method() !== "PATCH") return route.fulfill({ status: 405, json: { detail: "Method Not Allowed" } });
      const body = request.postDataJSON();
      ctl.patches.push({ id, body });
      if (ctl.patchRespond) return ctl.patchRespond(route, id, body);
      const target = ctl.items.find((h) => h.id === id);
      if (!target) return route.fulfill({ status: 404, json: { detail: "Hazard not found" } });
      const result = applyPatch(target, body);
      if (result.error) return route.fulfill({ status: result.error.status, json: { detail: result.error.detail } });
      return route.fulfill({ status: 200, json: result.hazard });
    },
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+\/report$/.test(url.pathname),
    async (route) => {
      const id = Number(new URL(route.request().url()).pathname.split("/")[3]);
      ctl.reportGets.push(id);
      if (ctl.reportRespond) return ctl.reportRespond(route, id, ctl.reportGets.length);
      const target = ctl.items.find((h) => h.id === id);
      if (!target) return route.fulfill({ status: 404, json: { detail: "Hazard not found" } });
      if (!["approved", "reported"].includes(target.status)) {
        return route.fulfill({ status: 409, json: { detail: `A report is only available for approved hazards, and this hazard is ${target.status}. Approve it first.` } });
      }
      return route.fulfill({ status: 200, json: buildReport(target) });
    },
  );
  await page.route(/^.*\/api\/hazards\/\d+\/photo$/, (route) => {
    const id = Number(new URL(route.request().url()).pathname.split("/")[3]);
    const status = ctl.photoStatus[id] ?? 200;
    return status === 200 ? route.fulfill({ status, contentType: "image/png", body: PIXEL }) : route.fulfill({ status, json: { detail: "Photo not found" } });
  });
  return ctl;
}

const item = (page, id) => page.locator(`[data-testid="hazard-list-item"][data-hazard-id="${id}"]`);
const pin = (page, id) => page.locator(`[data-testid="hazard-pin"][data-hazard-id="${id}"]`);
const popup = (page) => page.locator(".leaflet-popup").getByTestId("hazard-popup");
const panel = (page) => page.getByTestId("report-panel");
const opener = (page, id) => item(page, id).getByTestId("hazard-report-open");

async function openMap(page, ctl) {
  await page.goto("/#map");
  await expect(page.getByTestId("hazard-map-section")).toBeVisible();
  await expect(page.getByTestId("hazard-list-item")).toHaveCount(ctl.items.length);
  await expect(page.getByTestId("hazard-pin")).toHaveCount(ctl.items.length);
}

async function openReport(page, id) {
  await opener(page, id).click();
  await expect(panel(page)).toBeVisible();
  await expect(panel(page)).toHaveAttribute("data-phase", "ready");
  await expect(page.getByTestId("report-content")).toBeVisible();
}

function noAgencyRequests(ctl) {
  expect(ctl.agencyHits).toEqual([]);
  expect(ctl.requests.filter((u) => AGENCY_HOSTS.test(u))).toEqual([]);
}

// ----- the entry point -----------------------------------------------------------------------------------------

test("only approved and reported hazards have a Prefilled report button", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  for (const id of [1, 2, 3, 6]) await expect(opener(page, id)).toHaveAccessibleName(`Prefilled report for hazard ${id}`);
  for (const id of [4, 5]) await expect(opener(page, id)).toHaveCount(0);
  expect(ctl.reportGets).toEqual([]); // nothing is loaded until the user asks
});

test("approving a hazard makes the button appear, rejecting makes it go", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(opener(page, 4)).toHaveCount(0);
  await item(page, 4).getByTestId("hazard-approve").click();
  await expect(opener(page, 4)).toBeVisible();
  await item(page, 4).getByTestId("hazard-reject").click();
  await expect(opener(page, 4)).toHaveCount(0);
});

// ----- open and close, focus -----------------------------------------------------------------------------------

test("the panel is an accessible modal dialog with the nothing-is-submitted banner and focus inside", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  const dialog = panel(page);
  await expect(dialog).toHaveAttribute("role", "dialog");
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  await expect(dialog).toHaveAccessibleName("Prefilled report for hazard 1");
  await expect(page.getByTestId("report-banner")).toHaveText(NOTHING_SUBMITTED);
  await expect(dialog).toBeFocused();
  await expect(page.getByRole("dialog")).toHaveCount(1);
  await expect(dialog).toHaveAttribute("data-loads", "1"); // (React StrictMode fetches twice in dev; one reply counts)
  noAgencyRequests(ctl);
});

test("loading state is shown until the report arrives", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.reportRespond = async (route) => {
    await held.promise;
    return route.fulfill({ status: 200, json: buildReport(ctl.items[0]) });
  };
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-loading")).toBeVisible();
  await expect(page.getByTestId("report-banner")).toBeVisible(); // the banner never waits for the data
  await expect(page.getByTestId("report-content")).toHaveCount(0);
  held.open();
  await expect(page.getByTestId("report-content")).toBeVisible();
  await expect(page.getByTestId("report-loading")).toHaveCount(0);
});

test("Close button closes the panel and gives focus back to the button that opened it", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 2);
  await page.getByTestId("report-close").click();
  await expect(panel(page)).toHaveCount(0);
  await expect(opener(page, 2)).toBeFocused();
});

test("Escape closes the panel and gives focus back", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await expect(opener(page, 1)).toBeFocused();
});

test("clicking the dark backdrop closes the panel, clicking inside does not", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-title").click();
  await expect(panel(page)).toBeVisible();
  await page.getByTestId("report-backdrop").click({ position: { x: 5, y: 5 } });
  await expect(panel(page)).toHaveCount(0);
});

test("Tab and Shift+Tab stay inside the dialog", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.keyboard.press("Tab");
  await expect(page.getByTestId("report-close")).toBeFocused(); // first control
  await page.keyboard.press("Shift+Tab");
  await expect(page.getByTestId("report-text")).toBeFocused(); // wraps to the last control
  await page.keyboard.press("Tab");
  await expect(page.getByTestId("report-close")).toBeFocused();
  // nothing outside the dialog ever gets focus
  for (let i = 0; i < 25; i++) {
    await page.keyboard.press("Tab");
    expect(await page.evaluate(() => !!document.activeElement.closest('[data-testid="report-panel"]'))).toBe(true);
  }
});

test("opened from a map popup, closing returns focus to the popup's button", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await pin(page, 1).click();
  const button = popup(page).getByTestId("hazard-report-open");
  await expect(button).toBeVisible();
  await button.click();
  await expect(panel(page)).toHaveAttribute("data-hazard-id", "1");
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await expect(button).toBeFocused();
});

test("closing while the report loads shows nothing stale when another report is opened", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.reportRespond = async (route, id) => {
    if (id === 1) {
      await held.promise;
      try {
        await route.fulfill({ status: 200, json: buildReport(ctl.items[0]) });
      } catch {
        // the page already gave up on this request
      }
      return;
    }
    return route.fulfill({ status: 200, json: buildReport(ctl.items.find((h) => h.id === id)) });
  };
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-loading")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await opener(page, 2).click();
  await expect(panel(page)).toHaveAttribute("data-hazard-id", "2");
  await expect(page.getByTestId("report-category")).toHaveText(MDOT_TOPICS["debris in lane"]);
  held.open();
  await expect(page.getByTestId("report-title")).toHaveText("Prefilled report for hazard 2");
  await expect(page.getByTestId("report-category")).toHaveText(MDOT_TOPICS["debris in lane"]);
});

// ----- content: VDOT, MDOT SHA, no agency ----------------------------------------------------------------------

test("a Virginia hazard defaults to VDOT: category, fields, unverified notice, official form link", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await expect(page.getByTestId("report-agency-select")).toHaveValue("VDOT");
  await expect(page.getByTestId("report-agency-source")).toHaveText("Automatic (from the hazard's location)");
  await expect(page.getByTestId("report-agency-source")).toHaveAttribute("data-source", "auto");
  await expect(page.getByTestId("report-agency-auto")).toHaveCount(0);
  await expect(page.getByTestId("report-category")).toHaveText("Pothole repair");
  await expect(page.getByTestId("report-unverified-notice")).toContainText("Unverified field names, check on the live form");
  await expect(page.getByTestId("report-fields")).toHaveAttribute("data-verified", "false");
  const rows = page.getByTestId("report-field");
  await expect(rows).toHaveCount(4);
  await expect(rows.first()).toContainText("Kind of repair or problem");
  await expect(rows.first()).toContainText("(required)");
  await expect(page.locator('[data-field-key="blocks_traffic"]')).toHaveAttribute("data-empty", "true");
  await expect(page.locator('[data-field-key="blocks_traffic"]').getByTestId("report-field-value")).toHaveText("Left to you");
  const link = page.getByTestId("report-open-form");
  await expect(link).toHaveAttribute("href", "https://my.vdot.virginia.gov/");
  await expect(link).toHaveAttribute("target", "_blank");
  await expect(link).toHaveAttribute("rel", "noopener noreferrer");
  await expect(link).toHaveText("Open official form");
  noAgencyRequests(ctl);
});

test("a Maryland hazard defaults to MDOT SHA: Topic, sub-topic note, no unverified notice", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 2);
  await expect(page.getByTestId("report-agency-select")).toHaveValue("MDOT SHA");
  await expect(page.getByTestId("report-category")).toHaveText("Debris, Graffiti, & Litter");
  await expect(page.getByTestId("report-sub-topic-note")).toContainText("Choose on the form");
  await expect(page.getByTestId("report-unverified-notice")).toHaveCount(0);
  await expect(page.getByTestId("report-fields")).toHaveAttribute("data-verified", "true");
  await expect(page.locator('[data-field-key="topic"]')).toContainText("(required)");
  await expect(page.locator('[data-field-key="sub_topic"]').getByTestId("report-field-value")).toHaveText("Left to you");
  await expect(page.locator('[data-field-key="direction"]').getByTestId("report-field-value")).toHaveText("E");
  await expect(page.getByTestId("report-open-form")).toHaveAttribute("href", "https://mdotsha.my.salesforce-sites.com/customercare");
  await expect(page.getByTestId("report-agency-note")).toContainText("Change the agency");
});

test("description, location with a map link, and the blurred photo are shown", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await expect(page.getByTestId("report-description")).toContainText("Road hazard: pothole. Severity: medium (2 of 3).");
  await expect(page.getByTestId("report-location")).toContainText("38.850000, -77.420000");
  await expect(page.getByTestId("report-location")).toContainText("heading E");
  const map = page.getByTestId("report-map-link");
  await expect(map).toHaveAttribute("href", /^https:\/\/www\.openstreetmap\.org\/\?mlat=38\.850000&mlon=-77\.420000/);
  await expect(map).toHaveAttribute("target", "_blank");
  await expect(map).toHaveAttribute("rel", "noopener noreferrer");
  const photo = page.getByTestId("report-photo");
  await expect(photo).toHaveAttribute("src", "/api/hazards/1/photo");
  await expect.poll(() => photo.evaluate((img) => img.complete && img.naturalWidth > 0)).toBe(true);
});

test("a hazard without a photo, and a photo that fails to load, show a message instead of a broken image", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.items[1].photo_url = null;
  ctl.photoStatus[1] = 404;
  await openMap(page, ctl);
  await openReport(page, 2);
  await expect(page.getByTestId("report-photo-missing")).toHaveText("No photo for this hazard.");
  await expect(page.getByTestId("report-download-photo")).toHaveCount(0);
  await page.getByTestId("report-close").click();
  await openReport(page, 1);
  await expect(page.getByTestId("report-photo-missing")).toHaveText("Photo unavailable.");
});

test("a hazard outside Virginia and Maryland shows 'no agency': no category, no form link, copy and download remain", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 3);
  await expect(page.getByTestId("report-agency-select")).toHaveValue("none");
  await expect(page.getByTestId("report-no-agency")).toContainText("No agency for this hazard.");
  await expect(page.getByTestId("report-no-agency")).toContainText("not in Virginia or Maryland");
  await expect(page.getByTestId("report-category")).toHaveCount(0);
  await expect(page.getByTestId("report-fields")).toHaveCount(0);
  await expect(page.getByTestId("report-open-form")).toHaveCount(0);
  await expect(page.getByTestId("report-unverified-notice")).toHaveCount(0);
  await expect(page.getByTestId("report-copy")).toBeEnabled();
  await expect(page.getByTestId("report-download-text")).toBeVisible();
  await expect(page.getByTestId("report-agency-source")).toHaveText("Automatic (from the hazard's location)");
  await expect(page.getByTestId("report-banner")).toHaveText(NOTHING_SUBMITTED);
});

// ----- agency select -------------------------------------------------------------------------------------------

test("the agency select offers VDOT, MDOT SHA and No agency", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  const options = page.getByTestId("report-agency-select").locator("option");
  await expect(options).toHaveText(["VDOT", "MDOT SHA", "No agency"]);
  await expect(page.getByTestId("report-agency-select")).toHaveAccessibleName("Agency");
});

test("changing the agency PATCHes {agency}, reloads the report and shows 'Chosen by you'", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("MDOT SHA");
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  expect(ctl.patches).toEqual([{ id: 1, body: { agency: "MDOT SHA" } }]);
  await expect(panel(page)).toHaveAttribute("data-loads", "2");
  await expect(page.getByTestId("report-agency-select")).toHaveValue("MDOT SHA");
  await expect(page.getByTestId("report-agency-source")).toHaveText("Chosen by you");
  await expect(page.getByTestId("report-agency-source")).toHaveAttribute("data-source", "user");
  await expect(page.getByTestId("report-unverified-notice")).toHaveCount(0);
  await expect(page.getByTestId("report-open-form")).toHaveAttribute("href", /mdotsha\.my\.salesforce-sites\.com/);
  await expect(page.getByTestId("report-agency-auto")).toHaveText("Use automatic (VDOT)");
  noAgencyRequests(ctl);
});

test("choosing No agency sends 'none'; Use automatic sends null and brings the default back", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 2);
  await page.getByTestId("report-agency-select").selectOption("none");
  await expect(page.getByTestId("report-no-agency")).toBeVisible();
  await expect(page.getByTestId("report-agency-source")).toHaveText("Chosen by you");
  await expect(page.getByTestId("report-no-agency")).toContainText("No agency for this hazard.");
  expect(ctl.patches.at(-1)).toEqual({ id: 2, body: { agency: "none" } });
  await page.getByTestId("report-agency-auto").click();
  await expect(page.getByTestId("report-agency-source")).toHaveText("Automatic (from the hazard's location)");
  await expect(page.getByTestId("report-agency-select")).toHaveValue("MDOT SHA");
  expect(ctl.patches.at(-1)).toEqual({ id: 2, body: { agency: null } });
  await expect(page.getByTestId("report-agency-auto")).toHaveCount(0);
});

test("a user can pick VDOT for a hazard outside both states", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 3);
  await page.getByTestId("report-agency-select").selectOption("VDOT");
  await expect(page.getByTestId("report-category")).toHaveText("Pothole repair");
  await expect(page.getByTestId("report-agency-auto")).toHaveText("Use automatic (no agency)");
});

test("while the agency change is saved the select is disabled and only one request is sent", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.patchRespond = async (route, id, body) => {
    await held.promise;
    const result = applyPatch(ctl.items.find((h) => h.id === id), body);
    return route.fulfill({ status: 200, json: result.hazard });
  };
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("MDOT SHA");
  await expect(page.getByTestId("report-agency-select")).toBeDisabled();
  await expect(page.getByTestId("report-updating")).toBeVisible();
  await expect(page.getByTestId("report-mark-reported")).toBeDisabled();
  await expect(page.getByTestId("report-panel")).toHaveAttribute("aria-busy", "true");
  expect(ctl.patches).toHaveLength(1);
  held.open();
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  await expect(page.getByTestId("report-agency-select")).toBeEnabled();
  await expect(page.getByTestId("report-updating")).toHaveCount(0);
  expect(ctl.patches).toHaveLength(1);
});

test("re-opening the report after an agency change still shows the choice the user made", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("none");
  await expect(page.getByTestId("report-no-agency")).toBeVisible();
  await page.getByTestId("report-close").click();
  await openReport(page, 1); // re-open: the report is fetched again and still says the user's choice
  await expect(page.getByTestId("report-agency-select")).toHaveValue("none");
  await expect(page.getByTestId("report-agency-source")).toHaveText("Chosen by you");
});

test("a refused agency change (422) shows the message and keeps the old agency", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.patchRespond = (route) => route.fulfill({ status: 422, json: { detail: "agency must be one of VDOT, MDOT SHA, none, auto or null" } });
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("MDOT SHA");
  await expect(page.getByTestId("report-agency-error")).toHaveText("agency must be one of VDOT, MDOT SHA, none, auto or null");
  await expect(page.getByTestId("report-agency-select")).toHaveValue("VDOT");
  await expect(page.getByTestId("report-category")).toHaveText("Pothole repair");
  await expect(page.getByTestId("hazard-review-error")).toHaveCount(0); // the error stays in the panel
  await expect(panel(page)).toHaveAttribute("data-loads", "1");
});

test("losing the connection while changing the agency shows an error and keeps the old agency", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.patchRespond = (route) => route.abort();
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("none");
  await expect(page.getByTestId("report-agency-error")).toHaveText("Could not reach the server.");
  await expect(page.getByTestId("report-agency-select")).toHaveValue("VDOT");
  await expect(page.getByTestId("report-agency-select")).toBeEnabled();
});

// ----- copy ----------------------------------------------------------------------------------------------------

test("Copy report puts the whole report text on the clipboard and shows 'Copied'", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await expect(page.getByTestId("report-copy-status")).toHaveText("");
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
  await expect(page.getByTestId("report-copy-status")).toHaveAttribute("data-state", "copied");
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  const text = buildReport(ctl.items[0]).text;
  expect(copied.split(/\r?\n/).join("\n")).toBe(text); // (Windows clipboards turn LF into CRLF)
  await expect(page.getByTestId("report-text")).toHaveValue(text);
});

test("when navigator.clipboard is missing the execCommand fallback copies the text", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    window.__copied = [];
    document.execCommand = (command) => {
      if (command === "copy") window.__copied.push(document.getSelection().toString() || document.activeElement?.value || "");
      return command === "copy";
    };
  });
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
  const copied = await page.evaluate(() => window.__copied);
  expect(copied).toHaveLength(1);
  expect(copied[0]).toBe(buildReport(ctl.items[0]).text);
  await expect(page.getByTestId("report-copy")).toBeFocused(); // focus is given back
});

test("when copying fails everywhere a failure message is shown and the text stays selectable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: () => Promise.reject(new Error("denied")) },
      configurable: true,
    });
    document.execCommand = () => false;
  });
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toContainText("Could not copy automatically");
  await expect(page.getByTestId("report-copy-status")).toHaveAttribute("data-state", "failed");
  await expect(page.getByTestId("report-copy-status")).not.toContainText("Copied");
  await expect(page.getByTestId("report-text")).toHaveValue(buildReport(ctl.items[0]).text);
  await expect(page.getByTestId("report-copy")).toBeEnabled();
});

test("if the clipboard API rejects, the fallback still gets a chance", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: () => Promise.reject(new Error("denied")) },
      configurable: true,
    });
    document.execCommand = (command) => command === "copy";
  });
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
});

test("a double click on Copy report writes once", async ({ page }) => {
  await page.addInitScript(() => {
    window.__writes = 0;
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: () => {
          window.__writes += 1;
          return window.__hold;
        },
      },
      configurable: true,
    });
    window.__hold = new Promise((resolve) => (window.__release = resolve));
  });
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-copy").dblclick();
  await expect(page.getByTestId("report-copy")).toBeDisabled();
  expect(await page.evaluate(() => window.__writes)).toBe(1);
  await page.evaluate(() => window.__release());
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
});

// ----- downloads and the official form -------------------------------------------------------------------------

test("download links point at the text report and the photo download", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 2);
  const text = page.getByTestId("report-download-text");
  await expect(text).toHaveAttribute("href", "/api/hazards/2/report?format=txt");
  await expect(text).toHaveAttribute("download", "");
  await expect(text).toHaveText("Download report");
  const photo = page.getByTestId("report-download-photo");
  await expect(photo).toHaveAttribute("href", "/api/hazards/2/photo?download=1");
  await expect(photo).toHaveAttribute("download", "");
  await expect(photo).toHaveText("Download photo");
});

// ----- mark as reported ----------------------------------------------------------------------------------------

test("Mark as reported PATCHes status reported, updates the list and pin, and locks the button", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Mark as reported");
  await page.getByTestId("report-mark-reported").click();
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Marked as reported");
  await expect(page.getByTestId("report-mark-reported")).toBeDisabled();
  await expect(page.getByTestId("report-reported-note")).toBeVisible();
  expect(ctl.patches).toEqual([{ id: 1, body: { status: "reported" } }]);
  await expect(panel(page)).toHaveAttribute("data-loads", "2");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Reported");
  await expect(pin(page, 1)).toHaveAttribute("data-status", "reported");
  await expect(page.getByTestId("report-content")).toHaveAttribute("data-submitted", "false");
  await expect(page.getByTestId("report-banner")).toHaveText(NOTHING_SUBMITTED);
  noAgencyRequests(ctl);
});

test("an already reported hazard opens with the button locked", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 6);
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Marked as reported");
  await expect(page.getByTestId("report-mark-reported")).toBeDisabled();
  await expect(page.getByTestId("report-reported-note")).toBeVisible();
});

test("a double click on Mark as reported sends one request", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.patchRespond = async (route, id, body) => {
    await held.promise;
    const result = applyPatch(ctl.items.find((h) => h.id === id), body);
    return route.fulfill({ status: 200, json: result.hazard });
  };
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-mark-reported").dblclick();
  await expect(page.getByTestId("report-mark-reported")).toBeDisabled();
  expect(ctl.patches).toHaveLength(1);
  held.open();
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Marked as reported");
  expect(ctl.patches).toHaveLength(1);
});

test("a refused Mark as reported (409) shows the message, reloads the report and keeps the status", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.patchRespond = (route) =>
    route.fulfill({ status: 409, json: { detail: "only an approved hazard can be marked as reported; this one is rejected" } });
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-mark-reported").click();
  await expect(page.getByTestId("report-reported-error")).toContainText("only an approved hazard can be marked as reported");
  await expect(item(page, 1).getByTestId("hazard-list-status")).toHaveText("Approved");
  await expect(panel(page)).toHaveAttribute("data-loads", "2"); // fetched again because it may be out of date
  await expect(page.getByTestId("report-mark-reported")).toBeEnabled();
});

// ----- load errors ---------------------------------------------------------------------------------------------

test("409 from the report shows the server's message and no retry button", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route) =>
    route.fulfill({ status: 409, json: { detail: "A report is only available for approved hazards, and this hazard is rejected. Approve it first." } });
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-error")).toHaveText(/only available for approved hazards/);
  await expect(page.getByTestId("report-error")).toHaveAttribute("data-status", "409");
  await expect(page.getByTestId("report-retry")).toHaveCount(0);
  await expect(page.getByTestId("report-content")).toHaveCount(0);
  await expect(page.getByTestId("report-banner")).toHaveText(NOTHING_SUBMITTED);
  await page.getByTestId("report-close").click(); // a failed panel still closes
  await expect(panel(page)).toHaveCount(0);
});

test("404 from the report says the hazard was not found", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route) => route.fulfill({ status: 404, json: { detail: "Hazard not found" } });
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-error")).toHaveText("Hazard not found");
  await expect(page.getByTestId("report-retry")).toHaveCount(0);
});

test("500 shows the status and Try again loads the report", async ({ page }) => {
  const ctl = await mockApi(page);
  let failing = true;
  ctl.reportRespond = (route) =>
    failing ? route.fulfill({ status: 500, body: "boom" }) : route.fulfill({ status: 200, json: buildReport(ctl.items[0]) });
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-error")).toHaveText("Could not load the report (500).");
  failing = false;
  await page.getByTestId("report-retry").click();
  await expect(page.getByTestId("report-content")).toBeVisible();
  await expect(page.getByTestId("report-error")).toHaveCount(0);
});

test("a lost connection shows 'Could not reach the server.'", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route) => route.abort();
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-error")).toHaveText("Could not reach the server.");
  await expect(page.getByTestId("report-retry")).toBeVisible();
});

test("a malformed report body is an error, not a crash", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route) => route.fulfill({ status: 200, json: { hazard_id: 99 } });
  await openMap(page, ctl);
  await opener(page, 1).click();
  await expect(page.getByTestId("report-error")).toHaveText("Unexpected response from the server.");
});

// ----- nothing is ever sent to an agency -----------------------------------------------------------------------

test("a full walk through the report makes no request to any VDOT, Maryland or Salesforce host", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-agency-select").selectOption("MDOT SHA");
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
  await page.getByTestId("report-mark-reported").click();
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Marked as reported");
  await page.keyboard.press("Escape");
  await openReport(page, 2);
  await page.getByTestId("report-agency-select").selectOption("VDOT");
  await expect(page.getByTestId("report-unverified-notice")).toBeVisible();
  noAgencyRequests(ctl);
  // every PATCH body only ever contains our own fields
  for (const patch of ctl.patches) expect(Object.keys(patch.body).every((k) => ["agency", "status"].includes(k))).toBe(true);
  // the only non-local requests are the map tiles
  const external = ctl.requests.filter((u) => !/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//.test(u) && !u.startsWith("data:") && !u.startsWith("blob:"));
  expect(external.every((u) => /openstreetmap\.org/.test(u))).toBe(true);
});

test("the report buttons are plain buttons and links, not a form that could post anywhere", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await expect(panel(page).locator("form")).toHaveCount(0);
  await expect(panel(page).locator("button:not([type=button])")).toHaveCount(0);
  const hrefs = await panel(page).locator("a").evaluateAll((links) => links.map((a) => a.getAttribute("href")));
  expect(hrefs.every((h) => /^(\/api\/|https:\/\/(www\.openstreetmap\.org|my\.vdot\.virginia\.gov|mdotsha\.my\.salesforce-sites\.com)\/)/.test(h))).toBe(true);
});

// ----- keyboard focus, Escape on a select, inert background ----------------------------------------------------

const agencySelect = (page) => page.getByTestId("report-agency-select");

test("after a keyboard agency change focus comes back to the select once the report has reloaded", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.reportRespond = async (route, id) => {
    if (ctl.patches.length > 0) await held.promise; // hold only the reload that follows the change
    return route.fulfill({ status: 200, json: buildReport(ctl.items.find((h) => h.id === id)) });
  };
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).focus();
  await page.keyboard.press("ArrowDown"); // VDOT -> MDOT SHA
  await expect(page.getByTestId("report-updating")).toBeVisible();
  await expect(agencySelect(page)).toBeDisabled(); // held: still saving or reloading
  expect(ctl.patches).toEqual([{ id: 1, body: { agency: "MDOT SHA" } }]);
  held.open();
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  await expect(agencySelect(page)).toBeEnabled();
  await expect(agencySelect(page)).toBeFocused();
  await page.keyboard.press("Tab"); // the next Tab moves on from the select, not from <body>
  await expect(page.getByTestId("report-agency-auto")).toBeFocused();
});

test("after Use automatic (which then disappears) focus goes to the agency select", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).selectOption("MDOT SHA");
  await expect(page.getByTestId("report-agency-auto")).toBeVisible();
  await page.getByTestId("report-agency-auto").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("report-agency-source")).toHaveText("Automatic (from the hazard's location)");
  await expect(page.getByTestId("report-agency-auto")).toHaveCount(0);
  await expect(agencySelect(page)).toBeFocused();
  expect(ctl.patches.at(-1)).toEqual({ id: 1, body: { agency: null } });
});

test("after Mark as reported (a keyboard press) focus goes to the confirmation, not <body>", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-mark-reported").focus();
  await page.keyboard.press("Space");
  await expect(page.getByTestId("report-mark-reported")).toHaveText("Marked as reported");
  await expect(page.getByTestId("report-reported-note")).toBeFocused();
  await page.keyboard.press("Tab"); // moves on inside the dialog
  expect(await page.evaluate(() => !!document.activeElement.closest('[data-testid="report-panel"]'))).toBe(true);
  await expect(page.getByTestId("report-reported-note")).not.toBeFocused();
});

test("after a refused Mark as reported focus returns to that button", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.patchRespond = (route) =>
    route.fulfill({ status: 409, json: { detail: "only an approved hazard can be marked as reported; this one is rejected" } });
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-mark-reported").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("report-reported-error")).toBeVisible();
  await expect(panel(page)).toHaveAttribute("data-loads", "2");
  await expect(page.getByTestId("report-mark-reported")).toBeFocused();
});

test("after a refused agency change (no reload) focus returns to the select", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.patchRespond = (route) => route.fulfill({ status: 422, json: { detail: "agency must be one of VDOT, MDOT SHA, none, auto or null" } });
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("report-agency-error")).toBeVisible();
  await expect(agencySelect(page)).toBeEnabled();
  await expect(agencySelect(page)).toBeFocused();
});

test("focus is not pulled back when the user has already moved it elsewhere in the dialog", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.patchRespond = async (route, id, body) => {
    await held.promise;
    return route.fulfill({ status: 200, json: applyPatch(ctl.items.find((h) => h.id === id), body).hazard });
  };
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).focus();
  await page.keyboard.press("ArrowDown");
  await expect(agencySelect(page)).toBeDisabled();
  await page.getByTestId("report-text").focus();
  held.open();
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  await expect(agencySelect(page)).toBeEnabled();
  await expect(page.getByTestId("report-text")).toBeFocused();
});

test("if the reload fails after a change, focus goes to the dialog itself", async ({ page }) => {
  const ctl = await mockApi(page);
  ctl.reportRespond = (route, id) =>
    ctl.patches.length > 0
      ? route.fulfill({ status: 500, body: "boom" })
      : route.fulfill({ status: 200, json: buildReport(ctl.items.find((h) => h.id === id)) });
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("report-error")).toHaveText("Could not load the report (500).");
  await expect(panel(page)).toBeFocused();
});

test("Escape on the focused agency select does not close the dialog; Escape elsewhere does", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).focus();
  await page.keyboard.press("Escape");
  await expect(panel(page)).toBeVisible();
  await expect(agencySelect(page)).toBeFocused();
  await page.getByTestId("report-copy").focus();
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await expect(opener(page, 1)).toBeFocused();
});

test("the page behind the dialog is inert while it is open and live again after it closes", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await expect(page.locator("#root")).not.toHaveAttribute("inert", "");
  await openReport(page, 1);
  await expect(page.locator("#root")).toHaveAttribute("inert", "");
  await expect(page.getByTestId("report-backdrop")).not.toHaveAttribute("inert", "");
  // the list behind it cannot be reached
  expect(
    await page.evaluate(() => document.querySelector('[data-testid="hazard-list-item"]').matches(":is([inert], [inert] *)")),
  ).toBe(true);
  await page.getByTestId("report-close").click();
  await expect(page.locator("#root")).not.toHaveAttribute("inert", "");
  await expect(opener(page, 1)).toBeFocused();
  await opener(page, 1).click(); // clickable again
  await expect(panel(page)).toBeVisible();
});

// ----- reload states -------------------------------------------------------------------------------------------

test("while the report reloads after an agency change the dialog is locked and then shows the new report", async ({ page }) => {
  const ctl = await mockApi(page);
  const held = gate();
  ctl.reportRespond = async (route, id) => {
    if (ctl.patches.length > 0) await held.promise;
    return route.fulfill({ status: 200, json: buildReport(ctl.items.find((h) => h.id === id)) });
  };
  await openMap(page, ctl);
  await openReport(page, 1);
  await agencySelect(page).selectOption("MDOT SHA");
  await expect(panel(page)).toHaveAttribute("aria-busy", "true");
  await expect(page.getByTestId("report-updating")).toBeVisible();
  await expect(agencySelect(page)).toBeDisabled();
  await expect(page.getByTestId("report-mark-reported")).toBeDisabled();
  await expect(page.getByTestId("report-copy")).toBeDisabled(); // the old text must not be copied
  expect(ctl.patches).toHaveLength(1);
  held.open();
  await expect(panel(page)).toHaveAttribute("aria-busy", "false");
  await expect(page.getByTestId("report-category")).toHaveText("Potholes / Holes / Sinkholes");
  await expect(page.getByTestId("report-text")).toHaveValue(/Agency: MDOT SHA/);
  await expect(page.getByTestId("report-copy")).toBeEnabled();
  await expect(agencySelect(page)).toBeEnabled();
});

test("the 'Copied' confirmation is cleared when the report is reloaded", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await openReport(page, 1);
  await page.getByTestId("report-copy").click();
  await expect(page.getByTestId("report-copy-status")).toHaveText("Copied");
  await agencySelect(page).selectOption("none");
  await expect(page.getByTestId("report-no-agency")).toBeVisible();
  await expect(page.getByTestId("report-copy-status")).toHaveText("");
  await expect(page.getByTestId("report-copy-status")).toHaveAttribute("data-state", "idle");
});

// ----- keyboard activation, the official form link -------------------------------------------------------------

test("the Prefilled report button opens with Enter and with Space", async ({ page }) => {
  const ctl = await mockApi(page);
  await openMap(page, ctl);
  await opener(page, 2).focus();
  await page.keyboard.press("Enter");
  await expect(panel(page)).toHaveAttribute("data-hazard-id", "2");
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await expect(opener(page, 2)).toBeFocused();
  await page.keyboard.press("Space");
  await expect(panel(page)).toHaveAttribute("data-hazard-id", "2");
  await expect(panel(page)).toBeFocused();
});

test("clicking Open official form opens the agency's own page and is the only request to an agency host", async ({ page, context }) => {
  const ctl = await mockApi(page);
  const hits = [];
  // the new tab is another page, so it is stubbed at the context level; nothing real is contacted
  await context.route(AGENCY_HOSTS, (route) => {
    hits.push(route.request().url());
    return route.fulfill({ status: 200, contentType: "text/html", body: "<!doctype html><title>stub</title>" });
  });
  await openMap(page, ctl);
  await openReport(page, 1);
  expect(hits).toEqual([]); // opening the report and reading it contacts nobody
  const [form] = await Promise.all([context.waitForEvent("page"), page.getByTestId("report-open-form").click()]);
  await form.waitForLoadState("domcontentloaded");
  expect(form.url()).toBe("https://my.vdot.virginia.gov/");
  expect(hits.filter((u) => !/favicon/.test(u))).toEqual(["https://my.vdot.virginia.gov/"]);
  expect(ctl.agencyHits).toEqual([]); // the RoadScout page itself never called an agency
  expect(ctl.patches).toEqual([]); // and following the link changed nothing in RoadScout
  await expect(panel(page)).toBeVisible();
  await form.close();
});
