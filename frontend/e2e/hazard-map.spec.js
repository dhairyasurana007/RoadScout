import { test, expect } from "@playwright/test";
import zlib from "node:zlib";

// UI behaviour only. Every API call is mocked with page.route, and so are the OpenStreetMap tiles: they
// are answered with a small PNG generated here, so no test ever reaches the real tile server.

// ----- a tiny PNG generated in the test (a flat grey 256x256 tile) ---------------------------------

function crc32(buf) {
  let c = ~0;
  for (const byte of buf) {
    c ^= byte;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

function chunk(type, data) {
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const out = Buffer.alloc(body.length + 8);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(crc32(body), body.length + 4);
  return out;
}

function makePng(width, height, [r, g, b]) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 2; // RGB
  const row = Buffer.concat([Buffer.from([0]), Buffer.from(Array(width).fill([r, g, b]).flat())]);
  const raw = Buffer.concat(Array(height).fill(row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const TILE = makePng(256, 256, [220, 225, 215]);
const PHOTO = makePng(16, 12, [90, 90, 100]); // stands in for a blurred photo

// ----- fixtures --------------------------------------------------------------------------------------

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
  photo_url: `/api/hazards/${id}/photo`,
  crop_url: null,
  ...patch,
});

const THREE = [
  hazard(1, { hazard_class: "pothole", severity: 3, status: "new", sightings: 4 }),
  hazard(2, { hazard_class: "debris in lane", severity: 1, status: "approved", sightings: 2 }),
  hazard(3, { hazard_class: "damaged guardrail", severity: 2, status: "reported", sightings: 1 }),
];
const many = (n) => Array.from({ length: n }, (_, i) => hazard(i + 1, { status: i % 2 ? "approved" : "new" }));

/**
 * Mock the hazards API and the tile server. Returns a controller:
 *   ctl.items      what GET /api/hazards returns (filtered by ?status= like the server)
 *   ctl.queries    the query string of every GET /api/hazards request (React's dev StrictMode loads twice on
 *                  mount and aborts the first, so tests call ctl.mark() once loaded and use ctl.recent())
 *   ctl.respond    optional async (route, query) => ... that replaces the default answer
 *   ctl.photoStatus  status for the photo route (200 by default)
 *   ctl.tiles      every tile URL asked for, ctl.strays any other OpenStreetMap URL, ctl.external other hosts
 */
async function mockMap(page, items = []) {
  const ctl = {
    items,
    queries: [],
    base: 0,
    mark: () => (ctl.base = ctl.queries.length),
    recent: () => ctl.queries.slice(ctl.base),
    respond: null, photoStatus: 200,
    tiles: [],
    strays: [],
    external: [],
  };

  page.on("request", (request) => {
    const url = new URL(request.url());
    if (!["localhost", "127.0.0.1"].includes(url.hostname)) ctl.external.push(request.url());
  });
  // Any OpenStreetMap request that is not a tile is recorded and refused; the tile route below wins for tiles.
  await page.route(/openstreetmap\.org/, (route) => {
    ctl.strays.push(route.request().url());
    return route.abort();
  });
  await page.route(/^https:\/\/tile\.openstreetmap\.org\/\d+\/\d+\/\d+\.png$/, (route) => {
    ctl.tiles.push(route.request().url());
    return route.fulfill({ status: 200, contentType: "image/png", body: TILE });
  });
  await page.route(
    (url) => url.pathname === "/api/hazards",
    async (route) => {
      const request = route.request();
      if (request.method() !== "GET") return route.fulfill({ status: 405, json: {} });
      const query = new URL(request.url()).search;
      ctl.queries.push(query);
      if (ctl.respond) return ctl.respond(route, query);
      const status = new URLSearchParams(query).get("status");
      const list = status ? ctl.items.filter((h) => h.status === status) : ctl.items;
      return route.fulfill({ status: 200, json: list });
    },
  );
  await page.route(
    (url) => /^\/api\/hazards\/\d+\/photo$/.test(url.pathname),
    (route) =>
      ctl.photoStatus === 200
        ? route.fulfill({ status: 200, contentType: "image/png", body: PHOTO })
        : route.fulfill({ status: ctl.photoStatus, json: { detail: "Photo not found" } }),
  );
  return ctl;
}

/** A promise you resolve by hand, to hold a mocked response. */
function gate() {
  let open;
  const promise = new Promise((resolve) => (open = resolve));
  return { promise, open };
}

const pins = (page) => page.getByTestId("hazard-pin");
const pin = (page, id) => page.locator(`[data-testid="hazard-pin"][data-hazard-id="${id}"]`);
const items = (page) => page.getByTestId("hazard-list-item");
const item = (page, id) => page.locator(`[data-testid="hazard-list-item"][data-hazard-id="${id}"]`);
const mapBox = (page) => page.getByTestId("hazard-map");
const popup = (page) => page.locator(".leaflet-popup");

async function openMap(page, ctl, { hash = true } = {}) {
  await page.goto(hash ? "/#map" : "/");
  if (!hash) await page.getByTestId("tab-map").click();
  await expect(page.getByTestId("hazard-map-section")).toBeVisible();
  return ctl;
}

/** Is the element's box fully inside the map container's box? */
async function insideMap(page, locator) {
  const outer = await mapBox(page).boundingBox();
  const inner = await locator.boundingBox();
  if (!outer || !inner) return false;
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}

// ----- navigation and the map shell ----------------------------------------------------------------

test("the upload page is the default view and the map is a separate section", async ({ page }) => {
  await mockMap(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Upload a drive" })).toBeVisible();
  await expect(page.getByTestId("hazard-map-section")).toHaveCount(0);
  await expect(page.getByTestId("tab-drives")).toHaveAttribute("aria-pressed", "true");

  await page.getByTestId("tab-map").click();
  await expect(page.getByRole("heading", { name: "Hazard map" })).toBeVisible();
  await expect(mapBox(page)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Upload a drive" })).toHaveCount(0);
  await expect(page.getByTestId("tab-map")).toHaveAttribute("aria-pressed", "true");
  expect(new URL(page.url()).hash).toBe("#map");

  await page.getByTestId("tab-drives").click();
  await expect(page.getByRole("heading", { name: "Upload a drive" })).toBeVisible();
  await expect(page.getByTestId("hazard-map-section")).toHaveCount(0);
  expect(new URL(page.url()).hash).toBe("");
});

test("#map opens the map view directly", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  await expect(page.getByRole("heading", { name: "Upload a drive" })).toHaveCount(0);
});

test("the map shows OpenStreetMap tiles with the required attribution, all served by the mocked tile route", async ({
  page,
}) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(page.locator(".leaflet-control-attribution")).toContainText("© OpenStreetMap contributors");
  await expect(page.locator(".leaflet-control-attribution a[href*='openstreetmap.org/copyright']")).toBeVisible();
  await expect(page.locator(".leaflet-tile-loaded").first()).toBeAttached(); // the generated PNG decoded as a tile
  expect(ctl.tiles.length).toBeGreaterThan(0);
  for (const url of ctl.tiles) expect(url).toMatch(/^https:\/\/tile\.openstreetmap\.org\/\d+\/\d+\/\d+\.png$/);
  expect(ctl.strays).toEqual([]);
  expect([...ctl.external].sort()).toEqual([...ctl.tiles].sort()); // nothing but the (mocked) tiles left localhost
});

test("pins are not Leaflet's default marker images", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  await expect(page.locator("img.leaflet-marker-icon")).toHaveCount(0);
  await expect(page.locator(".leaflet-marker-icon.rs-pin-icon")).toHaveCount(3);
});

// ----- loading, empty and error states --------------------------------------------------------------

test("shows a loading state until the hazards arrive", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  const hold = gate();
  ctl.respond = async (route) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: THREE });
  };
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-loading")).toBeVisible();
  await expect(page.getByTestId("hazard-map-empty")).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-error")).toHaveCount(0);
  await expect(pins(page)).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-refresh")).toBeDisabled();
  hold.open();
  await expect(page.getByTestId("hazard-map-loading")).toHaveCount(0);
  await expect(pins(page)).toHaveCount(3);
  await expect(page.getByTestId("hazard-map-refresh")).toBeEnabled();
});

test("no hazards: says 'No hazards yet', has no pins and shows the Northern Virginia / Maryland default view", async ({
  page,
}) => {
  const ctl = await mockMap(page, []);
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-empty")).toHaveText("No hazards yet");
  await expect(pins(page)).toHaveCount(0);
  await expect(items(page)).toHaveCount(0);
  await expect(mapBox(page)).toHaveAttribute("data-map-zoom", "9");
  const lat = Number(await mapBox(page).getAttribute("data-map-lat"));
  const lon = Number(await mapBox(page).getAttribute("data-map-lon"));
  expect(lat).toBeGreaterThan(38.4);
  expect(lat).toBeLessThan(39.6);
  expect(lon).toBeGreaterThan(-78.2);
  expect(lon).toBeLessThan(-76.2);
  await expect(page.getByTestId("hazard-map-error")).toHaveCount(0);
});

test("a server error shows the message, no pins, and Refresh recovers", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  ctl.respond = (route) => route.fulfill({ status: 500, json: { detail: "database is locked" } });
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-error")).toContainText("database is locked");
  await expect(page.getByTestId("hazard-map-empty")).toHaveCount(0);
  await expect(pins(page)).toHaveCount(0);
  await expect(mapBox(page)).toBeVisible(); // the map itself stays usable

  ctl.respond = null;
  await page.getByTestId("hazard-map-refresh").click();
  await expect(page.getByTestId("hazard-map-error")).toHaveCount(0);
  await expect(pins(page)).toHaveCount(3);
});

test("an error without a detail falls back to the status code", async ({ page }) => {
  const ctl = await mockMap(page);
  ctl.respond = (route) => route.fulfill({ status: 503, body: "oops" });
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-error")).toContainText("503");
});

test("an unreachable server shows an error, and Refresh recovers", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  ctl.respond = (route) => route.abort();
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-error")).toHaveText("Could not reach the server.");
  await expect(pins(page)).toHaveCount(0);
  ctl.respond = null;
  await page.getByTestId("hazard-map-refresh").click();
  await expect(pins(page)).toHaveCount(3);
  await expect(page.getByTestId("hazard-map-error")).toHaveCount(0);
});

// ----- pins and list --------------------------------------------------------------------------------

test("one pin and one list item per hazard, matching ids, coloured and labelled by status and severity", async ({
  page,
}) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  await expect(items(page)).toHaveCount(3);
  await expect(page.getByTestId("hazard-map-count")).toHaveText("3 hazards");
  const pinIds = await pins(page).evaluateAll((els) => els.map((e) => e.dataset.hazardId));
  const itemIds = await items(page).evaluateAll((els) => els.map((e) => e.dataset.hazardId));
  expect(pinIds.sort()).toEqual(["1", "2", "3"]);
  expect(itemIds).toEqual(["1", "2", "3"]); // list order is the API order
  await expect(pin(page, 1)).toHaveAttribute("data-status", "new");
  await expect(pin(page, 2)).toHaveAttribute("data-status", "approved");
  await expect(pin(page, 3)).toHaveAttribute("data-status", "reported");
  await expect(pin(page, 1)).toHaveText("3"); // severity label
  await expect(pin(page, 2)).toHaveText("1");
  const colours = await pins(page).evaluateAll((els) => els.map((e) => getComputedStyle(e).backgroundColor));
  expect(new Set(colours).size).toBe(3); // three statuses, three colours
});

test("a list item shows photo thumbnail, class, severity, status, sightings and location", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  const first = item(page, 1);
  await expect(first.getByTestId("hazard-list-class")).toHaveText("pothole");
  await expect(first.getByTestId("hazard-list-severity")).toContainText("3 (High)");
  await expect(first.getByTestId("hazard-list-status")).toHaveText("New");
  await expect(first.getByTestId("hazard-list-sightings")).toHaveText("4 sightings");
  await expect(first.getByTestId("hazard-list-location")).toHaveText("38.85000, -77.41500");
  await expect(first.getByTestId("hazard-list-photo")).toHaveAttribute("src", "/api/hazards/1/photo");
  await expect.poll(() => first.getByTestId("hazard-list-photo").evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);
  await expect(item(page, 3).getByTestId("hazard-list-sightings")).toHaveText("1 sighting");
  await expect(item(page, 3).getByTestId("hazard-list-status")).toHaveText("Reported");
});

test("the singular count reads '1 hazard'", async ({ page }) => {
  const ctl = await mockMap(page, [THREE[0]]);
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-count")).toHaveText("1 hazard");
});

test("the default view fits all pins inside the map", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  for (const id of [1, 2, 3]) {
    await expect.poll(() => insideMap(page, pin(page, id))).toBe(true);
  }
  // fitted to the pins, not left at the regional default
  await expect(mapBox(page)).not.toHaveAttribute("data-map-zoom", "9");
});

test("a single hazard centres the map on its pin", async ({ page }) => {
  const ctl = await mockMap(page, [THREE[0]]);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(1);
  await expect(mapBox(page)).toHaveAttribute("data-map-lat", "38.85000");
  await expect(mapBox(page)).toHaveAttribute("data-map-lon", "-77.41500");
  await expect.poll(() => insideMap(page, pin(page, 1))).toBe(true);
});

test("a hazard without a position is listed but has no pin", async ({ page }) => {
  const ctl = await mockMap(page, [THREE[0], hazard(2, { lat: null, lon: null })]);
  await openMap(page, ctl);
  await expect(items(page)).toHaveCount(2);
  await expect(pins(page)).toHaveCount(1);
  await expect(item(page, 2).getByTestId("hazard-list-location")).toHaveText("no location");
  await item(page, 2).click(); // selecting it must not throw or open a popup
  await expect(item(page, 2)).toHaveAttribute("data-selected", "true");
  await expect(popup(page)).toHaveCount(0);
});

// ----- linking pins and list ------------------------------------------------------------------------

test("nothing is selected at first", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  await expect(page.locator('[data-testid="hazard-list-item"][data-selected="true"]')).toHaveCount(0);
  await expect(page.locator('[data-testid="hazard-pin"][data-selected="true"]')).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "");
  await expect(popup(page)).toHaveCount(0);
});

test("clicking a pin selects and highlights its list item and opens its popup", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await pin(page, 2).click();
  await expect(item(page, 2)).toHaveAttribute("data-selected", "true");
  await expect(item(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(item(page, 3)).toHaveAttribute("data-selected", "false");
  await expect(pin(page, 2)).toHaveAttribute("data-selected", "true");
  await expect(pin(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "2");
  await expect(popup(page).getByTestId("hazard-popup")).toHaveAttribute("data-hazard-id", "2");

  await pin(page, 3).click(); // moving the selection moves the highlight
  await expect(item(page, 3)).toHaveAttribute("data-selected", "true");
  await expect(item(page, 2)).toHaveAttribute("data-selected", "false");
  await expect(pin(page, 2)).toHaveAttribute("data-selected", "false");
  await expect(page.locator('[data-testid="hazard-list-item"][data-selected="true"]')).toHaveCount(1);
});

test("clicking a pin scrolls its list item into view", async ({ page }) => {
  const ctl = await mockMap(page, many(14));
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(14);
  const list = page.getByTestId("hazard-map-list");
  expect(await list.evaluate((el) => el.scrollHeight > el.clientHeight)).toBe(true); // the list does scroll
  const target = item(page, 14);
  await expect(target).not.toBeInViewport({ ratio: 0.5 }); // the last row starts out of sight
  expect(await list.evaluate((el) => el.scrollTop)).toBe(0);

  await pin(page, 14).click();
  await expect(target).toHaveAttribute("data-selected", "true");
  await expect(target).toBeInViewport({ ratio: 0.9 });
  expect(await list.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
});

test("clicking a list item pans and zooms to its pin and opens the popup", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(mapBox(page)).not.toHaveAttribute("data-map-zoom", "15");
  const before = Number(await mapBox(page).getAttribute("data-map-zoom"));
  expect(before).toBeLessThan(15);

  await item(page, 3).click();
  await expect(item(page, 3)).toHaveAttribute("data-selected", "true");
  await expect(pin(page, 3)).toHaveAttribute("data-selected", "true");
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "3");
  await expect.poll(async () => Number(await mapBox(page).getAttribute("data-map-zoom"))).toBeGreaterThanOrEqual(15);
  await expect(popup(page).getByTestId("hazard-popup")).toHaveAttribute("data-hazard-id", "3");
  await expect.poll(() => insideMap(page, pin(page, 3))).toBe(true);
  // zoomed in on hazard 3, so the others (about a kilometre away) are off the map
  await expect(pin(page, 3)).toBeInViewport();
  // the map is now centred close to hazard 3 (the popup may nudge it a little)
  const lat = Number(await mapBox(page).getAttribute("data-map-lat"));
  const lon = Number(await mapBox(page).getAttribute("data-map-lon"));
  expect(Math.abs(lat - THREE[2].lat)).toBeLessThan(0.01);
  expect(Math.abs(lon - THREE[2].lon)).toBeLessThan(0.01);
});

test("selecting a second list item moves the popup and the highlight", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await item(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup")).toHaveAttribute("data-hazard-id", "1");
  await item(page, 2).click();
  await expect(popup(page).getByTestId("hazard-popup")).toHaveAttribute("data-hazard-id", "2");
  await expect(popup(page)).toHaveCount(1);
  await expect(item(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(pin(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(pin(page, 2)).toHaveAttribute("data-selected", "true");
});

test("clicking the selected list item again re-centres on it after the map was moved", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await item(page, 1).click();
  await expect(pin(page, 1)).toBeInViewport();
  await page.getByRole("button", { name: "Zoom out" }).click();
  await page.getByRole("button", { name: "Zoom out" }).click();
  const zoomedOut = Number(await mapBox(page).getAttribute("data-map-zoom"));
  await item(page, 1).click();
  await expect.poll(async () => Number(await mapBox(page).getAttribute("data-map-zoom"))).toBeGreaterThanOrEqual(15);
  expect(zoomedOut).toBeLessThan(15);
  await expect(popup(page).getByTestId("hazard-popup")).toHaveAttribute("data-hazard-id", "1");
});

test("clicking an empty part of the map clears the selection", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await pin(page, 1).click();
  await expect(item(page, 1)).toHaveAttribute("data-selected", "true");
  const box = await mapBox(page).boundingBox();
  // the pins run along a diagonal; the middle of the top edge is clear of them and of the popup
  await page.mouse.click(box.x + box.width * 0.5, box.y + 12);
  await expect(item(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(pin(page, 1)).toHaveAttribute("data-selected", "false");
  await expect(popup(page)).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "");
});

// ----- popup -----------------------------------------------------------------------------------------

test("the popup shows class, severity, status, sightings and the blurred photo", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await pin(page, 1).click();
  const card = popup(page).getByTestId("hazard-popup");
  await expect(card.getByTestId("hazard-popup-class")).toHaveText("pothole");
  await expect(card.getByTestId("hazard-popup-severity")).toHaveText("Severity: 3 (High)");
  await expect(card.getByTestId("hazard-popup-status")).toHaveText("Status: New");
  await expect(card.getByTestId("hazard-popup-sightings")).toHaveText("Sightings: 4");
  const photo = card.getByTestId("hazard-popup-photo");
  await expect(photo).toHaveAttribute("src", "/api/hazards/1/photo");
  await expect.poll(() => photo.evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);

  await pin(page, 2).click();
  const second = popup(page).getByTestId("hazard-popup");
  await expect(second.getByTestId("hazard-popup-class")).toHaveText("debris in lane");
  await expect(second.getByTestId("hazard-popup-status")).toHaveText("Status: Approved");
  await expect(second.getByTestId("hazard-popup-sightings")).toHaveText("Sightings: 2");
});

test("only the hazard photo route is used for images (relative, same origin)", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  const photoRequests = [];
  page.on("request", (r) => {
    if (r.resourceType() === "image" && new URL(r.url()).pathname.startsWith("/api/")) photoRequests.push(r.url());
  });
  await openMap(page, ctl);
  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-photo")).toBeVisible();
  expect(photoRequests.length).toBeGreaterThan(0);
  for (const url of photoRequests) expect(new URL(url).pathname).toMatch(/^\/api\/hazards\/\d+\/photo$/);
});

test("a missing photo shows 'Photo unavailable' in the list and the popup", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  ctl.photoStatus = 404;
  await openMap(page, ctl);
  await expect(item(page, 1).getByTestId("hazard-list-photo-missing")).toHaveText("Photo unavailable");
  await expect(item(page, 1).getByTestId("hazard-list-photo")).toHaveCount(0);
  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-photo-missing")).toHaveText("Photo unavailable");
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText("pothole"); // the rest still shows
});

test("a hazard whose photo_url is null shows 'Photo unavailable' without requesting a photo", async ({ page }) => {
  const ctl = await mockMap(page, [hazard(1, { photo_url: null })]);
  const photoRequests = [];
  page.on("request", (r) => {
    if (r.url().includes("/photo")) photoRequests.push(r.url());
  });
  await openMap(page, ctl);
  await expect(item(page, 1).getByTestId("hazard-list-photo-missing")).toBeVisible();
  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText("pothole");
  await expect(popup(page).getByTestId("hazard-popup-photo")).toHaveCount(0);
  expect(photoRequests).toEqual([]);
});

test("hazard text is shown as text, never as markup", async ({ page }) => {
  const evil = "<b id='injected'>pothole</b><img src=x onerror='window.__pwned=1'>";
  const ctl = await mockMap(page, [hazard(1, { hazard_class: evil })]);
  await openMap(page, ctl);
  await pin(page, 1).click();
  await expect(popup(page).getByTestId("hazard-popup-class")).toHaveText(evil);
  await expect(item(page, 1).getByTestId("hazard-list-class")).toHaveText(evil);
  await expect(page.locator("#injected")).toHaveCount(0);
  expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
});

// ----- refresh and filter ----------------------------------------------------------------------------

test("Refresh reloads the hazards and shows the new ones", async ({ page }) => {
  const ctl = await mockMap(page, THREE.slice(0, 2));
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(2);
  ctl.mark();

  ctl.items = THREE;
  await page.getByTestId("hazard-map-refresh").click();
  await expect(pins(page)).toHaveCount(3);
  await expect(items(page)).toHaveCount(3);
  expect(ctl.recent()).toEqual([""]);

  ctl.items = [];
  await page.getByTestId("hazard-map-refresh").click();
  await expect(page.getByTestId("hazard-map-empty")).toHaveText("No hazards yet");
  await expect(pins(page)).toHaveCount(0);
  await expect(items(page)).toHaveCount(0);
});

test("Refresh is disabled while a request is in flight, so it cannot be double-clicked", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  ctl.mark();
  const hold = gate();
  ctl.respond = async (route) => {
    await hold.promise;
    return route.fulfill({ status: 200, json: THREE });
  };
  await page.getByTestId("hazard-map-refresh").click();
  await expect(page.getByTestId("hazard-map-refresh")).toBeDisabled();
  await expect(page.getByTestId("hazard-map-refresh")).toHaveText("Loading...");
  await expect.poll(() => ctl.recent().length).toBe(1);
  hold.open();
  await expect(page.getByTestId("hazard-map-refresh")).toBeEnabled();
  expect(ctl.recent().length).toBe(1);
});

test("a refresh keeps the selection and the view when the hazard is still there", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await pin(page, 2).click();
  await item(page, 3).click(); // zoomed in on hazard 3
  await expect.poll(async () => Number(await mapBox(page).getAttribute("data-map-zoom"))).toBeGreaterThanOrEqual(15);
  const zoom = await mapBox(page).getAttribute("data-map-zoom");

  await page.getByTestId("hazard-map-refresh").click();
  await expect(page.getByTestId("hazard-map-refresh")).toBeEnabled();
  await expect(pins(page)).toHaveCount(3);
  await expect(item(page, 3)).toHaveAttribute("data-selected", "true");
  await expect(pin(page, 3)).toHaveAttribute("data-selected", "true"); // the rebuilt pin is highlighted too
  await expect(mapBox(page)).toHaveAttribute("data-map-zoom", zoom);
});

test("a refresh drops the selection when the hazard is gone", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await item(page, 3).click();
  await expect(item(page, 3)).toHaveAttribute("data-selected", "true");
  ctl.items = THREE.slice(0, 2);
  await page.getByTestId("hazard-map-refresh").click();
  await expect(items(page)).toHaveCount(2);
  await expect(page.getByTestId("hazard-map-section")).toHaveAttribute("data-selected-hazard-id", "");
  await expect(page.locator('[data-testid="hazard-list-item"][data-selected="true"]')).toHaveCount(0);
});

test("the status filter asks the server for that status and shows only those hazards", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  await expect(page.getByTestId("hazard-map-filter")).toHaveValue("");

  await page.getByTestId("hazard-map-filter").selectOption("approved");
  await expect(pins(page)).toHaveCount(1);
  await expect(items(page)).toHaveCount(1);
  await expect(pin(page, 2)).toBeVisible();
  expect(ctl.queries.at(-1)).toBe("?status=approved");
  await expect.poll(() => insideMap(page, pin(page, 2))).toBe(true); // refitted to the single pin

  await page.getByTestId("hazard-map-filter").selectOption("possibly_fixed");
  await expect(pins(page)).toHaveCount(0);
  await expect(page.getByTestId("hazard-map-empty")).toHaveText("No hazards with status Possibly fixed");
  expect(ctl.queries.at(-1)).toBe("?status=possibly_fixed");

  await page.getByTestId("hazard-map-filter").selectOption("");
  await expect(pins(page)).toHaveCount(3);
  expect(ctl.queries.at(-1)).toBe("");
});

test("the filter offers every status the API accepts", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  const values = await page.getByTestId("hazard-map-filter").locator("option").evaluateAll((o) => o.map((x) => x.value));
  expect(values).toEqual(["", "new", "approved", "rejected", "reported", "possibly_fixed"]);
});

test("Refresh keeps the chosen filter", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await page.getByTestId("hazard-map-filter").selectOption("reported");
  await expect(pins(page)).toHaveCount(1);
  ctl.mark();
  await page.getByTestId("hazard-map-refresh").click();
  await expect.poll(() => ctl.recent().length).toBe(1);
  expect(ctl.recent()).toEqual(["?status=reported"]);
  await expect(pins(page)).toHaveCount(1);
});

test("an older reply that arrives late never overwrites a newer filter", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  const slow = gate();
  ctl.respond = async (route, query) => {
    const status = new URLSearchParams(query).get("status");
    if (!status) await slow.promise; // the unfiltered request is held
    const list = status ? THREE.filter((h) => h.status === status) : THREE;
    return route.fulfill({ status: 200, json: list }).catch(() => {}); // aborted by the page: nothing to answer
  };
  await openMap(page, ctl);
  await expect(page.getByTestId("hazard-map-loading")).toBeVisible();
  await page.getByTestId("hazard-map-filter").selectOption("approved");
  await expect(pins(page)).toHaveCount(1);
  slow.open(); // the stale full list is released only now
  await expect(pins(page)).toHaveCount(1);
  await expect(items(page)).toHaveCount(1);
  await expect(item(page, 2)).toBeVisible();
  await expect(page.getByTestId("hazard-map-loading")).toHaveCount(0);
});

// ----- hygiene ---------------------------------------------------------------------------------------

test("the page only reads (GET) and never contacts anything but localhost and the mocked tile server", async ({
  page,
}) => {
  const ctl = await mockMap(page, THREE);
  const methods = [];
  page.on("request", (r) => {
    if (new URL(r.url()).pathname.startsWith("/api/")) methods.push(r.method());
  });
  await openMap(page, ctl);
  await pin(page, 1).click();
  await item(page, 2).click();
  await page.getByTestId("hazard-map-refresh").click();
  await expect(pins(page)).toHaveCount(3);
  expect(methods.length).toBeGreaterThan(0);
  expect(new Set(methods)).toEqual(new Set(["GET"]));
  expect(ctl.strays).toEqual([]);
  expect([...ctl.external].sort()).toEqual([...ctl.tiles].sort());
});

test("leaving the map view and coming back loads the hazards again", async ({ page }) => {
  const ctl = await mockMap(page, THREE);
  await openMap(page, ctl);
  await expect(pins(page)).toHaveCount(3);
  ctl.mark();
  await page.getByTestId("tab-drives").click();
  await expect(page.getByTestId("hazard-map")).toHaveCount(0);
  ctl.items = THREE.slice(0, 1);
  await page.getByTestId("tab-map").click();
  await expect(pins(page)).toHaveCount(1);
  // StrictMode mounts twice in dev, so one or two requests; either way all unfiltered, and the new data shows.
  expect(ctl.recent().length).toBeGreaterThanOrEqual(1);
  expect(new Set(ctl.recent())).toEqual(new Set([""]));
});
