import { test, expect } from "@playwright/test";

// Tiny placeholder files: the API is mocked, so contents do not matter and no real footage is used.
const video = { name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("not-a-real-video") };
const gpx = {
  name: "track.gpx",
  mimeType: "application/gpx+xml",
  buffer: Buffer.from('<gpx><trk><trkseg><trkpt lat="38.8" lon="-77.4"/></trkseg></trk></gpx>'),
};

const createdDrive = {
  id: "abc123",
  status: "uploaded",
  started_at: "2026-10-20T12:30:00+00:00",
  started_at_source: "video_metadata",
  started_at_fell_back: false,
  clock_offset_seconds: 2.5,
};

async function fillFiles(page) {
  await page.getByLabel("Video file").setInputFiles(video);
  await page.getByLabel("GPX file").setInputFiles(gpx);
}

test("upload form renders and keeps the hello-world heading", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "RoadScout" })).toBeVisible();
  await expect(page.getByLabel("Video file")).toBeVisible();
  await expect(page.getByLabel("GPX file")).toBeVisible();
  await expect(page.getByLabel("Clock offset (seconds)")).toHaveValue("0");
  await expect(page.getByRole("button", { name: "Upload drive" })).toBeEnabled();
});

test("submitting without files shows validation errors and sends nothing", async ({ page }) => {
  let requests = 0;
  await page.route("**/api/drives", (route) => {
    requests += 1;
    return route.fulfill({ status: 201, json: createdDrive });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Choose a video file.");

  await page.getByLabel("Video file").setInputFiles(video);
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Choose a GPX file.");
  expect(requests).toBe(0);
});

test("a non-numeric clock offset is rejected before upload", async ({ page }) => {
  await page.goto("/");
  await fillFiles(page);
  await page.getByLabel("Clock offset (seconds)").fill("abc");
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toContainText("Clock offset must be a number");
});

test("successful upload posts the form fields and shows the drive status", async ({ page }) => {
  let posted = "";
  await page.route("**/api/drives", async (route) => {
    posted = route.request().postData() ?? "";
    await route.fulfill({ status: 201, json: createdDrive });
  });
  await page.goto("/");
  await fillFiles(page);
  await page.getByLabel("Clock offset (seconds)").fill("2.5");
  await page.getByRole("button", { name: "Upload drive" }).click();

  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");
  await expect(page.getByTestId("drive-id")).toHaveText("abc123");
  await expect(page.getByTestId("drive-offset")).toHaveText("2.5 s");
  await expect(page.getByTestId("drive-started-at")).toContainText("2026-10-20T12:30:00");
  await expect(page.getByRole("alert")).toHaveCount(0);

  expect(posted).toContain('name="video"; filename="clip.mp4"');
  expect(posted).toContain('name="gpx"; filename="track.gpx"');
  expect(posted).toContain('name="clock_offset_seconds"');
  expect(posted).toContain("2.5");
});

test("shows a note when the start time was not in the video metadata", async ({ page }) => {
  await page.route("**/api/drives", (route) =>
    route.fulfill({
      status: 201,
      json: { ...createdDrive, started_at_source: "gpx_start", started_at_fell_back: true },
    }),
  );
  await page.goto("/");
  await fillFiles(page);
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByTestId("drive-started-at")).toContainText("not in video metadata");
});

test("a server rejection is shown and no drive status appears", async ({ page }) => {
  await page.route("**/api/drives", (route) =>
    route.fulfill({ status: 400, json: { detail: "File is not a readable video" } }),
  );
  await page.goto("/");
  await fillFiles(page);
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("File is not a readable video");
  await expect(page.getByTestId("drive-status")).toHaveCount(0);
});

test("a network failure shows an error", async ({ page }) => {
  await page.route("**/api/drives", (route) => route.abort());
  await page.goto("/");
  await fillFiles(page);
  await page.getByRole("button", { name: "Upload drive" }).click();
  await expect(page.getByRole("alert")).toHaveText("Could not reach the server.");
});

function offsetField(postData) {
  const match = /name="clock_offset_seconds"\r?\n\r?\n([^\r\n]*)/.exec(postData ?? "");
  return match ? match[1] : null;
}

const upload = (page) => page.getByRole("button", { name: /Upload drive|Uploading/ });

test("a failed second upload clears the previous drive panel", async ({ page }) => {
  let call = 0;
  await page.route("**/api/drives", (route) => {
    call += 1;
    return call === 1
      ? route.fulfill({ status: 201, json: createdDrive })
      : route.fulfill({ status: 400, json: { detail: "File is not a readable video" } });
  });
  await page.goto("/");
  await fillFiles(page);
  await upload(page).click();
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");

  await upload(page).click();
  await expect(page.getByRole("alert")).toHaveText("File is not a readable video");
  await expect(page.getByTestId("drive-status")).toHaveCount(0);
  await expect(page.getByText("Drive created")).toHaveCount(0);
});

test("a second successful upload replaces the panel and a retry clears the error", async ({ page }) => {
  let call = 0;
  await page.route("**/api/drives", (route) => {
    call += 1;
    if (call === 1) return route.fulfill({ status: 400, json: { detail: "File is not a readable video" } });
    return route.fulfill({
      status: 201,
      json: { ...createdDrive, id: `drive-${call}` },
    });
  });
  await page.goto("/");
  await fillFiles(page);
  await upload(page).click();
  await expect(page.getByRole("alert")).toBeVisible();

  await upload(page).click();
  await expect(page.getByTestId("drive-id")).toHaveText("drive-2");
  await expect(page.getByRole("alert")).toHaveCount(0);

  await upload(page).click();
  await expect(page.getByTestId("drive-id")).toHaveText("drive-3");
  await expect(page.getByTestId("drive-id")).toHaveCount(1);
});

test("the button is disabled while uploading and a double click sends one request", async ({ page }) => {
  let requests = 0;
  await page.route("**/api/drives", async (route) => {
    requests += 1;
    await new Promise((resolve) => setTimeout(resolve, 1500));
    await route.fulfill({ status: 201, json: createdDrive });
  });
  await page.goto("/");
  await fillFiles(page);
  await page.getByRole("button", { name: "Upload drive" }).dblclick();

  const busy = page.getByRole("button", { name: "Uploading..." });
  await expect(busy).toBeVisible();
  await expect(busy).toBeDisabled();
  await expect(page.getByTestId("drive-status")).toHaveText("uploaded");
  await expect(page.getByRole("button", { name: "Upload drive" })).toBeEnabled();
  expect(requests).toBe(1);
});

test("FastAPI 422 validation errors are shown as a joined message", async ({ page }) => {
  await page.route("**/api/drives", (route) =>
    route.fulfill({
      status: 422,
      json: { detail: [{ msg: "Field required" }, { msg: "Input should be a valid number" }] },
    }),
  );
  await page.goto("/");
  await fillFiles(page);
  await upload(page).click();
  await expect(page.getByRole("alert")).toHaveText("Field required; Input should be a valid number");
});

test("a 500 with no JSON body shows a generic failure message", async ({ page }) => {
  await page.route("**/api/drives", (route) =>
    route.fulfill({ status: 500, contentType: "text/plain", body: "Internal Server Error" }),
  );
  await page.goto("/");
  await fillFiles(page);
  await upload(page).click();
  await expect(page.getByRole("alert")).toHaveText("Upload failed (500).");
  await expect(page.getByTestId("drive-status")).toHaveCount(0);
});

test("an empty clock offset is posted as 0", async ({ page }) => {
  let posted = "";
  await page.route("**/api/drives", async (route) => {
    posted = route.request().postData() ?? "";
    await route.fulfill({ status: 201, json: { ...createdDrive, clock_offset_seconds: 0 } });
  });
  await page.goto("/");
  await fillFiles(page);
  await page.getByLabel("Clock offset (seconds)").fill("");
  await upload(page).click();
  await expect(page.getByTestId("drive-offset")).toHaveText("0 s");
  expect(offsetField(posted)).toBe("0");
});

test("a negative clock offset is posted and displayed", async ({ page }) => {
  let posted = "";
  await page.route("**/api/drives", async (route) => {
    posted = route.request().postData() ?? "";
    await route.fulfill({ status: 201, json: { ...createdDrive, clock_offset_seconds: -3.5 } });
  });
  await page.goto("/");
  await fillFiles(page);
  await page.getByLabel("Clock offset (seconds)").fill("-3.5");
  await upload(page).click();
  await expect(page.getByTestId("drive-offset")).toHaveText("-3.5 s");
  expect(offsetField(posted)).toBe("-3.5");
});

test("file pickers have accept filters and clearing a file is noticed", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Video file")).toHaveAttribute("accept", /video\/\*/);
  await expect(page.getByLabel("GPX file")).toHaveAttribute("accept", /\.gpx/);

  await fillFiles(page);
  await page.getByLabel("Video file").setInputFiles([]);
  await upload(page).click();
  await expect(page.getByRole("alert")).toHaveText("Choose a video file.");
});
