import { test, expect } from "@playwright/test";

test("hello-world page renders", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("RoadScout");
  await expect(page.getByRole("heading", { name: "RoadScout" })).toBeVisible();
  await expect(page.getByText("Road hazards, found from your commute video.")).toBeVisible();
});
