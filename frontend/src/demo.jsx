import { createContext, useContext, useEffect, useState } from "react";

// True when the server is the read-only hosted demo (GET /api/demo answers {"demo": true}). Components read it
// with useDemo() and hide the controls that would change anything; the server refuses those requests anyway.
export const DemoContext = createContext(false);

export function useDemo() {
  return useContext(DemoContext);
}

export const DEMO_BANNER_TEXT = "Read-only demo: nothing you do here is saved.";

// Asks the server once. Any failure (offline, 404, odd answer) means "not a demo", so the normal UI shows.
export function useDemoStatus() {
  const [demo, setDemo] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch("/api/demo");
        if (!response.ok) return;
        const data = await response.json();
        if (!cancelled && data && data.demo === true) setDemo(true);
      } catch {
        /* not a demo, or the server is unreachable: leave the normal UI */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return demo;
}
