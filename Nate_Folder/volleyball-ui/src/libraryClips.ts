import type { LibraryClip } from "./types";

/** Sample indexed clips — replace with API / CV pipeline output. */
export const LIBRARY_CLIPS: LibraryClip[] = [
  {
    id: "clip-01",
    startSec: 8.2,
    endSec: 11.0,
    event: "serve",
    playerId: 3,
    label: "6v6 — float serve, deep zone 5",
  },
  {
    id: "clip-02",
    startSec: 24.6,
    endSec: 28.1,
    event: "dig",
    playerId: 12,
    label: "6v6 — first contact / pass to target",
  },
  {
    id: "clip-03",
    startSec: 41.0,
    endSec: 44.5,
    event: "set",
    playerId: 9,
    label: "6v6 — quick set middle",
  },
  {
    id: "clip-04",
    startSec: 56.3,
    endSec: 59.8,
    event: "attack",
    playerId: 7,
    label: "6v6 — pin attack cross-court",
  },
  {
    id: "clip-05",
    startSec: 72.1,
    endSec: 76.0,
    event: "defense",
    playerId: 4,
    label: "6v6 — block touch / defensive read",
  },
  {
    id: "clip-06",
    startSec: 88.4,
    endSec: 92.0,
    event: "dig",
    playerId: 11,
    label: "6v6 — hard-driven dig, back row",
  },
  {
    id: "clip-07",
    startSec: 105.0,
    endSec: 108.5,
    event: "defense",
    playerId: 6,
    label: "6v6 — defensive recovery after block, rally extended",
  },
  {
    id: "clip-08",
    startSec: 120.0,
    endSec: 124.2,
    event: "attack",
    playerId: 14,
    label: "6v6 — pipe attack after transition",
  },
];
