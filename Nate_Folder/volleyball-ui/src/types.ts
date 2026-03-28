export type PlayEventType = "dig" | "set" | "attack" | "defense" | "serve";

export type LiveTracking = {
  ballSpeedMph: number | null;
  event: PlayEventType | null;
  relativePlayerId: number | null;
  playerPositionsPct: Record<number, { left: number; top: number }>;
};

export function createEmptyLiveTracking(): LiveTracking {
  return {
    ballSpeedMph: null,
    event: null,
    relativePlayerId: null,
    playerPositionsPct: {},
  };
}

export type LibraryClip = {
  id: string;
  startSec: number;
  endSec: number;
  event: PlayEventType;
  playerId: number;
  label: string;
};
