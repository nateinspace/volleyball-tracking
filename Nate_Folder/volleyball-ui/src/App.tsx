import { useCallback, useMemo, useRef, useState } from "react";
import { LIBRARY_CLIPS } from "./libraryClips";
import { resolveMediaUrl } from "./mediaUrl";
import type { PlayEventType } from "./types";
import { createEmptyLiveTracking } from "./types";
import "./App.css";

const DEFAULT_PRACTICE_VIDEO = "/videos/corrected_overlay.web.mp4";

const PLAYER_COUNT = 24;

const ALL_EVENTS: PlayEventType[] = [
  "dig",
  "set",
  "attack",
  "defense",
  "serve",
];

const EVENT_LABELS: Record<PlayEventType, string> = {
  dig: "Dig",
  set: "Set",
  attack: "Attack",
  defense: "Defense",
  serve: "Serve",
};

function TrajectoryOverlay({
  playbackTime,
  visible,
}: {
  playbackTime: number;
  visible: boolean;
}) {
  if (!visible) return null;
  const u = (playbackTime % 4) / 4;
  const x0 = 14;
  const y0 = 58;
  const x1 = 86;
  const y1 = 36;
  const cx = 48 + 10 * Math.sin(playbackTime * 0.7);
  const cy = 14 + 8 * Math.cos(playbackTime * 0.55);
  const d = `M ${x0} ${y0} Q ${cx} ${cy} ${x1} ${y1}`;
  const bx = (1 - u) * (1 - u) * x0 + 2 * (1 - u) * u * cx + u * u * x1;
  const by = (1 - u) * (1 - u) * y0 + 2 * (1 - u) * u * cy + u * u * y1;

  return (
    <svg
      className="trajectory-svg"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden
    >
      <defs>
        <linearGradient id="traj-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgba(56, 189, 248, 0.15)" />
          <stop offset="100%" stopColor="rgba(251, 191, 36, 0.45)" />
        </linearGradient>
      </defs>
      <path
        d={d}
        fill="none"
        stroke="url(#traj-grad)"
        strokeWidth={1.2}
        vectorEffect="non-scaling-stroke"
        className="trajectory-path"
      />
      <path
        d={d}
        fill="none"
        stroke="rgba(250, 204, 21, 0.85)"
        strokeWidth={0.45}
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={bx} cy={by} r={2.2} fill="#fbbf24" className="trajectory-ball" />
    </svg>
  );
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [urlInput, setUrlInput] = useState(DEFAULT_PRACTICE_VIDEO);
  const [videoUrl, setVideoUrl] = useState(DEFAULT_PRACTICE_VIDEO);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [highlighted, setHighlighted] = useState<Set<number>>(() => new Set());
  const [trajectoryOn, setTrajectoryOn] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [eventFilter, setEventFilter] = useState<Set<PlayEventType>>(
    () => new Set(ALL_EVENTS)
  );
  const [playerFilter, setPlayerFilter] = useState<number | null>(null);

  const [tracking, setTracking] = useState(createEmptyLiveTracking);
  const { ballSpeedMph, event, relativePlayerId, playerPositionsPct } = tracking;

  const searchResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return LIBRARY_CLIPS.filter((c) => {
      if (!eventFilter.has(c.event)) return false;
      if (playerFilter != null && c.playerId !== playerFilter) return false;
      if (q && !c.label.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [searchQuery, eventFilter, playerFilter]);

  const loadUrl = useCallback(() => {
    const trimmed = urlInput.trim();
    if (!trimmed) return;
    setMediaError(null);
    setVideoUrl(trimmed);
    setTracking(createEmptyLiveTracking());
    const v = videoRef.current;
    if (v) {
      v.pause();
      v.currentTime = 0;
      v.load();
    }
  }, [urlInput]);

  const toggleEventFilter = useCallback((ev: PlayEventType) => {
    setEventFilter((prev) => {
      const next = new Set(prev);
      if (next.has(ev)) next.delete(ev);
      else next.add(ev);
      return next;
    });
  }, []);

  const togglePlayer = useCallback((n: number) => {
    setHighlighted((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setHighlighted(new Set(Array.from({ length: PLAYER_COUNT }, (_, i) => i + 1)));
  }, []);

  const clearAll = useCallback(() => {
    setHighlighted(new Set());
  }, []);

  const seekToClip = useCallback((startSec: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = startSec;
    void v.play();
  }, []);

  const resolvedSrc = videoUrl ? resolveMediaUrl(videoUrl) : "";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <div className="brand-text">
            <p className="event-eyebrow">Waves Innovation Summit</p>
            <p className="event-partners">Pepperdine × StatsPerform AI Hackathon</p>
            <h1>
              Project 1 — Volleyball Vision: AI-Powered Practice Analysis Platform
            </h1>
            <p className="overview">
              AI-processed practice video for coaches and athletes to search, analyze,
              and visualize performance — moving beyond slow manual review toward
              consistent, scalable insight.
            </p>
          </div>
        </div>
      </header>

      <p className="scope-banner" role="note">
        <strong>In scope:</strong> 6v6 full-team segments only (both sides populated).
        Warm-ups and small-group drills are out of scope.{" "}
        <strong>Identity:</strong> track players with learned appearance / re-ID over
        time — do not rely on jersey number or color alone.
      </p>

      <main className="layout">
        <div className="main-column">
          <section className="panel workspace" aria-label="Practice video">
            <div className="workspace-top">
              <div className="workspace-identity">
                <span className="workspace-badge" aria-hidden />
                <div className="workspace-titles">
                  <span className="workspace-kicker">Live analysis</span>
                  <span className="workspace-title">Practice feed</span>
                </div>
              </div>
              <div className="workspace-load">
                <label htmlFor="video-url" className="sr-only">
                  Video URL
                </label>
                <input
                  id="video-url"
                  type="text"
                  className="input-feed"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadUrl()}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button type="button" className="btn btn-load" onClick={loadUrl}>
                  Load feed
                </button>
              </div>
            </div>

            <div className="workspace-toolbar">
              <label className="toggle trajectory-toggle">
                <input
                  type="checkbox"
                  checked={trajectoryOn}
                  onChange={(e) => setTrajectoryOn(e.target.checked)}
                />
                <span className="toggle-ui" aria-hidden />
                <span className="toggle-label">Ball trajectory</span>
              </label>
              <p className="toolbar-caption">
                Analytical overlay — arc and ball path; hidden until enabled.
              </p>
            </div>

            <div className="video-shell">
              {videoUrl ? (
                <>
                  <video
                    ref={videoRef}
                    key={resolvedSrc}
                    className="video"
                    src={resolvedSrc}
                    controls
                    playsInline
                    preload="metadata"
                    onTimeUpdate={() => {
                      const v = videoRef.current;
                      if (v) setPlaybackTime(v.currentTime);
                    }}
                    onLoadedData={() => setMediaError(null)}
                    onError={(e) => {
                      const err = e.currentTarget.error;
                      if (!err) {
                        setMediaError("Could not load video.");
                        return;
                      }
                      const msg =
                        err.code === MediaError.MEDIA_ERR_NETWORK
                          ? "Network error — file missing (404) or blocked. Confirm the file is in public/videos/ and deployed."
                          : err.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
                            ? "Codec not supported. Use H.264 + AAC MP4 (run npm run transcode-video for the bundled clip)."
                            : err.code === MediaError.MEDIA_ERR_DECODE
                              ? "Decode error — file may be corrupt."
                              : `Playback error (code ${err.code}).`;
                      setMediaError(msg);
                    }}
                  />
                  <TrajectoryOverlay
                    visible={trajectoryOn}
                    playbackTime={playbackTime}
                  />
                  <div className="highlight-layer" aria-hidden>
                    {[...highlighted]
                      .sort((a, b) => a - b)
                      .filter((id) => playerPositionsPct[id] != null)
                      .map((id) => {
                        const pos = playerPositionsPct[id]!;
                        const isFocus = relativePlayerId === id;
                        return (
                          <div
                            key={id}
                            className={`player-chip ${isFocus ? "focus" : ""}`}
                            style={{ left: `${pos.left}%`, top: `${pos.top}%` }}
                          >
                            {id}
                          </div>
                        );
                      })}
                  </div>
                </>
              ) : (
                <div className="video-empty">
                  <div className="video-empty-inner">
                    <p className="video-empty-title">No video loaded</p>
                    <p className="video-empty-text">
                      Add a practice video URL in the field above to start review and
                      search.
                    </p>
                  </div>
                </div>
              )}
            </div>
            {mediaError ? (
              <p className="video-error" role="alert">
                {mediaError}
              </p>
            ) : null}
          </section>

          <section className="panel search-panel" aria-label="Offline play search">
            <div className="search-panel-head">
              <div>
                <h2 className="panel-title">Play library</h2>
                <p className="panel-lead">
                  Search indexed plays; selecting a row seeks the active feed.
                </p>
              </div>
            </div>
            <div className="search-row">
              <label htmlFor="lib-query" className="sr-only">
                Search library
              </label>
              <input
                id="lib-query"
                type="search"
                className="input-feed input-feed--grow"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <label htmlFor="player-filter" className="filter-label">
                Player
              </label>
              <select
                id="player-filter"
                className="select-input"
                value={playerFilter ?? ""}
                onChange={(e) => {
                  const v = e.target.value;
                  setPlayerFilter(v === "" ? null : Number(v));
                }}
              >
                <option value="">Any roster ID</option>
                {Array.from({ length: PLAYER_COUNT }, (_, i) => (
                  <option key={i + 1} value={i + 1}>
                    ID {i + 1}
                  </option>
                ))}
              </select>
            </div>
            <div className="event-chips" role="group" aria-label="Event types">
              {ALL_EVENTS.map((ev) => {
                const on = eventFilter.has(ev);
                return (
                  <button
                    key={ev}
                    type="button"
                    className={`event-chip ${on ? "on" : ""} evt-${ev}`}
                    aria-pressed={on}
                    onClick={() => toggleEventFilter(ev)}
                  >
                    {EVENT_LABELS[ev]}
                  </button>
                );
              })}
            </div>
            <div className="results-wrap">
              <div className="results-head">
                <span className="results-count">
                  {searchResults.length} match{searchResults.length === 1 ? "" : "es"}
                </span>
              </div>
              <ul className="results-list">
                {searchResults.map((c) => (
                  <li key={c.id}>
                    <button
                      type="button"
                      className="result-item"
                      onClick={() => seekToClip(c.startSec)}
                      disabled={!videoUrl}
                    >
                      <span className={`action-pill mini evt-${c.event}`}>
                        {EVENT_LABELS[c.event]}
                      </span>
                      <span className="result-meta">
                        <span className="result-time mono">
                          {formatTime(c.startSec)} – {formatTime(c.endSec)}
                        </span>
                        <span className="result-player">ID {c.playerId}</span>
                      </span>
                      <span className="result-label">{c.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
              {!videoUrl ? (
                <p className="results-hint">Load a feed to enable seek.</p>
              ) : null}
            </div>
          </section>
        </div>

        <aside className="panel side-panel">
          <div className="telemetry" role="status" aria-live="polite">
            <h2>Live telemetry</h2>
            <div className="metric">
              <span className="metric-label">Ball speed (estimated)</span>
              <span className="metric-value mono">
                {ballSpeedMph != null ? (
                  <>
                    {ballSpeedMph}
                    <small> mph</small>
                  </>
                ) : (
                  <span className="empty-value">—</span>
                )}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Event vs roster ID</span>
              <span className="metric-value phase-line">
                {event != null && relativePlayerId != null ? (
                  <>
                    <span className={`action-pill action-${event} inline`}>
                      {EVENT_LABELS[event]}
                    </span>
                    <span className="phase-connector">→</span>
                    <span>
                      Track ID <strong>{relativePlayerId}</strong>
                    </span>
                  </>
                ) : (
                  <span className="empty-value">—</span>
                )}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Trajectory layer</span>
              <span className="metric-value">
                {trajectoryOn ? (
                  <span className="status-on">Visible</span>
                ) : (
                  <span className="empty-value">Off</span>
                )}
              </span>
            </div>
          </div>

          <div className="players-block">
            <div className="players-head">
              <h2>Track highlights</h2>
              <div className="bulk">
                <button type="button" className="btn ghost" onClick={selectAll}>
                  All {PLAYER_COUNT}
                </button>
                <button type="button" className="btn ghost" onClick={clearAll}>
                  Clear
                </button>
              </div>
            </div>
            <p className="players-sub">
              Persistent IDs from detection / re-ID (not jersey-dependent).{" "}
              {highlighted.size} selected.
            </p>
            <div className="player-grid">
              {Array.from({ length: PLAYER_COUNT }, (_, i) => {
                const n = i + 1;
                const on = highlighted.has(n);
                return (
                  <button
                    key={n}
                    type="button"
                    className={`jersey ${on ? "on" : ""}`}
                    onClick={() => togglePlayer(n)}
                    aria-pressed={on}
                  >
                    {n}
                  </button>
                );
              })}
            </div>
          </div>

          <p className="data-note">
            Practice batches are distributed through the hackathon data link; additional
            batch per schedule. Questions via the event Slack.
          </p>
        </aside>
      </main>
    </div>
  );
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const f = Math.floor((sec % 1) * 10);
  return `${m}:${String(s).padStart(2, "0")}.${f}`;
}

export default App;
