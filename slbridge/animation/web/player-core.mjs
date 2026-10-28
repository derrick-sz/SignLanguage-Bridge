export class PlaybackClock {
  constructor() { this.duration = 0; this.position = 0; this.speed = 1; this.playing = false; this.anchor = 0; }
  load(duration) {
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Invalid duration");
    this.duration = duration; this.stop();
  }
  time(now) {
    const value = Math.min(this.duration, this.position + (this.playing ? Math.max(0, now - this.anchor) / 1000 * this.speed : 0));
    if (this.playing && value >= this.duration) { this.position = this.duration; this.playing = false; }
    return value;
  }
  play(now) { if (this.position >= this.duration) this.position = 0; this.anchor = now; this.playing = true; }
  pause(now) { this.position = this.time(now); this.playing = false; }
  stop() { this.position = 0; this.playing = false; this.anchor = 0; }
  seek(seconds, now) {
    if (!Number.isFinite(seconds)) throw new Error("Invalid seek position");
    this.position = Math.max(0, Math.min(this.duration, seconds)); this.anchor = now;
  }
  setSpeed(speed, now) {
    if (!Number.isFinite(speed) || speed < 0.25 || speed > 2) throw new Error("Invalid speed");
    this.position = this.time(now); this.anchor = now; this.speed = speed;
  }
}

export class PlaybackGeneration {
  constructor() { this.value = 0; }
  begin() { return ++this.value; }
  cancel() { ++this.value; }
  current(token) { return this.value === token; }
}

export function signingCamera(bounds, aspect, verticalFov = 34) {
  if (!(aspect > 0) || !Number.isFinite(aspect)) throw new Error("Invalid camera aspect");
  const half = Math.tan(verticalFov * Math.PI / 360);
  const height = bounds.maxY - bounds.minY + 0.24;
  const width = bounds.maxX - bounds.minX + 0.24;
  return { centerY: (bounds.minY + bounds.maxY) / 2,
    distance: Math.max(height / (2 * half), width / (2 * half * aspect)) };
}
