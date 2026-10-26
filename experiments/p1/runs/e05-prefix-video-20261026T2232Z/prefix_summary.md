# E05 observed-prefix probe summary (e05-prefix-video-20261026T2232Z)

> 已观测前缀 = 只给前 N% 的已观测姿态/视频帧，不填充未来帧；这是冻结前缀离线回放，不是实时流。

## Video path (trimmed mp4 -> local RTMPose -> mT5)

- modes: ['clip'], samples: 10, successes: 40/40, failures: 0, empty: 0
- feasibility: video end-to-end accepts observed prefixes down to 20 frames; no padding used
- 25%: ok 10/10, P50 e2e 820.389 ms, pose 585.105 ms, forward 51.719 ms
- 50%: ok 10/10, P50 e2e 870.196 ms, pose 598.589 ms, forward 52.662 ms
- 75%: ok 10/10, P50 e2e 937.597 ms, pose 583.613 ms, forward 53.372 ms
- 100%: ok 10/10, P50 e2e 901.426 ms, pose 557.968 ms, forward 53.953 ms

