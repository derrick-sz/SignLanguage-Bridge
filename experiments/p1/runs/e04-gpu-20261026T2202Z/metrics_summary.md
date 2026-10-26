# E04 measured summary (e04-gpu-20261026T2202Z)

| phase/model | attempts | ok | fail | P50 e2e ms | P95 e2e ms | P50 fwd ms | P50 pose ms | mean RTF | peak torch VRAM MB |
|---|---|---|---|---|---|---|---|---|---|
| cold/slrt_hrnet_npz | 3 | 3 | 0 | 9558.800 | 9694.894 | 3213.081 | 5.572 | 2.636 | 519.800 |
| cold/unisign_pose | 3 | 3 | 0 | 303.340 | 332.882 | 71.402 | 1.126 | 0.086 | 1202.700 |
| cold/unisign_video | 4 | 4 | 0 | 2950.236 | 3301.025 | 67.231 | 2203.557 | 0.834 | 1202.400 |
| load/slrt_hrnet_npz | 13 | 13 | 0 | 8038.375 | 10561.666 | 3329.419 | 5.921 | 2.190 | 373.900 |
| load/unisign_pose | 419 | 419 | 0 | 276.257 | 345.200 | 37.070 | 0.613 | 0.077 | 1213.000 |
| load/unisign_video | 42 | 42 | 0 | 2702.195 | 3245.768 | 32.718 | 2144.009 | 0.744 | 1211.100 |
| parallel/slrt_hrnet_npz | 5 | 5 | 0 | 6160.082 | 8282.564 | 2812.779 | 3.950 | 1.798 | 373.900 |
| parallel/unisign_video | 5 | 5 | 0 | 4800.039 | 5942.441 | 58.996 | 4157.308 | 1.139 | 1211.100 |
| serial/slrt_hrnet_npz | 5 | 5 | 0 | 5831.764 | 6575.811 | 2237.217 | 4.180 | 1.530 | 373.900 |
| serial/unisign_video | 5 | 5 | 0 | 2602.822 | 3198.180 | 49.755 | 2139.795 | 0.633 | 1211.100 |
| warm/slrt_hrnet_npz | 30 | 30 | 0 | 9865.084 | 11189.703 | 3701.439 | 6.144 | 2.534 | 373.900 |
| warm/unisign_pose | 30 | 30 | 0 | 277.642 | 344.328 | 38.986 | 0.675 | 0.078 | 1213.000 |
| warm/unisign_video | 30 | 30 | 0 | 2690.395 | 3283.920 | 35.896 | 2156.364 | 0.748 | 1211.100 |

- serial sum of batch seconds: 40.276
- parallel compute wall seconds: 34.1
- serial-sum / parallel-wall: 1.1811
