# E14 快速验证

| host | n | threads | bind | kernel | best_GBps | median_GBps |
| --- | --- | --- | --- | --- | --- | --- |
| ubuntu | 1048576 | 1 | close | copy | 25.606 | 25.242 |
| ubuntu | 1048576 | 1 | close | triad | 16.456 | 16.354 |
| ubuntu | 1048576 | 2 | close | copy | 39.274 | 32.143 |
| ubuntu | 1048576 | 2 | close | triad | 21.324 | 17.264 |
| ubuntu | 1048576 | 4 | close | copy | 44.909 | 44.094 |
| ubuntu | 1048576 | 4 | close | triad | 22.304 | 21.993 |
| ubuntu | 1048576 | 8 | close | copy | 251.18 | 249.33 |
| ubuntu | 1048576 | 8 | close | triad | 239.79 | 235.4 |
| ubuntu | 1048576 | 16 | close | copy | 456.15 | 455 |
| ubuntu | 1048576 | 16 | close | triad | 502.97 | 484.99 |
| ubuntu | 1048576 | 32 | close | copy | 550.97 | 547.08 |
| ubuntu | 1048576 | 32 | close | triad | 472.6 | 470.66 |
| ubuntu | 1048576 | 64 | close | copy | 590.87 | 532.09 |
| ubuntu | 1048576 | 64 | close | triad | 851.66 | 740.97 |
| ubuntu | 1048576 | 128 | close | copy | 222.71 | 220.19 |
| ubuntu | 1048576 | 128 | close | triad | 317.24 | 311.45 |
| ubuntu | 1048576 | 1 | spread | copy | 25.238 | 24.355 |
| ubuntu | 1048576 | 1 | spread | triad | 16.39 | 16.278 |
| ubuntu | 1048576 | 2 | spread | copy | 98.283 | 97.654 |
| ubuntu | 1048576 | 2 | spread | triad | 102.22 | 101.25 |
| ubuntu | 1048576 | 4 | spread | copy | 201.58 | 198.34 |
| ubuntu | 1048576 | 4 | spread | triad | 220.16 | 219.82 |
| ubuntu | 1048576 | 8 | spread | copy | 367.64 | 365.31 |
| ubuntu | 1048576 | 8 | spread | triad | 416.56 | 412.47 |
| ubuntu | 1048576 | 16 | spread | copy | 590.25 | 581.48 |
| ubuntu | 1048576 | 16 | spread | triad | 702.44 | 698.52 |
| ubuntu | 1048576 | 32 | spread | copy | 614.24 | 605.4 |
| ubuntu | 1048576 | 32 | spread | triad | 618.51 | 613.4 |
| ubuntu | 1048576 | 64 | spread | copy | 567.2 | 561.76 |
| ubuntu | 1048576 | 64 | spread | triad | 782.24 | 779.62 |
| ubuntu | 1048576 | 128 | spread | copy | 222.38 | 220.11 |
| ubuntu | 1048576 | 128 | spread | triad | 321.84 | 317.95 |


GB/s 采用十进制字节。Copy=16n 字节，Triad=24n 字节；每配置热身一次、测量五次，逐次验证输出。
