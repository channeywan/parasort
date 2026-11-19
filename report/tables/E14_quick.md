# E14 快速验证

| host | n | threads | bind | kernel | best_GBps | median_GBps |
| --- | --- | --- | --- | --- | --- | --- |
| ubuntu | 1048576 | 1 | close | copy | 24.464 | 23.595 |
| ubuntu | 1048576 | 1 | close | triad | 16.508 | 16.465 |
| ubuntu | 1048576 | 2 | close | copy | 38.103 | 29.977 |
| ubuntu | 1048576 | 2 | close | triad | 21.348 | 17.617 |
| ubuntu | 1048576 | 4 | close | copy | 45.772 | 45.416 |
| ubuntu | 1048576 | 4 | close | triad | 22.604 | 22.514 |
| ubuntu | 1048576 | 8 | close | copy | 337.75 | 326.54 |
| ubuntu | 1048576 | 8 | close | triad | 310.88 | 303.74 |
| ubuntu | 1048576 | 16 | close | copy | 455.28 | 418.12 |
| ubuntu | 1048576 | 16 | close | triad | 466.79 | 392.12 |
| ubuntu | 1048576 | 32 | close | copy | 659.29 | 491.07 |
| ubuntu | 1048576 | 32 | close | triad | 644.91 | 418.63 |
| ubuntu | 1048576 | 64 | close | copy | 580.21 | 572.69 |
| ubuntu | 1048576 | 64 | close | triad | 809.81 | 798.13 |
| ubuntu | 1048576 | 128 | close | copy | 214.94 | 214.03 |
| ubuntu | 1048576 | 128 | close | triad | 318.4 | 312.92 |
| ubuntu | 1048576 | 1 | spread | copy | 24.154 | 23.586 |
| ubuntu | 1048576 | 1 | spread | triad | 16.435 | 16.309 |
| ubuntu | 1048576 | 2 | spread | copy | 97.802 | 96.433 |
| ubuntu | 1048576 | 2 | spread | triad | 98.566 | 96.235 |
| ubuntu | 1048576 | 4 | spread | copy | 129.39 | 128.78 |
| ubuntu | 1048576 | 4 | spread | triad | 141.17 | 140.31 |
| ubuntu | 1048576 | 8 | spread | copy | 379.57 | 374.8 |
| ubuntu | 1048576 | 8 | spread | triad | 417.95 | 414.37 |
| ubuntu | 1048576 | 16 | spread | copy | 594.85 | 581.03 |
| ubuntu | 1048576 | 16 | spread | triad | 696.79 | 689.82 |
| ubuntu | 1048576 | 32 | spread | copy | 740.24 | 733.13 |
| ubuntu | 1048576 | 32 | spread | triad | 827.56 | 817.84 |
| ubuntu | 1048576 | 64 | spread | copy | 652.13 | 632.62 |
| ubuntu | 1048576 | 64 | spread | triad | 919.1 | 909.45 |
| ubuntu | 1048576 | 128 | spread | copy | 228.24 | 221.78 |
| ubuntu | 1048576 | 128 | spread | triad | 329.81 | 326.73 |


GB/s 采用十进制字节。Copy=16n 字节，Triad=24n 字节；每配置热身一次、测量五次，逐次验证输出。
