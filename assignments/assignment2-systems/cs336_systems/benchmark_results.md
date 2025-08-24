| model   |   d_model |   d_ff |   num_layers |   num_heads | function         | mean_time   | std_time   |
|:--------|----------:|-------:|-------------:|------------:|:-----------------|:------------|:-----------|
| small   |       768 |   3072 |           12 |          12 | forward_only     | 0.0227 s    | 0.0006 s   |
| small   |       768 |   3072 |           12 |          12 | forward_backward | 0.0755 s    | 0.0004 s   |
| medium  |      1024 |   4096 |           24 |          16 | forward_only     | 0.0677 s    | 0.0002 s   |
| medium  |      1024 |   4096 |           24 |          16 | forward_backward | 0.2338 s    | 0.0009 s   |
| large   |      1280 |   5120 |           36 |          20 | forward_only     | 0.1513 s    | 0.0008 s   |
| large   |      1280 |   5120 |           36 |          20 | forward_backward | 0.5250 s    | 0.0004 s   |