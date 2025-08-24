| model   |   d_model |   d_ff |   num_layers |   num_heads | function         | mean_time   | std_time   |
|:--------|----------:|-------:|-------------:|------------:|:-----------------|:------------|:-----------|
| small   |       768 |   3072 |           12 |          12 | forward_only     | 0.0595 s    | 0.0006 s   |
| small   |       768 |   3072 |           12 |          12 | forward_backward | 0.2022 s    | 0.0036 s   |
| medium  |      1024 |   4096 |           24 |          16 | forward_only     | 0.1716 s    | 0.0033 s   |
| medium  |      1024 |   4096 |           24 |          16 | forward_backward | 4.0455 s    | 0.0719 s   |