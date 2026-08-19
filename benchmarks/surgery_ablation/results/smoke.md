# Surgery ablation summary

Rows: 40

## Per-instance quality

| instance | arm | n | min tc | median tc |
| --- | --- | --- | --- | --- |
| dbn_13 | cold_only_r8 | 2 | 45.127720 | 45.215401 |
| dbn_13 | surg_greedy_local_r8 | 2 | 37.302401 | 39.316334 |
| dbn_13 | surg_greedy_root_r8 | 2 | 37.104869 | 38.298190 |
| dbn_13 | surg_warm_local_r8 | 2 | 31.632024 | 33.659685 |
| dbn_13 | surg_warm_root_r8 | 2 | 35.687346 | 36.944167 |
| petersen | cold_only_r8 | 2 | 8.491853 | 8.491853 |
| petersen | surg_greedy_local_r8 | 2 | 8.491853 | 8.491853 |
| petersen | surg_greedy_root_r8 | 2 | 8.491853 | 8.491853 |
| petersen | surg_warm_local_r8 | 2 | 8.491853 | 8.491853 |
| petersen | surg_warm_root_r8 | 2 | 8.491853 | 8.491853 |
| qft_27 | cold_only_r8 | 2 | 36.455245 | 38.003223 |
| qft_27 | surg_greedy_local_r8 | 2 | 30.100734 | 30.319744 |
| qft_27 | surg_greedy_root_r8 | 2 | 31.137963 | 32.121959 |
| qft_27 | surg_warm_local_r8 | 2 | 31.354538 | 31.915470 |
| qft_27 | surg_warm_root_r8 | 2 | 31.097943 | 31.226241 |
| surfacecode_d9 | cold_only_r8 | 2 | 21.918479 | 21.965192 |
| surfacecode_d9 | surg_greedy_local_r8 | 2 | 21.918479 | 21.965192 |
| surfacecode_d9 | surg_greedy_root_r8 | 2 | 21.807946 | 21.808134 |
| surfacecode_d9 | surg_warm_local_r8 | 2 | 21.918479 | 21.965192 |
| surfacecode_d9 | surg_warm_root_r8 | 2 | 21.855240 | 21.886859 |

## Surgery versus matched cold-only

| surgery arm vs cold-only | W | T | L |
| --- | --- | --- | --- |
| surg_greedy_local_r8 | 4 | 4 | 0 |
| surg_greedy_root_r8 | 6 | 2 | 0 |
| surg_warm_local_r8 | 4 | 4 | 0 |
| surg_warm_root_r8 | 5 | 3 | 0 |

## Work-matched TreeSA

| work-matched arm | n | median tc | median node visits | median wall s |
| --- | --- | --- | --- | --- |

## Accepted rebuilds

| arm | jobs | accepted rebuilds |
| --- | --- | --- |
| cold_only_r8 | 8 | 0 |
| surg_greedy_local_r8 | 8 | 16 |
| surg_greedy_root_r8 | 8 | 16 |
| surg_warm_local_r8 | 8 | 21 |
| surg_warm_root_r8 | 8 | 14 |
