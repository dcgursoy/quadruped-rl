# Robustness results

Policy: `results/checkpoints/best_8500k.zip`

## Push recovery (0.2 s horizontal push to trunk mid-walk; success = upright 5 s later; 10 trials/cell)

| direction | 25 N | 50 N | 75 N | 100 N | 125 N | 150 N |
|---|---|---|---|---|---|---|
| left | 10/10 | 9/10 | 9/10 | 5/10 | 3/10 | 1/10 |
| right | 10/10 | 3/10 | 1/10 | 0/10 | 0/10 | 0/10 |
| backward | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| forward | 10/10 | 10/10 | 10/10 | 10/10 | 6/10 | 7/10 |

## Observation noise (IMU/encoder-scaled Gaussian)

| noise scale | falls | mean speed |
|---|---|---|
| x0 | 0/20 | 1.36 m/s |
| x1 | 9/20 | 0.97 m/s |
| x2 | 20/20 | 0.63 m/s |
