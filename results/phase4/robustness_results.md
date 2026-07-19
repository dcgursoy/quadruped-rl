# Robustness results

Policy: `checkpoints/v3_full/ppo_go1_10000000_steps.zip`

## Push recovery (0.2 s horizontal push to trunk mid-walk; success = upright 5 s later; 10 trials/cell)

| direction | 25 N | 50 N | 75 N | 100 N | 125 N | 150 N |
|---|---|---|---|---|---|---|
| left | 10/10 | 10/10 | 10/10 | 10/10 | 8/10 | 0/10 |
| right | 10/10 | 10/10 | 10/10 | 9/10 | 8/10 | 3/10 |
| backward | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 8/10 |
| forward | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 2/10 |

## Observation noise (IMU/encoder-scaled Gaussian)

| noise scale | falls | mean speed |
|---|---|---|
| x0 | 0/20 | 1.90 m/s |
| x1 | 0/20 | 1.91 m/s |
| x2 | 0/20 | 1.80 m/s |
