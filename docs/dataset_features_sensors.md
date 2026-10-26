# Dataset, Features, Sensors and Settings

This document describes the data behind the Turbofan Engine RUL Predictor,
the characteristics of the FD001 subset, the feature engineering pipeline,
and the physical meaning of each operational setting and sensor channel
used by the model.

## 1. Source dataset

- **Name:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
  Turbofan Engine Degradation Simulation.
- **Subset used:** `FD001`.
- **Fault mode:** High-Pressure Compressor (HPC) degradation only.
- **Operating conditions:** Single (sea-level, standard day).
- **Format:** Whitespace-separated text files with no header.
  Columns: `unit_number, time_in_cycles, op_setting_1..3, sensor_1..21`.

### 1.1 Train split (`train_FD001.txt`)

| Property | Value |
|---|---|
| Engines (units) | 100 |
| Total rows (cycles) | 20,631 |
| Run type | Run-to-failure (each engine runs until failure) |
| Target | RUL = (engine's final cycle − current cycle), clipped at 125 |

### 1.2 Test split (`test_FD001.txt` + `RUL_FD001.txt`)

| Property | Value |
|---|---|
| Engines (units) | 100 |
| Total rows (cycles) | 13,096 |
| Run type | Partial runs (engines stopped before failure) |
| Evaluation | Last recorded cycle of each engine, against NASA's provided ground-truth RUL |
| Ground truth file | `RUL_FD001.txt` (one RUL value per test engine) |

The evaluation protocol follows the standard C-MAPSS benchmark: only the final
observed cycle of each test engine is scored.

## 2. Target variable

- **RUL (Remaining Useful Life):** number of flight cycles remaining before
  engine failure.
- **Clipping:** RUL is capped at `125` cycles (`RUL_CLIP = 125`). Early in an
  engine's life the degradation signal is flat, so the true RUL is not
  learnable from sensor readings alone. Clipping turns the target into a
  piecewise-linear curve: constant while healthy, then linearly decreasing
  near failure.
- All RUL values are non-negative integers in cycles.

## 3. Feature engineering pipeline

1. **Load raw files** (`src/data.py`): parse whitespace-separated CMAPSS files
   into a DataFrame with the standard column layout.
2. **Label training rows** with clipped RUL.
3. **Drop constant sensors:** 7 sensors with (near) zero variance across the
   whole FD001 run are removed because they carry no discriminative signal.
4. **Base feature columns (17):** 3 operational settings + 14 retained sensors.
5. **Rolling-window features:** for each engine unit, add rolling mean and
   rolling standard deviation (window = 5 cycles, `min_periods = 1`) on top
   of each base feature. This captures short-term degradation trends.
6. **Final feature vector (51):** 17 base + 17 rolling-mean + 17 rolling-std.
7. **Scaling:** `StandardScaler` fit on the training data only, then applied
   to train and test.

### 3.1 Base feature columns

```
op_setting_1, op_setting_2, op_setting_3,
sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9,
sensor_11, sensor_12, sensor_13, sensor_14, sensor_15,
sensor_17, sensor_20, sensor_21
```

### 3.2 Derived (rolling) feature columns

For each base feature `c`, two derived columns are produced:

- `c_rollmean5` — rolling mean over the last 5 cycles of that engine.
- `c_rollstd5` — rolling standard deviation over the last 5 cycles
  (filled with 0.0 when undefined, e.g. first cycle).

At inference time (single snapshot with no history), the rolling-mean
features fall back to the instantaneous reading and the rolling-std features
to 0.0.

## 4. Operational settings

The three operational settings describe the flight condition under which a
cycle was recorded. In FD001 there is effectively a single operating
condition, but the settings are still retained as features.

| Setting | Physical quantity | Description |
|---|---|---|
| `op_setting_1` | Altitude | Flight altitude at which the cycle was recorded (ft). |
| `op_setting_2` | Mach number | Flight Mach number (dimensionless). |
| `op_setting_3` | Throttle Resolver Angle (TRA) | Throttle position / power setting of the engine (degrees). |

## 5. Sensors (21 channels)

Physical interpretation of the C-MAPSS sensor channels (standard C-MAPSS
naming). FD001 drops 7 near-constant sensors; the remaining 14 are used.

### 5.1 Retained sensors (used by the model)

| Sensor | Physical measurement | Description |
|---|---|---|
| `sensor_2` | T24 | Total temperature at LPC (Low-Pressure Compressor) outlet (°R). |
| `sensor_3` | T30 | Total temperature at HPC (High-Pressure Compressor) outlet (°R). |
| `sensor_4` | T50 | Total temperature at LPT (Low-Pressure Turbine) outlet (°R). |
| `sensor_7` | P30 | Total pressure at HPC outlet (psia). |
| `sensor_8` | Nf | Physical fan speed (rpm). |
| `sensor_9` | Nc | Physical core speed (rpm). |
| `sensor_11` | Ps30 | Static pressure at HPC outlet (psia). |
| `sensor_12` | phi | Ratio of fuel flow to Ps30 (pps/psi). |
| `sensor_13` | NRf | Corrected fan speed (rpm). |
| `sensor_14` | NRc | Corrected core speed (rpm). |
| `sensor_15` | BPR | Bypass ratio (dimensionless). |
| `sensor_17` | htBleed | Bleed enthalpy (enthalpy units). |
| `sensor_20` | W31 | HPT (High-Pressure Turbine) coolant bleed (lbm/s). |
| `sensor_21` | W32 | LPT (Low-Pressure Turbine) coolant bleed (lbm/s). |

### 5.2 Dropped (near-constant) sensors

These 7 channels have (near) zero variance across the whole FD001 run and
carry no discriminative signal for RUL estimation, so they are excluded.

| Sensor | Physical measurement | Reason for dropping |
|---|---|---|
| `sensor_1` | T2 | Total temperature at fan inlet — essentially constant. |
| `sensor_5` | P2 | Pressure at fan inlet — essentially constant. |
| `sensor_6` | P15 | Total pressure in bypass-duct — essentially constant. |
| `sensor_10` | epr | Engine pressure ratio — essentially constant. |
| `sensor_16` | farB | Burner fuel-air ratio — essentially constant. |
| `sensor_18` | Nf_dmd | Demanded fan speed — essentially constant. |
| `sensor_19` | PCNfR_dmd | Demanded corrected fan speed — essentially constant. |

## 6. Model input / output

- **Input:** 51 features (17 base + 34 rolling), provided to the API as the
  17 base sensor/setting readings (rolling features are derived server-side).
- **Output:** `predicted_rul_cycles` (0–125), `status`
  (`healthy` / `warning` / `critical`), `urgency` (`low` / `medium` / `high`),
  and `model_used` (best model by test RMSE, currently XGBoost).

## 7. Cross-validation and evaluation

- **CV:** 5-fold `GroupKFold` grouped by engine unit, so no engine's cycles
  leak across train/validation folds.
- **Held-out test:** NASA's official FD001 test split, scored at each
  engine's last recorded cycle against NASA's provided ground-truth RUL.
- **Metrics:** MAE, RMSE, R² (in cycles).

## 8. Notes and limitations

- Trained/evaluated only on FD001 (single operating condition, single fault
  mode). Other subsets (FD002–FD004) have multiple operating conditions and
  would need condition normalization.
- The API accepts a single sensor snapshot; rolling features are approximated
  from that single reading.
- RUL is capped at 125 cycles, so "very healthy" engines beyond that cap
  cannot be distinguished.
- No sequence (LSTM/Transformer) model is trained; tree ensembles were
  sufficient to beat the linear baseline on a laptop.