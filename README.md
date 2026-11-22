# Solar DNI Forecasting Prototype

A PyTorch pipeline that forecasts Direct Normal Irradiance (DNI), the direct sunlight solar plants run on, one hour ahead from 27 years of NREL National Solar Radiation Database data. I built it in March 2026 to prove out the data pipeline and evaluation design before a Penn State graduate deep learning team project. It became the starting point for that project's Azure ML training setup.

## The problem

Solar operators plan dispatch and storage around irradiance forecasts. Forecasting time series makes it easy to fool yourself: random splits and scalers fit on all the data leak the future into training, and a model can look accurate just by repeating the last hour. This prototype is built to rule both out.

## Approach

1. **Pull the data reproducibly.** `download_nsrdb.py` fetches yearly NSRDB files for one location (default State College, PA, 1998 to 2024), with the API key read from `.env`.
2. **Split by time.** `build_dataset.py` trains on 1998 to 2022, validates on 2023 and tests on 2024, so the model never sees the future.
3. **Prevent leakage.** Scalers are fit on the training years only. Each prediction sees only the previous 48 hours.
4. **Train and select.** `train_lstm.py` trains a 2-layer LSTM (64 hidden units) and keeps the checkpoint with the best validation RMSE.
5. **Compare to persistence.** Test MAE, RMSE and R² are reported next to a persistence baseline that predicts this hour equals the last hour. A model that can't beat it adds nothing.

## What I learned

- **The evaluation design is the product.** Chronological splits, train-only scaling and a naive baseline mattered more to whether the results could be trusted than any architecture choice.
- **Small, scriptable steps scale.** Separate download, build and train scripts with settings in `.env` moved straight into an Azure ML training job.

## Setup

1. Get a free NREL API key.
2. Copy `.env.example` to `.env` and fill in your key and email.
3. `pip install -r requirements.txt`
4. Run `scripts/download_nsrdb.py`, then `scripts/build_dataset.py`, then `scripts/train_lstm.py`.

Data and trained weights are not in this repo.
