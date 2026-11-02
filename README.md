# NYC Taxi Trip Duration Predictor

A Streamlit app that predicts how long a New York City taxi trip will take (`trip_duration`, in seconds), using a trained LightGBM model.

This project was built for **Round 2 (AI & Data Challenge) of the UNETI AI Olympiad 2026**. The model was trained on the competition's training set (1,200,000 trips, based on the NYC Taxi Trip Duration dataset) and reached an RMSLE of 0.3366 on the validation split. The training notebook is not included in this repository.

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit interface |
| `features.py` | Feature engineering (same code used during training) |
| `model.joblib` | Trained model (an average of 3 LightGBM models) |
| `requirements.txt` | Python dependencies |

## Setup

Requires Python 3.10 or newer.

1. Create a virtual environment (only needed once):

   ```bash
   python -m venv .venv
   ```

2. Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS / Linux: `source .venv/bin/activate`

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. If that port is already in use, pick another one:

```bash
streamlit run app.py --server.port 8502
```

## Usage

The interface is in Vietnamese.

1. Choose the pickup point (**Điểm đón**) and drop-off point (**Điểm đến**) from the list, or choose **Tùy chỉnh** (custom) to enter latitude and longitude.
2. Set the pickup month, day of the week, hour and minute.
3. Enter the number of passengers and the vendor.
4. Click **Dự đoán** (Predict) to see the predicted trip time (minutes and seconds), the straight-line distance and a map of the trip.
