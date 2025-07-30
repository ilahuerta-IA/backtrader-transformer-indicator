import backtrader as bt
import torch
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# --- WARNING SUPPRESSION ---
import warnings
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
# --- END WARNING SUPPRESSION ---

# --- Path setup for model artifacts ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / 'src' / 'ml_models'
    DATA_PATH = PROJECT_ROOT / 'data' / 'EURUSD_5m_2Mon.csv'
    if not DATA_PATH.exists():
        print(f"FATAL: Data file not found at {DATA_PATH}")
        exit()
except Exception:
    print("FATAL: Could not determine project paths. Please run from the project root.")
    exit()

# --- Unchanged Helper Functions and Prediction Indicator ---
def create_time_features_for_window(dt_index: pd.DatetimeIndex) -> np.ndarray:
    features = pd.DataFrame(index=dt_index)
    features['hour'] = (dt_index.hour / 23.0) - 0.5
    features['day_of_week'] = (dt_index.dayofweek / 6.0) - 0.5
    features['day_of_month'] = ((dt_index.day - 1) / 30.0) - 0.5
    features['month'] = ((dt_index.month - 1) / 11.0) - 0.5
    return features.values

class TransformerPredictionIndicator(bt.Indicator):
    lines = ('prediction',)
    params = (('models_dir', str(MODELS_DIR)),)
    plotinfo = dict(subplot=False)
    
    def __init__(self):
        super().__init__()
        self.p.models_dir = Path(self.p.models_dir)
        from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
        with open(self.p.models_dir / 'model_config.json', 'r') as f:
            config = TimeSeriesTransformerConfig.from_dict(json.load(f))
        
        self.scaler = joblib.load(self.p.models_dir / 'target_scaler.pkl')
        self.model = TimeSeriesTransformerForPrediction(config)
        self.model.load_state_dict(torch.load(self.p.models_dir / 'best_transformer_model.pth', map_location='cpu'))
        self.model.eval()
        self.history_len = config.context_length + max(config.lags_sequence or [0])
        self.addminperiod(self.history_len)

    def next(self):
        datetimes = [self.data.num2date(self.data.datetime[-i]) for i in range(self.history_len)]
        datetimes.reverse()
        dt_index = pd.to_datetime(datetimes)
        close_prices = np.array(self.data.close.get(size=self.history_len))
        scaled_prices = self.scaler.transform(close_prices.reshape(-1, 1)).flatten()
        past_time_features = torch.tensor(create_time_features_for_window(dt_index), dtype=torch.float32).unsqueeze(0)
        future_dt_index = pd.to_datetime([dt_index[-1] + timedelta(minutes=5)])
        future_time_features = torch.tensor(create_time_features_for_window(future_dt_index), dtype=torch.float32).unsqueeze(0)
        past_values = torch.tensor(scaled_prices, dtype=torch.float32).unsqueeze(0)
        past_observed_mask = torch.ones_like(past_values)
        with torch.no_grad():
            outputs = self.model.generate(past_values=past_values, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_time_features=future_time_features)
        final_pred = self.scaler.inverse_transform(outputs.sequences.mean(dim=1).cpu().numpy()[:, -1].reshape(-1, 1))[0][0]
        self.lines.prediction[0] = final_pred

class AngleIndicator(bt.Indicator):
    lines = ('angle',)
    params = (('angle_lookback', 5), ('scale_factor', 50000),)
    plotinfo = dict(
        subplot=True,
        plotname='Smoothed Prediction Angle (Degrees)',
        plotlines={'angle': dict(_name='Angle', color='blue', linestyle='--', linewidth=1.5, alpha=0.7)}
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data0, period=self.p.angle_lookback)
        self.addminperiod(self.p.angle_lookback)

    def next(self):
        rise = (self.data0[0] - self.data0[-self.p.angle_lookback + 1]) * self.p.scale_factor
        self.lines.angle[0] = np.degrees(np.arctan2(rise, self.p.angle_lookback))

# --- Example of Trading Strategy ---
class TransformerSignalStrategy(bt.Strategy):
    params = (
        ('pred_smooth_period', 5),
        ('sma_momentum_period', 50),
        ('show_candlesticks', False),
        ('position_size', 10000),
        ('min_angle_for_entry', 85.0),
    )

    def __init__(self):
        # Indicators
        self.prediction = TransformerPredictionIndicator(self.data)
        self.smoothed_prediction = bt.indicators.SMA(self.prediction, period=self.p.pred_smooth_period)
        self.angle = AngleIndicator(self.data.close, angle_lookback=self.p.pred_smooth_period)
        self.sma_5 = bt.indicators.SMA(self.data.close, period=5)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=100)
        self.sma_momentum = bt.indicators.SMA(self.data.close, period=self.p.sma_momentum_period)
        self.smooth_cross_momentum = bt.indicators.CrossOver(self.smoothed_prediction, self.sma_momentum)

    def next(self):
        # --- Clean Debug Log ---
        # This will only print when a trade is being considered, keeping the log clean.
        if self.smooth_cross_momentum[0] > 0:
             print(f"Date: {self.data.datetime.date(0)} | Crossover detected. Angle is: {self.angle[0]:.2f}°")

        # --- Full original trading logic ---
        if self.position:
            if self.smooth_cross_momentum[0] < 0:
                self.close()
            return

        is_bullish_filter = self.sma_50[0] < self.prediction[0] and self.sma_momentum[0] < self.prediction[0]
        is_strong_momentum = self.smoothed_prediction[0] > self.smoothed_prediction[-1]
        is_crossover_signal = self.smooth_cross_momentum[0] > 0
        is_steep_angle = self.angle[0] > self.p.min_angle_for_entry
        
        # The angle condition is NOT used for entry. This is purely for visualization.
        # is_steep_angle = self.angle[0] > self.p.min_angle_for_entry

        if is_bullish_filter and is_strong_momentum and is_crossover_signal and is_steep_angle:
            print(f"--- BUY SIGNAL @ {self.data.datetime.date(0)} (Angle condition enabled) ---")
            self.buy(size=self.p.position_size)

# --- Cerebro Setup ---
if __name__ == '__main__':
    print("Starting Backtest with Transformer Signal Strategy...")
    cerebro = bt.Cerebro()
    
    cerebro.addstrategy(TransformerSignalStrategy)
    
    data = bt.feeds.GenericCSVData(
        dataname=str(DATA_PATH), dtformat=('%Y%m%d'), tmformat=('%H:%M:%S'),
        datetime=0, time=1, open=2, high=3, low=4, close=5, volume=6,
        timeframe=bt.TimeFrame.Minutes, compression=5)
    
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    
    print("--- Running Backtest in Bar-by-Bar Mode ---")
    cerebro.run()
    
    print("\n--- Backtest Finished. Generating Plot... ---")
    # This will now generate the plot correctly after the bar-by-bar run
    cerebro.plot(style='line')