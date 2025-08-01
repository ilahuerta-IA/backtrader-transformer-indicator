
# -----------------------------------------------------------------------------
# DISCLAIMER:
# This software is for educational and research purposes only.
# It is not intended for live trading or financial advice.
# Trading in financial markets involves substantial risk of loss.
# Use at your own risk. The author assumes no liability for any losses.
# -----------------------------------------------------------------------------

import backtrader as bt
import torch
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# --- WARNING SUPPRESSION (Unchanged) ---
import warnings
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
# --- END WARNING SUPPRESSION ---

# --- GLOBAL PATH CONFIGURATION (Unchanged) ---
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

# --- HELPER FUNCTION (Unchanged) ---
def create_time_features_for_window(dt_index: pd.DatetimeIndex) -> np.ndarray:
    features = pd.DataFrame(index=dt_index)
    features['hour'] = (dt_index.hour / 23.0) - 0.5
    features['day_of_week'] = (dt_index.dayofweek / 6.0) - 0.5
    features['day_of_month'] = ((dt_index.day - 1) / 30.0) - 0.5
    features['month'] = ((dt_index.month - 1) / 11.0) - 0.5
    return features.values

# --- INDICATOR: AI PRICE PREDICTION (Unchanged) ---
class TransformerPredictionIndicator(bt.Indicator):
    lines = ('prediction',)
    params = (('models_dir', str(MODELS_DIR)),)
    plotinfo = dict(subplot=False, plotname='AI Prediction')
    
    def __init__(self):
        super().__init__()
        self.p.models_dir = Path(self.p.models_dir)
        from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
        with open(self.p.models_dir / 'model_config.json', 'r') as f: config = TimeSeriesTransformerConfig.from_dict(json.load(f))
        self.scaler = joblib.load(self.p.models_dir / 'target_scaler.pkl')
        self.model = TimeSeriesTransformerForPrediction(config)
        self.model.load_state_dict(torch.load(self.p.models_dir / 'best_transformer_model.pth', map_location='cpu'))
        self.model.eval()
        self.history_len = config.context_length + max(config.lags_sequence or [0])
        self.addminperiod(self.history_len)

    def next(self):
        datetimes = [self.data.num2date(self.data.datetime[-i]) for i in range(self.history_len)]
        datetimes.reverse(); dt_index = pd.to_datetime(datetimes)
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

# --- INDICATOR: ANGLE OF A LINE (Unchanged) ---
class AngleIndicator(bt.Indicator):
    lines = ('angle',)
    params = (('angle_lookback', 5), ('scale_factor', 50000),)
    plotinfo = dict(subplot=True, plotname='Angle (Degrees)')
    
    def __init__(self):
        self.addminperiod(self.p.angle_lookback)
        super().__init__()

    def next(self):
        rise = (self.data0[0] - self.data0[-self.p.angle_lookback + 1]) * self.p.scale_factor
        run = self.p.angle_lookback
        self.lines.angle[0] = np.degrees(np.arctan2(rise, run))

# --- MAIN TRADING STRATEGY ---
class TransformerSignalStrategy(bt.Strategy):
    params = (
        ('pred_smooth_period', 5),
        ('sma_momentum_period', 50),
        ('sma_long_term_period', 100),
        ('sma_short_term_period', 5),
        ('min_angle_for_entry', 55.0),
        
        # --- NEW PARAMETER ---
        # The maximum allowed absolute divergence between the prediction angle
        # and the price angle to consider the signal "coherent".
        ('max_abs_divergence_entry', 0.30),
        
        ('position_size', 10000),
    )

    def __init__(self):
        # --- Standard Indicators (Unchanged) ---
        self.prediction = TransformerPredictionIndicator(self.data)
        self.smoothed_prediction = bt.indicators.SMA(self.prediction, period=self.p.pred_smooth_period)
        self.sma_short_term = bt.indicators.SMA(self.data.close, period=self.p.sma_short_term_period)
        self.sma_long_term = bt.indicators.SMA(self.data.close, period=self.p.sma_long_term_period)
        self.sma_momentum = bt.indicators.SMA(self.data.close, period=self.p.sma_momentum_period)
        self.smooth_cross_momentum = bt.indicators.CrossOver(self.smoothed_prediction, self.sma_momentum)
        
        # --- Angle Calculations ---
        self.angle_prediction = AngleIndicator(self.smoothed_prediction, angle_lookback=self.p.pred_smooth_period)
        self.angle_price = AngleIndicator(self.sma_short_term, angle_lookback=self.p.sma_short_term_period)
        
        # --- Min/Max Tracking Variables (for final report) ---
        self.max_abs_divergence = 0.0
        self.min_abs_divergence = float('inf')

    def next(self):
        # --- Synchronization Gate ---
        if np.isnan(self.angle_prediction[0]) or np.isnan(self.angle_price[0]):
            return

        # --- Calculate Absolute Divergence and Update Min/Max ---
        divergence = self.angle_prediction[0] - self.angle_price[0]
        abs_divergence = abs(divergence)
        
        self.max_abs_divergence = max(self.max_abs_divergence, abs_divergence)
        if abs_divergence > 0:
            self.min_abs_divergence = min(self.min_abs_divergence, abs_divergence)

        # --- Exit Logic (UNCHANGED) ---
        if self.position:
            if self.smooth_cross_momentum[0] < 0:
                self.close()
            return

        # --- Entry Conditions ---
        is_bullish_filter = (self.sma_long_term[0] < self.prediction[0] and self.sma_momentum[0] < self.prediction[0])
        is_strong_momentum = self.smoothed_prediction[0] > self.smoothed_prediction[-1]
        is_crossover_signal = self.smooth_cross_momentum[0] > 0
        is_steep_angle = self.angle_prediction[0] > self.p.min_angle_for_entry
        
        # --- NEW CONDITION ---
        # Checks if the divergence is below the configured threshold.
        is_coherent_signal = abs_divergence < self.p.max_abs_divergence_entry

        # --- Trade Execution with the new condition ---
        if is_bullish_filter and is_strong_momentum and is_crossover_signal and is_steep_angle and is_coherent_signal:
            print(f"--- BUY SIGNAL @ {self.data.datetime.date(0)} (Angle: {self.angle_prediction[0]:.2f}°, Abs Divergence: {abs_divergence:.2f}° < {self.p.max_abs_divergence_entry}°) ---")
            self.buy(size=self.p.position_size)
    
    def stop(self):
        """
        Called at the end of the backtest to print the final summary.
        """
        print("\n--- Backtest Finished ---")
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        
        print("\n--- Absolute Divergence Angle Analysis ---")
        if self.min_abs_divergence == float('inf'):
            print("No divergence data was calculated.")
        else:
            print(f"  - Minimum Absolute Divergence Recorded: {self.min_abs_divergence:.2f}°")
            print(f"  - Maximum Absolute Divergence Recorded: {self.max_abs_divergence:.2f}°")
        print("----------------------------------------\n")

# --- Cerebro Execution ---
if __name__ == '__main__':
    cerebro = bt.Cerebro(runonce=False)
    cerebro.addstrategy(TransformerSignalStrategy)
    
    data = bt.feeds.GenericCSVData(
        dataname=str(DATA_PATH), dtformat=('%Y%m%d'), tmformat=('%H:%M:%S'),
        datetime=0, time=1, open=2, high=3, low=4, close=5, volume=6,
        timeframe=bt.TimeFrame.Minutes, compression=5)
    
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    
    print("--- Running Backtest ---")
    cerebro.run()
    
    # Optional: uncomment to see the plot.
    print("Generating Plot...")
    cerebro.plot(style='line')