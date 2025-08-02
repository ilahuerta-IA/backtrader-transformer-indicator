
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

# --- HELPER FUNCTION and INDICATORS (All Unchanged) ---
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

# --- MAIN TRADING STRATEGY with DIVERGENCE ---
class TransformerSignalStrategy(bt.Strategy):
    params = (
        ('pred_smooth_period', 5),
        ('sma_momentum_period', 50),
        ('sma_long_term_period', 100),
        ('sma_short_term_period', 5),
        ('min_angle_for_entry', 55.0),
        ('max_abs_divergence_entry', 0.1),
        ('position_size', 10000),
    )

    def __init__(self):
        self.prediction = TransformerPredictionIndicator(self.data)
        self.smoothed_prediction = bt.indicators.SMA(self.prediction, period=self.p.pred_smooth_period)
        self.sma_short_term = bt.indicators.SMA(self.data.close, period=self.p.sma_short_term_period)
        self.sma_long_term = bt.indicators.SMA(self.data.close, period=self.p.sma_long_term_period)
        self.sma_momentum = bt.indicators.SMA(self.data.close, period=self.p.sma_momentum_period)
        self.smooth_cross_momentum = bt.indicators.CrossOver(self.smoothed_prediction, self.sma_momentum)
        
        self.angle_prediction = AngleIndicator(self.smoothed_prediction, angle_lookback=self.p.pred_smooth_period)
        self.angle_price = AngleIndicator(self.sma_short_term, angle_lookback=self.p.sma_short_term_period)

    def next(self):
        if np.isnan(self.angle_prediction[0]) or np.isnan(self.angle_price[0]):
            return

        if self.position:
            if self.smooth_cross_momentum[0] < 0:
                self.close()
            return

        abs_divergence = abs(self.angle_prediction[0] - self.angle_price[0])
        
        is_bullish_filter = (self.sma_long_term[0] < self.prediction[0] and self.sma_momentum[0] < self.prediction[0])
        is_strong_momentum = self.smoothed_prediction[0] > self.smoothed_prediction[-1]
        is_crossover_signal = self.smooth_cross_momentum[0] > 0
        is_steep_angle = self.angle_prediction[0] > self.p.min_angle_for_entry
        is_coherent_signal = abs_divergence < self.p.max_abs_divergence_entry

        if is_bullish_filter and is_strong_momentum and is_crossover_signal and is_steep_angle and is_coherent_signal:
            self.buy(size=self.p.position_size)
    
    def stop(self):
        # The 'stop' method is run for each optimization pass.
        # We can add a print here to see the progress.
        pass

# --- Cerebro Execution for OPTIMIZATION ---
if __name__ == '__main__':
    # We force bar-by-bar execution for the optimization runs. This ensures
    # our custom `AngleIndicator` with its Python `next()` method is calculated
    # correctly for every single bar in every run. It will be slower, but correct.
    cerebro = bt.Cerebro(runonce=False, optreturn=False)
    
    # --- Add the strategy for optimization ---
    cerebro.optstrategy(
        TransformerSignalStrategy,
        # Define the ranges of values to test for the key parameters
        #min_angle_for_entry=[65, 70],                      # Test angles: 65, 70
        #max_abs_divergence_entry=np.arange(0.1, 0.5, 0.2) # Test divergences: 0.1, 0.3
        min_angle_for_entry=range(65, 90, 5),      # Test angles: 75, 80, 85
        #max_abs_divergence_entry=np.arange(0.1, 0.6, 0.2) # Test divergences: 0.1, 0.3, 0.5
    )
    
    # --- Add Analyzers to measure performance ---
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Load data
    data = bt.feeds.GenericCSVData(
        dataname=str(DATA_PATH), dtformat=('%Y%m%d'), tmformat=('%H:%M:%S'),
        datetime=0, time=1, open=2, high=3, low=4, close=5, volume=6,
        timeframe=bt.TimeFrame.Minutes, compression=5)
    
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    
    print("--- Running Strategy Optimization ---")
    # cerebro.run() returns a list of lists, where each inner list contains the results of one run.
    optimization_results = cerebro.run()
    
    # --- Process and Print the Results ---
    final_results_list = []
    for single_run_results in optimization_results:
        for strategy_result in single_run_results:
            # Access parameters for this run
            params = strategy_result.p
            
            # Access analyzers for this run
            trade_analysis = strategy_result.analyzers.tradeanalyzer.get_analysis()
            return_analysis = strategy_result.analyzers.returns.get_analysis()
            
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            
            # Calculate Profit Factor
            profit_factor = 0.0
            if 'won' in trade_analysis and 'lost' in trade_analysis:
                total_won = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
                total_lost = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))
                if total_lost > 0:
                    profit_factor = total_won / total_lost

            final_results_list.append({
                'min_angle': params.min_angle_for_entry,
                'max_divergence': params.max_abs_divergence_entry,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                # The final portfolio value is in the 'returns' analyzer
                'final_value': return_analysis.get('rtot', 1) * cerebro.broker.startingcash
            })

    # Sort results to find the best combination
    sorted_results = sorted(final_results_list, key=lambda x: x['profit_factor'], reverse=True)
    
    print("\n--- Top Parameter Combinations by Profit Factor ---")
    print(f"{'Min Angle':<12} {'Max Divergence':<15} {'Profit Factor':<15} {'Total Trades':<15} {'Final Value':<15}")
    print("-" * 75)
    for res in sorted_results:
        print(f"{res['min_angle']:<12} {res['max_divergence']:<15.2f} {res['profit_factor']:<15.2f} {res['total_trades']:<15} {res['final_value']:<15.2f}")