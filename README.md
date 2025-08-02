# Backtrader Time Series Transformer Indicator

A ready-to-use example of a custom `backtrader` indicator that integrates a pre-trained Hugging Face `TimeSeriesTransformer` model for price prediction. This repository provides all the necessary components—code, model, and data—to run a backtest and visualize the model's predictions directly on a chart.

## ⚠️ Important Disclaimer

**This project is for educational and research purposes only. It is not financial advice.**

The software, models, and data provided in this repository are intended to demonstrate concepts in algorithmic trading and machine learning. They are not designed or tested for use in live trading environments.

- **High Risk:** Trading in financial markets is inherently risky. Any use of this code, or concepts derived from it, for live trading is done at your own risk. You are solely responsible for any financial losses you may incur.
- **No Warranty:** This software is provided "as is" without warranty of any kind, express or implied. The author makes no guarantees about its performance, accuracy, or profitability.
- **Regulatory Status (EU AI Act):** The AI models within this repository could be subject to regulations such as the EU AI Act if used in a commercial or live trading context. By downloading or using this code, you acknowledge that you are responsible for ensuring your own compliance with all applicable laws and regulations.
- **Not for Professional Use:** This is not a commercial-grade trading tool. Do not use this software with real money.

## Features

*   **`TransformerPredictionIndicator`**: A custom `bt.Indicator` that loads and runs a `TimeSeriesTransformerForPrediction` model on each bar.
*   **Pre-trained Model Included**: Comes with a pre-trained model for EUR/USD 5-minute data, along with the necessary configuration and scaler files.
*   **Example Strategy**: Includes a `TransformerSignalStrategy` that demonstrates how to use the prediction indicator alongside other standard indicators (like SMAs and Crossovers) to generate trading signals.
*   **Sample Data**: A sample `EURUSD_5m_2Mon.csv` data file is provided so you can run the example immediately.
*   **Self-Contained Script**: The main script `transformer_signal_strategy.py` can be run directly to produce a `backtrader` plot.

## File Structure

To run this project, your files should be organized in the following structure:

```
.
├── data/
│   └── EURUSD_5m_2Mon.csv
├── src/
│   ├── ml_models/
│   │   ├── best_transformer_model.pth
│   │   ├── model_config.json
│   │   └── target_scaler.pkl
│   └── strategies/
│       └── transformer_signal_strategy.py
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

It is highly recommended to use a Python virtual environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ilahuerta-IA/backtrader-transformer-indicator.git
    cd backtrader-transformer-indicator
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    You will need to create a `requirements.txt` file with the dependencies. Then, run:
    ```bash
    pip install -r requirements.txt
    ```

    The contents of your `requirements.txt` file should be:
    ```text
    backtrader
    torch
    transformers
    scikit-learn
    pandas
    numpy
    joblib
    ```

## Usage

Once the installation is complete, you can run the example directly from the root directory of the project:

```bash
python src/strategies/transformer_signal_strategy.py
```

This will execute the backtest and, upon completion, generate and display a `backtrader` plot. The plot will show the price data, the AI model's predictions, and the other indicators used in the sample strategy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
