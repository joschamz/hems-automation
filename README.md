[![Shipping files](https://github.com/joschamz/hems-automation/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/joschamz/hems-automation/actions/workflows/workflow-02.yml)

# Home Energy Management System (HEMS) - Solar Energy Forecasting



## Set up your Environment



### **`macOS`** type the following commands : 

- Install the OpenMP runtime required by `lightgbm`:

    ```BASH
    brew install libomp
    ```

- Install the virtual environment and required packages with the setup script:

    ```bash
    ./scripts/setup.sh
    source .venv/bin/activate
    ```

- Manual setup (alternative):

    ```bash
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```

If the notebook kernel was already running before `libomp` was installed, restart the kernel once before importing `lightgbm` again.
    
### **`WindowsOS`** type the following commands :

If you use the `pyenv` commands below on Windows, install and configure `pyenv-win` first.

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

Run notebooks from the project root so relative paths to `data/`, `models/`, and `user_config.json` resolve correctly.

Detailed usage examples are documented in notebooks:
- Solar utility usage: `notebooks/solar_forecast.ipynb`
- Price utility usage: `notebooks/Example_prices_utility.ipynb`

```bash
jupyter lab
```

## Solar Forecast Utility

You can now call one shared function from any notebook to get one full day of
solar prediction in 15-minute steps (96 rows) as per-interval `predicted_kwh`.

```python
from utils.solar_utils import get_daily_solar_kwh

# Default: tomorrow
tomorrow_df = get_daily_solar_kwh()

# Explicit forecast date
future_df = get_daily_solar_kwh(target_date="2026-03-15")

# Explicit historical date
past_df = get_daily_solar_kwh(target_date="2021-06-21")
```

The function reads system parameters from `user_config.json` by default and uses
Open-Meteo APIs. If API data cannot be retrieved, it falls back to a local
clear-sky approximation while keeping the same output shape.
Timestamps in the `time` column are returned in UTC.

## Day-Ahead Price Utility

You can fetch ENTSOE day-ahead electricity prices (DE_LU bidding zone) for any
date — past or future — using the same 96-row, 15-minute UTC format as the
solar utility.

`get_daily_prices()` always returns a strict UTC calendar day: first row at
`00:00:00+00:00`, last row at `23:45:00+00:00`.

Place your ENTSOE API key in `secrets/entsoe_api_key.txt` (UUID format). The `secrets/` folder is tracked, but files inside are ignored by git.
Register for a free key at [transparency.entsoe.eu](https://transparency.entsoe.eu).

```python
from utils.prices_utils import get_daily_prices

# Default: tomorrow's prices
tomorrow_df = get_daily_prices()

# Historical prices (any past date)
past_df = get_daily_prices("2025-01-15", mode="historical")

# Explicit future / forecast date
future_df = get_daily_prices("2026-03-15", mode="forecast")
```

Output columns (96 rows, 15-min UTC intervals):

| Column | Description |
|---|---|
| `time` | UTC timestamp (timezone-aware, matches solar `time` column) |
| `price_eur_mwh` | Day-ahead price in EUR/MWh |
| `price_cent_kwh` | Same price in ct/kWh (`price_eur_mwh / 10`) |
| `source` | `"entsoe_api"` · `"not_published"` · `"fallback_unavailable"` |

When tomorrow's prices have not been published yet (ENTSOE typically publishes
D+1 prices around 13:00 CET), the function returns 96 NaN rows with
`source="not_published"` instead of raising an error.

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


---

## Handling Merge Conflicts in Jupyter Notebooks

When working in teams, `.ipynb` files can cause messy merge conflicts because they’re JSON-based.  
We use **nbdime** to make this easy.

### Setup (run once)
```bash
nbdime config-git --enable
```

### When a conflict happens
```bash
nbdime mergetool
```

A web interface will open showing both notebook versions side by side.
Choose what to keep, save and close tool, then:
```bash
git add your_notebook.ipynb
git commit -m "Resolved notebook conflict"
```
That’s it — clean merges for notebooks!