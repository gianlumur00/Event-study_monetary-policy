# Required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from pathlib import Path

# ============================================
# LOAD EVENT DATES
# ============================================

# CSV file containing event dates
# Expected column name: "event.dates"
event_dates_df = pd.read_csv("event_dates.csv")

# Convert to datetime
event_dates = pd.to_datetime(
    event_dates_df["event.dates"],
    format="%d/%m/%Y"
)

num_dates = len(event_dates)

# ============================================
# LOAD DATABASE
# ============================================

# Main database containing stock prices and market index
DB = pd.read_csv(
    "database.csv",
    sep=";",
    decimal=".",
    na_values=["N/A", "#N/D"]
)

# Convert dates
DB["Dates"] = pd.to_datetime(DB["Dates"], format="%d/%m/%Y")

# Number of stocks
# First column = Dates
# Second column = STOXX.600
num_stocks = DB.shape[1] - 2

# ============================================
# COMPUTE DAILY LOG RETURNS
# ============================================

DB_logreturns = DB.copy()

# Compute log returns for all columns except Dates
for col in DB.columns[1:]:
    DB_logreturns[col] = np.log(DB[col]).diff()

# Reset index
DB_logreturns.reset_index(drop=True, inplace=True)

# ============================================
# INITIALIZATION
# ============================================

days = np.arange(-10, 11, 1)

start_window = -1
end_window = 3
window_length = end_window - start_window + 1

# Store all event-study results
results_list = []

# Store cumulative abnormal returns summary
TOTAL_DB_CAR = pd.DataFrame()

# ============================================
# EVENT STUDY LOOP
# ============================================

for i, event_date in enumerate(event_dates):

    # Find event position in the dataset
    event_position = DB_logreturns.index[
        DB_logreturns["Dates"] == event_date
    ][0]

    # DataFrame to store abnormal returns
    total_DB_AR = pd.DataFrame({"days": days})

    # DataFrame to store CAR statistics
    total_DB_CAR_event = pd.DataFrame({
        "EventDates": [event_date]
    })

    # Loop through stocks
    # Columns 2 onward excluding Dates
    for j in range(2, DB_logreturns.shape[1]):

        stock_name = DB_logreturns.columns[j]

        # ============================================
        # ESTIMATION WINDOW
        # ============================================

        estimation_start = event_position - 271
        estimation_end = event_position - 20

        y = DB_logreturns.iloc[
            estimation_start:estimation_end + 1,
            j
        ]

        x = DB_logreturns.loc[
            estimation_start:estimation_end,
            "STOXX.600"
        ]

        # Add constant for alpha
        x = sm.add_constant(x)

        # Market model regression
        model = sm.OLS(y, x, missing="drop").fit()

        alpha = model.params["const"]
        beta = model.params["STOXX.600"]

        residuals_sd = np.std(model.resid, ddof=1)

        # ============================================
        # EVENT WINDOW
        # ============================================

        exp_ret = []
        ab_ret = []
        CAR = []

        cumulative_car = 0

        for z in range(21):

            idx = event_position - 10 + z

            market_return = DB_logreturns.loc[idx, "STOXX.600"]

            expected_return = alpha + beta * market_return

            actual_return = DB_logreturns.iloc[idx, j]

            abnormal_return = actual_return - expected_return

            cumulative_car += abnormal_return

            exp_ret.append(expected_return)
            ab_ret.append(abnormal_return)
            CAR.append(cumulative_car)

        # Convert to arrays
        exp_ret = np.array(exp_ret)
        ab_ret = np.array(ab_ret)
        CAR = np.array(CAR)

        # ============================================
        # TEST STATISTICS
        # ============================================

        t_AR = ab_ret / residuals_sd

        pvalue_AR = 2 * (1 - norm.cdf(np.abs(t_AR)))

        # ============================================
        # STORE DAILY RESULTS
        # ============================================

        new_DB_AR = pd.DataFrame({
            f"E_ret_{stock_name}": exp_ret,
            f"AR_{stock_name}": ab_ret,
            f"t_{stock_name}": t_AR,
            f"pvalue_{stock_name}": pvalue_AR,
            f"CAR_{stock_name}": CAR
        })

        total_DB_AR = pd.concat(
            [total_DB_AR, new_DB_AR],
            axis=1
        )

        # ============================================
        # CUMULATIVE ABNORMAL RETURNS
        # ============================================

        start_idx = 10 + start_window
        end_idx = 10 + end_window + 1

        CAR_window = np.nansum(ab_ret[start_idx:end_idx])

        CAR_sd = np.sqrt(window_length) * residuals_sd

        t_CAR = CAR_window / CAR_sd

        pvalue_CAR = 2 * (1 - norm.cdf(np.abs(t_CAR)))

        new_DB_CAR = pd.DataFrame({
            f"CAR_window_{stock_name}": [CAR_window],
            f"CAR_sd_{stock_name}": [CAR_sd],
            f"t_{stock_name}": [t_CAR],
            f"pvalue_{stock_name}": [pvalue_CAR]
        })

        total_DB_CAR_event = pd.concat(
            [total_DB_CAR_event, new_DB_CAR],
            axis=1
        )

    # Save event-specific abnormal returns table
    results_list.append(total_DB_AR)

    # Append CAR summary
    TOTAL_DB_CAR = pd.concat(
        [TOTAL_DB_CAR, total_DB_CAR_event],
        ignore_index=True
    )

# ============================================
# EXPORT RESULTS
# ============================================

output_folder = Path.cwd()

# Export all AR results into separate Excel sheets
with pd.ExcelWriter(output_folder / "LISTA.xlsx") as writer:

    for idx, df in enumerate(results_list):

        sheet_name = f"DB_AR_{idx + 1}"

        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

# Export CAR summary
TOTAL_DB_CAR.to_excel(
    output_folder / "TOTAL_DB_CAR.xlsx",
    index=False
)

print("Event study completed successfully.")
