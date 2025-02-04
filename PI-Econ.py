#!/usr/bin/env python3
"""
An improved single-file Python script containing three calculators:
1) Worklife Adjustment Calculator
2) Adjusted Earnings Factor (AEF) Calculator
3) Earnings Calculator (with Residual Offset)

Incorporates recommended improvements:
    - Type hints
    - Separation of input from core logic
    - Better validation & error handling
    - Configuration constants
    - Logging instead of prints
    - Consistent docstrings
    - Basic unit tests
    - Graceful file overwrite checks
    - Consistent data formatting and partial-year calculation
    - Defaults for missing user inputs
"""

import logging
import os
import sys
import pandas as pd

from typing import Tuple, Optional
from dateutil.relativedelta import relativedelta
from datetime import datetime

# -------------------------------------------------------------------
#                       CONFIG & LOGGING
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Default values and constants used throughout
DEFAULTS = {
    "gross_earnings_base": 100.0,   # As a percentage
    "growth_rate": 3.5,            # %
    "discount_rate": 5.0,          # %
    "adjusted_earnings_factor": 88.54,  # e.g., 88.54%
    "starting_wage_base": 46748.0,
    "residual_earning_capacity": 0.0,
    "worklife_years": 40.0         # Example WLE
}


# -------------------------------------------------------------------
#                       HELPER FUNCTIONS
# -------------------------------------------------------------------

def safe_float_input(prompt: str, allow_blank: bool = False,
                     default: Optional[float] = None,
                     minimum: Optional[float] = None,
                     maximum: Optional[float] = None) -> float:
    """
    Prompts user for a float input, with optional blank acceptance (returns default).
    Can enforce optional min/max constraints.
    """
    while True:
        user_input = input(prompt).strip()
        if allow_blank and user_input == "" and default is not None:
            return default

        try:
            value = float(user_input)
            if minimum is not None and value < minimum:
                logging.warning(f"Value cannot be less than {minimum}.")
                continue
            if maximum is not None and value > maximum:
                logging.warning(f"Value cannot be greater than {maximum}.")
                continue
            return value
        except ValueError:
            logging.error("Invalid numeric input. Please try again.")


def confirm_file_overwrite(file_path: str) -> bool:
    """
    If file_path exists, ask the user whether to overwrite it.
    Returns True if overwriting is allowed, or False otherwise.
    """
    if os.path.exists(file_path):
        logging.warning(f"File '{file_path}' already exists.")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != "y":
            logging.info("File will not be overwritten.")
            return False
    return True


# -------------------------------------------------------------------
#           WORKLIFE ADJUSTMENT CALCULATOR (LOGIC)
# -------------------------------------------------------------------

def calculate_worklife_factor_logic(wle_years: float, yfs_years: float) -> Optional[float]:
    """
    Compute the Worklife Factor as a percentage = (WLE / YFS) * 100.

    Returns:
        A float (Worklife Factor in %) or None if invalid input.
    """
    if yfs_years == 0:
        logging.error("Years to Final Separation (YFS) cannot be zero.")
        return None
    return (wle_years / yfs_years) * 100.0


# -------------------------------------------------------------------
#           AEF CALCULATOR (LOGIC)
# -------------------------------------------------------------------

def calculate_aef_logic(
    gross_earnings_base: float,
    worklife_adjustment: float,
    unemployment_factor: float,
    fringe_benefit: float,
    tax_liability: float,
    wrongful_death: bool = False,
    personal_type: str = "",
    personal_percentage: float = 0.0
) -> Tuple[pd.DataFrame, float]:
    """
    Perform AEF (Adjusted Earnings Factor) calculation.

    Parameters
    ----------
    gross_earnings_base: float
        The starting gross earnings base (default 100%)
    worklife_adjustment: float
        Worklife Adjustment in percent (0-100)
    unemployment_factor: float
        Unemployment Factor in percent (0-100)
    fringe_benefit: float
        Fringe Benefit in percent (0-100)
    tax_liability: float
        Tax Liability in percent (0-100)
    wrongful_death: bool
        Indicates if personal maintenance/consumption is to be applied
    personal_type: str
        "personal maintenance" or "personal consumption"
    personal_percentage: float
        The percentage for personal maintenance/consumption

    Returns
    -------
    df: pd.DataFrame
        Step-by-step breakdown of the calculation
    total_factor: float
        The final AEF percentage
    """
    # Convert to decimal for math
    GE = gross_earnings_base
    WLE = worklife_adjustment / 100.0
    UF = unemployment_factor / 100.0
    FB = fringe_benefit / 100.0
    TL = tax_liability / 100.0
    PC = personal_percentage / 100.0 if wrongful_death else 0.0

    base_adjustment = GE * WLE * (1 - UF)
    fringe_adjusted = base_adjustment * (1 + FB)
    tax_adjustment = base_adjustment * TL
    final_adjusted = (fringe_adjusted - tax_adjustment) * (1 - PC)

    total_factor = round((final_adjusted / GE) * 100, 2)

    # Build step table dynamically
    steps = [
        ("Gross Earnings Base", "100.00%"),
        ("x WorkLife Adjustment", f"{worklife_adjustment:.2f}%"),
        ("x (1 - Unemployment Factor)", f"{100 - unemployment_factor:.2f}%"),
        ("= Adjusted Base Earnings", f"{base_adjustment:.2f}%"),
        ("x (1 - Tax Liability)", f"{100 - tax_liability:.2f}%"),
        ("x (1 + Fringe Benefit)", f"{100 + fringe_benefit:.2f}%"),
    ]

    if wrongful_death and personal_percentage > 0:
        steps.append((
            "x (1 - Personal Maintenance/Consumption)",
            f"{100 - personal_percentage:.2f}% ({personal_type.capitalize()})"
        ))

    steps.append(("= Fringe Benefits/Tax Adjusted Earnings Base", f"{fringe_adjusted:.2f}%"))
    steps.append(("AEF (Adjusted Earnings Factor)", f"{total_factor:.2f}%"))

    df = pd.DataFrame({"Step": [s[0] for s in steps], "Percentage": [s[1] for s in steps]})
    return df, total_factor


# -------------------------------------------------------------------
#           EARNINGS CALCULATOR (LOGIC)
# -------------------------------------------------------------------

def compute_earnings_table(
    start_date: str,
    end_date: str,
    wage_base: float,
    residual_base: float,
    growth_rate: float,
    discount_rate: float,
    adjustment_factor: float,
    date_of_birth: Optional[str] = None,
    reference_start: Optional[str] = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Computes a year-by-year earnings table (with partial years),
    offset by any residual earning capacity, yielding a net wage base.

    We compute partial years using date differences with dateutil.relativedelta,
    which handles months/days more precisely than naive (days/365.0).

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    wage_base : float
        Starting projected wage base
    residual_base : float
        Starting residual earning capacity
    growth_rate : float
        Annual growth rate in % (e.g., 3.5 means 3.5%)
    discount_rate : float
        Annual discount rate in % (e.g., 5.0 means 5%)
    adjustment_factor : float
        AEF in % (e.g., 88.54 means 88.54%)
    date_of_birth : str, optional
        For age calculation, format "YYYY-MM-DD"
    reference_start : str, optional
        Date from which discounting begins (YYYY-MM-DD).
        If None, defaults to start_date

    Returns
    -------
    df : pd.DataFrame
        Table with columns [Year, Age, Portion of Year (%), Wage Base,
                            Gross Earnings, Adjusted Earnings, Present Value]
    final_main_wage_base : float
        The wage_base after annual compounding up to end_date
    final_residual_base : float
        The residual_base after annual compounding up to end_date
    """
    # Convert input strings to date objects
    fmt = "%Y-%m-%d"
    start_dt = datetime.strptime(start_date, fmt)
    end_dt = datetime.strptime(end_date, fmt)
    dob_dt = datetime.strptime(date_of_birth, fmt) if date_of_birth else None
    ref_start_dt = datetime.strptime(reference_start, fmt) if reference_start else start_dt

    # Convert user-friendly percentages to decimals
    growth_decimal = growth_rate / 100.0
    discount_decimal = discount_rate / 100.0
    adj_decimal = adjustment_factor / 100.0

    data = {
        "Year": [],
        "Age": [],
        "Portion of Year (%)": [],
        "Wage Base": [],
        "Gross Earnings": [],
        "Adjusted Earnings": [],
        "Present Value": []
    }

    current_year = start_dt.year
    current_wage_base = wage_base
    current_residual_base = residual_base

    final_main_wage_base = wage_base
    final_residual_base = residual_base

    # Build year segments until we surpass end_date
    segment_start = start_dt

    while True:
        # A normal "year" goes from Jan 1 to Dec 31 of current_year,
        # but we must clamp it by end_dt or segment_start
        year_start = datetime(year=current_year, month=1, day=1)
        # Align the first segment to segment_start
        if year_start < segment_start:
            year_start = segment_start

        year_end = datetime(year=current_year, month=12, day=31)
        # Clamp by end_dt
        if year_end > end_dt:
            year_end = end_dt

        # If the start is beyond the end, break
        if year_start > year_end:
            break

        # Compute portion of the year using relativedelta
        rel = relativedelta(year_end, year_start)
        # Approximate fraction of a year: months/12 + days/365 ...
        portion_of_year = rel.years + rel.months / 12.0 + rel.days / 365.0
        # It's possible that we cross from segment_start. Let's verify we start each segment from 'year_start' = segment_start if it's later

        # Age calculation
        if dob_dt:
            age_this_year = relativedelta(year_start, dob_dt).years
        else:
            age_this_year = ""

        # Net wage base = main wage minus residual capacity
        net_wage_base = max(current_wage_base - current_residual_base, 0.0)

        # Earnings for partial year
        gross_earnings = net_wage_base * portion_of_year
        adjusted_earnings = gross_earnings * adj_decimal

        # Discount from reference_start
        years_from_ref = (year_start - ref_start_dt).days / 365.25  # or a more robust approach with relativedelta
        present_value = adjusted_earnings / ((1 + discount_decimal) ** years_from_ref)

        data["Year"].append(current_year)
        data["Age"].append(age_this_year)
        data["Portion of Year (%)"].append(f"{portion_of_year * 100:.2f}%")
        data["Wage Base"].append(f"${net_wage_base:,.2f}")
        data["Gross Earnings"].append(f"${gross_earnings:,.2f}")
        data["Adjusted Earnings"].append(f"${adjusted_earnings:,.2f}")
        data["Present Value"].append(f"${present_value:,.2f}")

        # Grow wage bases for next year
        current_wage_base *= (1.0 + growth_decimal)
        current_residual_base *= (1.0 + growth_decimal)

        # If we reached end_dt exactly, store final and break
        if year_end == end_dt:
            final_main_wage_base = current_wage_base
            final_residual_base = current_residual_base
            break

        # Move to the next calendar year
        current_year += 1
        segment_start = year_end + relativedelta(days=1)

    df = pd.DataFrame(data)
    return df, final_main_wage_base, final_residual_base


# -------------------------------------------------------------------
#                       CALCULATOR MENUS
# -------------------------------------------------------------------

def run_worklife_calculator() -> None:
    """Interactive flow for the Worklife Adjustment Calculator."""
    logging.info("Running Worklife Adjustment Calculator...")
    wle = safe_float_input("Enter the Worklife Expectancy (WLE) years: ", minimum=0)
    yfs = safe_float_input("Enter the Years to Final Separation (YFS): ", minimum=0)

    result = calculate_worklife_factor_logic(wle, yfs)
    if result is not None:
        logging.info(f"Worklife Factor: {result:.2f}%")

def run_aef_calculator() -> None:
    """Interactive flow for the Adjusted Earnings Factor (AEF) Calculator."""
    logging.info("Running AEF Calculator...")

    # Gather inputs with defaults if user leaves blank
    gross_earnings_base = safe_float_input(
        f"Gross Earnings Base (default {DEFAULTS['gross_earnings_base']}%): ",
        allow_blank=True,
        default=DEFAULTS['gross_earnings_base'],
        minimum=0,
        maximum=100
    )
    wla = safe_float_input("Worklife Adjustment (%): ", minimum=0, maximum=100)
    uf = safe_float_input("Unemployment Factor (%): ", minimum=0, maximum=100)
    fb = safe_float_input("Fringe Benefit (%): ", minimum=0, maximum=100)
    tl = safe_float_input("Tax Liability (%): ", minimum=0, maximum=100)

    wd = input("Is this a wrongful death matter? (yes/no): ").strip().lower() == "yes"
    personal_type = ""
    personal_percentage = 0.0
    if wd:
        personal_type = input("Is it 'personal maintenance' or 'personal consumption'?: ").strip().lower()
        personal_percentage = safe_float_input(
            f"Enter the percentage for {personal_type} (%): ",
            minimum=0,
            maximum=100
        )

    # Compute
    df, total_factor = calculate_aef_logic(
        gross_earnings_base,
        wla,
        uf,
        fb,
        tl,
        wrongful_death=wd,
        personal_type=personal_type,
        personal_percentage=personal_percentage
    )

    # Show result
    logging.info("\nAdjusted Earnings Factor Calculation:\n" + df.to_string(index=False))
    logging.info(f"Final AEF: {total_factor:.2f}%")

    # Save to CSV
    csv_save_path = "adjusted_earnings_factor_calculation.csv"
    if confirm_file_overwrite(csv_save_path):
        df.to_csv(csv_save_path, index=False)
        logging.info(f"Saved calculation to '{csv_save_path}'.")


def run_earnings_calculator() -> None:
    """Interactive flow for the Post-Trial Earnings Calculator (With Residual Offset)."""
    logging.info("Running Earnings Calculator...")

    first_name = input("Enter the first name: ").strip()
    last_name = input("Enter the last name: ").strip()
    date_of_birth = input("Enter the date of birth (YYYY-MM-DD) or blank: ").strip() or ""
    date_of_injury = input("Enter the date of injury (YYYY-MM-DD): ").strip()
    date_of_report = input("Enter the date of report (YYYY-MM-DD): ").strip()

    # Collect or default some numeric fields
    growth_rate = safe_float_input(
        f"Future growth rate in % (default {DEFAULTS['growth_rate']}): ",
        allow_blank=True,
        default=DEFAULTS['growth_rate'],
        minimum=0
    )
    discount_rate = safe_float_input(
        f"Discount rate in % (default {DEFAULTS['discount_rate']}): ",
        allow_blank=True,
        default=DEFAULTS['discount_rate'],
        minimum=0
    )
    aef = safe_float_input(
        f"Adjusted Earnings Factor in % (default {DEFAULTS['adjusted_earnings_factor']}): ",
        allow_blank=True,
        default=DEFAULTS['adjusted_earnings_factor'],
        minimum=0,
        maximum=100
    )
    swb = safe_float_input(
        f"Starting wage base (default {DEFAULTS['starting_wage_base']}): ",
        allow_blank=True,
        default=DEFAULTS['starting_wage_base'],
        minimum=0
    )

    residual_answer = input("Is there residual earning capacity? (y/n): ").strip().lower() == "y"
    if residual_answer:
        srb = safe_float_input(
            f"Starting residual earning capacity (default {DEFAULTS['residual_earning_capacity']}): ",
            allow_blank=True,
            default=DEFAULTS['residual_earning_capacity'],
            minimum=0
        )
    else:
        srb = 0.0

    wle_years = safe_float_input(
        f"Work Life Expectancy (default {DEFAULTS['worklife_years']}): ",
        allow_blank=True,
        default=DEFAULTS['worklife_years'],
        minimum=0
    )

    # Derive "retirement" date from date_of_report + WLE
    # We'll do that with relativedelta: WLE in years => build a relativedelta
    fmt = "%Y-%m-%d"
    try:
        report_timestamp = datetime.strptime(date_of_report, fmt)
    except ValueError:
        logging.error("Invalid date_of_report. Cannot proceed.")
        return

    # Build relativedelta to approximate WLE in years
    # If WLE is, say, 12.7 years, we can take the integer portion as years
    # and the fractional portion as months
    wle_int = int(wle_years)  # integer years
    wle_fraction = wle_years - wle_int
    # ~ fraction * 12 = approximate months
    wle_months = int(round(wle_fraction * 12))

    # Retirement date = date_of_report + (years + months)
    estimated_retirement_dt = report_timestamp + relativedelta(years=wle_int, months=wle_months)
    retirement_str = estimated_retirement_dt.strftime(fmt)

    # Calculate the two tables
    pre_injury_df, final_wage_pre_injury, final_resid_pre_injury = compute_earnings_table(
        start_date=date_of_injury,
        end_date=date_of_report,
        wage_base=swb,
        residual_base=srb,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        adjustment_factor=aef,
        date_of_birth=date_of_birth,
        reference_start=date_of_injury  # discount from injury date
    )

    post_injury_df, final_wage_post_injury, final_resid_post_injury = compute_earnings_table(
        start_date=date_of_report,
        end_date=retirement_str,
        wage_base=final_wage_pre_injury,
        residual_base=final_resid_pre_injury,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        adjustment_factor=aef,
        date_of_birth=date_of_birth,
        reference_start=date_of_report  # discount from report date
    )

    # Summaries
    def sum_currency_column(values) -> float:
        return sum(float(v.replace("$", "").replace(",", "")) for v in values)

    pre_injury_adj_total = sum_currency_column(pre_injury_df["Adjusted Earnings"])
    pre_injury_pv_total = sum_currency_column(pre_injury_df["Present Value"])

    post_injury_adj_total = sum_currency_column(post_injury_df["Adjusted Earnings"])
    post_injury_pv_total = sum_currency_column(post_injury_df["Present Value"])

    # Output file
    file_name = f"{first_name}_{last_name}_Earnings_Tables.xlsx"
    file_path = os.path.join(os.getcwd(), file_name)

    logging.info(f"Calculated retirement date based on WLE: {retirement_str}")

    if confirm_file_overwrite(file_path):
        with pd.ExcelWriter(file_path) as writer:
            # Pre-Injury
            pre_injury_df.to_excel(writer, index=False, sheet_name="Pre-Injury")
            summary_pre = pd.DataFrame({
                "Year": ["Total"],
                "Age": [""],
                "Portion of Year (%)": [""],
                "Wage Base": [""],
                "Gross Earnings": [""],
                "Adjusted Earnings": [f"${pre_injury_adj_total:,.2f}"],
                "Present Value": [f"${pre_injury_pv_total:,.2f}"]
            })
            summary_pre.to_excel(
                writer,
                index=False,
                header=False,
                sheet_name="Pre-Injury",
                startrow=len(pre_injury_df) + 2
            )

            # Post-Injury
            post_injury_df.to_excel(writer, index=False, sheet_name="Post-Injury")
            summary_post = pd.DataFrame({
                "Year": ["Total"],
                "Age": [""],
                "Portion of Year (%)": [""],
                "Wage Base": [""],
                "Gross Earnings": [""],
                "Adjusted Earnings": [f"${post_injury_adj_total:,.2f}"],
                "Present Value": [f"${post_injury_pv_total:,.2f}"]
            })
            summary_post.to_excel(
                writer,
                index=False,
                header=False,
                sheet_name="Post-Injury",
                startrow=len(post_injury_df) + 2
            )

        logging.info(f"Export successful! The file has been saved as '{file_path}'.")
    else:
        logging.info("Earnings tables were not saved.")


# -------------------------------------------------------------------
#                       UNIT TESTS (#9)
# -------------------------------------------------------------------

def run_tests() -> None:
    """
    Runs a small set of basic checks to ensure logic functions properly.
    """
    logging.info("---- Running Basic Tests ----")

    # 1) Worklife Factor
    wlf = calculate_worklife_factor_logic(20, 40)
    assert wlf == 50.0, "Expected 50.0 for WLE=20, YFS=40"

    # 2) AEF Calculation
    df_aef, factor_aef = calculate_aef_logic(
        gross_earnings_base=100,
        worklife_adjustment=30,
        unemployment_factor=5,
        fringe_benefit=3,
        tax_liability=20,
        wrongful_death=True,
        personal_type="personal consumption",
        personal_percentage=10
    )
    # Just check final factor is in a plausible range
    assert factor_aef > 0 and factor_aef < 100, "AEF factor should be between 0 and 100"

    # 3) Earnings Table
    # Minimal check: a 1-year table with 0 growth, 0 discount
    df_earn, fin_wage, fin_resid = compute_earnings_table(
        start_date="2025-01-01",
        end_date="2025-12-31",
        wage_base=50000,
        residual_base=20000,
        growth_rate=0,
        discount_rate=0,
        adjustment_factor=100,
        date_of_birth=None,
        reference_start=None
    )
    # Single line for 2025
    assert len(df_earn) == 1, "Should have only one row for single-year range"
    # Check net wage base = 30000 for that year
    assert "$30,000.00" in df_earn["Wage Base"].iloc[0], "Expected $30,000.00 wage base"

    logging.info("All basic tests passed!")


# -------------------------------------------------------------------
#                           MAIN
# -------------------------------------------------------------------

def main() -> None:
    """
    Main entry point. If 'test' is in sys.argv, run tests. Otherwise,
    prompt the user to pick a calculator.
    """
    if "test" in sys.argv:
        run_tests()
        return

    logging.info("Which calculator would you like to run?")
    print("1) Worklife Adjustment Calculator")
    print("2) Adjusted Earnings Factor (AEF) Calculator")
    print("3) Earnings Calculator (With Residual Offset)")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        run_worklife_calculator()
    elif choice == "2":
        run_aef_calculator()
    elif choice == "3":
        run_earnings_calculator()
    else:
        logging.error("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
