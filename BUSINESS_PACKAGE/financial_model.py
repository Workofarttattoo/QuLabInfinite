#!/usr/bin/env python3
"""
QuLab Infinite — Pre-Seed Financial Model
Corporation of Light | Joshua Hendricks Cole

Run: python BUSINESS_PACKAGE/financial_model.py
Outputs a full 18-month projection to terminal and CSV.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Assumptions (tweak these to run scenarios)
# ---------------------------------------------------------------------------

@dataclass
class Assumptions:
    # Funding
    raise_amount: int = 750_000
    post_money_valuation: int = 5_000_000
    runway_months: int = 18

    # Pricing tiers (monthly)
    starter_price: int = 99
    pro_price: int = 299
    enterprise_price: int = 499

    # Customer acquisition (new customers per month)
    # Ramp: months 1-3 slow, 4-9 medium, 10-18 accelerating
    new_customers_per_month_phase1: int = 3     # months 1-3
    new_customers_per_month_phase2: int = 8     # months 4-9
    new_customers_per_month_phase3: int = 15    # months 10-18

    # Tier mix (must sum to 1.0)
    starter_mix: float = 0.40
    pro_mix: float = 0.40
    enterprise_mix: float = 0.20

    # Churn (monthly)
    monthly_churn_rate: float = 0.05

    # Revenue expansion (upsell existing customers)
    monthly_expansion_rate: float = 0.02  # 2% of existing MRR from upsells

    # One-time / services revenue per quarter
    services_revenue_per_quarter: int = 10_000  # custom lab builds, consulting

    # Costs — Monthly
    founder_salary: int = 8_000
    eng_hire_1_salary: int = 12_000   # starts month 3
    eng_hire_2_salary: int = 12_000   # starts month 6
    scientist_salary: int = 10_000     # starts month 4
    cloud_infra_base: int = 2_000
    cloud_infra_per_customer: int = 50
    marketing_budget_phase1: int = 1_500   # months 1-6
    marketing_budget_phase2: int = 3_000   # months 7-18
    legal_patent_monthly: int = 1_500
    office_tools_misc: int = 1_000

    # CAC
    target_cac: int = 500


@dataclass
class MonthData:
    month: int = 0
    # Customers
    new_customers: int = 0
    churned_customers: int = 0
    total_customers: int = 0
    starter_customers: int = 0
    pro_customers: int = 0
    enterprise_customers: int = 0
    # Revenue
    subscription_mrr: float = 0
    expansion_revenue: float = 0
    services_revenue: float = 0
    total_revenue: float = 0
    # Costs
    salaries: float = 0
    cloud_infra: float = 0
    marketing: float = 0
    legal_patent: float = 0
    misc: float = 0
    total_costs: float = 0
    # Summary
    net_burn: float = 0
    cash_balance: float = 0
    cumulative_revenue: float = 0


def run_model(a: Assumptions) -> List[MonthData]:
    months: List[MonthData] = []
    prev_customers = 0
    prev_starter = 0
    prev_pro = 0
    prev_enterprise = 0
    cash = float(a.raise_amount)
    cumulative_rev = 0.0
    cumulative_mrr = 0.0

    for m in range(1, a.runway_months + 1):
        d = MonthData(month=m)

        # --- New customers ---
        if m <= 3:
            d.new_customers = a.new_customers_per_month_phase1
        elif m <= 9:
            d.new_customers = a.new_customers_per_month_phase2
        else:
            d.new_customers = a.new_customers_per_month_phase3

        new_starter = round(d.new_customers * a.starter_mix)
        new_pro = round(d.new_customers * a.pro_mix)
        new_enterprise = d.new_customers - new_starter - new_pro

        # --- Churn ---
        d.churned_customers = round(prev_customers * a.monthly_churn_rate)
        churn_starter = round(prev_starter * a.monthly_churn_rate)
        churn_pro = round(prev_pro * a.monthly_churn_rate)
        churn_enterprise = d.churned_customers - churn_starter - churn_pro

        # --- Totals ---
        d.starter_customers = max(0, prev_starter + new_starter - churn_starter)
        d.pro_customers = max(0, prev_pro + new_pro - churn_pro)
        d.enterprise_customers = max(0, prev_enterprise + new_enterprise - churn_enterprise)
        d.total_customers = d.starter_customers + d.pro_customers + d.enterprise_customers

        # --- Revenue ---
        d.subscription_mrr = (
            d.starter_customers * a.starter_price +
            d.pro_customers * a.pro_price +
            d.enterprise_customers * a.enterprise_price
        )
        d.expansion_revenue = cumulative_mrr * a.monthly_expansion_rate
        d.services_revenue = a.services_revenue_per_quarter / 3 if a.services_revenue_per_quarter else 0
        d.total_revenue = d.subscription_mrr + d.expansion_revenue + d.services_revenue
        cumulative_mrr = d.subscription_mrr

        # --- Costs ---
        d.salaries = a.founder_salary
        if m >= 3:
            d.salaries += a.eng_hire_1_salary
        if m >= 6:
            d.salaries += a.eng_hire_2_salary
        if m >= 4:
            d.salaries += a.scientist_salary

        d.cloud_infra = a.cloud_infra_base + (d.total_customers * a.cloud_infra_per_customer)
        d.marketing = a.marketing_budget_phase1 if m <= 6 else a.marketing_budget_phase2
        d.legal_patent = a.legal_patent_monthly
        d.misc = a.office_tools_misc

        d.total_costs = d.salaries + d.cloud_infra + d.marketing + d.legal_patent + d.misc

        # --- Summary ---
        d.net_burn = d.total_revenue - d.total_costs
        cumulative_rev += d.total_revenue
        d.cumulative_revenue = cumulative_rev
        cash += d.net_burn
        d.cash_balance = cash

        prev_customers = d.total_customers
        prev_starter = d.starter_customers
        prev_pro = d.pro_customers
        prev_enterprise = d.enterprise_customers

        months.append(d)

    return months


def print_model(months: List[MonthData], a: Assumptions):
    print("=" * 100)
    print("  QULAB INFINITE — 18-MONTH FINANCIAL MODEL")
    print("  Pre-Seed: ${:,}  |  Valuation Cap: ${:,}  |  Dilution: {:.1f}%".format(
        a.raise_amount, a.post_money_valuation,
        (a.raise_amount / a.post_money_valuation) * 100
    ))
    print("=" * 100)

    # Assumptions summary
    print("\n  ASSUMPTIONS")
    print("  " + "-" * 60)
    print(f"  Starter: ${a.starter_price}/mo | Pro: ${a.pro_price}/mo | Enterprise: ${a.enterprise_price}/mo")
    print(f"  Tier mix: {a.starter_mix:.0%} / {a.pro_mix:.0%} / {a.enterprise_mix:.0%}")
    print(f"  Monthly churn: {a.monthly_churn_rate:.0%}  |  Monthly expansion: {a.monthly_expansion_rate:.0%}")
    print(f"  New customers/mo: Phase 1={a.new_customers_per_month_phase1}, "
          f"Phase 2={a.new_customers_per_month_phase2}, Phase 3={a.new_customers_per_month_phase3}")
    blended_arpu = (
        a.starter_price * a.starter_mix +
        a.pro_price * a.pro_mix +
        a.enterprise_price * a.enterprise_mix
    )
    print(f"  Blended ARPU: ${blended_arpu:,.0f}/mo")

    # Monthly table
    print("\n  MONTHLY PROJECTIONS")
    print("  " + "-" * 96)
    header = (
        f"  {'Mo':>3} | {'New':>4} {'Churn':>5} {'Total':>5} | "
        f"{'MRR':>9} {'Expand':>8} {'Total Rev':>10} | "
        f"{'Costs':>9} {'Net':>10} | {'Cash':>11}"
    )
    print(header)
    print("  " + "-" * 96)

    for d in months:
        line = (
            f"  {d.month:>3} | {d.new_customers:>4} {d.churned_customers:>5} {d.total_customers:>5} | "
            f"${d.subscription_mrr:>8,.0f} ${d.expansion_revenue:>7,.0f} ${d.total_revenue:>9,.0f} | "
            f"${d.total_costs:>8,.0f} ${d.net_burn:>9,.0f} | ${d.cash_balance:>10,.0f}"
        )
        print(line)

    # Key milestones
    print("\n  KEY METRICS AT MILESTONES")
    print("  " + "-" * 60)
    for target_month in [6, 12, 18]:
        d = months[target_month - 1]
        arr = d.subscription_mrr * 12
        print(f"\n  Month {target_month}:")
        print(f"    Customers:       {d.total_customers}")
        print(f"    MRR:             ${d.subscription_mrr:,.0f}")
        print(f"    ARR:             ${arr:,.0f}")
        print(f"    Monthly Costs:   ${d.total_costs:,.0f}")
        print(f"    Net Burn:        ${d.net_burn:,.0f}")
        print(f"    Cash Remaining:  ${d.cash_balance:,.0f}")

    # Unit economics
    final = months[-1]
    total_customers_ever = sum(d.new_customers for d in months)
    total_marketing_spend = sum(d.marketing for d in months)
    actual_cac = total_marketing_spend / max(total_customers_ever, 1)
    ltv_months = 1 / a.monthly_churn_rate if a.monthly_churn_rate > 0 else 60
    ltv = blended_arpu * ltv_months
    ltv_cac = ltv / max(actual_cac, 1)

    print("\n  UNIT ECONOMICS")
    print("  " + "-" * 60)
    print(f"  Blended ARPU:          ${blended_arpu:,.0f}/mo")
    print(f"  Avg customer lifetime: {ltv_months:.0f} months")
    print(f"  LTV:                   ${ltv:,.0f}")
    print(f"  Total marketing spend: ${total_marketing_spend:,.0f}")
    print(f"  Total customers added: {total_customers_ever}")
    print(f"  Effective CAC:         ${actual_cac:,.0f}")
    print(f"  LTV:CAC ratio:         {ltv_cac:.1f}x")

    # Runway analysis
    print("\n  RUNWAY ANALYSIS")
    print("  " + "-" * 60)
    zero_cash_month = None
    breakeven_month = None
    for d in months:
        if d.cash_balance <= 0 and zero_cash_month is None:
            zero_cash_month = d.month
        if d.net_burn >= 0 and breakeven_month is None and d.month > 1:
            breakeven_month = d.month

    if zero_cash_month:
        print(f"  Cash runs out:     Month {zero_cash_month}")
    else:
        print(f"  Cash at month 18:  ${final.cash_balance:,.0f}")
        print(f"  Runway extends beyond 18 months")

    if breakeven_month:
        print(f"  Monthly breakeven: Month {breakeven_month}")
    else:
        print(f"  Monthly breakeven: Not reached in 18 months (need Series A)")

    final_arr = final.subscription_mrr * 12
    print(f"\n  End-state ARR:     ${final_arr:,.0f}")
    print(f"  End-state MRR:     ${final.subscription_mrr:,.0f}")
    print(f"  End-state customers: {final.total_customers}")

    # Series A readiness
    print("\n  SERIES A READINESS CHECK (Month 18)")
    print("  " + "-" * 60)
    checks = [
        ("ARR > $100K", final_arr >= 100_000),
        ("MRR > $10K", final.subscription_mrr >= 10_000),
        ("50+ customers", final.total_customers >= 50),
        ("LTV:CAC > 3x", ltv_cac >= 3),
        ("Monthly churn < 10%", a.monthly_churn_rate < 0.10),
        ("Cash remaining > $50K", final.cash_balance > 50_000),
    ]
    for label, passed in checks:
        status = "PASS" if passed else "MISS"
        print(f"  [{status}] {label}")

    all_pass = all(p for _, p in checks)
    print(f"\n  Series A readiness: {'STRONG' if all_pass else 'NEEDS WORK — focus on missed items'}")

    print("\n" + "=" * 100)


def export_csv(months: List[MonthData], filepath: str):
    fields = [
        "month", "new_customers", "churned_customers", "total_customers",
        "starter_customers", "pro_customers", "enterprise_customers",
        "subscription_mrr", "expansion_revenue", "services_revenue", "total_revenue",
        "salaries", "cloud_infra", "marketing", "legal_patent", "misc", "total_costs",
        "net_burn", "cash_balance", "cumulative_revenue"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in months:
            writer.writerow({k: round(getattr(d, k), 2) for k in fields})
    print(f"\n  CSV exported to: {filepath}")


def run_scenarios():
    print("\n\n")
    print("=" * 100)
    print("  SCENARIO COMPARISON")
    print("=" * 100)

    scenarios = {
        "Conservative": Assumptions(
            new_customers_per_month_phase1=2,
            new_customers_per_month_phase2=5,
            new_customers_per_month_phase3=10,
            monthly_churn_rate=0.08,
        ),
        "Base Case": Assumptions(),
        "Optimistic": Assumptions(
            new_customers_per_month_phase1=5,
            new_customers_per_month_phase2=12,
            new_customers_per_month_phase3=22,
            monthly_churn_rate=0.03,
            enterprise_mix=0.30,
            pro_mix=0.40,
            starter_mix=0.30,
        ),
    }

    print(f"\n  {'Metric':<30} {'Conservative':>15} {'Base Case':>15} {'Optimistic':>15}")
    print("  " + "-" * 78)

    results = {}
    for name, assumptions in scenarios.items():
        months = run_model(assumptions)
        final = months[-1]
        total_cust_ever = sum(d.new_customers for d in months)
        total_mktg = sum(d.marketing for d in months)
        blended_arpu = (
            assumptions.starter_price * assumptions.starter_mix +
            assumptions.pro_price * assumptions.pro_mix +
            assumptions.enterprise_price * assumptions.enterprise_mix
        )
        results[name] = {
            "End Customers": final.total_customers,
            "End MRR": final.subscription_mrr,
            "End ARR": final.subscription_mrr * 12,
            "Total Revenue (18mo)": final.cumulative_revenue,
            "Cash Remaining": final.cash_balance,
            "Effective CAC": total_mktg / max(total_cust_ever, 1),
            "Blended ARPU": blended_arpu,
        }

    for metric in results["Base Case"]:
        vals = []
        for name in ["Conservative", "Base Case", "Optimistic"]:
            v = results[name][metric]
            vals.append(f"${v:>13,.0f}" if metric != "End Customers" else f"{v:>14}")
        print(f"  {metric:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")


if __name__ == "__main__":
    assumptions = Assumptions()
    months = run_model(assumptions)
    print_model(months, assumptions)

    csv_path = os.path.join(os.path.dirname(__file__), "financial_projections.csv")
    export_csv(months, csv_path)

    run_scenarios()

    print("\n  Tip: Edit the Assumptions dataclass at the top of this file to model different scenarios.")
    print("  Run with: python BUSINESS_PACKAGE/financial_model.py\n")
