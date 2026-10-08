"""Synthetic small-business bank-transaction descriptions.

Everything here is SYNTHETIC. Merchant/counterparty names are invented or are
generic payment-processor style strings; no real customer data is used.

The generator is built to be *realistically messy*, not trivially separable:
  * bank-style prefixes (ACH DEBIT, POS PURCHASE, WEB PMT, ...), reference
    numbers, dates, store numbers, state codes;
  * truncation to a fixed descriptor width, random casing, character typos;
  * a share of rows get a *generic* descriptor ("ONLINE PAYMENT THANK YOU")
    that carries little category signal -> irreducible error, which is what
    makes confidence-based routing meaningful;
  * counterparty pools are split into SEEN (train/val/test_seen) and UNSEEN
    (test_unseen only) so we can measure generalisation to new merchants.
"""
from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CATEGORIES = [
    "payroll",
    "rent",
    "loan_repayment",
    "mca_remittance",
    "nsf_fee",
    "card_revenue",
    "customer_payment",
    "transfer",
    "utilities",
    "tax_payment",
    "supplier_payment",
    "insurance",
]

# direction: +1 credit (money in), -1 debit (money out), 0 either
DIRECTION = {
    "payroll": -1, "rent": -1, "loan_repayment": -1, "mca_remittance": -1,
    "nsf_fee": -1, "card_revenue": 1, "customer_payment": 1, "transfer": 0,
    "utilities": -1, "tax_payment": -1, "supplier_payment": -1, "insurance": -1,
}

# (low, high) amount ranges in USD, sampled log-uniformly
AMOUNT = {
    "payroll": (800, 60000), "rent": (1200, 25000), "loan_repayment": (150, 15000),
    "mca_remittance": (40, 2500), "nsf_fee": (25, 39), "card_revenue": (50, 30000),
    "customer_payment": (100, 80000), "transfer": (50, 50000), "utilities": (40, 4000),
    "tax_payment": (200, 40000), "supplier_payment": (100, 45000), "insurance": (80, 6000),
}

# Counterparty / descriptor stems per category (all invented or generic).
STEMS = {
    "payroll": ["GUSTOPAY PAYROLL", "ADPW WAGE PAY", "PAYCHEQ INC PAYROLL", "RIPLING PAY NET",
                "PAYROLL DIRECT DEP", "QBOOKS PAYROLL", "INTUIT PAYRL", "TRINET PAYROLL",
                "JUSTWORKS PAYRL", "PAYLOCITY WAGES", "ONPAY PAYROLL", "SQUARE PAYROLL",
                "EMPLOYEE WAGES BATCH", "PATRIOT SOFTWARE PAY"],
    "rent": ["RENT PAYMENT", "MAPLEWOOD PROPERTY MGMT", "CROSSPOINT PLAZA LLC LEASE",
             "BRIGHTON REALTY RENT", "SUMMIT COMMERCIAL LEASE", "LEASE PMT UNIT", "HARBORVIEW HOLDINGS RENT",
             "OAKRIDGE PROPERTIES", "RIVERSTONE MGMT LEASE", "APPFOLIO RENT PMT", "BUILDIUM RENT",
             "WESTGATE LANDLORD LLC", "TENANT RENT REMIT"],
    "loan_repayment": ["SBA LOAN PMT", "ONDEK CAPITAL LOAN", "BLUEVINE LOAN PAYMT", "FUNDBOX REPAY",
                       "TERM LOAN PAYMENT", "LOAN PYMT ACCT", "EQUIP FINANCE PMT", "KAPITAL LOAN DEBIT",
                       "LENDIO LOAN REPAY", "CREDIBLY LOAN PMT", "BANK LOAN AUTOPAY", "LOC PAYMENT",
                       "FORA FINANCIAL LOAN", "NATIONAL FUNDING PMT"],
    "mca_remittance": ["LIBERTY FUNDING DAILY", "YELLOWSTONE CAP ACH", "PEARL CAP RECV", "RAPID ADVANCE DLY",
                       "MERCHANT ADV REMIT", "CAPYBARA FUNDING", "EVEREST BIZ FUNDING", "KALAMATA CAP",
                       "MCA DAILY DEBIT", "FUNDKITE RECEIVABLES", "VADER MERCHANT SVCS", "SPLIT FUNDING RMT",
                       "IOU ADVANCE DLY", "FUTURE RECEIVABLES PURCH"],
    "nsf_fee": ["NSF FEE", "RETURNED ITEM FEE", "OVERDRAFT FEE", "OD ITEM FEE", "INSUFFICIENT FUNDS FEE",
                "RETURN ITEM CHARGEBACK FEE", "NSF RETURNED CHECK FEE", "OVERDRAFT ITEM PAID FEE",
                "UNCOLLECTED FUNDS FEE", "EXTENDED OD FEE"],
    "card_revenue": ["SQUARE INC DEP", "STRIPE TRANSFER", "TOASTPOS DEP", "CLOVER SETTLEMENT",
                     "SHOPIFY PAYOUT", "PAYPAL TRANSFER", "HEARTLAND PMT SYS DEP", "WORLDPAY DEPOSIT",
                     "BANKCARD MTOT DEP", "MERCH SETTLEMENT", "AMEX SETTLEMENT", "FISERV MERCH DEP",
                     "ELAVON BATCH DEP", "LIGHTSPEED PAYOUT", "DOORDASH PAYOUT"],
    "customer_payment": ["ACH CREDIT INV", "CUSTOMER PMT INV", "WIRE IN", "REMOTE DEPOSIT", "MOBILE DEPOSIT",
                         "BILL.COM RECEIVABLE", "ACH CR CORP PAY", "PAYMENT RECEIVED INV", "LOCKBOX DEPOSIT",
                         "INVOICE PMT ACME", "MELIO RECEIVABLE", "COUNTY OF CONTRACT PAY", "CHECK DEPOSIT"],
    "transfer": ["ONLINE TRANSFER TO CHK", "ONLINE TRANSFER FROM SAV", "ZELLE TO", "ZELLE FROM",
                 "INTERNAL TRANSFER", "XFER TO ACCT", "XFER FROM ACCT", "BOOK TRANSFER", "SWEEP TRANSFER",
                 "VENMO CASHOUT", "MOBILE TRANSFER", "TRANSFER TO OWNER", "FUNDS TRANSFER"],
    "utilities": ["CON ED ELECTRIC", "PSEG UTIL", "NATL GRID GAS", "VERIZON BUSINESS", "COMCAST BUSINESS",
                  "SPECTRUM BIZ", "WATER DEPT AUTOPAY", "WASTE MGMT", "ATT BUSINESS", "DUKE ENERGY",
                  "CITY WATER SEWER", "EVERSOURCE ELEC", "REPUBLIC SVCS TRASH", "OPTIMUM BIZ INTERNET"],
    "tax_payment": ["IRS USATAXPYMT", "EFTPS TAX PMT", "NYS DTF SALES TAX", "STATE TAX PAYMENT", "NJ DIV TAXATION",
                    "FRANCHISE TAX BD", "DEPT OF REVENUE", "IRS EST TAX", "CT DRS TAX PMT",
                    "COMPTROLLER SALES TAX", "PAYROLL TAX EFTPS", "TX COMPTROLLER"],
    "supplier_payment": ["SYSCO FOODS", "US FOODSERV", "RESTAURANT DEPOT", "ULINE SUPPLIES", "GRAINGER",
                         "HOME DEPOT PRO", "BILL.COM PAYABLE", "VENDOR PMT INV", "WEBSTAURANT", "STAPLES BIZ",
                         "AMAZON BUSINESS", "FASTENAL CO", "COSTCO WHOLESALE", "MCMASTER CARR", "MELIO PAYABLE"],
    "insurance": ["STATE FARM INS", "HISCOX INS PREM", "NEXT INSURANCE", "HARTFORD INS PMT", "PROGRESSIVE COMM",
                  "WORKERS COMP PREM", "LIBERTY MUT INS", "GEICO COMMERCIAL", "TRAVELERS PREM", "CHUBB PREMIUM",
                  "INSURANCE PREMIUM", "PIE INSURANCE"],
}

PREFIX = {
    -1: ["ACH DEBIT", "ACH WITHDRAWAL", "WEB PMT", "POS PURCHASE", "DEBIT CARD PURCHASE", "PREAUTHORIZED DEBIT",
         "ELECTRONIC PMT", "CHECKCARD", ""],
    1: ["ACH CREDIT", "DEPOSIT", "ELECTRONIC DEP", "INCOMING", ""],
}

# Low-signal descriptors that can show up for many categories.
GENERIC = ["ONLINE PAYMENT THANK YOU", "ACH DEBIT WEB PMT", "ELECTRONIC PAYMENT", "PREAUTHORIZED ACH",
           "PAYMENT", "ACH TRANSACTION", "RECURRING PAYMENT", "WEB INITIATED", "CCD PAYMENT", "ACH CREDIT"]

STATES = ["NY", "NJ", "CT", "PA", "MA", "TX", "CA", "FL", "IL", "GA"]
DESC_WIDTH = 40  # many core banking systems truncate descriptors


@dataclass
class GenConfig:
    seed: int = 7
    n_train: int = 8000
    n_val: int = 1500
    n_test_seen: int = 1500
    n_test_unseen: int = 1500
    generic_rate: float = 0.07   # share of rows with a low-signal descriptor
    typo_rate: float = 0.15
    unseen_frac: float = 0.25    # share of each category's stems held out


def split_stems(rng: random.Random, unseen_frac: float) -> tuple[dict, dict]:
    seen, unseen = {}, {}
    for cat, stems in STEMS.items():
        s = stems[:]
        rng.shuffle(s)
        k = max(2, round(len(s) * unseen_frac))
        unseen[cat], seen[cat] = s[:k], s[k:]
    return seen, unseen


def _ref(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return "".join(rng.choices(string.digits, k=rng.randint(6, 12)))
    if kind < 0.55:
        return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"
    if kind < 0.70:
        return f"#{rng.randint(100, 9999)} {rng.choice(STATES)}"
    if kind < 0.85:
        return "ID:" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
    return ""


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 6:
        return s
    i = rng.randrange(1, len(s) - 2)
    op = rng.random()
    if op < 0.4:
        return s[:i] + s[i + 1:]                      # drop
    if op < 0.8:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]    # swap
    return s[:i] + s[i] + s[i:]                       # double


def make_row(rng: random.Random, cat: str, stems: list[str], cfg: GenConfig) -> dict:
    d = DIRECTION[cat]
    if d == 0:
        d = rng.choice([-1, 1])
    lo, hi = AMOUNT[cat]
    amount = round(lo * (hi / lo) ** rng.random(), 2)
    if rng.random() < cfg.generic_rate:
        core = rng.choice(GENERIC)
    else:
        core = rng.choice(stems)
        if rng.random() < cfg.typo_rate:
            core = _typo(rng, core)
    parts = [rng.choice(PREFIX[d]), core, _ref(rng)]
    desc = " ".join(p for p in parts if p)
    desc = desc[:DESC_WIDTH]
    r = rng.random()
    if r < 0.15:
        desc = desc.lower()
    elif r < 0.25:
        desc = desc.title()
    return {
        "description": desc,
        "direction": "CREDIT" if d > 0 else "DEBIT",
        "amount": amount,
        "label": cat,
    }


def generate(cfg: GenConfig) -> dict[str, pd.DataFrame]:
    rng = random.Random(cfg.seed)
    seen, unseen = split_stems(rng, cfg.unseen_frac)
    # mildly imbalanced class prior, like a real SMB ledger
    weights = [1.0, 0.6, 0.8, 0.7, 0.4, 1.6, 1.2, 1.3, 0.9, 0.6, 1.4, 0.5]

    def draw(n: int, pool: dict) -> pd.DataFrame:
        cats = rng.choices(CATEGORIES, weights=weights, k=n)
        return pd.DataFrame([make_row(rng, c, pool[c], cfg) for c in cats])

    return {
        "train": draw(cfg.n_train, seen),
        "val": draw(cfg.n_val, seen),
        "test_seen": draw(cfg.n_test_seen, seen),
        "test_unseen": draw(cfg.n_test_unseen, unseen),
    }


def to_model_text(df: pd.DataFrame) -> list[str]:
    """Text fed to the models: direction + rounded amount + raw descriptor."""
    return [f"{d} {a:.0f} | {s}" for d, a, s in zip(df.direction, df.amount, df.description)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = generate(GenConfig(seed=args.seed))
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:12s} {len(df):6d} rows -> {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
