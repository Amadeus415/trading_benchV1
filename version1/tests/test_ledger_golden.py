"""Golden ledger test: hand-computed 3-trade / 1-split / 1-dividend scenario.

Asserted to the cent. Foundation of every downstream claim.
"""

from __future__ import annotations

from datetime import date

from tradingbench.sim.ledger import Ledger


def test_golden_three_trades_split_dividend():
    """
    Scenario (hand-computed):
      t0: cash = 1000.00
      buy  10 sh @ 100.00, fee 0.50  →  cash=1000-1000-0.50= - wait, notional 1000
      Use smaller notionals:

      cash = 1000.00
      1. BUY  5.0 sh XYZ @ 100.00, fee = 5*100 * 5bps = 0.25
         cash = 1000 - 500 - 0.25 = 499.75
         pos: qty=5, avg=100
      2. BUY  2.0 sh XYZ @ 110.00, fee = 220 * 5bps = 0.11
         cash = 499.75 - 220 - 0.11 = 279.64
         avg_cost = (5*100 + 2*110)/7 = 720/7 = 102.857142...
         qty=7
      3. SPLIT 2:1 → qty=14, avg_cost=51.428571...
      4. DIVIDEND $0.50/sh → cash += 14*0.50 = 7.00 → cash=286.64
      5. SELL 4.0 sh @ 60.00, fee = 240 * 5bps = 0.12
         cash = 286.64 + 240 - 0.12 = 526.52
         realized = 4*(60 - 51.428571...) - 0.12 = 4*8.571428... - 0.12 = 34.285714... - 0.12 = 34.165714...
         qty=10, avg unchanged
      Mark @ 60 → MV = 600, NAV = 526.52 + 600 = 1126.52
    """
    ledger = Ledger(cash=1000.0, as_of=date(2025, 1, 2))

    # 1. buy 5 @ 100, fee 0.25
    ledger.apply_buy("XYZ", qty=5.0, price=100.0, fee=0.25)
    assert abs(ledger.cash - 499.75) < 1e-9
    assert abs(ledger.positions["XYZ"].qty - 5.0) < 1e-9
    assert abs(ledger.positions["XYZ"].avg_cost - 100.0) < 1e-9
    assert abs(ledger.fees_paid - 0.25) < 1e-9

    # 2. buy 2 @ 110, fee 0.11
    ledger.apply_buy("XYZ", qty=2.0, price=110.0, fee=0.11)
    assert abs(ledger.cash - 279.64) < 1e-9
    pos = ledger.positions["XYZ"]
    assert abs(pos.qty - 7.0) < 1e-9
    assert abs(pos.avg_cost - (720.0 / 7.0)) < 1e-9
    assert abs(ledger.fees_paid - 0.36) < 1e-9

    # 3. split 2:1
    ledger.apply_split("XYZ", ratio=2.0)
    pos = ledger.positions["XYZ"]
    assert abs(pos.qty - 14.0) < 1e-9
    assert abs(pos.avg_cost - (720.0 / 7.0 / 2.0)) < 1e-9

    # 4. dividend 0.50
    ledger.apply_dividend("XYZ", amount_per_share=0.50)
    assert abs(ledger.cash - 286.64) < 1e-9

    # 5. sell 4 @ 60, fee 0.12
    ledger.apply_sell("XYZ", qty=4.0, price=60.0, fee=0.12)
    assert abs(ledger.cash - 526.52) < 1e-9
    pos = ledger.positions["XYZ"]
    assert abs(pos.qty - 10.0) < 1e-9
    expected_realized = 4.0 * (60.0 - (720.0 / 7.0 / 2.0)) - 0.12
    assert abs(ledger.realized_pnl - expected_realized) < 1e-6

    st = ledger.state({"XYZ": 60.0}, date(2025, 1, 10))
    assert abs(st.nav - (526.52 + 600.0)) < 1e-6
    # to the cent
    assert round(st.nav, 2) == 1126.52
    assert round(st.cash, 2) == 526.52
