"""The cost-rate robustness reader, and the limitations band that cites it.

These exist because the reader's output is a sentence printed on a report as a
finding. A reader that quietly returned "holds" on a file it could not understand
would put a robustness claim in front of a human that nothing had established --
the precise failure mode NULL is built to catch, committed by NULL.

So most of what is asserted here is refusal: the reader must raise on a missing
file, an empty one, a missing column, an unreadable number, and -- the one that
matters most -- a file whose own stated verdict disagrees with what its numbers
actually say.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from null.costs.robustness import (
    REQUIRED_COLUMNS,
    RobustnessFormatError,
    load_rate_robustness,
)

# Two grid points. The finding is "the gross pick loses money AND the net pick is
# a different point", so a fixture needs two distinct hashes to express it.
GROSS_PICK = "aaaa000000000000"
NET_PICK = "bbbb111111111111"

FIELDS = [
    "component",
    "factor",
    "best_gross_net_sharpe",
    "best_gross_hash",
    "best_net_hash",
    "inversion_holds",
]


def _row(
    component: str,
    factor: float,
    naive_net: float,
    *,
    net_hash: str = NET_PICK,
    stated: bool | None = None,
) -> dict[str, object]:
    holds = naive_net < 0.0 and net_hash != GROSS_PICK
    return {
        "component": component,
        "factor": f"{factor:.6f}",
        "best_gross_net_sharpe": f"{naive_net:.6f}",
        "best_gross_hash": GROSS_PICK,
        "best_net_hash": net_hash,
        "inversion_holds": str(holds if stated is None else stated),
    }


def _write(path: Path, rows: list[dict[str, object]], fields: list[str] = FIELDS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    return path


def test_a_sweep_where_the_finding_survives_reports_that_it_survived(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "sweep.csv",
        [
            _row("baseline", 1.00, -0.027),
            _row("dp_charge_per_scrip_per_sell", 0.75, -0.011),
            _row("dp_charge_per_scrip_per_sell", 1.25, -0.044),
        ],
    )
    result = load_rate_robustness(path)

    assert result.inversion_holds_everywhere is True
    assert result.broken_cases == ()
    assert result.n_cases == 3
    # Nearest to zero is the LEAST negative case -- the one nearest to breaking
    # the claim, not the one furthest from it.
    assert result.naive_pick_net_sharpe_nearest_zero == pytest.approx(-0.011)
    assert result.naive_pick_net_sharpe_most_negative == pytest.approx(-0.044)
    assert "held in all 3" in result.sentence
    assert "0.75x to 1.25x" in result.sentence


def test_a_sweep_where_the_finding_breaks_names_the_cases_that_broke_it(
    tmp_path: Path,
) -> None:
    """The whole point. A sweep that fails must produce a sentence that says so."""
    path = _write(
        tmp_path / "sweep.csv",
        [
            _row("baseline", 1.00, -0.027),
            _row("stt_buy_pct", 0.75, +0.055),
            _row("stt_buy_pct", 1.25, -0.110),
        ],
    )
    result = load_rate_robustness(path)

    assert result.inversion_holds_everywhere is False
    assert result.broken_cases == ("stt_buy_pct@0.75",)
    assert "does NOT survive" in result.sentence
    assert "stt_buy_pct@0.75" in result.sentence
    assert "1 of 3" in result.sentence


def test_the_gross_pick_being_the_net_pick_breaks_the_finding_too(
    tmp_path: Path,
) -> None:
    """Both halves are load-bearing.

    A case where the naive pick loses money but IS the best net variant is not the
    inversion -- there is nothing inverted about it. Only checking the sign would
    call this a pass.
    """
    path = _write(
        tmp_path / "sweep.csv",
        [_row("half_spread_bps", 1.25, -0.400, net_hash=GROSS_PICK)],
    )
    result = load_rate_robustness(path)

    assert result.inversion_holds_everywhere is False
    assert result.broken_cases == ("half_spread_bps@1.25",)


def test_a_writer_that_mislabels_its_own_verdict_is_caught(tmp_path: Path) -> None:
    """Two independent derivations of the same claim must agree.

    The sweep script computes ``inversion_holds`` itself; this reader recomputes it
    from the raw columns. If they ever disagree, one of them is broken and picking
    a winner silently would launder whichever is wrong onto a report.
    """
    path = _write(
        tmp_path / "sweep.csv",
        [_row("stt_buy_pct", 0.75, +0.055, stated=True)],  # numbers say False
    )
    with pytest.raises(RobustnessFormatError, match="disagree about what the sweep found"):
        load_rate_robustness(path)


def test_an_unreadable_stated_verdict_raises_rather_than_being_guessed(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "sweep.csv",
        [{**_row("stt_buy_pct", 0.75, -0.055), "inversion_holds": "probably"}],
    )
    with pytest.raises(RobustnessFormatError, match="not a boolean"):
        load_rate_robustness(path)


@pytest.mark.parametrize("dropped", [c for c in REQUIRED_COLUMNS])
def test_every_required_column_is_actually_required(
    tmp_path: Path, dropped: str
) -> None:
    """Parameterised over the declared list, so adding a column to REQUIRED_COLUMNS
    without enforcing it fails here rather than passing quietly."""
    fields = [f for f in FIELDS if f != dropped]
    path = _write(tmp_path / "sweep.csv", [_row("baseline", 1.0, -0.027)], fields=fields)
    with pytest.raises(RobustnessFormatError, match=dropped):
        load_rate_robustness(path)


def test_a_missing_sweep_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RobustnessFormatError, match="may only cite a sweep that exists"):
        load_rate_robustness(tmp_path / "nope.csv")


def test_a_header_with_no_cases_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "sweep.csv", [])
    with pytest.raises(RobustnessFormatError, match="no cases"):
        load_rate_robustness(path)


def test_an_unparseable_number_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "sweep.csv",
        [{**_row("baseline", 1.0, -0.027), "best_gross_net_sharpe": "n/a"}],
    )
    with pytest.raises(RobustnessFormatError, match="not a number"):
        load_rate_robustness(path)


# ---------------------------------------------------------------------------
# The limitations band.
# ---------------------------------------------------------------------------


def _cost_limitation(context: dict[str, object]):
    from null.verdict.limitations import _unverified_cost_rates

    return _unverified_cost_rates(None, context)  # type: ignore[arg-type]


def test_the_band_says_sensitivity_was_not_measured_when_no_sweep_is_supplied() -> None:
    """The default must stay conservative. An absent sweep is not a passing one."""
    limitation = _cost_limitation({"rates_are_verified": False})
    assert limitation is not None
    assert "NOT measured for this run" in limitation.text
    assert "unestablished rather than merely imprecise" in limitation.text
    assert limitation.severity == "blocking"


def test_the_band_states_the_measured_result_when_a_sweep_is_supplied() -> None:
    limitation = _cost_limitation(
        {
            "rates_are_verified": False,
            "cost_rate_robustness": "Rate sensitivity WAS measured and it broke.",
        }
    )
    assert limitation is not None
    assert "Rate sensitivity WAS measured and it broke." in limitation.text
    assert "NOT measured for this run" not in limitation.text


def test_the_band_separates_cost_levels_from_cost_rankings() -> None:
    """The change this rewrite exists for.

    The old text said every cost figure was 'indicative only', which warns the
    reader without telling them what it voids. Levels and rankings are affected
    differently and the band has to say which is which.
    """
    limitation = _cost_limitation({"rates_are_verified": False})
    assert limitation is not None
    assert "LEVELS" in limitation.text
    assert "RANKING" in limitation.text
    assert "indicative only" not in limitation.text


def test_verified_rates_still_produce_no_limitation_at_all() -> None:
    assert _cost_limitation({"rates_are_verified": True}) is None


# ---------------------------------------------------------------------------
# The rank measure. The sign test ("does the gross pick go net-negative") turned
# out to be fragile on the real M7 grid -- it flips when stt_buy_pct moves 25%,
# because the baseline number sits 0.027 from zero. The rank of the gross pick on
# the net-ranked grid measures the same selection error without balancing on that
# boundary, so these assert it is read, summarised, and never silently half-read.
# ---------------------------------------------------------------------------

RANK_FIELDS = FIELDS + [
    "gross_pick_rank_by_net",
    "net_sharpe_given_up_by_selecting_on_gross",
]


def _rank_row(
    component: str, factor: float, naive_net: float, rank: int, given_up: float
) -> dict[str, object]:
    return {
        **_row(component, factor, naive_net),
        "gross_pick_rank_by_net": str(rank),
        "net_sharpe_given_up_by_selecting_on_gross": f"{given_up:.6f}",
    }


def test_rank_columns_when_present_are_summarised_into_the_sentence(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "sweep.csv",
        [
            _rank_row("baseline", 1.00, -0.027, 63, 0.449),
            _rank_row("stt_buy_pct", 1.25, -0.110, 58, 0.512),
        ],
        fields=RANK_FIELDS,
    )
    result = load_rate_robustness(path, n_variants=108)

    assert result.gross_pick_rank_range == (58, 63)
    assert result.net_sharpe_given_up_range == pytest.approx((0.449, 0.512))
    assert "58-63 of 108" in result.sentence
    assert "0.449 to 0.512" in result.sentence


def test_a_broken_sign_test_still_reports_the_rank_that_survived(
    tmp_path: Path,
) -> None:
    """The case that actually happened on the M7 grid.

    A sweep may break the headline claim and still establish a weaker one. The
    sentence has to do both: say plainly that the sign test failed, and report the
    measure that did not -- without letting the second quietly stand in for the
    first.
    """
    path = _write(
        tmp_path / "sweep.csv",
        [
            _rank_row("baseline", 1.00, -0.027, 63, 0.449),
            _rank_row("stt_buy_pct", 0.75, +0.055, 61, 0.410),
        ],
        fields=RANK_FIELDS,
    )
    result = load_rate_robustness(path, n_variants=108)

    assert result.inversion_holds_everywhere is False
    assert "does NOT survive it" in result.sentence
    assert "stt_buy_pct@0.75" in result.sentence
    assert "What did survive every case" in result.sentence
    assert "61-63 of 108" in result.sentence


def test_a_partially_reported_rank_is_treated_as_no_rank_at_all(
    tmp_path: Path,
) -> None:
    """All-or-nothing.

    A range built from only the rows that happened to carry a rank is a range over
    an unknown subset, quoted on a report as though it covered the sweep.
    """
    rows = [
        _rank_row("baseline", 1.00, -0.027, 63, 0.449),
        {**_rank_row("stt_buy_pct", 1.25, -0.110, 58, 0.512),
         "gross_pick_rank_by_net": "",
         "net_sharpe_given_up_by_selecting_on_gross": ""},
    ]
    path = _write(tmp_path / "sweep.csv", rows, fields=RANK_FIELDS)
    result = load_rate_robustness(path, n_variants=108)

    assert result.gross_pick_rank_range is None
    assert result.net_sharpe_given_up_range is None
    assert "size of the selection error is unmeasured" in result.sentence


def test_the_rank_denominator_is_omitted_rather_than_invented(tmp_path: Path) -> None:
    """No n_variants supplied means no ``of 108`` -- not a guess at the grid size."""
    path = _write(
        tmp_path / "sweep.csv",
        [_rank_row("baseline", 1.00, -0.027, 63, 0.449)],
        fields=RANK_FIELDS,
    )
    result = load_rate_robustness(path)
    assert "63-63" in result.sentence
    assert " of " not in result.sentence.split("ranked ")[1].split(" on the")[0]
