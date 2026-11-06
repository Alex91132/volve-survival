"""Тесты: числа проекта воспроизводятся из данных репозитория.

Проверяем не «код не падает», а совпадение с тем, что написано в README и
напечатано в ноутбуке: когорта, медиана Каплана–Мейера, log-rank, таблица
чувствительности и отсутствие старых ошибочных формулировок.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from survival import (
    COX_BOUND, bootstrap_cox, build_cohort, km_curve, km_median, load_wells,
    logrank, nearest_injector_distance,
)

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily.csv"
LOCATIONS = ROOT / "data" / "locations.csv"


@pytest.fixture(scope="module")
def wells():
    return load_wells(str(DAILY))


@pytest.fixture(scope="module")
def cohort(wells):
    return build_cohort(wells)


def test_wells_are_producers_only(wells):
    assert wells["WELL"].nunique() == 6
    assert len(wells) == 9_143
    known = wells["wc"].dropna()
    assert known.max() <= 1.0, "обводнённость не может превышать 100 %"
    assert (known < 0).sum() <= 10, "отрицательные WC — единичные поправки в данных, не норма"


def test_cohort_shape(cohort):
    surv, dropped = cohort
    assert len(surv) == 6, "в когорте шесть скважин"
    assert int(surv["event"].sum()) == 5, "пять событий, одна цензурирована"
    assert int((1 - surv["event"]).sum()) == 1
    assert dropped.empty, "в текущих данных никто не отброшен молча"


def test_durations_match_notebook(cohort):
    surv, _ = cohort
    expected = [4.9, 5.5, 21.0, 23.2, 25.1, 27.8]
    assert sorted(round(v, 1) for v in surv["dur_months"]) == expected


def test_km_is_a_survival_curve(cohort):
    surv, _ = cohort
    ts, ss = km_curve(surv["dur_months"], surv["event"])
    assert ss[0] == 1.0 and ss[-1] < 1.0
    assert np.all(np.diff(ss) <= 1e-12), "выживаемость не может расти"
    assert np.all(np.diff(ts) >= 0)


def test_km_median_matches_readme(cohort):
    surv, _ = cohort
    ts, ss = km_curve(surv["dur_months"], surv["event"])
    assert km_median(ts, ss) == pytest.approx(23.2, abs=0.05)


def test_km_median_undefined_when_curve_stays_above_half():
    # два события на шесть наблюдений: S падает только до 0.667 — медианы нет
    ts, ss = km_curve([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 0, 0, 0, 0])
    assert ss[-1] > 0.5
    assert np.isnan(km_median(ts, ss))


def test_logrank_matches_readme(cohort):
    surv, _ = cohort
    surv = surv.assign(dist_inj_km=nearest_injector_distance(surv, str(LOCATIONS)))
    median = surv["dist_inj_km"].median()
    near = surv[surv["dist_inj_km"] <= median]
    far = surv[surv["dist_inj_km"] > median]
    chi2, p, observed, expected = logrank(near["dur_months"], near["event"], far["dur_months"], far["event"])
    assert chi2 == pytest.approx(0.83, abs=0.01)
    assert p == pytest.approx(0.364, abs=0.005)
    assert (observed, round(expected, 1)) == (2.0, 1.2)


def test_threshold_sensitivity_matches_readme(wells):
    events, medians = [], []
    for thr in (0.10, 0.30, 0.50, 0.70, 0.90):
        cohort, _ = build_cohort(wells, threshold=thr)
        ts, ss = km_curve(cohort["dur_months"], cohort["event"])
        events.append(int(cohort["event"].sum()))
        medians.append(round(km_median(ts, ss), 1))
    assert events == [5, 5, 5, 4, 2]
    assert medians == pytest.approx([13.0, 18.1, 23.2, 33.5, 61.4], abs=0.05)


def test_cox_estimate_is_capped_or_reported_as_separation(cohort):
    surv, _ = cohort
    surv = surv.assign(dist_inj_km=nearest_injector_distance(surv, str(LOCATIONS)))
    beta, lo, hi, at_bound = bootstrap_cox(
        surv["dur_months"].values, surv["event"].values, -surv["dist_inj_km"].values, n_boot=50
    )
    assert abs(beta) <= COX_BOUND + 1e-9, "коэффициент не выходит за границу поиска"
    assert lo <= beta <= hi or abs(beta) >= COX_BOUND - 1e-9
    assert 0.0 <= at_bound <= 100.0


def test_readme_has_no_stale_numbers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "~38" not in text, "старая ошибка: верхняя граница чувствительности не 38"
    assert "124.54" not in text, "β = 124.54 — артефакт сепарации, его не должно быть в README"
    for number in ("23 мес", "61 мес", "5 из 6"):
        assert number in text, f"в README нет числа {number!r}"
    assert ("0.36" in text) or ("0,36" in text), "в README нет p-value log-rank"
