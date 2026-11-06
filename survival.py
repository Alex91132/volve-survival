"""Ядро анализа выживаемости для проекта Volve Survival.

Здесь лежат все расчёты — когорта, Каплан–Мейер, log-rank, частичное
правдоподобие Кокса. Ноутбук и тесты вызывают эти функции, поэтому числа
в README проверяются той же реализацией, что строит графики.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

WC_THRESHOLD = 0.50          # порог обводнения по 30-дневному среднему WC
MIN_HISTORY_DAYS = 35        # минимальная длина истории скважины
COX_BOUND = 5.0              # за этой границей оценка Кокса не идентифицируется


def load_wells(path: str = "data/daily.csv") -> pd.DataFrame:
    """Суточная добыча только по добывающим скважинам + обводнённость."""
    df = pd.read_csv(path, parse_dates=["DATEPRD"])
    df["WELL"] = df["WELL_BORE_CODE"].str.strip().str.replace("NO 15/9-", "", regex=False)
    df = df[df["WELL_TYPE"] == "OP"].copy().sort_values(["WELL", "DATEPRD"])
    df["qliq"] = df["BORE_OIL_VOL"].fillna(0) + df["BORE_WAT_VOL"].fillna(0)
    df["wc"] = df["BORE_WAT_VOL"].fillna(0) / df["qliq"].replace(0, np.nan)
    return df


def build_cohort(df: pd.DataFrame, threshold: float = WC_THRESHOLD):
    """Когорта скважин с «сухим стартом» и их время до обводнения.

    Событие — первый день, когда 30-дневное скользящее среднее WC превысило
    порог. Скважина, стартовавшая сразу обводнённой, в анализ не входит:
    у неё нет перехода «сухо → мокро», то есть событие не определено.

    Возвращает (когорта, отброшенные) — во втором DataFrame видно, кого и
    почему исключили, чтобы отбор не был молчаливым.
    """
    rows, dropped = [], []
    for well, g in df.groupby("WELL"):
        g = g.set_index("DATEPRD")
        wc30 = g["wc"].rolling("30D", min_periods=15).mean()
        first_oil = g["BORE_OIL_VOL"][g["BORE_OIL_VOL"] > 0].index.min()
        if pd.isna(first_oil):
            dropped.append({"WELL": well, "причина": "нет добычи нефти"})
            continue
        if not (first_oil + pd.Timedelta(days=MIN_HISTORY_DAYS) <= g.index.max()):
            dropped.append({"WELL": well, "причина": f"история короче {MIN_HISTORY_DAYS} дней"})
            continue

        start_wc = wc30.loc[first_oil:first_oil + pd.Timedelta(days=MIN_HISTORY_DAYS)].max()
        if pd.notna(start_wc) and start_wc > threshold:
            dropped.append({"WELL": well, "причина": "стартовала обводнённой (WC выше порога в 1-й месяц)"})
            continue

        above = wc30[wc30 > threshold]
        event_day = above.index.min() if len(above) else pd.NaT
        end = event_day if pd.notna(event_day) else g.index.max()
        rows.append({
            "WELL": well,
            "first_oil": first_oil,
            "last_day": g.index.max(),
            "event_day": event_day,
            "event": int(pd.notna(event_day)),
            "dur_months": (end - first_oil).days / 30.4,
            "cum_oil_mln": g["BORE_OIL_VOL"].sum() / 1e6,
            "cum_wat_mln": g["BORE_WAT_VOL"].sum() / 1e6,
        })

    cohort = pd.DataFrame(rows).sort_values("first_oil").reset_index(drop=True)
    return cohort, pd.DataFrame(dropped, columns=["WELL", "причина"])


def nearest_injector_distance(cohort: pd.DataFrame, path: str = "data/locations.csv") -> pd.Series:
    """Расстояние от каждой скважины когорты до ближайшей нагнетательной, км."""
    loc = pd.read_csv(path)
    loc["WELL"] = loc["WELL_BORE_CODE"].str.strip().str.replace("NO 15/9-", "", regex=False)
    injected = loc[loc["WELL"].isin(["F-4 AH", "F-5 AH"])]
    distances = {}
    for _, well in loc[loc["WELL"].isin(cohort["WELL"])].iterrows():
        d = np.sqrt(((injected["Lat"] - well["Lat"]) * 111) ** 2
                    + ((injected["Long"] - well["Long"]) * 111 * np.cos(np.radians(well["Lat"]))) ** 2)
        distances[well["WELL"]] = d.min()
    return cohort["WELL"].map(distances)


def km_curve(durations, events):
    """Кривая Каплана–Мейера: S(t) = Π (1 − dᵢ/nᵢ) по моментам событий.

    Цензурированные наблюдения остаются в группе риска до своего момента и
    выбывают после него — поэтому доля событий считается до вычитания группы.
    """
    d = pd.DataFrame({"t": list(durations), "e": list(events)}).sort_values("t")
    at_risk = len(d)
    s = 1.0
    ts, ss = [0.0], [1.0]
    for t, grp in d.groupby("t"):
        deaths = grp["e"].sum()
        if deaths > 0:
            s *= 1 - deaths / at_risk
        at_risk -= len(grp)
        ts.append(t)
        ss.append(s)
    return np.array(ts), np.array(ss)


def km_median(ts, ss) -> float:
    """Медиана выживания: первый момент, где S(t) ≤ 0.5 (иначе NaN)."""
    if not (ss <= 0.5).any():
        return float("nan")
    return float(ts[np.argmax(ss <= 0.5)])


def logrank(d1, e1, d2, e2):
    """Log-rank для двух групп: (chi2, p, наблюдаемые, ожидаемые в первой группе)."""
    a = pd.DataFrame({"t": list(d1), "e": list(e1), "g": 1})
    b = pd.DataFrame({"t": list(d2), "e": list(e2), "g": 2})
    d = pd.concat([a, b])
    observed = expected = variance = 0.0
    for t, grp in d.groupby("t"):
        n1 = int((a["t"] >= t).sum())
        n2 = int((b["t"] >= t).sum())
        deaths = grp["e"].sum()
        if deaths == 0 or n1 + n2 <= 1:
            continue
        expected += deaths * n1 / (n1 + n2)
        observed += grp[grp["g"] == 1]["e"].sum()
        variance += deaths * n1 * n2 * (n1 + n2 - deaths) / (n1 + n2) ** 2 / (n1 + n2 - 1)
    if variance <= 0:
        return float("nan"), float("nan"), observed, expected
    chi2 = (observed - expected) ** 2 / variance
    p = math.erfc(math.sqrt(chi2 / 2))          # хвост хи-квадрата с 1 степенью свободы
    return chi2, p, observed, expected


def cox_slope(durations, events, covariate, bound: float = COX_BOUND) -> float:
    """Коэффициент Кокса с одной ковариатой: частичное правдоподобие (Breslow).

    Оценка ограничена [-bound; +bound]: на малых n максимум правдоподобия
    уходит в бесконечность (сепарация), и без ограничения оптимизатор вернёт
    произвольное большое число, которое нельзя выдавать за коэффициент.
    """
    from scipy.optimize import minimize

    durs = np.asarray(durations, dtype=float)
    evs = np.asarray(events, dtype=int)
    x = np.asarray(covariate, dtype=float)

    def neg_partial_ll(b):
        ll = 0.0
        for i in np.where(evs == 1)[0]:
            risk = durs >= durs[i]
            ll += b * x[i] - np.log(np.exp(b * x[risk]).sum())
        return -ll

    res = minimize(neg_partial_ll, 0.0, method="L-BFGS-B", bounds=[(-bound, bound)])
    return float(res.x[0])


def bootstrap_cox(durations, events, covariate, n_boot: int = 500, seed: int = 42,
                  bound: float = COX_BOUND):
    """Бутстреп коэффициента Кокса: (оценка, lo, hi, доля упоров в границу, %)."""
    durs = np.asarray(durations, dtype=float)
    evs = np.asarray(events, dtype=int)
    x = np.asarray(covariate, dtype=float)
    beta = cox_slope(durs, evs, x, bound=bound)
    rng = np.random.default_rng(seed)
    boot = np.array([
        cox_slope(durs[idx], evs[idx], x[idx], bound=bound)
        for idx in (rng.integers(0, len(durs), len(durs)) for _ in range(n_boot))
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    at_bound = float(np.mean(np.abs(boot) >= bound - 1e-9) * 100)
    return beta, float(lo), float(hi), at_bound
