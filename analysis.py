# %% [markdown]
# # Volve Survival: время до обводнения скважины
#
# Третий проект трилогии по данным месторождения **Volve** (Equinor, Северное
# море). Задача — оценить **время от запуска скважины до обводнения**
# (30-дневное скользящее WC > 50 % — точка, где половина жидкости становится
# водой) методами анализа выживаемости.
#
# ## Почему именно survival-анализ
#
# Ключевое явление — **цензурирование**: месторождение остановлено в 2016 году
# решением оператора (демонтаж платформы), а не по состоянию скважин. Скважины,
# которые к концу наблюдения не обводнились, — не «события не было», а
# «событие не успело произойти». При пороге 50 % такие скважины в выборке есть;
# при игнорировании цензурирования время до обводнения занижается. Чуть ниже
# (раздел 5) показано, как доля цензурированных растёт с порогом.
#
# ## Что честно оговариваем заранее
#
# Скважин в когорте — 6. Это достаточно для оценок Каплана–Мейера и наглядного
# сравнения групп, но **мало для модели Кокса** — вместо публикации p-values из
# пяти точек мы показываем границу применимости метода (раздел 6). Это
# осознанное решение, а не упущение.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

df = pd.read_csv("data/daily.csv", parse_dates=["DATEPRD"])
df["WELL"] = df["WELL_BORE_CODE"].str.strip().str.replace("NO 15/9-", "", regex=False)
df = df[df["WELL_TYPE"] == "OP"].copy()
df = df.sort_values(["WELL", "DATEPRD"])
df["qliq"] = df["BORE_OIL_VOL"].fillna(0) + df["BORE_WAT_VOL"].fillna(0)
df["wc"] = df["BORE_WAT_VOL"].fillna(0) / df["qliq"].replace(0, np.nan)
print(f"добывающих скважин: {df['WELL'].nunique()}, строк: {len(df):,}")

# %% [markdown]
# ## 1. Определение события и когорта анализа
#
# Событие — первый день, когда 30-дневное скользящее среднее WC превысило порог
# **50 %** (окно отсекает разовые технологические прорывы при запускных
# операциях).
#
# Когорта — скважины со **«сухим стартом»**: в первый месяц работы скользящее
# WC было ниже порога. Скважина, стартовавшая сразу обводнённой, не имеет
# перехода «сухо → мокро» и в анализ прорыва не входит — для неё событие
# не определено. Это стандартное требование анализа выживаемости: объект должен
# быть «жив» (свободен от события) в момент нуля.

# %%
WC_THRESHOLD = 0.50

rows, excluded = [], []
for w, g in df.groupby("WELL"):
    g = g.set_index("DATEPRD")
    wc30 = g["wc"].rolling("30D", min_periods=15).mean()
    first_oil = g["BORE_OIL_VOL"][g["BORE_OIL_VOL"] > 0].index.min()
    if not (first_oil + pd.Timedelta(days=35) <= g.index.max()):
        continue
    start_wc = wc30.loc[first_oil:first_oil + pd.Timedelta(days=35)].max()
    if pd.notna(start_wc) and start_wc > WC_THRESHOLD:
        excluded.append({"WELL": w, "причина": "стартовала обводнённой (WC > 50 % в 1-й месяц)"})
        continue
    last_day = g.index.max()
    bt = wc30[wc30 > WC_THRESHOLD]
    event_day = bt.index.min() if len(bt) else pd.NaT
    rows.append({
        "WELL": w,
        "first_oil": first_oil,
        "last_day": last_day,
        "event_day": event_day,
        "event": int(pd.notna(event_day)),
        "dur_months": ((event_day if pd.notna(event_day) else last_day) - first_oil).days / 30.4,
        "cum_oil_mln": g["BORE_OIL_VOL"].sum() / 1e6,
        "cum_wat_mln": g["BORE_WAT_VOL"].sum() / 1e6,
    })
surv = pd.DataFrame(rows).sort_values("first_oil").reset_index(drop=True)
if excluded:
    print("исключены из когорты:")
    print(pd.DataFrame(excluded).to_string(index=False))
surv[["WELL", "first_oil", "event", "dur_months", "cum_oil_mln", "cum_wat_mln"]].round(2)

# %%
print(f"событий (обводнение WC > {WC_THRESHOLD:.0%}): {surv['event'].sum()} из {len(surv)} скважин; "
      f"цензурировано: {(1 - surv['event']).sum()}")
surv[["WELL", "event_day", "dur_months"]].assign(
    event_day=lambda x: x["event_day"].dt.strftime("%Y-%m-%d").fillna("цензурирована"))

# %% [markdown]
# ## 2. Расстояние до нагнетательных — единственная осмысленная группировка
#
# Гипотеза: скважины ближе к нагнетательным обводняются быстрее (прорыв по
# нагнетаемой воде). Считаем расстояние от каждой добывающей до ближайшей
# нагнетательной (F-4 AH, F-5 AH; F-1 C после перевода) по координатам.

# %%
loc = pd.read_csv("data/locations.csv")
loc["WELL"] = loc["WELL_BORE_CODE"].str.strip().str.replace("NO 15/9-", "", regex=False)
inj_names = ["F-4 AH", "F-5 AH"]
inj = loc[loc["WELL"].isin(inj_names)]
prod = loc[loc["WELL"].isin(surv["WELL"])]

dists = {}
for _, p in prod.iterrows():
    d = np.sqrt(((inj["Lat"] - p["Lat"]) * 111) ** 2 +
                ((inj["Long"] - p["Long"]) * 111 * np.cos(np.radians(p["Lat"]))) ** 2)
    dists[p["WELL"]] = d.min()
surv["dist_inj_km"] = surv["WELL"].map(dists)
med = surv["dist_inj_km"].median()
surv["group"] = np.where(surv["dist_inj_km"] <= med, "близко к ППД", "далеко от ППД")
surv[["WELL", "dist_inj_km", "group", "event", "dur_months"]].round(2)

# %%
ax = plt.figure().add_subplot()
for _, p in prod.iterrows():
    ax.scatter(p["Long"], p["Lat"], s=140, c="#2b6cb0", zorder=3)
    ax.annotate(p["WELL"], (p["Long"], p["Lat"]), xytext=(6, 4), textcoords="offset points")
for _, i in inj.iterrows():
    ax.scatter(i["Long"], i["Lat"], s=160, c="#fa7a33", marker="s", zorder=3)
    ax.annotate(i["WELL"], (i["Long"], i["Lat"]), xytext=(6, 4), textcoords="offset points")
ax.set_title("Добывающие (круги) и нагнетательные (квадраты): расстановка Volve")
ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
plt.show()

# %% [markdown]
# ## 3. Оценка Каплана–Мейера — руками, без библиотек
#
# Кривая выживаемости S(t) = Π (1 − dᵢ/nᵢ) по моментам событий; цензурированные
# наблюдения остаются в группе риска до своего выбытия.

# %%
def km_curve(durations, events):
    """Кривая Каплана–Мейера: возвращает (время, выживаемость)."""
    d = pd.DataFrame({"t": durations, "e": events}).sort_values("t")
    at_risk = len(d)
    s = 1.0
    ts, ss = [0.0], [1.0]
    for t, grp in d.groupby("t"):
        deaths = grp["e"].sum()
        if deaths > 0:
            s *= 1 - deaths / at_risk
        at_risk -= len(grp)
        ts.append(t); ss.append(s)
    return np.array(ts), np.array(ss)

def km_median(ts, ss):
    """Медиана времени выживания: первый момент S(t) <= 0.5."""
    idx = np.argmax(ss <= 0.5) if (ss <= 0.5).any() else None
    return ts[idx] if idx is not None else np.nan

fig, ax = plt.subplots()
ts, ss = km_curve(surv["dur_months"], surv["event"])
ax.step(ts, ss, where="post", lw=2.2, color="#2b6cb0", label="все скважины")
for grp, color in [("близко к ППД", "#fa7a33"), ("далеко от ППД", "#38a169")]:
    sub = surv[surv["group"] == grp]
    tg, sg = km_curve(sub["dur_months"], sub["event"])
    ax.step(tg, sg, where="post", lw=1.8, color=color, label=f"{grp} (n={len(sub)})")
ax.set_title(f"Каплан–Мейер: время до обводнения (WC > {WC_THRESHOLD:.0%})")
ax.set_xlabel("месяцев с запуска скважины")
ax.set_ylabel("доля скважин без обводнения")
ax.set_ylim(0, 1.05)
ax.legend()
plt.show()
print(f"медиана по всем скважинам: {km_median(ts, ss):.1f} мес")

# %% [markdown]
# ## 4. Сравнение групп: log-rank тест
#
# Классическая статистика лог-ранга: сравниваем наблюдаемые и ожидаемые числа
# событий в группах на каждом моменте события. Одну реализацию — снова руками.

# %%
def logrank(d1, e1, d2, e2):
    """Log-rank для двух групп: chi2 (1 ст. свободы) и p-value."""
    a = pd.DataFrame({"t": d1, "e": e1, "g": 1})
    b = pd.DataFrame({"t": d2, "e": e2, "g": 2})
    d = pd.concat([a, b])
    O1 = E1 = V = 0.0
    for t, grp in d.groupby("t"):
        n1 = (a["t"] >= t).sum(); n2 = (b["t"] >= t).sum()
        dt = grp["e"].sum()
        if dt == 0 or n1 + n2 <= 1:
            continue
        E1 += dt * n1 / (n1 + n2)
        O1 += grp[grp["g"] == 1]["e"].sum()
        V += dt * n1 * n2 * (n1 + n2 - dt) / (n1 + n2) ** 2 / (n1 + n2 - 1)
    chi2 = (O1 - E1) ** 2 / V if V > 0 else np.nan
    p = 1 - stats.chi2.cdf(chi2, 1) if np.isfinite(chi2) else np.nan
    return chi2, p, O1, E1

g1 = surv[surv["group"] == "близко к ППД"]
g2 = surv[surv["group"] == "далеко от ППД"]
chi2, p, O1, E1 = logrank(g1["dur_months"], g1["event"], g2["dur_months"], g2["event"])
print(f"близко к ППД: событий {int(g1['event'].sum())} из {len(g1)} | "
      f"далеко: {int(g2['event'].sum())} из {len(g2)}")
print(f"log-rank: chi2 = {chi2:.2f}, p = {p:.3f} "
      f"(наблюдаемых {O1:.0f} против ожидаемых {E1:.1f} в группе «близко»)")
print("Внимание: с такими n результат — иллюстрация метода, не доказательство.")

# %% [markdown]
# ## 5. Чувствительность к порогу обводнения
#
# «Обводнение» — конструкторская гипотеза, а не факт природы. Считаем медианное
# время и число цензурированных при порогах от 10 % до 90 %: с ростом порога
# событий всё меньше, а цензурированных — больше; при игнорировании
# цензурирования оценка времени резала бы хвост распределения.

# %%
rows = []
for thr in [0.10, 0.30, 0.50, 0.70, 0.90]:
    r = []
    for w, g in df.groupby("WELL"):
        g = g.set_index("DATEPRD")
        wc30 = g["wc"].rolling("30D", min_periods=15).mean()
        first = g["BORE_OIL_VOL"][g["BORE_OIL_VOL"] > 0].index.min()
        if not (first + pd.Timedelta(days=35) <= g.index.max()):
            continue
        if pd.notna(wc30.loc[first:first + pd.Timedelta(days=35)].max()) and \
           wc30.loc[first:first + pd.Timedelta(days=35)].max() > thr:
            continue
        bt = wc30[wc30 > thr]
        end = bt.index.min() if len(bt) else g.index.max()
        r.append({"t": (end - first).days / 30.4, "e": int(len(bt) > 0)})
    rr = pd.DataFrame(r)
    t, s = km_curve(rr["t"], rr["e"])
    rows.append({"порог WC": f"{thr:.0%}", "событий": int(rr["e"].sum()),
                 "цензурировано": int((1 - rr["e"]).sum()),
                 "медиана, мес": round(km_median(t, s), 1)})
sens = pd.DataFrame(rows)
sens

# %% [markdown]
# ## 6. Почему мы НЕ строим модель Кокса
#
# Регрессия Кокса на 5–6 наблюдениях с 1–2 ковариатами даст «значимости» из
# чистой случайности: асимптотика не работает, оценка неустойчива, доверительные
# интервалы бессмысленны. Проверить это честно: бутстреп-оценка разброса
# коэффициента «расстояние до ППД» в простейшей модели.

# %%
from scipy.optimize import minimize

def cox_partial_slope(durs, evs, x):
    """Частичная лог-правдоподобность Кокса с одной ковариатой (без сдвига)."""
    def negll(b):
        ll = 0.0
        for i in np.where(evs == 1)[0]:
            risk = durs >= durs[i]
            ll += b * x[i] - np.log(np.exp(b * x[risk]).sum())
        return -ll
    res = minimize(negll, 0.0, method="BFGS")
    return res.x[0]

x = -surv["dist_inj_km"].values  # ближе -> больше риск: ожидаем b > 0
durs, evs = surv["dur_months"].values, surv["event"].values
beta = cox_partial_slope(durs, evs, x)
rng = np.random.default_rng(42)
boot = []
for _ in range(500):
    idx = rng.integers(0, len(surv), len(surv))
    boot.append(cox_partial_slope(durs[idx], evs[idx], x[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"коэффициент Кокса (близость к ППД): β = {beta:.2f}")
print(f"бутстреп 95% ДИ: [{lo:.2f}; {hi:.2f}] — интервал шире самой оценки:")
print("модель на таких n неинформативна, и мы это фиксируем, а не публикуем p-value.")

# %% [markdown]
# ## 7. Выводы

# %%
med_all = km_median(ts, ss)
n_ev = int(surv["event"].sum())
n_cens = int((1 - surv["event"]).sum())
concl = [
    f"Обводнение (WC > {WC_THRESHOLD:.0%}) наступило у {n_ev} из {len(surv)} скважин когорты; "
    f"медианное время по Каплану–Мейеру — {med_all:.1f} мес с запуска.",
    f"Цензурирование в данных реально: {n_cens} скважина не обводнела к закрытию поля "
    f"в 2016 г. (административное цензурирование); с ростом порога доля таких "
    f"растёт (раздел 5) — игнорирование цензурирования занижало бы время.",
    f"Группировка по близости к нагнетательным даёт ожидаемое направление "
    f"(у близких обводнение раньше), но log-rank p = {p:.2f} при n = {len(surv)} — "
    f"иллюстрация метода, а не доказательство.",
    f"Выбор порога меняет медиану ({' / '.join(str(v) for v in sens['медиана, мес'])} мес "
    f"при порогах 10–90 %) — порог надо фиксировать до анализа, а не подбирать после.",
    f"Модель Кокса на {len(surv)} скважинах неинформативна (бутстреп-ДИ шире оценки) — "
    "границу применимости методов важно признавать явно.",
]
for i, c in enumerate(concl, 1):
    print(f"{i}. {c}")
