import time
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from collections import Counter
from math import comb

# =========================
# 0) 한글 폰트 자동 설정
# =========================
def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    installed = set(f.name for f in font_manager.fontManager.ttflist)
    for name in candidates:
        if name in installed:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# 1) 로또 당첨번호 데이터 소스 (동행복권 API가 해외 IP를 차단하므로
#    GitHub에 미러링된 회차별 JSON을 사용)
# =========================
BASE_URL = "https://raw.githubusercontent.com/smok95/lotto/main/results"

def fetch_draw(drw_no: int, session: requests.Session = None, timeout: float = 5.0):
    s = session or requests.Session()
    try:
        r = s.get(f"{BASE_URL}/{drw_no}.json", timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        numbers = data.get("numbers")
        if not numbers or len(numbers) != 6:
            return None
        return {
            "drwNo": data.get("draw_no"),
            "drwNoDate": str(data.get("date", ""))[:10],
            "drwtNo1": numbers[0],
            "drwtNo2": numbers[1],
            "drwtNo3": numbers[2],
            "drwtNo4": numbers[3],
            "drwtNo5": numbers[4],
            "drwtNo6": numbers[5],
            "bnusNo": data.get("bonus_no"),
        }
    except Exception:
        return None

def fetch_all_draws(start_no: int = 1, max_consecutive_fail: int = 5, sleep_sec: float = 0.05):
    s = requests.Session()
    draws = []
    drw_no = start_no
    fails = 0

    while True:
        data = fetch_draw(drw_no, session=s)
        if data is not None:
            row = {
                "회차": data.get("drwNo"),
                "추첨일": data.get("drwNoDate"),
                "번호1": data.get("drwtNo1"),
                "번호2": data.get("drwtNo2"),
                "번호3": data.get("drwtNo3"),
                "번호4": data.get("drwtNo4"),
                "번호5": data.get("drwtNo5"),
                "번호6": data.get("drwtNo6"),
                "보너스": data.get("bnusNo"),
            }
            draws.append(row)
            fails = 0
        else:
            fails += 1
            if fails >= max_consecutive_fail:
                break
        drw_no += 1
        time.sleep(sleep_sec)

    draws.sort(key=lambda x: x["회차"])
    return pd.DataFrame(draws)

# =========================
# 2) 자리별 확률 분포 (실데이터 기반)
# =========================
def position_probability_from_real(df_draws: pd.DataFrame, n_numbers: int = 45, pick: int = 6):
    if df_draws.empty:
        raise ValueError("입력 데이터프레임이 비어 있습니다.")
    numbers = df_draws[["번호1","번호2","번호3","번호4","번호5","번호6"]].to_numpy()
    sorted_numbers = np.sort(numbers, axis=1)

    freq = {pos: {num: 0 for num in range(1, n_numbers+1)} for pos in range(1, pick+1)}
    for row in sorted_numbers:
        for pos in range(pick):
            num = int(row[pos])
            freq[pos+1][num] += 1

    total = len(sorted_numbers)
    for pos in freq:
        for num in freq[pos]:
            freq[pos][num] = freq[pos][num] / total

    df_prob = pd.DataFrame(freq)
    df_prob.index.name = "번호"
    df_prob.columns = [f"{i}번째 작은 수" for i in range(1, pick+1)]
    return df_prob

# =========================
# 3) 그래프 시각화 (자리분포)
# =========================
def plot_distributions(df_prob: pd.DataFrame, title: str):
    x = df_prob.index.to_numpy()
    plt.figure(figsize=(12, 6))
    for col in df_prob.columns:
        y = df_prob[col].to_numpy()
        plt.plot(x, y, label=col, linewidth=2)
    plt.title(title)
    plt.xlabel("번호")
    plt.ylabel("출현 확률")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.close()

# =========================
# 3-1) 이전 회차와 겹침 통계
# =========================
def overlap_with_previous_detail(df_draws: pd.DataFrame):
    number_cols = ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"]
    rows = []
    counts = Counter()

    for i in range(1, len(df_draws)):
        curr_round = int(df_draws.loc[i, "회차"])
        prev_round = int(df_draws.loc[i-1, "회차"])

        curr_set = set(int(x) for x in df_draws.loc[i, number_cols].to_list())
        prev_set = set(int(x) for x in df_draws.loc[i-1, number_cols].to_list())

        inter = sorted(list(curr_set & prev_set))
        k = len(inter)
        counts[k] += 1

        rows.append({
            "회차": curr_round,
            "이전회차": prev_round,
            "겹치는개수": k,
            "겹친번호": inter
        })

    df_overlap = pd.DataFrame(rows)
    return df_overlap, counts

def summarize_overlap_counts(counts: Counter):
    keys = sorted(counts.keys())
    total = sum(counts.values())
    data = {
        "겹치는개수": keys,
        "횟수(회)": [counts[k] for k in keys],
        "비율(%)": [counts[k] * 100.0 / total if total > 0 else 0.0 for k in keys]
    }
    df_summary = pd.DataFrame(data)
    return df_summary, total

def plot_overlap_bar(counts: Counter, title="이전 회차와 겹치는 번호 개수 분포(빈도)"):
    set_korean_font()
    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("겹치는 번호 개수")
    plt.ylabel("빈도(회)")
    plt.xticks(x)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.close()

# =========================
# 3-2) 이론 기대분포(하이퍼지오메트릭) & 비율 막대 + CDF
# =========================
def theoretical_overlap_pmf():
    denom = comb(45, 6)
    pmf = {k: (comb(6, k) * comb(39, 6 - k)) / denom for k in range(0, 7)}
    return pmf

def plot_overlap_ratio_vs_theory(counts: Counter, title="이전 회차 겹침: 실측 비율 vs 이론 분포"):
    set_korean_font()
    keys = list(range(0, 7))
    total = sum(counts.values())
    empirical = [counts.get(k, 0) / total if total > 0 else 0.0 for k in keys]
    theo_pmf = theoretical_overlap_pmf()
    theory = [theo_pmf.get(k, 0.0) for k in keys]

    plt.figure(figsize=(9, 5))
    plt.bar(keys, empirical)
    plt.plot(keys, theory, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("겹치는 번호 개수")
    plt.ylabel("비율")
    plt.xticks(keys)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.close()

def plot_overlap_cdf(counts: Counter, title="이전 회차 겹침: 실측 CDF vs 이론 CDF"):
    set_korean_font()
    keys = list(range(0, 7))
    total = sum(counts.values())
    empirical = [counts.get(k, 0) / total if total > 0 else 0.0 for k in keys]
    theo_pmf = theoretical_overlap_pmf()
    theory = [theo_pmf.get(k, 0.0) for k in keys]

    emp_cdf = np.cumsum(empirical)
    th_cdf = np.cumsum(theory)

    plt.figure(figsize=(9, 5))
    plt.step(keys, emp_cdf, where="post", linewidth=2, label="실측 CDF")
    plt.plot(keys, th_cdf, marker="o", linewidth=2, label="이론 CDF")
    plt.title(title)
    plt.xlabel("겹치는 번호 개수 (이하)")
    plt.ylabel("누적 비율")
    plt.xticks(keys)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.close()

# =========================
# 5) 겹칩 통계 기반 목표 벡터 생성 & 정확히 부합하도록 조합 생성
# =========================
def build_empirical_k_pmf(counts: Counter):
    total = sum(counts.values())
    if total == 0:
        return [1/7.0]*7
    return [counts.get(k, 0)/total for k in range(7)]

def sample_with_mask(nums, probs, mask_set=None, gt=None, le=None, exclude_set=None):
    cand, w = [], []
    for n, p in zip(nums, probs):
        if gt is not None and not (n > gt): continue
        if le is not None and not (n <= le): continue
        if exclude_set and (n in exclude_set): continue
        if mask_set is not None and (n not in mask_set): continue
        if p <= 0: continue
        cand.append(n); w.append(p)
    if not cand:
        return None, 0
    w = np.array(w, dtype=float); w = w / w.sum()
    choice = np.random.choice(cand, p=w)
    return int(choice), len(cand)

def generate_one_combo_with_fixed_k(df_draws: pd.DataFrame, df_prob: pd.DataFrame, fixed_k: int, max_tries=4000):
    last = df_draws.iloc[-1]
    last_set = set(int(x) for x in last[["번호1","번호2","번호3","번호4","번호5","번호6"]].to_list())
    all_nums = np.arange(1, 46)
    cols = [f"{i}번째 작은 수" for i in range(1, 7)]
    base_prob_cols = [df_prob[col].to_numpy() for col in cols]
    k = int(fixed_k)

    for _ in range(max_tries):
        selected, used, overlaps = [], set(), 0
        feasible = True
        for pos in range(6):
            remaining = 6 - (pos + 1)
            prev_val = selected[-1] if selected else 0
            upper = 45 - remaining

            need = k - overlaps
            must_from_last = None
            if need == remaining + 1:
                must_from_last = True
            elif need <= 0:
                must_from_last = False

            include_set = None
            if must_from_last is True:
                include_set = last_set
            elif must_from_last is False:
                include_set = set(range(1,46)) - last_set

            val, _ = sample_with_mask(
                nums=all_nums, probs=base_prob_cols[pos],
                mask_set=include_set, gt=prev_val, le=upper, exclude_set=used
            )
            if val is None:
                feasible = False; break
            selected.append(val); used.add(val)
            if val in last_set: overlaps += 1

        if feasible and overlaps == k and len(selected) == 6:
            return sorted(selected)

    # 백업(희귀): last_set에서 k개 + 평균확률 상위 채움
    mean_probs = np.mean(np.vstack([df_prob[c].to_numpy() for c in cols]), axis=0)
    base = sorted(list(np.random.choice(list(last_set), size=min(k,6), replace=False)))
    need = 6 - len(base)
    cand = [(n, mean_probs[n-1]) for n in range(1,46) if n not in set(base) and n not in last_set]
    cand.sort(key=lambda x: x[1], reverse=True)
    base = sorted(base + [n for n,_ in cand[:need]])
    return base[:6]

# ---- 통계 → 목표 벡터 만들기
def expand_ge2(counts: Counter, m_ge2: int, strategy="mode", seed=None):
    """
    'k>=2'로 모은 m_ge2개를 실제 k∈{2..6}로 분배.
    - strategy='mode': 2..6 구간의 최빈값으로 채움(성공률↑)
    - strategy='sample': 2..6 조건부 PMF로 샘플링
    """
    if seed is not None:
        np.random.seed(seed)
    ge2_total = sum(counts.get(k,0) for k in range(2,7))
    if ge2_total == 0:
        return [2]*m_ge2
    pmf_ge2 = np.array([counts.get(k,0)/ge2_total for k in range(2,7)], dtype=float)

    if strategy == "sample":
        return list(np.random.choice(np.arange(2,7), size=m_ge2, p=pmf_ge2))

    # mode
    k_mode = int(np.argmax(pmf_ge2) + 2)
    return [k_mode]*m_ge2

def target_ks_from_empirical(counts: Counter, n=10, collapse_ge2=True, ge2_strategy="mode", seed=None):
    """
    해밀턴 방식으로 counts(0..6) -> 길이 n의 목표 k 벡터 생성.
    collapse_ge2=True면 {0,1,GE2}로 좌석 배분 후 GE2를 2..6으로 펼침.
    """
    if seed is not None:
        np.random.seed(seed)
    total = sum(counts.values())
    if total == 0:
        return [0]*n
    pk = {k: counts.get(k,0)/total for k in range(7)}

    if collapse_ge2:
        probs = [pk[0], pk[1], sum(pk[k] for k in range(2,7))]
        quotas = [p*n for p in probs]
        base = [int(np.floor(q)) for q in quotas]
        rem = n - sum(base)
        fracs = [q - b for q,b in zip(quotas, base)]
        order = list(np.argsort(-np.array(fracs)))
        for idx in order[:rem]:
            base[idx] += 1

        k_list = [0]*base[0] + [1]*base[1]
        m_ge2 = base[2]
        if m_ge2 > 0:
            k_list += expand_ge2(counts, m_ge2, strategy=ge2_strategy, seed=seed)
        np.random.shuffle(k_list)
        return k_list
    else:
        quotas = [pk[k]*n for k in range(7)]
        base = [int(np.floor(q)) for q in quotas]
        rem = n - sum(base)
        fracs = [q - b for q,b in zip(quotas, base)]
        order = list(np.argsort(-np.array(fracs)))
        for idx in order[:rem]:
            base[idx] += 1
        k_list = []
        for k in range(7):
            k_list += [k]*base[k]
        np.random.shuffle(k_list)
        return k_list

# ---- 목표 벡터를 '정확히' 맞추는 배치 생성
def generate_combos_with_target_ks(df_draws: pd.DataFrame, df_prob: pd.DataFrame, target_ks,
                                   max_batch_tries=300, per_combo_max_tries=4000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    latest_set = set(int(x) for x in df_draws.iloc[-1][["번호1","번호2","번호3","번호4","번호5","번호6"]].to_list())
    n = len(target_ks)

    for _attempt in range(1, max_batch_tries+1):
        order = sorted(range(n), key=lambda i: target_ks[i], reverse=True)
        combos, seen, ok = [None]*n, set(), True

        for idx in order:
            k = int(target_ks[idx])
            c = generate_one_combo_with_fixed_k(df_draws, df_prob, fixed_k=k, max_tries=per_combo_max_tries)
            t = tuple(c)
            retry = 0
            while t in seen and retry < 100:
                c = generate_one_combo_with_fixed_k(df_draws, df_prob, fixed_k=k, max_tries=per_combo_max_tries)
                t = tuple(c); retry += 1
            if t in seen:
                ok = False; break
            seen.add(t); combos[idx] = c

        if not ok or any(c is None for c in combos):
            continue

        actual_ks = [len(set(c) & latest_set) for c in combos]
        from collections import Counter as C
        if C(actual_ks) == C(target_ks):
            return combos, target_ks, actual_ks

    return combos, target_ks, actual_ks

# =========================
# 6) 실행부
# =========================
if __name__ == "__main__":
    print("로또 실제 당첨 데이터 수집 중... (동행복권 API)")
    df_draws = fetch_all_draws(start_no=1, max_consecutive_fail=5, sleep_sec=0.05)
    print(f"수집 완료: {len(df_draws)} 회차")
    df_draws.to_csv("lotto_draws.csv", index=False, encoding="utf-8-sig")
    print("CSV 저장: lotto_draws.csv")

    # ----- 자리별 확률 분포 -----
    df_prob = position_probability_from_real(df_draws, n_numbers=45, pick=6)
    print("\n[자리별 확률 분포 - 일부 미리보기]")
    print(df_prob.head())
    plot_distributions(df_prob, title="실제 로또(대한민국 6/45) 번호 자리별 확률 분포")

    # ----- 이전 회차와의 겹침 통계 -----
    df_overlap, counts = overlap_with_previous_detail(df_draws)
    df_overlap.to_csv("lotto_overlap_with_prev.csv", index=False, encoding="utf-8-sig")
    print("\nCSV 저장: lotto_overlap_with_prev.csv (회차별 겹친 번호 상세)")

    df_summary, total_pairs = summarize_overlap_counts(counts)
    print("\n[이전 회차와 겹치는 번호 개수 요약]")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_summary.to_string(index=False, formatters={"비율(%)": lambda v: f"{v:.2f}"}))
    print(f"\n총 비교 횟수: {total_pairs} (2회차부터 현재 회차까지)")

    # ----- 시각화: (1) 빈도, (2) 실측 비율 vs 이론 PMF, (3) CDF 비교 -----
    plot_overlap_bar(counts, title="이전 회차와 겹치는 번호 개수 분포(빈도)")
    plot_overlap_ratio_vs_theory(counts, title="이전 회차 겹침: 실측 비율 vs 이론 분포")
    plot_overlap_cdf(counts, title="이전 회차 겹침: 실측 CDF vs 이론 CDF")

    # ----- 요약 CSV: 실측/이론 PMF·CDF -----
    theo = theoretical_overlap_pmf()
    keys = list(range(0,7))
    total = sum(counts.values())
    empirical_ratio = [counts.get(k,0)/total if total>0 else 0.0 for k in keys]
    df_compare = pd.DataFrame({
        "겹치는개수": keys,
        "실측_횟수": [counts.get(k,0) for k in keys],
        "실측_비율": empirical_ratio,
        "이론_비율": [theo[k] for k in keys],
        "실측_CDF": np.cumsum(empirical_ratio),
        "이론_CDF": np.cumsum([theo[k] for k in keys]),
    })
    df_compare.to_csv("lotto_overlap_empirical_vs_theory.csv", index=False, encoding="utf-8-sig")
    print("CSV 저장: lotto_overlap_empirical_vs_theory.csv (실측/이론 PMF·CDF 비교)")
    
    # ----- 가장 최근 로또 번호 -----
    latest_draw = df_draws.iloc[-1]
    latest_numbers = [latest_draw["번호1"], latest_draw["번호2"], latest_draw["번호3"],
                      latest_draw["번호4"], latest_draw["번호5"], latest_draw["번호6"]]
    latest_set = set(int(x) for x in latest_numbers)
    print("\n[가장 최근 로또 번호]")
    print(f"{latest_draw['회차']}회 ({latest_draw['추첨일']}): {sorted(latest_numbers)} + 보너스 {latest_draw['보너스']}")

    # ----- (핵심) 통계 기반 목표 벡터 생성 → 목표 정확히 만족하도록 10개 조합 생성 -----
    target_ks = target_ks_from_empirical(counts, n=5, collapse_ge2=True, ge2_strategy="mode", seed=None)
    print("\n[겹칩 통계 기반 목표 겹침 벡터]")
    print(target_ks)

    print("\n[겹칩 통계(목표 벡터)에 정확히 부합하도록 조합 10개 생성 중...]")
    combos, target_ks, actual_ks = generate_combos_with_target_ks(
        df_draws, df_prob, target_ks, max_batch_tries=300, per_combo_max_tries=4000, seed=None
    )

    from collections import Counter as C
    print("\n[목표 겹침 분포]")
    print(dict(C(target_ks)))
    print("[실제 생성 조합의 겹침 분포]")
    print(dict(C(actual_ks)))

    rows = []
    print("\n[생성된 조합 10개 - 최근 회차와의 겹침 정보]")
    for i, c in enumerate(combos, 1):
        inter = sorted(list(set(c) & latest_set))
        k = len(inter)
        print(f"{i:02d}) {c}  | 최근 회차와 {k}개 겹침: {inter}")
        rows.append({
            "번호조합": ",".join(map(str, c)),
            "겹치는개수": k,
            "겹친번호": ";".join(map(str, inter))
        })

    pd.DataFrame(rows).to_csv("generated_combos_WITH_TARGET_FROM_STATS.csv", index=False, encoding="utf-8-sig")
    print("\nCSV 저장: generated_combos_WITH_TARGET_FROM_STATS.csv (통계 기반 목표 벡터와 일치)")
