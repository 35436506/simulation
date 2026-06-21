"""
Monte Carlo Simulation App
Inspired by forecasting_app.py design language (navy/teal/gold palette, dark charts).
Deploy: streamlit run simulation_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import io, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (light theme)
# ══════════════════════════════════════════════════════════════════════════════
DARK   = "#ffffff"   # chart bg
PANEL  = "#f6f8fa"   # panel / card bg
GRID   = "#d0d7de"   # grid lines
WHITE  = "#1a1a2e"   # text on charts
GRAY   = "#57606a"   # secondary text
BLUE   = "#0969da"   # primary accent
TEAL   = "#1a7f37"   # success / positive
GOLD   = "#953800"   # warning / highlight
RED    = "#cf222e"   # danger / negative
PURPLE = "#6639ba"   # purple accent
PINK   = "#e85c8a"   # pink accent

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simulation Analyst",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background:#ffffff; }
[data-testid="stMain"] { background:#ffffff; }
[data-testid="stSidebar"]          { background:#f6f8fa; border-right:1px solid #d0d7de; }
[data-testid="stSidebar"] *        { color:#1f2328 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color:#57606a !important; }

/* ── Streamlit default text overrides ── */
p, li, span, label, div { color:#1f2328; }
h1, h2, h3, h4 { color:#1f2328; }

/* ── Hero ── */
.hero-title {
    font-size:2.1rem; font-weight:700; letter-spacing:-.03em;
    background:linear-gradient(90deg,#0969da,#1a7f37,#953800);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.15rem;
}
.hero-sub {
    font-size:.93rem; color:#57606a; margin-bottom:1.4rem; max-width:780px;
}

/* ── Section headers ── */
.section-hdr {
    font-size:.78rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#57606a;
    border-bottom:1px solid #d0d7de;
    padding:.35rem 0 .45rem; margin:1.2rem 0 .7rem;
}
.sec-num {
    display:inline-block; background:#0969da; color:#fff;
    border-radius:50%; width:1.35em; height:1.35em;
    text-align:center; line-height:1.35em;
    font-size:.8em; margin-right:.45rem;
}

/* ── Info / warn / error boxes ── */
.info-box {
    background:#ddf4ff; border:1px solid #54aeff; border-left:4px solid #0969da;
    border-radius:8px; padding:.75rem 1rem;
    font-size:.85rem; color:#1f2328; margin:.6rem 0;
}
.warn-box {
    background:#fff8c5; border:1px solid #d4a72c; border-left:4px solid #953800;
    border-radius:8px; padding:.65rem .9rem;
    font-size:.84rem; color:#6e4f00; margin:.5rem 0;
}
.err-box {
    background:#ffebe9; border:1px solid #ff8182; border-left:4px solid #cf222e;
    border-radius:8px; padding:.65rem .9rem;
    font-size:.84rem; color:#82071e; margin:.5rem 0;
}
.success-box {
    background:#dafbe1; border:1px solid #4ac26b; border-left:4px solid #1a7f37;
    border-radius:8px; padding:.65rem .9rem;
    font-size:.84rem; color:#116329; margin:.5rem 0;
}
.interpret-box {
    background:#f6f8fa; border:1px solid #d0d7de; border-left:4px solid #1a7f37;
    border-radius:8px; padding:.75rem 1rem;
    font-size:.84rem; color:#1f2328; margin:.6rem 0; line-height:1.6;
}

/* ── Metric cards ── */
.metric-card {
    background:#f6f8fa; border:1px solid #d0d7de; border-radius:10px;
    padding:.9rem 1.1rem; text-align:center;
}
.metric-label { font-size:.72rem; color:#57606a; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.6rem; font-weight:700; color:#0969da; }
.metric-sub   { font-size:.72rem; color:#6e7781; }

/* ── Formula box ── */
.formula-box {
    background:#eaeef2; border:1px solid #0969da; border-radius:8px;
    padding:.7rem 1.1rem; margin:.5rem 0;
    font-family:'Courier New',monospace; font-size:.92rem;
    color:#953800; letter-spacing:.02em;
}
.formula-desc { font-size:.78rem; color:#57606a; margin-top:.25rem; }

/* ── Step badge ── */
.step-badge {
    display:inline-block; background:#0969da; color:#fff;
    border-radius:6px; padding:.15rem .55rem;
    font-size:.75rem; font-weight:700; margin-right:.4rem;
}

/* ── VaR highlight ── */
.var-highlight {
    background:#eaeef2; border:1px solid #0969da; border-radius:8px;
    padding:.7rem 1rem; font-size:.85rem; color:#1f2328; margin:.6rem 0;
}

/* ── Distribution tag ── */
.dist-tag {
    display:inline-block; border-radius:99px;
    padding:.12rem .55rem; font-size:.72rem; font-weight:600;
    margin:.1rem .15rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (same style as forecasting_app)
# ══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=GRAY, fontsize=8, labelpad=5)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=8, labelpad=5)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


def new_fig(nrows=1, ncols=1, figsize=(10, 4), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(DARK)
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            ax.set_facecolor(PANEL)
    else:
        axes.set_facecolor(PANEL)
    return fig, axes


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def metric_html(label, value, sub="", color=BLUE):
    return f"""<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value" style="color:{color}">{value}</div>
    <div class="metric-sub">{sub}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def mc_health_insurance(n_emp, avg_claim, contrib, emp_change_min, emp_change_max,
                         claim_mean, claim_std, n_sims, seed=42):
    """Bài toán 1: Health Insurance (Fig 12-6)."""
    rng = np.random.default_rng(seed)
    total_costs = np.empty(n_sims)
    monthly_costs = np.zeros((n_sims, 12))
    for i in range(n_sims):
        emp, claim, total = float(n_emp), float(avg_claim), 0.0
        for m in range(12):
            emp_change  = rng.uniform(emp_change_min, emp_change_max)
            claim_change = rng.normal(claim_mean, claim_std)
            emp   = max(1, emp   * (1 + emp_change))
            claim = max(0, claim * (1 + claim_change))
            cost  = max(0.0, emp * claim - contrib * emp)
            monthly_costs[i, m] = cost
            total += cost
        total_costs[i] = total
    return total_costs, monthly_costs


@st.cache_data(show_spinner=False)
def mc_airline_overbooking(seats, ticket_price, bump_cost, p_noshow,
                            demand_vals, demand_probs,
                            res_min, res_max, n_sims, seed=42):
    """Bài toán 2: Airline Overbooking (Fig 12-18)."""
    rng = np.random.default_rng(seed)
    demand_vals  = np.array(demand_vals)
    demand_probs = np.array(demand_probs, dtype=float)
    demand_probs /= demand_probs.sum()
    results = {}
    for R in range(int(res_min), int(res_max) + 1):
        profits = np.empty(n_sims)
        for i in range(n_sims):
            demand      = rng.choice(demand_vals, p=demand_probs)
            tickets_sold = min(demand, R)
            showups      = rng.binomial(tickets_sold, 1 - p_noshow)
            bumped       = max(0, showups - seats)
            profits[i]   = tickets_sold * ticket_price - bumped * bump_cost
        results[R] = profits
    return results


@st.cache_data(show_spinner=False)
def mc_inventory(init_inv, reorder_pt, order_qty, n_days, n_sims,
                 holding_cost, order_cost, shortage_cost,
                 demand_vals, demand_probs, lead_vals, lead_probs, seed=42):
    """Bài toán 3: Inventory (s, Q) control (Fig 12-23)."""
    rng = np.random.default_rng(seed)
    dv = np.array(demand_vals)
    dp = np.array(demand_probs, dtype=float); dp /= dp.sum()
    lv = np.array(lead_vals)
    lp = np.array(lead_probs, dtype=float);  lp /= lp.sum()

    svc_levels, avg_invs, total_costs = np.empty(n_sims), np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        inv = float(init_inv)
        orders: dict = {}
        tot_demand = tot_served = inv_sum = n_orders = shortage_sum = 0
        for day in range(1, n_days + 1):
            inv += orders.pop(day, 0)
            inv_pos = inv + sum(orders.values())
            d = int(rng.choice(dv, p=dp))
            served = min(inv, d); short = d - served
            inv -= served
            tot_demand += d; tot_served += served
            inv_sum += max(0, inv); shortage_sum += short
            if inv_pos <= reorder_pt:
                lead = int(rng.choice(lv, p=lp))
                arrive = day + lead
                orders[arrive] = orders.get(arrive, 0) + order_qty
                n_orders += 1
        svc_levels[i]  = tot_served / tot_demand if tot_demand > 0 else 1.0
        avg_invs[i]    = inv_sum / n_days
        total_costs[i] = (holding_cost * inv_sum
                          + order_cost  * n_orders
                          + shortage_cost * shortage_sum)
    return svc_levels, avg_invs, total_costs


@st.cache_data(show_spinner=False)
def mc_project_selection(projects_df, n_sims, seed=42):
    """Bài toán 4: Project Selection (Fig 12-39)."""
    rng = np.random.default_rng(seed)
    sel = projects_df[projects_df["Selected"]].reset_index(drop=True)
    profits = np.zeros(n_sims)
    for i in range(n_sims):
        total = 0.0
        for _, row in sel.iterrows():
            success = rng.random() < row["P_Success"]
            if success:
                rev = rng.triangular(row["Rev_Min"], row["Rev_MLikely"], row["Rev_Max"])
                total += rev - row["Investment"]
            else:
                total -= row["Investment"]
        profits[i] = total
    return profits


@st.cache_data(show_spinner=False)
def mc_portfolio(means, stds, corr_matrix, weights, n_sims, seed=42):
    """Bài toán 5: Portfolio Optimization (Fig 12-43)."""
    rng = np.random.default_rng(seed)
    means = np.array(means); stds = np.array(stds)
    corr  = np.array(corr_matrix); weights = np.array(weights)
    cov   = np.outer(stds, stds) * corr
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(len(means)) * 1e-8
        L = np.linalg.cholesky(cov)
    Z       = rng.standard_normal((n_sims, len(means)))
    returns = Z @ L.T + means
    portfolio_returns = (returns * weights).sum(axis=1)
    return returns, portfolio_returns


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def summary_stats(data, target=None):
    d = np.asarray(data)
    s = {
        "Mean":   np.mean(d),
        "Std Dev": np.std(d, ddof=1),
        "Min":    np.min(d),
        "P10":    np.percentile(d, 10),
        "Median": np.median(d),
        "P90":    np.percentile(d, 90),
        "Max":    np.max(d),
    }
    if target is not None:
        s["P(Y ≤ Target)"] = np.mean(d <= target)
        s["P(Y > Target)"] = np.mean(d >  target)
    return s


def confidence_interval_mean(data, alpha=0.05):
    n = len(data)
    xbar = np.mean(data)
    s    = np.std(data, ddof=1)
    se   = s / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return xbar, xbar - t_crit * se, xbar + t_crit * se


def plot_histogram(data, title, xlabel, target=None, color=BLUE, bins=50):
    fig, axes = new_fig(1, 2, figsize=(11, 3.8))
    ax_hist, ax_cdf = axes

    # ── Histogram ──
    n_arr = np.asarray(data)
    ax_hist.hist(n_arr, bins=bins, color=color, alpha=0.75, density=True, edgecolor=DARK, linewidth=0.3)
    mu, sigma = np.mean(n_arr), np.std(n_arr)
    xr = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    ax_hist.plot(xr, stats.norm.pdf(xr, mu, sigma), color=GOLD, linewidth=1.5,
                 linestyle="--", label="Normal fit")
    if target is not None:
        ax_hist.axvline(target, color=RED, linewidth=1.5, linestyle=":", label=f"Target")
    ax_hist.axvline(mu, color=TEAL, linewidth=1.3, linestyle="-.", label=f"Mean")
    ax_hist.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
    style_ax(ax_hist, title, xlabel, "Probability Density")

    # ── CDF ──
    sorted_d = np.sort(n_arr)
    cdf_y    = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
    ax_cdf.plot(sorted_d, cdf_y, color=color, linewidth=1.8)
    if target is not None:
        prob = np.mean(n_arr <= target) * 100
        ax_cdf.axvline(target, color=RED, linewidth=1.3, linestyle=":")
        ax_cdf.axhline(prob,   color=RED, linewidth=1.0, linestyle=":")
        ax_cdf.annotate(f"P(≤T)={prob:.1f}%",
                        xy=(target, prob), xytext=(target + (sorted_d[-1]-sorted_d[0])*0.02, prob + 3),
                        color=RED, fontsize=7,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    style_ax(ax_cdf, "Cumulative Distribution (CDF)", xlabel, "Cumulative Probability (%)")
    fig.tight_layout(pad=1.5)
    return fig


def plot_tornado(data_dict, title, output_label, n_sims=2000, seed=99):
    """Sensitivity (Tornado) chart — vary one input at a time ±1 std."""
    impacts = {}
    for name, (samples_lo, samples_hi) in data_dict.items():
        lo = np.mean(np.asarray(samples_lo))
        hi = np.mean(np.asarray(samples_hi))
        impacts[name] = (lo, hi, abs(hi - lo))
    sorted_items = sorted(impacts.items(), key=lambda x: x[1][2], reverse=True)[:8]
    fig, ax = new_fig(figsize=(9, max(3, len(sorted_items) * 0.55 + 1)))
    y_pos = np.arange(len(sorted_items))
    for i, (name, (lo, hi, _)) in enumerate(sorted_items):
        ax.barh(i, hi - lo, left=lo, color=BLUE, alpha=0.75, height=0.55)
        ax.barh(i, 0, left=lo, color=TEAL, alpha=0.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x[0] for x in sorted_items], color=WHITE, fontsize=8)
    style_ax(ax, f"Tornado Chart — Sensitivity of {output_label}", output_label, "")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:18px; margin-bottom:.2rem; margin-top:.4rem;">

  <!-- Octagon SVG logo -->
  <svg width="72" height="72" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="octGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%"   stop-color="#e63946"/>
        <stop offset="16%"  stop-color="#f4a261"/>
        <stop offset="33%"  stop-color="#f7c948"/>
        <stop offset="50%"  stop-color="#43aa8b"/>
        <stop offset="66%"  stop-color="#4895ef"/>
        <stop offset="83%"  stop-color="#7b5ea7"/>
        <stop offset="100%" stop-color="#e63946"/>
      </linearGradient>
      <linearGradient id="innerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%"   stop-color="#0d1b2a"/>
        <stop offset="100%" stop-color="#1b2838"/>
      </linearGradient>
      <linearGradient id="chartGrad" x1="0%" y1="100%" x2="0%" y2="0%">
        <stop offset="0%"   stop-color="#4895ef" stop-opacity="0.3"/>
        <stop offset="100%" stop-color="#4895ef" stop-opacity="0.9"/>
      </linearGradient>
      <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"   stop-color="#f7c948"/>
        <stop offset="100%" stop-color="#e63946"/>
      </linearGradient>
    </defs>
    <!-- Octagon border (stroke gradient via thick stroke) -->
    <polygon
      points="22,4 50,4 68,22 68,50 50,68 22,68 4,50 4,22"
      fill="url(#innerGrad)"
      stroke="url(#octGrad)"
      stroke-width="3.5"
    />
    <!-- Mini bar chart inside -->
    <rect x="13" y="42" width="7"  height="16" rx="1.5" fill="url(#chartGrad)" opacity=".85"/>
    <rect x="23" y="34" width="7"  height="24" rx="1.5" fill="url(#chartGrad)" opacity=".85"/>
    <rect x="33" y="26" width="7"  height="32" rx="1.5" fill="url(#chartGrad)" opacity=".85"/>
    <rect x="43" y="30" width="7"  height="28" rx="1.5" fill="url(#chartGrad)" opacity=".85"/>
    <rect x="53" y="20" width="7"  height="38" rx="1.5" fill="url(#chartGrad)" opacity=".85"/>
    <!-- Trend line -->
    <polyline
      points="16,42 26,34 36,26 46,30 56,20"
      fill="none" stroke="url(#lineGrad)" stroke-width="2.2"
      stroke-linecap="round" stroke-linejoin="round"
    />
    <!-- Dots on line -->
    <circle cx="16" cy="42" r="2.2" fill="#f7c948"/>
    <circle cx="26" cy="34" r="2.2" fill="#43aa8b"/>
    <circle cx="36" cy="26" r="2.2" fill="#4895ef"/>
    <circle cx="46" cy="30" r="2.2" fill="#7b5ea7"/>
    <circle cx="56" cy="20" r="2.2" fill="#e63946"/>
  </svg>

  <!-- Title -->
  <div>
    <div style="
      font-size:2.2rem; font-weight:800; letter-spacing:-.04em; line-height:1.1;
      background: linear-gradient(90deg,
        #e63946 0%, #f4a261 18%, #f7c948 34%,
        #43aa8b 50%, #4895ef 66%, #7b5ea7 82%, #e63946 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    ">Simulation Analyst</div>
    <div style="font-size:.9rem; color:#57606a; margin-top:.15rem;">
      Monte Carlo Simulation
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Chọn bài toán từ sidebar → nhập tham số → chạy mô phỏng → đọc phân phối, VaR và khoảng tin cậy.</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation & shared settings
# ══════════════════════════════════════════════════════════════════════════════
MODULES = {
    "📂 Upload Excel Files":         "custom",
    "📖 Lý thuyết & Hướng dẫn":    "theory",
    "🏥 Health Insurance":          "health",
    "✈️ Airline Overbooking":       "airline",
    "📦 Inventory Control (s,Q)":   "inventory",
    "🗂️ Project Selection":         "project",
    "📈 Portfolio Optimization":    "portfolio",
    "🎰 RNG Playground":            "rng",
}

with st.sidebar:
    st.markdown('<div class="section-hdr"><span class="sec-num">①</span> CHỌN BÀI TOÁN</div>', unsafe_allow_html=True)
    module_label = st.radio("", list(MODULES.keys()), label_visibility="collapsed")
    MODULE = MODULES[module_label]

    st.markdown('<div class="section-hdr"><span class="sec-num">②</span> THIẾT LẬP MÔ PHỎNG</div>', unsafe_allow_html=True)
    n_sims = st.selectbox("Số lần lặp (replications)", [500, 1000, 2000, 5000, 10000], index=2)
    ci_alpha = st.selectbox("Độ tin cậy CI", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}")
    seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE: THEORY
# ══════════════════════════════════════════════════════════════════════════════
if MODULE == "theory":
    st.markdown('<div class="section-hdr"><span class="sec-num">①</span> TẠI SAO CẦN MÔ PHỎNG?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    Khi mô hình <b>Y = f(X₁, X₂, …, Xₖ)</b> có một hoặc nhiều Xᵢ là biến ngẫu nhiên
    (không chắc chắn), thì <b>Y cũng là biến ngẫu nhiên</b>. Chỉ dùng giá trị kỳ vọng
    E[Xᵢ] → mất đi toàn bộ thông tin về biến động và rủi ro.<br><br>
    <b>Monte Carlo Simulation</b> giải quyết điều này bằng cách lấy mẫu ngẫu nhiên
    hàng nghìn lần để xây dựng phân phối đầy đủ của Y.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""**✅ Best/Worst-case**
- Dễ thực hiện
- Chỉ có 2 kịch bản cực đoan
- ❌ Không biết xác suất""")
    c2.markdown("""**⚡ What-if Analysis**
- Khám phá 1 biến tại một thời điểm
- Thiên lệch chủ quan
- ❌ Không có xác suất""")
    c3.markdown("""**🎯 Monte Carlo Simulation**
- Phân phối đầy đủ
- Xác suất cụ thể (VaR)
- ✅ Khách quan & tự động""")

    st.divider()
    st.markdown('<div class="section-hdr"><span class="sec-num">②</span> QUY TRÌNH 6 BƯỚC</div>', unsafe_allow_html=True)

    steps = [
        ("Xây mô hình base case", "Tạo bảng tính / code với công thức tính Y từ Xᵢ. Chạy kịch bản 'trung bình' để kiểm tra logic."),
        ("Xác định biến ngẫu nhiên & chọn RNG", "Thay giá trị cố định bằng hàm phân phối phù hợp: Normal, Uniform, Triangular, Discrete, Binomial…"),
        ("Đánh dấu ô đầu ra cần theo dõi", "Chỉ định Y cần phân tích (tổng chi phí, lợi nhuận, service level…)."),
        ("Chọn số lần lặp (n)", "Thường 1,000–10,000. Nhiều hơn → chính xác hơn. Dùng SE = σ/√n để ước lượng."),
        ("Chạy mô phỏng", "Lặp n lần: lấy mẫu Xᵢ → tính Y → lưu. Kết quả: n giá trị Y."),
        ("Phân tích phân phối đầu ra", "Histogram, CDF, Mean, Std Dev, Percentile, P(Y > Target), Tornado chart."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f'<span class="step-badge">Bước {i}</span> <b style="color:{WHITE}">{title}</b>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-left:2.5rem;font-size:.84rem;color:{GRAY};margin-bottom:.6rem">{desc}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-hdr"><span class="sec-num">③</span> CÁC HÀM RNG (ANALYTIC SOLVER)</div>', unsafe_allow_html=True)

    rng_data = {
        "Normal(μ, σ)":         ["Normal", "Liên tục, đối xứng", "Chi phí, tốc độ tăng trưởng"],
        "Uniform(a, b)":        ["Uniform", "Mọi giá trị trong [a,b]", "Thay đổi % NV, tỷ lệ"],
        "Triangular(a, c, b)":  ["Triangular", "min / most-likely / max", "Ước tính 3-điểm (PM)"],
        "Discrete({v},{p})":    ["Discrete", "Rời rạc, xác suất cho trước", "Nhu cầu, lead time"],
        "Binomial(n, p)":       ["Binomial", "Số thành công trong n lần", "No-show, tỷ lệ lỗi"],
        "Lognormal(μ, σ)":      ["Lognormal", "Chỉ dương, lệch phải", "Giá tài sản, doanh thu"],
    }
    rng_df = pd.DataFrame(rng_data, index=["Phân phối", "Hình dạng", "Dùng khi"]).T
    rng_df.index.name = "Hàm phân phối"
    st.dataframe(rng_df, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-hdr"><span class="sec-num">④</span> CÔNG THỨC PHÂN TÍCH KẾT QUẢ</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="formula-box">CI = x̄ ± t*(s/√n)<div class="formula-desc">Khoảng tin cậy cho mean thực  ·  t* = t critical (df = n−1)</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">n = (z* · σ / E)²<div class="formula-desc">Số replications cần thiết để đạt sai số ±E</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="formula-box">P(Y ≤ T) = #{Y ≤ T} / n<div class="formula-desc">Xác suất kết quả không vượt ngưỡng T  (≡ CDF Target)</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">VaR: P(Y > T) ≤ α<div class="formula-desc">Value at Risk constraint  ·  α thường = 5% hoặc 2.5%</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    💡 <b>Gợi ý:</b> Chọn các bài toán từ sidebar để xem demo Monte Carlo trực tiếp.
    Mỗi bài toán giải thích từng bước và kết quả kèm diễn giải.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: HEALTH INSURANCE
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "health":
    st.markdown('<div class="section-hdr"><span class="sec-num">🏥</span> HEALTH INSURANCE — HUNGRY DAWG RESTAURANTS (Fig 12-6)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Bài toán:</b> Công ty có 18,533 NV tham gia bảo hiểm. Dự báo <b>tổng chi phí y tế</b> công ty phải bù
    đắp trong 12 tháng tới khi cả số NV lẫn chi phí bồi thường biến động ngẫu nhiên.<br>
    <b>Biến RNG:</b> Số NV thay đổi theo <code>Uniform</code>,
    chi phí BT/NV tăng theo <code>Normal</code>.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="section-hdr"><span class="sec-num">③</span> THAM SỐ</div>', unsafe_allow_html=True)
        n_emp    = st.number_input("Số NV ban đầu", value=18533, step=100)
        avg_claim = st.number_input("Chi phí BT TB/NV/tháng ($)", value=250, step=5)
        contrib   = st.number_input("Đóng góp NV/tháng ($)", value=125, step=5)
        emp_min   = st.slider("Thay đổi NV tối thiểu/tháng", -0.15, 0.0, -0.03, 0.005, format="%.3f")
        emp_max   = st.slider("Thay đổi NV tối đa/tháng",   0.0, 0.20,  0.07, 0.005, format="%.3f")
        c_mean    = st.slider("Tăng BT TB/tháng (mean)",    0.0, 0.05,  0.01, 0.001, format="%.3f")
        c_std     = st.slider("Tăng BT std dev/tháng",      0.0, 0.02, 0.003, 0.001, format="%.4f")
        target_m  = st.number_input("VaR Target ($M)", value=39.0, step=0.5) * 1e6
        run_btn   = st.button("▶ Chạy mô phỏng", use_container_width=True)

    with st.expander("📐 Công thức mô hình"):
        st.markdown(f"""
        <div class="formula-box">
        Nₜ = N₍ₜ₋₁₎ × (1 + Uniform({emp_min:.3f}, {emp_max:.3f}))
        <div class="formula-desc">Số nhân viên tháng t</div></div>
        <div class="formula-box">
        Cₜ = C₍ₜ₋₁₎ × (1 + Normal({c_mean:.3f}, {c_std:.4f}))
        <div class="formula-desc">Chi phí bồi thường/NV tháng t</div></div>
        <div class="formula-box">
        Company_Cost_t = MAX(0,  Nₜ × Cₜ  −  {contrib} × Nₜ)
        <div class="formula-desc">Chi phí công ty tháng t (NV đóng ${contrib}/tháng)</div></div>
        <div class="formula-box">
        Total_Cost = Σ Company_Cost_t  (t = 1 → 12)
        <div class="formula-desc">Kết quả cần phân tích</div></div>
        """, unsafe_allow_html=True)

    if run_btn or "health_data" not in st.session_state:
        with st.spinner("Đang chạy Monte Carlo…"):
            tc, mc = mc_health_insurance(
                n_emp, avg_claim, contrib, emp_min, emp_max,
                c_mean, c_std, n_sims, seed=seed)
            st.session_state.health_data = (tc, mc)

    tc, mc = st.session_state.health_data
    ss = summary_stats(tc, target=target_m)
    xbar, ci_lo, ci_hi = confidence_interval_mean(tc, alpha=1 - ci_alpha)

    # Metrics
    cols = st.columns(5)
    cols[0].markdown(metric_html("Mean cost", f"${np.mean(tc)/1e6:.2f}M"), unsafe_allow_html=True)
    cols[1].markdown(metric_html("Std Dev", f"${np.std(tc)/1e6:.2f}M", color=GOLD), unsafe_allow_html=True)
    cols[2].markdown(metric_html("P(≤ Target)", f"{ss['P(Y ≤ Target)']:.1%}", color=TEAL), unsafe_allow_html=True)
    cols[3].markdown(metric_html("P(> Target) [VaR]", f"{ss['P(Y > Target)']:.1%}", color=RED), unsafe_allow_html=True)
    cols[4].markdown(metric_html("90th Percentile", f"${np.percentile(tc,90)/1e6:.2f}M"), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="var-highlight">
    🎯 <b>VaR Analysis:</b> Ngưỡng Target = <b>${target_m/1e6:.0f}M</b> →
    P(cost ≤ Target) = <b>{ss['P(Y ≤ Target)']:.2%}</b>,
    P(cost &gt; Target) = <b style="color:{RED}">{ss['P(Y > Target)']:.2%}</b>.
    Khoảng tin cậy {ci_alpha:.0%} cho mean thực:
    [<b>${ci_lo/1e6:.2f}M — ${ci_hi/1e6:.2f}M</b>]
    </div>
    """, unsafe_allow_html=True)

    fig = plot_histogram(tc, "Phân phối tổng chi phí công ty (12 tháng)", "Total Company Cost ($)", target=target_m, color=BLUE)
    show_fig(fig)

    # Monthly trend
    fig2, ax2 = new_fig(figsize=(11, 3.2))
    months = [f"T{i+1}" for i in range(12)]
    p10 = np.percentile(mc, 10, axis=0)
    p50 = np.percentile(mc, 50, axis=0)
    p90 = np.percentile(mc, 90, axis=0)
    ax2.fill_between(months, p10, p90, alpha=0.2, color=BLUE, label="P10–P90 band")
    ax2.plot(months, p50, color=BLUE, linewidth=1.8, label="Median (P50)")
    ax2.plot(months, np.mean(mc, axis=0), color=GOLD, linewidth=1.4, linestyle="--", label="Mean")
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
    style_ax(ax2, "Chi phí hàng tháng — Dải P10/P50/P90", "Tháng", "Company Cost ($)")
    show_fig(fig2)

    st.markdown('<div class="section-hdr"><span class="sec-num">④</span> THỐNG KÊ CHI TIẾT</div>', unsafe_allow_html=True)
    stats_df = pd.DataFrame({"Chỉ số": list(ss.keys()), "Giá trị": [
        f"${v/1e6:.3f}M" if isinstance(v, float) and abs(v) > 1e4 else f"{v:.4f}" for v in ss.values()
    ]})
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with st.expander("💡 Diễn giải kết quả"):
        st.markdown(f"""
        <div class="interpret-box">
        Sau {n_sims:,} lần mô phỏng:<br>
        • Mean ≈ <b>${np.mean(tc)/1e6:.2f}M</b> — gần với kết quả base case (chạy 1 kịch bản duy nhất).<br>
        • Tuy nhiên, phân phối trải từ <b>${np.min(tc)/1e6:.1f}M</b> đến <b>${np.max(tc)/1e6:.1f}M</b> — 
          base case không cho thấy điều này.<br>
        • P(cost > ${target_m/1e6:.0f}M) = <b>{ss['P(Y > Target)']:.2%}</b>. 
          Nếu CFO muốn giảm rủi ro xuống ≤ 2%, có thể tăng đóng góp NV hoặc điều chỉnh chính sách bảo hiểm.<br>
        • Khoảng tin cậy {ci_alpha:.0%} cho mean: [${ci_lo/1e6:.3f}M — ${ci_hi/1e6:.3f}M]. 
          Với {n_sims:,} replications, sai số rất nhỏ.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: AIRLINE OVERBOOKING
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "airline":
    st.markdown('<div class="section-hdr"><span class="sec-num">✈️</span> AIRLINE OVERBOOKING (Fig 12-18)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Bài toán:</b> Máy bay có 19 ghế. Nhu cầu đặt chỗ theo phân phối rời rạc (<code>Discrete</code>).
    Xác suất no-show 10% → số khách show-up theo <code>Binomial</code>.
    Tìm số reservations R* tối ưu hoá lợi nhuận kỳ vọng.
    </div>
    """, unsafe_allow_html=True)

    # Default demand distribution from sách (Table 12-6)
    DEFAULT_DEMAND = {14:0.03,15:0.05,16:0.07,17:0.09,18:0.11,
                      19:0.15,20:0.18,21:0.14,22:0.08,23:0.05,24:0.03,25:0.02}

    with st.sidebar:
        st.markdown('<div class="section-hdr"><span class="sec-num">③</span> THAM SỐ</div>', unsafe_allow_html=True)
        seats        = st.number_input("Số ghế (capacity)", value=19, min_value=5, max_value=200)
        ticket_price = st.number_input("Giá vé ($)", value=150, step=10)
        bump_cost    = st.number_input("Chi phí bù đắp khách bị đẩy ($)", value=325, step=25)
        p_noshow     = st.slider("Xác suất no-show", 0.0, 0.5, 0.10, 0.01, format="%.2f")
        res_min      = st.number_input("R tối thiểu thử", value=int(seats), min_value=int(seats))
        res_max      = st.number_input("R tối đa thử", value=int(seats)+6, min_value=int(seats))
        run_btn      = st.button("▶ Chạy mô phỏng", use_container_width=True)

    with st.expander("📊 Phân phối nhu cầu (từ sách)"):
        dd = pd.DataFrame({"Demand": list(DEFAULT_DEMAND.keys()),
                            "Probability": list(DEFAULT_DEMAND.values())})
        fig_d, ax_d = new_fig(figsize=(8, 2.8))
        ax_d.bar(dd["Demand"], dd["Probability"], color=PURPLE, alpha=0.8, width=0.7)
        style_ax(ax_d, "Discrete Demand Distribution (Discrete)", "Demand", "Probability")
        show_fig(fig_d)

    if run_btn or "airline_data" not in st.session_state:
        with st.spinner("Quét tất cả giá trị R…"):
            results = mc_airline_overbooking(
                seats, ticket_price, bump_cost, p_noshow,
                list(DEFAULT_DEMAND.keys()), list(DEFAULT_DEMAND.values()),
                res_min, res_max, n_sims, seed=seed)
            st.session_state.airline_data = results

    results = st.session_state.airline_data
    summary_rows = []
    for R, profits in results.items():
        summary_rows.append({
            "Reservations (R)": R,
            "E[Profit]": np.mean(profits),
            "Std Dev": np.std(profits, ddof=1),
            "P10": np.percentile(profits, 10),
            "P90": np.percentile(profits, 90),
            "Min": np.min(profits),
            "Max": np.max(profits),
            "P(Loss)": np.mean(profits < 0),
        })
    summ_df = pd.DataFrame(summary_rows)
    best_R = int(summ_df.loc[summ_df["E[Profit]"].idxmax(), "Reservations (R)"])

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_html("Optimal R*", str(best_R), "max E[Profit]", TEAL), unsafe_allow_html=True)
    c2.markdown(metric_html("E[Profit] at R*", f"${np.mean(results[best_R]):,.0f}"), unsafe_allow_html=True)
    c3.markdown(metric_html("Seats available", str(seats), f"no-show rate = {p_noshow:.0%}", GOLD), unsafe_allow_html=True)

    # Comparison chart
    fig3, ax3 = new_fig(figsize=(10, 3.8))
    Rs = [r["Reservations (R)"] for r in summary_rows]
    means = [r["E[Profit]"] for r in summary_rows]
    p10s  = [r["P10"]  for r in summary_rows]
    p90s  = [r["P90"]  for r in summary_rows]
    ax3.fill_between(Rs, p10s, p90s, alpha=0.2, color=BLUE, label="P10–P90")
    ax3.plot(Rs, means, color=BLUE, linewidth=2, marker="o", markersize=5, label="E[Profit]")
    ax3.axvline(best_R, color=TEAL, linewidth=1.5, linestyle="--", label=f"R*={best_R}")
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
    style_ax(ax3, "E[Profit] theo số Reservations", "Reservations R", "Expected Profit ($)")
    show_fig(fig3)

    # Optimal distribution
    fig4 = plot_histogram(results[best_R], f"Phân phối lợi nhuận tại R*={best_R}",
                          "Profit ($)", target=0, color=PURPLE)
    show_fig(fig4)

    st.markdown(f"""
    <div class="interpret-box">
    Kết quả parameterized simulation với R từ {int(res_min)} đến {int(res_max)}:<br>
    • R* = <b>{best_R}</b> cho E[Profit] = <b>${np.mean(results[best_R]):,.0f}</b><br>
    • P(Loss) tại R* = <b>{np.mean(results[best_R] < 0):.1%}</b><br>
    • Nếu R = {seats} (không overbooking): E[Profit] = ${np.mean(results[int(seats)]):,.0f} — thấp hơn do bỏ lỡ doanh thu từ no-show.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Bảng tổng hợp tất cả giá trị R"):
        fmt_df = summ_df.copy()
        for c in ["E[Profit]","Std Dev","P10","P90","Min","Max"]:
            fmt_df[c] = fmt_df[c].apply(lambda x: f"${x:,.0f}")
        fmt_df["P(Loss)"] = fmt_df["P(Loss)"].apply(lambda x: f"{x:.1%}")
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: INVENTORY CONTROL
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "inventory":
    st.markdown('<div class="section-hdr"><span class="sec-num">📦</span> INVENTORY CONTROL — CHÍNH SÁCH (s, Q) (Fig 12-23)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Bài toán:</b> Khi tồn kho ≤ s (reorder point), đặt thêm Q đơn vị.
    Nhu cầu hàng ngày và lead time đều là biến ngẫu nhiên rời rạc (<code>Discrete</code>).
    Tìm (s, Q) tối thiểu chi phí với ràng buộc Service Level ≥ 98%.
    </div>
    """, unsafe_allow_html=True)

    # Default distributions from Fig 12-23
    DEMAND_DIST = {0:0.01,1:0.02,2:0.04,3:0.06,4:0.09,5:0.14,
                   6:0.18,7:0.22,8:0.16,9:0.06,10:0.02}
    LEAD_DIST   = {3:0.20, 4:0.60, 5:0.20}

    with st.sidebar:
        st.markdown('<div class="section-hdr"><span class="sec-num">③</span> THAM SỐ</div>', unsafe_allow_html=True)
        init_inv     = st.number_input("Tồn kho ban đầu", value=50, step=5)
        reorder_pt   = st.slider("Reorder Point (s)", 1, 100, 28)
        order_qty    = st.slider("Order Quantity (Q)", 5, 200, 50, step=5)
        n_days       = st.number_input("Số ngày mô phỏng", value=30, min_value=10, max_value=365)
        holding_cost = st.number_input("Holding cost/đvị/ngày ($)", value=0.10, step=0.05, format="%.2f")
        order_cost   = st.number_input("Ordering cost/đơn hàng ($)", value=50.0, step=5.0)
        shortage_cost= st.number_input("Shortage cost/đvị ($)", value=5.0, step=0.5)
        do_scan      = st.checkbox("Quét nhiều (s, Q) — Parameterized Simulation")
        if do_scan:
            s_vals_str = st.text_input("Các reorder point", "20,25,28,30,35,40")
            q_vals_str = st.text_input("Các order qty", "40,50,60")
        run_btn = st.button("▶ Chạy mô phỏng", use_container_width=True)

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown("**Demand distribution (Discrete)**")
        dd_df = pd.DataFrame({"Demand": list(DEMAND_DIST.keys()), "Prob": list(DEMAND_DIST.values())})
        fig_dd, ax_dd = new_fig(figsize=(5, 2.5))
        ax_dd.bar(dd_df["Demand"], dd_df["Prob"], color=TEAL, alpha=0.8, width=0.7)
        style_ax(ax_dd, "Nhu cầu hàng ngày", "Đơn vị", "Xác suất")
        show_fig(fig_dd)
    with col_dist2:
        st.markdown("**Lead time distribution (Discrete)**")
        lt_df = pd.DataFrame({"Lead": list(LEAD_DIST.keys()), "Prob": list(LEAD_DIST.values())})
        fig_lt, ax_lt = new_fig(figsize=(5, 2.5))
        ax_lt.bar(lt_df["Lead"], lt_df["Prob"], color=GOLD, alpha=0.8, width=0.5)
        style_ax(ax_lt, "Lead time (ngày)", "Ngày", "Xác suất")
        show_fig(fig_lt)

    if run_btn or "inv_data" not in st.session_state:
        with st.spinner("Đang mô phỏng tồn kho…"):
            sl, ai, tc = mc_inventory(
                init_inv, reorder_pt, order_qty, n_days, n_sims,
                holding_cost, order_cost, shortage_cost,
                list(DEMAND_DIST.keys()), list(DEMAND_DIST.values()),
                list(LEAD_DIST.keys()), list(LEAD_DIST.values()), seed=seed)
            st.session_state.inv_data = (sl, ai, tc)

            scan_results = None
            if do_scan:
                s_vals = [int(x) for x in s_vals_str.split(",")]
                q_vals = [int(x) for x in q_vals_str.split(",")]
                rows = []
                for sv in s_vals:
                    for qv in q_vals:
                        sl2, ai2, tc2 = mc_inventory(
                            init_inv, sv, qv, n_days, n_sims,
                            holding_cost, order_cost, shortage_cost,
                            list(DEMAND_DIST.keys()), list(DEMAND_DIST.values()),
                            list(LEAD_DIST.keys()), list(LEAD_DIST.values()), seed=seed)
                        rows.append({"s": sv, "Q": qv,
                                     "E[SL]": np.mean(sl2),
                                     "E[Avg Inv]": np.mean(ai2),
                                     "E[Total Cost]": np.mean(tc2),
                                     "P(SL≥98%)": np.mean(sl2 >= 0.98)})
                scan_results = pd.DataFrame(rows)
            st.session_state.inv_scan = scan_results

    sl, ai, tc = st.session_state.inv_data
    p_sl98 = np.mean(sl >= 0.98)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_html("Avg Service Level", f"{np.mean(sl):.1%}", color=TEAL), unsafe_allow_html=True)
    c2.markdown(metric_html("P(SL ≥ 98%)", f"{p_sl98:.1%}", color=TEAL if p_sl98 > 0.8 else RED), unsafe_allow_html=True)
    c3.markdown(metric_html("Avg Inventory", f"{np.mean(ai):.1f} units"), unsafe_allow_html=True)
    c4.markdown(metric_html("Avg Total Cost", f"${np.mean(tc):,.0f}"), unsafe_allow_html=True)

    col_h, col_sl = st.columns(2)
    with col_h:
        fig5 = plot_histogram(tc, "Phân phối tổng chi phí tồn kho", "Total Cost ($)", color=TEAL)
        show_fig(fig5)
    with col_sl:
        fig6, ax6 = new_fig(figsize=(5.5, 3.8))
        ax6.hist(sl * 100, bins=30, color=GOLD, alpha=0.8, density=True, edgecolor=DARK, linewidth=0.3)
        ax6.axvline(98, color=RED, linewidth=1.5, linestyle="--", label="98% target")
        ax6.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
        style_ax(ax6, "Phân phối Service Level", "Service Level (%)", "Density")
        show_fig(fig6)

    sl_color = TEAL if np.mean(sl) >= 0.98 else RED
    st.markdown(f"""
    <div class="interpret-box">
    Với s={reorder_pt}, Q={order_qty}:<br>
    • E[Service Level] = <b style="color:{sl_color}">{np.mean(sl):.1%}</b>
    {"✅ đạt mục tiêu ≥ 98%" if np.mean(sl) >= 0.98 else "❌ chưa đạt mục tiêu 98% — cần tăng s"}<br>
    • E[Total Cost] = <b>${np.mean(tc):,.0f}</b> (holding + ordering + shortage)<br>
    • Dùng Parameterized Scan để tìm (s, Q) tối ưu cân bằng giữa cost và service level.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("inv_scan") is not None:
        scan_df = st.session_state.inv_scan
        st.markdown('<div class="section-hdr"><span class="sec-num">④</span> PARAMETERIZED SCAN</div>', unsafe_allow_html=True)

        fig7, ax7 = new_fig(figsize=(10, 3.8))
        for qv in scan_df["Q"].unique():
            sub = scan_df[scan_df["Q"] == qv].sort_values("s")
            ax7.plot(sub["s"], sub["E[Total Cost]"], marker="o", markersize=5,
                     linewidth=1.5, label=f"Q={qv}")
        ax7.axhline(scan_df["E[Total Cost]"].min(), color=RED, linewidth=1, linestyle=":")
        ax7.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
        style_ax(ax7, "E[Total Cost] theo (s, Q)", "Reorder Point s", "E[Total Cost] ($)")
        show_fig(fig7)

        fmt_scan = scan_df.copy()
        fmt_scan["E[SL]"] = fmt_scan["E[SL]"].apply(lambda x: f"{x:.1%}")
        fmt_scan["E[Avg Inv]"] = fmt_scan["E[Avg Inv]"].apply(lambda x: f"{x:.1f}")
        fmt_scan["E[Total Cost]"] = fmt_scan["E[Total Cost]"].apply(lambda x: f"${x:,.0f}")
        fmt_scan["P(SL≥98%)"] = fmt_scan["P(SL≥98%)"].apply(lambda x: f"{x:.1%}")
        st.dataframe(fmt_scan, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: PROJECT SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "project":
    st.markdown('<div class="section-hdr"><span class="sec-num">📂</span> PROJECT SELECTION (Fig 12-39)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Bài toán:</b> Chọn tập hợp dự án trong ngân sách $2M để tối đa hoá lợi nhuận kỳ vọng.
    Revenue mỗi dự án theo <code>Triangular</code> (min/most-likely/max) nếu thành công,
    thành công theo <code>Binomial</code>.
    </div>
    """, unsafe_allow_html=True)

    DEFAULT_PROJECTS = pd.DataFrame({
        "Project": [1,2,3,4,5,6,7,8],
        "Investment": [250,650,250,500,700,30,350,70],
        "P_Success":  [0.9,0.7,0.6,0.4,0.8,0.6,0.7,0.9],
        "Rev_Min":    [600,1250,500,1600,1150,150,750,220],
        "Rev_MLikely":[750,1500,600,1800,1200,180,900,250],
        "Rev_Max":    [900,1600,750,1900,1400,250,1000,320],
        "Selected":   [True,True,False,True,False,True,True,True],
    })

    with st.sidebar:
        st.markdown('<div class="section-hdr"><span class="sec-num">③</span> THAM SỐ</div>', unsafe_allow_html=True)
        budget  = st.number_input("Ngân sách tổng ($K)", value=2000, step=100)
        run_btn = st.button("▶ Chạy mô phỏng", use_container_width=True)

    st.markdown("**Chỉnh sửa danh mục dự án:**")
    edited = st.data_editor(
        DEFAULT_PROJECTS, use_container_width=True,
        column_config={
            "Selected":   st.column_config.CheckboxColumn("Chọn?"),
            "P_Success":  st.column_config.NumberColumn("P(Success)", format="%.2f"),
            "Investment": st.column_config.NumberColumn("Investment ($K)"),
            "Rev_Min":    st.column_config.NumberColumn("Rev Min ($K)"),
            "Rev_MLikely":st.column_config.NumberColumn("Rev Most-Likely ($K)"),
            "Rev_Max":    st.column_config.NumberColumn("Rev Max ($K)"),
        },
        hide_index=True,
    )
    sel_df = edited[edited["Selected"]]
    total_inv = sel_df["Investment"].sum()

    c1, c2 = st.columns(2)
    c1.markdown(metric_html("Tổng đầu tư", f"${total_inv:,}K",
                            f"Budget: ${budget:,}K", RED if total_inv > budget else TEAL), unsafe_allow_html=True)
    c2.markdown(metric_html("Số dự án chọn", str(len(sel_df))), unsafe_allow_html=True)

    if total_inv > budget:
        st.markdown(f'<div class="err-box">⚠️ Tổng đầu tư ${total_inv}K vượt ngân sách ${budget}K!</div>', unsafe_allow_html=True)

    if run_btn or "proj_data" not in st.session_state:
        with st.spinner("Đang mô phỏng danh mục dự án…"):
            profits = mc_project_selection(edited, n_sims, seed=seed)
            st.session_state.proj_data = profits

    profits = st.session_state.proj_data
    ss = summary_stats(profits, target=0)
    xbar, ci_lo, ci_hi = confidence_interval_mean(profits, alpha=1 - ci_alpha)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_html("E[Total Profit]", f"${np.mean(profits):,.0f}K"), unsafe_allow_html=True)
    c2.markdown(metric_html("P(Loss)", f"{ss['P(Y > Target)']:.1%}",
                            color=RED if ss['P(Y > Target)'] > 0.2 else TEAL), unsafe_allow_html=True)
    c3.markdown(metric_html("90th Percentile", f"${np.percentile(profits,90):,.0f}K", color=GOLD), unsafe_allow_html=True)
    c4.markdown(metric_html("Std Dev", f"${np.std(profits,ddof=1):,.0f}K", color=GRAY), unsafe_allow_html=True)

    fig8 = plot_histogram(profits, "Phân phối tổng lợi nhuận danh mục", "Total Profit ($K)", target=0, color=GOLD)
    show_fig(fig8)

    # Per-project contribution
    if not sel_df.empty:
        fig9, ax9 = new_fig(figsize=(10, 3))
        proj_labels = [f"P{int(r['Project'])}" for _, r in sel_df.iterrows()]
        e_profits = [(r["Rev_MLikely"] - r["Investment"]) * r["P_Success"] - r["Investment"] * (1 - r["P_Success"])
                     for _, r in sel_df.iterrows()]
        colors = [TEAL if v > 0 else RED for v in e_profits]
        ax9.bar(proj_labels, e_profits, color=colors, alpha=0.8)
        ax9.axhline(0, color=WHITE, linewidth=0.8, linestyle="--")
        style_ax(ax9, "E[Profit] ước tính theo từng dự án ($K)", "Dự án", "E[Profit] ($K)")
        show_fig(fig9)

    st.markdown(f"""
    <div class="interpret-box">
    Với {len(sel_df)} dự án được chọn (đầu tư ${total_inv}K):<br>
    • E[Tổng lợi nhuận] = <b>${np.mean(profits):,.0f}K</b><br>
    • P(thua lỗ) = <b>{ss['P(Y > Target)']:.1%}</b><br>
    • Khoảng tin cậy {ci_alpha:.0%}: [${ci_lo:,.0f}K — ${ci_hi:,.0f}K]<br>
    • Tick/bỏ tick các dự án và nhấn "Chạy mô phỏng" để xem tác động tức thì.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "portfolio":
    st.markdown('<div class="section-hdr"><span class="sec-num">📈</span> PORTFOLIO OPTIMIZATION — EFFICIENT FRONTIER (Fig 12-43)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Bài toán McDaniel Group:</b> Phân bổ $1B vào 5 loại nhà máy điện.
    Return mỗi loại có tương quan nhau (<code>Correlated Returns</code>).
    Tìm danh mục tối ưu trên <b>efficient frontier</b>: max E[Return] với σ ≤ σ_max.
    </div>
    """, unsafe_allow_html=True)

    ASSETS = ["Gas","Coal","Oil","Nuclear","Wind"]
    MEANS  = [0.16,0.12,0.10,0.09,0.08]
    STDS   = [0.12,0.06,0.04,0.03,0.01]
    CORR   = np.array([
        [1.00,-0.49,-0.31, 0.16, 0.12],
        [-0.49,1.00,-0.41, 0.11, 0.07],
        [-0.31,-0.41,1.00, 0.13, 0.09],
        [0.16, 0.11, 0.13, 1.00, 0.04],
        [0.12, 0.07, 0.09, 0.04, 1.00],
    ])

    with st.sidebar:
        st.markdown('<div class="section-hdr"><span class="sec-num">③</span> PHÂN BỔ ($M)</div>', unsafe_allow_html=True)
        allocs = [st.slider(a, 0, 1000, 200, 50) for a in ASSETS]
        total_alloc = sum(allocs)
        st.markdown(f"**Tổng đầu tư: ${total_alloc}M** {'✅' if total_alloc==1000 else '⚠️ (mục tiêu $1,000M)'}")
        run_btn = st.button("▶ Chạy mô phỏng", use_container_width=True)

    weights = np.array(allocs) / 1000.0

    if run_btn or "port_data" not in st.session_state:
        with st.spinner("Đang mô phỏng portfolio…"):
            asset_rets, port_rets = mc_portfolio(MEANS, STDS, CORR, weights, n_sims, seed=seed)
            # Efficient frontier: random weights
            rng_ef = np.random.default_rng(99)
            ef_means, ef_stds = [], []
            for _ in range(3000):
                w = rng_ef.dirichlet(np.ones(5))
                z2 = rng_ef.standard_normal((300, 5))
                cov = np.outer(STDS, STDS) * CORR
                try:
                    L2 = np.linalg.cholesky(cov)
                except:
                    cov += np.eye(5)*1e-8; L2 = np.linalg.cholesky(cov)
                r2 = (z2 @ L2.T + np.array(MEANS)) @ w
                ef_means.append(np.mean(r2)); ef_stds.append(np.std(r2))
            st.session_state.port_data = (asset_rets, port_rets, ef_means, ef_stds)

    asset_rets, port_rets, ef_means, ef_stds = st.session_state.port_data
    port_mean = np.mean(port_rets); port_std = np.std(port_rets)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_html("E[Return]", f"{port_mean:.2%}"), unsafe_allow_html=True)
    c2.markdown(metric_html("Risk (σ)", f"{port_std:.2%}", color=GOLD), unsafe_allow_html=True)
    c3.markdown(metric_html("Return/Risk", f"{port_mean/port_std:.2f}" if port_std>0 else "—", color=PURPLE), unsafe_allow_html=True)
    c4.markdown(metric_html("P(Return < 0)", f"{np.mean(port_rets<0):.1%}", color=RED), unsafe_allow_html=True)

    col_ef, col_hist = st.columns(2)
    with col_ef:
        fig10, ax10 = new_fig(figsize=(5.5, 4.2))
        ax10.scatter(ef_stds, [m*100 for m in ef_means], s=4, alpha=0.3, color=GRAY, label="Random portfolios")
        ax10.scatter([port_std*100], [port_mean*100], s=120, color=GOLD, zorder=5, marker="*", label="Current portfolio")
        ax10.set_xlabel("Risk σ (%)", color=GRAY, fontsize=8)
        style_ax(ax10, "Efficient Frontier", "σ (%)", "E[Return] (%)")
        ax10.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
        show_fig(fig10)

    with col_hist:
        fig11 = plot_histogram(port_rets * 100, "Phân phối Return danh mục", "Return (%)", target=0, color=PURPLE)
        show_fig(fig11)

    # Asset allocation bar
    fig12, ax12 = new_fig(figsize=(10, 2.8))
    colors_a = [BLUE, TEAL, GOLD, RED, PURPLE]
    bars = ax12.bar(ASSETS, [w*100 for w in weights], color=colors_a, alpha=0.85, width=0.5)
    for bar, pct in zip(bars, weights):
        ax12.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{pct:.0%}",
                  ha="center", va="bottom", color=WHITE, fontsize=9)
    style_ax(ax12, "Phân bổ danh mục (%)", "Loại nhà máy", "Tỷ trọng (%)")
    show_fig(fig12)

    # Correlation heatmap
    with st.expander("📊 Ma trận tương quan (Correlation Matrix)"):
        fig13, ax13 = new_fig(figsize=(5, 4))
        cmap = matplotlib.colormaps.get_cmap("RdBu_r")
        im = ax13.imshow(CORR, cmap=cmap, vmin=-1, vmax=1)
        ax13.set_xticks(range(5)); ax13.set_yticks(range(5))
        ax13.set_xticklabels(ASSETS, color=WHITE, fontsize=8)
        ax13.set_yticklabels(ASSETS, color=WHITE, fontsize=8)
        for i in range(5):
            for j in range(5):
                ax13.text(j, i, f"{CORR[i,j]:.2f}", ha="center", va="center",
                          color=WHITE if abs(CORR[i,j]) > 0.5 else GRAY, fontsize=8)
        plt.colorbar(im, ax=ax13, fraction=0.046)
        style_ax(ax13, "Correlation Matrix (Return các loại nhà máy)", "", "")
        show_fig(fig13)

    st.markdown(f"""
    <div class="interpret-box">
    <b>Efficient Frontier:</b> Mỗi điểm xám = một danh mục ngẫu nhiên. Ngôi sao vàng = danh mục hiện tại của bạn.<br>
    • Các điểm nằm trên đường biên trên trái là "efficient" — không thể cải thiện return mà không tăng risk.<br>
    • Correlation âm giữa Gas và Coal (-0.49) tạo hiệu ứng đa dạng hoá giảm risk.<br>
    • Thử điều chỉnh phân bổ trong sidebar để di chuyển ngôi sao vàng trên frontier.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: RNG PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "rng":
    st.markdown('<div class="section-hdr"><span class="sec-num">🎰</span> RNG PLAYGROUND — KHÁM PHÁ PHÂN PHỐI</div>', unsafe_allow_html=True)

    dist = st.selectbox("Chọn phân phối", [
        "Normal (Normal)",
        "Uniform (Uniform)",
        "Triangular (Triangular)",
        "Discrete (Discrete)",
        "Binomial (Binomial)",
        "Lognormal (Lognormal)",
    ])

    col_p, col_c = st.columns([1, 2])
    rng_play = np.random.default_rng(int(seed))
    N = 10000

    with col_p:
        st.markdown("**Tham số**")
        if "Normal" in dist:
            mu    = st.number_input("Mean (μ)", value=0.0)
            sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01)
            samples = rng_play.normal(mu, sigma, N)
            formula = f"Normal({mu}, {sigma})"
        elif "Uniform" in dist:
            lo = st.number_input("Min (a)", value=0.0)
            hi = st.number_input("Max (b)", value=1.0)
            if hi <= lo: hi = lo + 1
            samples = rng_play.uniform(lo, hi, N)
            formula = f"Uniform({lo}, {hi})"
        elif "Triangular" in dist:
            lo  = st.number_input("Min (a)", value=0.0)
            mid = st.number_input("Most Likely (c)", value=5.0)
            hi  = st.number_input("Max (b)", value=10.0)
            if not (lo <= mid <= hi): mid = (lo+hi)/2
            samples = rng_play.triangular(lo, mid, hi, N)
            formula = f"Triangular({lo}, {mid}, {hi})"
        elif "Discrete" in dist:
            vals_s  = st.text_input("Giá trị (cách bởi dấu phẩy)", "10,20,30")
            probs_s = st.text_input("Xác suất (tổng = 1)", "0.3,0.5,0.2")
            try:
                vals  = [float(x) for x in vals_s.split(",")]
                probs = [float(x) for x in probs_s.split(",")]
                probs = [p/sum(probs) for p in probs]
                samples = rng_play.choice(vals, p=probs, size=N)
                formula = f"Discrete({{{vals_s}}}, {{{probs_s}}})"
            except:
                st.error("Lỗi nhập liệu"); samples = np.zeros(N); formula = ""
        elif "Binomial" in dist:
            n_tri = st.number_input("n (số lần thử)", value=20, min_value=1)
            p_suc = st.slider("p (xác suất thành công)", 0.0, 1.0, 0.3, 0.01)
            samples = rng_play.binomial(int(n_tri), p_suc, N)
            formula = f"Binomial({int(n_tri)}, {p_suc})"
        else:  # Lognormal
            mu_ln    = st.number_input("μ (log-scale)", value=0.0)
            sigma_ln = st.number_input("σ (log-scale)", value=0.5, min_value=0.01)
            samples  = rng_play.lognormal(mu_ln, sigma_ln, N)
            formula  = f"Lognormal({mu_ln}, {sigma_ln})"

        st.markdown(f'<div class="formula-box">{formula}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"**Mean:** {np.mean(samples):.4f}")
        st.markdown(f"**Std Dev:** {np.std(samples):.4f}")
        st.markdown(f"**Min / Max:** {np.min(samples):.3f} / {np.max(samples):.3f}")
        st.markdown(f"**P10 / P50 / P90:** {np.percentile(samples,10):.3f} / {np.percentile(samples,50):.3f} / {np.percentile(samples,90):.3f}")

    with col_c:
        nbins = 20 if ("Discrete" in dist or "Binomial" in dist) else 60
        fig_p, axes_p = new_fig(2, 1, figsize=(7, 5.5), sharex=False)
        ax_top, ax_bot = axes_p

        # Histogram
        ax_top.hist(samples, bins=nbins, color=PURPLE, alpha=0.78, density=True,
                    edgecolor=DARK, linewidth=0.3)
        if "Discrete" not in dist and "Binomial" not in dist:
            mu2, sig2 = np.mean(samples), np.std(samples)
            xr = np.linspace(np.min(samples), np.max(samples), 400)
            ax_top.plot(xr, stats.norm.pdf(xr, mu2, sig2), color=GOLD, linewidth=1.5, linestyle="--")
        style_ax(ax_top, f"{dist} — Histogram ({N:,} mẫu)", "", "Density")

        # Boxplot
        ax_bot.boxplot(samples, vert=False, patch_artist=True,
                       boxprops=dict(facecolor=PURPLE, alpha=0.5, color=PURPLE),
                       medianprops=dict(color=GOLD, linewidth=2),
                       whiskerprops=dict(color=GRAY),
                       capprops=dict(color=GRAY),
                       flierprops=dict(marker=".", color=GRAY, markersize=2, alpha=0.5))
        style_ax(ax_bot, "Boxplot", "", "")

        fig_p.tight_layout(pad=1.5)
        show_fig(fig_p)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE: CUSTOM SIMULATION (Excel Upload)
# ══════════════════════════════════════════════════════════════════════════════
elif MODULE == "custom":
    st.markdown('<div class="section-hdr"><span class="sec-num">📁</span> CUSTOM SIMULATION — TỰ ĐỊNH NGHĨA BÀI TOÁN</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <b>Hướng dẫn:</b> Upload file Excel (.xlsx/.xls) chứa dữ liệu đầu vào. Sau đó định nghĩa
    <b>các biến ngẫu nhiên</b> (chọn cột & phân phối) và <b>công thức đầu ra</b> (biểu thức Python).
    App sẽ chạy Monte Carlo và phân tích phân phối kết quả.<br><br>
    <b>Ví dụ công thức đầu ra:</b> <code>revenue - cost</code>, <code>price * quantity * (1 - discount)</code>,
    <code>income - tax * income</code>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Upload ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr"><span class="sec-num">①</span> UPLOAD FILE EXCEL</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Chọn file Excel (.xlsx hoặc .xls)",
        type=["xlsx", "xls"],
        help="File cần có ít nhất 1 sheet với dữ liệu dạng bảng (hàng đầu = tên cột)."
    )

    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names

            col_sheet, col_preview = st.columns([1, 3])
            with col_sheet:
                chosen_sheet = st.selectbox("Chọn sheet", sheet_names)

            raw_df = pd.read_excel(uploaded_file, sheet_name=chosen_sheet)

            with col_preview:
                st.markdown(f"**Preview — {chosen_sheet}** ({raw_df.shape[0]} hàng × {raw_df.shape[1]} cột)")
                st.dataframe(raw_df.head(10), use_container_width=True, height=200)

            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                st.markdown('<div class="warn-box">⚠️ Sheet này không có cột số nào. Vui lòng chọn sheet khác.</div>', unsafe_allow_html=True)
            else:
                # ── Step 2: Define random variables ────────────────────────
                st.markdown('<div class="section-hdr"><span class="sec-num">②</span> ĐỊNH NGHĨA BIẾN NGẪU NHIÊN</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box">
                Chọn các cột sẽ được coi là <b>biến ngẫu nhiên</b>. Với mỗi biến, chọn phân phối phù hợp.
                Các tham số mặc định tự động tính từ dữ liệu (mean, std, min, max).
                </div>
                """, unsafe_allow_html=True)

                n_vars = st.number_input("Số biến ngẫu nhiên", min_value=1, max_value=min(8, len(numeric_cols)), value=min(2, len(numeric_cols)), step=1)

                DIST_OPTIONS = ["Normal", "Uniform", "Triangular", "Lognormal", "Pert"]
                rv_configs = []

                for vi in range(int(n_vars)):
                    with st.expander(f"🎲 Biến ngẫu nhiên #{vi+1}", expanded=(vi < 2)):
                        c_col, c_dist, c_p1, c_p2, c_p3 = st.columns([2, 2, 1.2, 1.2, 1.2])
                        with c_col:
                            chosen_col = st.selectbox(f"Cột dữ liệu", numeric_cols, key=f"rv_col_{vi}",
                                                       index=min(vi, len(numeric_cols)-1))
                        col_data = raw_df[chosen_col].dropna()
                        col_mean  = float(col_data.mean())
                        col_std   = max(float(col_data.std()), 1e-9)
                        col_min   = float(col_data.min())
                        col_max   = float(col_data.max())
                        col_mid   = float(col_data.median())

                        with c_dist:
                            dist_choice = st.selectbox(f"Phân phối", DIST_OPTIONS, key=f"rv_dist_{vi}")

                        if dist_choice == "Normal":
                            with c_p1: p1 = st.number_input("Mean (μ)", value=round(col_mean,4), key=f"p1_{vi}")
                            with c_p2: p2 = st.number_input("Std (σ)", value=round(col_std,4), min_value=1e-6, key=f"p2_{vi}")
                            p3 = None
                        elif dist_choice == "Uniform":
                            with c_p1: p1 = st.number_input("Min", value=round(col_min,4), key=f"p1_{vi}")
                            with c_p2: p2 = st.number_input("Max", value=round(col_max,4), key=f"p2_{vi}")
                            p3 = None
                        elif dist_choice == "Triangular":
                            with c_p1: p1 = st.number_input("Min", value=round(col_min,4), key=f"p1_{vi}")
                            with c_p2: p2 = st.number_input("Most Likely", value=round(col_mid,4), key=f"p2_{vi}")
                            with c_p3: p3 = st.number_input("Max", value=round(col_max,4), key=f"p3_{vi}")
                        elif dist_choice == "Lognormal":
                            ln_mu  = float(np.log(max(col_mean,1e-9)))
                            ln_sig = float(np.sqrt(np.log(1 + (col_std/max(col_mean,1e-9))**2)))
                            with c_p1: p1 = st.number_input("μ (log-scale)", value=round(ln_mu,4), key=f"p1_{vi}")
                            with c_p2: p2 = st.number_input("σ (log-scale)", value=round(max(ln_sig,0.01),4), min_value=1e-6, key=f"p2_{vi}")
                            p3 = None
                        else:  # PERT
                            with c_p1: p1 = st.number_input("Min", value=round(col_min,4), key=f"p1_{vi}")
                            with c_p2: p2 = st.number_input("Most Likely", value=round(col_mid,4), key=f"p2_{vi}")
                            with c_p3: p3 = st.number_input("Max", value=round(col_max,4), key=f"p3_{vi}")

                        # Tên biến (dùng trong formula)
                        var_name = st.text_input(
                            "Tên biến (dùng trong công thức)",
                            value=chosen_col.lower().replace(" ", "_").replace("-", "_")[:20],
                            key=f"rv_name_{vi}",
                            help="Tên này sẽ dùng trong công thức đầu ra. Chỉ dùng chữ, số, dấu _"
                        )
                        rv_configs.append({
                            "col": chosen_col, "dist": dist_choice,
                            "p1": p1, "p2": p2, "p3": p3,
                            "var_name": var_name.strip() or f"x{vi+1}"
                        })

                # ── Step 3: Output formula ──────────────────────────────────
                st.markdown('<div class="section-hdr"><span class="sec-num">③</span> CÔNG THỨC ĐẦU RA (OUTPUT)</div>', unsafe_allow_html=True)

                var_names_hint = ", ".join([f"<code>{rv['var_name']}</code>" for rv in rv_configs])
                st.markdown(f"""
                <div class="info-box">
                Nhập biểu thức Python sử dụng các biến: {var_names_hint}<br>
                Ví dụ: <code>{rv_configs[0]['var_name']} * 1.2 - {rv_configs[-1]['var_name']}</code><br>
                Hỗ trợ: <code>+, -, *, /, **, np.log(), np.exp(), np.sqrt(), abs(), max(), min()</code>
                </div>
                """, unsafe_allow_html=True)

                default_formula = " - ".join([rv["var_name"] for rv in rv_configs]) if len(rv_configs) >= 2 else rv_configs[0]["var_name"]
                output_formula = st.text_input(
                    "Công thức đầu ra (Y =)",
                    value=default_formula,
                    placeholder="e.g. revenue - cost * quantity"
                )
                output_label = st.text_input("Tên chỉ số đầu ra", value="Output (Y)", placeholder="e.g. Profit ($)")
                target_val   = st.number_input("VaR Target (giá trị ngưỡng)", value=0.0, step=1.0,
                                                help="P(Y ≤ Target) và P(Y > Target) sẽ được tính.")

                # ── Step 4: Run ─────────────────────────────────────────────
                st.markdown('<div class="section-hdr"><span class="sec-num">④</span> CHẠY MÔ PHỎNG</div>', unsafe_allow_html=True)

                run_custom = st.button("▶ Chạy Custom Simulation", use_container_width=True, type="primary")

                if run_custom:
                    rng_c = np.random.default_rng(int(seed))
                    results_Y = np.empty(n_sims)
                    errors = []

                    # Sample each RV n_sims times
                    sampled = {}
                    for rv in rv_configs:
                        d, p1, p2, p3, nm = rv["dist"], rv["p1"], rv["p2"], rv["p3"], rv["var_name"]
                        try:
                            if d == "Normal":
                                sampled[nm] = rng_c.normal(p1, max(p2,1e-9), n_sims)
                            elif d == "Uniform":
                                lo_u, hi_u = min(p1,p2), max(p1,p2)
                                if lo_u == hi_u: hi_u += 1e-9
                                sampled[nm] = rng_c.uniform(lo_u, hi_u, n_sims)
                            elif d == "Triangular":
                                lo_t = min(p1, p2, p3)
                                hi_t = max(p1, p2, p3)
                                mid_t = sorted([p1, p2, p3])[1]
                                if lo_t == hi_t: hi_t += 1e-9
                                mid_t = np.clip(mid_t, lo_t, hi_t)
                                sampled[nm] = rng_c.triangular(lo_t, mid_t, hi_t, n_sims)
                            elif d == "Lognormal":
                                sampled[nm] = rng_c.lognormal(p1, max(p2,1e-9), n_sims)
                            else:  # PERT → approx via Beta
                                lo_b = min(p1, p2, p3)
                                hi_b = max(p1, p2, p3)
                                mode_b = sorted([p1, p2, p3])[1]
                                if hi_b == lo_b: hi_b += 1e-9
                                mu_b  = (lo_b + 4*mode_b + hi_b) / 6.0
                                sig_b = (hi_b - lo_b) / 6.0
                                a_b = ((mu_b - lo_b) / (hi_b - lo_b)) * ((mu_b - lo_b) * (hi_b - mu_b) / sig_b**2 - 1)
                                b_b = a_b * (hi_b - mu_b) / (mu_b - lo_b)
                                a_b = max(a_b, 0.1); b_b = max(b_b, 0.1)
                                sampled[nm] = lo_b + (hi_b - lo_b) * rng_c.beta(a_b, b_b, n_sims)
                        except Exception as e:
                            errors.append(f"Lỗi lấy mẫu biến '{nm}': {e}")

                    if errors:
                        for err in errors:
                            st.markdown(f'<div class="err-box">❌ {err}</div>', unsafe_allow_html=True)
                    else:
                        # Evaluate formula
                        eval_errors = []
                        for i in range(n_sims):
                            local_vars = {nm: sampled[nm][i] for nm in sampled}
                            local_vars["np"] = np
                            try:
                                results_Y[i] = float(eval(output_formula, {"__builtins__": {}}, local_vars))
                            except Exception as e:
                                eval_errors.append(str(e))
                                results_Y[i] = np.nan

                        nan_count = np.sum(np.isnan(results_Y))
                        if nan_count > n_sims * 0.5:
                            st.markdown(f'<div class="err-box">❌ Công thức lỗi ({nan_count}/{n_sims} lần). Lỗi: {eval_errors[0] if eval_errors else "?"}<br>Kiểm tra lại tên biến và cú pháp.</div>', unsafe_allow_html=True)
                        else:
                            valid_Y = results_Y[~np.isnan(results_Y)]
                            st.session_state["custom_results"] = {
                                "Y": valid_Y,
                                "sampled": sampled,
                                "formula": output_formula,
                                "label": output_label,
                                "target": target_val,
                                "rv_configs": rv_configs,
                            }
                            if nan_count > 0:
                                st.markdown(f'<div class="warn-box">⚠️ {nan_count} lần lặp bị lỗi, bỏ qua. Dùng {len(valid_Y)} kết quả hợp lệ.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="success-box">✅ Mô phỏng hoàn tất — {len(valid_Y):,} iterations.</div>', unsafe_allow_html=True)

                # ── Step 5: Display results ─────────────────────────────────
                if "custom_results" in st.session_state:
                    cr = st.session_state["custom_results"]
                    Y       = cr["Y"]
                    lbl     = cr["label"]
                    tgt     = cr["target"]
                    sampled = cr["sampled"]

                    st.markdown('<div class="section-hdr"><span class="sec-num">⑤</span> KẾT QUẢ MÔ PHỎNG</div>', unsafe_allow_html=True)

                    ss_c = summary_stats(Y, target=tgt)
                    xbar_c, ci_lo_c, ci_hi_c = confidence_interval_mean(Y, alpha=1-ci_alpha)

                    # Metrics row
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.markdown(metric_html("Mean", f"{np.mean(Y):,.2f}"), unsafe_allow_html=True)
                    mc2.markdown(metric_html("Std Dev", f"{np.std(Y):,.2f}", color=GOLD), unsafe_allow_html=True)
                    mc3.markdown(metric_html("P10 / P90", f"{np.percentile(Y,10):,.2f} / {np.percentile(Y,90):,.2f}", color=TEAL), unsafe_allow_html=True)
                    mc4.markdown(metric_html("P(Y ≤ Target)", f"{ss_c.get('P(Y ≤ Target)', 0):.1%}", color=BLUE), unsafe_allow_html=True)
                    mc5.markdown(metric_html("P(Y > Target) [VaR]", f"{ss_c.get('P(Y > Target)', 0):.1%}", color=RED), unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="var-highlight">
                    🎯 <b>Y = {cr['formula']}</b> &nbsp;|&nbsp;
                    Target = <b>{tgt:,.2f}</b> →
                    P(Y ≤ Target) = <b>{ss_c.get('P(Y ≤ Target)', 0):.2%}</b>,
                    P(Y > Target) = <b style="color:{RED}">{ss_c.get('P(Y > Target)', 0):.2%}</b><br>
                    CI {ci_alpha:.0%} cho Mean thực: [<b>{ci_lo_c:,.2f} — {ci_hi_c:,.2f}</b>]
                    </div>
                    """, unsafe_allow_html=True)

                    # Histogram + CDF
                    fig_c = plot_histogram(Y, f"Phân phối đầu ra: {lbl}", lbl, target=tgt, color=TEAL)
                    show_fig(fig_c)

                    # Input distributions side by side
                    st.markdown('<div class="section-hdr"><span class="sec-num">⑥</span> PHÂN PHỐI CÁC BIẾN ĐẦU VÀO</div>', unsafe_allow_html=True)
                    rv_list = list(sampled.items())
                    n_rv = len(rv_list)
                    cols_rv = st.columns(min(n_rv, 3))
                    for idx, (nm, smp) in enumerate(rv_list):
                        with cols_rv[idx % 3]:
                            fig_rv, ax_rv = new_fig(figsize=(4, 2.8))
                            ax_rv.hist(smp, bins=40, color=PURPLE, alpha=0.75, density=True, edgecolor=DARK, linewidth=0.3)
                            mu_rv, sig_rv = np.mean(smp), np.std(smp)
                            xr_rv = np.linspace(mu_rv - 4*sig_rv, mu_rv + 4*sig_rv, 300)
                            ax_rv.plot(xr_rv, stats.norm.pdf(xr_rv, mu_rv, sig_rv), color=GOLD, linewidth=1.2, linestyle="--")
                            style_ax(ax_rv, f"{nm}", nm, "Density")
                            fig_rv.tight_layout(pad=1)
                            show_fig(fig_rv)

                    # Sensitivity (Tornado) via correlation
                    if len(sampled) >= 2:
                        st.markdown('<div class="section-hdr"><span class="sec-num">⑦</span> PHÂN TÍCH ĐỘ NHẠY (CORRELATION)</div>', unsafe_allow_html=True)
                        corr_vals = {}
                        for nm, smp in sampled.items():
                            if np.std(smp) > 0 and np.std(Y) > 0:
                                corr_vals[nm] = float(np.corrcoef(smp, Y)[0, 1])
                        sorted_corr = sorted(corr_vals.items(), key=lambda x: abs(x[1]), reverse=True)

                        fig_t, ax_t = new_fig(figsize=(8, max(2.5, len(sorted_corr)*0.55 + 1)))
                        ys_t = np.arange(len(sorted_corr))
                        colors_t = [TEAL if v >= 0 else RED for _, v in sorted_corr]
                        ax_t.barh(ys_t, [v for _, v in sorted_corr], color=colors_t, alpha=0.8, height=0.55)
                        ax_t.set_yticks(ys_t)
                        ax_t.set_yticklabels([n for n, _ in sorted_corr], color=WHITE, fontsize=9)
                        ax_t.axvline(0, color=GRAY, linewidth=0.8)
                        for i, (nm, v) in enumerate(sorted_corr):
                            ax_t.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.3f}",
                                      va="center", ha="left" if v >= 0 else "right", color=WHITE, fontsize=8)
                        style_ax(ax_t, f"Hệ số tương quan Pearson — Input vs {lbl}", "Correlation với Y", "")
                        show_fig(fig_t)

                    # Detail stats table
                    with st.expander("📋 Bảng thống kê chi tiết"):
                        ss_df = pd.DataFrame({
                            "Chỉ số": list(ss_c.keys()),
                            "Giá trị": [f"{v:,.4f}" if isinstance(v, float) else str(v) for v in ss_c.values()]
                        })
                        st.dataframe(ss_df, use_container_width=True, hide_index=True)

                    # Export results
                    with st.expander("💾 Xuất kết quả ra Excel"):
                        out_df = pd.DataFrame({"Y_output": Y})
                        for nm, smp in sampled.items():
                            out_df[f"input_{nm}"] = smp[:len(Y)]
                        out_buf = io.BytesIO()
                        with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                            out_df.to_excel(writer, index=False, sheet_name="SimResults")
                            ss_df.to_excel(writer, index=False, sheet_name="Summary")
                        st.download_button(
                            label="⬇️ Tải file kết quả (.xlsx)",
                            data=out_buf.getvalue(),
                            file_name="simulation_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        except Exception as e:
            st.markdown(f'<div class="err-box">❌ Không đọc được file Excel: {e}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
        📂 <b>Chưa có file nào được upload.</b><br><br>
        File Excel cần có dạng bảng đơn giản:<br>
        • Hàng đầu tiên = tên cột (header)<br>
        • Các hàng tiếp theo = dữ liệu số<br>
        • Ví dụ: cột <code>revenue</code>, <code>cost</code>, <code>quantity</code>, <code>price</code>…<br><br>
        Sau khi upload, bạn sẽ chọn phân phối xác suất cho từng biến và nhập công thức
        để tính chỉ số đầu ra (Y). App sẽ tự chạy Monte Carlo và hiện kết quả.
        </div>
        """, unsafe_allow_html=True)

        # Demo template download
        demo_data = pd.DataFrame({
            "revenue":  [1000, 1200, 950, 1100, 1300, 980, 1050, 1150, 1020, 1080],
            "cost":     [600,  620,  580, 610,  650,  590, 605,  625,  595,  615],
            "quantity": [500,  520,  480, 510,  540,  490, 505,  515,  495,  508],
            "discount": [0.05, 0.08, 0.03, 0.06, 0.10, 0.04, 0.05, 0.07, 0.04, 0.06],
        })
        demo_buf = io.BytesIO()
        with pd.ExcelWriter(demo_buf, engine="openpyxl") as writer:
            demo_data.to_excel(writer, index=False, sheet_name="Data")
        st.download_button(
            label="📥 Tải file mẫu (demo_data.xlsx)",
            data=demo_buf.getvalue(),
            file_name="demo_simulation_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ── End ────────────────────────────────────────────────────────────────────
