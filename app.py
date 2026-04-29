
import math
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 让同目录下的数据库模块可直接导入
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bunge_rhf_h_to_kr import (  # noqa: E402
    get_element,
    list_orbitals,
    list_supported_elements,
    orbital_radial,
    orbital_wavefunction,
    radial_probability_density,
    real_spherical_harmonic,
)

FONT_PATH = THIS_DIR / "NotoSansSC-Regular.ttf"

if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


ELEMENT_NAMES = {
    "H": "氢", "He": "氦", "Li": "锂", "Be": "铍", "B": "硼", "C": "碳", "N": "氮", "O": "氧", "F": "氟", "Ne": "氖",
    "Na": "钠", "Mg": "镁", "Al": "铝", "Si": "硅", "P": "磷", "S": "硫", "Cl": "氯", "Ar": "氩",
    "K": "钾", "Ca": "钙", "Sc": "钪", "Ti": "钛", "V": "钒", "Cr": "铬", "Mn": "锰", "Fe": "铁",
    "Co": "钴", "Ni": "镍", "Cu": "铜", "Zn": "锌", "Ga": "镓", "Ge": "锗", "As": "砷", "Se": "硒",
    "Br": "溴", "Kr": "氪",
}
SUBSHELL_ORDER = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1)]
SUBSHELL_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
L_LETTER = {0: "s", 1: "p", 2: "d", 3: "f"}
# m 的展开顺序（实球谐习惯）：l=0 只有 0；l=1 为 z,x,y；l=2 为 z²,xz,yz,x²-y²,xy
M_ORDER_REAL = {0: [0], 1: [0, 1, -1], 2: [0, 1, -1, 2, -2], 3: [0, 1, -1, 2, -2, 3, -3]}

# 颜色池，供多轨道比较自动分配
COLOR_POOL = [
    'black', 'darkred', 'saddlebrown', 'darkgreen', 'darkblue',
    'darkviolet', ]

SINGLE_PLOT_TYPES = [
    "径向函数图 R(r)-r",
    "径向密度函数图 R²(r)-r",
    "径向分布函数图 D(r)-r",
    "原子轨道角度分布图 |Y|-角度",
    "电子云角度分布图 |Y|²-角度",
    "原子轨道等值线图",
    "原子轨道网格图",
    "电子云黑点图",
]
MULTI_PLOT_TYPES = [
    "径向函数图 R(r)-r",
    "径向密度函数图 R²(r)-r",
    "径向分布函数图 D(r)-r",
]


# ==========================================================================
# 基础工具函数
# ==========================================================================
def _parse_configuration_string(config_str: str) -> Dict[str, int]:
    """
    解析数据库里的组态字符串（如 "[Ar] 3d5 4s1"、"1s2 2s2 2p4"）。
    返回 {子壳层标签: 电子数}，如 {"3d": 5, "4s": 1}。

    "[Ar]"、"[Ne]" 等惰性气体核心会被展开为对应满壳层。
    解析失败时返回空字典，调用方应当回退到 Aufbau。
    """
    import re as _re

    # 惰性气体核心展开
    NOBLE_CORES = {
        "He": {"1s": 2},
        "Ne": {"1s": 2, "2s": 2, "2p": 6},
        "Ar": {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6},
        "Kr": {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 10, "4s": 2, "4p": 6},
    }

    out: Dict[str, int] = {}
    s = config_str.strip()

    # 先处理 [核心]
    m = _re.match(r"\[(He|Ne|Ar|Kr)\]\s*(.*)", s)
    if m:
        core = m.group(1)
        if core not in NOBLE_CORES:
            return {}
        out.update(NOBLE_CORES[core])
        s = m.group(2)

    # 再解析剩余的 nlX 项
    for tok in s.split():
        m2 = _re.match(r"^(\d)([spdf])(\d+)$", tok)
        if not m2:
            return {}
        n = int(m2.group(1))
        l_letter = m2.group(2)
        count = int(m2.group(3))
        label = f"{n}{l_letter}"
        out[label] = out.get(label, 0) + count
    return out


def build_ground_state_config(z: int, symbol: str = None) -> Dict[str, int]:
    """
    构造基态电子组态。

    优先从数据库的 configuration 字符串解析（这能正确处理 Cr=[Ar]3d5 4s1、
    Cu=[Ar]3d10 4s1 等反常组态）；解析不到时回退到简单 Aufbau 顺序填充。
    """
    if symbol is not None:
        try:
            elem = get_element(symbol)
            parsed = _parse_configuration_string(elem.configuration)
            if parsed and sum(parsed.values()) == z:
                return parsed
        except (KeyError, AttributeError):
            pass

    # 回退：按 SUBSHELL_ORDER 的简单 Aufbau
    remain = z
    config = {}
    for n, l in SUBSHELL_ORDER:
        if remain <= 0:
            break
        cap = SUBSHELL_CAPACITY[l]
        take = min(cap, remain)
        config[f"{n}{L_LETTER[l]}"] = take
        remain -= take
    return config


def superscript_int(num: int) -> str:
    return str(num).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def config_to_text(config: Dict[str, int]) -> str:
    parts = []
    for n, l in SUBSHELL_ORDER:
        label = f"{n}{L_LETTER[l]}"
        if label in config and config[label] > 0:
            parts.append(f"{label}{superscript_int(config[label])}")
    return " ".join(parts)


def subshell_label(n: int, l: int) -> str:
    return f"{n}{L_LETTER[l]}"


# ---------- 轨道符号统一函数 ----------
def get_orbital_symbol(n: int, l: int, m: int, style: str = "plain") -> str:
    """返回类似 2p_x, 3d_{z^2}, 4f_{xyz} 的轨道符号。style='plain' 给 Unicode 上下标，'latex' 给 LaTeX。"""
    subshell_map = {0: "s", 1: "p", 2: "d", 3: "f"}
    base = f"{n}{subshell_map.get(l, chr(ord('g') + l - 4))}"

    suffix_plain = ""
    suffix_latex = ""

    if l == 1:
        if m == 0:
            suffix_plain, suffix_latex = "z", "z"
        elif m == 1:
            suffix_plain, suffix_latex = "x", "x"
        elif m == -1:
            suffix_plain, suffix_latex = "y", "y"
    elif l == 2:
        if m == 0:
            suffix_plain, suffix_latex = "z²", "z^2"
        elif m == 1:
            suffix_plain, suffix_latex = "xz", "xz"
        elif m == -1:
            suffix_plain, suffix_latex = "yz", "yz"
        elif m == 2:
            suffix_plain, suffix_latex = "x²-y²", "x^2-y^2"
        elif m == -2:
            suffix_plain, suffix_latex = "xy", "xy"
    elif l == 3:
        if m == 0:
            suffix_plain, suffix_latex = "z³", "z^3"
        elif m == 1:
            suffix_plain, suffix_latex = "xz²", "xz^2"
        elif m == -1:
            suffix_plain, suffix_latex = "yz²", "yz^2"
        elif m == 2:
            suffix_plain, suffix_latex = "xyz", "xyz"
        elif m == -2:
            suffix_plain, suffix_latex = "z(x²-y²)", "z(x^2-y^2)"
        elif m == 3:
            suffix_plain, suffix_latex = "x(x²-3y²)", "x(x^2-3y^2)"
        elif m == -3:
            suffix_plain, suffix_latex = "y(3x²-y²)", "y(3x^2-y^2)"

    if style == "latex":
        return rf"{base}_{{{suffix_latex}}}" if suffix_latex else base
    return f"{base}_{suffix_plain}" if suffix_plain else base


def orbital_pretty_label(n: int, l: int, m: int) -> str:
    """界面显示用的紧凑标签。"""
    return get_orbital_symbol(n, l, m, style="plain")


def orbital_key(symbol: str, n: int, l: int, m: int) -> str:
    return f"{symbol}|{n}|{l}|{m}"


def parse_orbital_key(key: str) -> Tuple[str, int, int, int]:
    s, n, l, m = key.split("|")
    return s, int(n), int(l), int(m)


# ---------- 智能平面选择 ----------
def choose_plane(l: int, m: int) -> str:
    """为(l,m)挑选最能展现轨道形状的二维平面。"""
    if l == 0:
        return 'xy'
    if l == 1:
        if m == 0:
            return 'xz'
        elif m == 1:
            return 'xy'
        else:
            return 'xy'
    if l == 2:
        if m == 0:
            return 'xz'
        elif m == 1:
            return 'xz'
        elif m == -1:
            return 'yz'
        elif m == 2:
            return 'xy'
        else:
            return 'xy'
    if l == 3:
        if m == 0:
            return 'xz'
        elif m == 1:
            return 'xz'
        elif m == -1:
            return 'yz'
        elif m == 2:
            return 'xz'
        elif m == -2:
            return 'xy'
        elif m == 3:
            return 'xy'
        else:
            return 'xy'
    # l >= 4
    if m == 0:
        return 'xz'
    elif abs(m) == 1:
        return 'xz'
    elif abs(m) == 2:
        return 'xy'
    elif abs(m) == 3:
        return 'xy'
    return 'xy'


# ==========================================================================
# 记录构造
# ==========================================================================
def element_orbital_records(symbol: str) -> List[dict]:
    elem = get_element(symbol)
    config = build_ground_state_config(elem.Z, symbol)
    records = []
    y_rank = 0
    for n, l in SUBSHELL_ORDER:
        sub = subshell_label(n, l)
        electron_count = config.get(sub, 0)
        if sub not in elem.orbitals:
            continue
        energy = elem.orbitals[sub].energy_hartree
        for i, m in enumerate(M_ORDER_REAL[l]):
            occupied = i < electron_count
            records.append({
                "symbol": symbol,
                "n": n,
                "l": l,
                "m": m,
                "subshell": sub,
                "energy_hartree": energy,
                "occupied": occupied,
                "label": orbital_pretty_label(n, l, m),
                "latex_label": get_orbital_symbol(n, l, m, style="latex"),
                "key": orbital_key(symbol, n, l, m),
                "y_rank": y_rank,
            })
        y_rank += 1
    return records


# ==========================================================================
# 物理量计算
# ==========================================================================
def find_radial_peak(symbol: str, subshell: str, r_max: float = 25.0) -> float:
    """径向概率密度 D(r) = r²R² 的最概然半径。"""
    r = np.linspace(0.001, r_max, 2000)
    d = radial_probability_density(symbol, subshell, r)
    idx = int(np.argmax(d))
    return float(r[idx])


def fix_phase_by_first_lobe(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    仅用于 R(r) 图的显示相位约定：
    如果第一个零点以前的第一波瓣整体为负，则将整条 R(r) 乘以 -1。

    说明：
    - 原子轨道的整体相位任意，R(r) 与 -R(r) 物理等价；
    - 该函数只改变 R(r) 曲线的显示方向；
    - R²(r)、D(r)=r²R²(r)、|ψ|² 等物理量不受影响。
    """
    R = np.asarray(R, dtype=float).copy()

    # 忽略非常接近 0 的数值噪声
    significant = np.abs(R) > eps
    if not np.any(significant):
        return R

    # 从第一个显著非零点开始，避免 r=0 附近的 0 造成误判
    first_nonzero = int(np.argmax(significant))
    R_work = R[first_nonzero:]

    # 寻找第一个真正的符号变化，即第一个径向节点
    signs = np.sign(R_work)
    valid = signs != 0
    signs = signs[valid]
    if len(signs) < 2:
        region = R_work[np.abs(R_work) > eps]
    else:
        change_candidates = np.where(signs[1:] * signs[:-1] < 0)[0]
        if len(change_candidates) == 0:
            region = R_work[np.abs(R_work) > eps]
        else:
            first_zero_relative = change_candidates[0] + 1
            region = R_work[:first_zero_relative]
            region = region[np.abs(region) > eps]

    if len(region) == 0:
        return R

    # 用中位数比均值更不容易被节点附近的小振荡影响
    if np.median(region) < 0:
        return -R
    return R



def get_radial_arrays(symbol: str, subshell: str, r_min: float = 0.0, r_max: float = 20.0, npts: int = 1200):
    """
    生成 r, R(r), R²(r), D(r)=r²R² 四个数组。

    R(r) 仅用于曲线显示时采用“第一波瓣为正”的相位约定：
    如果第一个零点以前的函数值整体小于 0，则整条 R(r) 乘以 -1。
    R²(r) 和 D(r) 仍由原始 R_raw 计算，不受相位显示约定影响。
    """
    r = np.linspace(max(r_min, 0.0), r_max, npts)
    R_raw = orbital_radial(symbol, subshell, r)
    R = fix_phase_by_first_lobe(R_raw)
    R2 = R_raw**2
    D = r**2 * R2
    return r, R, R2, D


def estimate_extent(symbol: str, subshell: str, tail_fraction: float = 0.005) -> float:
    """
    估计绘图区域大小：找径向概率分布 D(r)=r²R² 衰减到峰值 tail_fraction 倍的地方。
    这样能包住轨道的主要形状，同时避免大片空白。
    """
    r = np.linspace(0.001, 30.0, 3000)
    D = radial_probability_density(symbol, subshell, r)
    peak = np.max(D)
    if peak <= 0:
        return 8.0
    threshold = peak * tail_fraction
    peak_idx = int(np.argmax(D))
    tail_r = r[-1]
    # 从峰后往外找：第一次衰减到阈值且之后基本不再反弹
    for i in range(peak_idx, len(r)):
        if D[i] < threshold and (i + 20 >= len(r) or np.all(D[i:] < peak * 0.02)):
            tail_r = r[i]
            break
    # 留一点边距，保证最小尺度
    return max(4.0, tail_r * 1.15)


def evaluate_on_plane(symbol: str, n: int, l: int, m: int, plane: str = None,
                      extent: float = None, ngrid: int = 220):
    """在指定平面上计算 ψ 的实部。"""
    sub = subshell_label(n, l)
    if plane is None:
        plane = choose_plane(l, m)
    if extent is None:
        extent = estimate_extent(symbol, sub)

    a = np.linspace(-extent, extent, ngrid)
    b = np.linspace(-extent, extent, ngrid)
    A, B = np.meshgrid(a, b, indexing="xy")

    if plane == "xy":
        x, y, z = A, B, np.zeros_like(A)
        labels = ("x", "y")
    elif plane == "xz":
        x, y, z = A, np.zeros_like(A), B
        labels = ("x", "z")
    else:  # yz
        x, y, z = np.zeros_like(A), A, B
        labels = ("y", "z")

    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(np.where(r > 0, z / r, 1.0))
        phi = np.arctan2(y, x)
    theta = np.nan_to_num(theta, nan=0.0)
    phi = np.nan_to_num(phi, nan=0.0)
    psi = orbital_wavefunction(symbol, sub, m, r, theta, phi, real_form=True)
    return A, B, psi, plane, labels, extent


# ---------- 角向分布采样 ----------
def sample_unit_circle_on_plane(plane: str, num: int = 720):
    """在给定平面的单位圆上取方向，换算成球坐标(theta, phi)。"""
    t = np.linspace(0, 2 * np.pi, num, endpoint=False)
    if plane == 'xy':
        Xc, Yc, Zc = np.cos(t), np.sin(t), np.zeros_like(t)
    elif plane == 'xz':
        Xc, Yc, Zc = np.cos(t), np.zeros_like(t), np.sin(t)
    else:  # yz
        Xc, Yc, Zc = np.zeros_like(t), np.cos(t), np.sin(t)
    r_unit = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta_sph = np.arccos(np.clip(Zc / r_unit, -1.0, 1.0))
    phi_sph = np.arctan2(Yc, Xc)
    return t, Xc, Yc, Zc, theta_sph, phi_sph


def sample_angular_curve(n: int, l: int, m: int, squared: bool = False):
    """
    采样角向分布曲线。仅依赖 Y_{lm}，与元素无关——角向形状只由 (l,m) 决定。
    返回归一化后的笛卡尔坐标用于"花瓣图"。
    """
    plane = choose_plane(l, m)
    t, Xc, Yc, Zc, theta_sph, phi_sph = sample_unit_circle_on_plane(plane)
    Y_vals = real_spherical_harmonic(l, m, theta_sph, phi_sph)
    magnitude = np.abs(Y_vals)

    # 若所选平面上 Y 恒为零（节面），自动换平面
    if np.allclose(magnitude, 0) or np.std(magnitude) < 1e-6:
        best_plane = plane
        best_std = np.std(magnitude)
        best_dataset = (t, Xc, Yc, Zc, theta_sph, phi_sph, magnitude)
        for cand in ['xy', 'xz', 'yz']:
            t2, Xc2, Yc2, Zc2, th2, ph2 = sample_unit_circle_on_plane(cand)
            Y2 = real_spherical_harmonic(l, m, th2, ph2)
            mag2 = np.abs(Y2)
            s = np.std(mag2)
            if s > best_std + 1e-12:
                best_std = s
                best_plane = cand
                best_dataset = (t2, Xc2, Yc2, Zc2, th2, ph2, mag2)
        plane = best_plane
        t, Xc, Yc, Zc, theta_sph, phi_sph, magnitude = best_dataset

    if squared:
        magnitude = magnitude ** 2
    max_mag = np.max(magnitude) if magnitude.size > 0 else 0.0
    mag_norm = magnitude / max_mag if max_mag > 0 else magnitude

    if plane == 'xy':
        plot_x, plot_y = mag_norm * Xc, mag_norm * Yc
        labels = ('x', 'y')
    elif plane == 'xz':
        plot_x, plot_y = mag_norm * Xc, mag_norm * Zc
        labels = ('x', 'z')
    else:
        plot_x, plot_y = mag_norm * Yc, mag_norm * Zc
        labels = ('y', 'z')

    return np.nan_to_num(plot_x), np.nan_to_num(plot_y), labels, plane


# ---------- 电子云蒙特卡洛采样（径向反 CDF + 角向拒绝，对 Bunge 轨道高效） ----------
def sample_cloud(symbol: str, n: int, l: int, m: int, npts: int = 20000):
    """
    采样电子云 |ψ|² = R²(r)|Y(θ,φ)|²。
    用两步解耦采样以适配 Bunge 紧凑轨道：
      (1) 沿 r 按 D(r)=r²R²(r) 的逆累积分布取样（概率密度自带 r² 权重）；
      (2) 沿方向按 |Y_{lm}(θ,φ)|² 做拒绝采样得到 (θ,φ)。
    这样每个保留点都在有意义的概率密度处，不会出现大部分被拒绝的情况。
    返回 (x, y, z, |ψ|², r_max_for_display).
    """
    sub = subshell_label(n, l)

    # --- 径向：按 D(r) 做逆 CDF 采样 ---
    r_grid = np.linspace(1e-4, 30.0, 4000)
    D = radial_probability_density(symbol, sub, r_grid)
    D = np.maximum(D, 0.0)
    if D.sum() <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 8.0
    cdf = np.cumsum(D)
    cdf /= cdf[-1]
    u = np.random.rand(npts)
    r_samp = np.interp(u, cdf, r_grid)

    # --- 角向：按 |Y|² 做拒绝采样（在 4π 立体角均匀候选） ---
    # 先估计 |Y|² 的上界（保守取一个全向离散最大）
    # 粗采样找上界
    tt_pre = np.arccos(1 - 2 * np.random.rand(4000))
    pp_pre = 2 * np.pi * np.random.rand(4000)
    Y_pre = real_spherical_harmonic(l, m, tt_pre, pp_pre)
    y2_max = float(np.max(Y_pre ** 2))
    if y2_max <= 0:
        y2_max = 1e-12

    theta_samp = np.empty(npts)
    phi_samp = np.empty(npts)
    filled = 0
    safety = 0
    while filled < npts and safety < 80:
        safety += 1
        batch = max(npts - filled, 4000) * 3
        # 在球面上均匀采样方向：cosθ 均匀 in [-1,1], φ 均匀 in [0,2π]
        cos_t = 1 - 2 * np.random.rand(batch)
        t = np.arccos(np.clip(cos_t, -1, 1))
        ph = 2 * np.pi * np.random.rand(batch)
        Yv = real_spherical_harmonic(l, m, t, ph)
        acc = (np.random.rand(batch) * y2_max) < (Yv ** 2)
        got = np.sum(acc)
        if got == 0:
            # 对 l=0 时 |Y|² 常数，上面应当全收；此处为极端兜底
            y2_max *= 0.5
            continue
        take = min(got, npts - filled)
        idx = np.where(acc)[0][:take]
        theta_samp[filled:filled + take] = t[idx]
        phi_samp[filled:filled + take] = ph[idx]
        filled += take

    # 截到实际填充数（防极端情况）
    theta_samp = theta_samp[:filled]
    phi_samp = phi_samp[:filled]
    r_samp = r_samp[:filled]

    # --- 合成笛卡尔坐标 ---
    sin_t = np.sin(theta_samp)
    x = r_samp * sin_t * np.cos(phi_samp)
    y = r_samp * sin_t * np.sin(phi_samp)
    z = r_samp * np.cos(theta_samp)

    # --- 计算对应 |Y|² 值（用于颜色映射）---
    psi = orbital_wavefunction(symbol, sub, m, r_samp, theta_samp, phi_samp, real_form=True)
    prob = psi ** 2

    # 用于显示坐标轴的 r_max 取采样点最远 r 和 extent 的更大者
    r_max_display = max(float(np.max(np.abs([x.max() if len(x) else 0,
                                               y.max() if len(y) else 0,
                                               z.max() if len(z) else 0,
                                               x.min() if len(x) else 0,
                                               y.min() if len(y) else 0,
                                               z.min() if len(z) else 0]))) ,
                        estimate_extent(symbol, sub))
    return x, y, z, prob, r_max_display


# ==========================================================================
# 绘图函数
# ==========================================================================
def fig_radial(selected_orbs: List[dict], mode: str,
               x_min: float = 0.0, x_max: float = 10.0,
               y_min: float = None, y_max: float = None):
    """径向相关图像：R(r), R²(r), D(r) = r²R²。支持多轨道叠加比较。"""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ylabel, title = "", ""
    for i, orb in enumerate(selected_orbs):
        color = orb.get("color") or COLOR_POOL[i % len(COLOR_POOL)]
        # 采样点取 1200，保证核附近 R(r) 的剧烈变化（尤其是 n≥2 轨道的核区反冲）被细致捕捉
        r, R, R2, D = get_radial_arrays(orb["symbol"], orb["subshell"], x_min, x_max, npts=1200)
        if mode == "径向函数图 R(r)-r":
            y, ylabel, title = R, r"$R(r)$", "径向函数图 R(r)"
        elif mode == "径向密度函数图 R²(r)-r":
            y, ylabel, title = R2, r"$R^2(r)$", "径向密度函数图 R²(r)"
        else:
            y, ylabel, title = D, r"$D(r)=r^2 R^2(r)$", "径向分布函数图 D(r)"

        label = fr'${orb["symbol"]}\ {orb["latex_label"]}$'
        ax.plot(r, y, linewidth=2.0, label=label, color=color, alpha=0.9)

    ax.set_xlabel(r"$r / a_0$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontname='Noto Sans SC', fontsize=13)
    ax.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    return fig


def fig_angular(orb: dict, squared: bool = False):
    """
    角向分布花瓣图。
    到原点的距离使用 |Y_l^m| 或 |Y_l^m|²，而不是完整波函数 psi。
    """
    plot_x, plot_y, labels, plane = sample_angular_curve(
        orb["n"], orb["l"], orb["m"], squared=squared
    )

    lim = 1.1
    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    square = Rectangle((-lim, -lim), 2 * lim, 2 * lim,
                       facecolor='white', edgecolor='lightgray', lw=1.0, zorder=0)
    ax.add_patch(square)
    ax.axhline(0, color='k', linewidth=1.0, zorder=3)
    ax.axvline(0, color='k', linewidth=1.0, zorder=3)

    px = np.concatenate([plot_x, plot_x[:1]])
    py = np.concatenate([plot_y, plot_y[:1]])
    ax.plot(px, py, '-', linewidth=2.2, zorder=5, color='#0099ff')
    ax.fill(px, py, alpha=0.35, color='#74a9ff', zorder=4)

    ax.set_xticks(np.linspace(-lim, lim, 3))
    ax.set_yticks(np.linspace(-lim, lim, 3))
    ax.grid(True, linestyle='--', alpha=0.25, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(axis='both', labelsize=7.5)
    ax.set_xlabel(f"${labels[0]}$", fontsize=7.5)
    ax.set_ylabel(f"${labels[1]}$", fontsize=7.5)

    if squared:
        title_kind = r"电子云角向分布 $|Y_l^m(\theta,\phi)|^2$"
    else:
        title_kind = r"原子轨道角向分布 $|Y_l^m(\theta,\phi)|$"

    ax.set_title(
        f"{title_kind}",
        fontsize=8, fontname='Noto Sans SC'
    )
    fig.tight_layout()
    return fig


def fig_contour(orb: dict, extent: float = None):
    """轨道等值线图 Re[ψ]。红色表示正值，蓝色表示负值，黑色虚线为 ψ=0 节面。"""
    A, B, psi, plane, labels, extent = evaluate_on_plane(
        orb["symbol"], orb["n"], orb["l"], orb["m"],
        extent=extent, ngrid=220
    )
    psi_real = np.real(psi)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=85)
    vmax = np.max(np.abs(psi_real))
    if vmax < 1e-12:
        vmax = 1e-12
    num_levels = 12
    levels = np.linspace(-vmax, vmax, num_levels)

    cf = ax.contourf(A, B, psi_real, levels=levels, cmap='RdBu_r', alpha=1)

    # ψ=0 节面
    ax.contour(A, B, psi_real, levels=[0], colors='black',
               linestyles='dashed', linewidths=1.5)

    # 细线等高线
    pos_lv = levels[levels > 0]
    neg_lv = levels[levels < 0]
    if len(pos_lv) > 0:
        ax.contour(A, B, psi_real, levels=pos_lv, colors='darkred',
                   linestyles='solid', linewidths=0.3, alpha=0.7)
    if len(neg_lv) > 0:
        ax.contour(A, B, psi_real, levels=neg_lv, colors='darkblue',
                   linestyles='solid', linewidths=0.3, alpha=0.7)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label('波函数值 ψ', fontsize=8)

    ax.set_aspect("equal")
    ax.set_title(
        f"轨道等值线图",
        fontname='Noto Sans SC', fontsize=8
    )
    ax.set_xlabel(f"${labels[0]} / a_0$", fontsize=5)
    ax.set_ylabel(f"${labels[1]} / a_0$", fontsize=5)
    ax.grid(True, linestyle=':', alpha=0.5)

    ax.text(0.02, 0.02,
            "红色: 正值\n蓝色: 负值\n黑色虚线: 节面",
            transform=ax.transAxes, fontsize=5,
            fontname='Noto Sans SC', 
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))

    fig.tight_layout()
    return fig


def fig_surface(orb: dict):
    """3D 网格变形图：把平面网格按 ψ 值'拉起/压下'。"""
    A, B, psi, plane, labels, extent = evaluate_on_plane(
        orb["symbol"], orb["n"], orb["l"], orb["m"],
        extent=None, ngrid=60
    )
    psi_real = np.real(psi)
    max_abs_psi = np.max(np.abs(psi_real))
    if max_abs_psi < 1e-12:
        scale_factor = 1.0
    else:
        scale_factor = 0.5 * extent / max_abs_psi

    custom_colorscale = [
        [0.0, 'darkblue'], [0.25, 'blue'], [0.4, 'lightblue'],
        [0.5, 'white'],
        [0.6, 'lightcoral'], [0.75, 'red'], [1.0, 'darkred']
    ]

    fig = go.Figure()
    grid_size = A.shape[0]

    # 根据平面选择：把"波函数值"放到垂直于平面的轴上
    if plane == 'xy':
        Xs, Ys, Zs = A, B, scale_factor * psi_real
        axis_labels = {'x': 'x / a₀', 'y': 'y / a₀', 'z': '波函数值 (缩放)'}
    elif plane == 'xz':
        Xs, Ys, Zs = A, scale_factor * psi_real, B
        axis_labels = {'x': 'x / a₀', 'y': '波函数值 (缩放)', 'z': 'z / a₀'}
    else:  # yz
        Xs, Ys, Zs = scale_factor * psi_real, A, B
        axis_labels = {'x': '波函数值 (缩放)', 'y': 'y / a₀', 'z': 'z / a₀'}

    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs,
        surfacecolor=psi_real,
        colorscale=custom_colorscale,
        cmin=-max_abs_psi, cmax=max_abs_psi,
        showscale=True, opacity=0.85,
        name='波函数',
        colorbar=dict(title='ψ', thickness=18, len=0.7)
    ))

    # 叠加黑色网格线（每隔一行/列画一次，降低渲染压力）
    for i in range(0, grid_size, 2):
        fig.add_trace(go.Scatter3d(
            x=Xs[i, :], y=Ys[i, :], z=Zs[i, :],
            mode='lines',
            line=dict(color='black', width=1.2),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=Xs[:, i], y=Ys[:, i], z=Zs[:, i],
            mode='lines',
            line=dict(color='black', width=1.2),
            showlegend=False, hoverinfo='skip'
        ))

    # 原点标记
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=3, color='black'),
        showlegend=False, hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"轨道网格变形图  {orb['symbol']} {orb['label']}  ({plane} 平面, 缩放×{scale_factor:.2f})",
        scene=dict(
            xaxis_title=axis_labels['x'],
            yaxis_title=axis_labels['y'],
            zaxis_title=axis_labels['z'],
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=560,
        margin=dict(l=10, r=10, b=10, t=50),
    )
    return fig


def fig_cloud(orb: dict, npts: int = 20000):
    """电子云图（蒙特卡洛拒绝采样）。颜色映射到概率密度，附三色坐标轴。"""
    xk, yk, zk, pk, r_max = sample_cloud(
        orb["symbol"], orb["n"], orb["l"], orb["m"], npts=npts
    )

    fig = go.Figure()

    if len(xk) == 0:
        fig.add_annotation(text="采样点不足，请增加采样数", showarrow=False)
    else:
        fig.add_trace(go.Scatter3d(
            x=xk, y=yk, z=zk,
            mode='markers',
            marker=dict(
                size=2.5,
                color=pk,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='|ψ|²', thickness=18, len=0.7)
            ),
            name='电子云',
            hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>|ψ|²=%{marker.color:.4g}<extra></extra>'
        ))

    # 三色坐标轴
    axis_line_len = r_max
    fig.add_trace(go.Scatter3d(
        x=[-axis_line_len, axis_line_len], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=3), name='X轴'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_line_len, axis_line_len], z=[0, 0],
        mode='lines', line=dict(color='green', width=3), name='Y轴'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_line_len, axis_line_len],
        mode='lines', line=dict(color='blue', width=3), name='Z轴'
    ))

    fig.update_layout(
        title=f"电子云图 |ψ|²  {orb['symbol']} ${orb['latex_label']}$  (保留 {len(xk)} 点)",
        scene=dict(
            xaxis_title='x / a₀',
            yaxis_title='y / a₀',
            zaxis_title='z / a₀',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=560,
        margin=dict(l=10, r=10, b=10, t=40),
        showlegend=True,
    )
    return fig


# ==========================================================================
# 分发 & 布局
# ==========================================================================
def draw_plot(plot_type: str, selected_orbs: List[dict], plot_prefix: str = ""):
    """绘制一种图类型。plot_prefix 用于给用户控件分配唯一 key。"""
    if plot_type in MULTI_PLOT_TYPES:
        # 径向类图：允许用户调整 x/y 轴范围。
        # 默认 x_max 按所选轨道的物理尺度（estimate_extent）自适应——
        # Bunge 轨道非常紧凑（Ar 2p 只到 r~2 a0），若用固定的 20 会把细节压成一根刺。
        default_xmax = max(
            estimate_extent(orb["symbol"], orb["subshell"])
            for orb in selected_orbs
        )
        # 适当放宽一些，方便看轨道的衰减尾部
        default_xmax = round(default_xmax * 1.3, 1)

        # 关键：key 必须携带"当前选中轨道"的指纹，否则 Streamlit 的 number_input
        # 会缓存旧 value（比如 20.0），切换元素/轨道时新 default 无法生效。
        orb_fingerprint = "_".join(sorted(o["key"] for o in selected_orbs))
        widget_key_base = f"{plot_prefix}_{plot_type}_{orb_fingerprint}"

        with st.expander(f"⚙ {plot_type} - 轴范围控制", expanded=False):
            # 注意：此处不能再用 st.columns()，因为 draw_plot 已经处于
            # 顶层 columns（main 的 left/center/right）和中层 columns（adaptive_plot_layout）
            # 的双重嵌套之中——Streamlit 限制 columns 嵌套最多 1 层（即 2 层深度）。
            # 因此 4 个控件直接顺序排列。
            x_min = st.number_input("X 最小", value=0.0, format="%.3f",
                                    key=f"{widget_key_base}_xmin")
            x_max = st.number_input("X 最大", value=default_xmax, format="%.2f",
                                    key=f"{widget_key_base}_xmax")

            # Y 轴默认按全局范围展示。
            # R(r) 已在 fix_phase_by_first_lobe 中统一为"第一波瓣为正"的相位约定，
            # 因此所有 s 轨道都从核处一个正大值出发，单调下降经过节点（若有）后衰减——
            # 视觉风格统一，无需再做聚焦裁剪。
            all_y = []
            for orb in selected_orbs:
                _, R, R2, D = get_radial_arrays(orb["symbol"], orb["subshell"],
                                                 x_min, x_max, npts=600)
                if plot_type == "径向函数图 R(r)-r":
                    all_y.append(R)
                elif plot_type == "径向密度函数图 R²(r)-r":
                    all_y.append(R2)
                else:
                    all_y.append(D)

            y_all = np.concatenate(all_y) if all_y else np.array([0.0, 1.0])
            y_min_def = float(np.min(y_all))
            y_max_def = float(np.max(y_all))
            pad = 0.05 * (y_max_def - y_min_def + 1e-9)
            y_min_def -= pad
            y_max_def += pad

            y_min = st.number_input("Y 最小", value=y_min_def, format="%.4f",
                                    key=f"{widget_key_base}_ymin")
            y_max = st.number_input("Y 最大", value=y_max_def, format="%.4f",
                                    key=f"{widget_key_base}_ymax")

        fig = fig_radial(selected_orbs, plot_type, x_min, x_max, y_min, y_max)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    orb = selected_orbs[0]
    if plot_type == "原子轨道角度分布图 |Y|-角度":
        fig = fig_angular(orb, squared=False)
        st.pyplot(fig, width='content')
        plt.close(fig)
    elif plot_type == "电子云角度分布图 |Y|²-角度":
        fig = fig_angular(orb, squared=True)
        st.pyplot(fig, width='content')
        plt.close(fig)
    elif plot_type == "原子轨道等值线图":
        fig = fig_contour(orb)
        st.pyplot(fig, width='content')
        plt.close(fig)
    elif plot_type == "原子轨道网格图":
        fig = fig_surface(orb)
        st.plotly_chart(fig, width='content')
    elif plot_type == "电子云黑点图":
        npts = st.slider("采样点数", 5000, 60000, 20000, step=5000,
                         key=f"{plot_prefix}_cloud_npts")
        fig = fig_cloud(orb, npts=npts)
        st.plotly_chart(fig)


def render_energy_diagram(records: List[dict], selected_keys: List[str]):
    """
    教材风格电子能级图（可点击选择轨道）。

    - 纵轴为真实 orbital energy / Eh；
    - 每个 subshell 画成一排小方框；
    - s/p/d/f 简并轨道横向展开；
    - 用户可直接点击方框上的透明 marker 来选择/取消选择轨道；
    - 选中的轨道用红色边框加粗显示。
    """
    if not records:
        st.info("暂无轨道数据。")
        return

    shell_colors = {
        1: "#d65f9e",
        2: "#58b97f",
        3: "#ff6b6b",
        4: "#4d9de0",
        5: "#9b59b6",
    }

    groups = []
    seen = set()
    for rec in records:
        key = (rec["n"], rec["l"], rec["subshell"], round(float(rec["energy_hartree"]), 8))
        if key in seen:
            continue
        seen.add(key)
        members = [r for r in records if r["subshell"] == rec["subshell"]]
        order = M_ORDER_REAL.get(rec["l"], [r["m"] for r in members])
        members = sorted(members, key=lambda r: order.index(r["m"]) if r["m"] in order else r["m"])
        groups.append({
            "n": rec["n"],
            "l": rec["l"],
            "subshell": rec["subshell"],
            "energy": float(rec["energy_hartree"]),
            "members": members,
        })

    groups = sorted(groups, key=lambda g: g["energy"])

    energies = [g["energy"] for g in groups]
    e_min, e_max = min(energies), max(energies)
    e_pad = max((e_max - e_min) * 0.12, 0.8)

    fig = go.Figure()

    x_axis = -1.05
    y0_axis = e_min - e_pad * 0.45
    y1_axis = max(0.20, e_max + e_pad * 0.65)

    # 能量轴
    fig.add_shape(
        type="line",
        x0=x_axis, x1=x_axis,
        y0=y0_axis,
        y1=y1_axis,
        line=dict(color="black", width=2),
        layer="below",
    )
    fig.add_annotation(
        x=x_axis,
        y=y1_axis,
        ax=x_axis,
        ay=y1_axis - max((y1_axis - y0_axis) * 0.08, 0.3),
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor="black",
        text="",
    )
    fig.add_annotation(
        x=x_axis - 0.10,
        y=(y0_axis + y1_axis) / 2,
        text="Energy",
        textangle=-90,
        showarrow=False,
        font=dict(size=13, color="black"),
    )

    # 轨道方框横向位置。不同 l 略错开，同一 subshell 中的简并轨道展开。
    x_start = {0: -0.76, 1: -0.48, 2: -0.15, 3: 0.18}
    box_w = 0.095
    box_h = max((e_max - e_min) * 0.020, 0.055)
    gap = 0.030

    click_x, click_y, click_text, click_customdata = [], [], [], []

    for g in groups:
        n = g["n"]
        l = g["l"]
        subshell = g["subshell"]
        energy = g["energy"]
        members = g["members"]
        color = shell_colors.get(n, "#777777")

        total_w = len(members) * box_w + (len(members) - 1) * gap
        x0 = x_start.get(l, -0.50)
        x1 = x0 + total_w

        # 水平能级线
        fig.add_shape(
            type="line",
            x0=x_axis,
            x1=max(x1 + 0.18, 0.55),
            y0=energy,
            y1=energy,
            line=dict(color="rgba(120,120,120,0.25)", width=1),
            layer="below",
        )

        # subshell 标签
        fig.add_annotation(
            x=x0 - 0.075,
            y=energy,
            text=subshell,
            showarrow=False,
            font=dict(size=12, color=color),
            xanchor="right",
            yanchor="middle",
        )

        for i, rec in enumerate(members):
            bx0 = x0 + i * (box_w + gap)
            bx1 = bx0 + box_w
            by0 = energy - box_h / 2
            by1 = energy + box_h / 2

            selected = rec["key"] in selected_keys
            occupied = rec["occupied"]

            line_style = dict(
                color="#d62728" if selected else color,
                width=3.2 if selected else 1.7,
                dash="solid" if occupied else "dot",
            )
            fill = (
                "rgba(214,95,158,0.13)" if n == 1 else
                "rgba(88,185,127,0.13)" if n == 2 else
                "rgba(255,107,107,0.13)" if n == 3 else
                "rgba(77,157,224,0.13)"
            )
            if not occupied:
                fill = "rgba(220,220,220,0.06)"

            fig.add_shape(
                type="rect",
                x0=bx0, x1=bx1,
                y0=by0, y1=by1,
                line=line_style,
                fillcolor=fill,
                layer="above",
            )

            cx = (bx0 + bx1) / 2
            cy = energy
            click_x.append(cx)
            click_y.append(cy)
            click_text.append(rec["label"])
            click_customdata.append([
                rec["key"],
                rec["subshell"],
                float(rec["energy_hartree"]),
                "已占据" if occupied else "未占据",
            ])

    # 透明但可点击的 marker。点击事件由 st.plotly_chart(..., on_select="rerun") 捕获。
    fig.add_trace(go.Scatter(
        x=click_x,
        y=click_y,
        mode="markers",
        marker=dict(size=24, color="rgba(0,0,0,0.001)", line=dict(width=0)),
        text=click_text,
        customdata=click_customdata,
        hovertemplate=(
            "轨道: %{text}<br>"
            "子层: %{customdata[1]}<br>"
            "真实能量: %{customdata[2]:.4f} Eh<br>"
            "状态: %{customdata[3]}<br>"
            "点击选择/取消选择"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # n 壳层标注
    shell_groups = {}
    for g in groups:
        shell_groups.setdefault(g["n"], []).append(g["energy"])
    for n, es in shell_groups.items():
        fig.add_annotation(
            x=0.70,
            y=max(es),
            text=f"n = {n}",
            showarrow=False,
            font=dict(size=11, color=shell_colors.get(n, "#777777")),
            xanchor="left",
        )

    # n=inf 参考线
    fig.add_shape(
        type="line",
        x0=x_axis,
        x1=0.75,
        y0=0,
        y1=0,
        line=dict(color="rgba(80,80,80,0.55)", width=1, dash="dash"),
        layer="below",
    )
    fig.add_annotation(
        x=0.72,
        y=0,
        text="n = inf",
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        title="电子能级图（点击方框选择轨道）",
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1.15, 0.88],
            title="",
            fixedrange=True,
        ),
        yaxis=dict(
            title="Orbital Energy / Eh",
            showgrid=False,
            zeroline=False,
            range=[e_min - e_pad, max(0.30, e_max + e_pad)],
            fixedrange=False,
        ),
        height=620,
        margin=dict(l=10, r=10, t=45, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        clickmode="event+select",
        dragmode=False,
    )

    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"energy_diagram_{records[0]['symbol']}",
            on_select="rerun",
            selection_mode="points",
        )
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
        if points:
            clicked_key = points[0].get("customdata", [None])[0]
            if clicked_key:
                if clicked_key in st.session_state.selected_orbital_keys:
                    st.session_state.selected_orbital_keys.remove(clicked_key)
                    st.session_state.orbital_colors.pop(clicked_key, None)
                else:
                    st.session_state.selected_orbital_keys.append(clicked_key)
                    used = set(st.session_state.orbital_colors.values())
                    avail = [c for c in COLOR_POOL if c not in used]
                    st.session_state.orbital_colors[clicked_key] = (
                        random.choice(avail) if avail else random.choice(COLOR_POOL)
                    )
                st.rerun()
    except TypeError:
        # 兼容旧版 Streamlit：不支持 on_select 时仍正常显示图，但无法点击选择。
        st.plotly_chart(fig, use_container_width=True)
        st.caption("当前 Streamlit 版本不支持 plotly 点击选择；请升级 Streamlit 或使用下方备用选择器。")


def adaptive_plot_layout(plot_types: List[str], selected_orbs: List[dict]):
    n = len(plot_types)

    if n == 0:
        st.info("请先在右侧选择 1–4 种可视化类型。")
        return

    # 1 张图：单独显示
    if n == 1:
        draw_plot(plot_types[0], selected_orbs, plot_prefix="p0")
        return

    # 2 张图：一行两列
    if n == 2:
        cols = st.columns(2)
        for i, p in enumerate(plot_types):
            with cols[i]:
                draw_plot(p, selected_orbs, plot_prefix=f"p{i}")
        return

    # 3 或 4 张图：统一 2×2 网格，保证尺寸一致
    cols_row1 = st.columns(2)
    cols_row2 = st.columns(2)

    grid_cols = [cols_row1[0], cols_row1[1], cols_row2[0], cols_row2[1]]

    for i, p in enumerate(plot_types):
        with grid_cols[i]:
            draw_plot(p, selected_orbs, plot_prefix=f"p{i}")

# ==========================================================================
# 主页面
# ==========================================================================
def main():
    st.set_page_config(page_title="前四周期原子轨道可视化", layout="wide")
    st.title("原子轨道可视化")
    st.caption("Produced by Jinghao Wang & Haibei Li; Data Source: Bunge RHF database")

    supported = list_supported_elements()
    # 仅保留前四周期（Z ≤ 36）；如果数据库只到 Ar，则按原数据库范围
    supported = [s for s in supported if get_element(s).Z <= 36]

    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = "O"
    if "selected_orbital_keys" not in st.session_state:
        st.session_state.selected_orbital_keys = []
    if "selected_plot_types" not in st.session_state:
        st.session_state.selected_plot_types = ["径向函数图 R(r)-r"]
    if "orbital_colors" not in st.session_state:
        st.session_state.orbital_colors = {}

    left, center, right = st.columns([0.70, 1.00, 1.80])

    with left:
        st.subheader("原子选择与信息")
        symbol = st.selectbox(
            "查看原子",
            options=supported,
            index=supported.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in supported else 0,
        )
        st.session_state.selected_symbol = symbol
        elem = get_element(symbol)
        config = build_ground_state_config(elem.Z, symbol)
        st.markdown(f"**元素**：{symbol}（{ELEMENT_NAMES.get(symbol, symbol)}）")
        st.markdown(f"**原子序数**：{elem.Z}")
        st.markdown(f"**电子排布**：{config_to_text(config)}")
        st.markdown(f"**基态项符号**：{elem.term_symbol}")
        st.markdown(f"**数据库占据轨道**：{', '.join(list_orbitals(symbol))}")

        if st.button("清空已选轨道", use_container_width=True):
            st.session_state.selected_orbital_keys = []
            st.session_state.orbital_colors = {}

    records = element_orbital_records(symbol)

    with center:
        st.subheader("电子能级图与轨道选择")
        st.caption("点击能级图中的方框即可选择/取消选择轨道。")
        render_energy_diagram(records, st.session_state.selected_orbital_keys)

        if st.session_state.selected_orbital_keys:
            selected_names = []
            for rec in records:
                if rec["key"] in st.session_state.selected_orbital_keys:
                    selected_names.append(rec["label"])
            st.markdown("**已选轨道**：" + "，".join(selected_names))

        # 备用选择器：仅当 Plotly 点击选择在当前环境不可用时使用
        with st.expander("备用：手动选择轨道", expanded=False):
            option_map = {
                f'{rec["label"]} | {rec["energy_hartree"]:.4f} Eh | {"已占据" if rec["occupied"] else "未占据"}': rec["key"]
                for rec in records
            }
            reverse_map = {v: k for k, v in option_map.items()}
            default_labels = [
                reverse_map[k] for k in st.session_state.selected_orbital_keys
                if k in reverse_map
            ]
            manual_selected = st.multiselect(
                "选择轨道",
                options=list(option_map.keys()),
                default=default_labels,
                key=f"manual_orbital_select_{symbol}",
            )
            if st.button("应用手动选择", key=f"apply_manual_{symbol}"):
                manual_keys = [option_map[label] for label in manual_selected]
                st.session_state.selected_orbital_keys = manual_keys
                
                for key in list(st.session_state.orbital_colors.keys()):
                    if key not in manual_keys:
                        st.session_state.orbital_colors.pop(key, None)
                
                for key in manual_keys:
                    if key not in st.session_state.orbital_colors:
                        used = set(st.session_state.orbital_colors.values())
                        avail = [c for c in COLOR_POOL if c not in used]
                        st.session_state.orbital_colors[key] = (
                            random.choice(avail) if avail else random.choice(COLOR_POOL)
                        )
                        
                st.rerun()

    with right:
        st.subheader("可视化结果")
        selected_records = [r for r in records if r["key"] in st.session_state.selected_orbital_keys]
        # 附加颜色
        for r in selected_records:
            r["color"] = st.session_state.orbital_colors.get(r["key"])

        if not selected_records:
            st.info("请先在中间栏至少选择一个轨道。")
            return

        options = SINGLE_PLOT_TYPES if len(selected_records) == 1 else MULTI_PLOT_TYPES
        current_default = [p for p in st.session_state.selected_plot_types if p in options]
        if not current_default:
            current_default = [options[0]]

        selected_plot_types = st.multiselect(
            "原子轨道可视化类型（最多 4 种）",
            options=options,
            default=current_default,
            max_selections=4,
        )
        st.session_state.selected_plot_types = selected_plot_types

        selected_labels = ", ".join([f'{r["symbol"]} {r["label"]}' for r in selected_records])
        st.markdown(f"**当前选中轨道**：{selected_labels}")
        if len(selected_records) > 1:
            st.caption("多轨道比较时仅显示三类径向相关图像。")
        adaptive_plot_layout(selected_plot_types, selected_records)


if __name__ == "__main__":
    main()
