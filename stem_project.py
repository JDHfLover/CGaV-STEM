"""
STEM-проєкт з дисципліни «Комп'ютерна графіка та візуалізація».
Тема: Аналіз та обробка зображень за допомогою класичних методів
фільтрації, сегментації та аналізу текстур.
Виконав: Бобенко Сергій, група ІПЗ-23(08).

Усі алгоритми обробки зображень (6 методів із 6 різних категорій)
та метрики якості (MSE, PSNR, SSIM, ентропія Шеннона, IoU, F1, GLCM)
реалізовані вручну з використанням базових матричних операцій NumPy.
Бібліотеки PIL та Matplotlib використовуються виключно для введення/виведення
та візуалізації результатів.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


# ============================================================================
# 1. МЕТРИКИ ЯКОСТІ ЗОБРАЖЕНЬ (РЕАЛІЗОВАНІ ВРУЧНУ)
# ============================================================================

def validate_and_prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Перевіряє та приводить вхідний масив зображення до двовимірного
    масиву uint8 (градації сірого) з обробкою межових випадків (критерій C2).
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Вхідне зображення має бути непорожнім масивом NumPy.")
    if img.size == 0:
        raise ValueError("Зображення є порожнім (0 пікселів).")

    if img.ndim == 3:
        if img.shape[2] >= 3:
            # Стандартне перетворення RGB/RGBA -> Y (градації сірого)
            gray = (
                0.299 * img[:, :, 0].astype(np.float32)
                + 0.587 * img[:, :, 1].astype(np.float32)
                + 0.114 * img[:, :, 2].astype(np.float32)
            )
            return np.clip(np.round(gray), 0, 255).astype(np.uint8)
        if img.shape[2] == 1:
            return np.clip(np.round(img[:, :, 0]), 0, 255).astype(np.uint8)
        raise ValueError(f"Непідтримувана кількість каналів: {img.shape[2]}")

    if img.ndim == 2:
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    raise ValueError(f"Непідтримувана розмірність масиву: {img.ndim}")


def calc_mse(img_ref: np.ndarray, img_proc: np.ndarray) -> float:
    """
    Обчислює середньоквадратичну похибку (MSE) між двома зображеннями:
    MSE = (1 / N) * sum((I_ref - I_proc)^2).
    """
    ref = validate_and_prepare_image(img_ref).astype(np.float64)
    proc = validate_and_prepare_image(img_proc).astype(np.float64)
    if ref.shape != proc.shape:
        raise ValueError(f"Розміри зображень не збігаються: {ref.shape} != {proc.shape}")
    diff = ref - proc
    return float(np.mean(diff * diff))


def calc_psnr(img_ref: np.ndarray, img_proc: np.ndarray, max_val: float = 255.0) -> float:
    """
    Обчислює пікове відношення сигнал/шум (PSNR, дБ):
    PSNR = 10 * log10(MAX^2 / MSE).
    Якщо MSE == 0 (ідентичні зображення), повертає 100.0 дБ.
    """
    mse = calc_mse(img_ref, img_proc)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10((max_val * max_val) / mse))


def calc_ssim(
    img_ref: np.ndarray,
    img_proc: np.ndarray,
    window_size: int = 7
) -> float:
    """
    Обчислює індекс структурної подібності (SSIM) вручну за формулою Ванга та ін.:
    SSIM(x, y) = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) /
                 ((mu_x^2 + mu_y^2 + C1)*(sigma_x^2 + sigma_y^2 + C2)).
    Для малих зображень (< window_size) обчислюється глобальна оцінка SSIM.
    """
    x = validate_and_prepare_image(img_ref).astype(np.float64)
    y = validate_and_prepare_image(img_proc).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Розміри зображень не збігаються: {x.shape} != {y.shape}")

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    h, w = x.shape
    eff_win = min(window_size, h, w)
    if eff_win % 2 == 0:
        eff_win = max(1, eff_win - 1)

    if eff_win < 3:
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.mean((x - mu_x) ** 2))
        var_y = float(np.mean((y - mu_y) ** 2))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        return float(num / den)

    pad = eff_win // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = np.pad(y, pad_width=pad, mode="reflect")

    win_x = np.lib.stride_tricks.sliding_window_view(x_pad, (eff_win, eff_win))
    win_y = np.lib.stride_tricks.sliding_window_view(y_pad, (eff_win, eff_win))

    mu_x = np.mean(win_x, axis=(-2, -1))
    mu_y = np.mean(win_y, axis=(-2, -1))

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = np.mean(win_x * win_x, axis=(-2, -1)) - mu_x2
    sigma_y2 = np.mean(win_y * win_y, axis=(-2, -1)) - mu_y2
    sigma_xy = np.mean(win_x * win_y, axis=(-2, -1)) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.clip(np.mean(ssim_map), -1.0, 1.0))


def calc_shannon_entropy(img: np.ndarray) -> float:
    """
    Обчислює інформаційну ентропію Шеннона H = -sum(p_i * log2(p_i)).
    """
    arr = validate_and_prepare_image(img)
    counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    probs = counts / float(arr.size)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def calc_iou_and_f1(binary_pred: np.ndarray, binary_gt: np.ndarray) -> Tuple[float, float]:
    """
    Обчислює коефіцієнт Жаккара (IoU) та F1-міру для бінарних масок сегментації/контурів.
    """
    pred = validate_and_prepare_image(binary_pred) >= 128
    gt = validate_and_prepare_image(binary_gt) >= 128

    # Враховуємо можливу інверсію полярності класів при пороговій сегментації
    inter1 = int(np.logical_and(pred, gt).sum())
    union1 = int(np.logical_or(pred, gt).sum())
    iou1 = inter1 / union1 if union1 > 0 else 1.0

    pred_inv = np.logical_not(pred)
    inter2 = int(np.logical_and(pred_inv, gt).sum())
    union2 = int(np.logical_or(pred_inv, gt).sum())
    iou2 = inter2 / union2 if union2 > 0 else 1.0

    if iou2 > iou1 and np.mean(gt) > 0.15:
        pred = pred_inv
        inter = inter2
        union = union2
    else:
        inter = inter1
        union = union1

    iou = float(inter / union) if union > 0 else 1.0
    denom_f1 = int(pred.sum()) + int(gt.sum())
    f1 = float(2.0 * inter / denom_f1) if denom_f1 > 0 else 1.0
    return iou, f1


# ============================================================================
# 2. КЛАСИЧНІ МЕТОДИ ОБРОБКИ ЗОБРАЖЕНЬ (6 МЕТОДІВ ІЗ 6 КАТЕГОРІЙ)
# ============================================================================

# --- МЕТОД 1 (Категорія: Ретушування та шумоподавлення зі збереженням контурів) ---
def bilateral_filter(
    img: np.ndarray,
    ksize: int = 5,
    sigma_spatial: float = 2.0,
    sigma_range: float = 28.0
) -> np.ndarray:
    """
    Реалізує білатеральний фільтр вручну.
    Комбінує просторову гауссівську вагу w_s = exp(-(dx^2 + dy^2)/(2*sigma_s^2))
    та яскравісну гауссівську вагу w_r = exp(-(|I(p)-I(q)|^2)/(2*sigma_r^2)),
    що дозволяє ефективно пригнічувати шум зі збереженням чітких границь.
    """
    arr = validate_and_prepare_image(img).astype(np.float32)
    h, w = arr.shape
    if h < 2 or w < 2 or ksize <= 1:
        return arr.astype(np.uint8)

    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    sigma_s = max(float(sigma_spatial), 1e-3)
    sigma_r = max(float(sigma_range), 1e-3)

    padded = np.pad(arr, pad_width=pad, mode="reflect")
    accum_weighted = np.zeros((h, w), dtype=np.float32)
    accum_norm = np.zeros((h, w), dtype=np.float32)

    two_sigma_s2 = 2.0 * sigma_s * sigma_s
    two_sigma_r2 = 2.0 * sigma_r * sigma_r

    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            spatial_weight = math.exp(-(dx * dx + dy * dy) / two_sigma_s2)
            shifted = padded[pad + dy: pad + dy + h, pad + dx: pad + dx + w]
            intensity_diff = shifted - arr
            range_weight = np.exp(-(intensity_diff * intensity_diff) / two_sigma_r2)
            total_weight = spatial_weight * range_weight
            accum_weighted += shifted * total_weight
            accum_norm += total_weight

    result = accum_weighted / np.maximum(accum_norm, 1e-8)
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


# --- МЕТОД 2 (Категорія: Гістограмна корекція) ---
def _compute_clipped_cdf(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    """
    Допоміжна функція для CLAHE: будує гістограму тайла, обмежує її піки
    порогом clip_limit, рівномірно перерозподіляє надлишок і повертає таблицю CDF (0..255).
    """
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    num_pixels = float(tile.size)
    if num_pixels <= 0:
        return np.arange(256, dtype=np.float64)

    avg_bin = num_pixels / 256.0
    actual_limit = max(float(clip_limit), clip_limit * avg_bin)

    excess = np.maximum(hist - actual_limit, 0.0)
    total_excess = float(np.sum(excess))
    hist = np.minimum(hist, actual_limit)
    hist += total_excess / 256.0

    cdf = np.cumsum(hist)
    cdf_min = float(cdf[cdf > 0][0]) if np.any(cdf > 0) else 0.0
    denom = max(num_pixels - cdf_min, 1e-6)
    lut = np.clip((cdf - cdf_min) / denom * 255.0, 0.0, 255.0)
    return lut


def clahe_enhancement(
    img: np.ndarray,
    grid_size: int = 4,
    clip_limit: float = 2.5
) -> np.ndarray:
    """
    Реалізує адаптивне гістограмне вирівнювання з обмеженням контрасту (CLAHE) вручну.
    Зображення розбивається на сітку тайлів (grid_size x grid_size), для кожного тайла
    будується обмежена за контрастом функція розподілу (CDF), після чого значення
    пікселів обчислюються через білінійну інтерполяцію між центрами сусідніх тайлів.
    """
    arr = validate_and_prepare_image(img)
    h, w = arr.shape
    if h < 4 or w < 4 or grid_size <= 1:
        lut = _compute_clipped_cdf(arr, clip_limit)
        return np.round(lut[arr]).astype(np.uint8)

    gh = min(grid_size, h // 2)
    gw = min(grid_size, w // 2)
    luts = np.zeros((gh, gw, 256), dtype=np.float64)

    y_bounds = np.linspace(0, h, gh + 1, dtype=int)
    x_bounds = np.linspace(0, w, gw + 1, dtype=int)
    y_centers = 0.5 * (y_bounds[:-1] + y_bounds[1:] - 1)
    x_centers = 0.5 * (x_bounds[:-1] + x_bounds[1:] - 1)

    for ty in range(gh):
        for tx in range(gw):
            tile = arr[y_bounds[ty]:y_bounds[ty + 1], x_bounds[tx]:x_bounds[tx + 1]]
            luts[ty, tx] = _compute_clipped_cdf(tile, clip_limit)

    y_coords = np.arange(h, dtype=np.float64)
    x_coords = np.arange(w, dtype=np.float64)

    ty1 = np.clip(np.searchsorted(y_centers, y_coords, side="right") - 1, 0, gh - 1)
    ty2 = np.clip(ty1 + 1, 0, gh - 1)
    tx1 = np.clip(np.searchsorted(x_centers, x_coords, side="right") - 1, 0, gw - 1)
    tx2 = np.clip(tx1 + 1, 0, gw - 1)

    dy_denom = np.maximum(y_centers[ty2] - y_centers[ty1], 1e-6)
    dx_denom = np.maximum(x_centers[tx2] - x_centers[tx1], 1e-6)
    wy = np.where(ty1 == ty2, 0.0, np.clip((y_coords - y_centers[ty1]) / dy_denom, 0.0, 1.0))
    wx = np.where(tx1 == tx2, 0.0, np.clip((x_coords - x_centers[tx1]) / dx_denom, 0.0, 1.0))

    ty1_grid, tx1_grid = np.meshgrid(ty1, tx1, indexing="ij")
    ty2_grid, tx2_grid = np.meshgrid(ty2, tx2, indexing="ij")
    wy_grid, wx_grid = np.meshgrid(wy, wx, indexing="ij")

    v11 = luts[ty1_grid, tx1_grid, arr]
    v12 = luts[ty1_grid, tx2_grid, arr]
    v21 = luts[ty2_grid, tx1_grid, arr]
    v22 = luts[ty2_grid, tx2_grid, arr]

    top = (1.0 - wx_grid) * v11 + wx_grid * v12
    bottom = (1.0 - wx_grid) * v21 + wx_grid * v22
    interp = (1.0 - wy_grid) * top + wy_grid * bottom

    return np.clip(np.round(interp), 0, 255).astype(np.uint8)


# --- МЕТОД 3 (Категорія: Контурний аналіз) ---
def sobel_edge_detector(img: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Реалізує виділення контурів оператором Собеля вручну.
    Обчислює частинні похідні яскравості за допомогою ядер 3x3:
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    та модуль градієнта G = sqrt(Gx^2 + Gy^2).
    """
    arr = validate_and_prepare_image(img).astype(np.float32)
    h, w = arr.shape
    if h < 2 or w < 2:
        return np.zeros_like(arr, dtype=np.uint8)

    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)

    padded = np.pad(arr, pad_width=1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    gx = np.sum(windows * kx, axis=(-2, -1))
    gy = np.sum(windows * ky, axis=(-2, -1))
    mag = np.sqrt(gx * gx + gy * gy)

    if normalize:
        max_mag = float(np.max(mag))
        if max_mag > 1e-6:
            mag = (mag / max_mag) * 255.0

    return np.clip(np.round(mag), 0, 255).astype(np.uint8)


# --- МЕТОД 4 (Категорія: Сегментація зображень) ---
def otsu_threshold_value(img: np.ndarray) -> int:
    """
    Знаходить оптимальний глобальний поріг бінаризації за методом Оцу
    шляхом максимізації міжкласової дисперсії sigma_B^2(t).
    """
    arr = validate_and_prepare_image(img)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = float(arr.size)
    if total <= 0:
        return 128

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_total = float(mu[-1])

    best_thresh = 0
    max_sigma_b2 = -1.0

    for t in range(255):
        w0 = omega[t]
        w1 = 1.0 - w0
        if w0 <= 1e-9 or w1 <= 1e-9:
            continue
        mu0 = mu[t] / w0
        mu1 = (mu_total - mu[t]) / w1
        sigma_b2 = w0 * w1 * ((mu0 - mu1) ** 2)
        if sigma_b2 > max_sigma_b2:
            max_sigma_b2 = sigma_b2
            best_thresh = t

    return int(best_thresh)


def otsu_segmentation(
    img: np.ndarray,
    reconstruct_class_means: bool = False
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Виконує автоматичну сегментацію методом Оцу вручну.
    Повертає кортеж:
      (segmented_image, optimal_threshold, binary_mask),
    де binary_mask має значення {0, 255}, а segmented_image — або бінарна маска,
    або двокласова реконструкція з середніми рівнями класів mu0, mu1.
    """
    arr = validate_and_prepare_image(img)
    t_opt = otsu_threshold_value(arr)
    mask_fg = arr > t_opt
    binary_mask = np.where(mask_fg, 255, 0).astype(np.uint8)

    if not reconstruct_class_means:
        return binary_mask, t_opt, binary_mask

    bg_vals = arr[~mask_fg]
    fg_vals = arr[mask_fg]
    mean_bg = int(round(float(np.mean(bg_vals)))) if bg_vals.size > 0 else 0
    mean_fg = int(round(float(np.mean(fg_vals)))) if fg_vals.size > 0 else 255
    reconstructed = np.where(mask_fg, mean_fg, mean_bg).astype(np.uint8)
    return reconstructed, t_opt, binary_mask


# --- МЕТОД 5 (Категорія: Морфологічна обробка) ---
def grayscale_erosion(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Виконує сірошкальну морфологічну ерозію (мінімум у межах плоского
    структурувального елемента ksize x ksize).
    """
    arr = validate_and_prepare_image(img)
    h, w = arr.shape
    if h < 2 or w < 2 or ksize <= 1:
        return arr.copy()
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    return np.min(windows, axis=(-2, -1)).astype(np.uint8)


def grayscale_dilation(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Виконує сірошкальну морфологічну дилатацію (максимум у межах плоского
    структурувального елемента ksize x ksize).
    """
    arr = validate_and_prepare_image(img)
    h, w = arr.shape
    if h < 2 or w < 2 or ksize <= 1:
        return arr.copy()
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    return np.max(windows, axis=(-2, -1)).astype(np.uint8)


def morphological_filter(
    img: np.ndarray,
    ksize: int = 3,
    operation: str = "open_close_smooth"
) -> np.ndarray:
    """
    Реалізує операції математичної морфології вручну:
    - 'erosion': ерозія
    - 'dilation': дилатація
    - 'opening': відкриття (дилатація від ерозії)
    - 'closing': закриття (ерозія від дилатації)
    - 'gradient': морфологічний градієнт Беше (дилатація мінус ерозія)
    - 'open_close_smooth': півсума відкриття та закриття 0.5 * (gamma(I) + phi(I)),
      яка симетрично видаляє як світлі, так і темні імпульсні та структурні дефекти.
    """
    arr = validate_and_prepare_image(img)
    if operation == "erosion":
        return grayscale_erosion(arr, ksize)
    if operation == "dilation":
        return grayscale_dilation(arr, ksize)

    opened = grayscale_dilation(grayscale_erosion(arr, ksize), ksize)
    if operation == "opening":
        return opened

    closed = grayscale_erosion(grayscale_dilation(arr, ksize), ksize)
    if operation == "closing":
        return closed

    if operation == "gradient":
        dil = grayscale_dilation(arr, ksize).astype(np.int32)
        ero = grayscale_erosion(arr, ksize).astype(np.int32)
        return np.clip(dil - ero, 0, 255).astype(np.uint8)

    # За замовчуванням: симетричне морфологічне згладжування відкриттям-закриттям
    smoothed = 0.5 * (opened.astype(np.float32) + closed.astype(np.float32))
    return np.clip(np.round(smoothed), 0, 255).astype(np.uint8)


# --- МЕТОД 6 (Категорія: Аналіз текстур — LBP та GLCM) ---
def compute_lbp_image(img: np.ndarray) -> np.ndarray:
    """
    Обчислює карту локальних бінарних шаблонів (LBP — Local Binary Patterns) вручну.
    Для кожного пікселя (y, x) порівнює яскравість 8 сусідів за годинниковою стрілкою
    починаючи з верхнього лівого кута:
    LBP(y, x) = sum_{p=0..7} s(I_p - I_c) * 2^p, де s(z) = 1 при z >= 0, інакше 0.
    """
    arr = validate_and_prepare_image(img)
    h, w = arr.shape
    if h < 2 or w < 2:
        return np.zeros_like(arr, dtype=np.uint8)

    padded = np.pad(arr, pad_width=1, mode="edge")
    center = padded[1:h + 1, 1:w + 1]

    # 8 сусідів за годинниковою стрілкою: TL, T, TR, R, BR, B, BL, L
    offsets = [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (2, 2), (2, 1),
        (2, 0), (1, 0)
    ]

    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit_idx, (dy, dx) in enumerate(offsets):
        neighbor = padded[dy:dy + h, dx:dx + w]
        bit_mask = (neighbor >= center).astype(np.uint8)
        lbp |= (bit_mask << bit_idx)

    return lbp


def compute_glcm_features(
    img: np.ndarray,
    levels: int = 16,
    distance: int = 1
) -> Dict[str, float]:
    """
    Будує симетричну нормалізовану матрицю спільної появи рівнів сірого (GLCM)
    у горизонтальному напрямку (dx = distance, dy = 0) та обчислює 4 текстурні
    ознаки Хараліка: контраст, гомогенність (IDM), енергію (ASM) та кореляцію.
    """
    arr = validate_and_prepare_image(img)
    h, w = arr.shape
    quant = np.clip((arr.astype(np.int32) * levels) // 256, 0, levels - 1)

    if w <= distance:
        return {
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 1.0,
            "glcm_energy": 1.0,
            "glcm_correlation": 0.0
        }

    left = quant[:, :-distance].ravel()
    right = quant[:, distance:].ravel()

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (left, right), 1.0)
    np.add.at(glcm, (right, left), 1.0)

    total = float(np.sum(glcm))
    if total > 0:
        glcm /= total

    i_idx, j_idx = np.meshgrid(
        np.arange(levels, dtype=np.float64),
        np.arange(levels, dtype=np.float64),
        indexing="ij"
    )
    diff = i_idx - j_idx

    contrast = float(np.sum((diff * diff) * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(diff))))
    energy = float(np.sum(glcm * glcm))

    mu_i = float(np.sum(i_idx * glcm))
    mu_j = float(np.sum(j_idx * glcm))
    std_i = math.sqrt(float(np.sum(((i_idx - mu_i) ** 2) * glcm)))
    std_j = math.sqrt(float(np.sum(((j_idx - mu_j) ** 2) * glcm)))
    if std_i > 1e-9 and std_j > 1e-9:
        correlation = float(
            np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (std_i * std_j)
        )
    else:
        correlation = 0.0

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation
    }


# ============================================================================
# 3. ГЕНЕРАЦІЯ ТА ЗАВАНТАЖЕННЯ ВИБІРКИ ЗОБРАЖЕНЬ (100 ЗОБРАЖЕНЬ, 5 КЛАСІВ)
# ============================================================================

def _create_base_scene(
    category: str,
    variant_idx: int,
    size: Tuple[int, int],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Створює чисте еталонне зображення (clean), бінарну маску об'єктів (gt_mask)
    та еталонну карту контурів (gt_edges) для заданого класу об'єктів.
    """
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    if category == "aerial_runway":
        # Аерокосмічний знімок аеродрому та злітно-посадкової смуги
        bg = 55.0 + 12.0 * np.sin((x + variant_idx * 5.0) / 18.0) * np.cos(y / 20.0)
        angle = -0.35 + 0.04 * (variant_idx % 15)
        cx, cy = w * 0.5, h * 0.5
        xr = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
        yr = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)

        runway_width = 12.0 + (variant_idx % 4) * 2.0
        mask_runway = (np.abs(xr) <= runway_width) & (np.abs(yr) <= h * 0.42)
        mask_taxiway = (np.abs(yr - 15.0) <= 5.0) & (xr >= 0.0) & (xr <= 38.0)
        mask_apron = (np.abs(xr - 34.0) <= 10.0) & (np.abs(yr) <= 22.0)
        gt_mask = mask_runway | mask_taxiway | mask_apron

        clean = bg.copy()
        clean[gt_mask] = 195.0
        # Розмітка смуги
        centerline = (np.abs(xr) <= 1.2) & (np.abs(yr) <= h * 0.38) & ((np.floor(yr / 6.0) % 2) == 0)
        clean[centerline] = 240.0

    elif category == "urban_blocks":
        # Аерофотознімок міських кварталів та будівель
        clean = np.full((h, w), 62.0, dtype=np.float32)
        gt_mask = np.zeros((h, w), dtype=bool)
        step = 28 + (variant_idx % 3) * 4
        offset_x = (variant_idx * 3) % 10
        offset_y = (variant_idx * 5) % 10

        for r0 in range(10 + offset_y, h - 18, step):
            for c0 in range(10 + offset_x, w - 18, step):
                bh = 14 + int(rng.integers(0, 6))
                bw = 14 + int(rng.integers(0, 6))
                r1 = min(h - 4, r0 + bh)
                c1 = min(w - 4, c0 + bw)
                gt_mask[r0:r1, c0:c1] = True
                roof_val = 175.0 + float(rng.integers(-20, 35))
                clean[r0:r1, c0:c1] = roof_val

    elif category == "biometric_fingerprint":
        # Біометричний малюнок папілярних ліній та концентричних дуг
        cx = w * (0.45 + 0.01 * (variant_idx % 10))
        cy = h * (0.50 + 0.01 * ((variant_idx // 2) % 10))
        dist = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.15) ** 2)
        theta = np.arctan2(y - cy, x - cx)
        freq = 0.52 + 0.02 * (variant_idx % 5)
        ridges = np.sin(dist * freq + 1.5 * np.sin(theta))
        envelope = np.exp(-((x - w * 0.5) ** 2 + (y - h * 0.5) ** 2) / (2.0 * (w * 0.42) ** 2))
        gt_mask = (ridges > 0.0) & (envelope > 0.22)
        clean = 45.0 + 170.0 * ((0.5 * (ridges + 1.0)) * envelope)

    elif category == "satellite_river_forest":
        # Супутниковий знімок річкового русла серед лісового масиву
        forest_tex = 58.0 + 10.0 * np.sin(x / 3.0) * np.cos(y / 3.0)
        meander = (
            w * 0.5
            + (18.0 + (variant_idx % 5) * 2.0)
            * np.sin(y / (18.0 + (variant_idx % 4) * 2.0) + variant_idx * 0.4)
        )
        river_width = 11.0 + (variant_idx % 4) * 1.5
        gt_mask = np.abs(x - meander) <= river_width
        sandbank = (np.abs(x - meander) > river_width) & (np.abs(x - meander) <= river_width + 4.0)

        clean = forest_tex.copy()
        clean[sandbank] = 145.0
        clean[gt_mask] = 210.0

    else:
        # Історичний рукопис / геометричні символи на пергаменті
        clean = 68.0 + 8.0 * np.cos(x / 25.0) + 8.0 * np.sin(y / 25.0)
        gt_mask = np.zeros((h, w), dtype=bool)
        num_lines = 4 + (variant_idx % 3)
        for line_i in range(num_lines):
            ry = 18 + line_i * 24
            if ry + 10 >= h:
                break
            for seg_j in range(4):
                cx0 = 14 + seg_j * 26 + ((line_i + variant_idx) % 5)
                cw = 14 + ((seg_j + variant_idx) % 6)
                if cx0 + cw < w - 6:
                    gt_mask[ry:ry + 6, cx0:cx0 + cw] = True
                    if (seg_j + line_i) % 2 == 0 and ry + 14 < h:
                        gt_mask[ry:ry + 14, cx0:cx0 + 5] = True
        clean[gt_mask] = 215.0

    clean_u8 = np.clip(np.round(clean), 0, 255).astype(np.uint8)
    mask_u8 = np.where(gt_mask, 255, 0).astype(np.uint8)
    edges_u8 = np.where(sobel_edge_detector(mask_u8, normalize=True) >= 64, 255, 0).astype(np.uint8)
    return clean_u8, mask_u8, edges_u8


def generate_stem_dataset(
    dataset_dir: Path,
    num_images: int = 100,
    image_size: Tuple[int, int] = (128, 128),
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Генерує репрезентативну вибірку зі 100 зображень (5 класів об'єктів х 4 режими
    спотворень х 5 варіацій) разом із чистими еталонами та масками для обчислення
    PSNR, MSE, SSIM, IoU та F1.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = dataset_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        "aerial_runway",
        "urban_blocks",
        "biometric_fingerprint",
        "satellite_river_forest",
        "historical_manuscript"
    ]
    conditions = [
        "low_noise",
        "gaussian_noise",
        "impulse_noise",
        "low_contrast"
    ]

    rng = np.random.default_rng(seed)
    manifest: List[Dict[str, Any]] = []

    for idx in range(num_images):
        cat = categories[idx % len(categories)]
        cond = conditions[(idx // len(categories)) % len(conditions)]
        clean_u8, mask_u8, edges_u8 = _create_base_scene(cat, idx, image_size, rng)

        clean_f = clean_u8.astype(np.float32)
        h, w = clean_u8.shape

        if cond == "low_noise":
            observed = clean_f + rng.normal(0.0, 6.0, size=(h, w))
        elif cond == "gaussian_noise":
            observed = clean_f + rng.normal(0.0, 22.0, size=(h, w))
        elif cond == "impulse_noise":
            observed = clean_f.copy()
            rnd = rng.random((h, w))
            observed[rnd < 0.04] = 0.0
            observed[(rnd >= 0.04) & (rnd < 0.08)] = 255.0
        else:
            # Знижений контраст + плавна нерівномірність освітлення
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            illum = 15.0 * np.sin(x_grid / 40.0) - 10.0 * np.cos(y_grid / 40.0)
            observed = (clean_f - 128.0) * 0.45 + 125.0 + illum + rng.normal(0.0, 5.0, size=(h, w))

        obs_u8 = np.clip(np.round(observed), 0, 255).astype(np.uint8)

        img_name = f"img_{idx:03d}_{cat}_{cond}.png"
        img_path = dataset_dir / img_name
        ref_clean_path = ref_dir / f"img_{idx:03d}_clean.png"
        ref_mask_path = ref_dir / f"img_{idx:03d}_mask.png"
        ref_edge_path = ref_dir / f"img_{idx:03d}_edges.png"

        Image.fromarray(obs_u8, mode="L").save(img_path)
        Image.fromarray(clean_u8, mode="L").save(ref_clean_path)
        Image.fromarray(mask_u8, mode="L").save(ref_mask_path)
        Image.fromarray(edges_u8, mode="L").save(ref_edge_path)

        manifest.append({
            "index": idx,
            "filename": img_name,
            "category": cat,
            "condition": cond,
            "image_path": f"stem_dataset/{img_name}",
            "ref_clean_path": f"stem_dataset/references/img_{idx:03d}_clean.png",
            "ref_mask_path": f"stem_dataset/references/img_{idx:03d}_mask.png",
            "ref_edge_path": f"stem_dataset/references/img_{idx:03d}_edges.png"
        })

    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_grayscale_image_safe(path: Path) -> Optional[np.ndarray]:
    """
    Безпечно завантажує зображення з диску та конвертує його у 2D-масив uint8.
    При пошкодженому або порожньому файлі повертає None без аварійного завершення (C2).
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with Image.open(path) as img:
            arr = np.array(img)
            return validate_and_prepare_image(arr)
    except (OSError, ValueError, UnidentifiedImageError):
        return None


# ============================================================================
# 4. ЗАПУСК МЕТОДІВ, ОБЧИСЛЕННЯ МЕТРИК ТА ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ============================================================================

def run_single_method(
    method_name: str,
    img: np.ndarray,
    method_cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Виконує вказаний метод обробки зображень з параметрами із конфігурації
    та повертає оброблене зображення і додаткові діагностичні показники.
    """
    extra_info: Dict[str, float] = {}

    if method_name == "bilateral":
        out = bilateral_filter(
            img,
            ksize=int(method_cfg.get("ksize", 5)),
            sigma_spatial=float(method_cfg.get("sigma_spatial", 2.0)),
            sigma_range=float(method_cfg.get("sigma_range", 28.0))
        )
    elif method_name == "clahe":
        out = clahe_enhancement(
            img,
            grid_size=int(method_cfg.get("grid_size", 4)),
            clip_limit=float(method_cfg.get("clip_limit", 2.5))
        )
    elif method_name == "sobel":
        out = sobel_edge_detector(
            img,
            normalize=bool(method_cfg.get("normalize", True))
        )
    elif method_name == "otsu":
        reconstruct = bool(method_cfg.get("reconstruct_class_means", True))
        out, t_opt, binary_mask = otsu_segmentation(img, reconstruct_class_means=reconstruct)
        extra_info["otsu_threshold"] = float(t_opt)
        extra_info["foreground_ratio"] = float(np.mean(binary_mask > 0))
    elif method_name == "morphology":
        out = morphological_filter(
            img,
            ksize=int(method_cfg.get("ksize", 3)),
            operation=str(method_cfg.get("operation", "open_close_smooth"))
        )
    elif method_name == "lbp_glcm":
        out = compute_lbp_image(img)
        glcm_feats = compute_glcm_features(
            img,
            levels=int(method_cfg.get("glcm_levels", 16)),
            distance=int(method_cfg.get("distance", 1))
        )
        extra_info.update(glcm_feats)
    else:
        raise ValueError(f"Невідомий метод обробки: {method_name}")

    return out, extra_info


def execute_stem_benchmark(
    config_path: Path,
    selected_method: str = "all",
    input_dir_override: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Виконує повний цикл дослідження STEM-проєкту:
    1. Завантажує конфігурацію та за потреби генерує вибірку зі 100 зображень.
    2. Проганяє обрані класичні методи на всіх зображеннях вибірки.
    3. Зберігає оброблені зображення, обчислює метрики MSE, PSNR, SSIM, ентропію,
       IoU/F1 та час виконання і записує їх у CSV та JSON.
    4. Будує зведені таблиці й графіки порівняння методів.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = config_path.parent
    dataset_cfg = config.get("dataset", {})
    dataset_dir = input_dir_override or (base_dir / dataset_cfg.get("input_dir", "stem_dataset"))
    num_images = int(dataset_cfg.get("num_images", 100))
    img_size = tuple(dataset_cfg.get("image_size", [128, 128]))
    seed = int(config.get("seed", 42))

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = generate_stem_dataset(
            dataset_dir=dataset_dir,
            num_images=num_images,
            image_size=(int(img_size[0]), int(img_size[1])),
            seed=seed
        )
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    out_cfg = config.get("output", {})
    out_dir = output_dir_override or (base_dir / out_cfg.get("output_dir", "stem_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    methods_cfg: Dict[str, Dict[str, Any]] = config.get("methods", {})
    if selected_method != "all":
        if selected_method not in methods_cfg:
            raise ValueError(f"Метод '{selected_method}' відсутній у конфігурації.")
        active_methods = [selected_method]
    else:
        active_methods = [m for m, mcfg in methods_cfg.items() if mcfg.get("enabled", True)]

    all_records: List[Dict[str, Any]] = []

    for m_name in active_methods:
        (out_dir / m_name).mkdir(parents=True, exist_ok=True)

    for item in manifest:
        img_path = Path(item["image_path"])
        if not img_path.is_absolute() and not img_path.exists():
            img_path = base_dir / img_path

        obs_img = load_grayscale_image_safe(img_path)
        if obs_img is None:
            continue

        ref_clean = load_grayscale_image_safe(Path(item["ref_clean_path"]))
        if ref_clean is None:
            ref_clean = obs_img

        ref_mask = load_grayscale_image_safe(Path(item["ref_mask_path"]))
        ref_edge = load_grayscale_image_safe(Path(item["ref_edge_path"]))

        init_mse = calc_mse(ref_clean, obs_img)
        init_psnr = calc_psnr(ref_clean, obs_img)
        init_ssim = calc_ssim(ref_clean, obs_img)

        for m_name in active_methods:
            m_cfg = methods_cfg[m_name]
            t0 = time.perf_counter()
            proc_img, extra = run_single_method(m_name, obs_img, m_cfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Зберігаємо перші 20 зображень кожного методу для візуальної верифікації
            if int(item["index"]) < 20:
                save_path = out_dir / m_name / item["filename"]
                Image.fromarray(proc_img, mode="L").save(save_path)

            # Визначаємо цільовий еталон залежно від призначення методу
            if m_name == "sobel" and ref_edge is not None:
                mse_val = calc_mse(ref_edge, proc_img)
                psnr_val = calc_psnr(ref_edge, proc_img)
                ssim_val = calc_ssim(ref_edge, proc_img)
                thresh_e = int(m_cfg.get("edge_threshold", 60))
                iou_val, f1_val = calc_iou_and_f1(
                    np.where(proc_img >= thresh_e, 255, 0).astype(np.uint8),
                    ref_edge
                )
            elif m_name == "otsu" and ref_mask is not None:
                mse_val = calc_mse(ref_clean, proc_img)
                psnr_val = calc_psnr(ref_clean, proc_img)
                ssim_val = calc_ssim(ref_clean, proc_img)
                _, _, bin_mask = otsu_segmentation(obs_img, reconstruct_class_means=False)
                iou_val, f1_val = calc_iou_and_f1(bin_mask, ref_mask)
            else:
                mse_val = calc_mse(ref_clean, proc_img)
                psnr_val = calc_psnr(ref_clean, proc_img)
                ssim_val = calc_ssim(ref_clean, proc_img)
                if ref_mask is not None:
                    _, _, bin_mask = otsu_segmentation(proc_img, reconstruct_class_means=False)
                    iou_val, f1_val = calc_iou_and_f1(bin_mask, ref_mask)
                else:
                    iou_val, f1_val = 0.0, 0.0

            entropy_val = calc_shannon_entropy(proc_img)
            contrast_std = float(np.std(proc_img.astype(np.float32)))

            record: Dict[str, Any] = {
                "image_index": item["index"],
                "filename": item["filename"],
                "scene_category": item["category"],
                "condition": item["condition"],
                "method": m_name,
                "method_category": m_cfg.get("category", ""),
                "input_mse": round(init_mse, 3),
                "input_psnr_db": round(init_psnr, 3),
                "input_ssim": round(init_ssim, 4),
                "mse": round(mse_val, 3),
                "psnr_db": round(psnr_val, 3),
                "ssim": round(ssim_val, 4),
                "iou": round(iou_val, 4),
                "f1_score": round(f1_val, 4),
                "entropy_bits": round(entropy_val, 4),
                "std_contrast": round(contrast_std, 3),
                "time_ms": round(elapsed_ms, 3)
            }
            for k_ext, v_ext in extra.items():
                record[k_ext] = round(float(v_ext), 4)
            all_records.append(record)

    # Збереження детальних метрик у CSV та JSON
    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    summary_csv_path = out_dir / "summary_by_method.csv"

    if all_records:
        all_keys: List[str] = []
        for rec in all_records:
            for k in rec.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_records)

    summary_by_method = _aggregate_summary(all_records, active_methods, methods_cfg)

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        if summary_by_method:
            writer = csv.DictWriter(f, fieldnames=list(summary_by_method[0].keys()))
            writer.writeheader()
            writer.writerows(summary_by_method)

    full_report_data = {
        "project": config.get("project_name"),
        "author": config.get("author"),
        "num_images_processed": len(manifest),
        "methods_evaluated": active_methods,
        "summary_by_method": summary_by_method,
        "records": all_records
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2, ensure_ascii=False)

    if generate_plots and all_records:
        generate_report_figures(dataset_dir, out_dir, manifest, all_records, summary_by_method)

    return full_report_data


def _aggregate_summary(
    records: List[Dict[str, Any]],
    active_methods: List[str],
    methods_cfg: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Обчислює середні показники якості по кожному методу загалом та за умовами зйомки.
    """
    summary: List[Dict[str, Any]] = []
    for m_name in active_methods:
        m_recs = [r for r in records if r["method"] == m_name]
        if not m_recs:
            continue

        def avg(key: str, subset: Optional[List[Dict[str, Any]]] = None) -> float:
            target = subset if subset is not None else m_recs
            vals = [float(r[key]) for r in target if key in r]
            return round(float(np.mean(vals)), 3) if vals else 0.0

        gauss_recs = [r for r in m_recs if r["condition"] == "gaussian_noise"]
        imp_recs = [r for r in m_recs if r["condition"] == "impulse_noise"]
        lowc_recs = [r for r in m_recs if r["condition"] == "low_contrast"]

        summary.append({
            "method": m_name,
            "category": methods_cfg[m_name].get("category", ""),
            "mean_mse": avg("mse"),
            "mean_psnr_db": avg("psnr_db"),
            "mean_ssim": round(avg("ssim"), 4),
            "mean_iou": round(avg("iou"), 4),
            "mean_f1": round(avg("f1_score"), 4),
            "mean_entropy": round(avg("entropy_bits"), 3),
            "mean_contrast_std": avg("std_contrast"),
            "psnr_gaussian_db": avg("psnr_db", gauss_recs),
            "psnr_impulse_db": avg("psnr_db", imp_recs),
            "psnr_low_contrast_db": avg("psnr_db", lowc_recs),
            "mean_time_ms": avg("time_ms")
        })
    return summary


# ============================================================================
# 5. ПОБУДОВА ІЛЮСТРАЦІЙ ТА ГРАФІКІВ ДЛЯ ЗВІТУ
# ============================================================================

def generate_report_figures(
    dataset_dir: Path,
    out_dir: Path,
    manifest: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    summary_by_method: List[Dict[str, Any]]
) -> None:
    """
    Створює наочні графічні ілюстрації для звіту STEM-проєкту:
    1) stem_fig1_dataset_samples.png — зразки 5 класів зображень та 4 режимів спотворень;
    2) stem_fig2_methods_visual.png — візуальне порівняння дії всіх 6 методів до/після;
    3) stem_fig3_metrics_charts.png — кількісне порівняння PSNR, SSIM, IoU/F1 та часу обробки;
    4) stem_fig4_conditions_heatmap.png — ефективність методів за різними умовами зйомки.
    """
    # --- Рисунок 1: Приклади зображень вибірки з 5 категорій ---
    fig1, axes1 = plt.subplots(2, 5, figsize=(13, 5.5))
    cat_titles = {
        "aerial_runway": "Аеродром (ЗПС)",
        "urban_blocks": "Міська забудова",
        "biometric_fingerprint": "Відбиток пальця",
        "satellite_river_forest": "Русло річки та ліс",
        "historical_manuscript": "Стародрук / рукопис"
    }
    categories = list(cat_titles.keys())
    for col_i, cat in enumerate(categories):
        # Рядок 0: малий шум, Рядок 1: сильний шум або низький контраст
        item_clean = next(it for it in manifest if it["category"] == cat and it["condition"] == "low_noise")
        item_noisy = next(it for it in manifest if it["category"] == cat and it["condition"] == "gaussian_noise")

        img_c = load_grayscale_image_safe(Path(item_clean["image_path"]))
        img_n = load_grayscale_image_safe(Path(item_noisy["image_path"]))

        axes1[0, col_i].imshow(img_c, cmap="gray", vmin=0, vmax=255)
        axes1[0, col_i].set_title(f"{cat_titles[cat]}\n(Малий шум, σ=6)", fontsize=9)
        axes1[0, col_i].axis("off")

        axes1[1, col_i].imshow(img_n, cmap="gray", vmin=0, vmax=255)
        axes1[1, col_i].set_title(f"{cat_titles[cat]}\n(Гаусів шум, σ=22)", fontsize=9)
        axes1[1, col_i].axis("off")

    fig1.suptitle("Рисунок 1 — Репрезентативні зразки 5 класів об'єктів створеної вибірки (N = 100 зображень, 128×128)", fontsize=11)
    fig1.tight_layout()
    fig1.savefig(out_dir / "stem_fig1_dataset_samples.png", dpi=160)
    plt.close(fig1)

    # --- Рисунок 2: Візуальне порівняння роботи 6 методів на 3 характерних сценах ---
    sample_indices = [5, 12, 16]  # Гаусів шум, імпульсний шум, низький контраст
    fig2, axes2 = plt.subplots(len(sample_indices), 7, figsize=(15, 6.8))
    method_labels = [
        ("Вхідне\nзображення", None),
        ("1. Білатеральний\nфільтр 5×5", "bilateral"),
        ("2. Гістограмна\nкорекція CLAHE", "clahe"),
        ("3. Контури\nСобеля", "sobel"),
        ("4. Сегментація\nметодом Оцу", "otsu"),
        ("5. Морфологічне\nвідкр.-закр. 3×3", "morphology"),
        ("6. Текстурна\nкарта LBP", "lbp_glcm")
    ]

    for row_i, s_idx in enumerate(sample_indices):
        item = manifest[s_idx]
        obs_img = load_grayscale_image_safe(Path(item["image_path"]))
        ref_clean = load_grayscale_image_safe(Path(item["ref_clean_path"]))
        init_p = calc_psnr(ref_clean, obs_img)

        axes2[row_i, 0].imshow(obs_img, cmap="gray", vmin=0, vmax=255)
        axes2[row_i, 0].set_title(
            f"{item['category'].split('_')[0]} ({item['condition']})\nPSNR={init_p:.1f} дБ",
            fontsize=8
        )
        axes2[row_i, 0].axis("off")

        for col_i, (m_title, m_key) in enumerate(method_labels[1:], start=1):
            proc_img, _ = run_single_method(m_key, obs_img, {})
            rec_match = next(
                (r for r in records if r["image_index"] == s_idx and r["method"] == m_key),
                None
            )
            metric_str = ""
            if rec_match is not None:
                if m_key in ("sobel", "otsu"):
                    metric_str = f"F1={rec_match['f1_score']:.2f}, IoU={rec_match['iou']:.2f}"
                elif m_key == "lbp_glcm":
                    metric_str = f"H={rec_match['entropy_bits']:.2f} біт"
                else:
                    metric_str = f"PSNR={rec_match['psnr_db']:.1f} дБ, SSIM={rec_match['ssim']:.2f}"

            axes2[row_i, col_i].imshow(proc_img, cmap="gray", vmin=0, vmax=255)
            axes2[row_i, col_i].set_title(f"{m_title if row_i == 0 else ''}\n{metric_str}".strip(), fontsize=8)
            axes2[row_i, col_i].axis("off")

    fig2.suptitle("Рисунок 2 — Порівняння результатів обробки зображень 6 реалізованими класичними методами", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(out_dir / "stem_fig2_methods_visual.png", dpi=160)
    plt.close(fig2)

    # --- Рисунок 3: Стовпчикові діаграми метрик (PSNR, SSIM, F1/IoU, Час виконання) ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(12.5, 8.2))
    names = [s["method"] for s in summary_by_method]
    psnrs = [s["mean_psnr_db"] for s in summary_by_method]
    ssims = [s["mean_ssim"] for s in summary_by_method]
    f1s = [s["mean_f1"] for s in summary_by_method]
    ious = [s["mean_iou"] for s in summary_by_method]
    times = [s["mean_time_ms"] for s in summary_by_method]

    colors = ["#2b5c8f", "#388e3c", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]

    bars1 = axes3[0, 0].bar(names, psnrs, color=colors[:len(names)], edgecolor="black")
    axes3[0, 0].set_title("Середній PSNR (дБ) на вибірці N=100")
    axes3[0, 0].set_ylabel("PSNR, дБ")
    axes3[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars1:
        axes3[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}", ha="center", fontsize=8)

    bars2 = axes3[0, 1].bar(names, ssims, color=colors[:len(names)], edgecolor="black")
    axes3[0, 1].set_title("Середній індекс структурної подібності SSIM")
    axes3[0, 1].set_ylabel("SSIM")
    axes3[0, 1].set_ylim(0, 1.05)
    axes3[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars2:
        axes3[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", fontsize=8)

    x_pos = np.arange(len(names))
    axes3[1, 0].bar(x_pos - 0.18, ious, width=0.35, label="IoU (Жаккар)", color="#4daf4a", edgecolor="black")
    axes3[1, 0].bar(x_pos + 0.18, f1s, width=0.35, label="F1-міра", color="#ff7f00", edgecolor="black")
    axes3[1, 0].set_xticks(x_pos)
    axes3[1, 0].set_xticklabels(names)
    axes3[1, 0].set_ylim(0, 1.05)
    axes3[1, 0].set_title("Точність виділення структур та сегментації (IoU та F1)")
    axes3[1, 0].legend()
    axes3[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    bars4 = axes3[1, 1].bar(names, times, color="#5c6bc0", edgecolor="black")
    axes3[1, 1].set_title("Середній час обробки 1 зображення 128×128 (мс)")
    axes3[1, 1].set_ylabel("Час, мс")
    axes3[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars4:
        axes3[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", fontsize=8)

    fig3.suptitle("Рисунок 3 — Кількісне порівняння ефективності та швидкодії досліджуваних методів", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(out_dir / "stem_fig3_metrics_charts.png", dpi=160)
    plt.close(fig3)

    # --- Рисунок 4: Залежність PSNR від типу спотворення (умов зйомки) ---
    fig4, ax4 = plt.subplots(figsize=(10, 4.8))
    cond_keys = ["low_noise", "gaussian_noise", "impulse_noise", "low_contrast"]
    cond_labels = ["Малий шум (σ=6)", "Гаусів шум (σ=22)", "Імпульсний шум (8%)", "Низький контраст"]
    x_cond = np.arange(len(cond_keys))
    width_c = 0.13

    for idx_m, m_name in enumerate(names):
        vals_c = []
        for ck in cond_keys:
            sub = [r["psnr_db"] for r in records if r["method"] == m_name and r["condition"] == ck]
            vals_c.append(float(np.mean(sub)) if sub else 0.0)
        offset = (idx_m - len(names) / 2.0 + 0.5) * width_c
        ax4.bar(x_cond + offset, vals_c, width=width_c, label=m_name, color=colors[idx_m % len(colors)], edgecolor="black")

    ax4.set_xticks(x_cond)
    ax4.set_xticklabels(cond_labels)
    ax4.set_ylabel("PSNR, дБ")
    ax4.set_title("Рисунок 4 — Порівняння стійкості методів (PSNR, дБ) за різних типів зашумлення та контрасту")
    ax4.legend(ncol=3, fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig(out_dir / "stem_fig4_conditions_heatmap.png", dpi=160)
    plt.close(fig4)


def print_summary_table(report_data: Dict[str, Any]) -> None:
    """
    Виводить зведену таблицю результатів виконання STEM-проєкту у консоль.
    """
    print("=" * 108)
    print(f"ПРОЄКТ: {report_data.get('project')}")
    print(f"АВТОР:  {report_data.get('author')}")
    print(f"ОБРОБЛЕНО ЗОБРАЖЕНЬ: {report_data.get('num_images_processed')}")
    print("=" * 108)
    header = (
        f"{'Метод':<12} | {'Категорія':<30} | {'MSE':>7} | {'PSNR,дБ':>7} | "
        f"{'SSIM':>6} | {'IoU':>6} | {'F1':>6} | {'H,біт':>6} | {'Час,мс':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in report_data.get("summary_by_method", []):
        print(
            f"{row['method']:<12} | {row['category']:<30} | "
            f"{row['mean_mse']:>7.1f} | {row['mean_psnr_db']:>7.2f} | "
            f"{row['mean_ssim']:>6.3f} | {row['mean_iou']:>6.3f} | "
            f"{row['mean_f1']:>6.3f} | {row['mean_entropy']:>6.2f} | "
            f"{row['mean_time_ms']:>7.2f}"
        )
    print("=" * 108)


def parse_cli_args() -> argparse.Namespace:
    """
    Парсер аргументів командного рядка для керування запуском STEM-проєкту без зміни коду.
    """
    parser = argparse.ArgumentParser(
        description="STEM-проєкт: Класичні методи фільтрації, сегментації та аналізу текстур."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="stem_config.json",
        help="Шлях до JSON-файлу конфігурації параметрів методів."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "bilateral", "clahe", "sobel", "otsu", "morphology", "lbp_glcm"],
        help="Вибір конкретного методу обробки або 'all' для запуску всіх методів."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Шлях до директорії з вибіркою вхідних зображень."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Шлях до директорії для збереження результатів та метрик."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Вимкнути генерацію підсумкових графіків."
    )
    return parser.parse_args()


def main() -> None:
    """
    Точка входу програми при запуску з командного рядка.
    """
    args = parse_cli_args()
    cfg_path = Path(args.config).resolve()
    in_dir = Path(args.input_dir).resolve() if args.input_dir else None
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None

    report_data = execute_stem_benchmark(
        config_path=cfg_path,
        selected_method=args.method,
        input_dir_override=in_dir,
        output_dir_override=out_dir,
        generate_plots=not args.no_plots
    )
    print_summary_table(report_data)


if __name__ == "__main__":
    main()
