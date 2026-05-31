import numpy as np
import scipy.io as sio
import pickle
from pathlib import Path
from itertools import combinations
from scipy.optimize import least_squares


# ═══════════════════════════════════════════════════════════════════════════
# 1. Geometric RANSAC Anchor — C(18,3) = 816 전수 탐색
# ═══════════════════════════════════════════════════════════════════════════

_ALL_COMBS = list(combinations(range(18), 3))
_COMBS = [_ALL_COMBS[i] for i in np.random.RandomState(42).permutation(816)]


def multilateration_3bs(p1, p2, p3, d1, d2, d3):
    """3개 BS 좌표와 거리로 2D 위치 추정."""
    A = np.array([
        [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
        [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])],
    ])
    B = np.array([
        d1**2 - d2**2 + p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2,
        d1**2 - d3**2 + p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2,
    ])
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None


def geometric_ransac_anchor(d, p_bs, noise_margin=3.0):
    """Geometric RANSAC — 816 전수 탐색. Returns: (pos, n_inliers)."""
    best_pos = None
    best_inliers = np.array([], dtype=int)
    best_violation_count = 999

    for bs_idx in _COMBS:
        pos = multilateration_3bs(
            p_bs[:, bs_idx[0]], p_bs[:, bs_idx[1]], p_bs[:, bs_idx[2]],
            d[bs_idx[0]], d[bs_idx[1]], d[bs_idx[2]],
        )
        if pos is None:
            continue
        if not (-70 <= pos[0] <= 70 and -40 <= pos[1] <= 40):
            continue

        est_ranges = np.sqrt(np.sum((p_bs.T - pos) ** 2, axis=1))
        violations = est_ranges - d - noise_margin
        violation_count = int(np.sum(violations > 0))
        inliers = np.where(np.abs(est_ranges - d) <= noise_margin)[0]

        if (violation_count < best_violation_count) or \
           (violation_count == best_violation_count and len(inliers) > len(best_inliers)):
            best_violation_count = violation_count
            best_pos = pos
            best_inliers = inliers

    if best_pos is not None and len(best_inliers) >= 3:
        bs_sub = p_bs[:, best_inliers].T
        d_sub = d[best_inliers]
        def residual(x):
            return np.sqrt(np.sum((bs_sub - x) ** 2, axis=1)) - d_sub
        res = least_squares(residual, best_pos, loss='linear', max_nfev=50)
        return res.x, len(best_inliers)

    w = 1.0 / (d + 1e-6)
    return (p_bs * w).sum(axis=1) / w.sum(), 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Iterative RANSAC-Correct Loop
# ═══════════════════════════════════════════════════════════════════════════

def iterative_ransac_correct(d, p_bs, n_iter=3, alpha=0.9, noise_margin=3.0):
    """반복적 RANSAC-Correct. Returns: (pos, n_inliers)."""
    d_current = d.copy()
    pos = None
    n_inliers = 0

    for _ in range(n_iter):
        pos, n_inliers = geometric_ransac_anchor(d_current, p_bs, noise_margin)
        est_ranges = np.sqrt(np.sum((p_bs.T - pos) ** 2, axis=1))
        bias = np.maximum(0, d_current - est_ranges)
        d_current = d - alpha * bias
        d_current = np.maximum(d_current, 0.1)

    return pos, n_inliers


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature Engineering (88D)
# ═══════════════════════════════════════════════════════════════════════════

def make_features_one(d_u, p_bs, anchor):
    """단일 사용자 88D feature vector."""
    bs = p_bs.T             # (18, 2)
    anchor_ranges = np.sqrt(np.sum((bs - anchor) ** 2, axis=1))  # (18,)
    range_residual = d_u - anchor_ranges                          # (18,)

    sorted_d = np.sort(d_u)
    stats = np.concatenate([
        [d_u.mean(), d_u.std(), d_u.min(), d_u.max(), np.median(d_u)],
        sorted_d[:3], sorted_d[-3:],
    ])                                                             # (11,)

    rank = np.argsort(np.argsort(d_u)).astype(float) / 17.0      # (18,)

    top3 = sorted_d[:3]
    pair_diffs = np.array([
        top3[1] - top3[0], top3[2] - top3[0], top3[2] - top3[1],
    ])                                                             # (3,)

    return np.concatenate([
        d_u, anchor, range_residual, np.abs(range_residual),
        stats, rank, pair_diffs,
    ])                                                             # (88,)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Inference — README 규격: your_algorithm(d_hat[:, u], p_bs)
# ═══════════════════════════════════════════════════════════════════════════

_SAVED = None

def _load_model():
    global _SAVED
    if _SAVED is None:
        model_path = Path(__file__).parent.resolve() / 'model.pkl'
        with open(model_path, 'rb') as f:
            _SAVED = pickle.load(f)
    return _SAVED


def your_algorithm(d_hat_u, p_bs):
    """
    단일 사용자의 RTT로 위치를 추정한다.

    Parameters
    ----------
    d_hat_u : (18,)  — 한 사용자의 RTT 측정값
    p_bs    : (2,18) — 18개 기지국 좌표

    Returns
    -------
    pos : (2,)  — 추정 위치 [x, y]
    """
    saved = _load_model()
    noise_margin = saved.get('noise_margin', 3.0)
    n_iter       = saved.get('n_iter', 3)
    alpha        = saved.get('alpha', 0.9)

    # 1. Iterative RANSAC-Correct Anchor
    pos, _ = iterative_ransac_correct(
        d_hat_u, p_bs, n_iter, alpha, noise_margin
    )

    # 2. Feature 추출 (단일 사용자)
    X = make_features_one(d_hat_u, p_bs, pos).reshape(1, -1)

    # 3. ML 잔차 보정
    residual = saved['model'].predict(X)[0]
    return pos + residual


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main — README 규격
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = 'DH_FR1.mat'

    data = sio.loadmat(mat_path, squeeze_me=False)
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    else:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p     = np.asarray(data['p'],     dtype=float)

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    main()
