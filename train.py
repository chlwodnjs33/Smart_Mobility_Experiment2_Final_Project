"""
train.py — IRAC: Iterative RANSAC-Correct + Symmetric Augmentation Residual Learning

알고리즘 구조:
  1. Geometric RANSAC Anchor: NLOS 양의 바이어스 물리 제약(||x-p_i|| <= d_i)을
     단방향 부등식으로 활용, C(18,3)=816 전수 탐색하여 물리적 일관 BS 선별
  2. 반복적 NLOS 보정 루프 (3회): RANSAC → 바이어스 추정 → RTT 보정 → 재탐색
  3. 대칭 데이터 증강: 6x3 BS 그리드 대칭으로 700 → 2800 샘플
  4. 88D Feature + Huber 회귀 잔차 보정 (outlier-robust 선형, 트리 모델 대비 OOF 우수)
  5. p_hat = irac_anchor + Huber_residual
"""

import numpy as np
import scipy.io as sio
import pickle
import time
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone


# ═══════════════════════════════════════════════════════════════════════════
# 1. Physics: Geometric RANSAC Anchor
# ═══════════════════════════════════════════════════════════════════════════

# C(18,3) = 816 조합 — 자연 순서로 전수 탐색 (시드 비의존, 재현성 보장)
_COMBS = list(combinations(range(18), 3))


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
    """
    Geometric RANSAC — C(18,3) = 816 전수 탐색.
    Returns: (pos, n_inliers)
    """
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
    """
    반복적 RANSAC-Correct 루프.
    Returns: (pos, n_inliers)
    """
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


def compute_irac_anchors(d_hat, p_bs, n_iter=3, alpha=0.9, noise_margin=3.0):
    """전체 사용자: anchors (N,2), n_inliers_arr (N,)."""
    N = d_hat.shape[1]
    anchors = np.zeros((N, 2))
    n_inliers_arr = np.zeros(N)
    for u in range(N):
        anchors[u], n_inliers_arr[u] = iterative_ransac_correct(
            d_hat[:, u], p_bs, n_iter, alpha, noise_margin
        )
    return anchors, n_inliers_arr


# ═══════════════════════════════════════════════════════════════════════════
# 3. Symmetric Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════

X_SYM_MAP = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
Y_SYM_MAP = [12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
ORIGIN_SYM_MAP = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def augment_symmetric(d_hat, y):
    """4배 대칭 증강: 원본 + X축 + Y축 + 원점."""
    d1 = d_hat[X_SYM_MAP, :];  y1 = y.copy(); y1[:, 0] = -y1[:, 0]
    d2 = d_hat[Y_SYM_MAP, :];  y2 = y.copy(); y2[:, 1] = -y2[:, 1]
    d3 = d_hat[ORIGIN_SYM_MAP, :]; y3 = -y
    return np.hstack([d_hat, d1, d2, d3]), np.vstack([y, y1, y2, y3])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering (88D)
# ═══════════════════════════════════════════════════════════════════════════

def make_features(d_hat, p_bs, anchors):
    """
    88D feature vector.
    raw RTT(18) + anchor(2) + range_residual(18) + abs_residual(18)
    + statistics(11) + rank(18) + top3_pair_diffs(3) = 88
    """
    raw = d_hat.T           # (N, 18)
    bs = p_bs.T             # (18, 2)

    anchor_ranges = np.sqrt(
        np.sum((anchors[:, None, :] - bs[None, :, :]) ** 2, axis=2)
    )
    range_residual = raw - anchor_ranges

    sorted_raw = np.sort(raw, axis=1)
    stats = np.column_stack([
        raw.mean(axis=1), raw.std(axis=1),
        raw.min(axis=1), raw.max(axis=1),
        np.median(raw, axis=1),
        sorted_raw[:, :3], sorted_raw[:, -3:],
    ])

    rank = (
        np.argsort(np.argsort(raw, axis=1), axis=1).astype(float)
        / (raw.shape[1] - 1)
    )

    top3 = sorted_raw[:, :3]
    pair_diffs = np.column_stack([
        top3[:, 1] - top3[:, 0],
        top3[:, 2] - top3[:, 0],
        top3[:, 2] - top3[:, 1],
    ])

    X = np.hstack([
        raw, anchors, range_residual, np.abs(range_residual),
        stats, rank, pair_diffs,
    ])                                   # (N, 88)
    return X


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── 데이터 로드 ─────────────────────────────────────────────────────
    data = sio.loadmat('InF_DH_FR1.mat', squeeze_me=False)
    p_bs = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p = np.asarray(data['p'], dtype=float)
    y = p.T

    print(f'데이터 로드 완료: {d_hat.shape[1]}명')

    # ── IRAC Anchor (원본 700명) ──────────────────────────────────────
    print('IRAC Anchor 계산 중 (816 전수탐색, 3회 반복, alpha=0.9)...')
    t0 = time.time()
    anchors_orig, inliers_orig = compute_irac_anchors(d_hat, p_bs)
    anchor_rmse = float(np.mean(np.sqrt(np.sum((anchors_orig - y) ** 2, axis=1))))
    print(f'  IRAC Anchor RMSE: {anchor_rmse:.4f} m ({time.time()-t0:.1f}s)')
    print(f'  평균 inlier 수  : {inliers_orig.mean():.1f} / 18')

    # ── OOF Cross-Validation ──────────────────────────────────────────
    print('\n5-Fold OOF Cross-Validation...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Huber 회귀: outlier(큰 NLOS 잔차)에 강건한 선형 모델.
    # 트리 모델(GBR/RF)은 anchor 잔차의 노이즈를 과적합하여 OOF가 악화되는 반면,
    # Huber loss는 노이즈에 휘둘리지 않고 체계적 선형 성분만 보정한다.
    # 입력 스케일 차이가 크므로 StandardScaler로 표준화 후 적용.
    model_proto = MultiOutputRegressor(make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=2000),
    ))

    oof_preds = np.zeros_like(y)
    train_rmses = []
    val_rmses = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(y)):
        # Train fold: 대칭 증강
        d_hat_tr_aug, y_tr_aug = augment_symmetric(d_hat[:, tr_idx], y[tr_idx])

        # IRAC anchor (증강 데이터)
        anchors_tr, _ = compute_irac_anchors(d_hat_tr_aug, p_bs)
        X_tr = make_features(d_hat_tr_aug, p_bs, anchors_tr)
        res_tr = y_tr_aug - anchors_tr

        # Validation fold: 원본만
        anchors_va, _ = compute_irac_anchors(d_hat[:, va_idx], p_bs)
        X_va = make_features(d_hat[:, va_idx], p_bs, anchors_va)

        m = clone(model_proto)
        m.fit(X_tr, res_tr)

        tr_pred = m.predict(X_tr)
        tr_pos = anchors_tr + tr_pred
        tr_rmse = float(np.mean(np.sqrt(np.sum((tr_pos - y_tr_aug) ** 2, axis=1))))
        train_rmses.append(tr_rmse)

        va_pred = m.predict(X_va)
        va_pos = anchors_va + va_pred
        va_rmse = float(np.mean(np.sqrt(np.sum((va_pos - y[va_idx]) ** 2, axis=1))))
        val_rmses.append(va_rmse)

        oof_preds[va_idx] = va_pos

        print(f'  Fold {fold+1}: Train {tr_rmse:.4f}m | Val {va_rmse:.4f}m '
              f'(gap {va_rmse - tr_rmse:.4f}m)')

    oof_rmse = float(np.mean(np.sqrt(np.sum((oof_preds - y) ** 2, axis=1))))
    avg_train = np.mean(train_rmses)
    avg_val = np.mean(val_rmses)

    print(f'\n  ── OOF 종합 ──')
    print(f'  Train RMSE 평균 : {avg_train:.4f} m')
    print(f'  Val RMSE 평균   : {avg_val:.4f} m')
    print(f'  과적합 갭       : {avg_val - avg_train:.4f} m')
    print(f'  OOF 전체 RMSE   : {oof_rmse:.4f} m  ← hidden test 예상치')

    # ── 전체 데이터로 최종 모델 학습 ──────────────────────────────────
    print('\n전체 데이터로 최종 모델 학습 (대칭 증강 포함)...')
    d_hat_aug, y_aug = augment_symmetric(d_hat, y)
    anchors_aug, _ = compute_irac_anchors(d_hat_aug, p_bs)

    X_all = make_features(d_hat_aug, p_bs, anchors_aug)
    res_all = y_aug - anchors_aug

    final_model = clone(model_proto)
    final_model.fit(X_all, res_all)

    final_pred = final_model.predict(X_all)
    final_pos = anchors_aug + final_pred
    final_rmse = float(np.mean(np.sqrt(np.sum((final_pos - y_aug) ** 2, axis=1))))
    print(f'  전체 학습 Train RMSE: {final_rmse:.4f} m')

    # ── 저장 ────────────────────────────────────────────────────────────
    payload = {
        'type': 'irac_huber_residual',
        'model': final_model,
        'noise_margin': 3.0,
        'n_iter': 3,
        'alpha': 0.9,
        'oof_rmse': oof_rmse,
        'anchor_rmse': anchor_rmse,
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t_start
    print(f'\n{"="*50}')
    print(f'  저장 완료: model.pkl')
    print(f'  IRAC Anchor RMSE   : {anchor_rmse:.4f} m')
    print(f'  OOF RMSE (test 추정) : {oof_rmse:.4f} m')
    print(f'  과적합 갭            : {avg_val - avg_train:.4f} m')
    print(f'  총 소요 시간         : {elapsed:.1f}s')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
