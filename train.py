"""
train.py — Iterative RANSAC-Correct + Symmetric Augmentation Residual Learning

알고리즘 구조 (IRAC: Iterative RANSAC-Correct):
  1. Geometric RANSAC Anchor: NLOS 양의 바이어스 물리 제약(||x-p_i|| ≤ d_i)을
     단방향 부등식으로 활용하여 C(18,3) 조합 중 물리적으로 일관된 BS 부분집합을
     선별하고, inlier BS만으로 정밀 위치 추정
  2. 반복적 NLOS 보정 루프: RANSAC 추정 위치 → NLOS 바이어스 추정 → RTT 부분 보정
     → 보정된 RTT로 다시 RANSAC → 수렴까지 반복 (3회)
  3. 대칭 데이터 증강: 6×3 BS 그리드의 X축/Y축/원점 대칭을 이용하여
     700 → 2800 샘플로 4배 증강
  4. 88D Feature 추출 + GBR 잔차 학습 (얕은 depth=3으로 정규화)
  5. p_hat = iterative_ransac_anchor + GBR_residual
"""

import numpy as np
import scipy.io as sio
import pickle
import time
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone


# ═══════════════════════════════════════════════════════════════════════════
# 1. Physics: Geometric RANSAC Anchor
# ═══════════════════════════════════════════════════════════════════════════

def multilateration_3bs(p1, p2, p3, d1, d2, d3):
    """3개 BS 좌표와 거리로 2D 위치 추정 (원의 교차 선형화)."""
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


def geometric_ransac_anchor(d, p_bs, noise_margin=3.0, max_iter=150):
    """
    Geometric RANSAC Consensus Anchor.

    핵심 원리: NLOS 환경에서 RTT 측정값은 항상 실제 거리 이상이므로
    (d_measured >= d_true), 올바른 위치 x에서는 모든 BS i에 대해
    ||x - p_bs_i|| <= d_hat_i + noise_margin 이 성립해야 함.
    이 단방향 부등식 위반 횟수를 최소화하는 위치를 찾음.
    """
    n_bs = len(d)
    best_pos = None
    best_inliers = np.array([], dtype=int)
    best_violation_count = 999

    comb = list(combinations(range(n_bs), 3))
    rng = np.random.RandomState(42)
    selected = rng.choice(len(comb), min(max_iter, len(comb)), replace=False)

    for idx in selected:
        bs_idx = comb[idx]
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

def iterative_ransac_correct(d, p_bs, n_iter=3, alpha=0.9,
                              noise_margin=3.0, max_ransac_iter=816):
    """
    반복적 RANSAC-Correct 루프.

    핵심 아이디어: NLOS 바이어스는 항상 양수이므로, 추정 위치에서 계산한
    '예측 거리'와 '측정 거리'의 양의 차이가 곧 NLOS 바이어스 추정치임.
    이를 부분적으로(alpha 비율) 차감한 보정 RTT로 다시 RANSAC을 수행하면
    더 정확한 위치를 얻을 수 있고, 이 과정을 반복하면 수렴함.

    Step 1: RANSAC으로 초기 위치 추정
    Step 2: 추정 위치에서 NLOS 바이어스 추정 (양수만 취함)
    Step 3: RTT 부분 보정 (alpha 비율만큼 차감)
    Step 4: 보정된 RTT로 다시 RANSAC
    → 수렴까지 반복 (기본 3회)
    """
    d_current = d.copy()
    pos = None
    n_inliers = 0

    for _ in range(n_iter):
        # RANSAC으로 현재 RTT 기반 위치 추정
        pos, n_inliers = geometric_ransac_anchor(
            d_current, p_bs, noise_margin, max_ransac_iter
        )

        # 추정 위치에서 각 BS까지의 기하학적 거리
        est_ranges = np.sqrt(np.sum((p_bs.T - pos) ** 2, axis=1))

        # NLOS 바이어스 추정: 양수인 것만 (NLOS는 항상 양의 바이어스)
        bias = np.maximum(0, d_current - est_ranges)

        # 원본 RTT에서 alpha 비율만큼 바이어스 차감
        d_current = d - alpha * bias

        # 음수 방지 (물리적으로 거리는 항상 양수)
        d_current = np.maximum(d_current, 0.1)

    return pos, n_inliers, d_current


def compute_iterative_ransac_anchors(d_hat, p_bs, n_iter=3, alpha=0.9,
                                      noise_margin=3.0, max_ransac_iter=816):
    """전체 사용자에 대해 Iterative RANSAC-Correct anchor 계산 → anchors (N,2), d_corrected (18,N)."""
    N = d_hat.shape[1]
    anchors = np.zeros((N, 2))
    d_corrected = np.zeros_like(d_hat)
    for u in range(N):
        anchors[u], _, d_corrected[:, u] = iterative_ransac_correct(
            d_hat[:, u], p_bs, n_iter, alpha, noise_margin, max_ransac_iter
        )
    return anchors, d_corrected


# ═══════════════════════════════════════════════════════════════════════════
# 3. Symmetric Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════

X_SYM_MAP = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
Y_SYM_MAP = [12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
ORIGIN_SYM_MAP = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def augment_symmetric(d_hat, y):
    """
    BS 그리드 대칭을 이용한 4배 데이터 증강.
    (x,y) → 원본, (-x,y), (x,-y), (-x,-y) 의 4가지 대칭 변환.
    """
    d0, y0 = d_hat, y
    d1 = d_hat[X_SYM_MAP, :];  y1 = y.copy(); y1[:, 0] = -y1[:, 0]
    d2 = d_hat[Y_SYM_MAP, :];  y2 = y.copy(); y2[:, 1] = -y2[:, 1]
    d3 = d_hat[ORIGIN_SYM_MAP, :];  y3 = -y
    return np.hstack([d0, d1, d2, d3]), np.vstack([y0, y1, y2, y3])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering (88D)
# ═══════════════════════════════════════════════════════════════════════════

def make_features(d_hat, p_bs, anchors):
    """
    88D feature vector.
    raw RTT(18) + anchor(2) + range_residual(18) + abs_residual(18)
    + statistics(11) + rank(18) + top3_pair_diffs(3) = 88
    """
    raw = d_hat.T
    bs = p_bs.T

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
    ])
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

    # ── Iterative RANSAC-Correct Anchor (원본 700명) ──────────────────
    print('Iterative RANSAC-Correct Anchor 계산 중 (전수탐색 816개, 2회 반복, alpha=0.9)...')
    t0 = time.time()
    anchors_orig, _ = compute_iterative_ransac_anchors(d_hat, p_bs)
    anchor_rmse = float(np.mean(np.sqrt(np.sum((anchors_orig - y) ** 2, axis=1))))
    print(f'  Iterative RANSAC Anchor RMSE: {anchor_rmse:.4f} m ({time.time()-t0:.1f}s)')

    # ── OOF Cross-Validation ──────────────────────────────────────────
    print('\n5-Fold OOF Cross-Validation...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    model_proto = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=3,
        subsample=0.8, random_state=42,
    ))

    oof_preds = np.zeros_like(y)
    train_rmses = []
    val_rmses = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(y)):
        # Train fold: 대칭 증강
        d_hat_tr_aug, y_tr_aug = augment_symmetric(d_hat[:, tr_idx], y[tr_idx])

        # Iterative RANSAC anchor (증강 데이터)
        anchors_tr, _ = compute_iterative_ransac_anchors(
            d_hat_tr_aug, p_bs
        )
        X_tr = make_features(d_hat_tr_aug, p_bs, anchors_tr)
        res_tr = y_tr_aug - anchors_tr

        # Validation fold: 원본만
        anchors_va, _ = compute_iterative_ransac_anchors(
            d_hat[:, va_idx], p_bs
        )
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
    anchors_aug, _ = compute_iterative_ransac_anchors(d_hat_aug, p_bs)

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
        'type': 'irac_residual',
        'model': final_model,
        'noise_margin': 3.0,
        'max_iter': 816,
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
    print(f'  Iterative RANSAC Anchor RMSE : {anchor_rmse:.4f} m')
    print(f'  OOF RMSE (test 추정)         : {oof_rmse:.4f} m')
    print(f'  과적합 갭                    : {avg_val - avg_train:.4f} m')
    print(f'  총 소요 시간                 : {elapsed:.1f}s')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
