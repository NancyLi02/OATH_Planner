from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal
import numpy as np

try:
    # sklearn >= 1.2
    from sklearn.cluster import AgglomerativeClustering
    _HAS_NEW_SK = True
except Exception:
    AgglomerativeClustering = None
    _HAS_NEW_SK = False


@dataclass
class Robot:
    name: str
    is_hybrid: bool                   # used only in the 2-type legacy scoring
    U: Optional[np.ndarray] = None    # shape (|τ|,), used for multi-type
    cap: Optional[int] = None         # optional capacity (not used in legacy formula)


def _agglomerative_from_precomputed(M: np.ndarray,
                                    n_clusters: Optional[int] = None,
                                    distance_threshold: Optional[float] = None,
                                    linkage: str = "average") -> np.ndarray:
    """
    Cluster with a precomputed distance matrix using agglomerative clustering.
    Returns labels of shape (T,).
    """
    if AgglomerativeClustering is None:
        raise ImportError("scikit-learn is required for clustering.")

    # sklearn changed API: older versions use `affinity='precomputed'`, newer use `metric='precomputed'`
    if _HAS_NEW_SK:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage=linkage,
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            affinity="precomputed",
            linkage=linkage,
        )
    labels = model.fit_predict(M)
    return labels


def _cluster_medoids(M: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    """
    For each cluster label, pick the medoid task index (with minimal average distance to others in the cluster).
    Returns dict: cluster_label -> medoid_task_index
    """
    medoids = {}
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        subM = M[np.ix_(idx, idx)]
        # average distance from each member to all others
        avg = subM.mean(axis=1)
        medoids[int(k)] = int(idx[np.argmin(avg)])
    return medoids


def _compute_two_type_gamma(task_types: np.ndarray,
                            labels: np.ndarray,
                            theta: float) -> Dict[int, float]:
    """
    For each cluster: gamma_k = (N1 + theta) / (N2 + theta)
    Assumes type labels are 1 and 2.
    """
    gammas = {}
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        N1 = int(np.sum(task_types[idx] == 1))
        N2 = int(np.sum(task_types[idx] == 2))
        gammas[int(k)] = (N1 + theta) / (N2 + theta)
    return gammas


def _compute_multi_type_composition(task_types: np.ndarray,
                                    labels: np.ndarray,
                                    num_types: int,
                                    theta: float) -> Dict[int, np.ndarray]:
    """
    For each cluster: psi_k = (N_k + theta) / ||N_k + theta||_1
    Returns dict: k -> psi_k (shape (|τ|,))
    """
    comp = {}
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        counts = np.bincount(task_types[idx], minlength=num_types + 1)[1:]  # ignore 0
        vec = counts.astype(float) + theta
        comp[int(k)] = vec / np.sum(vec)
    return comp


def _normalize_capability(U: np.ndarray) -> np.ndarray:
    """zeta_r = U_r / ||U_r||_1 (safe for all-zero by returning zeros)."""
    s = np.sum(np.abs(U))
    if s <= 0:
        return np.zeros_like(U, dtype=float)
    return (U.astype(float) / s)


def _robot_to_cluster_distance(
    delta_rt: Optional[np.ndarray],
    robot_positions: Optional[np.ndarray],
    task_positions: Optional[np.ndarray],
    medoid_task_index: int,
    r_idx: int,
) -> float:
    """
    Returns obstacle-aware distance s_{rk}. If delta_rt is provided (R x T), use it.
    Otherwise, compute Euclidean distance from robot_positions[r] to task_positions[medoid].
    """
    if delta_rt is not None:
        return float(delta_rt[r_idx, medoid_task_index])
    if robot_positions is None or task_positions is None:
        raise ValueError("Provide either delta_rt (R x T) or (robot_positions, task_positions).")
    return float(np.linalg.norm(robot_positions[r_idx] - task_positions[medoid_task_index]))


def score_clusters_and_robots(
    M: np.ndarray,
    task_types: np.ndarray,          # shape (T,), integer in {1..|τ|}
    robots: List[Robot],
    theta: float = 1e-3,
    mode: Literal["TwoTypeLegacy", "MultiType"] = "TwoTypeLegacy",
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    delta_rt: Optional[np.ndarray] = None,  # optional R x T robot->task distances
    robot_positions: Optional[np.ndarray] = None,  # optional R x d
    task_positions: Optional[np.ndarray] = None,   # optional T x d
) -> Tuple[np.ndarray, Dict[int, float], np.ndarray, Dict[int, int]]:
    """
    Runs:
      1) Agglomerative clustering (precomputed M) -> labels
      2) Cluster statistics:
           - TwoTypeLegacy: gamma_k = (N1+theta)/(N2+theta)
           - MultiType: psi_k = (N_k + theta)/||N_k + theta||_1
      3) Scoring:
           - TwoTypeLegacy:
                if robot.is_hybrid:  s_{rk} = delta / gamma_k
                else:                s_{rk} = delta * gamma_k
           - MultiType:
                zeta_r = U_r / ||U_r||_1
                gamma_{rk} = <psi_k, zeta_r>
                s_{rk} = delta / gamma_{rk}
    Returns:
      labels: (T,) cluster labels for each task (values 0..K-1)
      gamma_or_empty: dict k->gamma_k (legacy) or empty dict (multi-type)
      S: (R x K) score matrix s_{rk}
      medoids: dict k-> medoid task index (used as cluster center)
    """
    T = M.shape[0]
    assert task_types.shape[0] == T, "task_types length must equal number of tasks"
    num_types = int(np.max(task_types))

    # 1) Clustering
    labels = _agglomerative_from_precomputed(
        M, n_clusters=n_clusters, distance_threshold=distance_threshold, linkage="average"
    )

    # 2) Medoids as cluster centers
    medoids = _cluster_medoids(M, labels)
    cluster_ids = sorted(np.unique(labels).tolist())
    K = len(cluster_ids)

    # 3) Cluster statistics
    gamma_legacy: Dict[int, float] = {}
    psi: Dict[int, np.ndarray] = {}

    if mode == "TwoTypeLegacy":
        if num_types != 2:
            raise ValueError("TwoTypeLegacy mode requires exactly two task types (labels 1 and 2).")
        gamma_legacy = _compute_two_type_gamma(task_types, labels, theta)
    else:
        psi = _compute_multi_type_composition(task_types, labels, num_types, theta)

    # 4) Scores
    R = len(robots)
    S = np.zeros((R, K), dtype=float)

    if mode == "MultiType":
        # Precompute zeta_r
        zetas = []
        for r in robots:
            if r.U is None:
                raise ValueError("MultiType mode requires each robot to provide U_r (capability vector).")
            if len(r.U) != num_types:
                raise ValueError(f"Robot {r.name}: U_r must have length |τ|={num_types}.")
            zetas.append(_normalize_capability(r.U))
        zetas = np.stack(zetas, axis=0)  # R x |τ|

    for r_idx, r in enumerate(robots):
        for j, k in enumerate(cluster_ids):
            # center = medoid task index for cluster k
            med = medoids[k]
            delta = _robot_to_cluster_distance(delta_rt, robot_positions, task_positions, med, r_idx)

            if mode == "TwoTypeLegacy":
                gk = gamma_legacy[k]
                if r.is_hybrid:
                    s_rk = delta / gk
                else:
                    s_rk = delta * gk
            else:
                # MultiType
                gamma_rk = float(np.dot(psi[k], zetas[r_idx]))
                # safeguard: if gamma_rk is zero, treat as infeasible (inf score)
                s_rk = delta / gamma_rk if gamma_rk > 0 else np.inf

            S[r_idx, j] = s_rk

    return labels, gamma_legacy, S, medoids


# ------------------------
# Example usage (toy)
# ------------------------
if __name__ == "__main__":
    np.random.seed(0)
    T = 20
    # symmetric nonnegative distance matrix (toy)
    X = np.random.rand(T, 2)
    M = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    # types in {1,2} for legacy
    task_types_2 = np.random.randint(1, 3, size=T)

    robots_legacy = [
        Robot(name="r1", is_hybrid=True),
        Robot(name="r2", is_hybrid=False),
    ]

    labels, gamma_legacy, S_legacy, medoids = score_clusters_and_robots(
        M=M,
        task_types=task_types_2,
        robots=robots_legacy,
        theta=1e-2,
        mode="TwoTypeLegacy",
        n_clusters=4,                      # or use distance_threshold=...
        robot_positions=np.array([[0.0, 0.0], [1.0, 1.0]]),
        task_positions=X,
    )
    print("Legacy gamma_k:", gamma_legacy)
    print("Legacy scores S (R x K):\n", S_legacy)

    # Multi-type example (|τ|=5)
    task_types_5 = np.random.randint(1, 6, size=T)
    robots_multi = [
        Robot(name="rA", is_hybrid=True,  U=np.array([0,1,1,1,0])),
        Robot(name="rB", is_hybrid=False, U=np.array([1,0,0,1,1])),
    ]

    labels, _, S_multi, medoids = score_clusters_and_robots(
        M=M,
        task_types=task_types_5,
        robots=robots_multi,
        theta=1e-2,
        mode="MultiType",
        n_clusters=4,
        robot_positions=np.array([[0.0, 0.0], [1.0, 1.0]]),
        task_positions=X,
    )
    print("Multi-type scores S (R x K):\n", S_multi)
