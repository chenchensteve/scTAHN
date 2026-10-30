from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import dropout_edge

import anndata
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from ppo import PPOAgent
import sknetwork as skn

from sklearn.preprocessing import normalize
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_graph(data, mask_prob: float = 0.2, dropout_prob: float = 0.2):
    x, edge_index = data.x, data.edge_index
    mask = torch.empty_like(x).uniform_() > mask_prob
    x_aug = x * mask.float()
    edge_index_aug, _ = dropout_edge(edge_index, p=dropout_prob, force_undirected=True)
    return x_aug, edge_index_aug.to(x.device)


# Leiden

def _safe_leiden(
    adata: anndata.AnnData,      # 输入：单细胞 AnnData 对象
    *,
    resolution: float,           # 核心参数：分辨率，值越高分出的群越多
    key_added: str,              # 输出结果：存放在 adata.obs 中的列名
    random_state: int = 0,       # 随机种子：保证聚类结果的可重复性
    flavor: str = "igraph",      # 算法后端：默认使用 igraph，较新版本支持更多优化
    n_iterations: int = 2,       # 迭代次数：Leiden 算法通过多次迭代优化模块度
    directed: bool = False,      # 有向图：指定是否将细胞相似性图视为有向图
) -> None:
    """
    调用 Leiden 聚类算法。
    """

    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        key_added=key_added,
        random_state=random_state,
        flavor=flavor,
        n_iterations=n_iterations,
        directed=directed,
    )


def _estimate_target_clusters(n_cells: int) -> int:
    # sqrt(n/2)，避免过碎；并且限制上界,round四舍五入取整
    k = int(max(2, round(np.sqrt(max(n_cells, 2) / 2.0))))
    return int(np.clip(k, 2, 50))


def _avg_pairwise_nmi(partitions: Sequence[np.ndarray]) -> float:
    """
        计算多个聚类结果之间的平均两两互信息 (NMI)，用于评估聚类稳定性
        Args:
            partitions: 包含多个分群结果（数组格式）的列表，通常是不同随机种子产生的结果
        Returns:
            平均 NMI 分数 (0 到 1 之间，越高代表分群越稳定)
        """
    # 如果只有一个分群结果，无法进行两两比较，直接返回 0.0
    if len(partitions) <= 1:
        return 0.0
    scores: List[float] = []
    # 使用双重循环遍历所有可能的分群组合，进行两两对比
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            # 计算第 i 个结果和第 j 个结果之间的归一化互信息 (NMI)
            # NMI 越接近 1，说明不同种子下跑出的分群结果越一致
            scores.append(normalized_mutual_info_score(partitions[i], partitions[j]))

    # 返回所有组合的平均值作为最终的稳定性指标
    # 如果没有得分记录，则返回 0.0
    return float(np.mean(scores)) if scores else 0.0



def auto_leiden_resolution(
        embeddings: np.ndarray,  # 输入：模型提取的细胞高维特征表征
        true_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,  # 可选：真实标签，用于有监督调优
        *,
        n_neighbors: int = 15,  # 构建 KNN 图时的邻居数量
        metric: str = "cosine",  # 距离度量方式，默认使用余弦距离
        resolution_grid: Optional[Sequence[float]] = None,  # 搜索网格：初始的分辨率范围
        refine: bool = True,  # 是否进行二次精细化搜索
        target_n_clusters: Optional[int] = None,  # 预期分群数：无标签模式下希望得到的聚类数量
        stability_seeds: Sequence[int] = (0, 1, 2),  # 稳定性测试种子：用于多次运行对比
        stability_weight: float = 1.0,  # 无监督模式下“稳定性”指标的权重
        count_weight: float = 0.2,  # 无监督模式下“分群数量偏离度”的权重
) -> Tuple[float, Dict[str, Any]]:
    # 1. 数据校验与参数初始化
    emb = np.asarray(embeddings)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {emb.shape}")

    # 如果没有指定搜索范围，默认在 0.1 到 2.0 之间均匀取 20 个点
    if resolution_grid is None:
        # https://numpy.org/devdocs/reference/generated/numpy.linspace.html
        # tolist转化为一个python标量
        resolution_grid = np.round(np.linspace(0.1, 1.0, 20), 3).tolist()

    # 如果没有指定目标群数，根据细胞总数自动预估一个
    if target_n_clusters is None:
        target_n_clusters = _estimate_target_clusters(emb.shape[0])

    # 统一处理 true_labels 格式
    # view改变元数据，内存必须连续，reshape自动判断是否拷贝数据，内存不一定连续，选择的时候看内存是否有要求
    y_true = None
    if true_labels is not None:
        if isinstance(true_labels, torch.Tensor):
            y_true = true_labels.detach().cpu().numpy().reshape(-1)
        else:
            y_true = np.asarray(true_labels).reshape(-1)

    # 2. 构建单细胞对象并计算邻居关系
    adata = anndata.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=int(n_neighbors), metric=str(metric))

    results: List[Dict[str, Any]] = []

    # 3. 开始遍历分辨率网格进行测试
    for r in resolution_grid:
        r = float(r)
        # A 模式：有标签（有监督）
        if y_true is not None:
            # 运行 Leiden 聚类
            _safe_leiden(adata, resolution=r, key_added="leiden_tmp", random_state=0)
            pred = adata.obs["leiden_tmp"].to_numpy()

            # 计算准确性指标
            ari = adjusted_rand_score(y_true, pred)
            nmi = normalized_mutual_info_score(y_true, pred)

            # 以 ARI 和 NMI 的均值作为评分
            score = 0.5 * (ari + nmi)
            results.append(dict(resolution=r, score=float(score),
                                ari=float(ari), nmi=float(nmi),
                                n_clusters=int(len(np.unique(pred))),
                                stability=None))

        # B 模式：无标签（无监督）
        else:
            parts = []
            # 使用多个随机种子运行多次聚类
            for s in stability_seeds:
                key = f"leiden_{s}"
                _safe_leiden(adata, resolution=r, key_added=key, random_state=int(s))
                parts.append(adata.obs[key].to_numpy())

            # 计算多次运行之间的平均互信息量（稳定性）
            stability = _avg_pairwise_nmi(parts)
            n_found = int(len(np.unique(parts[0])))  # 找到每个分辨率下分出几个类别

            # 综合评分：稳定性要高，且发现的分群数不能偏离目标太远
            score = stability_weight * stability - count_weight * abs(n_found - int(target_n_clusters))
            results.append(dict(resolution=r, score=float(score),
                                ari=None, nmi=None,
                                n_clusters=n_found,
                                stability=float(stability)))

    # 4. 选出第一轮网格搜索中的最佳分辨率
    # https://blog.csdn.net/u013250071/article/details/118220139
    best = max(results, key=lambda d: d["score"])
    best_r = float(best["resolution"])

    # 5. 二次精细化搜索
    # 在第一轮最佳值附近缩小范围再次搜索，以获得更精确的数值
    if refine:
        # 生成更密集的网格（在 best_r 的 ±0.2 范围内取 9 个点）
        fine_grid = np.round(np.linspace(max(0.05, best_r - 0.2), best_r + 0.2, 9), 3).tolist()
        fine_grid = sorted(set([float(x) for x in fine_grid]))

        if len(fine_grid) > 1:
            # 递归调用自身进行二次搜索
            fine_r, fine_info = auto_leiden_resolution(
                emb, true_labels=true_labels,
                n_neighbors=n_neighbors, metric=metric,
                resolution_grid=fine_grid, refine=False,
                target_n_clusters=target_n_clusters,
                stability_seeds=stability_seeds,
                stability_weight=stability_weight, count_weight=count_weight,
            )
            # 如果精细化搜索找到了更好的得分，则更新结果
            if fine_info["best"]["score"] >= best["score"]:
                best_r = float(fine_r)
                best = fine_info["best"]
                results = fine_info["grid_results"]

    return best_r, dict(
        mode=("supervised" if y_true is not None else "unsupervised"),
        target_n_clusters=int(target_n_clusters),
        best=best,
        grid_results=results,
        n_neighbors=int(n_neighbors),
        metric=str(metric),
    )


def evaluate_clustering_leiden(
        embeddings: np.ndarray,  # 输入：模型提取的细胞高维特征（Embedding）
        true_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,  # 可选：真实标签，用于验证准确性
        *,  # 后面的必须要告诉具体的参数名
        resolution: Union[str, float, None] = "auto",  # 关键参数：分辨率。默认 "auto" 开启自动寻优
        n_neighbors: int = 15,  # KNN 图的邻居数
        metric: str = "cosine",  # 距离度量（余弦距离）
        random_state: int = 0,  # 随机种子，保证结果可重复
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # 1. 数据预处理与格式转换
    emb = np.asarray(embeddings)
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html数组维度数，判断是否为2维
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {emb.shape}")

    # 2. 自动化决策逻辑：判断是否需要自动寻找最佳分辨率
    # 如果 resolution 被设为 None 或 "auto"
    if resolution is None or (isinstance(resolution, str) and resolution.lower() == "auto"):
        # 调用我们之前分析过的 auto_leiden_resolution 函数
        # 它会通过网格搜索（Grid Search）找到让结果最准确或最稳定的 resolution
        best_r, info = auto_leiden_resolution(
            emb, true_labels=true_labels, n_neighbors=n_neighbors, metric=metric, refine=True
        )
        resolution = best_r  # 将寻优得到的“冠军”分辨率赋值给 resolution
        # 打印自动化搜索的结果，方便查阅
        print(f"\n[Auto-Leiden] resolution={resolution:.3f} | mode={info['mode']} | best={info['best']}")
    else:
        # 如果用户手动指定了数字（如 0.5），则直接强制转换为浮点数
        resolution = float(resolution)

    # 3. 执行最终聚类
    # 创建 Scanpy 专用的 AnnData 对象
    adata = anndata.AnnData(X=emb)
    # 重新构建最终的细胞邻居图
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=int(n_neighbors), metric=str(metric))
    # 调用Leiden 函数执行聚类，结果存入 adata.obs["leiden"]
    _safe_leiden(adata, resolution=float(resolution), key_added="leiden", random_state=int(random_state))

    # 4. 提取聚类结果
    pred = adata.obs["leiden"].to_numpy()  # 获取每个细胞被分到了哪个簇（Cluster）
    n_found = int(len(np.unique(pred)))  # 统计总共分出了多少个簇

    # 5. 性能评估与指标计算
    # 初始化一个字典，用来存储所有的评估数据
    metrics: Dict[str, Any] = dict(resolution=float(resolution), n_clusters=n_found, ari=None, nmi=None)

    # 如果提供了真实标签，则计算学术界认可的 ARI 和 NMI 评分
    if true_labels is not None:
        if isinstance(true_labels, torch.Tensor):
            # 将 Tensor 转移到 CPU 并展平为一维数组
            y_true = true_labels.detach().cpu().numpy().reshape(-1)
        else:
            y_true = np.asarray(true_labels).reshape(-1)

        # 计算调整兰德系数 (ARI) 和 归一化互信息 (NMI)
        ari = adjusted_rand_score(y_true, pred)
        nmi = normalized_mutual_info_score(y_true, pred)

        # 更新 metrics 字典并打印结果
        metrics.update(dict(ari=float(ari), nmi=float(nmi)))
        print(f"Leiden clusters={n_found} | ARI={ari:.4f} | NMI={nmi:.4f}")
    else:
        # 如果没有标签，只打印分出的簇数量
        print(f"Leiden clusters={n_found} (没有true_labels，没有计算 ARI，NMI)")

    # 返回聚类结果数组和完整的评估指标字典
    return pred, metrics

def evaluate_clustering_leiden2(
        embeddings: np.ndarray,
        true_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        n_neighbors: int = 15,
        metric: str = "cosine"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    改进版评估：强制通过二分查找匹配真实簇数 K
    """
    emb = np.asarray(embeddings)
    adata_eval = anndata.AnnData(X=emb)
    sc.pp.neighbors(adata_eval, use_rep="X", n_neighbors=n_neighbors, metric=metric)

    # 统一转换标签格式
    if isinstance(true_labels, torch.Tensor):
        y_true = true_labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(true_labels).reshape(-1)

    target_k = len(np.unique(y_true))

    # --- 二分搜索核心逻辑 ---
    low, high = 0.0, 5.0
    best_res = 0.5

    # 1. 动态确认 high 范围
    for _ in range(5):
        _safe_leiden(adata_eval, resolution=high, key_added="tmp")
        if len(np.unique(adata_eval.obs["tmp"])) >= target_k:
            break
        high *= 2

    # 2. 迭代搜索
    for _ in range(30):
        mid = (low + high) / 2.0
        _safe_leiden(adata_eval, resolution=mid, key_added="tmp")
        n = len(np.unique(adata_eval.obs["tmp"]))
        if n == target_k:
            best_res = mid
            break
        elif n < target_k:
            low = mid
        else:
            high = mid
        best_res = mid

    # 3. 得到最终预测结果
    _safe_leiden(adata_eval, resolution=best_res, key_added="leiden")
    pred = adata_eval.obs["leiden"].to_numpy().astype(int)

    # 4. 计算指标
    ari = adjusted_rand_score(y_true, pred)
    nmi = normalized_mutual_info_score(y_true, pred)

    metrics = {
        "ari": float(ari),
        "nmi": float(nmi),
        "res": float(best_res),
        "n_clusters": int(len(np.unique(pred)))
    }

    print(f"Final Alignment: K={target_k} | Found_K={metrics['n_clusters']} | Res={best_res:.4f} | ARI={ari:.4f} | NMI={nmi:.4f}")
    return pred, metrics

def evaluate_clustering_kmeans(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        random_state: int = 42
):
    """
    使用 K-Means 强制输出真实簇数 K，最公平的特征评估法。
    """
    emb = np.asarray(embeddings)

    # 统一转换标签格式
    if isinstance(true_labels, torch.Tensor):
        y_true = true_labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(true_labels).reshape(-1)

    # 1. 获取真实簇数 K
    target_k = len(np.unique(y_true))

    # 2. 运行 K-Means 聚类
    kmeans = KMeans(n_clusters=target_k, random_state=random_state, n_init=10)
    pred_labels = kmeans.fit_predict(emb)

    # 3. 计算指标
    ari = adjusted_rand_score(y_true, pred_labels)
    nmi = normalized_mutual_info_score(y_true, pred_labels)

    metrics = {
        "ari": float(ari),
        "nmi": float(nmi),
        "n_clusters": target_k,
        "method": "K-Means"
    }

    print(f"[K-Means Eval] Target_K={target_k} | ARI={ari:.4f} | NMI={nmi:.4f}")
    return pred_labels, metrics


def evaluate_clustering_leiden_adaptive(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_neighbors: int = 15,
        metric: str = "cosine",
        res_range: list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
):
    """
    全自动无监督寻优：通过轮廓系数寻找最适合当前数据集的 Resolution，再算分。
    """
    emb = np.asarray(embeddings)

    if isinstance(true_labels, torch.Tensor):
        y_true = true_labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(true_labels).reshape(-1)

    target_k = len(np.unique(y_true))

    adata_eval = anndata.AnnData(X=emb)
    sc.pp.neighbors(adata_eval, use_rep="X", n_neighbors=n_neighbors, metric=metric)

    best_res = None
    best_sil = -1.0
    best_pred = None

    print(f"  [Adaptive Search] 正在为新数据集寻找最佳 Resolution...")

    # 1. 无监督遍历扫描
    for res in res_range:
        sc.tl.leiden(adata_eval, resolution=res, random_state=42, key_added="tmp")
        pred = adata_eval.obs["tmp"].to_numpy().astype(int)

        n_clusters = len(np.unique(pred))
        if n_clusters > 1:
            # 使用轮廓系数（完全无监督）作为唯一评判标准
            sil = silhouette_score(emb, pred, metric='cosine')
        else:
            sil = -1.0

        if sil > best_sil:
            best_sil = sil
            best_res = res
            best_pred = pred.copy()

    # 2. 使用选出的最佳预测计算最终分数（解封标签）
    ari = adjusted_rand_score(y_true, best_pred)
    nmi = normalized_mutual_info_score(y_true, best_pred)
    found_k = len(np.unique(best_pred))

    metrics = {
        "ari": float(ari),
        "nmi": float(nmi),
        "res": float(best_res),
        "n_clusters": int(found_k),
        "method": "Adaptive Leiden"
    }

    print(
        f"[Adaptive Leiden] Selected Res={best_res} (Sil={best_sil:.3f}) | True_K={target_k} vs Found_K={found_k} | ARI={ari:.4f} | NMI={nmi:.4f}")
    return best_pred, metrics

def evaluate_clustering_target_k(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_neighbors: int = 15,
        metric: str = "cosine"
):
    """
    终极评估法：目标 K 锚定的图模块度寻优。
    （已集成 NaN 保护、Connectivities 修复、消除 Leiden 警告）
    """
    emb = np.asarray(embeddings)

    # 【新增保护层】：防止上游梯度爆炸导致 NaN 报错
    if np.isnan(emb).any() or np.isinf(emb).any():
        print("💥 致命错误：特征空间出现 NaN 或 Inf，模型输出崩溃！返回 0 分跳过。")
        return np.zeros(len(true_labels)), {"ari": 0.0, "nmi": 0.0}

    if isinstance(true_labels, torch.Tensor):
        y_true = true_labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(true_labels).reshape(-1)

    target_k = len(np.unique(y_true))

    adata_eval = anndata.AnnData(X=emb)
    sc.pp.neighbors(adata_eval, use_rep="X", n_neighbors=n_neighbors, metric=metric)

    # 【修复1】：之前这里被不小心删掉了！现在找回来了，并且使用的是正确的连通性矩阵
    adj_matrix = adata_eval.obsp['connectivities']

    res_range = np.arange(0.02, 1.0, 0.02)

    best_res = None
    best_modularity = -1.0
    best_pred = None
    min_k_diff = float('inf')

    for res in res_range:
        # 【修复2】：加上了 flavor="igraph" 等参数，彻底消灭烦人的黄字警告
        sc.tl.leiden(
            adata_eval,
            resolution=res,
            random_state=42,
            key_added="tmp",
            flavor="igraph",
            n_iterations=2,
            directed=False
        )
        pred = adata_eval.obs["tmp"].to_numpy().astype(int)
        n_clusters = len(np.unique(pred))

        k_diff = abs(n_clusters - target_k)

        if k_diff <= min_k_diff and n_clusters > 1:
            modularity = skn.clustering.get_modularity(adj_matrix, pred)

            if k_diff < min_k_diff or (k_diff == min_k_diff and modularity > best_modularity):
                min_k_diff = k_diff
                best_modularity = modularity
                best_res = res
                best_pred = pred.copy()

    # 极端情况兜底：如果所有 resolution 都没有切出 >1 个簇
    if best_pred is None:
        best_pred = np.zeros(len(y_true))
        best_res = 0.0

    # 最后计算分数
    ari = adjusted_rand_score(y_true, best_pred)
    nmi = normalized_mutual_info_score(y_true, best_pred)
    found_k = len(np.unique(best_pred))

    print(
        f"[Target-K Leiden] 锚定 Res={best_res:.2f} (Modularity={best_modularity:.4f}) | True_K={target_k} vs Found_K={found_k} | ARI={ari:.4f} | NMI={nmi:.4f}")

    return best_pred, {"ari": float(ari), "nmi": float(nmi)}


def evaluate_clustering_multistart_leiden(
        embeddings: np.ndarray,
        true_labels,
        *,
        n_neighbors: int = 15,
        metric: str = "cosine",
        n_seeds: int = 10,
        top_k_candidates: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    多起点图集成 Leiden 聚类评估（无数据泄露）。

    仅以类别数 K 作为先验知识，不使用真实标签参与任何优化决策。

    设计思路（三阶段）：
      Phase-1  全局粗扫描 [0.01, 2.0] step=0.01，收集所有恰好产生 K 个簇的 resolution，
               同时为每个候选计算 seed=0 时的模块度，按模块度降序排列。
               覆盖全区间的意义在于：Leiden 在不同 resolution 区段产生的 K 个簇
               对应完全不同的图划分，高分辨率下的划分不一定比低分辨率差，必须全搜。
      Phase-2  精选 top_k_candidates 个模块度最高的候选 resolution。
      Phase-3  对每个候选以 n_seeds 个随机起点反复运行 Leiden，
               全程以模块度作为无监督选择信号，输出模块度最高的划分。

    Args:
        embeddings:        模型输出的 L2 归一化嵌入 [N, D]
        true_labels:       真实标签（仅用于提取 K 和最终 ARI/NMI 评分）
        n_neighbors:       构建 KNN 图的邻居数
        metric:            距离度量，默认余弦距离
        n_seeds:           Phase-3 每个候选 resolution 的随机起点数
        top_k_candidates:  Phase-2 保留的高模块度候选数量

    Returns:
        pred_labels: 最终聚类标签数组
        metrics:     包含 ari, nmi, modularity, n_clusters, res 的字典
    """
    emb = np.asarray(embeddings)

    # ── 保护性检查 ──────────────────────────────────────────────────────────
    if np.isnan(emb).any() or np.isinf(emb).any():
        print("[MultiStart-Leiden] 致命错误：嵌入中存在 NaN/Inf，返回 0 分。")
        dummy = np.zeros(len(true_labels), dtype=int)
        return dummy, {"ari": 0.0, "nmi": 0.0, "modularity": -1.0, "n_clusters": 1}

    if isinstance(true_labels, torch.Tensor):
        y_true = true_labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(true_labels).reshape(-1)

    target_k = int(len(np.unique(y_true)))

    # ── 构建 KNN 图 ──────────────────────────────────────────────────────────
    adata = anndata.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors, metric=metric)
    adj_matrix = adata.obsp['connectivities']

    def _leiden(res: float, seed: int) -> np.ndarray:
        sc.tl.leiden(
            adata, resolution=float(res), key_added="_ms_tmp_",
            flavor="igraph", n_iterations=2, directed=False,
            random_state=int(seed)
        )
        return adata.obs["_ms_tmp_"].to_numpy().astype(int)

    # ── Phase-1：全局粗扫描，找出所有 K-matching resolution ─────────────────
    # 关键修正：必须扫全 [0.01, 2.0]，因为不同区段的 K 个簇对应本质不同的划分，
    # 仅围绕某个锚点搜索会系统性地漏掉其他高模块度区间（上一版本的根本缺陷）。
    upper_bound = 2.0
    for _ in range(4):  # 动态扩展上界，应对 K 很大的数据集
        if len(np.unique(_leiden(upper_bound, 0))) >= target_k:
            break
        upper_bound *= 2.0

    coarse_grid = np.round(np.arange(0.01, upper_bound + 0.01, 0.01), 4)

    # 收集所有恰好产生 K 个簇的候选，同时记录其 seed=0 模块度（用于粗排）
    candidates: List[Dict] = []
    for res in coarse_grid:
        pred = _leiden(res, 0)
        if len(np.unique(pred)) == target_k:
            mod = float(skn.clustering.get_modularity(adj_matrix, pred))
            candidates.append({"res": float(res), "mod": mod, "pred": pred.copy()})

    # 若完全没有精确 K-matching 的 resolution（极罕见），则回退到最近 K
    if not candidates:
        min_diff = float('inf')
        fallback_pred = None
        fallback_res = 0.5
        for res in np.arange(0.01, upper_bound, 0.02):
            pred = _leiden(res, 0)
            diff = abs(len(np.unique(pred)) - target_k)
            if diff < min_diff:
                min_diff = diff
                fallback_pred = pred.copy()
                fallback_res = float(res)
        mod = float(skn.clustering.get_modularity(adj_matrix, fallback_pred))
        ari = float(adjusted_rand_score(y_true, fallback_pred))
        nmi = float(normalized_mutual_info_score(y_true, fallback_pred))
        print(f"[MultiStart-Leiden] 无精确 K-match，fallback Res={fallback_res:.3f} | "
              f"ARI={ari:.4f} | NMI={nmi:.4f}")
        return fallback_pred, {"ari": ari, "nmi": nmi, "modularity": mod,
                               "n_clusters": len(np.unique(fallback_pred)), "res": fallback_res}

    # ── Phase-2：按模块度降序，保留 top_k_candidates 个候选 ─────────────────
    # 模块度高 → 簇内图连接密、簇间连接稀 → 与余弦嵌入结构高度对应
    candidates.sort(key=lambda d: d["mod"], reverse=True)
    top_candidates = candidates[:top_k_candidates]

    # ── Phase-3：对每个 top 候选运行 n_seeds 随机起点，最终择优 ──────────────
    best_pred = top_candidates[0]["pred"]  # 至少保证有 seed=0 的结果
    best_modularity = top_candidates[0]["mod"]
    best_res_used = top_candidates[0]["res"]
    best_seed_used = 0

    for cand in top_candidates:
        res = cand["res"]
        for seed in range(n_seeds):
            if seed == 0:
                # seed=0 已在 Phase-1 跑过，直接复用，不重复计算
                pred = cand["pred"]
                mod = cand["mod"]
            else:
                pred = _leiden(res, seed)
                if len(np.unique(pred)) != target_k:
                    continue
                mod = float(skn.clustering.get_modularity(adj_matrix, pred))

            if mod > best_modularity:
                best_modularity = mod
                best_pred = pred.copy()
                best_res_used = res
                best_seed_used = seed

    # ── 最终评分 ──────────────────────────────────────────────────────────────
    ari = float(adjusted_rand_score(y_true, best_pred))
    nmi = float(normalized_mutual_info_score(y_true, best_pred))
    found_k = int(len(np.unique(best_pred)))

    print(
        f"[MultiStart-Leiden] "
        f"K={target_k} | KMatchRes={len(candidates)} | TopCands={len(top_candidates)} | "
        f"BestRes={best_res_used:.3f} Seed={best_seed_used} | "
        f"Mod={best_modularity:.4f} | Found_K={found_k} | "
        f"ARI={ari:.4f} | NMI={nmi:.4f}"
    )

    return best_pred, {
        "ari": ari,
        "nmi": nmi,
        "modularity": best_modularity,
        "n_clusters": found_k,
        "res": best_res_used,
    }
# RL: state和reward

@dataclass
class RLConfig:
    # 训练节奏控制
    warmup_epochs: int = 10     # 预热轮数：前10轮不更新RL，学习基本特征分布
    min_ratio: float = 0.15     # 最小硬样本比例：限制智能体最少必须选择 15% 的细胞作为“难点”
    max_ratio: float = 0.35     # 最大硬样本比例：限制智能体最多选择 35% 的细胞
    warmup_ratio: float = 0.15  # 预热期比例：预热阶段固定使用 15% 的比例进行训练
    ema_alpha: float = 0.8      # 指数移动平均系数：用于平滑奖励信号，防止模型更新步子迈得太大

    # FNR计算阈值：用于识别罕见细胞
    # 定义“高置信度”细胞：取模型预测概率前 10% 的细胞作为参考
    high_conf_quantile: float = 0.1
    # 绝对阈值：如果模型对某个类别的信心低于 0.8，则认为该类别可能存在问题
    min_abs_conf_thr: float = 0.8

    # 奖励函数权重
    w_loss: float = 2.5         # 损失函数权重：智能体选择能让模型 Loss 下降最明显的样本
    loss_scale: float = 100.0   # 损失缩放：将微小的 Loss 变化放大 100 倍，便于强化学习捕捉信号
    w_consistency: float = 1.0  # 一致性权重：智能体让同一类细胞在特征空间里更靠拢
    w_fnr: float = 1.0          # FNR 权重：智能体去关注那些容易被漏掉的 假负样本细胞
    fnr_scale_count: float = 5.0 # FNR 数量缩放：对发现假负样本细胞细胞的行为给予额外的倍率奖励
    w_ratio: float = 0.05       # 比例惩罚权重：防止智能体为了偷懒总是选择极高或极低的比例
    w_sep: float = 0.5          # 分离度权重：鼓励模型把不同种类的细胞拉得更开
    base_ratio: float = 0.20    # 基准比例：模 型默认倾向于保持 20%的硬样本采样率
    reward_clip: float = 5.0    # 奖励裁剪：将奖励值限制在 [-5, 5] 之间，防止突发的极端数值破坏梯度

    # 当fnr超过 20% 时，强制智能体提高关注度
    safety_fnr_rate: float = 0.2
    # 强制执行 18% 的最低采样率，确保基础训练量
    safety_ratio: float = 0.18


def map_action_to_ratio(raw_action: float, cfg: RLConfig) -> float:
    """
    将 RL 智能体输出的 [-1, 1] 或 [0, 1] 原始动作，映射到设定的 [min_ratio, max_ratio] 范围内。

    参数:
        raw_action: 智能体神经网络直接输出的数值（通常是 Sigmoid 后的 0~1）
        cfg: 上面定义的配置对象
    """
    # 计算逻辑：最小值 + (范围区间 * 动作比例)
    # 例如：raw_action=0.5，则返回 0.15 + (0.35 - 0.15) * 0.5 = 0.25
    return float(cfg.min_ratio + (cfg.max_ratio - cfg.min_ratio) * float(raw_action))


def build_state(
        sim_pos: float,  # 正样本对（同一细胞的不同增强视图）的平均相似度
        sim_neg: float,  # 负样本对（队列中的异构细胞）的平均相似度
        loss_change: float,  # Loss的变化量（当前平滑Loss - 上一轮平滑Loss）
        fnr_rate: float,  # 假阴性率（本应是同类却被当成负样本的比例）
        consistency: float,  # 一致性分数（反映特征空间的局部结构稳定性）
        hard_ratio: float,  # 当前执行的困难样本采样比例动作
) -> np.ndarray:
    """
    构造强化学习的5维状态向量：[区分度, 一致性, FNR率, 当前采样比例, Loss趋势]
    """

    # 1. 计算正负样本区分度（Contrastive Discrimination）
    # 使用 log(pos/neg) 衡量两者的相对倍数。
    # abs(...) + 1e-6: 防止分母为0或负数；+ 1.0: 确保 log 参数 >= 1，使 sim_ratio 始终 >= 0。
    sim_ratio = float(np.log((float(sim_pos) / (abs(float(sim_neg)) + 1e-6)) + 1.0))

    # 2. 将余弦一致性分数归一化到 [0, 1] 区间
    # 原始 consistency（余弦相似度）范围在 [-1, 1]。
    # 通过 (x + 1) / 2 线性映射：-1 映射为 0（完全不一致），1 映射为 1（完全一致）。
    cons01 = float((float(consistency) + 1.0) / 2.0)

    # 3. 提取 Loss 的变化特征（Loss Momentum）
    # loss_change 数值通常很小，乘以 50.0 是为了放大波动，让智能体对收敛速度变慢更敏感。
    # np.tanh 将放大后的数值压缩到 [-1, 1] 之间，既保留了正负方向（上升/下降），又防止数值爆炸。
    loss_feat = float(np.tanh(float(loss_change) * 50.0))

    # 4. 组合成 5 维 NumPy 状态数组
    # 这个向量涵盖了特征空间的几何特征（sim）、任务难度（fnr）、策略现状（hard_ratio）和收敛状态（loss）。
    state = np.array([sim_ratio, cons01, float(fnr_rate), float(hard_ratio), loss_feat], dtype=np.float32)

    # 5. 数值裁剪（Boundary Clipping）
    # 强制将所有维度的特征值限制在 [-1.0, 5.0] 范围内。
    # 作用是剔除极端的离群值（Outliers），确保输入到 PPO 神经网络的数据分布相对集中，防止梯度崩坏。
    return np.clip(state, -1.0, 5.0)


def calculate_reward(
        loss_drop: float,     # 损失下降值
        fnr_rate: float,      # 假阴性率（本应是同类却被当成负样本推开的比例）
        consistency: float,   # 一致性分数（反映特征空间的局部结构是否稳定）
        hard_ratio: float,    # 当前智能体选择的动作：困难负样本采样比例
        sim_pos: float,       # 正样本对平均相似度
        sim_neg: float,       # 负样本对平均相似度
        cfg: RLConfig,        # 包含各项权重系数的配置对象
) -> float:
    # 1. 计算损失下降奖励，从
    # 使用 tanh 将 loss_drop 压缩到 [-1, 1]，cfg.loss_scale (100.0) 用于放大微小波动。
    r_loss = cfg.w_loss * np.tanh(float(loss_drop) * cfg.loss_scale)

    # 2. 计算一致性奖励
    # 将余弦相似度 [-1, 1] 映射到 [0, 1]。
    cons01 = float((float(consistency) + 1.0) / 2.0)
    r_cons = cfg.w_consistency * cons01

    # 3. 计算分离度奖励项
    # 计算正样本相似度与负样本相似度的差值，并用 tanh 规范化。
    # 模型拉大“类内距离”与“类间距离”的差距，让细胞边界更清晰。引入离散度奖励（r_sep）来增强类间异质性
    r_sep = cfg.w_sep * float(np.tanh(float(sim_pos) - float(sim_neg)))

    # 4. 计算假阴性惩罚项
    # x_rate 是 FNR 相对于 0.5 的比例。使用 x^2 是为了非线性惩罚。将 0.5 设为惩罚基准，模型在检测到 FNR 偏高时会获得非线性的负反馈。
    # 如果 FNR 太高，说明模型把同类细胞推开了，给一个负分（惩罚），迫使智能体反思采样策略。如果fnr一直在0.5左右徘徊，可以适当的调整cfg.w_fnr惩罚权重
    # 通过将随机水平（0.5）作为基准归一化，并引入tanh(x^2)结构
    x_rate = float(fnr_rate) / 0.5
    p_fnr = cfg.w_fnr * float(np.tanh(x_rate * x_rate))

    # 5. 计算比例偏离惩罚项
    # 计算当前 hard_ratio 偏离预设基准（base_ratio, 0.20）的平方。当前批次困难样本比例与预设基准 base_ratio 的偏差，并进行平方级缩放，挖掘细微亚群特征
    # 保证采样的难度，克服了单细胞数据高噪声导致的 FNR 震荡问题
    p_ratio = cfg.w_ratio * float(((float(hard_ratio) - cfg.base_ratio) / (cfg.max_ratio - cfg.min_ratio)) ** 2)

    # 6. 总奖励值合成
    # 总奖励 = (下降奖励 + 一致性奖励 + 分离度奖励) - (FNR惩罚 + 偏离惩罚)
    reward = float(r_loss + r_cons + r_sep - p_fnr - p_ratio)

    # 7. 奖励裁剪
    # 将奖励强制限制在配置文件定义的 [-5.0, 5.0] 范围内。
    # 防止由于某个 Batch 数据异常导致奖励爆炸，破坏强化学习网络的稳定性。
    return float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))



@torch.no_grad()
def calculate_consistency_and_fnr(
        model,
        batch,
        device: torch.device,
        hard_ratio: float,               # 当前动态调整的困难样本采样比例
        *,
        high_conf_quantile: float = 0.1, # 置信度分位数
        min_abs_conf_thr: float = 0.8,   # 最小绝对置信度阈值（保底阈值）
) -> Tuple[float, float, float, float]:
    """返回 (一致性得分, 假阴性率, 每个锚点的平均假阴性数, 置信度阈值)."""

    model.eval()                         # 将模型设为评估模式（关闭 Dropout 等）
    im_q = batch.x.to(device)            # 将细胞特征矩阵搬运到 GPU
    edge_index = batch.edge_index.to(device) # 将图的边索引搬运到 GPU

    # 1. 提取细胞嵌入表示
    z, _ = model.encoder_q(im_q, edge_index) # 使用查询编码器提取特征
    z = torch.nn.functional.normalize(z, dim=1) # 对特征进行 L2 归一化，确保余弦相似度计算准确
    batch_size = int(z.shape[0])         # 获取当前 Batch 的细胞数量

    # 2. 计算图中已知邻居的一致性
    u, v = edge_index                    # 取出图中连接的节点对，u起始点，v终止点
    # 过滤掉采样产生的边缘节点，只保留当前 Batch 核心节点的边，neihgborloader这个函数得到512个核心节点，然后剩下的都是拉进来的邻居节点，边缘节点
    mask = (u < batch_size) & (v < batch_size)
    u_valid, v_valid = u[mask], v[mask]

    if u_valid.numel() > 0:
        # 计算在原始图中本身就是邻居的细胞之间的余弦相似度均值
        # 就是计算起始点和终止点的点积，也就是自身*自身的转置
        consistency = torch.sum(z[u_valid] * z[v_valid], dim=1).mean().item()
    else:
        consistency = 0.0

    # 3. 构建全 batch 相似度矩阵
    sim = torch.matmul(z, z.T)           # 计算 Batch 内所有细胞两两之间的余弦相似度 [N, N]
    sim.fill_diagonal_(-float("inf"))    # 将对角线（细胞自身）设为负无穷，排除自对比

    # 4. 动态确定置信度阈值
    valid = sim[sim > -9999]             # 排除掉被屏蔽的无效数值
    if valid.numel() > 0:
        # 取相似度分布的前 10%  位置的值作为动态阈值
        qv = torch.quantile(valid, float(1.0 - high_conf_quantile)).item()
        # 取动态值和保底值 (0.8) 中的较大者，防止阈值过低
        conf_thr = float(max(qv, float(min_abs_conf_thr)))
    else:
        conf_thr = float(min_abs_conf_thr)

    # 5. 准备计算 FN：排除掉已知的图邻居
    adj_mask = torch.zeros_like(sim, dtype=torch.bool)
    if u_valid.numel() > 0:
        adj_mask[u_valid, v_valid] = True # 在掩码中标记已知的邻居
        adj_mask[v_valid, u_valid] = True
    sim_for_hard = sim.clone()
    sim_for_hard[adj_mask] = -float("inf") # 将已知邻居设为负无穷，只看“非邻居”中相似度很高的样本

    # 6. 挖掘最难的负样本
    k_hard = max(5, int(batch_size * float(hard_ratio))) # 根据 RL 传进来的比例计算 K 值
    _, hard_idx = torch.topk(sim_for_hard, k_hard, dim=1) # 找出每个细胞最相似的 K 个非邻居
    rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k_hard)
    hard_vals = sim[rows, hard_idx]      # 提取这些最难负样本的实际相似度数值

    # 7. 计算假阴性率（FNR）
    if hard_vals.numel() > 0:
        # 物理意义：如果一个“非邻居”的相似度竟然超过了阈值，说明模型可能错误地把同类细胞推开了
        is_fn = (hard_vals > conf_thr)   # 判断是否为假阴性
        fnr_rate = is_fn.float().mean().item()           # 计算假阴性在该 batch 中的总体比例
        fnr_count = is_fn.float().sum(dim=1).mean().item() # 每个细胞平均对应的假阴性个数
    else:
        fnr_rate = 0.0
        fnr_count = 0.0

    return float(consistency), float(fnr_rate), float(fnr_count), float(conf_thr) # 一致性。在所有被当作负样本的细胞对中，有多少比例的相似度超过了阈值。每个细胞在负样本池中有多少个同类。置信度阈值。


class RunningAverage:
    """
    指数移动平均 (EMA) 类。
    用于平滑时间序列数据（如 Loss 或 Reward），使其对瞬时噪声不敏感，反映长期趋势。
    """

    def __init__(self, alpha: float = 0.9):
        """
        初始化平滑器。
        Args:
            alpha (float): 平滑系数（衰减率）。
                           alpha 越大（接近 1.0），历史数据的权重越高，曲线越平滑，但反应越迟钝；
                           alpha 越小（接近 0.0），当前数据的权重越高，反应越灵敏。
        """
        self.alpha = float(alpha)
        # val 存储当前的平滑结果，初始化为 None 表示还未收到第一个数据
        self.val: Optional[float] = None

    def update(self, x: float) -> float:
        """
        输入一个新的数值，更新平滑后的平均值。
        公式: val = (val * alpha) + (x * (1 - alpha))

        Args:
            x (float): 当前时刻观察到的原始数值（如当前 batch 的 Loss）。
        Returns:
            float: 更新后的平滑值。
        """
        x = float(x)

        # 如果是第一个数据点，直接将其设为初始平滑值
        if self.val is None:
            self.val = x
        else:
            # 核心公式：将之前的历史累积值与当前新值进行加权分配
            # self.val * self.alpha: 保留 90% 的历史信息
            # x * (1.0 - self.alpha): 吸收 10% 的当前新信息
            self.val = self.val * self.alpha + x * (1.0 - self.alpha)

        return float(self.val)

def _build_global_adj_dense(
    edge_index: torch.Tensor,  # 输入：图的边索引，形状为 [2, E]，E 是边的数量
    num_nodes: int,           # 输入：节点（细胞）的总数 N
    device: torch.device      # 输入：运算设备
) -> torch.Tensor:
    """
    将稀疏的 edge_index 转换为稠密的布尔邻接矩阵。
    """

    # 1. 内存安全检查
    if num_nodes > 60000:
        print(f"警告: num_nodes={num_nodes} 很大，构建 dense global_adj 可能导致显存溢出 (OOM)")

    # 2. 初始化全零矩阵
    # 创建一个形状为 [num_nodes, num_nodes] 的矩阵，初始值全部为 False (0)。
    global_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    # 3. 填充邻居关系
    # edge_index[0] 是源节点索引，edge_index[1] 是目标节点索引
    # 将所有存在边连接的位置一次性设为 True
    global_adj[edge_index[0], edge_index[1]] = True

    # 4. 填充自环
    # 矩阵对角线代表细胞自身
    global_adj.fill_diagonal_(True)

    # 返回构建好的稠密布尔矩阵
    return global_adj


def train_moco(
        model,
        data,
        *,
        use_mask: bool = False,  # 👑 核心新增：控制是否过滤假负样本
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        num_neighbors: Sequence[int] = (10, 10),
        num_workers: int = 4,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # 纯 Adam，无 weight_decay，对齐 RL
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 👑 新增：即使是基础 MoCo，也需要提前计算好全局邻接矩阵（以备开启 mask 时使用）
    num_nodes = int(data.x.shape[0])
    global_adj_full = _build_global_adj_dense(data.edge_index.to(device), num_nodes, device)

    print(f"🚀 开始基础 MoCo 训练 | Use Mask: {use_mask} | (无困难样本挖掘)")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = len(loader)

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            batch_indices = batch.n_id[: batch.batch_size]

            im_q = batch.x.clone()
            edge_index_q = batch.edge_index

            # 统一增强概率
            im_k, edge_index_k = augment_graph(batch, mask_prob=0.2, dropout_prob=0.2)

            # 👑 核心逻辑：根据开关决定是否传入邻接矩阵
            active_adj = global_adj_full if use_mask else None

            # 注意：这里不传 hard_ratio，模型内部就会走普通的 MoCo 负采样逻辑
            out = model(
                im_q, im_k, edge_index_q, edge_index_k,
                batch_indices=batch_indices,
                global_adj=active_adj
            )

            logits, labels = out[0], out[1]
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"[MoCo] Epoch {epoch:03d} | loss={epoch_loss / n_batches:.4f} ")

    model.eval()
    with torch.no_grad():
        h, _ = model.encoder_q(data.x.to(device), data.edge_index.to(device))
        h = torch.nn.functional.normalize(h, dim=1) # 必须 L2 归一化

    return h.cpu().numpy()

def train_moco_rl(
        model,
        data,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        num_neighbors: Sequence[int] = (10, 10),
        num_workers: int = 8,
        rl_cfg: Optional[RLConfig] = None,
        custom_gamma=0.1,
        update_interval_val=7
) -> np.ndarray:
    cfg = rl_cfg or RLConfig()# 如果没传配置，就用默认的 RLConfig 参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # NeighborLoader 通过采样小规模的运行
    # num_neighbors=(10, 10) 表示采样两层邻居，每层 10 个。
    #https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html
    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # 优化器只更新 encoder_q（Query 编码器）
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_nodes = int(data.x.shape[0])
    # 构建全局邻接矩阵，这样负采样时就会避开它们
    global_adj = _build_global_adj_dense(data.edge_index.to(device), num_nodes, device)

    # global_adj = torch.zeros((num_nodes, num_nodes)).to(device)
    #
    # print("当前处于消融实验模式，假阴性掩码已关闭（global_adj 为全 0）")

    agent = PPOAgent(state_dim=5, action_dim=1,gamma=custom_gamma)
    current_state = np.zeros(5, dtype=np.float32)


    # 将采样比例的“平滑值”初始化为预热期的比例
    ratio_ema = float(cfg.warmup_ratio)
    # 平滑 Loss 的工具。因为单个 batch 的 Loss 跳动很大，用 EMA（指数移动平均）处理后的 Loss 更有参考价值。
    loss_smoother = RunningAverage(alpha=0.9)
    prev_smooth_loss = 2.0  # 初始设想的一个参考 Loss

    # 每间隔 interval 个 batch 更新一次 RL
    update_interval = 10
    step_buffer = {
        'loss': [], 'sim_pos': [], 'sim_neg': [],
        'consistency': [], 'fnr_rate': []
    }

    # 每个反馈区间只决策一次：区间第一个批次采样比例，整个区间沿用，区间结束时与该区间的奖励一起存入记忆
    interval_target = None

    print(f"开始 RL-MoCo 训练 (Interval: {update_interval})...")

    for epoch in range(1, epochs + 1):
        model.train()

        # 用于打印 Epoch 日志的累加器
        epoch_log = {'loss': 0, 'reward': 0, 'ratio': 0, 'fnr': 0}
        n_batches = len(loader)#多少个batch是一个epoch

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            batch_indices = batch.n_id[: batch.batch_size]

            # 1. RL 决策阶段
            if epoch <= cfg.warmup_epochs:
                target_ratio = float(cfg.warmup_ratio)
            else:
                if interval_target is None:
                    raw_action = agent.select_action(current_state)
                    interval_target = map_action_to_ratio(raw_action, cfg)
                    #print("消融实验：关闭安全门逻辑")
                    #消融实验：关闭安全门逻辑
                    #安全门：FNR 过高时强制降比例，防止模型开始后突然的飙升
                    if float(current_state[2]) > cfg.safety_fnr_rate:
                         interval_target = min(interval_target, float(cfg.safety_ratio))
                target_ratio = interval_target

            # 平滑ppo的决策
            ratio_ema = cfg.ema_alpha * ratio_ema + (1.0 - cfg.ema_alpha) * float(target_ratio)
            # 最后肚饿认为限制困难样本的比例
            hard_ratio = float(np.clip(ratio_ema, cfg.min_ratio, cfg.max_ratio))

            # 2. 训练阶段
            # 清除优化器中上一个批次（Batch）计算留下的梯度
            optimizer.zero_grad()
            im_q = batch.x.clone()
            #  print("消融q编码器")
            #  im_q, edge_index_q = augment_graph(batch, mask_prob=0.2, dropout_prob=0.2)
            im_k, edge_index_k = augment_graph(batch, mask_prob=0.2, dropout_prob=0.2)

            out = model(
                im_q, im_k, batch.edge_index, edge_index_k,
                batch_indices=batch_indices,
                hard_ratio=hard_ratio,
                global_adj=global_adj
            )

            logits, labels, sim_pos, sim_neg, _ = out[:5]
            #计算 InfoNCE Loss
            loss = criterion(logits, labels)
            #反向传播
            loss.backward()
            # 利用 Adam 算法更新模型的权重
            optimizer.step()

            # 3. 收集指标进入缓存
            consistency, fnr_rate, _, _ = calculate_consistency_and_fnr(
                model, batch, device, hard_ratio,
                high_conf_quantile=cfg.high_conf_quantile,
                min_abs_conf_thr=cfg.min_abs_conf_thr,
            )
            model.train()

            step_buffer['loss'].append(loss.item())
            step_buffer['sim_pos'].append(float(sim_pos))
            step_buffer['sim_neg'].append(float(sim_neg))
            step_buffer['consistency'].append(consistency)
            step_buffer['fnr_rate'].append(fnr_rate)

            # 4. 间隔式强化学习更新.为了解决单细胞数据噪声大、批次反馈不稳定的问题.显著提高策略更新的置信度
            if (batch_idx + 1) % update_interval == 0 or (batch_idx + 1) == n_batches:
                avg_batch_loss = np.mean(step_buffer['loss'])
                avg_sim_pos = np.mean(step_buffer['sim_pos'])
                avg_sim_neg = np.mean(step_buffer['sim_neg'])
                avg_fnr = np.mean(step_buffer['fnr_rate'])
                avg_consistency = np.mean(step_buffer['consistency'])

                # 计算 Loss 变化趋势
                curr_smooth = loss_smoother.update(avg_batch_loss)
                loss_change = float(prev_smooth_loss - curr_smooth)
                prev_smooth_loss = curr_smooth

                # 计算 Reward
                fnr_signal = avg_fnr if epoch > cfg.warmup_epochs else 0.0
                reward = calculate_reward(
                    loss_drop=loss_change,
                    fnr_rate=fnr_signal,
                    consistency=avg_consistency,
                    hard_ratio=hard_ratio,
                    sim_pos=avg_sim_pos,
                    sim_neg=avg_sim_neg,
                    cfg=cfg,
                )

                # 构造状态并更新 Agent
                next_state = build_state(
                    avg_sim_pos, avg_sim_neg, loss_change,
                    fnr_signal, avg_consistency, hard_ratio
                )

                if epoch > cfg.warmup_epochs:
                    agent.store_transition(float(reward), next_state, False)# 存入记忆
                    # is_last_interval = ((batch_idx + 1) == n_batches)
                    # agent.store_transition(float(reward), next_state, is_last_interval)
                    #agent.update()
                    if len(agent.memory) >= update_interval_val:
                        agent.update()

                current_state = next_state
                interval_target = None

                # 记录 reward 到日志
                epoch_log['reward'] += float(reward) * (len(step_buffer['loss']) / n_batches)
                # 清空缓存
                for k in step_buffer: step_buffer[k] = []

            # 累加基础指标用于 Epoch 日志打印
            epoch_log['loss'] += loss.item()
            epoch_log['ratio'] += hard_ratio
            epoch_log['fnr'] += fnr_rate
        #新加两行
        #if epoch > cfg.warmup_epochs:
            #agent.update()

        # 打印本轮统计
        status = "Warm-up" if epoch <= cfg.warmup_epochs else "RL-Active"
        print(
            f"Epoch: {epoch:03d} [{status}] | "
            f"Loss: {epoch_log['loss'] / n_batches:.4f} | Reward: {epoch_log['reward']:.2f} | "
            f"Ratio: {epoch_log['ratio'] / n_batches:.3f} | FNR: {epoch_log['fnr'] / n_batches:.3f}"
        )

    # 提取最终嵌入
    model.eval()
    with torch.no_grad():
        h, _ = model.encoder_q(data.x.to(device), data.edge_index.to(device))
        h = torch.nn.functional.normalize(h, dim=1)
    return h.cpu().numpy()


def train_moco_no_rl(
        model,
        data,
        *,
        static_ratio: float = 0.2,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        num_neighbors: Sequence[int] = (10, 10),
        num_workers: int = 4,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # 去掉 weight_decay=1e-4，对齐 RL
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_nodes = int(data.x.shape[0])
    global_adj = _build_global_adj_dense(data.edge_index.to(device), num_nodes, device)

    print(f"🚀 开始静态消融训练 | Fixed Ratio: {static_ratio} | 已完全对齐 RL 版本")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = len(loader)

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            batch_indices = batch.n_id[: batch.batch_size]
            im_q = batch.x.clone()
            edge_index_q = batch.edge_index

            im_k, edge_index_k = augment_graph(batch, mask_prob=0.2, dropout_prob=0.2)

            out = model(
                im_q, im_k, edge_index_q, edge_index_k,
                batch_indices=batch_indices,
                hard_ratio=float(static_ratio),
                global_adj=global_adj
            )

            logits, labels = out[0], out[1]
            loss = criterion(logits, labels)

            loss.backward()
            # 去掉梯度裁剪，对齐 RL
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"[MoCo Ablation] Epoch {epoch:03d} | loss={epoch_loss / n_batches:.4f} ")

    model.eval()
    with torch.no_grad():
        h, _ = model.encoder_q(data.x.to(device), data.edge_index.to(device))
        h = torch.nn.functional.normalize(h, dim=1) # 统一执行 L2 归一化

    return h.cpu().numpy()