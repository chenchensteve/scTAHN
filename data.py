import scanpy as sc
import numpy as np
import pandas as pd
import torch
import anndata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import scipy.sparse



class SingleCellGraphProcessor:
    def __init__(self,
                 hvg_n=4000,
                 knn_k=15,
                 seeds=[42, 66, 88, 2023, 2024]):
        """
        通用的单细胞图数据预处理类
        Args:
            hvg_n: 高变基因数量 (默认4000)
            knn_k: 构建图时的邻居数量 (默认15)
            seeds: 用于划分数据集的随机种子列表
        """
        self.hvg_n = hvg_n
        self.knn_k = knn_k
        self.seeds = seeds
        self.le = preprocessing.LabelEncoder()  # 标签编码器

    def process(self, file_path, label_col='cell_type'):
        """
        主处理函数

            file_path: .h5ad 文件路径
            label_col: 原始数据中存放在 .obs 里的细胞类型列名
        Returns:
            processed_adata: 处理后的 AnnData 对象 (包含 size_factors 等)
            data_list: 包含 5 个种子对应划分信息的列表。
                       每个元素包含 {'train_idx', 'val_idx', 'test_idx', 'ood_idx', 'pyg_data'}#pyg_data封装了图神经网络（GNN）训练所需的所有组件
        """
        print(f"加载数据集{file_path}")
        adata = anndata.read_h5ad(file_path)

        # 1. 基础过滤与标签处理
        # 统一标签列名为 'cell'
        if label_col in adata.obs.keys():
            adata.obs['cell'] = adata.obs[label_col]
        else:
            raise ValueError(f"Column '{label_col}' not found in adata.obs")

        # 过滤掉细胞数量极少的类别 (<=10)
        adata = self._filter_rare_cells(adata, min_count=10)

        # 2. 提取高变基因 (Seurat v3 / Cell Ranger 风格)
        print(f"使用cell ranger方式提取高变基因")
        adata = self._select_hvg(adata, n_top=self.hvg_n)

        # 3. 归一化、Log化、标准化
        print("归一化，log化，标准化")
        adata = self._normalize_and_scale(adata)

        # 4. 准备 PyTorch 需要的特征 (X) 和 标签 (y)
        x_tensor = torch.from_numpy(adata.X).float()

        # 标签编码
        y_encoded = self.le.fit_transform(adata.obs['cell'].values)
        y_tensor = torch.from_numpy(y_encoded).long()

        # 5. 构建 KNN 图
        print("构建细胞相似图")
        edge_index = self._build_knn_graph(adata.X, k=self.knn_k)

        # 6. 创建 PyG 对象，data包括特征，标签，边索引
        pyg_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        pyg_data.num_nodes = adata.shape[0]

        # 7. 数据划分 226
        print("划分数据")
        split_info_list = self._split_data(adata.obs['cell'].values, self.seeds)

        # 将 PyG Data 对象放入每个种子的结果中
        for item in split_info_list:
            item['pyg_data'] = pyg_data

        print("预处理结束")
        return adata, split_info_list, self.le#数据，数据划分列表，标签查找器

    def _filter_rare_cells(self, adata, min_count=10):
        """过滤掉样本数极少的细胞类型"""
        counts = adata.obs['cell'].value_counts()
        keep_classes = counts[counts > min_count].index
        adata = adata[adata.obs['cell'].isin(keep_classes)].copy()
        return adata

    def _select_hvg(self, adata, n_top):
        """
        提取高变基因的核心逻辑
        必须使用 Raw Counts (整数)
        """
        #判断是否是稀疏矩阵或稀疏数组
        if scipy.sparse.issparse(adata.X):
            X = adata.X.toarray()#换成稠密矩阵，方便计算欧式距离
        else:
            X = adata.X
        X = np.ceil(X).astype(int)#向上取整，变成整数

        # 创建临时 adata 进行计算，避免破坏原数据结构
        # temp_adata = sc.AnnData(X, dtype=np.float32)
        # temp_adata.obs = adata.obs.copy()#存储 细胞（行）元数据 的表格
        # temp_adata.var = adata.var.copy()#存储 基因（列）元数据 的表格

        # 【核心修复】：在创建 AnnData 时直接传入 obs 和 var，这样细胞名和基因名索引会被完美继承！
        temp_adata = sc.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy(), dtype=np.float32)

        # 基础过滤
        sc.pp.filter_genes(temp_adata, min_counts=3)#计算每个基因在所有细胞中的表达量之和
        sc.pp.filter_cells(temp_adata, min_counts=1)#计算每个细胞里所有基因表达量的总和

        # 核心
        sc.pp.highly_variable_genes(
            temp_adata,
            n_top_genes=n_top,
            flavor='cell_ranger',
            min_mean=0.0125,
            max_mean=4,
            min_disp=0.5,
            subset=True
        )
        # 1.在原始数据集上计算每个基因的均值和方差
        # 2.对所有基因的均值和方差取对数ln
        # 3.使用局部多项式回归拟合一条曲线，表示在该表达量水平下，基因通常具有的技术方差是多少
        # 4.算出每个基因的预期方差
        # 5.计算标准方差，（原始表达量 - 均值） / 预期方差的开根号
        # 6.排序

        return temp_adata

    def _normalize_and_scale(self, adata):
        """
        归一化流水线 (对应原代码 normalize_adata)
        1. 备份 Raw Counts (用于 ZINB Loss)
        2. 计算 Size Factors
        3. Log1p
        4. Scale
        """
        # 1. 备份 Raw Counts
        adata.raw = adata.copy()

        # 2. 计算 Size Factors
        n_counts = adata.X.sum(axis=1)#（N，1），每一个细胞计算所有的基因表达量的总和
        if isinstance(n_counts, np.matrix):
            n_counts = n_counts.A1  # 如果要是一个一列则必须转为一维数组一行

        # 归一化 (Normalize per cell)
        sc.pp.normalize_per_cell(adata, min_counts=0)#不过滤任何细胞

        # 计算因子：当前细胞总数 / 所有细胞总数的中位数  用于ZINB
        adata.obs['size_factors'] = n_counts / np.median(n_counts)

        # 3. 取对数 (Log1p)
        sc.pp.log1p(adata)

        # 4. 标准化 (Scale: 均值0，方差1)
        sc.pp.scale(adata)

        return adata

    def _build_knn_graph(self, X, k):
        """
        构建 KNN 图
        使用 欧氏距离 (l2)
        """
        # 确保是 Dense Matrix 有0的矩阵
        if scipy.sparse.issparse(X):
            X = X.toarray()#解压缩，变成稠密矩阵

        # 使用 sklearn 计算 KNN
        # n_neighbors = k + 1 (因为包含自己)
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='l2').fit(X)#找出自己最近的领居
        _, indices = nbrs.kneighbors(X)


        source_nodes = []#边起点
        target_nodes = []#边终点

        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:

                source_nodes.append(i)
                target_nodes.append(neighbor)

        # （2，E）https://docs.pytorch.org/docs/stable/tensors.html
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        return edge_index

    def _split_data(self, labels, seeds):
        """
        划分 ID/OOD 和 Train/Val/Test
        """
        #https://pandas.pydata.org/docs/reference/api/pandas.Series.html
        y_series = pd.Series(labels)#就是变成带索引的，原本只是一个普通的列表或数组，方便后面进行统计

        results = []

        for seed in seeds:
            np.random.seed(seed)

            # 1. 确定 OOD 类别 (人数最少)
            ood_class = y_series.value_counts().idxmin()

            # 2. 获取索引
            all_indices = np.arange(len(labels))
            is_ood = (labels == ood_class)

            ood_idx = all_indices[is_ood]  # 新型细胞
            id_idx_all = all_indices[~is_ood]  # 普通细胞

            # 3. 进行划分 (6:2:2)
            # 先切出 20% 作为 Test
            train_val_idx, test_idx = train_test_split(
                id_idx_all, test_size=0.2, random_state=seed
            )
            # 再从剩下的 80% 里切出 25% 作为 Val (即总数的 20%)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.25, random_state=seed
            )

            # 4. 转换为 Tensor 格式 (模型需要 LongTensor 索引)
            results.append({
                'seed': seed,
                'ood_class': ood_class,
                'train_idx': torch.tensor(train_idx, dtype=torch.long),
                'val_idx': torch.tensor(val_idx, dtype=torch.long),
                'test_idx': torch.tensor(test_idx, dtype=torch.long),
                'ood_idx': torch.tensor(ood_idx, dtype=torch.long),
                'id_idx': torch.tensor(id_idx_all, dtype=torch.long)
            })

        return results




if __name__ == "__main__":
    # 1. 初始化处理器
    processor = SingleCellGraphProcessor(hvg_n=4000, knn_k=15)

    # 2. 【关键！】运行处理，生成 adata
    # 这一行必须在验证代码之前！而且变量名必须叫 adata
    adata, split_data_list, le = processor.process('../data/10x_5cl_processed.h5ad', label_col='cell_type')
