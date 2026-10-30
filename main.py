import numpy as np
import torch
import os
import time
#导入自定义模块
from data import SingleCellGraphProcessor
from model import MoCo,GATEncoder
from train import train_moco_rl,evaluate_clustering_leiden,train_moco,evaluate_clustering_leiden2,evaluate_clustering_kmeans
from train import RLConfig
my_rl_cfg = RLConfig(
    max_ratio=0.35,
    warmup_epochs=10,
    # min_ratio = 0.1117,
    # min_abs_conf_thr = 0.6809
    # safety_fnr_rate=0.1862,
    # safety_ratio=0.1589
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from train import set_seed, evaluate_clustering_target_k,evaluate_clustering_multistart_leiden

import umap
import matplotlib.pyplot as plt



TARGET_RESOLUTION = 0.05

VISUALIZATION_AVAILABLE = True
DATA_ROOT = os.environ.get('SCTAHN_DATA_DIR', './data/')  # 数据集存放根目录

# --- 待运行的数据集列表及对应的标签列 ---
# 格式: (文件名, 标签列名)
TARGET_DATASETS = [
    #  ("Klein.h5ad", "cluster"),
    #  ("Pollen.h5ad", "cluster"),
    #  ("10x_5cl_processed.h5ad", "cell"),
    # ("CelSeq2_5cl_processed.h5ad", "cell"),
    # ("biase.h5ad", "cluster"),
    # ("Quake_10x_Bladder.h5ad", "cluster"),
    # ("deng.h5ad", "cluster"),
    # ("Quake_10x_Spleen.h5ad", "cluster"),
    # ("Quake_Smart-seq2_Heart.h5ad", "cluster"),
    ("Quake_Smart-seq2_Limb_Muscle.h5ad", "cluster"),
    # ("Quake_10x_Limb_Muscle.h5ad", "cluster"),
    # ("Camp.h5ad", "cluster"),
    # ("Quake_Smart-seq2_Lung.h5ad", "celltype"),
    # ("Pancreas_human1.h5ad", "celltype"),
    #("Bach.h5ad","cluster"),
    #("Plasschaert.h5ad","cluster"),
    # ("Romanov.h5ad", "celltype"),
    # ("darmanis.h5ad","cluster"),
    # ("goolam.h5ad","cluster"),
    # ("Zeisel.h5ad","cluster"),
    # ("Baron.h5ad","cluster"),
    # ("Quake_Smart-seq2_Trachea.h5ad","cluster"),
    # ("Covid_saliva_processed.h5ad","cluster"),
    # ("Covid_SP_processed.h5ad","cluster"),
    ("My_Limb_Muscle_Real.h5ad", "true_label"),
    #("Kidney_10X_Pristine.h5ad", "true_label")



]



def main():

    #file_path = os.path.join(DATA_ROOT, 'Klein.h5ad')



    summary_results = []

    for file_name, label_col in TARGET_DATASETS:
        file_path = os.path.join(DATA_ROOT, file_name)

        if not os.path.exists(file_path):
            print(f"跳过：文件不存在 -> {file_path}")
            continue

        print(f"\n" + "=" * 60)
        print(f"正在处理数据集: {file_name} | 标签列: {label_col}")
        print("=" * 60)
        batch_size_val = 512


        all_ari = []
        all_nmi = []

        timestamp = time.strftime("%Y%m%d-%H:%M:%S")

        try:
            #1. 初始化处理器
            #设置高变基因数(4000)和KNN邻居数(15)
            processor = SingleCellGraphProcessor(hvg_n=4000, knn_k=15)

            #2. 运行处理
            print(f"[{timestamp}] 开始处理数据集: {file_path}")
            #执行完整的预处理流程：加载、过滤、HVG、归一化、构建图、划分数据
            adata, split_data_list, le = processor.process(file_path, label_col=label_col)

            # import numpy as np
            #
            #
            # save_path = os.path.join(os.environ.get("SCTAHN_EXPORT_DIR", "./data/"), "Kidney_10X_Pristine")
            # os.makedirs(save_path, exist_ok=True)
            #
            # # 1. 导出 Z-score 归一化后的数据 (用于 DAE 编码器输入)
            # np.save(f"{save_path}/x_norm.npy", adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray())
            #
            # # 2. 导出 Raw Counts (用于 ZINB Loss 计算似然)
            # # 注意：scLEGA 需要原始整数计数
            # raw_counts = adata.raw.X if isinstance(adata.raw.X, np.ndarray) else adata.raw.X.toarray()
            # np.save(f"{save_path}/x_raw.npy", raw_counts)
            #
            # # 3. 导出 Size Factors
            # np.save(f"{save_path}/size_factors.npy", adata.obs['size_factors'].values)
            #
            # # 4. 导出标签 (用于最后统一评估)
            # np.save(f"{save_path}/y_true.npy", adata.obs['cell'].values)
            #
            # print("标准数据已导出，请将文件夹拷贝到目录下")
            #
            # exit()
            #3. 初始化模型参数
            in_channels = adata.X.shape[1]
            latent_dim = 256  #嵌入维度
            proj_dim = 128 #投影维度
            queue_size = 4096  #队列大小
            momentum = 0.99  #动量
            temperature = 0.2  #温度
            gat_heads = 4  #GAT 头数
            use_mlp = True  #MLP 投影头


            print(f"\n模型参数: 输入={in_channels}, Latent={latent_dim}, Proj={proj_dim}")
            print(f"真实类别数: {len(le.classes_)}")

            #3. 循环处理所有种子
            for split_info in split_data_list:
                seed = split_info['seed']
                set_seed(seed)

                data = split_info['pyg_data']
                true_labels = data.y

                start_str = time.strftime("%H:%M:%S")
                start_ticks = time.time()
                print(f"  [Seed {seed}] 开始训练: {start_str}")



                model = MoCo(
                    base_encoder=GATEncoder,
                    num_genes=in_channels,
                    dim=latent_dim,
                    out_dim=proj_dim,
                    r=queue_size,
                    m=momentum,
                    T=temperature,
                    head=gat_heads,
                    mlp=use_mlp
                )

                #训练
                final_embeddings= train_moco_rl(model, data, epochs=200, lr=0.001, batch_size=batch_size_val, rl_cfg=my_rl_cfg)


                #final_embeddings = train_moco(model, data, epochs=200, lr=0.001, batch_size=batch_size_val)

                #评估
                #pred_labels, metrics = evaluate_clustering_leiden(final_embeddings, true_labels, resolution="auto")
                #, metrics = evaluate_clustering_leiden(final_embeddings, true_labels)
                #print(f"Final Test Result: ARI={metrics['ari']:.4f}, NMI={metrics['nmi']:.4f}, Res={metrics['res']:.4f}, K={metrics['n_clusters']}")
                pred_labels, metrics = evaluate_clustering_target_k(final_embeddings, true_labels)

                #pred_labels, metrics = evaluate_clustering_leiden2(final_embeddings, true_labels)
                all_ari.append(metrics['ari'])  # 核心修复
                all_nmi.append(metrics['nmi'])  # 核心修复

                cost_time = time.time() - start_ticks
                end_str = time.strftime("%H:%M:%S")
                print(
                    f"  [ Seed {seed}] 结束时间: {end_str} | 耗时: {cost_time:.2f} 秒 | 当前 ARI: {metrics['ari']:.4f}\n")
                #可视化
                if VISUALIZATION_AVAILABLE:
                    print(f"正在生成可视化图表...")
                    #降维：初始化 UMAP 降维器，n_components嵌入空间的维度。默认值为 tx2，以便于可视化.min_dist嵌入点之间的有效最小距离。
                    #https://umap-learn.readthedocs.io/en/latest/api.html
                    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                    #对高维嵌入进行 UMAP 降维到 2D (N, 2)
                    embedding_2d = reducer.fit_transform(final_embeddings)#Fit计算所有细胞之间的距离。Transform计算出每个细胞在二维平面（X轴, Y轴）上的具体坐标。
                    #https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/pyplot.py#L1621-L1781 宽16，高7
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                    #左图：Ground Truth (真实标签)
                    scatter1 = ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                                           c=true_labels.cpu().numpy(), cmap='Spectral', s=5)
                    ax1.set_title(f'Ground Truth (Seed {seed})')
                    ax1.set_xlabel('UMAP 1')
                    ax1.set_ylabel('UMAP 2')
                    legend1 = ax1.legend(*scatter1.legend_elements(), loc="lower left", title="Class")
                    ax1.add_artist(legend1)

                    #右图：Leiden Clustering
                    #Leiden 标签可能是字符串，转换为 int 以便画图配色

                    pred_labels_int = pred_labels.astype(int)
                    scatter2 = ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                                           c=pred_labels_int, cmap='Set1', s=5)
                    n_leiden_clusters = len(np.unique(pred_labels_int))
                    ax2.set_title(f'Leiden Clustering (Res=0.5, K={n_leiden_clusters})\nARI={metrics["ari"]:.3f}, NMI={metrics["nmi"]:.3f}')
                    ax2.set_xlabel('UMAP 1')

                    #自动调整https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
                    plt.tight_layout()
                    save_name = f'gat_moco_leiden_seed_{seed}.png'
                    plt.savefig(save_name)
                    plt.close()  #关闭画布，防止内存堆积
                    print(f"已保存: {save_name}")

            if len(all_ari) > 0:
                avg_ari = np.mean(all_ari)
                std_ari = np.std(all_ari)
                avg_nmi = np.mean(all_nmi)
                std_nmi = np.std(all_nmi)

                print("\n" + "=" * 30)
                print(f"最终统计结果 (时间戳: {timestamp})")
                print(f"运行种子数: {len(all_ari)}")
                print(f"ARI: {avg_ari:.4f} ± {std_ari:.4f}")
                print(f"NMI: {avg_nmi:.4f} ± {std_nmi:.4f}")
                print("=" * 30)

                # 可选：将统计结果保存到本地文件
                with open(f"summary_stats_{timestamp}.txt", "w") as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Seeds: {len(all_ari)}\n")
                    f.write(f"ARI: {avg_ari:.4f} (+/- {std_ari:.4f})\n")
                    f.write(f"NMI: {avg_nmi:.4f} (+/- {std_nmi:.4f})\n")

        except Exception as e:#捕获错误命名为e
            print(f"\n运行时发生错误：{e}")
            import traceback#追踪错误足迹的工具包
            #https://docs.python.org/3/library/traceback.html
            traceback.print_exc()#打印完整的堆栈追踪 print_exception(sys.exception(), limit=limit, file=file, chain=chain).


    # 8. 最终打印所有数据集的汇总报告
    if summary_results:
        print("\n\n" + "实验最终汇总报告".center(50, "="))
        df_final = pd.DataFrame(summary_results)
        # 生成 Mean ± Std 字符串方便复制到论文
        df_final['ARI_Summary'] = df_final.apply(lambda x: f"{x['ARI_mean']:.4f} ± {x['ARI_std']:.4f}", axis=1)
        df_final['NMI_Summary'] = df_final.apply(lambda x: f"{x['NMI_mean']:.4f} ± {x['NMI_std']:.4f}", axis=1)

        print(df_final[['Dataset', 'ARI_Summary', 'NMI_Summary']].to_string(index=False))
        df_final.to_csv(f"batch_summary_{timestamp}.csv", index=False)
        print("=" * 50)


if __name__ == "__main__":
    main()

