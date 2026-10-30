import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, num_genes):
        super(LinearDecoder, self).__init__()
        # 简单的一层或两层 MLP，负责把 h (256维) 还原回原始基因数 (如 4000维)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, h):
        return self.net(h)
#辅助函数：全连接块
#用于构建 MoCo v2 中的 MLP 投影头，最后一层drop为0
def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        #线性层
        nn.Linear(in_features, out_features, bias=True),
        #激活函数
        nn.ReLU(),
        #随机丢弃神经元，防止过拟合
        nn.Dropout(p=p_drop),
    )


#GAT 编码器
class GATEncoder(nn.Module):
    def __init__(self, num_genes, latent_dim, num_heads=4, dropout=0.6, fc=None):
        """
        初始化 GAT 编码器
        Args:
            num_genes: 输入特征维度
            latent_dim: 最终输出的嵌入维度
            num_heads: 多头注意力的头数
            dropout: Dropout 概率。  https://arxiv.org/abs/1710.10903
            （During training, we apply L2 regularization with λ = 0.0005. Furthermore, dropout (Srivastava et al., 2014) with p = 0.6 is applied to
both layers’ inputs, as well as to the normalized attention coefficients (critically, this means that at
each training iteration, each node is exposed to a stochastically sampled neighborhood).）
            fc: 可选的全连接层 (在 MoCo 中会被替换为 MLP 投影头)
        """
        super(GATEncoder, self).__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        #第一层 GAT
        #输入维度: num_genes
        #输出维度: 128 * heads
        #https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
        #concat（布尔值，可选）– 如果设置为 true False，则多头注意力机制的结果将取平均值而不是连接起来。
        self.gat_layer_1 = GATConv(
            in_channels=num_genes,  # 输入特征数
            out_channels=128,  # 每个头的输出特征数
            heads=num_heads,  # 注意力头数
            dropout=dropout,  # 注意力系数的 dropout
            concat=True,# 拼接多头结果
            negative_slope=0.2)#2.1 当输入是负数时，不要直接把这一刀切成 0，而是保留 0.2 倍的数值。

        #计算第二层的输入维度 (128 * 4 = 512)
        in_dim2 = 128 * num_heads

        #第二层 GAT
        # 输入维度: 128 * heads
        # 输出维度: latent_dim
        self.gat_layer_2 = GATConv(
            in_channels=in_dim2,  # 输入维度
            out_channels=latent_dim,  # 输出维度
            heads=num_heads,  # 注意力头数
            concat=False,  # 不拼接，而是取平均 (回归到 latent_dim)
            dropout=dropout,
            negative_slope=0.2)

        # 投影头占位符 (在 MoCo 初始化时会被覆盖为 MLP)
        self.fc = fc

    def forward(self, x, edge_index):
        """
        前向传播逻辑
        x: 节点特征矩阵 [N, num_genes]
        edge_index: 图的连接关系 [2, E]
        """
        #Layer 1
        x = self.gat_layer_1(x, edge_index)
        # 激活函数 3.3Transductive learning  followed by an exponential linear unit (ELU) (Clevert et al., 2016) nonlinearity.
        x = F.elu(x)
        # Dropout：防止过拟合
        x = F.dropout(x, p=0.4, training=self.training)

        #Layer 2
        h = self.gat_layer_2(x, edge_index)

        if self.fc is not None:
            #如果有投影头，z 是投影后的结果
            z = self.fc(h)
        else:
            z = h

        return h, z


class MoCo(nn.Module):
    """
    自定义的 MoCo (Momentum Contrast) 模型类。
    集成了 GAT 编码器、动量更新、负样本队列、图邻居过滤和 RL 动态负采样。
    """

    def __init__(self, base_encoder, num_genes=4000, dim=256, out_dim=128, r=4096, m=0.99, T=0.2, head=4, mlp=True):
        """
        初始化函数。
        Args:
            base_encoder: 基础编码器类 (这里是 GATEncoder)。
            num_genes: 输入特征维度 (基因数)。
            dim: 隐藏层维度。
            out_dim: 输出投影维度 (Projection Head 的输出)。
            r: 负样本队列大小 (Queue Size, e.g., 4096)。
            m: 动量系数 (Momentum)，控制 Key Encoder 更新的平滑程度。
            T: 温度系数 (Temperature)，用于缩放 Logits，控制分布的尖锐程度。
            head: GAT 的注意力头数。
            mlp: 是否使用 MLP 投影头 (MoCo v2 的标配)。
        """
        super(MoCo, self).__init__()  # 调用父类 nn.Module 的初始化

        # 保存超参数
        self.r = r
        self.m = m
        self.T = T

        # 1. 创建双编码器 (Query 和 Key)
        # Query Encoder: 梯度更新的主网络
        self.encoder_q = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)
        # Key Encoder: 动量更新的目标网络
        self.encoder_k = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)

        # 2. 构建 MLP 投影头 (Projection Head)
        if mlp:
            hidden_dim = 256
            # Query 的投影头: Linear -> ReLU -> Linear
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            # Key 的投影头: 结构完全相同
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

        # 3. 初始化 Key Encoder
        # 遍历 Q 和 K 的所有参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # 将 Q 的初始权重复制给 K，确保训练起点一致
            param_k.data.copy_(param_q.data)
            # 关键：关闭 K 的梯度计算。K 不通过反向传播更新，而是通过动量公式更新。
            param_k.requires_grad = False

        # 4. 初始化负样本队列 (Queue)
        # register_buffer: 注册一个张量为模型的"状态"，它会随模型保存/加载，但不是可训练参数(parameter)。
        # queue: [out_dim, r]，存储 r 个历史负样本的特征向量。
        self.register_buffer("queue", torch.randn(out_dim, r))
        # 对随机初始化的队列进行归一化，因为对比学习基于余弦相似度。
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # 关键修改 1: 增加 queue_idx 存储对应的节点索引
        # 这是一个并行队列，专门存储 queue 中每个特征向量对应的原始节点 ID (Cell Index)。
        # 初始化为 -1，表示队列初始时没有有效数据。
        # 形状: [r,] (一维长整型)
        self.register_buffer("queue_idx", torch.full((r,), -1, dtype=torch.long))

        # 5. 初始化队列指针
        # 记录当前写到了队列的哪个位置 (0 到 r-1)。
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.decoder = LinearDecoder(latent_dim=dim, num_genes=num_genes)

    @torch.no_grad()  # 装饰器：该函数内不计算梯度，节省显存和计算资源
    def _momentum_update_key_encoder(self):
        """
        动量更新：theta_k = m * theta_k + (1-m) * theta_q
        让 Key Encoder 缓慢跟随 Query Encoder 变化，保持特征的一致性。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # 原地更新 param_k
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, indices):
        """
        队列更新操作：入队新 batch，出队最旧数据。
        同时维护 特征队列(queue) 和 索引队列(queue_idx)。
        """
        batch_size = keys.shape[0]  # 获取当前 batch 大小
        ptr = int(self.queue_ptr)  # 获取当前指针位置

        # Write the tail first, then continue from the queue beginning.
        tail_size = min(batch_size, self.r - ptr)
        self.queue[:, ptr:ptr + tail_size] = keys[:tail_size].T
        self.queue_idx[ptr:ptr + tail_size] = indices[:tail_size]
        remaining = batch_size - tail_size
        if remaining > 0:
            self.queue[:, :remaining] = keys[tail_size:].T
            self.queue_idx[:remaining] = indices[tail_size:]

        # 移动指针，并对队列长度取模，实现循环队列
        ptr = (ptr + batch_size) % self.r
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, edge_index_q, edge_index_k=None,
                batch_indices=None, hard_ratio=None, global_adj=None, is_eval=False):
        """
        Args:
            global_adj: 全局布尔邻接矩阵 [Num_Nodes, Num_Nodes]，用于查找真邻居
        """
        if is_eval:
            h, _ = self.encoder_q(im_q, edge_index_q)
            return h

        if edge_index_k is None: edge_index_k = edge_index_q

        # 1. Compute Features
        # 此时 q 的形状是 [N_total, dim] (例如 1095)，包含了邻居节点
        _, q = self.encoder_q(im_q, edge_index_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            # 此时 k 的形状也是 [N_total, dim]
            _, k = self.encoder_k(im_k, edge_index_k)
            k = nn.functional.normalize(k, dim=1)

        # NeighborLoader 保证了 Batch 中的前 batch_size 个节点是 Target Nodes

        if batch_indices is not None:
            # batch_indices 的长度就是真正的 batch_size (例如 512)
            curr_batch_size = batch_indices.shape[0]

            # 只保留前 512 个节点
            q = q[:curr_batch_size]
            k = k[:curr_batch_size]


        # 2. Positive Logits
        # 现在 q, k 形状都是 [512, dim]，可以正常计算了
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # 3. Negative Logits
        queue_clone = self.queue.clone().detach()
        # l_neg 形状变为 [512, 4096]
        l_neg = torch.einsum('nc,ck->nk', [q, queue_clone])

        # 真正的邻居过滤逻辑

        if batch_indices is not None:
            queue_indices = self.queue_idx.clone().detach()

            # final_mask 形状 [512, 4096]
            final_mask = torch.zeros_like(l_neg, dtype=torch.bool)

            # 自身过滤
            mask_self = batch_indices.unsqueeze(1) == queue_indices.unsqueeze(0)
            final_mask = final_mask | mask_self

            # 邻居过滤
            if global_adj is not None:
                valid_queue_mask = (queue_indices != -1)
                valid_queue_indices = queue_indices[valid_queue_mask]

                if len(valid_queue_indices) > 0:
                    neighbor_mask = global_adj[batch_indices][:, valid_queue_indices]
                    #final_mask[:, valid_queue_mask] = final_mask[:, valid_queue_mask] | neighbor_mask
                    #消融假阴性
                    # 确保两边都是布尔型再进行位运算
                    final_mask[:, valid_queue_mask] = final_mask[:, valid_queue_mask].bool() | neighbor_mask.bool()

            l_neg.masked_fill_(final_mask, -float('inf'))

        # Hard Negative Selection
        if hard_ratio is not None:
            num_neg = l_neg.shape[1]
            k_hard = int(num_neg * hard_ratio)
            k_hard = max(10, min(k_hard, num_neg))

            topk_values, _ = torch.topk(l_neg, k_hard, dim=1)

            with torch.no_grad():
                valid_hard_mask = topk_values > -9999
                valid_hard_values = topk_values[valid_hard_mask]

                if valid_hard_values.numel() > 0:
                    high_conf_count = (valid_hard_values > 0.9).sum().item()
                    fnr_ratio = high_conf_count / valid_hard_values.numel()
                    sim_neg_mean = valid_hard_values.mean().item()
                else:
                    sim_neg_mean = -1.0

            l_neg = topk_values
            sim_pos_mean = l_pos.mean().item()

        else:
            sim_pos_mean = l_pos.mean().item()
            sim_neg_mean = l_neg.mean().item()
            fnr_ratio = 0.0

        # 4. Final Logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # 5. Update Queue
        if batch_indices is not None:
            self._dequeue_and_enqueue(k, batch_indices)

        return logits, labels, sim_pos_mean, sim_neg_mean, fnr_ratio
