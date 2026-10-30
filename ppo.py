import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic 神经网络架构，用于 PPO (Proximal Policy Optimization) 算法。
    该类包含两个主要子网络：
    1. Actor (策略网络): 负责根据当前状态输出动作的概率分布参数（均值 mu 和 标准差 std）。
    2. Critic (价值网络): 负责评估当前状态的好坏，输出状态价值 V(s)。
    """

    def __init__(self, state_dim=5, action_dim=1, hidden_dim=64):
        """
        初始化 Actor-Critic 网络。
        Args:
            state_dim (int): 输入状态向量的维度 (例如 5: [信噪比, 一致性, FNR, 预留位, Loss变化])。
            action_dim (int): 输出动作的维度 (例如 1: hard_ratio 比例系数)。
            hidden_dim (int): 隐藏层神经元数量 (默认 64)。
        """
        # 调用父类 nn.Module 的构造函数，初始化 PyTorch 模块的内部状态
        super(ActorCritic, self).__init__()

        # ================= Actor 网络定义 (策略函数 π) =================
        # 这是一个共享的特征提取层序列，用于提取状态的高阶特征
        # 结构: Linear -> Tanh -> Linear -> Tanh
        self.actor_fc = nn.Sequential(
            # 第一层全连接：将输入维度映射到隐藏层维度
            nn.Linear(state_dim, hidden_dim),
            # 激活函数 Tanh：将输出压缩到 (-1, 1)。
            # 在 PPO 中，Tanh 通常优于 ReLU，因为它输出也是中心化的，且梯度更平滑。
            nn.Tanh(),
            # 第二层全连接：增加网络深度，提升非线性拟合能力
            nn.Linear(hidden_dim, hidden_dim),
            # 再次使用 Tanh 激活
            nn.Tanh()
        )

        # --- 输出头 1: 均值 (mu) ---
        # 这是一个线性层，将提取的特征映射到动作维度
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        # --- 输出头 2: 对数标准差 (log_std) ---
        # 这里设计为一个可学习的参数 (Parameter)，而不是神经网络的输出。
        # 含义：整个动作空间共享同一个标准差，不随输入状态变化（State-Independent Log Std）。
        # torch.zeros(1, action_dim): 初始化为 0，意味着初始标准差为 exp(0) = 1.0。
        # 随着训练进行，网络会自动调整这个值来控制探索程度 (熵)。
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim))

        # ================= Critic 网络定义 (价值函数 V) =================
        # 这是一个独立的网络，用于估算 V(s)，即处于当前状态的预期累积回报
        # 结构: Linear -> Tanh -> Linear -> Tanh -> Linear (输出标量)
        self.critic = nn.Sequential(
            # 输入层：同样接收状态向量
            nn.Linear(state_dim, hidden_dim),
            # 激活函数
            nn.Tanh(),
            # 隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            # 激活函数
            nn.Tanh(),
            # 输出层：输出维度为 1，因为价值 V(s) 是一个标量数值
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        前向传播函数。
        Args:
            state: 输入的状态张量 [Batch_Size, state_dim]
        Returns:
            mu: 动作分布的均值 [Batch_Size, action_dim]
            std: 动作分布的标准差 [Batch_Size, action_dim]
            value: 状态价值估算 [Batch_Size, 1]
        """
        # 1. 计算 Critic Value (状态价值)
        # 直接将状态传入 Critic 网络，得到标量 V(s)
        value = self.critic(state)

        # 2. 计算 Actor 输出
        # 先通过共享的全连接层提取特征
        x = self.actor_fc(state)

        # --- 计算均值 mu ---
        # self.mu_head(x) 输出的是未经过激活的值 (logits)。
        # 使用 Sigmoid 激活函数将其压缩到 (0, 1) 区间。
        # 原因：你的动作是 hard_ratio (困难样本比例)，这是一个概率值，物理意义上必须在 0 到 1 之间。
        mu = torch.sigmoid(self.mu_head(x))

        # --- 计算标准差 std ---
        # 也就是从对数域转换回实数域: std = e^(log_std)
        # 这样做是为了保证标准差永远非负 (因为 e^x 恒大于 0)。
        # expand_as(mu): 将 [1, action_dim] 的参数扩展成 [Batch_Size, action_dim] 以匹配 mu 的形状
        std = torch.exp(self.log_std_head).expand_as(mu)

        # 返回三个关键变量，供 PPO 构建正态分布并计算 Loss
        return mu, std, value


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 强化学习代理类。
    负责管理策略网络 (Actor-Critic)、选择动作、存储经验回放以及执行 PPO 核心更新算法。
    """

    def __init__(self, state_dim=5, action_dim=1, lr=3e-4, gamma=0.1, eps_clip=0.2, k_epochs=4):
        """
        初始化 PPO Agent。
        Args:
            state_dim (int): 输入状态的维度 (默认为 5)。
            lr (float): 学习率 (Learning Rate)，通常为 3e-4 (0.0003)。
            gamma (float): 折扣因子 (Discount Factor)，用于计算未来回报，但这儿针对单步任务简化了。
            eps_clip (float): PPO 裁剪参数 (Epsilon Clip)，通常为 0.1 或 0.2，限制策略更新幅度。
            k_epochs (int): 每次更新时，复用这批数据训练多少轮 (PPO 的 Sample Efficiency 来源)。
        """
        # 1. 初始化策略网络 (Actor-Critic)
        # 实例化前面定义的神经网络，并自动检测是否有 GPU，有则移动到 GPU 上加速。
        self.policy = ActorCritic(state_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 2. 初始化优化器
        # 使用 Adam 优化器来更新策略网络的所有参数 (Actor 和 Critic 的参数)。
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 3. 保存超参数
        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # PPO 截断范围 (0.8 ~ 1.2)
        self.k_epochs = k_epochs  # 每次 update 循环的次数

        # 4. 定义损失函数
        # 均方误差损失 (MSE)，用于 Critic 网络估算 Value 时的 Loss (预测值 vs 真实回报)。
        self.mse_loss = nn.MSELoss()

        # 5. 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 6. 初始化记忆库 (Memory)
        # 用于暂时存储轨迹数据 (State, Action, Reward 等)，用于 On-policy 更新。
        # 注意：PPO 是 On-policy 算法，更新完一次后必须清空 Memory。
        self.memory = []

    def select_action(self, state, training=True):
        """
        根据当前状态选择动作。
        Args:
            state (array): 当前环境状态向量。
            training (bool): 是否处于训练模式。
                             True -> 从分布中采样 (Exploration, 探索)。
                             False -> 直接取均值 (Exploitation, 利用/推理)。
        Returns:
            action (float): 具体的动作值 (标量)。
        """
        # 1. 状态预处理
        # 将 numpy 数组转换为 FloatTensor，增加 Batch 维度 [1, state_dim]，并移至 GPU。
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 2. 网络前向传播
        # policy 返回三个值：均值 mu，标准差 std，状态价值 (这里不需要价值，用 _ 忽略)。
        mu, std, _ = self.policy(state)

        # 3. 构建高斯分布 (正态分布)
        # 根据 mu 和 std 构建分布，用于后续采样。
        dist = Normal(mu, std)

        if training:
            # --- 训练模式：随机采样 ---
            # sample(): 从分布中随机抽取一个动作，带有随机性 (探索)。
            # 例如 mu=0.5, std=0.1，可能采样到 0.48 或 0.62。
            action = dist.sample()

            # log_prob(): 计算该动作在当前分布下的对数概率。
            # 这是 PPO 计算 Loss 时计算“重要性采样比率”所必须的。
            action_log_prob = dist.log_prob(action)

            # --- 缓存转换 (Transition) ---
            # 暂时保存 (状态, 动作, 动作概率)，等待外部传入 Reward 后一起存入 memory。
            self.current_transition = [state, action, action_log_prob]

            # --- 动作截断 ---
            # 将动作强制限制在 [0.01, 0.99] 之间。
            # 1. 防止超出物理意义 (比例不能小于0或大于1)。
            # 2. 防止取到边界值 0 或 1 导致后续数值计算不稳定。
            action = torch.clamp(action, 0.01, 0.99)

            # item(): 将单元素 Tensor 转换为 Python float 返回。
            return action.item()
        else:
            # --- 测试模式：确定性输出 ---
            # 直接返回分布的均值 mu，这是模型认为“概率最大/最优”的动作。
            return mu.item()

    def store_transition(self, reward, next_state, done):
        """
        存储完整的经验步 (Transition)。
        外部环境执行完动作并算出 Reward 后调用此函数。
        """
        # 取出之前在 select_action 中缓存的前半部分数据
        state, action, action_log_prob = self.current_transition

        # 将完整的元组 (s, a, log_p, r, s', done) 加入记忆库。
        # 注意：next_state 和 done 在这套简化的代码里暂时没用到，但保留接口是为了通用性。
        self.memory.append((state, action, action_log_prob, reward, next_state, done))

    def update(self):
        """
        PPO 核心更新函数。
        使用 Memory 中的数据，执行多次梯度下降更新策略网络。
        """

        if len(self.memory) < 2:
            return
        # 如果记忆库为空，直接跳过 (例如预热阶段)
        if len(self.memory) == 0:
            return

        #修改奖励
        # returns = []
        # R = 0.0
        # for _, _, _, reward, _, done in reversed(self.memory):
        #     R = reward +self.gamma  * R * (1.0 - float(done))
        #     returns.insert(0, R)
        #
        # returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        #
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 1. 整理 Batch 数据
        # 将 memory 列表中的数据解包并堆叠成 Tensor 块
        # detach(): 确保这里的数据切断了之前的计算图，只作为固定的数值参与本次更新。

        # states: [Batch_Size, State_Dim]
        states = torch.cat([x[0] for x in self.memory]).detach()
        # actions: [Batch_Size, 1]
        actions = torch.cat([x[1] for x in self.memory]).detach()
        # old_log_probs: [Batch_Size, 1] (旧策略产生的概率，用于计算 Ratio)
        old_log_probs = torch.cat([x[2] for x in self.memory]).detach().sum(dim=-1)
        # rewards: [Batch_Size]
        rewards = torch.tensor([x[3] for x in self.memory], dtype=torch.float32).to(self.device)

        # 2. 奖励归一化 (Reward Normalization)
        # 这是一个 Trick。针对你的场景，每个 Batch 是独立的任务，没有时间序列上的“未来折扣回报”。
        # 所以直接对 Reward 进行 Z-Score 归一化 (减均值除方差)。
        # 作用：统一 Reward 的尺度，让梯度更加稳定，避免因为 Reward 忽大忽小导致训练崩溃。
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 3. PPO 更新循环 (K Epochs)
        # 利用同一批数据反复训练 k_epochs 次，榨取数据价值。
        for _ in range(self.k_epochs):
            # --- A. 评估旧动作在新策略下的表现 ---
            # 将保存的 states 再次输入网络 (此时网络参数可能已经被更新过微小幅度)
            mu, std, state_values = self.policy(states)

            # 构建新策略的分布
            dist = Normal(mu, std)

            # 计算新策略下，当初那些 actions 的对数概率
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # 计算分布的熵 (Entropy): 衡量分布的随机程度。熵越大，探索性越强。
            # 我们希望在 Loss 里加上熵奖励，鼓励模型多探索，防止过早收敛到局部最优。
            dist_entropy = dist.entropy().sum(dim=-1)

            # 将 Critic 输出的 Value 从 [Batch, 1] 压缩成 [Batch] 以匹配 rewards 形状
            state_values = state_values.squeeze()
            state_values = state_values.view(-1)

            # --- B. 计算优势函数 (Advantage) ---
            # Advantage = 实际回报 - Critic预测的基准价值
            # 含义：这个动作比“平均水平”好多少？
            # A > 0: 动作好，应该增加概率；A < 0: 动作差，应该降低概率。
            # detach(): Critic 的梯度不传给 Actor。
            advantages = rewards - state_values.detach()

            # --- C. 计算重要性采样比率 (Ratio) ---
            # Ratio = P_new(a|s) / P_old(a|s)
            # log(a/b) = log(a) - log(b)  =>  a/b = exp(log(a) - log(b))
            ratios = torch.exp(log_probs - old_log_probs)
            N = states.shape[0]
            assert ratios.shape == advantages.shape == (N,)

            # --- D. 计算 PPO 裁剪损失 (Surrogate Loss) ---
            # 第一项: 无裁剪的原始目标 (Ratio * Advantage)
            surr1 = ratios * advantages
            assert surr1.shape == (N,)

            # 第二项: 裁剪后的目标
            # torch.clamp: 将 ratio 强制限制在 [0.8, 1.2] 范围内 (假设 eps_clip=0.2)。
            # 作用: 防止一次更新步子迈得太大，导致策略崩溃 (Trust Region)。
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # PPO Loss = -min(surr1, surr2)
            # 取负号是因为我们要最大化目标，但优化器是最小化 Loss。
            # min 的作用是取“保守下界”，确保更新是安全的。

            # Critic Loss: 0.5 * MSE(预测值, 真实回报) -> 让 Critic 估值更准
            # Entropy Bonus: - 0.01 * entropy -> 减去负的熵(即加上熵)，鼓励探索
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            # --- E. 反向传播与更新 ---
            self.optimizer.zero_grad()  # 清空梯度
            loss.mean().backward()  # 反向传播，计算梯度 (取 batch 均值)
            self.optimizer.step()  # 更新参数

        # 4. 清空记忆库
        # PPO 是 On-policy 算法，旧数据是用旧策略采样的，
        # 更新策略后，旧数据分布已不再匹配，必须丢弃，重新采样。
        self.memory = []