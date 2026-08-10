import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

# ========== 全局设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 经验条目 ==========
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state']
)


# ========== Actor（保持不变） ==========
class Actor(nn.Module):
    def __init__(self, grid, k_pool=11, tau=1.0):
        super().__init__()
        self.grid = grid
        self.grid2 = grid * grid
        self.tau = nn.Parameter(torch.tensor(float(tau)))  # 可学习温度

        # 仍然保持超快的逐点融合（1x1 conv）
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),  # -> [B,1,G,G]
        )

        # 计算局部空白度 prior: 1 - AvgPool(canvas)
        self.pool = nn.AvgPool2d(kernel_size=k_pool, stride=1, padding=k_pool // 2, count_include_pad=False)

        # 门控：根据 wiremask 的强弱控制“数据分支”和“先验分支”的占比
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        # 将空间先验拼合后再做 1x1 调整，便于学习不同任务的加权
        self.prior_fuse = nn.Conv2d(2, 1, 1, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        # 预生成边界距离 prior（越靠中间越大，贴边越小）
        with torch.no_grad():
            yy, xx = torch.meshgrid(
                torch.arange(grid, dtype=torch.float32),
                torch.arange(grid, dtype=torch.float32),
                indexing='ij'
            )
            dist_to_border = torch.minimum(
                torch.minimum(xx, grid - 1 - xx),
                torch.minimum(yy, grid - 1 - yy)
            ) / (grid / 2.0)
            self.register_buffer("border_prior", dist_to_border.unsqueeze(0).unsqueeze(0))  # [1,1,G,G]

    def forward(self, x):
        """
        x: [B, 3, G, G], order = [canvas, wiremask, position_mask]
        returns: [B, G*G] probs
        """
        assert x.dim() == 4 and x.size(2) == self.grid and x.size(3) == self.grid
        B = x.size(0)

        canvas = x[:, 0:1]  # [B,1,G,G]
        wire = x[:, 1:2]  # [B,1,G,G]
        posmsk = x[:, 2:3]  # [B,1,G,G]

        # 数据分支：逐点融合得到 logits_data
        logits_data = self.cnn(x)  # [B,1,G,G]

        # 先验分支：局部空白度 + 边界先验
        # 局部空白度（越空越好）
        local_occ = self.pool(canvas)  # [B,1,G,G], 0~1
        prior_space = 1.0 - torch.clamp(local_occ, 0, 1)  # 空白越大越好

        # 边界先验（远离边界一点更好）
        prior_border = self.border_prior.expand(B, -1, -1, -1)  # [B,1,G,G]

        # 融合先验成单通道
        prior_map = torch.cat([prior_space, prior_border], dim=1)  # [B,2,G,G]
        logits_prior = self.prior_fuse(prior_map)  # [B,1,G,G]

        # 计算 wire 的“有效性” —— 如果几乎全 0，则 gate→0，更依赖先验
        # 用两个标量特征：均值与最大值的绝对值
        wire_abs = wire.abs()
        wire_feat = torch.stack([
            wire_abs.mean(dim=[1, 2, 3]),  # [B]
            wire_abs.amax(dim=[1, 2, 3])  # [B]
        ], dim=1)  # [B,2]
        gate = self.gate_mlp(wire_feat).view(B, 1, 1, 1)  # [B,1,1,1] in (0,1)
        # gate 越大表示 wire 越“有信号”，更信任数据分支；反之更信任先验

        logits = gate * logits_data + (1.0 - gate) * logits_prior  # [B,1,G,G]

        # 非法位置屏蔽
        invalid = (posmsk >= 1.0)  # [B,1,G,G]
        logits = logits.masked_fill(invalid, -1.0e10)

        # 温度缩放 + softmax
        logits = logits.view(B, -1)  # [B, G*G]
        probs = self.softmax(logits / self.tau.clamp_min(1e-3))

        return probs


# ========== Critic ==========
class Critic(nn.Module):
    def __init__(self, grid=224):
        super(Critic, self).__init__()
        self.input_dim = grid * grid * 3
        self.fc1 = nn.Linear(self.input_dim, 64)  # 假设输入是 64 维
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)  # 输出一个标量，即 [B, 1]

    def forward(self, x):
        # x 是 [B, 3, 224, 224] 形状
        # 展平输入：将 [B, 3, 224, 224] 展平为 [B, 3 * 224 * 224]
        x = x.view(x.size(0), -1)  # 展平操作，将每个样本从 [3, 224, 224] 展平为 [3 * 224 * 224]

        # 继续通过 fc1 和 fc2 层
        x1 = F.relu(self.fc1(x))  # 处理
        x2 = F.relu(self.fc2(x1))  # 处理

        # 输出一个标量，即每个状态对应的价值
        value = self.state_value(x2)  # [B, 1]
        return value


class RDC(nn.Module):
    def __init__(self, grid=224, hidden_dim=128, last_dim=64):
        super().__init__()
        self.input_dim = grid * grid * 3
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, last_dim)

        # 三个独立头
        self.value_total = nn.Linear(last_dim, 1)  # 总价值
        self.value_imm = nn.Linear(last_dim, 1)  # 即时分量
        self.value_future = nn.Linear(last_dim, 1)  # 长期分量

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        l = F.relu(self.fc3(h))

        v_total = self.value_total(l)  # 独立预测
        v_imm = self.value_imm(l)
        v_future = self.value_future(l)

        return v_total, v_imm, v_future


class RDC2(nn.Module):
    def __init__(self, input_channels=3, grid=224, hidden_dim=64, lstm_hidden_dim=64):
        super().__init__()
        self.image_size = grid

        # 共享的CNN特征提取器（处理原始图像）
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小，避免尺寸问题
        )

        # 计算CNN输出维度
        self.cnn_output_dim = 64 * 4 * 4

        # 共享的全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 总价值头（MLP-based）
        self.value_total_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 即时奖励头（CNN-based，使用早期特征）
        self.value_imm_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 将 32 改为 64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        self.value_imm_fc = nn.Linear(64, 1)

        # 未来奖励头（考虑时序依赖）
        self.value_future_lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.value_future_fc = nn.Linear(lstm_hidden_dim, 1)

        # 存储中间特征
        self.early_features = None

    def hook_fn(self, module, input, output):
        """钩子函数获取中间层特征"""
        self.early_features = output

    def forward(self, x, sequence_length=1):
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            sequence_length: 用于LSTM的序列长度（默认为1）
        """
        batch_size = x.size(0)

        # 注册钩子获取中间特征（用于即时奖励头）
        hook_handle = None
        if hasattr(self.shared_cnn[0], 'register_forward_hook'):
            hook_handle = self.shared_cnn[2].register_forward_hook(self.hook_fn)  # 在第二个卷积层后获取特征

        # 共享CNN特征提取
        cnn_features = self.shared_cnn(x)
        cnn_features_flat = cnn_features.view(batch_size, -1)

        # 共享全连接层
        shared_features = self.shared_fc(cnn_features_flat)

        # 总价值预测
        v_total = self.value_total_head(shared_features)

        # 即时奖励预测（使用CNN中间特征）
        v_imm = torch.zeros(batch_size, 1, device=x.device)
        if self.early_features is not None:
            imm_features = self.value_imm_conv(self.early_features)
            imm_features_flat = imm_features.view(batch_size, -1)
            v_imm = self.value_imm_fc(imm_features_flat)

        # 移除钩子
        if hook_handle is not None:
            hook_handle.remove()
        self.early_features = None

        # 未来奖励预测（LSTM-based）
        # 如果只有单帧，复制特征创建伪序列
        if sequence_length > 1:
            # 假设x已经包含序列维度 [batch, seq_len, channels, height, width]
            # 需要先处理整个序列的CNN特征
            raise NotImplementedError("多帧序列处理需要重新设计数据流")
        else:
            # 单帧情况：创建伪序列或使用当前特征
            shared_features_seq = shared_features.unsqueeze(1)  # [batch, 1, features]
            # 如果需要更长的序列，可以复制当前特征
            if sequence_length > 1:
                shared_features_seq = shared_features_seq.repeat(1, sequence_length, 1)

            lstm_out, _ = self.value_future_lstm(shared_features_seq)
            v_future = self.value_future_fc(lstm_out[:, -1, :])  # 取最后一个时间步

        return v_total, v_imm, v_future\

class RDC3(nn.Module):
    def __init__(self, input_channels=3, grid=224, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.image_size = grid

        # 共享的CNN特征提取器（处理原始图像）
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小，避免尺寸问题
        )

        # 计算CNN输出维度
        self.cnn_output_dim = 64 * 4 * 4

        # 共享的全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout增加正则化
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Dropout增加正则化
        )

        # 总价值头（MLP-based）
        self.value_total_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 即时奖励头（CNN-based，使用早期特征）
        self.value_imm_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 将 32 改为 64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        self.value_imm_fc = nn.Linear(64, 1)

        # 未来奖励头（替换LSTM为全连接层）
        self.value_future_fc = nn.Linear(hidden_dim, 1)  # 去除LSTM，直接用全连接层

        # 存储中间特征
        self.early_features = None

    def hook_fn(self, module, input, output):
        """钩子函数获取中间层特征"""
        self.early_features = output

    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        """
        batch_size = x.size(0)

        # 注册钩子获取中间特征（用于即时奖励头）
        hook_handle = None
        if hasattr(self.shared_cnn[0], 'register_forward_hook'):
            hook_handle = self.shared_cnn[2].register_forward_hook(self.hook_fn)  # 在第二个卷积层后获取特征

        # 共享CNN特征提取
        cnn_features = self.shared_cnn(x)
        cnn_features_flat = cnn_features.view(batch_size, -1)

        # 共享全连接层
        shared_features = self.shared_fc(cnn_features_flat)

        # 总价值预测
        v_total = self.value_total_head(shared_features)

        # 即时奖励预测（使用CNN中间特征）
        v_imm = torch.zeros(batch_size, 1, device=x.device)
        if self.early_features is not None:
            imm_features = self.value_imm_conv(self.early_features)
            imm_features_flat = imm_features.view(batch_size, -1)
            v_imm = self.value_imm_fc(imm_features_flat)

        # 移除钩子
        if hook_handle is not None:
            hook_handle.remove()
        self.early_features = None

        # 未来奖励预测（直接通过全连接层处理）
        v_future = self.value_future_fc(shared_features)

        return v_total, v_imm, v_future



# ========== PPO ==========
class PPO:
    def __init__(self, env, args):
        self.env = env
        self.device = 'cuda'
        self.gcn = None
        self.actor_net = Actor(grid=args.grid).float().to(device)
        self.critic_net = RDC3(grid=args.grid).float().to(device)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.A_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), args.C_lr)

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epoch = 10
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.args = args
        self.placed_num_macro = args.pnm
        self.buffer_capacity = 5 * args.pnm
        # 存储上一个选择的位置
        self.last_position = None

    def select_action(self, state, Eval=False):
        state = state.clone().detach().to(device).float()

        state = state.unsqueeze(0)  # 增加批次维度
        with torch.no_grad():
            probs = self.actor_net(state)  # 获取策略网络的输出

        dist = Categorical(probs)

        if Eval:  # 如果在评估模式下，选择概率最大的动作
            action = torch.argmin(probs, dim=-1)  # 选择最大概率的动作
            log_prob = dist.log_prob(action)  # 计算该动作的log_prob
        else:  # 否则按概率分布采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob

    def store_transition(self, transition):
        # Helper function to convert data to tensor if it's not already in tensor format
        def to_tensor(data, dtype=torch.float):
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data, dtype=dtype, device=self.device)

        # Standardizing transition components
        state = to_tensor(transition.state, dtype=torch.float)
        next_state = to_tensor(transition.next_state, dtype=torch.float)
        reward = to_tensor(transition.reward, dtype=torch.float)
        a_log_prob = to_tensor(transition.a_log_prob, dtype=torch.float)
        action = to_tensor(transition.action, dtype=torch.long)
        # Append the transition to the buffer
        self.buffer.append(Transition(state, action, reward, a_log_prob, next_state))

        # Increment the counter and check if buffer capacity is reached
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self, writer=None):
        states = torch.stack([t.state for t in self.buffer]).to(device)
        actions = torch.stack([t.action for t in self.buffer]).view(-1, 1).to(device)
        rewards = torch.stack([t.reward for t in self.buffer]).view(-1, 1).to(device)
        old_log_probs = torch.stack([t.a_log_prob for t in self.buffer]).view(-1, 1).to(device)

        # 计算 target value (回报)
        target_list = []
        target = 0
        for i in reversed(range(rewards.shape[0])):
            if self.env.t >= self.placed_num_macro - 1:
                target = 0
            target = rewards[i, 0].item() + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(target_list, dtype=torch.float, device=device).view(-1, 1)

        self.buffer.clear()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                self.training_step += 1

                probs = self.actor_net(states[index].to(device))
                dist = Categorical(probs)
                action_log_prob = dist.log_prob(actions[index].squeeze())
                ratio = torch.exp(action_log_prob - old_log_probs[index].squeeze())

                # ------- Critic with Return Decomposition -------
                v_total, v_imm, v_future = self.critic_net(states[index].to(device))

                # 计算优势 = 目标 - 总价值
                advantage = (target_v_all[index] - v_total).detach()

                # Actor loss (标准 PPO)
                surr1 = ratio * advantage.squeeze()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage.squeeze()
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss = 总体监督 + 分解一致性
                value_loss_total = F.smooth_l1_loss(v_total, target_v_all[index])
                value_loss_balance = F.mse_loss(v_total, v_imm + v_future)
                value_loss = value_loss_total + 0.1 * value_loss_balance

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                if writer:
                    writer.add_scalar('ppo/action_loss', actor_loss.item(), self.training_step)
                    writer.add_scalar('ppo/value_loss_total', value_loss_total.item(), self.training_step)
                    writer.add_scalar('ppo/value_loss_balance', value_loss_balance.item(), self.training_step)

    def save_param(self, path):
        torch.save({
            'actor_net_dict': self.actor_net.state_dict(),
            'critic_net_dict': self.critic_net.state_dict()
        }, path)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
