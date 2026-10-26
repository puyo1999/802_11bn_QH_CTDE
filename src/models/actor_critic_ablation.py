import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCritic(nn.Module):
    """조건 1: 자기 관측(o_i)만 입력받는 로컬 비평가 네트워크"""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(LocalCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, local_obs, action):
        x = torch.cat([local_obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


class CentralCritic(nn.Module):
    """조건 2~5: 전체 글로벌 관측(o_1 ... o_N)을 입력받는 중앙화 비평가 네트워크"""

    def __init__(self, global_obs_dim, global_action_dim, hidden_dim=128):
        super(CentralCritic, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim + global_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, global_obs, global_action):
        x = torch.cat([global_obs, global_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


class SpatialAttentionLayer(nn.Module):
    """공간 주의집중(Spatial Attention) 레이어"""

    def __init__(self, feature_dim, hidden_dim=128):
        super(SpatialAttentionLayer, self).__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)

    def forward(self, self_feat, neighbor_feats, adj_mask=None):
        # self_feat: [batch, feat_dim]
        # neighbor_feats: [batch, num_neighbors, feat_dim]
        q = self.query(self_feat).unsqueeze(1)  # [batch, 1, hidden_dim]
        k = self.key(neighbor_feats)  # [batch, num_neighbors, hidden_dim]
        v = self.value(neighbor_feats)  # [batch, num_neighbors, hidden_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
        if adj_mask is not None:
            scores = scores.masked_fill(adj_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v).squeeze(1)
        return context


class UnifiedActor(nn.Module):
    """조건 1~5 통합 행위자 네트워크"""

    def __init__(self, obs_dim, action_dim, actor_type="single", hidden_dim=128):
        super(UnifiedActor, self).__init__()
        self.actor_type = actor_type
        self.obs_dim = obs_dim

        self.self_encoder = nn.Linear(obs_dim, hidden_dim)

        if actor_type in ["joint_attention", "vq_joint", "dummy_joint"]:
            self.neighbor_encoder = nn.Linear(obs_dim, hidden_dim)
            self.attention = SpatialAttentionLayer(hidden_dim, hidden_dim)
            self.fc_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc_combine = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, self_obs, neighbor_obs=None, adj_mask=None):
        self_feat = F.relu(self_encoder(self_obs))

        if self.actor_type in ["joint_attention", "vq_joint"]:
            neigh_feat = F.relu(self.neighbor_encoder(neighbor_obs))
            context = self.attention(self_feat, neigh_feat, adj_mask)
            combined = torch.cat([self_feat, context], dim=-1)
            x = F.relu(self.fc_combine(combined))

        elif self.actor_type == "dummy_joint":
            # 조건 5: 이웃 관측을 0으로 채워(Zero-Padding) 정보 유입 차단
            dummy_neigh_obs = torch.zeros_like(neighbor_obs)
            neigh_feat = F.relu(self.neighbor_encoder(dummy_neigh_obs))
            context = self.attention(self_feat, neigh_feat, adj_mask)
            combined = torch.cat([self_feat, context], dim=-1)
            x = F.relu(self.fc_combine(combined))

        else:  # "single" (C1, C2)
            x = F.relu(self.fc_combine(self_feat))

        action_logits = self.policy_head(x)
        return F.softmax(action_logits, dim=-1)