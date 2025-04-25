import torch
import torch.nn as nn
from ssmarl.algorithms.utils.util import init, check
from ssmarl.algorithms.utils.gnn import GNNbase
from ssmarl.algorithms.utils.act import ACTLayer
from ssmarl.algorithms.utils.rnn import RNNLayer, LSTMLayer
from ssmarl.utils.util import get_shape_from_obs_space


class G_Actor(nn.Module):
    """
    Actor network class for SSMARL.
    """
    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(G_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.gnn_hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_lstm = args.use_lstm
        self.tpdv = dict(dtype=torch.float32, device=device)

        # graph_obs_shape = get_shape_from_obs_space(obs_space)[0]
        self.base = GNNbase(args, graph_aggr='agent', device=device)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self._use_lstm:
                self.rnn = LSTMLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.to(device)

    def forward(self, agent_id, nodes_feats, edge_index, edge_attr, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        """

        agent_id = check(agent_id).to(**self.tpdv)
        nodes_feats = check(nodes_feats).to(**self.tpdv)
        edge_index = check(edge_index).to(**self.tpdv)
        edge_attr = check(edge_attr).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(nodes_feats, edge_index, edge_attr, agent_id)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, agent_id, nodes_feats, edge_index, edge_attr, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        agent_id = check(agent_id).to(**self.tpdv)
        nodes_feats = check(nodes_feats).to(**self.tpdv)
        edge_index = check(edge_index).to(**self.tpdv)
        edge_attr = check(edge_attr).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(nodes_feats, edge_index, edge_attr, agent_id)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        action_log_probs, dist_entropy, action_mu, action_std = self.act.evaluate_actions_trpo(actor_features,
                                                                                               action,
                                                                                               available_actions,
                                                                                               active_masks=
                                                                                               active_masks if self._use_policy_active_masks
                                                                                               else None)

        return action_log_probs, dist_entropy, action_mu, action_std


class G_Critic(nn.Module):
    """
    Critic network class for SSMARL.
    """
    def __init__(self, args, device=torch.device("cpu")):
        super(G_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_lstm = args.use_lstm
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.base = GNNbase(args, graph_aggr='graph', device=device)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self._use_lstm:
                self.rnn = LSTMLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, agent_id, nodes_feats, edge_index, edge_attr, rnn_states, masks):
        """
        Compute value function predictions from the given inputs.
        """
        
        agent_id = check(agent_id).to(**self.tpdv)
        nodes_feats = check(nodes_feats).to(**self.tpdv)
        edge_index = check(edge_index).to(**self.tpdv)
        edge_attr = check(edge_attr).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # bacth_nodes_feats = bacth_nodes_feats.reshape(-1,*bacth_nodes_feats.shape[2:])
        # bacth_edge_index = bacth_edge_index.reshape(-1,*bacth_edge_index.shape[2:])
        # bacth_edge_attr = bacth_edge_attr.reshape(-1,*bacth_edge_attr.shape[2:])

        # 每个agent的cent_obs一样，取一个就行了
        # 不需要每个agent的cent_obs都保存，显存/内存优化
        # nodes_feats = bacth_nodes_feats[:,0].reshape(-1,*bacth_nodes_feats.shape[2:])
        # edge_index = bacth_edge_index[:,0].reshape(-1,*bacth_edge_index.shape[2:])
        # edge_attr = bacth_edge_attr[:,0].reshape(-1,*bacth_edge_attr.shape[2:])
        critic_features = self.base(nodes_feats, edge_index, edge_attr, agent_id)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
