import torch
from ssmarl.algorithms.ssmarl.algorithm.g_actor_critic import G_Actor, G_Critic
from ssmarl.utils.util import update_linear_schedule


class SSMARLPolicy:
    """
    SSMARL Policy class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = G_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = G_Critic(args, self.share_obs_space, self.device)
        self.cost_critic = G_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                                               lr=self.critic_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.cost_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, nodes_feats, edge_index, edge_attr, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, rnn_states_cost=None):
        """
        Compute actions and value function predictions for the given inputs.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(agent_id,
                                                                 nodes_feats,
                                                                 edge_index,
                                                                 edge_attr,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_critic, masks)
        if rnn_states_cost is None:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        else:
            cost_preds, rnn_states_cost = self.cost_critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_cost, masks)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, cost_preds, rnn_states_cost


    def get_values(self, agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_critic, masks):
        """
        Get value function predictions.
        """
        values, _ = self.critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_critic, masks)
        return values

    def get_cost_values(self, agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_cost, masks):
        """
        Get constraint cost predictions.
        """
        cost_preds, _ = self.cost_critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_cost, masks)
        return cost_preds

    def evaluate_actions(self, agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, nodes_feats, edge_index, edge_attr, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, rnn_states_cost=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        """
        action_log_probs, dist_entropy, action_mu, action_std = self.actor.evaluate_actions(agent_id, nodes_feats, edge_index, edge_attr,
                                                                                            rnn_states_actor,
                                                                                            action,
                                                                                            masks,
                                                                                            available_actions,
                                                                                            active_masks)
        values, _ = self.critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_critic, masks)
        cost_values, _ = self.cost_critic(agent_id, cent_nodes_feats, cent_edge_index, cent_edge_attr, rnn_states_cost, masks)
        # values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, cost_values, action_mu, action_std


    def act(self, agent_id, nodes_feats, edge_index, edge_attr, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        """
        actions, _, rnn_states_actor = self.actor(agent_id, nodes_feats, edge_index, edge_attr, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
