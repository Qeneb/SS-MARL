import unittest
import torch
import numpy as np
from ssmarl.algorithms.ssmarl.algorithm.SSMARLPolicy import SSMARLPolicy
from types import SimpleNamespace
from gym.spaces import Box


class TestSSMARLPolicy(unittest.TestCase):
    def setUp(self):
        # Mock arguments
        self.args = SimpleNamespace(
            hidden_size=64,
            gain=0.01,
            use_orthogonal=True,
            use_policy_active_masks=False,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=True,
            use_lstm=False,
            recurrent_N=1,
            gnn_out_channels=16,
            gnn_num_heads=2,
            gnn_num_layers=2,
            gnn_hidden_size=64,
            gnn_edge_dim=4,
            embedding_dim=9,
            embedding_nums=5,
            embedding_hidden_size=16,
            lr=0.001,
            critic_lr=0.001,
            opti_eps=1e-5,
            weight_decay=0.0
        )
        self.device = torch.device("cpu")

        # Mock observation and action spaces
        self.obs_space = [torch.Size([10])]
        self.cent_obs_space = [torch.Size([20])]
        low_action = np.array([-1.0, -1.0])
        high_action = np.array([1.0, 1.0])
        self.act_space = Box(low=low_action, high=high_action, dtype=np.float32)

        # Initialize SSMARLPolicy
        self.policy = SSMARLPolicy(self.args, self.obs_space, self.cent_obs_space, self.act_space, self.device)

        # Mock inputs
        self.agent_id = torch.tensor([[0]])
        self.nodes_feats = torch.rand(1, 5, 4)
        self.edge_index = torch.randint(0, 5, (1, 2, 5))
        self.edge_attr = torch.rand(1, 5, 4)
        self.rnn_states_actor = torch.zeros(1, 1, self.args.hidden_size)
        self.rnn_states_critic = torch.zeros(1, 1, self.args.hidden_size)
        self.rnn_states_cost = torch.zeros(1, 1, self.args.hidden_size)
        self.masks = torch.ones(1, 1)
        self.available_actions = torch.ones(1, 5)
        self.action = torch.rand(1, 2)

    def test_get_actions(self):
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, cost_preds, rnn_states_cost = self.policy.get_actions(
            self.agent_id,
            self.nodes_feats,
            self.edge_index,
            self.edge_attr,
            self.rnn_states_actor,
            self.rnn_states_critic,
            self.rnn_states_cost,
            self.masks,
            self.available_actions
        )

        # Assertions
        self.assertEqual(actions.shape, torch.Size([1, 2]))
        self.assertEqual(action_log_probs.shape, torch.Size([1, 2]))
        self.assertEqual(values.shape, torch.Size([1, 1]))
        self.assertEqual(rnn_states_actor.shape, self.rnn_states_actor.shape)
        self.assertEqual(rnn_states_critic.shape, self.rnn_states_critic.shape)
        self.assertEqual(cost_preds.shape, torch.Size([1, 1]))
        self.assertEqual(rnn_states_cost.shape, self.rnn_states_critic.shape)

    def test_get_values(self):
        values = self.policy.get_values(
            self.agent_id,
            self.nodes_feats,
            self.edge_index,
            self.edge_attr,
            self.rnn_states_critic,
            self.masks
        )

        # Assertions
        self.assertEqual(values.shape, torch.Size([1, 1]))

    def test_get_cost_values(self):
        cost_values = self.policy.get_cost_values(
            self.agent_id,
            self.nodes_feats,
            self.edge_index,
            self.edge_attr,
            self.rnn_states_cost,
            self.masks
        )

        # Assertions
        self.assertEqual(cost_values.shape, torch.Size([1, 1]))

    def test_evaluate_actions(self):
        values, action_log_probs, dist_entropy, cost_values, action_mu, action_std = self.policy.evaluate_actions(
            self.agent_id,
            self.nodes_feats,
            self.edge_index,
            self.edge_attr,
            self.rnn_states_actor,
            self.rnn_states_critic,
            self.rnn_states_cost,
            self.action,
            self.masks,
            self.available_actions
        )

        # Assertions
        self.assertEqual(values.shape, torch.Size([1, 1]))
        self.assertEqual(action_log_probs.shape, torch.Size([1, 2]))
        self.assertIsInstance(dist_entropy, torch.Tensor)
        self.assertEqual(cost_values.shape, torch.Size([1, 1]))
        self.assertEqual(action_mu.shape, torch.Size([1, 2]))
        self.assertEqual(action_std.shape, torch.Size([1, 2]))


if __name__ == "__main__":
    unittest.main()

# python -m unittest tests/test_ssmarl_policy.py