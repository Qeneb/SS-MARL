import torch
import unittest
import numpy as np
from ssmarl.algorithms.ssmarl.algorithm.g_actor_critic import G_Actor, G_Critic
from types import SimpleNamespace
from gym.spaces import Box

class TestGActorCritic(unittest.TestCase):
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
            embedding_hidden_size=16
        )
        self.device = torch.device("cpu")

        # Mock action space
        low_action = np.array([-1.0, -1.0])  # Minimum action values
        high_action = np.array([1.0, 1.0])   # Maximum action values
        self.action_space = Box(low=low_action, high=high_action, dtype=np.float32)

        # Initialize actor and critic
        self.actor = G_Actor(self.args, self.action_space, self.device)
        self.critic = G_Critic(self.args, self.device)

        torch.manual_seed(2)

    def test_actor_forward(self):
        # Mock inputs
        agent_id = torch.tensor([[0]])
        nodes_feats = torch.rand(1, 5, 4)
        edge_index = torch.randint(0, 5, (1, 2, 5))
        edge_attr = torch.rand(1, 5, 4)
        rnn_states = torch.zeros(1, 1, self.args.hidden_size)
        masks = torch.ones(1, 1)

        # In DiagGaussian, available_actions is not used
        available_actions = None

        # Forward pass
        actions, action_log_probs, rnn_states_out = self.actor(
            agent_id, nodes_feats, edge_index, edge_attr, rnn_states, masks, available_actions
        )

        # Assertions
        self.assertEqual(actions.shape, torch.Size([1, 2]))
        self.assertEqual(action_log_probs.shape, torch.Size([1, 2]))
        self.assertEqual(rnn_states_out.shape, rnn_states.shape)

    def test_actor_evaluate_actions(self):
        # Mock inputs
        agent_id = torch.tensor([[0]])
        nodes_feats = torch.rand(1, 5, 4)
        edge_index = torch.randint(0, 5, (1, 2, 5))
        edge_attr = torch.rand(1, 5, 4)
        rnn_states = torch.zeros(1, 1, self.args.hidden_size)
        action = torch.rand(1, 2)
        masks = torch.ones(1, 1)
        available_actions = None

        # Forward pass
        action_log_probs, dist_entropy, action_mu, action_std = self.actor.evaluate_actions(
            agent_id, nodes_feats, edge_index, edge_attr, rnn_states, action, masks, available_actions
        )

        # Assertions
        self.assertEqual(action_log_probs.shape, torch.Size([1, 2]))
        self.assertEqual(dist_entropy.shape, torch.Size([]))
        self.assertEqual(action_mu.shape, torch.Size([1, 2]))
        self.assertEqual(action_std.shape, torch.Size([1, 2]))

    def test_critic_forward(self):
        # Mock inputs
        agent_id = torch.tensor([[0]])
        nodes_feats = torch.rand(1, 5, 4)
        edge_index = torch.randint(0, 5, (1, 2, 5))
        edge_attr = torch.rand(1, 5, 4)
        rnn_states = torch.zeros(1, 1, self.args.hidden_size)
        masks = torch.ones(1, 1)

        # Forward pass
        values, rnn_states_out = self.critic(
            agent_id, nodes_feats, edge_index, edge_attr, rnn_states, masks
        )

        # Assertions
        self.assertEqual(values.shape, torch.Size([1, 1]))
        self.assertEqual(rnn_states_out.shape, rnn_states.shape)

if __name__ == "__main__":
    unittest.main()

# python -m unittest tests/test_g_actor_critic.py