import unittest
import torch
import numpy as np
from ssmarl.algorithms.ssmarl.ssmarl import SSMARL
from ssmarl.algorithms.ssmarl.algorithm.SSMARLPolicy import SSMARLPolicy
from types import SimpleNamespace
from gym.spaces import Box


class TestSSMARLTrainer(unittest.TestCase):
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
            weight_decay=0.0,
            clip_param=0.2,
            num_mini_batch=1,
            data_chunk_length=10,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            huber_delta=1.0,
            episode_length=100,
            kl_threshold=0.01,
            safety_bound=0.1,
            ls_step=10,
            accept_ratio=0.1,
            EPS=1e-5,
            gamma=0.99,
            line_search_fraction=0.5,
            fraction_coef=0.5,
            n_rollout_threads=1,
            use_max_grad_norm=True,
            use_clipped_value_loss=False,
            use_huber_loss=False,
            use_popart=False,
            use_value_active_masks=False,
            num_agents=1
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

        # Initialize SSMARL trainer
        self.trainer = SSMARL(self.args, self.policy, device=self.device)

        # Mock inputs
        self.sample = (
            torch.tensor([[0]]),  # agent_id
            # torch.rand(1, 5, 4),  # share_nodes_feats
            # torch.randint(0, 5, (1, 2, 5)),  # share_edge_index
            # torch.rand(1, 5, 4),  # share_edge_attr
            torch.rand(1, 5, 4),  # nodes_feats
            # torch.randint(0, 5, (1, 2, 5)),  # edge_index
            torch.tensor([[[1, 2, 3, 4], [0, 0, 0, 0]]]), # edge_index
            torch.rand(1, 4, 4),  # edge_attr
            torch.zeros(1, 1, self.args.hidden_size),  # rnn_states_batch
            torch.zeros(1, 1, self.args.hidden_size),  # rnn_states_critic_batch
            torch.rand(1, 2),  # actions_batch
            torch.rand(1, 1),  # value_preds_batch
            torch.rand(1, 1),  # return_batch
            torch.ones(1, 1),  # masks_batch
            torch.ones(1, 1),  # active_masks_batch
            torch.rand(1, 2),  # old_action_log_probs_batch
            torch.rand(1, 1),  # adv_targ
            torch.ones(1, 5),  # available_actions_batch
            torch.rand(1, 1),  # factor_batch
            torch.rand(1, 2),  # cost_preds_batch
            torch.rand(1, 2),  # cost_returns_batch
            torch.zeros(1, 1, self.args.hidden_size),  # rnn_states_cost_batch
            torch.rand(1, 2),  # cost_adv_targ
            torch.rand(1, 2)  # aver_episode_costs
        )

    def test_cal_value_loss(self):
        values = torch.rand(1, 1)
        value_preds_batch = torch.rand(1, 1)
        return_batch = torch.rand(1, 1)
        active_masks_batch = torch.ones(1, 1)

        value_loss = self.trainer.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # Assertions
        self.assertIsInstance(value_loss, torch.Tensor)
        self.assertEqual(value_loss.shape, torch.Size([]))

    def test_flat_grad(self):
        grads = [torch.rand(2, 2), torch.rand(3, 3)]
        flat_grads = self.trainer.flat_grad(grads)

        # Assertions
        self.assertIsInstance(flat_grads, torch.Tensor)
        self.assertEqual(flat_grads.shape, torch.Size([13]))

    def test_flat_hessian(self):
        hessians = [torch.rand(2, 2), torch.rand(3, 3)]
        flat_hessians = self.trainer.flat_hessian(hessians)

        # Assertions
        self.assertIsInstance(flat_hessians, torch.Tensor)
        self.assertEqual(flat_hessians.shape, torch.Size([13]))

    def test_flat_params(self):
        params = self.trainer.flat_params(self.policy.actor)

        # Assertions
        self.assertIsInstance(params, torch.Tensor)
        self.assertEqual(len(params.shape), 1)

    def test_update_model(self):
        params = self.trainer.flat_params(self.policy.actor)
        self.trainer.update_model(self.policy.actor, params)

        # Assertions
        updated_params = self.trainer.flat_params(self.policy.actor)
        self.assertTrue(torch.equal(params, updated_params))

    def test_kl_divergence(self):
        agent_id, nodes_feats, edge_index, edge_attr, rnn_states, rnn_states_critic, action, value, returns, masks, active_masks, _, _, available_actions = self.sample[:14]
        kl = self.trainer.kl_divergence(agent_id, nodes_feats, edge_index, edge_attr, rnn_states, action, masks, available_actions, active_masks, self.policy.actor, self.policy.actor)

        # Assertions
        self.assertIsInstance(kl, torch.Tensor)

    def test_fisher_vector_product(self):
        agent_id, nodes_feats, edge_index, edge_attr, rnn_states, rnn_states_critic, action, value, returns, masks, active_masks, _, _, available_actions = self.sample[:14]
        b = torch.rand(64865)
        vvp = self.trainer.fisher_vector_product(self.policy.actor, agent_id, nodes_feats, edge_index, edge_attr, rnn_states, action, masks, available_actions, active_masks, b)

        # Assertions
        self.assertIsInstance(vvp, torch.Tensor)

    def test_conjugate_gradient(self):
        agent_id, nodes_feats, edge_index, edge_attr, rnn_states, rnn_states_critic, action, value, returns, masks, active_masks, _, _, available_actions = self.sample[:14]
        b = torch.rand(64865)
        x = self.trainer.conjugate_gradient(self.policy.actor, agent_id, nodes_feats, edge_index, edge_attr, rnn_states, action, masks, available_actions, active_masks, b, 10)

        # Assertions
        self.assertIsInstance(x, torch.Tensor)

    def test_trpo_update(self):
        train_info = self.trainer.trpo_update(self.sample)

        # Assertions
        self.assertIsInstance(train_info, tuple)
        self.assertGreaterEqual(len(train_info), 1)

    # def test_train(self):
    #     # Mock buffer
    #     class MockBuffer:
    #         def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length, cost_adv=None):
    #             yield self.sample

    #     buffer = MockBuffer()
    #     train_info = self.trainer.train(buffer)

    #     # Assertions
    #     self.assertIsInstance(train_info, dict)
    #     self.assertIn('value_loss', train_info)
    #     self.assertIn('kl', train_info)
    #     self.assertIn('dist_entropy', train_info)

if __name__ == "__main__":
    unittest.main()

# python -m unittest tests/test_ssmarl_trainer.py
