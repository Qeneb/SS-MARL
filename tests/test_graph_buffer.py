import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from ssmarl.utils.graph_share_buffer import GraphShareReplayBuffer

class TestGraphShareReplayBufferInsert(unittest.TestCase):
    def setUp(self):
        # Create a mock args object
        self.args = MagicMock()
        self.args.episode_length = 5
        self.args.n_rollout_threads = 2
        self.args.hidden_size = 64
        self.args.recurrent_N = 1
        self.args.gamma = 0.99
        self.args.gae_lambda = 0.95
        self.args.use_gae = True
        self.args.use_popart = False
        self.args.use_valuenorm = False
        self.args.use_proper_time_limits = False
        self.args.algorithm_name = "PPO"
        self.args.num_costs = 1
        self.args.num_agents = 3
        
        # Mock graph observation shapes
        NODE_NUM = 10
        NODE_FEAT_DIM = 2
        EDGE_NUM = 5
        EDGE_FEAT_DIM = 2
        self.graph_obs_shape = ((NODE_NUM, NODE_FEAT_DIM), (EDGE_NUM, 2), (EDGE_NUM, EDGE_FEAT_DIM))  # (nodes_feats_shape, edge_index_shape, edge_attr_shape)
        
        # Mock action space
        self.act_space = MagicMock()
        self.act_space.__class__.__name__ = 'Discrete'
        self.act_space.n = 4
        
        # Create buffer instance
        self.buffer = GraphShareReplayBuffer(self.args, self.graph_obs_shape, self.act_space)
        self.buffer.factor = np.ones((self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, self.act_space.n), dtype=np.float32)
        
        # Prepare test data
        self._prepare_test_data()

    def _prepare_test_data(self):
        """Prepare test input data."""
        # Define data shapes using variables from the config
        episode_length = self.args.episode_length
        n_threads = self.args.n_rollout_threads
        n_agents = self.args.num_agents
        nodes_feats_shape, edge_index_shape, edge_attr_shape = self.graph_obs_shape
        
        # Observation data
        self.agent_id = np.random.randint(0, 10, size=(n_threads, n_agents, 1)).astype(np.int32)
        self.nodes_feats = np.random.rand(n_threads, *nodes_feats_shape).astype(np.float32)
        self.edge_index = np.random.rand(n_threads, *edge_index_shape).astype(np.float32)
        self.edge_attr = np.random.rand(n_threads, *edge_attr_shape).astype(np.float32)
        
        # RNN states
        self.rnn_states = np.random.rand(n_threads, n_agents, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)
        self.rnn_states_critic = np.random.rand(n_threads, n_agents, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)
        self.rnn_states_costs = np.random.rand(n_threads, n_agents, self.args.num_costs, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)
        
        # Action data
        act_shape = 1  # Shape of Discrete action space
        self.actions = np.random.randint(0, 4, size=(n_threads, n_agents, act_shape)).astype(np.float32)
        self.action_log_probs = np.random.rand(n_threads, n_agents, act_shape).astype(np.float32)
        
        # Rewards and value predictions
        self.rewards = np.random.rand(n_threads, n_agents, 1).astype(np.float32)
        self.value_preds = np.random.rand(n_threads, n_agents, 1).astype(np.float32)
        self.costs = np.random.rand(n_threads, n_agents, self.args.num_costs).astype(np.float32)
        self.cost_preds = np.random.rand(n_threads, n_agents, self.args.num_costs).astype(np.float32)
        
        # Masks
        self.masks = np.random.randint(0, 2, size=(n_threads, n_agents, 1)).astype(np.float32)
        self.bad_masks = np.random.randint(0, 2, size=(n_threads, n_agents, 1)).astype(np.float32)
        self.active_masks = np.random.randint(0, 2, size=(n_threads, n_agents, 1)).astype(np.float32)
        self.available_actions = np.random.randint(0, 2, size=(n_threads, n_agents, self.act_space.n)).astype(np.float32)

        # Fill next_value
        self.next_value = np.random.rand(n_threads, n_agents, 1).astype(np.float32)

    def test_insert_basic_data(self):
        """Test basic data insertion functionality."""
        # Record initial step value
        initial_step = self.buffer.step
        
        # Call insert method
        self.buffer.insert(
            agent_id=self.agent_id,
            nodes_feats=self.nodes_feats,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            rnn_states=self.rnn_states,
            rnn_states_critic=self.rnn_states_critic,
            rnn_states_costs=self.rnn_states_costs,
            actions=self.actions,
            action_log_probs=self.action_log_probs,
            rewards=self.rewards,
            value_preds=self.value_preds,
            costs=self.costs,
            cost_preds=self.cost_preds,
            masks=self.masks
        )
        
        # Verify step value has been updated
        self.assertEqual(self.buffer.step, (initial_step + 1) % self.args.episode_length)
        
        # Verify data has been correctly inserted into the buffer
        np.testing.assert_array_equal(self.buffer.agent_id[initial_step + 1], self.agent_id)
        np.testing.assert_array_equal(self.buffer.nodes_feats[initial_step + 1], self.nodes_feats)
        np.testing.assert_array_equal(self.buffer.edge_index[initial_step + 1], self.edge_index)
        np.testing.assert_array_equal(self.buffer.edge_attr[initial_step + 1], self.edge_attr)
        np.testing.assert_array_equal(self.buffer.rnn_states[initial_step + 1], self.rnn_states)
        np.testing.assert_array_equal(self.buffer.rnn_states_critic[initial_step + 1], self.rnn_states_critic)
        np.testing.assert_array_equal(self.buffer.rnn_states_costs[initial_step + 1], self.rnn_states_costs)
        np.testing.assert_array_equal(self.buffer.actions[initial_step], self.actions)
        np.testing.assert_array_equal(self.buffer.action_log_probs[initial_step], self.action_log_probs)
        np.testing.assert_array_equal(self.buffer.rewards[initial_step], self.rewards)
        np.testing.assert_array_equal(self.buffer.value_preds[initial_step], self.value_preds)
        np.testing.assert_array_equal(self.buffer.costs[initial_step], self.costs)
        np.testing.assert_array_equal(self.buffer.cost_preds[initial_step], self.cost_preds)
        np.testing.assert_array_equal(self.buffer.masks[initial_step + 1], self.masks)

    def test_insert_optional_data(self):
        """Test insertion of optional data (bad_masks, active_masks, available_actions)."""
        # Call insert method with all optional parameters
        self.buffer.insert(
            agent_id=self.agent_id,
            nodes_feats=self.nodes_feats,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            rnn_states=self.rnn_states,
            rnn_states_critic=self.rnn_states_critic,
            rnn_states_costs=self.rnn_states_costs,
            actions=self.actions,
            action_log_probs=self.action_log_probs,
            rewards=self.rewards,
            value_preds=self.value_preds,
            costs=self.costs,
            cost_preds=self.cost_preds,
            masks=self.masks,
            bad_masks=self.bad_masks,
            active_masks=self.active_masks,
            available_actions=self.available_actions
        )
        
        # Verify optional data has been correctly inserted
        np.testing.assert_array_equal(self.buffer.bad_masks[1], self.bad_masks)
        np.testing.assert_array_equal(self.buffer.active_masks[1], self.active_masks)
        np.testing.assert_array_equal(self.buffer.available_actions[1], self.available_actions)

    def test_insert_step_wrapping(self):
        """Test step value wrapping at the end of the buffer."""
        # Set step to the end of the buffer
        self.buffer.step = self.args.episode_length - 1
        
        # Call insert method
        self.buffer.insert(
            agent_id=self.agent_id,
            nodes_feats=self.nodes_feats,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            rnn_states=self.rnn_states,
            rnn_states_critic=self.rnn_states_critic,
            rnn_states_costs=self.rnn_states_costs,
            actions=self.actions,
            action_log_probs=self.action_log_probs,
            rewards=self.rewards,
            value_preds=self.value_preds,
            costs=self.costs,
            cost_preds=self.cost_preds,
            masks=self.masks
        )
        
        # Verify step value has wrapped around to 0
        self.assertEqual(self.buffer.step, 0)

    def test_compute_returns_with_gae(self):
        """Test computation of returns using GAE."""
        # Save original data for verification
        original_rewards = self.buffer.rewards.copy()
        original_value_preds = self.buffer.value_preds.copy()
        
        # Call the returns computation
        self.buffer.compute_returns(self.next_value)
        
        # Verify the shape of returns is correct
        self.assertEqual(self.buffer.returns.shape, (self.args.episode_length + 1, self.args.n_rollout_threads, self.args.num_agents, 1))
        
        # Verify the returns of the last step equal next_value
        np.testing.assert_array_equal(self.buffer.value_preds[-1], self.next_value)
        
        # Verify original data has not been modified
        np.testing.assert_array_equal(self.buffer.rewards, original_rewards)
        np.testing.assert_array_equal(self.buffer.value_preds[:-1], original_value_preds[:-1])

    def test_feed_forward_generator(self):
        """Test the feed_forward_generator function."""
        # Prepare test data
        advantages = np.random.rand(self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, 1).astype(np.float32)
        cost_advantages = np.random.rand(self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, self.args.num_costs).astype(np.float32)
        num_mini_batch = 4
        mini_batch_size = None
        
        # Call the generator
        generator = self.buffer.feed_forward_generator(advantages, num_mini_batch, mini_batch_size, cost_advantages)
        
        # Verify the generator output
        for batch in generator:
            # Check the structure of each batch
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 19)  # Number of items yielded by the generator
            
            # Check the shape of each item in the batch
            for item in batch:
                self.assertIsNotNone(item)
                if isinstance(item, np.ndarray):
                    self.assertTrue(item.size > 0, f"Expected non-empty array, got {item}")  # Ensure non-empty arrays
                elif isinstance(item, torch.Tensor):
                    self.assertTrue(item.numel() > 0)  # Ensure non-empty tensors

        # Test with mini_batch_size specified
        mini_batch_size = 2
        generator = self.buffer.feed_forward_generator(advantages, num_mini_batch, mini_batch_size, cost_advantages)
        
        # Verify the generator output
        for batch in generator:
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 19)  # Number of items yielded by the generator
            for k, item in enumerate(batch):
                self.assertIsNotNone(item, f"Expected non-empty item at index {k}, got {item}")
                if isinstance(item, np.ndarray):
                    self.assertTrue(item.size > 0, f"Expected non-empty array, got {item}")
                elif isinstance(item, torch.Tensor):
                    self.assertTrue(item.numel() > 0)

    def test_recurrent_generator(self):
        """Test the recurrent_generator function."""
        # Prepare test data
        advantages = np.random.rand(self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, 1).astype(np.float32)
        cost_advantages = np.random.rand(self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, self.args.num_costs).astype(np.float32)
        num_mini_batch = self.args.n_rollout_threads
        data_chunk_length = self.args.episode_length  # Adjusted to match episode_length
        
        # Call the generator
        generator = self.buffer.recurrent_generator(advantages, num_mini_batch, data_chunk_length, cost_advantages)
        
        # Verify the generator output
        for batch in generator:
            # Check the structure of each batch
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 19)  # Number of items yielded by the generator
            
            # Check the shape of each item in the batch
            for item in batch:
                self.assertIsNotNone(item)
                if isinstance(item, np.ndarray):
                    self.assertTrue(item.size > 0)  # Ensure non-empty arrays
                elif isinstance(item, torch.Tensor):
                    self.assertTrue(item.numel() > 0)  # Ensure non-empty tensors

if __name__ == '__main__':
    unittest.main()

# python -m unittest tests.test_graph_buffer