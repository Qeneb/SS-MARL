import unittest
import numpy as np
from unittest.mock import MagicMock
from ssmarl.envs.mpe_env.multiagent.environment import MultiAgentGraphConstrainEnv
from ssmarl.utils.graph_share_buffer import GraphShareReplayBuffer
from tests.test_scenario import Scenario


class TestMPEEnv(unittest.TestCase):
    def setUp(self):
        args = MagicMock()
        args.episode_length = 10
        args.n_rollout_threads = 1
        args.hidden_size = 64
        args.recurrent_N = 1
        args.gamma = 0.99
        args.gae_lambda = 0.95
        args.use_gae = True
        args.use_popart = False
        args.use_valuenorm = False
        args.use_proper_time_limits = False
        args.algorithm_name = "PPO"
        args.num_agents = 3
        args.num_obstacles = 4
        args.num_landmarks = 3
        args.num_costs = 2
        self.args = args
        self.NODE_DIM = 1
        self.EDGE_DIM = 4
        scenario = Scenario()
        # Initialize environment with the scenario
        self.env = MultiAgentGraphConstrainEnv(scenario.make_world(args), 
                                               scenario.reset_world, 
                                               scenario.reward, 
                                               scenario.observation,
                                               scenario.costs, 
                                               scenario.info)
        
        # Mock graph observation shapes
        self.graph_obs_shape = (((args.num_agents + args.num_obstacles + args.num_landmarks), self.NODE_DIM), (self.args.num_agents * (self.args.num_agents + self.args.num_obstacles), 2), (self.args.num_agents * (self.args.num_agents + self.args.num_obstacles), self.EDGE_DIM))  # (nodes_feats_shape, edge_index_shape, edge_attr_shape)
        
        # Mock action space
        self.act_space = MagicMock()
        self.act_space.__class__.__name__ = 'Box'
        self.act_space.n = 2

        self.buffer = GraphShareReplayBuffer(self.args, self.graph_obs_shape, self.act_space)
        self.buffer.factor = np.ones((self.args.episode_length, self.args.n_rollout_threads, self.args.num_agents, self.act_space.n), dtype=np.float32)
    def test_step_with_scenario(self):
        """Test the step process of the environment using the scenario."""
        # Reset the environment
        obs = self.env.reset()
        self.assertIsInstance(obs, tuple)
        self.assertEqual(len(obs), 3)  # Node features, edge indices, edge features

        # Run a single episode
        for step in range(self.env.world.world_length):
            # Generate random actions for each agent
            actions = [np.array([1.0, 1.0]) for _ in range(self.env.n)]
            
            # Step the environment
            nodes_feats_n, edge_index_n, edge_attr_n, reward_n, cost_n, done_n, info_n = self.env.step(actions)
            
            # Verify the output
            # Nodes number = number of agents + number of obstacles + number of landmarks
            # number of agents = number of landmarks
            # Max edges = (Agent number) * (Agent number + Obstacle number)
            if step == 0:
                print(f"Nodes features: {np.array(nodes_feats_n).shape}")
                self.assertEqual(np.array(nodes_feats_n).shape, (self.args.num_agents, self.args.num_agents + self.args.num_obstacles + self.args.num_landmarks))
                print(f"Edge index: {np.array(edge_index_n).shape}")
                self.assertEqual(np.array(edge_index_n).shape, (self.args.num_agents, 2, self.args.num_agents * (self.args.num_agents + self.args.num_obstacles)))
                print(f"Edge attributes: {np.array(edge_attr_n).shape}")
                self.assertEqual(np.array(edge_attr_n).shape, (self.args.num_agents, self.args.num_agents * (self.args.num_agents + self.args.num_obstacles), self.EDGE_DIM))
                print(f"Rewards: {np.array(reward_n).shape}")
                self.assertEqual(np.array(reward_n).shape, (self.args.num_agents, 1))
                print(f"Costs: {np.array(cost_n).shape}")
                self.assertEqual(np.array(cost_n).shape, (self.args.num_agents, self.args.num_costs))
                print(f"Info: {np.array(info_n).shape}")
        
    def test_buffer_insert(self):
        """Test the step and buffer insert process of the environment using the scenario."""
        # Reset the environment
        obs = self.env.reset()
        self.assertIsInstance(obs, tuple)
        self.assertEqual(len(obs), 3)  # Node features, edge indices, edge features

        # prepare data
        self.agent_id = np.arange(self.args.num_agents).reshape(self.args.n_rollout_threads, self.args.num_agents, 1)
        # RNN states
        n_threads=self.args.n_rollout_threads
        n_agents=self.args.num_agents
        self.rnn_states = np.random.rand(n_threads, n_agents, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)
        self.rnn_states_critic = np.random.rand(n_threads, n_agents, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)
        self.rnn_states_costs = np.random.rand(n_threads, n_agents, self.args.num_costs, self.args.recurrent_N, self.args.hidden_size).astype(np.float32)

        self.action_log_probs = np.random.rand(n_threads, n_agents, self.act_space.n).astype(np.float32)

        self.value_preds = np.random.rand(n_threads, n_agents, 1).astype(np.float32)

        self.cost_preds = np.random.rand(n_threads, n_agents, self.args.num_costs).astype(np.float32)
        
        self.masks = np.random.randint(0, 2, size=(n_threads, n_agents, 1)).astype(np.float32)

        # Run a single episode
        for step in range(self.env.world.world_length):
            # Generate random actions for each agent
            actions = [np.array([1.0, 1.0]) for _ in range(self.env.n)]
            
            # Step the environment
            nodes_feats_n, edge_index_n, edge_attr_n, reward_n, cost_n, done_n, info_n = self.env.step(actions)
            
            # Put all states into buffer
             # Call insert method
            self.buffer.insert(
                agent_id=self.agent_id,
                nodes_feats=np.expand_dims(nodes_feats_n[0], -1),
                edge_index=edge_index_n[0],
                edge_attr=edge_attr_n[0],
                ##############################
                # fake data
                rnn_states=self.rnn_states,
                rnn_states_critic=self.rnn_states_critic,
                rnn_states_costs=self.rnn_states_costs,
                ##############################
                actions=actions,
                action_log_probs=self.action_log_probs,
                rewards=reward_n,
                value_preds=self.value_preds,
                costs=cost_n,
                cost_preds=self.cost_preds,
                masks=self.masks
            )

if __name__ == "__main__":
    unittest.main()

# python -m unittest tests.test_env