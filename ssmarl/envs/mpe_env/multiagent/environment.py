import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from typing import Callable, List, Tuple, Dict, Union, Optional
from ssmarl.envs.mpe_env.multiagent.multi_discrete import MultiDiscrete


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self,
                 world,
                 reset_callback=None,
                 reward_callback=None,
                 observation_callback=None,
                 info_callback=None,
                 done_callback=None,
                 shared_viewer=True,
                 discrete_action=False,
                 scenario_name="simple_spread",
                 ):

        self.world = world
        self.current_step = 0
        self.agents = self.world.policy_agents
        self.scenario_name = scenario_name
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters True for MAPPO, False for CBF-QP
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.current_step = 0
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                from ssmarl.envs.mpe_env.multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from ssmarl.envs.mpe_env.multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from ssmarl.envs.mpe_env.multiagent import rendering
            # update bounds to center around agent
            cam_range = 2
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


class MultiAgentConstrainEnv(MultiAgentEnv):
    def __init__(self,
                 world,
                 reset_callback=None,
                 reward_callback=None,
                 observation_callback=None,
                 cost_callback=None,
                 info_callback=None,
                 done_callback=None,
                 shared_viewer=True,
                 discrete_action=False,
                 scenario_name="simple_n",
                 ):

        self.world = world
        self.agents = self.world.policy_agents
        self.scenario_name = scenario_name
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.cost_callback = cost_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = True
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        cost_n = []
        info_n = []
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            cost_n.append(self._get_cost(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        return obs_n, reward_n, cost_n, done_n, info_n

    def _get_cost(self, agent):
        if self.cost_callback is None:
            return np.array([0.0])
        else:
            return self.cost_callback(agent, self.world)


class MultiAgentGraphConstrainEnv(MultiAgentConstrainEnv):
    def __init__(self,
                 world,
                 reset_callback=None,
                 reward_callback=None,
                 observation_callback=None,
                 cost_callback=None,
                 info_callback=None,
                 done_callback=None,
                 shared_viewer=True,
                 discrete_action=False,
                 scenario_name="simple_n_graph",
                 ):

        self.world = world
        self.agents = self.world.policy_agents
        self.scenario_name = scenario_name
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.cost_callback = cost_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = True
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            max_N = len(self.world.entities)
            self.max_N = max_N
            self.max_Edges = int(max_N*max_N/2) + 1
            num_agents = len(self.agents)
            self.num_agents = num_agents
            if isinstance(observation_callback(agent, self.world)[0][0],int):
                obs_dim = 1
                nodes_feats_shape = (max_N, 1)
                share_nodes_feats_shape = (num_agents, max_N, 1)
            else:
                obs_dim = len(observation_callback(agent, self.world)[0][0])
                nodes_feats_shape = (max_N, obs_dim)
                share_nodes_feats_shape = (num_agents, max_N, obs_dim)
            
            edge_index_shape = (2, self.max_Edges)
            edge_attr_shape = (self.max_Edges, observation_callback(agent, self.world)[2][0].size)
            share_edge_index_shape = (num_agents, 2, self.max_Edges)
            share_edge_attr_shape = (num_agents, self.max_Edges, observation_callback(agent, self.world)[2][0].size)

            self.graph_obs_shape = (nodes_feats_shape, edge_index_shape, edge_attr_shape)
            self.share_graph_obs_shape = (share_nodes_feats_shape, share_edge_index_shape, share_edge_attr_shape)
            share_obs_dim = obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        self.current_step += 1
        reward_n, cost_n, done_n, info_n = [], [], [], []
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            
        # advance world state
        self.world.step()
        # record observation for each agent
        nodes_feats_n, edge_index_n, edge_attr_n = [], [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_i = list(self._get_obs(agent))
            obs_i[0] += [None] * (self.max_N - len(obs_i[0]))
            nodes_feats_n.append(obs_i[0])

            obs_i[1][0] += [None] * (self.max_Edges - len(obs_i[1][0]))
            obs_i[1][1] += [None] * (self.max_Edges - len(obs_i[1][1]))
            edge_index_n.append(obs_i[1])

            obs_i[2] += [np.array([None]*self.graph_obs_shape[-1][-1])] * (self.max_Edges - len(obs_i[2]))
            edge_attr_n.append(obs_i[2])
        for i, agent in enumerate(self.world.agents):
            reward = self._get_reward(agent)
            cost = self._get_cost(agent)
            env_info = self._get_info(agent)
            info = {"individual_reward": reward}
            info.update(env_info)
            done_n.append(self._get_done(agent))
            reward_n.append(reward)
            cost_n.append(cost)
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        cost = np.sum(cost_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n  
            cost_n = [[cost]] * self.n
        self.update_graph(edge_index_n[0])
        return nodes_feats_n, edge_index_n, edge_attr_n, reward_n, cost_n, done_n, info_n
    
    def reset(self):
        # reset world
        self.current_step = 0
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        nodes_feats_n, edge_index_n, edge_attr_n = [], [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_i = list(self._get_obs(agent))
            obs_i[0] += [None] * (self.max_N - len(obs_i[0]))
            nodes_feats_n.append(obs_i[0])

            obs_i[1][0] += [None] * (self.max_Edges - len(obs_i[1][0]))
            obs_i[1][1] += [None] * (self.max_Edges - len(obs_i[1][1]))
            edge_index_n.append(obs_i[1])

            obs_i[2] += [np.array([None]*self.graph_obs_shape[-1][-1])] * (self.max_Edges - len(obs_i[2]))
            edge_attr_n.append(obs_i[2])
        self.update_graph(edge_index_n[0])
        return nodes_feats_n, edge_index_n, edge_attr_n

    def update_graph(self, edge_idx):
        edge_list = []
        for i in range(len(edge_idx[0])):
            if edge_idx[0][i] is not None:
                index_1 = edge_idx[0][i]
                index_2 = edge_idx[1][i]
                edge_list.append([self.world.entities[index_1].name, self.world.entities[index_2].name])
            else:
                break
        self.world.edge_list = np.array(edge_list)

    def render(self, mode: str = "human", close: bool = False) -> List:
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == "human":
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            message = ""
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = "_"
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += other.name + " to " + agent.name + ": " + word + "   "
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it
                # (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from macpo.envs.mpe_env.multiagent import rendering

                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it
            # (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from macpo.envs.mpe_env.multiagent import rendering

            self.render_geoms = []
            self.render_geoms_xform = []

            self.comm_geoms = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                entity_comm_geoms = []

                if "agent" in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = entity.size / dim_c
                            offset.set_translation(
                                ci * comm_size * 2 - entity.size + comm_size, 0
                            )
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                else:
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = entity.size / dim_c
                            offset.set_translation(
                                ci * comm_size * 2 - entity.size + comm_size, 0
                            )
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)


            # add geoms to viewer
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        cam_range = self.world.world_size/2
        for i in range(len(self.viewers)):
            # cam_range = 2
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(
                pos[0] - cam_range,
                pos[0] + cam_range,
                pos[1] - cam_range,
                pos[1] + cam_range,
            )
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if "agent" in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(color, color, color)

            # render the graph connections
            edge_list = self.world.edge_list
            assert edge_list is not None, "Edge list should not be None"
            for entity1 in self.world.entities:
                for entity2 in self.world.entities:
                    e1_id, e2_id = entity1.name, entity2.name
                    if e1_id == e2_id:
                        continue
                    # if edge exists draw a line
                    if [e1_id, e2_id] in edge_list.tolist():
                        src = entity1.state.p_pos
                        dest = entity2.state.p_pos
                        self.viewers[i].draw_line(start=src, end=dest)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == "rgb_array"))

        return results