import numpy as np
from ssmarl.envs.mpe_env.multiagent.core import World, Agent, Landmark
from ssmarl.envs.mpe_env.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.sensor_range = 1
        self.communication_range = 0.7

    def make_world(self,args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_landmarks = 2
        world.world_length = args.episode_length
        self.world_size = 3
        world.world_size = self.world_size
        world.collaborative = True
        self.arena_size = 1.5
        self.num_agents = args.num_agents
        num_agents = self.num_agents
        self.total_sep = 1.25*self.arena_size
        self.ideal_sep = self.total_sep / (self.num_agents-1)
        self.dist_thres = 0.05
        self.identity_size = 0
        world.num_landmarks = 2
        world.num_obstacles = 0
        world.num_agents = args.num_agents

        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.adversary = False
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        world.obstacles = []
        # make initial conditions
        self.reset_world(world)
        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        # colors = [np.array([0,0,0.1]), np.array([0,1,0]), np.array([0,0,1]), np.array([1,1,0]), np.array([1,0,0])]
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # agent.color = colors[i]

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.random.uniform(-.25*self.arena_size, .25*self.arena_size, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        
        theta = np.random.uniform(0, 2*np.pi)
        loc = world.landmarks[0].state.p_pos + self.total_sep*np.array([np.cos(theta), np.sin(theta)])
        # find a suitable theta such that landmark 1 is within the bounds
        while not(abs(loc[0])<self.arena_size and abs(loc[1])<self.arena_size):
            theta += np.radians(5)
            loc = world.landmarks[0].state.p_pos + self.total_sep*np.array([np.cos(theta), np.sin(theta)])

        world.landmarks[1].state.p_pos = loc
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        self.expected_positions = [world.landmarks[0].state.p_pos + i*self.ideal_sep*np.array([np.cos(theta), np.sin(theta)]) 
                                   for i in range(len(world.agents))]

        world.steps = 0
        world.dists = []

    def reward(self, agent, world):
        dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_positions] for a in world.agents])
        # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        self.delta_dists = self._bipartite_min_dists(dists) 
        world.dists = self.delta_dists

        total_penalty = np.mean(np.clip(self.delta_dists, 0, 2))
        self.joint_reward = -total_penalty
            
        return self.joint_reward

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # Graph feature initialization
        # Returns a large graph containing local observations of all agents
        # node_feature is listed in the following order: Agent 0 | landmark 1 | obstacle 2
        # The agents are numbered in sequential order
        node_feature = [0] * world.num_agents + [1] * 2 + [2] * world.num_obstacles
        edge_index = [[], []]
        edge_feature = []
        edge_num = 0
        
        for i, entity_i in enumerate(world.agents):
            for j in range(i+1, world.num_agents):
                entity_j = world.agents[j]
                # The entity within the communication radius of entity_i will be added to the edge list
                dist = np.linalg.norm(entity_i.state.p_pos - entity_j.state.p_pos)
                if dist < self.communication_range and entity_i.name != entity_j.name:
                    edge_num += 1
                    edge_index[0].append(j)
                    edge_index[1].append(i)
                    relative_state = np.hstack((entity_j.state.p_pos-entity_i.state.p_pos, entity_j.state.p_vel-entity_i.state.p_vel))
                    edge_feature.append(relative_state)

            # There is only two landmarks in line task
            edge_num += 1
            edge_index[0].append(world.num_agents)
            edge_index[1].append(i)
            goal = world.landmarks[0]
            relative_state = np.hstack((goal.state.p_pos-entity_i.state.p_pos, goal.state.p_vel-entity_i.state.p_vel))
            edge_feature.append(relative_state)

            edge_num += 1
            edge_index[0].append(world.num_agents + 1)
            edge_index[1].append(i)
            goal = world.landmarks[1]
            relative_state = np.hstack((goal.state.p_pos-entity_i.state.p_pos, goal.state.p_vel-entity_i.state.p_vel))
            edge_feature.append(relative_state)
            
            for j, obstacle in enumerate(world.obstacles):
                # Obstacles that are within the perception radius of entity_i will be added to the edge list
                dist = np.linalg.norm(entity_i.state.p_pos - obstacle.state.p_pos)
                if dist < self.sensor_range:
                    edge_num += 1
                    edge_index[0].append(2*world.num_agents + j)
                    edge_index[1].append(i)
                    relative_state = np.hstack((obstacle.state.p_pos-entity_i.state.p_pos, obstacle.state.p_vel-entity_i.state.p_vel))
                    edge_feature.append(relative_state)

        return node_feature, edge_index, edge_feature

    def cost(self, agent, world):
        cost = 0.0
        if agent.collide:
            for a in world.agents:
                # do not consider collision with itself
                if a.name == agent.name:
                    continue
                if self.is_collision(a, agent):
                    cost += 1.0
            for b in world.obstacles:
                if self.is_collision(agent, b):
                    cost += 1.0
        return np.array([cost])

    def info(self, agent, world):
        agent_id = id = int(agent.name.split(' ')[1])
        info = {'agent_id':agent_id}
        return info
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (agent1.size + agent2.size)
        return True if dist < dist_min else False