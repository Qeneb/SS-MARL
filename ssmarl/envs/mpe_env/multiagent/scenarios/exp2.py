import numpy as np
from ssmarl.envs.mpe_env.multiagent.core import World, Agent, Landmark, Obstacle
from ssmarl.envs.mpe_env.multiagent.scenario import BaseScenario
import math


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        self.world_size = math.sqrt(16*args.num_agents/3)
        world.world_size = self.world_size
        self.sensor_range = 1
        self.communication_range = 1
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        self.num_agents = world.num_agents
        world.num_obstacles = args.num_obstacles
        self.num_obstacles = world.num_obstacles
        world.num_landmarks = args.num_landmarks
        self.num_landmarks = world.num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add obstacles
        world.obstacles = [Obstacle() for i in range(world.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.silent = True
            obstacle.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        world.assign_obstacle_colors()
        world.assign_landmark_colors()
        # Randomly place agents and landmarks
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            obstacle.state.p_vel = np.zeros(world.dim_p)
        # set agents at random positions not colliding with obstacles
        num_agents_added = 0
        agents_added = []
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            agent_size = world.agents[num_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            agent_collision = self.check_agent_collision(
                random_pos, agent_size, agents_added
            )
            if not obs_collision and not agent_collision:
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                agents_added.append(world.agents[num_agents_added])
                num_agents_added += 1
        num_agents_added = 0
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            agent_size = world.agents[num_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            agent_collision = self.check_agent_collision(
                random_pos, agent_size, agents_added
            )
            if not obs_collision and not agent_collision:
                world.agents[
                    num_agents_added
                ].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(
                    world.dim_p
                )
                world.agents[num_agents_added].state.c = np.zeros(
                    world.dim_c
                )
                agents_added.append(world.agents[num_agents_added])
                num_agents_added += 1
        num_goals_added = 0
        goals_added = []
        while True:
            if num_goals_added == self.num_agents:
                break
            random_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            goal_size = world.landmarks[num_goals_added].size
            obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
            landmark_collision = self.is_landmark_collision(
                random_pos, goal_size, world.landmarks[:num_goals_added]
            )
            if not landmark_collision and not obs_collision:
                world.landmarks[num_goals_added].state.p_pos = random_pos
                world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
                num_goals_added += 1

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (agent1.size + agent2.size)
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        goal = world.get_entity('landmark', agent.name)

        dist_to_goal = np.sqrt(
            np.sum(np.square(agent.state.p_pos - goal.state.p_pos))
        )
        if dist_to_goal < agent.size/3:
            rew += 10
        else:
            rew -= dist_to_goal

        return rew

    def observation(self, agent, world):
        # Graph feature initialization
        # Returns a large graph containing local observations of all agents
        # node_feature is listed in the following order: Agent 0 | end point 1 | obstacle 2
        # The agents are numbered in sequential order
        node_feature = [0] * world.num_agents + [1] * world.num_agents + [2] * world.num_obstacles
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

            # The goal of entity_i is added to the edge list
            edge_num += 1
            edge_index[0].append(world.num_agents+i)
            edge_index[1].append(i)
            goal = world.get_entity('landmark', entity_i.name)
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

    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of agent with other agents
    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.p_pos - pos
                dist = np.linalg.norm(delta_pos)
                if dist < (agent.size + agent_size):
                    collision = True
                    break
        return collision

    def is_landmark_collision(self, pos, size: float, landmark_list: list) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + landmark.size
            if dist < dist_min:
                collision = True
                break
        return collision