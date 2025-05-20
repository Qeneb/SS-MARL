import numpy as np
import math
from ssmarl.envs.mpe_env.multiagent.core import World, Agent, Landmark, Obstacle
from ssmarl.envs.mpe_env.multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        self.world_size = 4
        world.world_size = self.world_size
        self.sensor_range = 1.0
        self.communication_range = 1.0
        world.world_length = args.episode_length
        # set any world properties first
        world.num_agents = args.num_agents
        self.num_agents = world.num_agents
        world.num_obstacles = args.num_obstacles
        self.num_obstacles = world.num_obstacles
        world.num_landmarks = args.num_landmarks
        self.num_landmarks = world.num_landmarks
        world.num_costs = args.num_costs
        self.num_costs = world.num_costs
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
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        # make initial conditions
        self.reset_world(world)
        return world

    def generate_circles(self, num_circles, square_size):
        center, circles = [], []
        max_radius = square_size / 8
        for _ in range(num_circles):
            while True:
                x = random.uniform(-square_size / 2, square_size / 2)
                y = random.uniform(-square_size / 2, square_size / 2)
                radius = random.uniform(0.1, max_radius)

                valid = True
                for cx, cy, cr in circles:
                    distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if distance < (radius + cr):
                        valid = False
                        break

                if x - radius < -square_size / 2 or x + radius > square_size / 2 or y - radius < -square_size / 2 or y + radius > square_size / 2:
                    valid = False

                if valid:
                    circles.append((round(x, 2), round(y, 2), round(radius, 2)))
                    break
        radius = []
        for i, (x, y, r) in enumerate(circles):
            center.append([x, y])
            radius.append(r)
        return center, radius

    def reset_world(self, world):

        world.assign_agent_colors()
        world.assign_obstacle_colors()
        world.assign_landmark_colors()
        obs_pos, obs_size = self.generate_circles(self.num_obstacles, self.world_size)
        for i, obstacle in enumerate(world.obstacles):
            obstacle.state.p_pos = np.array(obs_pos[i])
            obstacle.state.p_vel = np.zeros(world.dim_p)
            obstacle.size = obs_size[i]

        for i, agent in enumerate(world.agents):
            rand_angle = 2 * np.pi * (random.random() - 0.5) / 20
            agent.state.p_pos = 0.4 * np.array(
                [np.cos(2 * i * np.pi / world.num_agents + rand_angle),
                 np.sin(2 * i * np.pi / world.num_agents + rand_angle)]) - np.array([1.5, 1.5])
            # y = (self.world_size - agent.size) * (0.5-i/(self.num_agents-1))
            # agent.state.p_pos = np.array([-2.8, y])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.state.p_pos = 0.4 * np.array([np.cos(2 * i * np.pi / world.num_agents + np.pi / 6),
                                                   np.sin(2 * i * np.pi / world.num_agents + np.pi / 6)]) + np.array(
                [1.4, 1.5])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = (1, 1, 1)

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
        dist_min = (agent1.size + agent2.size) * 1.1
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0

        goal = world.get_entity('landmark', agent.name)

        dist_to_goal = np.sqrt(
            np.sum(np.square(agent.state.p_pos - goal.state.p_pos))
        )
        if dist_to_goal < agent.size / 2:
            rew += 1
        else:
            rew -= dist_to_goal
        return rew

    def formation_cost(self, agent, world):
        positions = []
        expected_positions = []
        for a in world.agents:
            positions.append(a.state.p_pos)

        for a in world.landmarks:
            expected_positions.append(a.state.p_pos)

        positions = np.vstack(positions)
        expected_positions = np.vstack(expected_positions)

        formation_error = self.formation_error(expected_positions.T, positions.T) / self.num_agents

        if formation_error < 0.05:
            formation_costs = 0
        else:
            formation_costs = formation_error
        
        return formation_costs

    def formation_error(self, Target, P):
        N = Target.shape[1]
        # SVD
        L = np.eye(N) - np.ones([N, N]) * 1 / N
        S = np.matmul(np.matmul(P, L), Target.T)
        U, _, V = np.linalg.svd(S)

        # calculate gamma mat and T mat
        gamma = np.matmul(V.T, U)
        t = np.matmul((Target - np.matmul(gamma, P)), np.ones([N, 1])) * 1 / N

        # V mat
        v_x = np.kron(np.ones([N, 1]), np.expand_dims(np.array([1, 0]), axis=-1))
        v_y = np.kron(np.ones([N, 1]), np.expand_dims(np.array([0, 1]), axis=-1))

        # rotation correction and transit correction
        P_final = np.expand_dims(np.matmul(np.kron(np.eye(N), gamma), P.flatten(order='F')), axis=-1) + \
                  np.matmul(np.concatenate([v_x, v_y], axis=1), t)

        # calculate formation error
        error = P_final - np.expand_dims(Target.reshape(-1, order='F'), axis=-1)
        f_error = np.sqrt(1 / N * np.sum(error ** 2))
        return f_error

    def observation(self, agent, world):
        node_feature = [0] * world.num_agents + [1] * world.num_agents + [2] * world.num_obstacles
        edge_index = [[], []]
        edge_feature = []
        edge_num = 0

        for i, entity_i in enumerate(world.agents):
            for j in range(i + 1, world.num_agents):
                entity_j = world.agents[j]
                dist = np.linalg.norm(entity_i.state.p_pos - entity_j.state.p_pos)
                if dist < self.communication_range and entity_i.name != entity_j.name:
                    edge_num += 1
                    edge_index[0].append(j)
                    edge_index[1].append(i)
                    relative_state = np.hstack(
                        (entity_j.state.p_pos - entity_i.state.p_pos, entity_j.state.p_vel - entity_i.state.p_vel))
                    edge_feature.append(relative_state)

            edge_num += 1
            edge_index[0].append(world.num_agents + i)
            edge_index[1].append(i)
            goal = world.get_entity('landmark', entity_i.name)
            relative_state = np.hstack(
                (goal.state.p_pos - entity_i.state.p_pos, - entity_i.state.p_vel))
            edge_feature.append(relative_state)

            for j, obstacle in enumerate(world.obstacles):
                dist = np.linalg.norm(entity_i.state.p_pos - obstacle.state.p_pos)
                if dist < self.sensor_range:
                    edge_num += 1
                    edge_index[0].append(2 * world.num_agents + j)
                    edge_index[1].append(i)
                    relative_state = np.hstack(
                        (obstacle.state.p_pos - entity_i.state.p_pos, obstacle.state.p_vel - entity_i.state.p_vel))
                    edge_feature.append(relative_state)

        return node_feature, edge_index, edge_feature

    def collision_cost(self, agent, world):
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
        return cost
    
    def costs(self, agent, world):
        return [self.collision_cost, self.formation_cost]

    def info(self, agent, world):
        agent_id = id = int(agent.name.split(' ')[1])
        positions = []
        expected_positions = []
        for a in world.agents:
            positions.append(a.state.p_pos)

        for a in world.landmarks:
            expected_positions.append(a.state.p_pos)

        positions = np.vstack(positions)
        expected_positions = np.vstack(expected_positions)

        formation_error = self.formation_error(expected_positions.T, positions.T) / self.num_agents

        info = {'agent_id': agent_id, 'formation_error': formation_error}
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

    # check collision of agent with another agent
    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

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

    def done_callback(self, agent, world):
        done = False
        goal = world.get_entity('landmark', agent.name)

        dist_to_goal = np.sqrt(
            np.sum(np.square(agent.state.p_pos - goal.state.p_pos))
        )
        if dist_to_goal < agent.size:
            done = True
        else:
            done = False
        return done