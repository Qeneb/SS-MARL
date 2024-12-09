import time
import imageio
import wandb
import numpy as np
import torch
from ssmarl.runner.base_runner import Runner
import csv


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.retrun_average_cost = 0

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer[0].policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                rnn_states_cost = self.collect(step)

                # observe reward cost and next obs
                nodes_feats, edge_index, edge_attr, rewards, costs, dones, infos = self.envs.step(actions)
                nodes_feats = np.expand_dims(nodes_feats, -1)
                edge_index = np.array(edge_index)
                edge_attr = np.array(edge_attr)
                share_nodes_feats = np.expand_dims(nodes_feats, 1).repeat(self.num_agents, axis=1)
                share_edge_index = np.expand_dims(edge_index, 1).repeat(self.num_agents, axis=1)
                share_edge_attr = np.expand_dims(edge_attr, 1).repeat(self.num_agents, axis=1)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                cost_env = np.mean(costs, axis=1).flatten()
                train_episode_rewards += reward_env
                train_episode_costs += cost_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_costs.append(train_episode_costs[t])
                        train_episode_costs[t] = 0
                done_episodes_costs_aver = np.mean(train_episode_costs)

                data = nodes_feats, edge_index, edge_attr, share_nodes_feats, share_edge_index, share_edge_attr, rewards, costs, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic,  cost_preds, rnn_states_cost, done_episodes_costs_aver

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_costs = np.mean(done_episodes_costs)
                    self.return_aver_cost(aver_episode_costs)
                    print("some episodes done, average rewards: {}, average costs: {}".format(aver_episode_rewards,
                                                                                              aver_episode_costs))
    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        if self.n_rollout_threads == 1:
            nodes_feats, edge_index, edge_attr = self.envs.reset()[0]
        else:
            result = self.envs.reset()
            nodes_feats, edge_index, edge_attr = [], [], []
            for i in range(self.n_rollout_threads):
                nodes_feats.append(result[i][0])
                edge_index.append(result[i][1])
                edge_attr.append(result[i][2])
        # replay buffer
        if self.n_rollout_threads == 1:
            graph_obs_shape = self.envs.envs[0].graph_obs_shape
            share_graph_obs_shape = self.envs.envs[0].share_graph_obs_shape
        else:
            graph_obs_shape = self.envs.graph_obs_shape
            share_graph_obs_shape = self.envs.share_graph_obs_shape
        nodes_feats = np.array(nodes_feats).reshape(self.n_rollout_threads,*share_graph_obs_shape[0])
        edge_index = np.array(edge_index).reshape(self.n_rollout_threads,*share_graph_obs_shape[1])
        edge_attr = np.array(edge_attr).reshape(self.n_rollout_threads,*share_graph_obs_shape[2])
        share_nodes_feats = np.expand_dims(nodes_feats, 1).repeat(self.num_agents, axis=1)
        share_edge_index = np.expand_dims(edge_index, 1).repeat(self.num_agents, axis=1)
        share_edge_attr = np.expand_dims(edge_attr, 1).repeat(self.num_agents, axis=1)

        agent_id_list = np.tile(np.linspace(0,self.num_agents - 1, self.num_agents),(self.n_rollout_threads)).reshape(self.n_rollout_threads, self.num_agents, 1).astype(int)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_nodes_feats[0] = share_nodes_feats[:, agent_id].copy()
            self.buffer[agent_id].share_edge_index[0] = share_edge_index[:, agent_id].copy()
            self.buffer[agent_id].share_edge_attr[0] = share_edge_attr[:, agent_id].copy()
            self.buffer[agent_id].nodes_feats[0] = nodes_feats[:, agent_id].copy()
            self.buffer[agent_id].edge_index[0] = edge_index[:, agent_id].copy()
            self.buffer[agent_id].edge_attr[0] = edge_attr[:, agent_id].copy()
            self.buffer[agent_id].agent_id[0] = agent_id_list[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        cost_preds_collector = []
        rnn_states_cost_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].agent_id[step],
                                                            self.buffer[agent_id].share_nodes_feats[step],
                                                            self.buffer[agent_id].share_edge_index[step],
                                                            self.buffer[agent_id].share_edge_attr[step],
                                                            self.buffer[agent_id].nodes_feats[step],
                                                            self.buffer[agent_id].edge_index[step],
                                                            self.buffer[agent_id].edge_attr[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step]
                                                            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            cost_preds_collector.append(_t2n(cost_pred))
            rnn_states_cost_collector.append(_t2n(rnn_state_cost))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        cost_preds = np.array(cost_preds_collector).transpose(1, 0, 2)
        rnn_states_cost = np.array(rnn_states_cost_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    def insert(self, data, aver_episode_costs = 0):
        aver_episode_costs = aver_episode_costs
        nodes_feats, edge_index, edge_attr, share_nodes_feats, share_edge_index, share_edge_attr, rewards, costs, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost, done_episodes_costs_aver = data # fixme:!!!
        agent_id_list = []
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                agent_id_list.append(infos[i][j]['agent_id'])
        agent_id_list = np.array(agent_id_list).reshape(self.n_rollout_threads, self.num_agents, 1)
        dones_env = np.all(dones, axis=1)

        

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        rnn_states_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(agent_id_list[:, agent_id], nodes_feats[:, agent_id], edge_index[:, agent_id], edge_attr[:, agent_id],
                                        share_nodes_feats[:, agent_id], share_edge_index[:, agent_id], share_edge_attr[:, agent_id],
                                        rnn_states[:, agent_id], rnn_states_critic[:, agent_id], actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                        active_masks[:, agent_id],  None, costs=costs[:, agent_id],
                                        cost_preds=cost_preds[:, agent_id],
                                        rnn_states_cost=rnn_states_cost[:, agent_id], done_episodes_costs_aver=done_episodes_costs_aver, aver_episode_costs=aver_episode_costs)

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)/self.num_agents))
        train_infos[0][0]["average_step_rewards"] = 0
        for agent_id in range(self.num_agents):
            train_infos[0][agent_id]["max_average_step_rewards"]= np.max(np.mean(self.buffer[agent_id].rewards, axis=0))/self.num_agents
            train_infos[0][agent_id]["min_average_step_rewards"]= np.min(np.mean(self.buffer[agent_id].rewards, axis=0))/self.num_agents
            train_infos[0][agent_id]["average_step_rewards"]= np.mean(self.buffer[agent_id].rewards)/self.num_agents
            train_infos[0][agent_id]["max_average_step_costs"]= np.max(np.mean(self.buffer[agent_id].costs, axis=0))/self.num_agents
            train_infos[0][agent_id]["min_average_step_costs"]= np.min(np.mean(self.buffer[agent_id].costs, axis=0))/self.num_agents
            train_infos[0][agent_id]["average_step_costs"]= np.mean(self.buffer[agent_id].costs)/self.num_agents
            for k, v in train_infos[0][agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        rews = 0
        costs = 0
        finishes = 0
        t = 0
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            episode_costs = []
            # reset env
            nodes_feats, edge_index, edge_attr = self.envs.reset()[0]

            graph_obs_shape = self.envs.envs[0].graph_obs_shape
            share_graph_obs_shape = self.envs.envs[0].share_graph_obs_shape
            nodes_feats = np.array(nodes_feats).reshape(self.n_rollout_threads, *share_graph_obs_shape[0])
            edge_index = np.array(edge_index).reshape(self.n_rollout_threads, *share_graph_obs_shape[1])
            edge_attr = np.array(edge_attr).reshape(self.n_rollout_threads, *share_graph_obs_shape[2])

            share_nodes_feats = np.expand_dims(nodes_feats, 1).repeat(self.num_agents, axis=1)
            share_edge_index = np.expand_dims(edge_index, 1).repeat(self.num_agents, axis=1)
            share_edge_attr = np.expand_dims(edge_attr, 1).repeat(self.num_agents, axis=1)

            agent_id_list = np.tile(np.linspace(0, self.num_agents - 1, self.num_agents),
                                    (self.n_rollout_threads)).reshape(self.n_rollout_threads, self.num_agents,
                                                                      1).astype(int)

            if self.all_args.use_render:
                image = self.envs.render("rgb_array")[0][0]
                if self.all_args.save_gifs:
                    all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                if self.all_args.parameter_share:
                    action, rnn_state = self.trainer[0].policy.act(np.array([range(0, self.num_agents)]),
                                                                   nodes_feats[:, 0],
                                                                   edge_index[:, 0],
                                                                   edge_attr[:, 0],
                                                                   rnn_states[:, 0],
                                                                   masks[:, 0],
                                                                   deterministic=True)
                    action = action.detach().cpu().numpy()
                    temp_actions_env = list(action.reshape(-1, 1, 2))
                else:
                    for agent_id in range(self.num_agents):
                        self.trainer[agent_id].prep_rollout()
                        action, rnn_state = self.trainer[agent_id].policy.act(agent_id_list[:, agent_id],
                                                                              nodes_feats[:, agent_id],
                                                                              edge_index[:, agent_id],
                                                                              edge_attr[:, agent_id],
                                                                              rnn_states[:, agent_id],
                                                                              masks[:, agent_id],
                                                                              deterministic=True)
                        action = action.detach().cpu().numpy()
                        # rearrange action
                        if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                            for i in range(self.envs.action_space[agent_id].shape):
                                uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                                if i == 0:
                                    action_env = uc_action_env
                                else:
                                    action_env = np.concatenate((action_env, uc_action_env), axis=1)
                        elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                            action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                        elif self.envs.action_space[agent_id].__class__.__name__ == 'Box':
                            action_env = action
                        else:
                            raise NotImplementedError

                        temp_actions_env.append(action_env)
                        rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # observe reward and next obs
                nodes_feats, edge_index, edge_attr, rewards, cost, dones, infos = self.envs.step(actions_env)
                nodes_feats = nodes_feats.reshape(*nodes_feats.shape, 1)
                agent_id_list = []
                for i in range(self.n_rollout_threads):
                    for j in range(self.num_agents):
                        agent_id_list.append(infos[i][j]['agent_id'])
                agent_id_list = np.array(agent_id_list).reshape(self.n_rollout_threads, self.num_agents, 1)

                episode_rewards.append(rewards)
                episode_costs.append(cost)

                if step != self.episode_length - 1:
                    if self.all_args.use_render:
                        image = self.envs.render("rgb_array")[0][0]
                        if self.all_args.save_gifs:
                            all_frames.append(image)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                calc_end = time.time()

            episode_rewards = np.array(episode_rewards)
            episode_costs = np.array(episode_costs)
            average_episode_costs = 0
            for agent_id in range(self.num_agents):
                average_episode_costs += np.mean(np.sum(episode_costs[:, :, agent_id], axis=0))

            average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, 0], axis=0))
            print(f'episode:{episode}')
            print("eval average episode rewards: " + str(average_episode_rewards))
            print("eval average episode costs: " + str(average_episode_costs))

            rews += average_episode_rewards
            costs += average_episode_costs

            print(f'rews:{rews}, costs:{costs}')
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)