
def make_env(args, benchmark=False):
    '''
    Creates a MultiAgentEnv / MultiAgentConstrainEnv / MultiAgentGraphConstrainEnv object as env.
    '''
    from ssmarl.envs.mpe_env.multiagent.environment import MultiAgentEnv, MultiAgentConstrainEnv, MultiAgentGraphConstrainEnv
    import ssmarl.envs.mpe_env.multiagent.scenarios as scenarios

    scenario_name = args.scenario_name

    scenario = scenarios.load(scenario_name + ".py").Scenario()

    world = scenario.make_world(args)

    env = MultiAgentGraphConstrainEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                          scenario.cost, scenario.info)

    return env
