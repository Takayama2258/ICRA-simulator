from gym.envs.registration import register

register(
    id='icra-simulator-v0',
    entry_point='ICRA_simulator.envs:kernal',
)
