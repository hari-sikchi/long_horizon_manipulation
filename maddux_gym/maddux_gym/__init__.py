from gym.envs.registration import register
import sys
sys.path.append('maddux_gym/')
register(
    id='Maddux-v0',
    entry_point='maddux_gym.envs:MadduxEnv',
)