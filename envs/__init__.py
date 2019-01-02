from gym.envs.registration import register

register(
    id='FourRoom-v0',
    entry_point='envs.fourroom:FourRoomEnv'
)

register(
    id='Maze-v0',
    entry_point='envs.maze:MazeEnv'
)