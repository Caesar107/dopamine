python dopamine/labs/redo/train.py --base_dir="C:/dopamine-master/results" --gin_bindings="create_agent_recycled.agent_name='dqn'" --gin_bindings="atari_lib.create_atari_environment.game_name='Pong'" --gin_bindings="OutOfGraphReplayBuffer.replay_capacity=100000" --gin_bindings="OutOfGraphReplayBuffer.batch_size=32" --gin_bindings="DQNAgent.epsilon_decay_period=50000" --gin_bindings="Runner.num_iterations=10" --gin_bindings="Runner.base_dir='C:/dopamine-master/results/metrics/tensorboard'" --gin_bindings="Runner.training_steps=10000" --gin_bindings="create_agent_recycled.debug_mode=True"


tensorboard --logdir="D:/onedrive/OneDrive - The University of Auckland/S3/COMPSCI 764/project/data/dopamine-master/results/metrics/tensorboard"

