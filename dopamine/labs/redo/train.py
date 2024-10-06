# coding=utf-8
# Copyright 2023 ReDo authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running a Dopamine agent.

"""
#import some library to cammand file log app
from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains import run_experiment #运行强化学习任务。它将结合用户传入的命令行参数（如选择的代理、游戏环境等），并在实验过程中执行强化学习的核心步骤。
from dopamine.labs.atari_100k import eval_run_experiment  #评估训练后的代理
from dopamine.labs.redo import recycled_atari100k_rainbow_agent #Rainbow DQN 是一种增强的 DQN 算法
from dopamine.labs.redo import recycled_dqn_agents  #另一类强化学习代理模型，基于经典的 DQN（Deep Q-Learning Network）算法，并通过回收和再利用神经元的策略（"ReDo" 算法）来改进模型的表现
from dopamine.labs.redo import recycled_rainbow_agent 

import gin
import tensorflow as tf

#控制训练的基础目录（base_dir）、配置文件路径、日志等
ATARI_REPLAY_DIR = None

flags.DEFINE_string('replay_dir', ATARI_REPLAY_DIR, 'Data dir.')
flags.DEFINE_string(
    'replay_dir_suffix',
    'replay_logs',
    'Data is to be read from "replay_dir/.../{replay_dir_suffix}"',
)

flags.DEFINE_string(
    'base_dir', None, 'Base directory to host all required sub-directories.'
)
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.'
)
flags.DEFINE_multi_string(
    'gin_bindings',
    [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").',
)


FLAGS = flags.FLAGS

#根据传入的 agent_name 参数创建不同类型的代理
@gin.configurable
def create_agent_recycled(
    sess,
    environment,
    agent_name=None,
    summary_writer=None,
    debug_mode=True,#False,
):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  del sess
  if not debug_mode:
    summary_writer = None
  if agent_name.startswith('dqn'):
    return recycled_dqn_agents.RecycledDQNAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer
    )
  elif agent_name.startswith('rainbow'):
    return recycled_rainbow_agent.RecycledRainbowAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer
    )
  elif agent_name.startswith('atari100k'):
    return recycled_atari100k_rainbow_agent.RecycledAtari100kRainbowAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer
    )
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))

#创建 Runner，它用于控制训练和评估过程
@gin.configurable
def create_runner_recycled(
    base_dir,
    schedule='continuous_train_and_eval',
    max_episode_eval=False,
):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.
    max_episode_eval: Whether to use `MaxEpisodeEvalRunner` or not.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  if schedule == 'continuous_train_and_eval':
    if max_episode_eval:
      runner_fn = eval_run_experiment.MaxEpisodeEvalRunner
      logging.info('Using MaxEpisodeEvalRunner for evaluation.')
      return runner_fn(base_dir, create_agent_recycled)
    else:
      return run_experiment.Runner(base_dir, create_agent_recycled)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_recycled)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_v2_behavior()

  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = create_runner_recycled(base_dir)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
