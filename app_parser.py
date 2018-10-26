import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--usage', type=str, default='app',
							   help='Usage of the application')
parser.add_argument('--mode', type=str, default='rgb',
							  help='Rendering mode')
parser.add_argument('--env', type=str, default='list_data_practice',
							 help='Environment list')
parser.add_argument('--env_presence', type=str, default='env_spec',
							 	      help='Environment spec')
parser.add_argument('--environment', type=str, default='MsPacman-v0',
									 help='Environment name')
parser.add_argument('--episodes', type=int, default=20,
								  help='Seens episode')
parser.add_argument('--timesteps', type=int, default=200,
								   help='Watchout series')
parser.add_argument('--policy_construct_file_path', type=str, default='weights/solid_state.h5',
													help='Constructing buma')
parser.add_argument('--policy_builder_file_path', type=str, default='solid_state.h5',
												  help='Builder buma')
parser.add_argument('--state_size', type=int, default=100800,
									help='Stateless definition of space based on observation')
parser.add_argument('--action_size', type=int, default=0,
									 help='Action condition')
parser.add_argument('--epochs', type=int, default=1,
								help='Train epochs')
parser.add_argument('--batch_size', type=int, default=64,
									help='Batching size')
parser.add_argument('--state_size_environment', type=str, default='manual',
												help='Common interactively recognition')
parser.add_argument('--reinforce', type=int, default=1,
								   help='Reinforce train based on all caption')
parser.add_argument('--daemonize', type=str, default='dqn',
								   help='Deep reinforcement learning based on a daemonization')
parser.add_argument('--dqn', type=str, default='haxlem',
							 help='Double model layers for your capacity of learn')
