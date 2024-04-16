from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=False,
                        help='The number of epochs for each task.')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=[None, 'cosine', 'simple', 'bic', 'ntueffnet'])
    parser.add_argument('--scheduler_rate', type=float, default=None)
    parser.add_argument('--opt_steps', type=int, nargs='+', default=None)

    parser.add_argument('--timeme', action='store_true')
    parser.add_argument('--nowand', choices=[0, 1], default=0, type=int)
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--job_number', type=int, default=None,
                        help='The job ID in Slurm.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')

    parser.add_argument('--ignore_other_metrics', type=int, choices=[0, 1], default=0,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help="If set, run program with partial epochs.")

    parser.add_argument('--disable_log', action='store_true',
                        help='Disable results logging.')
    parser.add_argument('--loss_log', action='store_true',
                        help='Enable loss logging')
    parser.add_argument('--conf_external_path', type=str, default=None)
    parser.add_argument('--examples_log', action='store_true',
                        help='Enable example logging')
    parser.add_argument('--examples_full_log', action='store_true',
                        help='Enable full example logging. Don\'t do this: very dangerous')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--savecheck', action='store_true',
                        help='Save checkpoint?')
    parser.add_argument('--loadcheck', type=str, default=None,
                        help='Path of the checkpoint to load (pkl file with \'interpr\')')
    parser.add_argument('--start_from', type=int, default=None, help="Task to start from")
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=False,
                        help='Minibatch size.')
