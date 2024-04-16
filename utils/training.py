import torch
from utils import get_memory_mb
from utils.status import ProgressBar, create_stash, update_status, update_accs
from utils.loggers import *
from utils.loggers import LossLogger, ExampleFullLogger, DictxtLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from tqdm import tqdm
import sys
import pickle
import math
from copy import deepcopy
import torch.optim

import wandb


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, verbose=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    iterator = enumerate(dataset.test_loaders)
    if verbose:
        iterator = tqdm(iterator, total=len(dataset.test_loaders))

    for k, test_loader in iterator:
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for idx, data in enumerate(test_loader):
            if model.args.debug_mode and idx > 2:  # len(test_loader) / 2:
                continue
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)  # [:,0:100]

            _, pred = torch.max(outputs.data, 1)
            matches = pred == labels
            correct += torch.sum(matches).item()

            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                masked_matches = pred == labels
                correct_mask_classes += torch.sum(masked_matches).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def compute_average_logit(model: ContinualModel, dataset: ContinualDataset, subsample: float):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    prio = torch.zeros(dataset.N_CLASSES_PER_TASK *
                       dataset.N_TASKS).to(model.device)
    c = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        for idx, data in enumerate(test_loader):
            if idx / len(test_loader) > subsample:
                break
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                prio += outputs.sum(0)
                c += len(outputs)
    model.net.train(status)
    return prio.cpu() / c


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # global sig_pause
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print(args)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, mode='disabled' if args.nowand else 'online')

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = DictxtLogger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, test_loader = dataset.get_data_loaders()
            if hasattr(model, 'end_task'):
                model.end_task(dataset)

    if args.loadcheck is not None:
        dict_keys = torch.load(args.loadcheck.replace('interpr_', '').replace(
            '.pkl', '.pt'), map_location=torch.device("cpu"))
        for k in list(dict_keys):
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        for k in list(dict_keys):
            if '_features' in dict_keys:
                dict_keys.pop(k)

        model.load_state_dict(dict_keys)
        if os.path.exists(args.loadcheck.replace('interpr', 'bufferoni')):
            model.buffer = pickle.load(
                open(args.loadcheck.replace('interpr', 'bufferoni'), 'rb'))
        results, results_mask_classes, csvdump = pickle.load(
            open(args.loadcheck.replace('interpr', 'results'), 'rb'))
        if not args.disable_log:
            logger.load(csvdump)
        loaded_args = pickle.load(open(args.loadcheck, 'rb'))
        ignored_args = ['loadcheck', 'start_from', 'stop_after', 'conf_jobnum', 'conf_host', 'conf_timestamp', 'examples_log', 'examples_full_log',
                        'job_number', 'conf_git_commit', 'loss_log', 'seed', 'savecheck', 'notes', 'non_verbose', 'autorelaunch', 'force_compat', 'conf_external_path']
        mismatched_args = [x for x in vars(args) if x not in ignored_args and (
            x not in vars(loaded_args) or getattr(args, x) != getattr(loaded_args, x))]
        if len(mismatched_args):
            if 'force_compat' not in vars(args) or args.force_compat:
                print(
                    "WARNING: The following arguments do not match between loaded and current model:")
                print(mismatched_args)
            else:
                raise ValueError(
                    'The loaded model was trained with different arguments: {}'.format(mismatched_args))

        model.net.to(model.device)
        if os.path.exists(args.loadcheck.replace('interpr', 'bufferoni')):
            model.buffer.to(model.device)
        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    model_stash = create_stash(model, args, dataset)

    if args.loss_log:
        loss_logger = LossLogger(dataset.SETTING, dataset.NAME, model.NAME)
        loss_loggerstream = LossLogger(
            dataset.SETTING, dataset.NAME, model.NAME + 'stream')
        loss_loggerbuffer = LossLogger(
            dataset.SETTING, dataset.NAME, model.NAME + 'buffer')
        loss_loggerstreamunm = LossLogger(
            dataset.SETTING, dataset.NAME, model.NAME + 'streamUnm')
        loss_loggerbufferunm = LossLogger(
            dataset.SETTING, dataset.NAME, model.NAME + 'bufferUnm')
    if args.examples_full_log:
        assert not args.examples_log, "You cannot log both examples and full examples"
        example_full_logger = ExampleFullLogger(
            dataset.SETTING, dataset.NAME, model.NAME, args.batch_size)

    if "joint" not in args.model and not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()

    print(file=sys.stderr)

    for t in range(0 if args.start_from is None else args.start_from, dataset.N_TASKS if args.stop_after is None else args.stop_after):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if "joint" not in args.model and t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t - 1] = results[t - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

        model.evaluator = lambda: evaluate(model, dataset)
        model.evaluation_dsets = dataset.test_loaders
        model.evaluate = lambda dataset: evaluate(model, dataset)

        scheduler = model.get_scheduler()

        for epoch in range(args.n_epochs):
            if scheduler is not None:
                assert model.opt == scheduler.optimizer, "Optimizer changed uncontrollably, scheduler has no effect!!"

            if args.model == 'co2l' and t > 0 and args.co2l_task_epoch is not None and epoch > args.co2l_task_epoch:
                break
            if args.model.startswith('joint'):
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 12:
                    break

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss, loss_stream, loss_buff, loss_streamu, loss_buffu = model.observe(
                        inputs, labels, not_aug_inputs, logits, epoch=epoch)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss, loss_stream, loss_buff, loss_streamu, loss_buffu = model.observe(
                        inputs, labels, not_aug_inputs, epoch=epoch)
                    assert not math.isnan(loss)

                if args.loss_log:
                    loss_logger.log(loss)
                    loss_loggerstream.log(loss_stream)
                    loss_loggerbuffer.log(loss_buff)
                    loss_loggerstreamunm.log(loss_streamu)
                    loss_loggerbufferunm.log(loss_buffu)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

                if i % 100 == 0:
                    update_status(i, len(train_loader), epoch, t,
                                  loss, job_number=args.job_number)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0

            if (epoch % 10 == 0 or epoch == args.n_epochs - 1) and args.examples_full_log:
                print("Gathering full log task %d epoch %d" %
                      (t, epoch), file=sys.stderr)
                evaluate(model, dataset, example_logger=example_full_logger)

            if scheduler is not None:
                scheduler.step()

        if args.loss_log:
            loss_logger.write(task=t)
            loss_loggerstream.write(task=t)
            loss_loggerbuffer.write(task=t)
            loss_loggerstreamunm.write(task=t)
            loss_loggerbufferunm.write(task=t)

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # possible checkpoint saving
        if "joint" not in args.model or t == dataset.N_TASKS - 1:

            accs = evaluate(model, dataset,
                            verbose=not model.args.non_verbose)
            print(accs)

            results.append(accs[0])
            results_mask_classes.append(accs[1])
            mean_acc = np.mean(accs, axis=1)

            postfix = "" if epoch is None else f"_epoch_{epoch}"
            d2 = {f'RESULT_class_mean_accs{postfix}': mean_acc[0], f'RESULT_task_mean_accs{postfix}': mean_acc[1],
                  **{f'RESULT_class_acc_{i}{postfix}': a for i, a in enumerate(accs[0])},
                  **{f'RESULT_task_acc_{i}{postfix}': a for i, a in enumerate(accs[1])},
                  'Task': t}

            wandb.log(d2)

            update_accs(mean_acc, dataset.SETTING, args.job_number)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            model_stash['mean_accs'].append(mean_acc)
            if not args.disable_log:
                logger.log(mean_acc)
                logger.log_fullacc(accs)

            if not os.path.isdir('checkpoints'):
                create_if_not_exists("checkpoints")
            if args.savecheck:
                print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)
                torch.save(model.state_dict(), f'checkpoints/{args.ckpt_name}_{t}.pt')
                if 'buffer_size' in model.args:
                    with open(f'checkpoints/{args.ckpt_name_replace.format("bufferoni")}_{t}.pkl', 'wb') as f:
                        pickle.dump(obj=deepcopy(
                            model.buffer).to('cpu'), file=f)
                with open(f'checkpoints/{args.ckpt_name_replace.format("interpr")}_{t}.pkl', 'wb') as f:
                    pickle.dump(obj=args, file=f)
                with open(f'checkpoints/{args.ckpt_name_replace.format("results")}_{t}.pkl', 'wb') as f:
                    pickle.dump(
                        obj=[results, results_mask_classes, logger.dump()], file=f)

    if args.examples_full_log:
        example_full_logger.write()

    if "joint" not in args.model and not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)

    if not args.disable_log:
        logger.write(vars(args))

    mem = get_memory_mb()
    res = {
        "local_proc_mem_MB": mem["self"],
        "child_proc_mem_MB": mem["children"],
        "total_proc_mem_MB": mem["total"],
    }

    # pretty print res
    print("Memory usage:")
    for k, v in res.items():
        print(f"\t{k}: {v} MB")

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
