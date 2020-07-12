import numpy as np
from configuration import conf
from models.model_definition import get_model_multi_view_models
from actions.multi_view import Multi_view
from tqdm import tqdm


def multi_view_train(data_loader, args, load_weights=False):
    epochs = args.epochs
    verbose = args.verbose
    model_list = []
    initial_weights = None
    multi_view_functions = []
    if args.same_initial:
        model = get_model_multi_view_models(len(conf.task_labels[0]) if conf.multi_head else 10, args.num_centers,
                                            args.deep)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        model = get_model_multi_view_models(len(conf.task_labels[task_idx]) if conf.multi_head else 10,
                                            args.num_centers, args.deep)
        if initial_weights is not None:
            model.set_weights(initial_weights)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        if task_idx == 0:
            print(model.summary())
        multi_view_functions.append(Multi_view())
        multi_view_functions[task_idx].fit(x)
        if load_weights:
            model.load_weights('./ckpt/{}/task{}.pkl'.format(conf.dataset_name, task_idx))
        else:
            bar = tqdm(range(epochs), ascii=True, desc='Training start')
            for e in bar:
                num_batches = 0
                while num_batches < len(x) / conf.batch_size:
                    x_batch, y_batch = data_loader.sample(task_idx=task_idx)
                    x_batch, y_batch = multi_view_functions[task_idx].augment(x_batch, y_batch, concat=True, num_runs=1)
                    y_batch = {'task_output': np.ones(y_batch.shape[0]), 'clf_output': y_batch}
                    loss = model.train_on_batch(x_batch, y_batch)
                    num_batches += 1
                    desc = 'Task {} Epochs {}/{}, '.format(task_idx, e, epochs)
                    for l in range(len(loss)):
                        desc += str(model.metrics_names[l]) + '  ' + str(loss[l])[:6] + '  '
                    if verbose:
                        bar.set_description(desc=desc, refresh=True)
                    if num_batches >= len(x) / conf.batch_size:
                        break
                print(' ')
        model_list.append(model)

    return model_list, multi_view_functions
