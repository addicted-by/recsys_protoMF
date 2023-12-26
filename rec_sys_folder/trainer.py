import os
import optuna
import torch
from ray import train
from torch import nn
from torch.utils import data
from pathlib import Path
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys_folder.rec_sys import RecSys
from utilities.consts import OPTIMIZING_METRIC, MAX_PATIENCE
from utilities.eval import Evaluator
from datetime import datetime 
from rich.progress import track
import torch.nn.functional as F

def get_module_by_name(module, name):
    """ """
    for s in name.split("."):
        module = getattr(module, s)

    return module



class Trainer:

    def __init__(self, train_loader: data.DataLoader, val_loader: data.DataLoader, conf, user_factors=None, item_factors=None):
        """
        Train and Evaluate the model.
        :param train_loader: Training DataLoader (check music4all_data.Music4AllDataset for more info)
        :param val_loader: Validation DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.rec_sys_param = conf.rec_sys_param
        self.ft_ext_param = conf.ft_ext_param
        self.optim_param = conf.optim_param

        self.n_epochs = conf.n_epochs
        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.device = conf.device

        self.optimizing_metric = OPTIMIZING_METRIC
        self.max_patience = MAX_PATIENCE

        self.user_factors = user_factors
        self.item_factors = item_factors

        self.exp = datetime.now().strftime("%m%d%Y.%H%M%S")
        self.model = self._build_model()
        if conf.checkpoint_dir:
            (Path(conf.checkpoint_dir) / self.exp).mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = str(Path(conf.checkpoint_dir) / self.exp) if conf.checkpoint_dir else "."
        self.optimizer = self._build_optimizer()

        print(f'Built Trainer module \n'
              f'- n_epochs: {self.n_epochs} \n'
              f'- loss_func_name: {self.loss_func_name} \n'
              f'- loss_func_aggr: {self.loss_func_aggr} \n'
              f'- device: {self.device} \n'
              f'- optimizing_metric: {self.optimizing_metric} \n',
              f'- checkpoint_dir: {self.checkpoint_dir} \n',
              )

    def _build_model(self):
        # Step 1 --- Building User and Item Feature Extractors
        n_users = self.train_loader.dataset.n_users
        n_items = self.train_loader.dataset.n_items
        user_feature_extractor, item_feature_extractor = \
            FeatureExtractorFactory.create_models(self.ft_ext_param, n_users, n_items, self.user_factors, self.item_factors)
        

        # Step 2 --- Building RecSys Module
        rec_sys = RecSys(n_users, n_items, self.rec_sys_param, user_feature_extractor, item_feature_extractor,
                         self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()


        rec_sys = nn.DataParallel(rec_sys)
        rec_sys = rec_sys.to(self.device)

        return rec_sys

    def _build_optimizer(self):
        self.lr = self.optim_param['lr'] if 'lr' in self.optim_param else 1e-3
        self.wd = self.optim_param['wd'] if 'wd' in self.optim_param else 1e-4

        optim_name = self.optim_param['optim']
        if optim_name == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optim_name == 'adagrad':
            optim = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError('Optimizer not yet included')

        print(f'Built Optimizer  \n'
              f'- name: {optim_name} \n'
              f'- lr: {self.lr} \n'
              f'- wd: {self.wd} \n')

        return optim

    def run(self, trial=None):
        """
        Runs the Training procedure
        """
        metrics_values = self.val()
        best_value = metrics_values[self.optimizing_metric]
        train.report(metrics_values)
        print('Init - Avg Val Value {:.3f} \n'.format(best_value))

        patience = 0
        for epoch in range(self.n_epochs):

            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break

            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in track(self.train_loader, total=len(self.train_loader)):
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(u_idxs, i_idxs)

                loss = self.model.module.loss_func(out, labels)

                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_train_loss /= len(self.train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))

            metrics_values = self.val()
            curr_value = metrics_values[self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value))
            train.report({**metrics_values, 'epoch_train_loss': epoch_train_loss})

            if curr_value > best_value:
                best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                # with train.checkpoint_dir(0) as checkpoint_dir:

                torch.save(self.model.module.state_dict(), os.path.join(self.checkpoint_dir, f'best_model_{epoch + 1}.pth'))
                patience = 0
            else:
                patience += 1
            if trial:
                trial.report(best_value, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return best_value

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: A scalar float value, output of the validation (e.g. NDCG@10).
        """
        self.model.eval()
        print('Validation started')
        val_loss = 0
        eval = Evaluator(self.val_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.val_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            val_loss += self.model.module.loss_func(out, labels).item()

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out)

        val_loss /= len(self.val_loader)
        metrics_values = {**eval.get_results(), 'val_loss': val_loss}

        return metrics_values


class DistanceTrainer:
    def __init__(self, train_loader: data.DataLoader, val_loader: data.DataLoader, conf, load_only: bool=False, user_factors=None, item_factors=None):
        """
        Train and Evaluate the model.
        :param train_loader: Training DataLoader (check music4all_data.Music4AllDataset for more info)
        :param val_loader: Validation DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.rec_sys_param = conf.rec_sys_param
        self.ft_ext_param = conf.ft_ext_param
        self.optim_param = conf.optim_param

        self.n_epochs = conf.n_epochs
        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.user_alpha = conf.user_alpha
        self.item_alpha = conf.item_alpha

        self.device = conf.device
        self.conf = conf
        self.optimizing_metric = OPTIMIZING_METRIC
        self.max_patience = MAX_PATIENCE

        self.user_factors = user_factors
        self.item_factors = item_factors

        self.exp = datetime.now().strftime("%m%d%Y.%H%M%S")
        if conf.checkpoint_dir:
            if not load_only:
                (Path(conf.checkpoint_dir) / self.exp).mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = str(Path(conf.checkpoint_dir) / self.exp) if conf.checkpoint_dir else "."

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

        print(
            f'Built Trainer module \n'
            f'- n_epochs: {self.n_epochs} \n'
            f'- loss_func_name: {self.loss_func_name} \n'
            f'- loss_func_aggr: {self.loss_func_aggr} \n'
            f'- device: {self.device} \n'
            f'- optimizing_metric: {self.optimizing_metric} \n',
            f'- checkpoint_dir: {self.checkpoint_dir} \n',
            f'- user_alpha: {self.user_alpha} \n',
            f'- item_alpha: {self.item_alpha} \n',
        )

    def _build_model(self):
        # Step 1 --- Building User and Item Feature Extractors
        n_users = self.train_loader.dataset.n_users
        n_items = self.train_loader.dataset.n_items
        user_feature_extractor, item_feature_extractor = \
            FeatureExtractorFactory.create_models(self.ft_ext_param, n_users, n_items, self.user_factors, self.item_factors)
        # Step 2 --- Building RecSys Module
        rec_sys = RecSys(n_users, n_items, self.rec_sys_param, user_feature_extractor, item_feature_extractor,
                         self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()
        rec_sys = nn.DataParallel(rec_sys)
        rec_sys = rec_sys.to(self.device)

        return rec_sys

    def _build_optimizer(self):
        self.lr = self.optim_param['lr'] if 'lr' in self.optim_param else 1e-3
        self.wd = self.optim_param['wd'] if 'wd' in self.optim_param else 1e-4

        optim_name = self.optim_param['optim']
        if optim_name == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optim_name == 'adagrad':
            optim = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError('Optimizer not yet included')

        print(f'Built Optimizer  \n'
              f'- name: {optim_name} \n'
              f'- lr: {self.lr} \n'
              f'- wd: {self.wd} \n')

        return optim

    def run(self, trial=None):
        """
        Runs the Training procedure
        """
        metrics_values = self.val()
        best_value = metrics_values[self.optimizing_metric]
        train.report(metrics_values)
        print('Init - Avg Val Value {:.3f} \n'.format(best_value))

        patience = 0
        for epoch in range(self.n_epochs):

            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break

            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in track(self.train_loader, total=len(self.train_loader)):
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(u_idxs, i_idxs)

                user_protos = get_module_by_name(self.model, 'module.user_feature_extractor.model_1.prototypes')
                normalized_user_protos = F.normalize(user_protos, p=2, dim=1)
                user_dist = normalized_user_protos @ normalized_user_protos.T
                user_dist = user_dist.abs().mean()
                #user_dist = (user_dist ** 2).sum() * self.user_alpha

                item_protos = get_module_by_name(self.model, 'module.item_feature_extractor.model_1.prototypes') #self.model['item_feature_extractor.model_1.prototypes']\
                normalized_items_protos = F.normalize(item_protos, p=2, dim=1)
                item_dist = normalized_items_protos @ normalized_items_protos.T #item_protos @ item_protos.T
                item_dist = item_dist.abs().mean()#(item_dist ** 2).sum() * self.item_alpha

                # print(self.model.module.loss_func(out, labels), user_dist, item_dist)
                loss = self.model.module.loss_func(out, labels) + user_dist + item_dist

                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_train_loss /= len(self.train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))

            metrics_values = self.val()
            curr_value = metrics_values[self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value))
            train.report({**metrics_values, 'epoch_train_loss': epoch_train_loss})

            if curr_value > best_value:
                best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                # with tune.checkpoint_dir(0) as checkpoint_dir:
                ckpt_name = f'best_model_{epoch + 1}_{best_value}.pth'
                print("Saving checkpoint", ckpt_name)
                torch.save(
                    {
                        "state_dict": self.model.module.state_dict(),
                        "config": self.conf,
                        "checkpoint_dir": self.checkpoint_dir
                    }, os.path.join(self.checkpoint_dir, ckpt_name)
                )
                patience = 0
            else:
                patience += 1

            if trial:
                trial.report(best_value, epoch)
                print("Should prune:", trial.should_prune())
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
        return best_value

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: A scalar float value, output of the validation (e.g. NDCG@10).
        """
        self.model.eval()
        print('Validation started')
        val_loss = 0
        eval = Evaluator(self.val_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.val_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            val_loss += self.model.module.loss_func(out, labels).item()

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out)

        val_loss /= len(self.val_loader)
        metrics_values = {**eval.get_results(), 'val_loss': val_loss}

        return metrics_values
