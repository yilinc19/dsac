import wandb
import abc
from tensorboardX import SummaryWriter
import torch
import gtimer as gt
from datetime import datetime
from numpy import random
from rlkit.core import eval_util, logger
from tabulate import tabulate
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.samplers.data_collector import (VecMdpPathCollector,
                                           VecMdpStepCollector)




class VecOnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            alg,
            exploration_env,
            evaluation_env,
            exploration_data_collector: VecMdpStepCollector,
            evaluation_data_collector: VecMdpPathCollector,
            replay_buffer: TorchReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_paths_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            s
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.alg = alg
        self.writer = SummaryWriter('temp/')
        self.count = 0
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_paths_per_epoch = num_eval_paths_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.f= open("result.txt","w+")
        self.result_table = [["Trial", "Reward", "CVar Min"]]
        self.g= open("train.txt","w+")
        self.train_table = [["Epoch", "Reward", "CVar Min"]]
        self.p= open("percent.txt","w+")
        self.percent_table = []
        for i in range (100):
          i += 1
          self.percent_table.append([i * 0.01])
        self.risk = 0
        self.train_bool = False

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

        if self.alg == 0:
            wandb.init(
                project="Hopper New",
                config={
                "epochs": 1000,
                }
            )
        else:
            wandb.init(
                project="Hopper Old",
                config={
                "epochs": 1000,
                }
            )

    def _train(self):
        self.train_bool = True
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training // self.expl_env.env_num,
                discard_incomplete_paths=False,
                random=True,  # whether random sample from action_space
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=False)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        num_trains_per_expl_step *= self.expl_env.env_num
        numm = 0
        train_data = self.replay_buffer.next_batch(self.batch_size)
        for epoch in gt.timed_for(
                range(self._start_epoch, 1000),
                save_itrs=True,
        ):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("Current Trial: ", numm)
            numm += 1
            #self.trainer.clear_history()
            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop // self.expl_env.env_num):
                    new_expl_steps = self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_paths(new_expl_steps)
                    gt.stamp('data storing', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        train_data = self.replay_buffer.next_batch(self.batch_size)
                        gt.stamp('data sampling', unique=False)
                    #self.trainer.clear_history()
                    self.training_mode(False)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_paths_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)
            self.count += 1
            self._end_epoch(epoch)
        self.g.write(tabulate(self.train_table))
        self.g.close()
        self.train_bool = False

    def test(self):
        self.count = 0
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training // self.expl_env.env_num,
                discard_incomplete_paths=False,
                random=True,  # whether random sample from action_space
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=False)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        num_trains_per_expl_step *= self.expl_env.env_num
        train_data = self.replay_buffer.next_batch(self.batch_size)
        for epoch in gt.timed_for(
                range(self._start_epoch, 500),
                save_itrs=True,
        ):
            noise = random.normal(loc=0, scale=self.risk * 0.001, size=(self.expl_data_collector.get_size()))
            self.expl_data_collector.set_count(noise)
            self.eval_data_collector.set_count(noise)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop // self.expl_env.env_num):
                    new_expl_steps = self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_paths(new_expl_steps)
                    gt.stamp('data storing', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        #self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        train_data = self.replay_buffer.next_batch(self.batch_size)
                        gt.stamp('data sampling', unique=False)
                    #self.trainer.clear_history()
                    self.training_mode(False)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_paths_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)
            self.count += 1
            self._end_epoch(epoch)
            self.risk += 1
        self.f.write(tabulate(self.result_table))
        self.f.close()
        self.p.write(tabulate(self.percent_table))
        self.p.close()
        self.writer.close()

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def combine_list(self, ls1, ls2):
        for i in range(len(ls1)):
          ls1[i] = ls1[i] + ls2[i]

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )
        def flatten(l):
          return [item for sublist in l for item in sublist]
        returns = []
        for path in eval_paths:
          returns += flatten(path["rewards"])
        returns.sort()
        print (sum(returns))
        num = int(0.05 * len(returns))
        minReturns = int(sum(returns[:num]))
        if self.train_bool == True:
            self.train_table.append([self.count, eval_util.get_generic_path_information(eval_paths)['Average Returns'], minReturns])  
            self.writer.add_scalar('Train Average Returns', eval_util.get_generic_path_information(eval_paths)['Average Returns'],
                               self.count)
            self.writer.add_scalar('Train Returns Min CVar', minReturns, self.count)
            wandb.log({"Train Average Returns": eval_util.get_generic_path_information(eval_paths)['Average Returns'], "Train Returns Min CVar": minReturns})
        else:
            self.result_table.append([self.count, eval_util.get_generic_path_information(eval_paths)['Average Returns'], minReturns]) 
            if self.risk % 5 == 0:
                ls2 = []
                for i in range(100):
                    i += 1
                    nn =  int(i * 0.01 * len(returns))
                    minR = int(sum(returns[:nn]))
                    ls2.append([minR])
                    wandb.log({f"{self.risk}_Percentage": minR})
                self.combine_list(self.percent_table, ls2)
            self.writer.add_scalar('Test Average Returns', eval_util.get_generic_path_information(eval_paths)['Average Returns'],
                               self.count)
            self.writer.add_scalar('Test Returns Min CVar', minReturns, self.count)      
            wandb.log({"Test Average Returns": eval_util.get_generic_path_information(eval_paths)['Average Returns'], "Test Returns Min CVar": minReturns})
        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
