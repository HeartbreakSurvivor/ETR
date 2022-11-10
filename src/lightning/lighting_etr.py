

from sklearn.utils import resample
import torch
import numpy as np
from loguru import logger
import pytorch_lightning as pl

from src.utils.misc import lower_config
from src.losses.msfft_loss import construct_loss
from src.optimizers.optimizer import build_optimizer
from src.optimizers.scheduler import build_scheduler
from src.model.rmt.rmt_matcher import ReMatcher
from src.utils.profiler import PassThroughProfiler
from src.utils.evaluate import Evaluator

class PL_ETR(pl.LightningModule):
    def __init__(self, config, evaluator:Evaluator=None, pretrained_ckpt=None, profiler=None):
        super().__init__() 
        # Misc
        self.config = lower_config(config)  # full config
        self.profiler = profiler or PassThroughProfiler()

        # Module: ReMatcher
        self.rmt = ReMatcher(self.config['rmt'])
        self.lossfunc = construct_loss(self.config['trainer'])

        # # Pretrained weights
        # if pretrained_ckpt:
        #     state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
        #     # print('keys: ', state_dict.keys(), len(state_dict.keys()))
        #     # remove prefix 'msfft_arcface.'
        #     # state_dict = {k[14:]: v for k, v in state_dict.items()}
        #     # self.msfft_arcface.load_state_dict(state_dict, strict=True)
        #     logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Val Dataset
        self.evaluator = evaluator 

    def eval(self):
        if self.evaluator:
            self.evaluator.eval(self.rmt)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.rmt, self.config['trainer'])
        scheduler = build_scheduler(optimizer, self.config['trainer'])
        return [optimizer], [scheduler]

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # logger.info('optimizer step...')
        # learning rate warm up
        trainer_conf = self.config['trainer']
        warmup_step = trainer_conf['warmup_step']
        warmup_type = trainer_conf['warmup_type']

        # manually warm up lr without a scheduler
        if self.trainer.global_step < warmup_step:
            if warmup_type == 'linear':
                base_lr = trainer_conf['warmup_ratio'] * trainer_conf['true_lr']
                lr = base_lr + (self.trainer.global_step / warmup_step) * \
                    abs(trainer_conf['true_lr'] - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif warmup_type == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {warmup_type}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        global_feats, local_feats, local_mask, scales, positions = batch

        # logger.info("trainging step, batch_idx: {}", batch_idx)
        # logger.info("img shape: {}, label: {}", img.shape, label.shape)
        # logger.info('label: ', label)

        with self.profiler.profile("ReMatcher inference"):
            # 正样本
            p_logits = self.rmt(
            global_feats[0::3], local_feats[0::3], local_mask[0::3], scales[0::3], positions[0::3],
            global_feats[1::3], local_feats[1::3], local_mask[1::3], scales[1::3], positions[1::3])

            # 负样本
            n_logits = self.rmt(
                global_feats[0::3], local_feats[0::3], local_mask[0::3], scales[0::3], positions[0::3],
                global_feats[2::3], local_feats[2::3], local_mask[2::3], scales[2::3], positions[2::3])

            logits = torch.cat([p_logits, n_logits], 0)
            bsize = logits.size(0)
            assert (bsize % 2 == 0)
            labels = logits.new_ones(logits.size()).float()
            labels[(bsize//2):] = 0

            loss = self.lossfunc(logits, labels).mean()

            # 计算准确率
            acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        # logging
        if self.trainer.global_rank == 0 and \
            self.global_step % self.trainer.log_every_n_steps == 0:
            logger.info(f"current steps: {self.global_step}, \n current bacth_idx: {batch_idx}, \n current batch loss: {loss}, \n current batch acc: {acc}")
        
        logs = {"train_batch_loss": loss.detach()}
        batch_dict = {
            'loss': loss,
            'acc': acc,
            'log': logs #optional for batch logging purposes 
        }
        return batch_dict

    def training_epoch_end(self, outputs):
        # logger.info(f"training_epoch_end")

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        logger.info(f"current_epoch:{self.current_epoch} \n avg_loss: {avg_loss} \n avg_acc: {avg_acc}")

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('avg_loss/train', avg_loss, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_acc/train', avg_acc, global_step=self.current_epoch)

        self.log("train_avg_loss", avg_loss, sync_dist=True) # ckpt monitors on this
        self.log("train_avg_acc", avg_acc, sync_dist=True) # ckpt monitors on this
        
        # #`training_epoch_end` expects a return of None. HINT: remove the return statement in `training_epoch_end`.
        # epoch_dict = {
        #     'loss': avg_loss,
        #     'log': avg_loss.detach()
        # }
        # return epoch_dict

        if self.trainer.global_rank == 0:
            logger.info("start evaluate on val dataset....")
            res = self.evaluator.eval(self.rmt)[100]

            for k, v in res.items():
                logger.info("====> Recall@{}: {:.4f}", k, v)
                self.logger.experiment.add_scalar(
                    'Recall@' + str(k), v, self.current_epoch)
                self.log(f'Recall@{k}', v)  # ckpt monitors on this

    def on_validation_start(self): 
        # 只在主GPU上运算
        if self.trainer.global_rank == 0:
            logger.info("start evaluate on val dataset....")
            res = self.evaluator.eval(self.rmt)[0]

            for k, v in res.items():
                logger.info("====> Recall@{}: {:.4f}", k, v)
                self.logger.experiment.add_scalar(
                    'Recall@' + str(k), v, self.current_epoch)
                self.log(f'Recall@{k}', v)  # ckpt monitors on this

    def validation_step(self, batch, batch_idx):
        # the pythonlightning has put the model in eval mode and PyTorch gradients have been disabled
        # at the end of validation, the model will go back to training mode, so we don't need set eval mode manually
        ...

    def validation_epoch_end(self, outputs):
        logger.info(f'validation_epoch_end, start to calculate average loss and accuracy')
