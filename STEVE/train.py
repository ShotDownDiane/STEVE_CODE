import os
import traceback
from datetime import datetime
import sys
sys.path.append('/home/zhangwt/StableST')

import warnings

from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import yaml
import argparse
import time
import torch

from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, get_log_dir,
)

from lib.dataloader import get_dataloader
from lib.logger import get_logger, PD_Stats
from lib.utils import dwa
import numpy as np
from models.our_model import StableST


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, lr_scheduler,args, graph2=None,load_state=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.lr_scheduler=lr_scheduler
        self.args = args
        if graph2 != None:
            self.test_graph=graph2
        else:
            self.test_graph=graph

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch, loss_weights):
        self.model.train()
        p=epoch/self.args.epochs*1.0
        total_loss = 0
        total_sep_loss = np.zeros(3)
        lms=[]
        t1=datetime.now()
        for batch_idx, (data, target, time_label,c) in enumerate(self.train_loader):
            

            self.optimizer.zero_grad()
            # input shape: n,l,v,c; graph shape: v,v;
            
            Z, H = self.model(data)  # nvc
            loss, sep_loss,lm = self.model.calculate_loss(Z, H, target, c, time_label, self.scaler, loss_weights,p,True)
            if type(lm) == int:
                lms.append(lm)
            else:
                lms.append(lm.item())
            # t2=datetime.now()
            assert not torch.isnan(loss)

            loss.backward()
            # t3=datetime.now()
            # gradient clipping
            # import pdb
            # pdb.set_trace()
            # for param in self.model.parameters():
            #     print("param=%s, grad=%s" % (param.data, param.grad))
            
            # import pdb
            # pdb.set_trace()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            # t4=datetime.now()
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss
        t5=datetime.now()
        print(f"train_time:{t5-t1}")


        train_epoch_loss = total_loss / self.train_per_epoch
        total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))
        self.logger.info('*******Train Epoch {}: averaged lm : {:.6f}'.format(epoch, np.mean(lms)))
        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()

        total_val_loss = 0
        total_sep_loss = np.zeros(3)
        with torch.no_grad():
            for batch_idx, (data, target,time_label,c) in enumerate(val_dataloader):
                Z, H = self.model(data)
                # c_hat=self.model.predict_con(data)
                loss, sep_loss,lm = self.model.calculate_loss(Z, H, target, c, time_label, self.scaler, loss_weights)
                # loss = self.model.pred_loss(repr1, repr1, target, self.scaler)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                total_sep_loss += sep_loss
        val_loss = total_val_loss / len(val_dataloader)
        total_sep_loss = total_sep_loss /len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f} sep loss : {}'.format(epoch, val_loss, total_sep_loss))
        return val_loss

    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3)  # (1.0, 1.0, 1.0)
        for epoch in range(1, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks

            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights = dwa(loss_tm1, loss_tm2, self.args.temp)
            self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            # train_epoch_loss = self.train_epoch(epoch, loss_weights)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)

            if epoch in [1,16,32,64,128]:
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_dir=os.path.join(self.args.log_dir,'epoch{}.pth'.format(epoch))
                self.logger.info('**************Current {} model saved to {}'.format(epoch,save_dir))
                torch.save(save_dict, save_dir)

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1
            
            self.lr_scheduler.step(val_epoch_loss)

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args.early_stop_patience))
                break


        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                         "Total training time: {:.2f} min\t"
                         "best loss: {:.4f}\t"
                         "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))

        # test
        self.logger.info("load best model from {}".format(self.best_path))
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler,
                                 self.test_graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }

        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        x=[]
        atts=[]
        Cs=[]
        Hs=[]
        start_time=time.time()
        with torch.no_grad():
            for batch_idx, (data, target, c) in enumerate(dataloader):
                # weather
                # if batch_idx!=10:
                #     continue
                x.append(data)
                Z, H  = model(data,graph)
                # c_hat=model.predict_con(data)
                pred_output,att,C_tensor, H = model.predict_test(Z, H)
                pred_output = pred_output.squeeze(1)
                target = target.squeeze(1)
                y_true.append(target)
                y_pred.append(pred_output)
                atts.append(att.cpu().detach())
                Cs.append(C_tensor.cpu().detach())
                Hs.append(H.cpu().detach())
        
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        x=torch.cat(x,dim=0)
        atts=torch.cat(atts,dim=0)
        Cs=torch.cat(Cs,dim=0)
        Hs=torch.cat(Hs,dim=0)

        end_time=time.time()
        print(start_time)
        print(end_time-start_time)
        logger.info(end_time-start_time)

        save_path=os.path.join(args.log_dir,'result.npz')
        np.savez(save_path,y_true=y_true.cpu().numpy(),y_pred=y_pred.cpu().numpy(),x=x.cpu().numpy(),atts=atts.cpu().numpy())
        rep_path=os.path.join(args.log_dir,'representation.npz')
        np.savez(rep_path,C=Cs.cpu().numpy(),H=Hs.cpu().numpy())

        test_results = []
        # inflow
        # mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        # logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        # test_results.append([mae, mape])
        # outflow
        mae, mape = test_metrics(y_pred, y_true)
        logger.info("FLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])

        return np.stack(test_results, axis=0)


def make_one_hot(labels, classes):
    # labels=labels.to('cuda:1')
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def main(args):

    # A,A2 = load_graph(args.graph_file, device=args.device)  # �ڽӾ���
    A = load_graph(args.graph_file, device=args.device)

    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    # current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # log_dir = os.path.join(current_dir, 'experiments', 'NYCBike1', current_time)
    model = StableST(args=args, adj=A, in_channels=args.d_input, embed_size=args.d_model,
                T_dim=args.input_length, output_T_dim=1, output_dim=args.d_output,device=args.device).to(args.device)

    

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000005, eps=1e-08)

    # start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=A,
        graph2=A,
        lr_scheduler=lr_scheduler,
        args=args
    )

    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        A, trainer.logger, trainer.args)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())

    trainer.logger.info("abulation is {}".format(args.ablation))
    trainer.logger.info("bank gradient!")
    trainer.logger.info("gamma {}".format(args.bank_gamma))
    trainer.logger.info("kw {}".format(args.kw))











