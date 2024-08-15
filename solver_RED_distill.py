import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict
from matplotlib import image
from prep import printProgressBar
from networks import RED_CNN,RED_CNN_transformer
from measure import compute_measure,updata_te
# from loss import CrossEntropyLossForSoftTarget
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from tensorboard_logger import configure, log_value
import random

class Solver(object):
    def __init__(self, args, data_loader):
        # data params
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.patch_size = args.patch_size

        # training params
        self.mode = args.mode
        #self.device = torch.device("cuda:1")
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = args.num_epochs
        self.multi_gpu = args.multi_gpu
        self.lr = args.lr
        #self.decay_iters = args.decay_iters

        self.save_path = args.save_path
        self.ckpt_path = args.ckpt_path
        self.print_iters = args.print_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig
        self.logs_dir = args.logs_dir
        self.use_tensorboard = args.use_tensorboard
        self.T = args.T
        self.alpha = args.alpha

        self.model_num = args.model_num
        self.REDCNN = RED_CNN()
        self.Rformer = RED_CNN_transformer()
        self.models = [self.REDCNN, self.Rformer]

        self.optimizer_REDCNN = optim.Adam(self.REDCNN.parameters(), self.lr)
        self.optimizer_Rformer = optim.Adam(self.Rformer.parameters(), self.lr)
        self.optimizers = [self.optimizer_REDCNN,self.optimizer_Rformer]

        self.criterion = nn.MSELoss()
        #self.criterion_soft = CrossEntropyLossForSoftTarget()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')


        if self.use_tensorboard:
            tensorboard_dir = os.path.join(args.save_path, args.logs_dir)
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            # configure(tensorboard_dir)
        # num_cuda = 0
        for i in range(self.model_num):
            model = self.models[i]
            if (self.multi_gpu) and (torch.cuda.device_count() > 1):
                print('Use {} GPUs'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(self.device)
            # learning rate decay
            # scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=60, gamma=self.gamma, last_epoch=-1)
            # self.schedulers.append(scheduler)


    def save_model(self, model, iter_):
        f = os.path.join(self.save_path, 'ckpt','Rformer_{}iter.ckpt'.format(iter_))
        #if not os.path.exists(f):
        #    os.makedirs(f)
        #    print("makedirs:save_model")
        torch.save(model.module.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'ckpt','Rformer_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.Rformer.load_state_dict(state_d)
        else:
            self.Rformer.load_state_dict(torch.load(f))


    #def lr_decay(self):
    #    lr = self.lr * 0.5
    #    for param_group in self.optimizer.param_groups:
    #        param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def save_result(self, pred, fig_name):
        if self.result_fig:
            result_path = os.path.join(self.save_path, 'results')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print('Create path : {}'.format(result_path))
        pred = pred.cpu().numpy()
        path = os.path.join(self.save_path, 'results', 'result_{}.png'.format(fig_name))
        image.imsave(path, pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)

    def save_pred(self, pred, fig_name):
        if self.result_fig:
            result_path = os.path.join(self.save_path, 'results_pred')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print('Create path : {}'.format(result_path))
        pred = pred.cpu().numpy()
        path = os.path.join(self.save_path, 'results_pred', 'result_{}.npy'.format(fig_name))
        np.save(path, pred)

    def save_y(self, y, fig_name):
        if self.result_fig:
            result_path = os.path.join(self.save_path, 'results_y')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print('Create path : {}'.format(result_path))
        y = y.cpu().numpy()
        path = os.path.join(self.save_path, 'results_y', 'result_{}.png'.format(fig_name))
        image.imsave(path, y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)

    def save_x(self, x, fig_name):
        if self.result_fig:
            result_path = os.path.join(self.save_path, 'results_x')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print('Create path : {}'.format(result_path))
        x = x.cpu().numpy()
        path = os.path.join(self.save_path, 'results_x', 'result_{}.png'.format(fig_name))
        image.imsave(path, x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)

    def train(self):
        train_losses = []
        total_iters = 0
        total_epoch = 0
        start_time = time.time()

        self.work_dir = os.path.join(self.save_path, 'work_dir')
        self.trainwriter = SummaryWriter(os.path.join(self.work_dir, 'Train'))

        for epoch in range(1, self.num_epochs):
            for i in range(self.model_num):
                self.models[i].train(True)

            for iter_, (x1, y1,x2,y2) in enumerate(self.data_loader):
                total_iters += 1
                # add 1 channel  二维卷积输入维度必须为4维  此处为5维
                x1 = x1.unsqueeze(0).float().to(self.device)
                y1 = y1.unsqueeze(0).float().to(self.device)
                x2 = x2.unsqueeze(0).float().to(self.device)
                y2 = y2.unsqueeze(0).float().to(self.device)
                if self.patch_size: # patch training  回到4维 根据patch_size规定形状
                    x1 = x1.view(-1, 1, self.patch_size, self.patch_size)
                    y1 = y1.view(-1, 1, self.patch_size, self.patch_size)
                    x2 = x2.view(-1, 1, self.patch_size, self.patch_size)
                    y2 = y2.view(-1, 1, self.patch_size, self.patch_size)
                for i, model in enumerate(self.models):
                    if i == 0:
                        kl_loss = 0
                        model1 = self.models[i]
                        pred1 = model1(x2)
                        ce_loss = self.criterion(pred1, y2)

                        model2 = self.models[1]
                        pred2 = model2(x1)
                        kl_loss += self.criterion_kl(F.log_softmax(pred1 / self.T, dim=1),
                                                     F.softmax(Variable(pred2 / self.T), dim=1)
                                                     )* self.T * self.T
                    else:
                        kl_loss = 0
                        model3 = self.models[i]
                        pred3 = model3(x1)
                        ce_loss = self.criterion(pred3,y1)
                        model4 = self.models[0]
                        pred4 = model4(x2)
                        kl_loss += self.criterion_kl(F.log_softmax(pred3 / self.T, dim=1),
                                                     F.softmax(Variable(pred4 / self.T), dim=1)
                                                     ) * self.T * self.T
                    loss = (1-self.alpha) * ce_loss + kl_loss * self.alpha
                    self.models[0].zero_grad() 
                    self.models[1].zero_grad()
                    self.optimizers[0].zero_grad()
                    self.optimizers[1].zero_grad()

                    loss.backward()  
                    self.optimizers[0].step() 
                    self.optimizers[1].step()
                    train_losses.append(loss.item())

                    #i = epoch-1

                    # print
                    if total_iters % self.print_iters == 0:
                        print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch,
                                                                                                            self.num_epochs, iter_+1,
                                                                                                            len(self.data_loader), loss.item(),
                                                                                                            time.time() - start_time))
                        self.trainwriter.add_scalar('Loss', loss.item(), total_iters)
                        self.trainwriter.close()
                    # learning rate decay
                    #if total_iters % self.decay_iters == 0:
                    #    self.lr_decay()
                    # save model
                    if total_iters % self.save_iters == 0:
                        self.save_model(self.models[i],total_iters)
                        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


    def test(self):
        del self.Rformer
        # load
        self.Rformer = RED_CNN_transformer().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        #ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = np.zeros(343), np.zeros(343), np.zeros(343)
        #pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = np.zeros(343), np.zeros(343), np.zeros(343)
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = np.zeros(211), np.zeros(211), np.zeros(211)
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = np.zeros(211), np.zeros(211), np.zeros(211)  

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                #print(x.size())
                pred = self.Rformer(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg[i] = original_result[0]
                ori_ssim_avg[i] = original_result[1]
                ori_rmse_avg[i] = original_result[2]
                pred_psnr_avg[i] = pred_result[0]
                pred_ssim_avg[i] = pred_result[1]
                pred_rmse_avg[i] = pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)
                    self.save_result(pred, i)
                    #self.save_pred(pred, i)
                    self.save_x(x, i)
                    self.save_y(y, i)
                    self.save_pred(pred, i)
                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print(self.test_iters)
            print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(np.mean(ori_psnr_avg), 
                                                                                            np.mean(ori_ssim_avg), 
                                                                                            np.mean(ori_rmse_avg)))
            print('After learnin\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(np.mean(pred_psnr_avg), 
                                                                                            np.mean(pred_ssim_avg), 
                                                                                            np.mean(pred_rmse_avg)))
            print('Original\nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(np.std(ori_psnr_avg,ddof=1), 
                                                                                            np.std(ori_ssim_avg,ddof=1), 
                                                                                            np.std(ori_rmse_avg,ddof=1)))
            print('After learning\nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(np.std(pred_psnr_avg,ddof=1), 
                                                                                                  np.std(pred_ssim_avg,ddof=1), 
                                                                                                  np.std(pred_rmse_avg,ddof=1)))


