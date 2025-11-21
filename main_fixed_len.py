import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_fixed_eval #Fixed
# from data_utils_vl import * #Var
from model import *
from utils import read_metadata, my_collate, reproducibility
import numpy as np
import copy
# import ast
# from src.models.mamba_models import *
#!/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 

Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has tje CM protocol and ASV score.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track

Example:
$: python evaluate.py score.txt ./keys eval
"""

import sys, os.path
import numpy as np
# import pandas

from glob import glob



def save_results(exp_num, epoch, train_loss, train_acc, val_loss, val_acc, 
                 train_max_memory_allocated, train_max_memory_reserved, 
                 val_max_memory_allocated, val_max_memory_reserved,
                 start_train_time, train_end_time, val_end_time):
    """
    保存训练和验证的结果到指定文件，包括损失值、准确率、显存占用和时间信息
    """
    result_path = f'./results/exp{exp_num}/result_1_14.txt'
    
    # 如果文件路径不存在，则创建它
    if not os.path.exists(f'./results/exp{exp_num}'):
        os.makedirs(f'./results/exp{exp_num}')
    
    # 格式化时间戳
    start_train_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_train_time))
    train_end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))
    val_end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(val_end_time))
    
    # 将结果写入文件
    with open(result_path, 'a') as f:
        f.write(
            f'Epoch {epoch}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%, '
            f'Train Max GPU Allocated: {train_max_memory_allocated:.4f} GiB, '
            f'Train Max GPU Reserved: {train_max_memory_reserved:.4f} GiB, '
            f'Val Max GPU Allocated: {val_max_memory_allocated:.4f} GiB, '
            f'Val Max GPU Reserved: {val_max_memory_reserved:.4f} GiB, '
            f'Train Start Time: {start_train_time_str}, '
            f'Train End Time: {train_end_time_str}, '
            f'Validation End Time: {val_end_time_str}\n'
        )
    
    print(f"Results saved to {result_path}")



import torch
import torch.nn as nn

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct = 0
    model.eval()
    
    # 设置损失权重
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i = 0

    # 清理 GPU 内存统计
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            # 将数据移至设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # 模型前向传播
            batch_out = model(batch_x)  # Baseline
            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size
            
            # 计算预测结果
            pred = batch_out.max(1)[1]  # 获取预测值
            correct += pred.eq(batch_y).sum().item()  # 统计预测正确的数量
            
            # 计算当前批次的准确率
            batch_accuracy = (pred == batch_y).sum().item() / batch_size
            
            i += 1

            # 输出损失和准确率
            print("Batch %i of %i - Loss: %.4f, Accuracy: %.4f%% (Memory: %.4f of %.4f GiB reserved)" 
                  % (
                     i,
                     num_batch,
                     batch_loss.item(),
                     batch_accuracy * 100,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
    
    # 计算整个验证集的平均损失和准确率
    val_loss /= num_total
    val_accuracy = 100. * correct / num_total

    # 获取 GPU 内存峰值
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (2 ** 30)  # 转换为 GiB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (2 ** 30)  # 转换为 GiB
    
    # 输出验证集结果
    print('\nVal accuracy: {:.2f}%, Max Memory Allocated: {:.4f} GiB, Max Memory Reserved: {:.4f} GiB'.format(
        val_accuracy, max_memory_allocated, max_memory_reserved
    ))

    return val_loss, val_accuracy, max_memory_allocated, max_memory_reserved



from tqdm import tqdm  # 导入tqdm用于可视化进度条

def produce_fixed_evaluation_file(dataset, model, device, save_path):
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False,collate_fn=collate_fn_pad)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    total_samples = len(dataset)  # 获取总样本数
    
    fname_list = []
    score_list = []

    # 使用tqdm包裹data_loader以显示进度
    with tqdm(total=total_samples, desc="Processing Samples", unit="samples") as pbar:
        for batch_x, utt_id in data_loader:
            # print("batch_x",batch_x.shape)
            fname_list = []
            score_list = []
            batch_x = batch_x.to(device)
            # batch_x = F.interpolate(batch_x.unsqueeze(1), size=26112, mode='linear', align_corners=False).squeeze(1)
            batch_out = model(batch_x)
            # batch_out, _ = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
            
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            # 保存结果到文件
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {}\n'.format(f, cm))
            
            # 更新进度条，累加处理过的样本数
            pbar.update(len(batch_x))  # len(batch_x) 是本次批次的样本数量

    print('Scores saved to {}'.format(save_path))


from tqdm import tqdm  # 用于进度条显示
import time  # 用于计算时间
from torch.utils.data import DataLoader

def produce_val_evaluation_file(dataset, model, device, save_path):
    # 设置数据加载器
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=my_collate)
    # data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    
    model.eval()  # 将模型设置为评估模式
    
    # 清空保存结果的文件
    f = open(save_path, 'w')
    f.close()
    
    # 记录开始时间
    start_time = time.time()
    
    # 总样本数量
    total_samples = len(dataset)

    # print(f"Dataset total samples: {len(dataset)}")

    
    # 设置进度条，显示处理的样本数
    with tqdm(total=total_samples, desc="Processing Samples", unit="samples") as pbar:
        for batch_x, _, utt_id in data_loader:
        # for batch_x, utt_id in data_loader:
            # print(f"Batch size: {len(batch_x)}, Utterance IDs: {utt_id}")
            fname_list = []
            score_list = []  
            
            # 推理模型输出
            batch_out = model(batch_x)
            # print(f"Batch output shape: {batch_out.shape}")

            
            # 取第二列的输出分数并移动到 CPU
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
            
            # 添加文件名和分数到列表
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            # 将结果追加到文件中
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {}\n'.format(f, cm))
            
            # 更新进度条，每处理一个 batch 则更新对应数量的样本
            pbar.update(len(batch_x))
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    print('Scores saved to {}'.format(save_path))
    print(f'Task completed in {total_time:.2f} seconds.')


import torch
import torch.nn as nn
import torch.optim as optim
import sys

def var_train_epoch(train_loader, model, lr, optimizer, device):
    num_total = 0.0
    num_correct = 0.0  # 用于统计正确预测的数量
    total_loss = 0.0  # 用于累加每个批次的损失
    model.train()

    # 设置损失函数
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i = 0

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        # 将数据移至设备
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # 模型前向传播
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        
        # 计算预测结果
        _, predicted = torch.max(batch_out, 1)  # 获取预测值
        num_correct += (predicted == batch_y).sum().item()  # 统计预测正确的数量

        # 优化器操作
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 计算当前批次的准确率
        batch_accuracy = (predicted == batch_y).sum().item() / batch_size
        
        i += 1
        
        # 输出损失和准确率
        print("Batch %i of %i - Loss: %.4f, Accuracy: %.2f%% (Memory: %.2f of %.2f GiB reserved)" 
              % (
                 i,
                 num_batch,
                 batch_loss.item(),
                 batch_accuracy * 100,
                 torch.cuda.max_memory_allocated(device) / (2 ** 30),
                 torch.cuda.max_memory_reserved(device) / (2 ** 30),
                 ),
              end="\r",
              )

    # 计算整个 epoch 的平均损失和准确率
    epoch_loss = total_loss / num_total
    epoch_accuracy = num_correct / num_total * 100
    print("\nEpoch Loss: %.4f, Accuracy: %.2f%%" % (epoch_loss, epoch_accuracy))
    sys.stdout.flush()

    return epoch_loss, epoch_accuracy


import sys
import torch
import torch.nn as nn

def train_epoch(train_loader, model, lr, optimizer, device):
    num_total = 0.0
    num_correct = 0.0  # 用于统计正确预测的数量
    total_loss = 0.0  # 用于累加每个批次的损失
    
    model.train()

    # 设置损失函数
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i = 0

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        # 将数据移至设备
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # 模型前向传播
        batch_out = model(batch_x)
        batch_out = batch_out.view(-1, 2)  # 确保适配交叉熵损失的形状
        batch_loss = criterion(batch_out, batch_y)     
        
        # 计算预测结果
        _, predicted = torch.max(batch_out, 1)  # 获取预测值
        num_correct += (predicted == batch_y).sum().item()  # 统计预测正确的数量

        # 优化器操作
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 计算当前批次的准确率
        batch_accuracy = (predicted == batch_y).sum().item() / batch_size
        
        i += 1
        
        # 输出损失和准确率
        print("Batch %i of %i - Loss: %.4f, Accuracy: %.4f%% (Memory: %.4f of %.4f GiB reserved)" 
              % (
                 i,
                 num_batch,
                 batch_loss.item(),
                 batch_accuracy * 100,
                 torch.cuda.max_memory_allocated(device) / (2 ** 30),
                 torch.cuda.max_memory_reserved(device) / (2 ** 30),
                 ),
              end="\r",
              )

    # 计算整个 epoch 的平均损失和准确率
    epoch_loss = total_loss / num_total
    epoch_accuracy = num_correct / num_total * 100
    
    # 获取最大 GPU 内存占用
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (2 ** 30)  # 转换为 GiB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (2 ** 30)  # 转换为 GiB
    
    # 打印 epoch 总结
    print("\nEpoch Loss: %.4f, Accuracy: %.4f%%, Max Memory Allocated: %.4f GiB, Max Memory Reserved: %.4f GiB" % 
          (epoch_loss, epoch_accuracy, max_memory_allocated, max_memory_reserved))
    sys.stdout.flush()

    # 返回值增加 GPU 内存信息
    return epoch_loss, epoch_accuracy, max_memory_allocated, max_memory_reserved



       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/datasets/cleaned/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--exp_num', type=int, default=200, help='Experiment number for saving results')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    # parser.add_argument('--emb-size', type=int, default=256, metavar='N',
    #                 help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=7, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--train', default='true', type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)



    ##===================================================Rawboost data augmentation ======================================================================#
    #LA
    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    # ##DF
    # parser.add_argument('--algo', type=int, default=3, 
    #                     help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
    #                         5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    exp_tag = 'Exp{}'.format(args.exp_num)
    if args.comment:
        exp_tag = exp_tag + '_{}'.format(args.comment)

    model_save_path = os.path.join('/nlp/nlp-xxuan/models', exp_tag)
    
    print('exp_tag: '+ exp_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    
    #GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    

    model = Fake_Mamba(args,device)

    

    
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
     
    # define train dataloader
    label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of training trials',len(files_id_train))
    
    train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=20, num_workers = 32, shuffle=True, drop_last = True)
    
    del train_set, label_trn
    
    # define validation dataloader
    labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(args,list_IDs = files_id_dev,
		    labels = labels_dev,
		    base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=20, num_workers=32, shuffle=False)
    del dev_set,labels_dev

    import time
    # #################### Training and validation #####################
    # Early Stopping - 停止训练的条件，当验证集损失没有改善超过7个epoch时，停止训练
    early_stopping_patience = 7

    num_epochs = args.num_epochs
    not_improving=0
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')
    if args.train:
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        # while not_improving<args.num_epochs:
        while not_improving < early_stopping_patience:  # 修改为7个epoch没有改进时停止
            start_train_time = time.time()  # 记录训练开始时间
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_train_time))}")
            print('######## Epoch {} ########'.format(epoch))

            train_loss, train_acc, train_max_memory_allocated, train_max_memory_reserved = train_epoch(train_loader, model, args.lr, optimizer, device)
            train_end_time = time.time()  # 记录训练结束时间
            print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))}")

        
            val_loss, val_acc, val_max_memory_allocated, val_max_memory_reserved = evaluate_accuracy(dev_loader, model, device)

            val_end_time = time.time()  # 记录验证结束时间
            print(f"Validation ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(val_end_time))}")

            # 保存训练和验证结果，包括时间信息
            save_results(
                args.exp_num,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                train_max_memory_allocated,
                train_max_memory_reserved,
                val_max_memory_allocated,
                val_max_memory_reserved,
                start_train_time,
                train_end_time,
                val_end_time
            )

            if val_loss<best_loss:
                best_loss=val_loss
                # 将 val_loss 和 val_accuracy 格式化为文件名的一部分

                torch.save(copy.deepcopy(model.state_dict()), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1

            # 打印当前epoch的损失和准确率
            print(f'\nEpoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}%')

            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_loss
                    torch.save(copy.deepcopy(model.state_dict()), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            print('\n{} - {}'.format(epoch, val_loss))
            print('n-best loss:', bests)
            torch.save(copy.deepcopy(model.state_dict()), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

            # 检查是否达到 early stopping 的条件
            if not_improving >= early_stopping_patience:
                print(f'Early stopping applied. No improvement for {early_stopping_patience} consecutive epochs.')
                break

            epoch+=1
            if epoch>74:
                break
        print('Total epochs: ' + str(epoch) +'\n')


    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))



    # #fixed_len
    # eval_tracks=['LA','DF']
    # if args.comment_eval:
    #     exp_tag = exp_tag + '_{}'.format(args.comment_eval)

    # for tracks in eval_tracks:
    #     if not os.path.exists('Scores/{}/{}.txt'.format(tracks, exp_tag)):
    #         prefix      = 'ASVspoof_{}'.format(tracks)
    #         prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
    #         prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

    #         file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)), is_eval=True)
    #         print('no. of eval trials',len(file_eval))
    #         eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)),track=tracks)
    #         produce_fixed_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, exp_tag))
    #     else:
    #         print('Score file already exists')

    #fixed_len   
    
    # eval_tracks = ['LA', 'DF']
    # eval_tracks = ['DF']
    eval_tracks = ['LA']

    if args.comment_eval:
        exp_tag = exp_tag + '_{}'.format(args.comment_eval)

    for tracks in eval_tracks:
        # 在每次循环中确保路径存在
        output_dir = 'Scores/Fixed_Train_Fixed_test/{}/'.format(tracks)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 确定完整的输出文件路径
        output_file = os.path.join(output_dir, '{}.txt'.format(exp_tag))
        
        if not os.path.exists(output_file):
            prefix = 'ASVspoof_{}'.format(tracks)
            prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
            prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

            file_eval = read_metadata(dir_meta=os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_eval=True)
            print('no. of eval trials', len(file_eval))
            
            eval_set = Dataset_fixed_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)), track=tracks)
            produce_fixed_evaluation_file(eval_set, model, device, output_file)
        else:
            print('Score file already exists')




        
