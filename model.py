import torch
import torch.nn as nn
import fairseq
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conformer00 import *
from conformer import ConformerBlock,BiMambaEncoder
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
from torch.autograd import Variable

class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    for layer in self.encoder_blocks:
            x = layer(x) #[bs,1+tiempo,emb_size]
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, embedding


class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = '/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/pre-model/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        # print("input_tmp1",input_tmp.shape)
        # if input_tmp.shape[-1] < 10:
        #     input_tmp = F.pad(input_tmp, (0, 10 - input_tmp.shape[-1]), "constant", 0)
        # print("input_tmp2",input_tmp.shape)
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb
    
class SSLModel24(nn.Module):  # W2V
    def __init__(self, device):
        super(SSLModel24, self).__init__()
        cp_path = './pre-model/xlsr2_300m.pt'  # 修改为预训练模型的路径
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)  # 确保模型加载到指定设备
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # 确保模型在正确的设备上
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train() 

        # 将输入传递给特征提取器以获得适当的形状
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # 使用模型的特征提取器
        features = self.model.feature_extractor(input_tmp)
        features = features.transpose(1, 2)  # 转置以符合编码器输入形状 [batch, seq_len, feature_dim]
        
        # 如果有 post_extract_proj，则使用它将特征投影到 1024 维
        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        # 输出特征
        outputs = []
        for i, layer in enumerate(self.model.encoder.layers):
            features = layer(features)[0]  # 处理每个 transformer 层
            if (i + 1) % 2 == 0:  # 每隔 4 个层抽取一次特征
                layer_output = layer.final_layer_norm(features)  # 提取 final_layer_norm 后的特征
                # print(f"Layer {i} final_layer_norm output shape: {layer_output.shape}")  # 打印输出形状
                outputs.append(layer_output)

        return torch.stack(outputs, dim=1)  # 将每层的特征堆叠到一起


class Fake_Mamba(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('Fake_Mamba')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = PN_BiMambas(dim=144, depth=7)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension


        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #------------------------------Fix len-------------------------------------------
        # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # # 提取 Encoder 输出的最后一个时间步特征（或其他特征形式）
        # embedding = x[:, 0, :]  # 选取时间步 0 的特征，形状 (B, 144)
        

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)



class XLSR_Transformer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Transformer')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = Transformer(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #------------------------------Fix len-------------------------------------------
        # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # # 提取 Encoder 输出的最后一个时间步特征（或其他特征形式）
        # embedding = x[:, 0, :]  # 选取时间步 0 的特征，形状 (B, 144)
        

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)






#-------------------------------------

import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq


class SSLModel00(nn.Module):
    def __init__(self,device):
        super(SSLModel00, self).__init__()
        
        cp_path = './pre-model/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
            layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
            
        
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:

        layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature



class XLSR_Conformer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('XLSR_Conformer')
        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        #dim_head=int(emb_size/heads)=144/4=36
        self.encoder_blocks=_get_clones( ConformerBlock( dim = 144, dim_head=36, heads= 4, 
        ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 16), 12)#BOLCK数量

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        for layer in self.encoder_blocks:
            x = layer(x) #[bs,tiempo,emb_size]


        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)
   
class XLSR_AttW_Conformer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('XLSR_Conformer')
        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        #dim_head=int(emb_size/heads)=144/4=36
        self.encoder_blocks=_get_clones(ConformerBlock( dim = 144, dim_head=36, heads= 4, 
        ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 16), 4)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)
        # print("x",x.shape)#torch.Size([20, 199, 144])

        for layer in self.encoder_blocks:
            x = layer(x) #[bs,tiempo,emb_size]


        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)
   

class ML_XLSR_Conformer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('ML_XLSR_Conformer')
        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        heads=args.heads, kernel_size=args.kernel_size)

    def forward(self, x, return_embedding=False):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        # x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        # x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        # x = self.first_bn(x)
        # x = self.selu(x)
        # x = x.squeeze(dim=1)

        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)
        # print("x",x.shape)#torch.Size([20, 199, 144])

        
        out, emb =self.conformer(x,self.device)
        if return_embedding:
            return out, emb
        return out




#-------------------------------------

import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq


class SSLModel00(nn.Module):
    def __init__(self,device):
        super(SSLModel00, self).__init__()
        
        cp_path = './pre-model/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
            layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []

    for layer in layerResult:
        # print("layer",layer[0].shape)#layer torch.Size([208, 20, 1024])

        layery = layer[0].transpose(0, 1)#x(201,b,1024) (b,201,1024) 

        # print("layery1",layery.shape)#torch.Size([20, 208, 1024])

        
        layery = layery.transpose(1, 2) #(x,z)  (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


def getAttenF10(layerResult):

    target_layer_index = 9  # 第10层的索引（从0开始）

    for idx, layer in enumerate(layerResult):
        if idx != target_layer_index:
            continue  # 跳过不是第10层的内容

        # print("idx",idx)

        # 转置并进行平均池化操作
        layery = layer[0].transpose(0, 1)  # (201,b,1024) -> (b,201,1024)

    return layery


class ML_MambaModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V(ML) + BiMambas')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        # Encoder (BiMamba)
        self.encoder = BiMambas(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x):
        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        #------------------------------Var len-------------------------------------------
        nUtterances = len(x)
        output = torch.zeros(nUtterances, 2).to(self.device)
        for n, feat in enumerate(x):
            # print("feat",feat.shape)#[1,seq_len]
            
            input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
            # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
            x_ssl_feat, layerResult = self.ssl_model.extract_feat(input_x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
            # print("layerResult",layerResult.shape)

            y0, fullfeature = getAttenF(layerResult)
            # print("y0",y0.shape)#([20, 24, 1024])
            # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
            y0 = self.fc0(y0)
            # print("y01",y0.shape)#([20, 24, 1])

            y0 = self.sig(y0)
            # print("y02",y0.shape)#([20, 24, 1])

            y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
            # print("y03",y0.shape)#[20, 24, 1, 1]
            fullfeature = fullfeature * y0
            fullfeature = torch.sum(fullfeature, 1)

            # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

            # Step 2: Apply linear layer to reduce feature dimension
            x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

            # Step 3: Preprocess features for encoder
            x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
            x = self.first_bn(x)
            x = self.selu(x)
            x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

            # Step 4: Pass through BiMamba encoder
            x = self.encoder(x)  # (B, T, 144)

            # Step 5: Attention pooling along the time dimension
            attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
            x_pooled = torch.matmul(
                attention_weights.transpose(-1, -2), x
            ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

            # Step 6: Classification head
            out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)


        return out
    
class ML_XLSX_BiMamba_FFN_SLS_Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('ML_W2V + BiMambas_FFN_SLS')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(48, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(48, 2)  # Binary classification (real vs fake)

    def forward(self, x):
        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # -------------------------XLSR-SLS--------------------------
        fullfeature = x.unsqueeze(dim=1)
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.squeeze(x, dim=1)  # 删除第二维度的 1
        # print("x3_after_squeeze", x.shape)  # [20, 69, 341]
        # x = torch.flatten(x, 1)
        # # print("x4",x.shape)#[20, 23529]
        # x = self.fc1(x)
        # # print("x5",x.shape)#[20, 1024])
        # x = self.selu(x)
        # # print("x6",x.shape)#[20, 1024])

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        #===========================Var Len=======================================
        # nUtterances = len(x)
        # output = torch.zeros(nUtterances, 2).to(self.device)
        # for n, feat in enumerate(x):
        #     # print("feat",feat.shape)#[1,seq_len]
            
        #     input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
        #     # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        #     x_ssl_feat = self.ssl_model.extract_feat(input_x.squeeze(-1))  # (B, T, 1024)

        #     # Step 2: Apply linear layer to reduce feature dimension
        #     x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        #     # Step 3: Preprocess features for encoder
        #     x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        #     x = self.first_bn(x)
        #     x = self.selu(x)
        #     x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        #     # Step 4: Pass through BiMamba encoder
        #     x = self.encoder(x)  # (B, T, 144)

        #     # Step 5: Attention pooling along the time dimension
        #     attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        #     x_pooled = torch.matmul(
        #         attention_weights.transpose(-1, -2), x
        #     ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        #     # Step 6: Classification head
        #     out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        return out

class ML_XLSX_10_BiMamba_FFN_Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('ML_W2V10 + BiMambas_FFN')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        # # Step 2: Apply linear layer to reduce feature dimension
        # x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # # Step 3: Preprocess features for encoder
        # x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        # x = self.first_bn(x)
        # x = self.selu(x)
        # x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        fullfeature = getAttenF10(layerResult)
        # # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 199, 1024])
        # y0 = self.fc0(y0)
        # # print("y01",y0.shape)#([20, 24, 1])

        # y0 = self.sig(y0)
        # # print("y02",y0.shape)#([20, 24, 1])

        # y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # # print("y03",y0.shape)#[20, 24, 1, 1]
        # fullfeature = fullfeature * y0
        # fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)


class ML_XLSX_BiMamba_FFN_Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('ML_W2V + BiMambas_FFN')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=4)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x, return_embedding=False):
        #------------------------------Fix len-------------------------------------------
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        # # Step 2: Apply linear layer to reduce feature dimension
        # x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # # Step 3: Preprocess features for encoder
        # x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        # x = self.first_bn(x)
        # x = self.selu(x)
        # x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        if return_embedding:
            return self.fc5(x_pooled), x_pooled
            # return self.fc5(x_pooled), embedding
        
        return self.fc5(x_pooled)

        #===========================Var Len=======================================
        # nUtterances = len(x)
        # output = torch.zeros(nUtterances, 2).to(self.device)
        # for n, feat in enumerate(x):
        #     # print("feat",feat.shape)#[1,seq_len]
            
        #     input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
        #     # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        #     x_ssl_feat = self.ssl_model.extract_feat(input_x.squeeze(-1))  # (B, T, 1024)

        #     # Step 2: Apply linear layer to reduce feature dimension
        #     x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        #     # Step 3: Preprocess features for encoder
        #     x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        #     x = self.first_bn(x)
        #     x = self.selu(x)
        #     x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        #     # Step 4: Pass through BiMamba encoder
        #     x = self.encoder(x)  # (B, T, 144)

        #     # Step 5: Attention pooling along the time dimension
        #     attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        #     x_pooled = torch.matmul(
        #         attention_weights.transpose(-1, -2), x
        #     ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        #     # Step 6: Classification head
        #     out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        # return out
    
class XLSX_SLS_Mamba_Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel00(self.device)
        
        self.JusBiMamba=JustBiMamba(num_classes=2,encoder_dim=1024, num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss

        
        # #---------------ASP----------------------
        # self.attention = nn.Sequential(
        #     nn.Conv1d(3072, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(512, 1024, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(2048)
        # self.fc6 = nn.Linear(2048, 1024)
        # self.bn6 = nn.BatchNorm1d(1024)
        # #--------------ASP-=-----------------------
        
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(23529, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        fullfeature = self.JusBiMamba(fullfeature)

        # print("fullfeature2",fullfeature.shape)#torch.Size([20, 208, 1024])


        # -------------------------XLSR-SLS--------------------------
        fullfeature = fullfeature.unsqueeze(dim=1)
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#[20, 23529]
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])

        # #-------------------------ASP--------------------------
        # x = fullfeature.permute(0, 2, 1)#(B, T, C)
        # t = x.size()[-1]#(B, C, T)# ([20, 1024, 208])
        # # print("t:", t)#t: 208
        # # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t))
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # # print("global_x:", global_x.shape)#torch.Size([20, 3072, 208])

        # w = self.attention(global_x)
        # # print("w:", w.shape)#w: torch.Size([20, 1024, 208])
        # # print("x * w",(x * w).shape)#x * w torch.Size([20, 1024, 208])
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # # print("mu:", mu.shape)#mu: torch.Size([20, 1024])
        # # print("sg:", sg.shape)#sg: torch.Size([20, 1024])
        # x = torch.cat((mu,sg),1)
        # # print("x:", x.shape)#x: torch.Size([20, 1024])
        # x = self.bn5(x)##
        # x = self.fc6(x)##
        # x = self.bn6(x)##



        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = x #
        # output = self.logsoftmax(x)#exp51开始删除，之前加了这个
        # print("output",output.shape)#[20, 2])

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMAttention(nn.Module):
    def __init__(self, feature_dim, reduction_ratio=16):
        super(CBAMAttention, self).__init__()
        # 通道注意力的全连接网络
        self.channel_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction_ratio, feature_dim, bias=False)
        )
        # 空间注意力的卷积层
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        # 输入形状: [batch_size, num_layers, seq_length, feature_dim]

        # 通道注意力机制
        channel_attention = self.channel_attention(x)  # [batch_size, num_layers, seq_length, feature_dim]
        x = x * channel_attention

        # 空间注意力机制
        spatial_attention = self.spatial_attention(x)  # [batch_size, num_layers, seq_length, feature_dim]
        x = x * spatial_attention

        return x

    def channel_attention(self, x):
        # 计算通道维度注意力
        # [batch_size, num_layers, seq_length, feature_dim] -> [batch_size, num_layers * seq_length, feature_dim]
        batch_size, num_layers, seq_length, feature_dim = x.size()
        x = x.view(batch_size, num_layers * seq_length, feature_dim)

        # 平均池化和最大池化
        avg_pool = torch.mean(x, dim=1)  # [batch_size, feature_dim]
        max_pool = torch.max(x, dim=1)[0]  # [batch_size, feature_dim]

        # 通过全连接网络
        avg_out = self.channel_fc(avg_pool)  # [batch_size, feature_dim]
        max_out = self.channel_fc(max_pool)  # [batch_size, feature_dim]

        # 合并并生成通道注意力
        channel_attention = torch.sigmoid(avg_out + max_out)  # [batch_size, feature_dim]
        channel_attention = channel_attention.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, feature_dim]

        return channel_attention

    def spatial_attention(self, x):
        # 计算空间维度注意力
        # [batch_size, num_layers, seq_length, feature_dim] -> [batch_size, feature_dim, seq_length, num_layers]
        x = x.permute(0, 3, 2, 1)

        # 平均池化和最大池化
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, seq_length, num_layers]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [batch_size, 1, seq_length, num_layers]

        # 拼接池化结果
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, 2, seq_length, num_layers]

        # 卷积生成空间注意力
        spatial_attention = torch.sigmoid(self.spatial_conv(pooled))  # [batch_size, 1, seq_length, num_layers]

        # 恢复原始维度顺序
        spatial_attention = spatial_attention.permute(0, 3, 2, 1)  # [batch_size, num_layers, seq_length, feature_dim]

        return spatial_attention
    
class XLSX_CBAM_SLS_Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel00(self.device)
        # 实例化 CBAMAttention 模块
        self.cbam_attention = CBAMAttention(feature_dim=1024, reduction_ratio=16)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(23529, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        #-------------------------SE------------------------------
        # y0 = self.fc0(y0)
        # # print("y01",y0.shape)#([20, 24, 1])

        # y0 = self.sig(y0)
        # # print("y02",y0.shape)#([20, 24, 1])

        # y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # # print("y03",y0.shape)#[20, 24, 1, 1]
        # fullfeature = fullfeature * y0
        # fullfeature = torch.sum(fullfeature, 1)
        # # print("fullfeature1",fullfeature.shape)#[20, 208, 1024]
        # ------------------CBAM 注意力机制------------------
        # 使用 CBAM 模块对 fullfeature 进行注意力加权
        fullfeature = self.cbam_attention(fullfeature)  # 加权后的特征 [20, 24, 208, 1024]
    
        # 将多层特征沿层维度融合
        fullfeature = torch.sum(fullfeature, 1)  # [batch_size, seq_length, feature_dim]

        # --------------------------------------------------
        fullfeature = fullfeature.unsqueeze(dim=1)
        
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#[20, 23529]
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])
        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = self.logsoftmax(x)
        # print("output",output.shape)#[20, 2])

        return output

class ML_CBAM_MambaModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel00(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V(ML_CBAM) + BiMambas')

        # 实例化 CBAMAttention 模块
        self.cbam_attention = CBAMAttention(feature_dim=1024, reduction_ratio=16)

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        # Encoder (BiMamba)
        self.encoder = BiMambas(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension

        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, x):
        # # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        # x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        # ------------------CBAM 注意力机制------------------
        # 使用 CBAM 模块对 fullfeature 进行注意力加权
        fullfeature = self.cbam_attention(fullfeature)  # 加权后的特征 [20, 24, 208, 1024]
    
        # 将多层特征沿层维度融合
        fullfeature = torch.sum(fullfeature, 1)  # [batch_size, seq_length, feature_dim]

        # --------------------------------------------------

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(fullfeature)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        out = self.fc5(x_pooled)  # (B, 144) -> (B, 2)

        return out

class XLSX_SLS_att_Mamba_Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel00(self.device)

        # 实例化 CBAMAttention 模块
        self.cbam_attention = CBAMAttention(feature_dim=1024, reduction_ratio=16)
        
        self.JusBiMamba=JustBiMamba(num_classes=2,encoder_dim=1024, num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss

        
        # #---------------ASP----------------------
        # self.attention = nn.Sequential(
        #     nn.Conv1d(3072, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(512, 1024, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(2048)
        # self.fc6 = nn.Linear(2048, 1024)
        # self.bn6 = nn.BatchNorm1d(1024)
        # #--------------ASP-=-----------------------
        
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(23529, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        #------------------SE-------------------------------
        # y0 = self.fc0(y0)
        # # print("y01",y0.shape)#([20, 24, 1])

        # y0 = self.sig(y0)
        # # print("y02",y0.shape)#([20, 24, 1])

        # y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # # print("y03",y0.shape)#[20, 24, 1, 1]
        # fullfeature = fullfeature * y0
        # fullfeature = torch.sum(fullfeature, 1)
        # ------------------CBAM 注意力机制------------------
        # 使用 CBAM 模块对 fullfeature 进行注意力加权
        fullfeature = self.cbam_attention(fullfeature)  # 加权后的特征 [20, 24, 208, 1024]
    
        # 将多层特征沿层维度融合
        fullfeature = torch.sum(fullfeature, 1)  # [batch_size, seq_length, feature_dim]

        # --------------------------------------------------
        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        fullfeature = self.JusBiMamba(fullfeature)

        # print("fullfeature2",fullfeature.shape)#torch.Size([20, 208, 1024])


        # -------------------------XLSR-SLS--------------------------
        fullfeature = fullfeature.unsqueeze(dim=1)
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#[20, 23529]
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])

        # #-------------------------ASP--------------------------
        # x = fullfeature.permute(0, 2, 1)#(B, T, C)
        # t = x.size()[-1]#(B, C, T)# ([20, 1024, 208])
        # # print("t:", t)#t: 208
        # # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t))
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # # print("global_x:", global_x.shape)#torch.Size([20, 3072, 208])

        # w = self.attention(global_x)
        # # print("w:", w.shape)#w: torch.Size([20, 1024, 208])
        # # print("x * w",(x * w).shape)#x * w torch.Size([20, 1024, 208])
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # # print("mu:", mu.shape)#mu: torch.Size([20, 1024])
        # # print("sg:", sg.shape)#sg: torch.Size([20, 1024])
        # x = torch.cat((mu,sg),1)
        # # print("x:", x.shape)#x: torch.Size([20, 1024])
        # x = self.bn5(x)##
        # x = self.fc6(x)##
        # x = self.bn6(x)##



        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = x #
        # output = self.logsoftmax(x)#exp51开始删除，之前加了这个
        # print("output",output.shape)#[20, 2])

        return output

class XLSX_SLS_Dual_Mamba_Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel00(self.device)
        
        self.JusBiMamba=JustBiMamba(num_classes=2,encoder_dim=1024, num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss

        
        # #---------------ASP----------------------
        # self.attention = nn.Sequential(
        #     nn.Conv1d(3072, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(512, 1024, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(2048)
        # self.fc6 = nn.Linear(2048, 1024)
        # self.bn6 = nn.BatchNorm1d(1024)
        # #--------------ASP-=-----------------------
        
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(23529, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        fullfeature = self.JusBiMamba(fullfeature)

        # print("fullfeature2",fullfeature.shape)#torch.Size([20, 208, 1024])


        # -------------------------XLSR-SLS--------------------------
        fullfeature = fullfeature.unsqueeze(dim=1)
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#[20, 23529]
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])

        # #-------------------------ASP--------------------------
        # x = fullfeature.permute(0, 2, 1)#(B, T, C)
        # t = x.size()[-1]#(B, C, T)# ([20, 1024, 208])
        # # print("t:", t)#t: 208
        # # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t))
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # # print("global_x:", global_x.shape)#torch.Size([20, 3072, 208])

        # w = self.attention(global_x)
        # # print("w:", w.shape)#w: torch.Size([20, 1024, 208])
        # # print("x * w",(x * w).shape)#x * w torch.Size([20, 1024, 208])
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # # print("mu:", mu.shape)#mu: torch.Size([20, 1024])
        # # print("sg:", sg.shape)#sg: torch.Size([20, 1024])
        # x = torch.cat((mu,sg),1)
        # # print("x:", x.shape)#x: torch.Size([20, 1024])
        # x = self.bn5(x)##
        # x = self.fc6(x)##
        # x = self.bn6(x)##



        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = x #
        # output = self.logsoftmax(x)#exp51开始删除，之前加了这个
        # print("output",output.shape)#[20, 2])

        return output

class XLSX_SLS_Mamba2_Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel00(self.device)
        
        self.JusBiMamba2=JustBiMamba2(num_classes=2,encoder_dim=1024, num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss

        
        # #---------------ASP----------------------
        # self.attention = nn.Sequential(
        #     nn.Conv1d(3072, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(512, 1024, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(2048)
        # self.fc6 = nn.Linear(2048, 1024)
        # self.bn6 = nn.BatchNorm1d(1024)
        # #--------------ASP-=-----------------------
        
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(23529, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        # print("layerResult",layerResult.shape)
        y0, fullfeature = getAttenF(layerResult)
        # print("y0",y0.shape)#([20, 24, 1024])
        # print("fullfeature",fullfeature.shape)#([20, 24, 208, 1024])
        y0 = self.fc0(y0)
        # print("y01",y0.shape)#([20, 24, 1])

        y0 = self.sig(y0)
        # print("y02",y0.shape)#([20, 24, 1])

        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        # print("y03",y0.shape)#[20, 24, 1, 1]
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)

        # print("fullfeature1",fullfeature.shape)#torch.Size([20, 208, 1024])

        fullfeature = self.JusBiMamba2(fullfeature)

        # print("fullfeature2",fullfeature.shape)#torch.Size([20, 208, 1024])


        #-------------------------XLSR-SLS--------------------------
        fullfeature = fullfeature.unsqueeze(dim=1)
        # print("fullfeature2",fullfeature.shape)#[20, 1, 208, 1024]
        x = self.first_bn(fullfeature)
        # print("x1",x.shape)#[20, 1, 208, 1024])
        x = self.selu(x)
        # print("x2",x.shape)#[20, 1, 208, 1024]
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#[20, 1, 69, 341]
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#[20, 23529]
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])
        #-------------------------XLSR-SLS--------------------------

        # #-------------------------ASP--------------------------
        # x = fullfeature.permute(0, 2, 1)#(B, T, C)
        # t = x.size()[-1]#(B, C, T)# ([20, 1024, 208])
        # # print("t:", t)#t: 208
        # # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t))
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # # print("global_x:", global_x.shape)#torch.Size([20, 3072, 208])

        # w = self.attention(global_x)
        # # print("w:", w.shape)#w: torch.Size([20, 1024, 208])
        # # print("x * w",(x * w).shape)#x * w torch.Size([20, 1024, 208])
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # # print("mu:", mu.shape)#mu: torch.Size([20, 1024])
        # # print("sg:", sg.shape)#sg: torch.Size([20, 1024])
        # x = torch.cat((mu,sg),1)
        # # print("x:", x.shape)#x: torch.Size([20, 1024])
        # x = self.bn5(x)##
        # x = self.fc6(x)##
        # x = self.bn6(x)##
        # #-------------------------ASP--------------------------



        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = x #
        # output = self.logsoftmax(x)#exp51开始删除，之前加了这个
        # print("output",output.shape)#[20, 2])

        return output

class XLSX_Mamba1_Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, 512)
        print('XLSX_Mamba1')
        # self.first_bn = nn.BatchNorm2d(num_features=1)
        # self.selu = nn.SELU(inplace=True)
        # self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        # heads=args.heads, kernel_size=args.kernel_size)
        # self.ConBiMamba_conformer=ConBiMamba(num_classes=512)#sc-aam
        # self.ConBiMamba_conformer=ConBiMamba(num_classes=2,input_dim=143,num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss
        # self.ConBiMamba_conformer=ConBiMamba(num_classes=2,input_dim=143,encoder_dim=144,num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss
        
        # self.layers = _get_clones(ExBimamba(d_model=256), args.num_encoders)
        
        # self.num_layers = 12  # 定义层数
        # self.layers = nn.ModuleList([ExBimamba(d_model=512) for _ in range(self.num_layers)])

        
        #(num_classes=2,input_dim=143,encoder_dim=512,num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)
        
        # self.class_token = nn.Parameter(torch.rand(1, args.emb_size))

        # #---------------ASP----------------------
        # self.attention = nn.Sequential(
        #     nn.Conv1d(1536, 256, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(256, 512, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(1024)
        # self.fc6 = nn.Linear(1024, 512)
        # self.bn6 = nn.BatchNorm1d(512)
        # #--------------ASP-=-----------------------

        # self.fc5 = nn.Linear(512, 2)
        # self.fc5 = nn.Linear(144, 2)

        #---------------CLASSFIER--------------------
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(11730 , 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # self.fc5 = nn.Linear(144, 2)

    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        # print("x",x.shape) #torch.Size([32, 66800])
        # if isinstance(x, list):#val-test加上
        #     x = torch.tensor(x)#val-test加上
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        # print("x1",x_ssl_feat.shape) #torch.Size([32, 208, 1024])
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        # x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        # # print("x2",x.shape) #torch.Size([32, 1, 208, 144])
        # x = self.first_bn(x)
        # # print("x3",x.shape) #torch.Size([32, 1, 208, 144])
        # x = self.selu(x)
        # x = x.squeeze(dim=1)
        # # print("x4",x.shape) #torch.Size([32, 208, 144])
        
        # x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
        # print("x5",x.shape) #torch.Size([32, 209, 144])

        # 逐层处理输入
        # fullf = []
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)  # 每层更新输入
        #     fullf.append(x)
        #     # print(f"Layer {i}: mean={fullf.mean().item()}, std={fullf.std().item()}")
        # fullfeature = torch.cat(fullf, dim=2)

        # for layer in self.layers:
        #     x = layer(x)

        # out, _ =self.ConBiMamba_conformer(x,self.device)#MyConformer_mamba
        # print("x6",x.shape) ## torch.Size([32, 51, 512])
        #----------------------ASP------------------------------
        # x = x.permute(0, 2, 1)#(B, T, C)
        # t = x.size()[-1]#(B, C, T)# torch.Size([32, 512, 51])
        # # print("t:", t)#t: 51
        # # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t))
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # # print("global_x:", global_x.shape)#torch.Size([32, 1536, 51])

        # w = self.attention(global_x)
        # # print("w:", w.shape)#w: torch.Size([32, 512, 51])
        # # print("x * w",(x * w).shape)#x * w torch.Size([32, 512, 51])
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # # print("mu:", mu.shape)#mu: torch.Size([32, 512])
        # # print("sg:", sg.shape)#sg: torch.Size([32, 512])
        # x = torch.cat((mu,sg),1)
        # # print("x:", x.shape)#x: torch.Size([32, 1024])
        # x = self.bn5(x)##
        # x = self.fc6(x)##
        # x = self.bn6(x)##
        # #print("x:", x.shape)
        # embedding=x
        # print("embedding:", embedding.shape)#embedding: torch.Size([32, 512])
        #--------------ASP-------------------------------
        # embedding=x[:,0,:] #[bs, emb_size]
        # # print("emb",embedding.shape)
        # out=self.fc5(embedding) #[bs,2]
        # # print("out",out.shape)
        #--------------CLASSFIER---------------------------
        # print("x:", x.shape)#torch.Size([20, 208, 512])
        x = x.unsqueeze(dim=1)
        # print("x0:", x.shape)#torch.Size([20, 208, 512])
        x = self.first_bn(x)
        # print("x1",x.shape)#torch.Size([20, 208, 512])
        x = self.selu(x)
        # print("x2",x.shape)#torch.Size([20, 208, 512])
        x = F.max_pool2d(x, (3, 3))
        # print("x3",x.shape)#([20, 1, 69, 170])
        x = torch.flatten(x, 1)
        # print("x4",x.shape)#([20, 11730])
        x = self.fc1(x)
        # print("x5",x.shape)#[20, 1024])
        x = self.selu(x)
        # print("x6",x.shape)#[20, 1024])
        x = self.fc3(x)
        # print("x7",x.shape)#[20, 2])
        x = self.selu(x)
        # print("x8",x.shape)#[20, 2])
        output = self.logsoftmax(x)
        # print("output",output.shape)#[20, 2])
        
        return output


class XLSX_Mamba1_Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + XLSX_Mamba1')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        # self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        # heads=args.heads, kernel_size=args.kernel_size)
        # self.ConBiMamba_conformer=ConBiMamba(num_classes=512)#sc-aam
        self.ConBiMamba_conformer=ConBiMamba(num_classes=2,num_encoder_layers = args.num_encoders, num_attention_heads=args.heads, conv_kernel_size=args.kernel_size)#celoss
        # self.ConBiMamba_conformer=MyConformer_mamba(emb_size=args.emb_size, n_encoders=args.num_encoders,
        # heads=args.heads, kernel_size=args.kernel_size)
        # self.fc5 = nn.Linear(512, 2)#EXP30
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        # print("x",x.shape) #torch.Size([32, 66800])
        # if isinstance(x, list):#val-test加上
        #     x = torch.tensor(x)#val-test加上
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        # print("x1",x_ssl_feat.shape) #torch.Size([32, 208, 1024])
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        # print("x2",x.shape) #torch.Size([32, 1, 208, 144])
        x = self.first_bn(x)
        # print("x3",x.shape) #torch.Size([32, 1, 208, 144])
        x = self.selu(x)
        x = x.squeeze(dim=1)
        # print("x4",x.shape) #torch.Size([32, 208, 144])
        # out, _ =self.conformer(x,self.device)
        # print("input_lengths",x.shape[0])
        # out, _ =self.ConBiMamba_conformer(x, input_lengths=x.shape[0])#celoss
        out =self.ConBiMamba_conformer(x, input_lengths=x.shape[0])#no_upsamp
        # out, _ =self.ConBiMamba_conformer(x,self.device)#MyConformer_mamba
        # print("x5",out.shape) #torch.Size([32, 51, 512])

        # out=self.fc5(out) 

        # embedding=out[:,0,:] #[bs, emb_size]#EXP30
        # # print("emb",embedding.shape)
        # out=self.fc5(embedding) #[bs,2]#EXP30
        # # print("out",out.shape)

        return out





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
import os
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from pytorch_model_summary import summary
import math
# from .loss import *
from typing import Union


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):#原始的ASSIST
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out





class XLSR_AASIST(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        

        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, 128)
        print('XLSR_AASIST')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)


        ####
        # create network wav2vec 2.0
        ####

        # self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        # self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.LL = nn.Linear(1024, 128)
        # self.LL = nn.Linear(768, 128)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        # self.arcface = ArcMarginProduct()
        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        # self.ocsoftmax = OCSoftmax().cuda()
    def forward(self, x, label=None,ya = None, yb = None, lam =None, inference=False):
        # Step 1: Extract features using SSL (e.g., Wav2Vec 2.0)
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))  # (B, T, 1024)

        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(x_ssl_feat)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)
    # -------pre-trained Wav2vec model fine tunning ------------------------##
        # x = x.squeeze(dim=1)
        # # print("x1:", x.shape)#x1: torch.Size([256, 768, 199])
        # x = x.transpose(1, 2)
        # # print("x2:", x.shape)#x2: torch.Size([256, 199, 768])

        # x = self.LL(x)
        # # print("x3:", x.shape)#x3: torch.Size([256, 199, 128])

        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        # print("x4:", x.shape)#x4: torch.Size([256, 128, 199])

        x = x.unsqueeze(dim=1)  # add channel
        # print(x.shape)#torch.Size([256, 1, 128, 199])
        x = F.max_pool2d(x, (3, 3))
        # print("x5:", x.shape)#x5: torch.Size([256, 1, 42, 66])

        x = self.first_bn(x)
        # print("x6:", x.shape)#x6: torch.Size([256, 1, 42, 66])

        x = self.selu(x)
        # print("x7:", x.shape)#x7: torch.Size([256, 1, 42, 66])

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        
        # if inference ==True:
        #     loss,prototype = self.ocsoftmax(last_hidden, label)
        #     output = self.out_layer(last_hidden)
        #     return last_hidden, output, loss, prototype
        # loss,prototype = self.ocsoftmax(last_hidden, label)
 

        output = self.out_layer(last_hidden)
        
        
        # return last_hidden, output, loss, prototype
        return output
