import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import torch.distributed as dist
class LoRALinear(nn.Module):

    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.out_features = linear_layer.out_features
        # 冻结原始线性层参数
        for param in self.linear.parameters():
            param.requires_grad = False
        
        self.rank = rank
        self.alpha = alpha

        # LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))

        self.scaling = alpha / rank

        
        # 训练/评估模式标志
        self.merged_weights = False
    
    def forward(self, x):
        # 处理训练/评估模式切换
        if self.merged_weights and self.training:
            # 如果从评估切换到训练模式，则恢复原始权重
            self._unmerge_weights()
            self.merged_weights = False
        
        if not self.training:
            # 评估模式
            if not self.merged_weights:
                # 如果权重尚未合并，则合并权重以提升推理效率
                self._merge_weights()
                self.merged_weights = True
            # 使用合并后的权重进行推理
            return self.linear(x)
        else:
            # 获取原始权重
            original_output = self.linear(x)
            lora_A =self.lora_A
            lora_B = self.lora_B
            if len(x.shape) == 3:
                # 处理3D输入 (batch_size, seq_len, in_features)
                batch_size, seq_len, _ = x.shape
                lora_output = torch.matmul(torch.matmul(x, lora_A), lora_B) * self.scaling
            else:
                # 处理2D输入 (batch_size, in_features)
                lora_output = (x @ lora_A @ lora_B) * self.scaling
            return original_output + lora_output
    
    def _merge_weights(self):
        with torch.no_grad():
            original_weight = self.linear.weight.data

            # 选择对应类别的LoRA适配器
            lora_A = self.lora_A
            lora_B = self.lora_B

            lora_weight = (lora_A @ lora_B) * self.scaling

            # 合并权重
            self.linear.weight.data = original_weight + lora_weight
    
    def _unmerge_weights(self):
        """恢复原始权重"""
        if hasattr(self, 'original_weight'):
            self.linear.weight.data = self.original_weight
        else:
            # 首次调用时保存原始权重
            self.original_weight = self.linear.weight.data.clone()



class LoRAConv2d(nn.Module):
    """带有LoRA适配器的Conv2d层"""

    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        if isinstance(conv_layer, LoRAConv2d):
            # 如果传入的是DoRAConv2d实例，则获取其内部的原始卷积层
            self.conv = conv_layer.conv

        else:
            # 正常情况，传入的是普通卷积层
            self.conv = conv_layer
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.rank = rank
        self.alpha = alpha
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        # 计算展开后的权重维度
        # 卷积权重形状: [out_channels, in_channels, kernel_height, kernel_width]
        # 展开后形状: [out_channels, in_channels * kernel_height * kernel_width]
        if isinstance(self.kernel_size, tuple):
            self.weight_shape = (self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        else:
            self.weight_shape = (self.out_channels, self.in_channels * self.kernel_size * self.kernel_size)

        # 冻结原始卷积参数
        for param in self.conv.parameters():
            param.requires_grad = False


        # 低秩矩阵A和B (二维矩阵形式)
        self.lora_A = nn.Parameter(torch.zeros(self.weight_shape[0], rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.weight_shape[1]))
        
        # 初始化LoRA适配器权重
        nn.init.normal_(self.lora_A, mean=0, std=1 / math.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank

        # 注册原始权重矩阵
        self.register_buffer('W0', self.conv.weight.detach().clone())
        
        # 训练/评估模式标志
        self.merged_weights = False

    def forward(self, x):
        # 处理训练/评估模式切换
        if self.merged_weights and self.training:
            # 如果从评估切换到训练模式，则恢复原始权重
            self._unmerge_weights()
            self.merged_weights = False
        
        if not self.training:
            # 评估模式
            if not self.merged_weights:
                # 如果权重尚未合并，则合并权重以提升推理效率
                self._merge_weights()
                self.merged_weights = True
            # 使用合并后的权重进行推理
            return self.conv(x)
        else:
            # 获取原始权重并展开为二维矩阵
            original_weight = self.W0
            lora_A = self.lora_A
            lora_B = self.lora_B
            delta = (lora_A @ lora_B) * self.scaling  # [out_channels, in_channels*kh*kw]
            original_weight_flat = original_weight.reshape(self.out_channels, -1)  # [out_channels, in_channels*kh*kw]
            new_weight_flat = original_weight_flat + delta  # [out_channels, in_channels*kh*kw]
            new_weight = new_weight_flat.reshape(self.out_channels, self.in_channels, *self._get_kernel_size())
            
            # 使用最终权重进行卷积
            return F.conv2d(x, new_weight, bias=self.conv.bias,
                           stride=self.stride, padding=self.padding, dilation=self.conv.dilation)
    
    def _merge_weights(self):
        device = self.W0.device  # 获取当前设备

        with torch.no_grad():  # 禁用梯度跟踪
            # 所有进程预分配最终权重张量
            final_weight = torch.empty_like(self.W0)
            is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
            is_rank0 = (not is_distributed) or (dist.get_rank() == 0)
            if is_rank0:
                original_weight_flat = self.W0.reshape(
                    self.out_channels, -1
                )  # [C_out, C_in*kH*kW]

                # 计算LoRA增量
                delta = (self.lora_A @ self.lora_B) * self.scaling  # [C_out, C_in*kH*kW]
                new_weight_flat = original_weight_flat + delta

                # 重塑回卷积形状
                final_weight = new_weight_flat.reshape_as(self.conv.weight)

                # 显式释放中间变量
                del delta, new_weight_flat

            if is_distributed:
                backend = dist.get_backend()
                if backend == "gloo":
                    final_weight = final_weight.cpu()
                dist.broadcast(final_weight, src=0)
                if backend == "gloo":
                    final_weight = final_weight.to(device)


            # 安全更新权重
            # 保存原始权重（仅首次合并时）
            if not hasattr(self, 'original_weight'):
                self.original_weight = self.conv.weight.detach().clone()

            # 原地更新权重数据（保留计算图链接）
            self.conv.weight.data.copy_(final_weight)

    
    def _unmerge_weights(self):
        """恢复原始权重"""
        if hasattr(self, 'original_weight'):
            self.conv.weight.data = self.original_weight
        else:
            # 首次调用时保存原始权重
            self.original_weight = self.conv.weight.data.clone()
    
    def _get_kernel_size(self):
        """获取卷积核尺寸作为元组"""
        if isinstance(self.kernel_size, tuple):
            return self.kernel_size
        else:
            return (self.kernel_size, self.kernel_size)


def apply_lora_to_conv_block(conv_block, rank=4, alpha=1.0, rank_conv1=None, rank_conv2=None, alpha_conv1=None, alpha_conv2=None):
    """
    支持为不同卷积层设置不同的rank和alpha值
    """

    # 替换第一个卷积层，使用特定参数或默认参数
    conv1_rank = rank_conv1 if rank_conv1 is not None else rank
    conv1_alpha = alpha_conv1 if alpha_conv1 is not None else alpha
    conv_block.conv1 = LoRAConv2d(conv_block.conv1, rank=conv1_rank, alpha=conv1_alpha)
    
    # 替换第二个卷积层，使用特定参数或默认参数
    if hasattr(conv_block, 'conv2'):
        conv2_rank = rank_conv2 if rank_conv2 is not None else rank
        conv2_alpha = alpha_conv2 if alpha_conv2 is not None else alpha
        conv_block.conv2 = LoRAConv2d(conv_block.conv2, rank=conv2_rank, alpha=conv2_alpha)
    
    return conv_block


def apply_lora_to_film(film_module, rank=4, alpha=1.0):

    def _apply_lora_to_module(module):
        # 递归遍历所有子模块
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            else:
                # 递归处理子模块
                _apply_lora_to_module(child)

    # 开始递归处理FiLM模块
    _apply_lora_to_module(film_module)
    return film_module


def apply_lora_to_resunet(model, rank=4, alpha=1.0, **layer_params):

    use_encoder1_lora = layer_params.get('use_encoder1_lora', True)
    if use_encoder1_lora:
        encoder1_rank = layer_params.get('encoder1_rank', rank)
        encoder1_alpha = layer_params.get('encoder1_alpha', alpha)
        model.base.encoder_block1.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block1.conv_block1, 
            encoder1_rank, encoder1_alpha,
            rank_conv1=layer_params.get('encoder1_rank_conv1', encoder1_rank),
            rank_conv2=layer_params.get('encoder1_rank_conv2', encoder1_rank),
            alpha_conv1=layer_params.get('encoder1_alpha_conv1', encoder1_alpha),
            alpha_conv2=layer_params.get('encoder1_alpha_conv2', encoder1_alpha)
        )
    
    # encoder_block2
    use_encoder2_lora = layer_params.get('use_encoder2_lora', True)
    if use_encoder2_lora:
        encoder2_rank = layer_params.get('encoder2_rank', rank)
        encoder2_alpha = layer_params.get('encoder2_alpha', alpha)
        model.base.encoder_block2.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block2.conv_block1, 
            encoder2_rank, encoder2_alpha,
            rank_conv1=layer_params.get('encoder2_rank_conv1', encoder2_rank),
            rank_conv2=layer_params.get('encoder2_rank_conv2', encoder2_rank),
            alpha_conv1=layer_params.get('encoder2_alpha_conv1', encoder2_alpha),
            alpha_conv2=layer_params.get('encoder2_alpha_conv2', encoder2_alpha)
        )
    
    # encoder_block3
    use_encoder3_lora = layer_params.get('use_encoder3_lora', True)
    if use_encoder3_lora:
        encoder3_rank = layer_params.get('encoder3_rank', rank)
        encoder3_alpha = layer_params.get('encoder3_alpha', alpha)
        model.base.encoder_block3.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block3.conv_block1, 
            encoder3_rank, encoder3_alpha,
            rank_conv1=layer_params.get('encoder3_rank_conv1', encoder3_rank),
            rank_conv2=layer_params.get('encoder3_rank_conv2', encoder3_rank),
            alpha_conv1=layer_params.get('encoder3_alpha_conv1', encoder3_alpha),
            alpha_conv2=layer_params.get('encoder3_alpha_conv2', encoder3_alpha)
        )
    
    # encoder_block4
    use_encoder4_lora = layer_params.get('use_encoder4_lora', True)
    if use_encoder4_lora:
        encoder4_rank = layer_params.get('encoder4_rank', rank)
        encoder4_alpha = layer_params.get('encoder4_alpha', alpha)
        model.base.encoder_block4.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block4.conv_block1, 
            encoder4_rank, encoder4_alpha,
            rank_conv1=layer_params.get('encoder4_rank_conv1', encoder4_rank),
            rank_conv2=layer_params.get('encoder4_rank_conv2', encoder4_rank),
            alpha_conv1=layer_params.get('encoder4_alpha_conv1', encoder4_alpha),
            alpha_conv2=layer_params.get('encoder4_alpha_conv2', encoder4_alpha)
        )
    
    # encoder_block5
    use_encoder5_lora = layer_params.get('use_encoder5_lora', True)  # 默认应用
    if use_encoder5_lora:
        encoder5_rank = layer_params.get('encoder5_rank', rank)
        encoder5_alpha = layer_params.get('encoder5_alpha', alpha)
        model.base.encoder_block5.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block5.conv_block1, 
            encoder5_rank, encoder5_alpha,
            rank_conv1=layer_params.get('encoder5_rank_conv1', encoder5_rank),
            rank_conv2=layer_params.get('encoder5_rank_conv2', encoder5_rank),
            alpha_conv1=layer_params.get('encoder5_alpha_conv1', encoder5_alpha),
            alpha_conv2=layer_params.get('encoder5_alpha_conv2', encoder5_alpha)
        )
    
    # encoder_block6
    use_encoder6_lora = layer_params.get('use_encoder6_lora', True)  # 默认应用
    if use_encoder6_lora:
        encoder6_rank = layer_params.get('encoder6_rank', rank)
        encoder6_alpha = layer_params.get('encoder6_alpha', alpha)
        model.base.encoder_block6.conv_block1 = apply_lora_to_conv_block(
            model.base.encoder_block6.conv_block1, 
            encoder6_rank, encoder6_alpha,
            rank_conv1=layer_params.get('encoder6_rank_conv1', encoder6_rank),
            rank_conv2=layer_params.get('encoder6_rank_conv2', encoder6_rank),
            alpha_conv1=layer_params.get('encoder6_alpha_conv1', encoder6_alpha),
            alpha_conv2=layer_params.get('encoder6_alpha_conv2', encoder6_alpha)
        )
    
    # 瓶颈层的lora应用
    use_bottleneck_lora = layer_params.get('use_bottleneck_lora', True)  # 默认应用
    if use_bottleneck_lora:
        bottleneck_rank = layer_params.get('bottleneck_rank', rank)
        bottleneck_alpha = layer_params.get('bottleneck_alpha', alpha)
        model.base.conv_block7a.conv_block1 = apply_lora_to_conv_block(
            model.base.conv_block7a.conv_block1, 
            bottleneck_rank, bottleneck_alpha,
            rank_conv1=layer_params.get('bottleneck_rank_conv1', bottleneck_rank),
            rank_conv2=layer_params.get('bottleneck_rank_conv2', bottleneck_rank),
            alpha_conv1=layer_params.get('bottleneck_alpha_conv1', bottleneck_alpha),
            alpha_conv2=layer_params.get('bottleneck_alpha_conv2', bottleneck_alpha)
        )
    
    # 解码器各层的lora应用
    # decoder_block1
    use_decoder1_lora = layer_params.get('use_decoder1_lora', True)  # 默认应用
    if use_decoder1_lora:
        decoder1_rank = layer_params.get('decoder1_rank', rank)
        decoder1_alpha = layer_params.get('decoder1_alpha', alpha)
        model.base.decoder_block1.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block1.conv_block2, 
            decoder1_rank, decoder1_alpha,
            rank_conv1=layer_params.get('decoder1_rank_conv1', decoder1_rank),
            rank_conv2=layer_params.get('decoder1_rank_conv2', decoder1_rank),
            alpha_conv1=layer_params.get('decoder1_alpha_conv1', decoder1_alpha),
            alpha_conv2=layer_params.get('decoder1_alpha_conv2', decoder1_alpha)
        )
    
    # decoder_block2
    use_decoder2_lora = layer_params.get('use_decoder2_lora', True)  # 默认应用
    if use_decoder2_lora:
        decoder2_rank = layer_params.get('decoder2_rank', rank)
        decoder2_alpha = layer_params.get('decoder2_alpha', alpha)
        model.base.decoder_block2.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block2.conv_block2, 
            decoder2_rank, decoder2_alpha,
            rank_conv1=layer_params.get('decoder2_rank_conv1', decoder2_rank),
            rank_conv2=layer_params.get('decoder2_rank_conv2', decoder2_rank),
            alpha_conv1=layer_params.get('decoder2_alpha_conv1', decoder2_alpha),
            alpha_conv2=layer_params.get('decoder2_alpha_conv2', decoder2_alpha)
        )
    
    # decoder_block3
    use_decoder3_lora = layer_params.get('use_decoder3_lora', True)
    if use_decoder3_lora:
        decoder3_rank = layer_params.get('decoder3_rank', rank)
        decoder3_alpha = layer_params.get('decoder3_alpha', alpha)
        model.base.decoder_block3.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block3.conv_block2, 
            decoder3_rank, decoder3_alpha,
            rank_conv1=layer_params.get('decoder3_rank_conv1', decoder3_rank),
            rank_conv2=layer_params.get('decoder3_rank_conv2', decoder3_rank),
            alpha_conv1=layer_params.get('decoder3_alpha_conv1', decoder3_alpha),
            alpha_conv2=layer_params.get('decoder3_alpha_conv2', decoder3_alpha)
        )
    
    # decoder_block4
    use_decoder4_lora = layer_params.get('use_decoder4_lora', True)
    if use_decoder4_lora:
        decoder4_rank = layer_params.get('decoder4_rank', rank)
        decoder4_alpha = layer_params.get('decoder4_alpha', alpha)
        model.base.decoder_block4.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block4.conv_block2, 
            decoder4_rank, decoder4_alpha,
            rank_conv1=layer_params.get('decoder4_rank_conv1', decoder4_rank),
            rank_conv2=layer_params.get('decoder4_rank_conv2', decoder4_rank),
            alpha_conv1=layer_params.get('decoder4_alpha_conv1', decoder4_alpha),
            alpha_conv2=layer_params.get('decoder4_alpha_conv2', decoder4_alpha)
        )
    
    # decoder_block5
    use_decoder5_lora = layer_params.get('use_decoder5_lora', True)
    if use_decoder5_lora:
        decoder5_rank = layer_params.get('decoder5_rank', rank)
        decoder5_alpha = layer_params.get('decoder5_alpha', alpha)
        model.base.decoder_block5.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block5.conv_block2, 
            decoder5_rank, decoder5_alpha,
            rank_conv1=layer_params.get('decoder5_rank_conv1', decoder5_rank),
            rank_conv2=layer_params.get('decoder5_rank_conv2', decoder5_rank),
            alpha_conv1=layer_params.get('decoder5_alpha_conv1', decoder5_alpha),
            alpha_conv2=layer_params.get('decoder5_alpha_conv2', decoder5_alpha)
        )
    
    # decoder_block6
    use_decoder6_lora = layer_params.get('use_decoder6_lora', True)
    if use_decoder6_lora:
        decoder6_rank = layer_params.get('decoder6_rank', rank)
        decoder6_alpha = layer_params.get('decoder6_alpha', alpha)
        model.base.decoder_block6.conv_block2 = apply_lora_to_conv_block(
            model.base.decoder_block6.conv_block2, 
            decoder6_rank, decoder6_alpha,
            rank_conv1=layer_params.get('decoder6_rank_conv1', decoder6_rank),
            rank_conv2=layer_params.get('decoder6_rank_conv2', decoder6_rank),
            alpha_conv1=layer_params.get('decoder6_alpha_conv1', decoder6_alpha),
            alpha_conv2=layer_params.get('decoder6_alpha_conv2', decoder6_alpha)
        )

    # 应用到FiLM层
    use_film = layer_params.get('use_film', False)
    if use_film:
        film_rank = layer_params.get('film_rank', rank)
        film_alpha = layer_params.get('film_alpha', alpha)
        model.film = apply_lora_to_film(model.film, film_rank, film_alpha)
    
    return model


class LoRAResUNet(nn.Module):
    """
    支持为不同层设置不同的rank和alpha值
    """
    def __init__(self, base_model, rank=4, alpha=1.0, **layer_params):
        super().__init__()
        self.model = apply_lora_to_resunet(base_model, rank, alpha, **layer_params)
        # 获取设备信息
        self.device = next(base_model.parameters()).device
    
    def forward(self, input_dict):
        # 获取输入
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']
        device = next(self.model.parameters()).device
        # 确保输入在正确的设备上
        if mixtures.device != device:
            mixtures = mixtures.to(device)
        if conditions.device != device:
            conditions = conditions.to(device)
        
        # 获取FiLM条件
        film_dict = self.model.film(conditions=conditions)
        
        # 前向传播
        output_dict = self.model.base(mixtures=mixtures, film_dict=film_dict)
        return output_dict
    
 

class LoRAManager:
    """
    支持为不同层设置不同的rank和alpha值
    """
    def __init__(self, base_model, rank=4, alpha=1.0, **layer_params):
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.layer_params = layer_params
        # 获取设备信息
        self.device = next(base_model.parameters()).device

        # 处理layer_params中的特殊参数
        # 确保所有必要的参数都存在，即使在JSON中没有明确指定
        # 这样可以避免模型加载时缺少键的问题
        default_params = {
            'use_film': False,
            'film_rank': 4,
            'film_alpha': 1,
            'use_bottleneck_lora': True,
            'bottleneck_rank': rank,
            'bottleneck_alpha': 16,
            'use_encoder1_lora': True,
            'encoder1_rank': rank,
            'encoder1_alpha': 1,
            'use_encoder2_lora': True,
            'encoder2_rank': rank,
            'encoder2_alpha': 2,
            'use_encoder3_lora': True,
            'encoder3_rank': rank,
            'encoder3_alpha': 4,
            'use_encoder4_lora': True,
            'encoder4_rank': rank,
            'encoder4_alpha': 8,
            'use_encoder5_lora': True,
            'encoder5_rank': rank,
            'encoder5_alpha': 12,
            'use_encoder6_lora': True,
            'encoder6_rank': rank,
            'encoder6_alpha': 16,
            'use_decoder1_lora': True,
            'decoder1_rank': rank,
            'decoder1_alpha': 16,
            'use_decoder2_lora': True,
            'decoder2_rank': rank,
            'decoder2_alpha': 16,
            'use_decoder3_lora': True,
            'decoder3_rank': rank,
            'decoder3_alpha': 12,
            'use_decoder4_lora': True,
            'decoder4_rank': rank,
            'decoder4_alpha': 8,
            'use_decoder5_lora': True,
            'decoder5_rank': rank,
            'decoder5_alpha': 4,
            'use_decoder6_lora': True,
            'decoder6_rank': rank,
            'decoder6_alpha': 2,
        }

        # 更新默认参数
        for key, value in layer_params.items():
            default_params[key] = value

        self.lora_model = LoRAResUNet(base_model, rank, alpha, **default_params)
        self.lora_model = self.lora_model.to(self.device)
    
    def get_lora_model(self):
        return self.lora_model

    def forward(self, input_dict):
        # 确保输入在正确的设备上
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and value.device != self.device:
                input_dict[key] = value.to(self.device)
        return self.lora_model(input_dict)
