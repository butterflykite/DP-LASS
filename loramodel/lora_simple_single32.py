import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class DoRALinear(nn.Module):
    """
    带有DoRA适配器的Linear层 - 单模型版本
    """
    def __init__(self, linear_layer, rank=8, alpha=1.0):
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
        
        # 初始化幅度参数
        with torch.no_grad():
            # 计算原始权重的行范数（对应每个输出特征）
            magnitude_init = torch.norm(self.linear.weight, p=2, dim=1)  # shape: [out_features]
        
        # 可训练的幅度参数
        self.magnitude = nn.Parameter(magnitude_init.clone())  # 初始化为各行范数
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        
        # 初始化LoRA适配器权重
        nn.init.normal_(self.lora_A, mean=0, std=1 / math.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
        
        # 注册原始权重矩阵（用于方向计算）
        self.register_buffer('W0', self.linear.weight.detach().clone())
        
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
            # 训练模式：计算DoRA权重
            # 获取原始权重
            original_weight = self.W0
     
            # 计算LoRA增量
            delta = (self.lora_A @ self.lora_B) * self.scaling #ΔV=BA
            
            # 计算新的权重向量
            new_weight_v = original_weight + delta.T # V+ΔV
            
            # 计算权重范数（按列归一化）
            weight_norm = torch.linalg.norm(new_weight_v, dim=1, keepdim=True) # ||V+ΔV||_r
            normalized_direction = new_weight_v / weight_norm  # (V+ΔV)/||V+ΔV||_r
            
            magnitude = self.magnitude.view(self.out_features, 1)  # m
            
            final_weight = magnitude * normalized_direction  # m·(V+ΔV)/||V+ΔV||_r
            
            dora_output = F.linear(x, final_weight, self.linear.bias)
            return dora_output
    
    def _merge_weights(self):
        """合并权重以提升推理效率（兼容 DDP）"""
        device = self.linear.weight.device
        with torch.no_grad():
            # 1. 所有进程预分配 final_weight
            final_weight = torch.empty_like(self.linear.weight)

            # 2. 仅主进程计算 final_weight
            if torch.distributed.get_rank() == 0:
                original_weight = self.W0
                delta = (self.lora_A @ self.lora_B) * self.scaling
                new_weight_v = original_weight + delta.T
                weight_norm = torch.norm(new_weight_v, dim=1, keepdim=True)
                normalized_direction = new_weight_v / weight_norm
                final_weight = (self.magnitude.view(self.out_features, 1) * normalized_direction)
                del delta, new_weight_v  # 显式释放内存
            # 3. 设备与后端处理
            backend = torch.distributed.get_backend()
            if backend == "gloo":
                final_weight = final_weight.cpu()

            # 4. 广播张量
            torch.distributed.broadcast(final_weight, src=0)

            # 5. 恢复设备（Gloo 需要）
            if backend == "gloo":
                final_weight = final_weight.to(device)
            if not hasattr(self, 'original_linear_weight'):
                self.original_linear_weight = self.linear.weight.detach().clone()
            # 6. 原地更新
            self.linear.weight.data.copy_(final_weight)
    
    def _unmerge_weights(self):
        """恢复原始权重"""
        if hasattr(self, 'original_weight'):
            self.linear.weight.data = self.original_weight
        else:
            # 首次调用时保存原始权重
            self.original_weight = self.linear.weight.data.clone()


class DoRAConv2d(nn.Module):
    """
    带有LoRA适配器的Conv2d层
    """
    def __init__(self, conv_layer, rank=8, alpha=1.0):
        super().__init__()
        # 检查是否已经是DoRAConv2d实例，避免嵌套问题
        if isinstance(conv_layer, DoRAConv2d):
            # 如果传入的是DoRAConv2d实例，则获取其内部的原始卷积层
            self.conv = conv_layer.conv
            # 直接复用原始DoRAConv2d的W0作为初始权重
            # original_weight = conv_layer.W0
        else:
            # 正常情况，传入的是普通卷积层
            self.conv = conv_layer
            # 获取原始权重
            # original_weight = self.conv.weight
        self.conv = conv_layer
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.rank = rank
        self.alpha = alpha
        # 冻结原始卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 创建单个LoRA适配器
        self.lora_adapter = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            bias=False
        )
        
        # 初始化LoRA适配器权重
        nn.init.zeros_(self.lora_adapter.weight)
        
        # 计算缩放因子
        if isinstance(self.kernel_size, tuple):
            kernel_size_product = self.kernel_size[0] * self.kernel_size[1]
        else:
            kernel_size_product = self.kernel_size * self.kernel_size
            
        self.scaling = self.alpha / (kernel_size_product * self.rank)
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def forward(self, x):
        # 获取输入张量的设备
        device = x.device
        # 训练和推理使用相同的逻辑
        # 原始卷积输出
        original_output = self.conv(x)
        # 确保LoRA适配器在正确的设备上
        if next(self.lora_adapter.parameters()).device != device:
            self.lora_adapter = self.lora_adapter.to(device)
        # LoRA适配器输出
        lora_output = self.lora_adapter(x) * self.scaling
        # 确保所有输出在同一设备上
        original_output = original_output.to(device)
        lora_output = lora_output.to(device)
        # 组合原始输出和LoRA输出
        return original_output + lora_output

    # def forward(self, x):
    #     # 处理训练/评估模式切换
    #     if self.merged_weights and self.training:
    #         # 如果从评估切换到训练模式，则恢复原始权重
    #         self._unmerge_weights()
    #         self.merged_weights = False
        
    #     if not self.training:
    #         # 评估模式
    #         if not self.merged_weights:
    #             # 如果权重尚未合并，则合并权重以提升推理效率
    #             self._merge_weights()
    #             self.merged_weights = True
    #         # 使用合并后的权重进行推理
    #         return self.conv(x)
    #     else:
    #         # 训练模式：使用DoRA方式计算
    #         # 获取原始权重并展开为二维矩阵
    #         original_weight = self.W0
    #         original_weight_flat = original_weight.reshape(self.out_channels, -1)  # [out_channels, in_channels*kh*kw]
            
    #         delta = (self.lora_A @ self.lora_B) * self.scaling  # [out_channels, in_channels*kh*kw]
            
    #         # 计算新的权重向量（在二维空间中）
    #         new_weight_flat = original_weight_flat + delta  # [out_channels, in_channels*kh*kw]
            
    #         # 计算权重范数（按输出通道归一化）
    #         weight_norm = torch.norm(new_weight_flat, p=2, dim=1, keepdim=True)  # [out_channels, 1]
            
    #         # 归一化权重（方向分量）
    #         normalized_weight_flat = new_weight_flat / weight_norm   # [out_channels, in_channels*kh*kw]
            
    #         # 应用幅度调整
    #         magnitude_expanded = self.magnitude.view(self.out_channels, 1)  # [out_channels, 1]
    #         final_weight_flat = normalized_weight_flat * magnitude_expanded  # [out_channels, in_channels*kh*kw]
            
    #         # 将最终权重重塑回卷积权重的形状
    #         final_weight = final_weight_flat.reshape(self.out_channels, self.in_channels, *self._get_kernel_size())
            
    #         # 使用最终权重进行卷积
    #         return F.conv2d(x, final_weight, bias=self.conv.bias, 
    #                        stride=self.stride, padding=self.padding, dilation=self.conv.dilation)
    
    # def _merge_weights(self):
    #     """合并权重以提升推理效率"""
    #     # 使用 W0 的设备，而不是直接访问 self.conv.weight.device
    #     device = self.W0.device  # 获取当前设备

    #     with torch.no_grad():  # 禁用梯度跟踪
    #         # 所有进程预分配最终权重张量
    #         final_weight = torch.empty_like(self.W0)

    #         # 仅主进程执行计算逻辑
    #         if torch.distributed.get_rank() == 0:
    #             # 展平原始权重
    #             original_weight = self.W0
    #             original_weight_flat = self.W0.reshape(
    #                 self.out_channels, -1
    #             )  # [C_out, C_in*kH*kW]

    #             # 计算LoRA增量
    #             delta = (self.lora_A @ self.lora_B) * self.scaling  # [C_out, C_in*kH*kW]
    #             new_weight_flat = original_weight_flat + delta

    #             # 归一化方向
    #             weight_norm = torch.norm(new_weight_flat, dim=1, keepdim=True)
    #             normalized_weight_flat = new_weight_flat / weight_norm

    #             # 应用幅度参数
    #             magnitude_expanded = self.magnitude.view(self.out_channels, -1)
    #             final_weight_flat = normalized_weight_flat * magnitude_expanded

    #             # 重塑回卷积形状
    #             final_weight = final_weight_flat.reshape_as(self.conv.weight)

    #             # 显式释放中间变量
    #             del delta, new_weight_flat, normalized_weight_flat

    #         # 分布式通信
    #         backend = torch.distributed.get_backend()
    #         # Gloo后端需要CPU张量通信
    #         if backend == "gloo":
    #             final_weight = final_weight.cpu()

    #         # 主进程广播最终权重
    #         torch.distributed.broadcast(final_weight, src=0)
    #         # 恢复Gloo后端的设备位置
    #         if backend == "gloo":
    #             final_weight = final_weight.to(device)

    #         # 安全更新权重
    #         # 保存原始权重（仅首次合并时）
    #         if not hasattr(self, 'original_weight'):
    #             self.original_weight = self.conv.weight.detach().clone()

    #         # 原地更新权重数据（保留计算图链接）
    #         self.conv.weight.data.copy_(final_weight)
    
    # def _unmerge_weights(self):
    #     """恢复原始权重"""
    #     if hasattr(self, 'original_weight'):
    #         self.conv.weight.data = self.original_weight
    
    # def _get_kernel_size(self):
    #     """获取卷积核尺寸作为元组"""
    #     if isinstance(self.kernel_size, tuple):
    #         return self.kernel_size
    #     else:
    #         return (self.kernel_size, self.kernel_size)
            
    # def __repr__(self):
    #     return f"DoRAConv2d(in={self.in_channels}, out={self.out_channels}, " \
    #            f"kernel={self.kernel_size}, rank={self.rank})"

def apply_dora_to_conv_block(conv_block, rank=4, alpha=1.0, rank_conv1=None, rank_conv2=None, alpha_conv1=None, alpha_conv2=None):
    """
    将DoRA适配器应用到ConvBlock中的卷积层
    支持为不同卷积层设置不同的rank和alpha值
    """

    # 替换第一个卷积层，使用特定参数或默认参数
    conv1_rank = rank_conv1 if rank_conv1 is not None else rank
    conv1_alpha = alpha_conv1 if alpha_conv1 is not None else alpha
    conv_block.conv1 = DoRAConv2d(conv_block.conv1, rank=conv1_rank, alpha=conv1_alpha)
    
    # 替换第二个卷积层，使用特定参数或默认参数
    if hasattr(conv_block, 'conv2'):
        conv2_rank = rank_conv2 if rank_conv2 is not None else rank
        conv2_alpha = alpha_conv2 if alpha_conv2 is not None else alpha
        conv_block.conv2 = DoRAConv2d(conv_block.conv2, rank=conv2_rank, alpha=conv2_alpha)
    
    return conv_block


def apply_dora_to_film(film_module, rank=4, alpha=1.0):
    """
    将DoRA适配器选择性地应用到FiLM模块中的重要线性层
    只对特定的重要线性层应用DoRA适配器，而不是所有线性层
    """

    # 用于存储重要层的路径
    important_layers = set()

    # 遍历模块结构，找出重要的线性层
    def _identify_important_layers(module, path=''):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Linear):
                # 根据输出特征数量判断层的重要性
                # 输出特征数量较大的层通常更重要
                if child.out_features >= 256:
                    important_layers.add(current_path)
                # 特定模块的线性层（如编码器和解码器的关键部分）
                elif 'encoder_block' in current_path or 'decoder_block' in current_path:
                    if child.out_features >= 256:
                        important_layers.add(current_path)
            else:
                _identify_important_layers(child, current_path)

    # 开始识别重要层
    _identify_important_layers(film_module)

    # 选择性地应用DoRA适配器
    def _apply_dora_selectively(module, path=''):
        for name, child in list(module.named_children()):
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Linear):
                # 只对重要的线性层应用DoRA
                if current_path in important_layers:
                    setattr(module, name, DoRALinear(child, rank=rank, alpha=alpha))
            else:
                # 递归处理子模块
                _apply_dora_selectively(child, current_path)

    # 开始选择性地处理FiLM模块
    _apply_dora_selectively(film_module)
    return film_module

def apply_dora_to_resunet(model, rank=4, alpha=1.0, **layer_params):
    """
    将DoRA适配器应用到ResUNet模型的关键层
    支持为不同层设置不同的rank和alpha值，以及条件应用DoRA
    
    参数:
        model: 原始模型
        rank: 默认的秩值
        alpha: 默认的缩放因子
        layer_params: 各层的特定参数，例如:
            use_encoder1_dora: 是否在encoder_block1应用DoRA
            encoder1_rank: encoder_block1的秩
            encoder1_alpha: encoder_block1的缩放因子
            use_encoder2_dora: 是否在encoder_block2应用DoRA
            encoder2_rank: encoder_block2的秩
            encoder2_alpha: encoder_block2的缩放因子
            ...
            use_film: 是否在FiLM层应用DoRA
            film_rank: FiLM层的秩
            film_alpha: FiLM层的缩放因子
            use_after_conv: 是否在after_conv层应用DoRA
            after_conv_rank: after_conv层的秩
            after_conv_alpha: after_conv层的缩放因子
            use_conv_transpose: 是否在ConvTranspose2d层应用DoRA
            conv_transpose_rank: ConvTranspose2d层的秩
            conv_transpose_alpha: ConvTranspose2d层的缩放因子
    """
    # 编码器各层的DoRA应用
    # encoder_block1
    use_encoder1_dora = layer_params.get('use_encoder1_dora', False)
    if use_encoder1_dora:
        encoder1_rank = layer_params.get('encoder1_rank', rank)
        encoder1_alpha = layer_params.get('encoder1_alpha', alpha)
        model.base.encoder_block1.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block1.conv_block1, 
            encoder1_rank, encoder1_alpha,
            rank_conv1=layer_params.get('encoder1_rank_conv1', encoder1_rank),
            rank_conv2=layer_params.get('encoder1_rank_conv2', encoder1_rank),
            alpha_conv1=layer_params.get('encoder1_alpha_conv1', encoder1_alpha),
            alpha_conv2=layer_params.get('encoder1_alpha_conv2', encoder1_alpha)
        )
    
    # encoder_block2
    use_encoder2_dora = layer_params.get('use_encoder2_dora', False)
    if use_encoder2_dora:
        encoder2_rank = layer_params.get('encoder2_rank', rank)
        encoder2_alpha = layer_params.get('encoder2_alpha', alpha)
        model.base.encoder_block2.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block2.conv_block1, 
            encoder2_rank, encoder2_alpha,
            rank_conv1=layer_params.get('encoder2_rank_conv1', encoder2_rank),
            rank_conv2=layer_params.get('encoder2_rank_conv2', encoder2_rank),
            alpha_conv1=layer_params.get('encoder2_alpha_conv1', encoder2_alpha),
            alpha_conv2=layer_params.get('encoder2_alpha_conv2', encoder2_alpha)
        )
    
    # encoder_block3
    use_encoder3_dora = layer_params.get('use_encoder3_dora', False)
    if use_encoder3_dora:
        encoder3_rank = layer_params.get('encoder3_rank', rank)
        encoder3_alpha = layer_params.get('encoder3_alpha', alpha)
        model.base.encoder_block3.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block3.conv_block1, 
            encoder3_rank, encoder3_alpha,
            rank_conv1=layer_params.get('encoder3_rank_conv1', encoder3_rank),
            rank_conv2=layer_params.get('encoder3_rank_conv2', encoder3_rank),
            alpha_conv1=layer_params.get('encoder3_alpha_conv1', encoder3_alpha),
            alpha_conv2=layer_params.get('encoder3_alpha_conv2', encoder3_alpha)
        )
    
    # encoder_block4
    use_encoder4_dora = layer_params.get('use_encoder4_dora', False)
    if use_encoder4_dora:
        encoder4_rank = layer_params.get('encoder4_rank', rank)
        encoder4_alpha = layer_params.get('encoder4_alpha', alpha)
        model.base.encoder_block4.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block4.conv_block1, 
            encoder4_rank, encoder4_alpha,
            rank_conv1=layer_params.get('encoder4_rank_conv1', encoder4_rank),
            rank_conv2=layer_params.get('encoder4_rank_conv2', encoder4_rank),
            alpha_conv1=layer_params.get('encoder4_alpha_conv1', encoder4_alpha),
            alpha_conv2=layer_params.get('encoder4_alpha_conv2', encoder4_alpha)
        )
    
    # encoder_block5
    use_encoder5_dora = layer_params.get('use_encoder5_dora', True)  # 默认应用
    if use_encoder5_dora:
        encoder5_rank = layer_params.get('encoder5_rank', rank)
        encoder5_alpha = layer_params.get('encoder5_alpha', alpha)
        model.base.encoder_block5.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block5.conv_block1, 
            encoder5_rank, encoder5_alpha,
            rank_conv1=layer_params.get('encoder5_rank_conv1', encoder5_rank),
            rank_conv2=layer_params.get('encoder5_rank_conv2', encoder5_rank),
            alpha_conv1=layer_params.get('encoder5_alpha_conv1', encoder5_alpha),
            alpha_conv2=layer_params.get('encoder5_alpha_conv2', encoder5_alpha)
        )
    
    # encoder_block6
    use_encoder6_dora = layer_params.get('use_encoder6_dora', True)  # 默认应用
    if use_encoder6_dora:
        encoder6_rank = layer_params.get('encoder6_rank', rank)
        encoder6_alpha = layer_params.get('encoder6_alpha', alpha)
        model.base.encoder_block6.conv_block1 = apply_dora_to_conv_block(
            model.base.encoder_block6.conv_block1, 
            encoder6_rank, encoder6_alpha,
            rank_conv1=layer_params.get('encoder6_rank_conv1', encoder6_rank),
            rank_conv2=layer_params.get('encoder6_rank_conv2', encoder6_rank),
            alpha_conv1=layer_params.get('encoder6_alpha_conv1', encoder6_alpha),
            alpha_conv2=layer_params.get('encoder6_alpha_conv2', encoder6_alpha)
        )
    
    # 瓶颈层的DoRA应用
    use_bottleneck_dora = layer_params.get('use_bottleneck_dora', True)  # 默认应用
    if use_bottleneck_dora:
        bottleneck_rank = layer_params.get('bottleneck_rank', rank)
        bottleneck_alpha = layer_params.get('bottleneck_alpha', alpha)
        model.base.conv_block7a.conv_block1 = apply_dora_to_conv_block(
            model.base.conv_block7a.conv_block1, 
            bottleneck_rank, bottleneck_alpha,
            rank_conv1=layer_params.get('bottleneck_rank_conv1', bottleneck_rank),
            rank_conv2=layer_params.get('bottleneck_rank_conv2', bottleneck_rank),
            alpha_conv1=layer_params.get('bottleneck_alpha_conv1', bottleneck_alpha),
            alpha_conv2=layer_params.get('bottleneck_alpha_conv2', bottleneck_alpha)
        )
    
    # 解码器各层的DoRA应用
    # decoder_block1
    use_decoder1_dora = layer_params.get('use_decoder1_dora', True)  # 默认应用
    if use_decoder1_dora:
        decoder1_rank = layer_params.get('decoder1_rank', rank)
        decoder1_alpha = layer_params.get('decoder1_alpha', alpha)
        model.base.decoder_block1.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block1.conv_block2, 
            decoder1_rank, decoder1_alpha,
            rank_conv1=layer_params.get('decoder1_rank_conv1', decoder1_rank),
            rank_conv2=layer_params.get('decoder1_rank_conv2', decoder1_rank),
            alpha_conv1=layer_params.get('decoder1_alpha_conv1', decoder1_alpha),
            alpha_conv2=layer_params.get('decoder1_alpha_conv2', decoder1_alpha)
        )
    
    # decoder_block2
    use_decoder2_dora = layer_params.get('use_decoder2_dora', True)  # 默认应用
    if use_decoder2_dora:
        decoder2_rank = layer_params.get('decoder2_rank', rank)
        decoder2_alpha = layer_params.get('decoder2_alpha', alpha)
        model.base.decoder_block2.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block2.conv_block2, 
            decoder2_rank, decoder2_alpha,
            rank_conv1=layer_params.get('decoder2_rank_conv1', decoder2_rank),
            rank_conv2=layer_params.get('decoder2_rank_conv2', decoder2_rank),
            alpha_conv1=layer_params.get('decoder2_alpha_conv1', decoder2_alpha),
            alpha_conv2=layer_params.get('decoder2_alpha_conv2', decoder2_alpha)
        )
    
    # decoder_block3
    use_decoder3_dora = layer_params.get('use_decoder3_dora', False)
    if use_decoder3_dora:
        decoder3_rank = layer_params.get('decoder3_rank', rank)
        decoder3_alpha = layer_params.get('decoder3_alpha', alpha)
        model.base.decoder_block3.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block3.conv_block2, 
            decoder3_rank, decoder3_alpha,
            rank_conv1=layer_params.get('decoder3_rank_conv1', decoder3_rank),
            rank_conv2=layer_params.get('decoder3_rank_conv2', decoder3_rank),
            alpha_conv1=layer_params.get('decoder3_alpha_conv1', decoder3_alpha),
            alpha_conv2=layer_params.get('decoder3_alpha_conv2', decoder3_alpha)
        )
    
    # decoder_block4
    use_decoder4_dora = layer_params.get('use_decoder4_dora', False)
    if use_decoder4_dora:
        decoder4_rank = layer_params.get('decoder4_rank', rank)
        decoder4_alpha = layer_params.get('decoder4_alpha', alpha)
        model.base.decoder_block4.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block4.conv_block2, 
            decoder4_rank, decoder4_alpha,
            rank_conv1=layer_params.get('decoder4_rank_conv1', decoder4_rank),
            rank_conv2=layer_params.get('decoder4_rank_conv2', decoder4_rank),
            alpha_conv1=layer_params.get('decoder4_alpha_conv1', decoder4_alpha),
            alpha_conv2=layer_params.get('decoder4_alpha_conv2', decoder4_alpha)
        )
    
    # decoder_block5
    use_decoder5_dora = layer_params.get('use_decoder5_dora', False)
    if use_decoder5_dora:
        decoder5_rank = layer_params.get('decoder5_rank', rank)
        decoder5_alpha = layer_params.get('decoder5_alpha', alpha)
        model.base.decoder_block5.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block5.conv_block2, 
            decoder5_rank, decoder5_alpha,
            rank_conv1=layer_params.get('decoder5_rank_conv1', decoder5_rank),
            rank_conv2=layer_params.get('decoder5_rank_conv2', decoder5_rank),
            alpha_conv1=layer_params.get('decoder5_alpha_conv1', decoder5_alpha),
            alpha_conv2=layer_params.get('decoder5_alpha_conv2', decoder5_alpha)
        )
    
    # decoder_block6
    use_decoder6_dora = layer_params.get('use_decoder6_dora', False)
    if use_decoder6_dora:
        decoder6_rank = layer_params.get('decoder6_rank', rank)
        decoder6_alpha = layer_params.get('decoder6_alpha', alpha)
        model.base.decoder_block6.conv_block2 = apply_dora_to_conv_block(
            model.base.decoder_block6.conv_block2, 
            decoder6_rank, decoder6_alpha,
            rank_conv1=layer_params.get('decoder6_rank_conv1', decoder6_rank),
            rank_conv2=layer_params.get('decoder6_rank_conv2', decoder6_rank),
            alpha_conv1=layer_params.get('decoder6_alpha_conv1', decoder6_alpha),
            alpha_conv2=layer_params.get('decoder6_alpha_conv2', decoder6_alpha)
        )

    # 应用到FiLM层
    use_film = layer_params.get('use_film', True)  # 默认应用
    if use_film:
        film_rank = layer_params.get('film_rank', rank)
        film_alpha = layer_params.get('film_alpha', alpha)
        model.film = apply_dora_to_film(model.film, film_rank, film_alpha)
    
    # 应用到after_conv层
    use_after_conv = layer_params.get('use_after_conv', False)
    if use_after_conv and hasattr(model.base, 'after_conv'):
        after_conv_rank = layer_params.get('after_conv_rank', rank)
        after_conv_alpha = layer_params.get('after_conv_alpha', alpha)
        model.base.after_conv = apply_dora_to_after_conv(
            model.base.after_conv,
            rank=after_conv_rank,
            alpha=after_conv_alpha
        )
    
    # 应用到ConvTranspose2d层
    use_conv_transpose = layer_params.get('use_conv_transpose', False)
    if use_conv_transpose:
        conv_transpose_rank = layer_params.get('conv_transpose_rank', rank)
        conv_transpose_alpha = layer_params.get('conv_transpose_alpha', alpha)
        
        # 遍历模型中的所有模块，查找ConvTranspose2d层
        for name, module in model.named_modules():
            if isinstance(module, nn.ConvTranspose2d):
                # 获取模块的父模块和属性名
                parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent = model
                
                # 导航到父模块
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # 应用DoRA适配器
                setattr(parent, attr_name, apply_dora_to_conv_transpose(
                    module,
                    rank=conv_transpose_rank,
                    alpha=conv_transpose_alpha
                ))
    
    return model


class DoRAResUNet(nn.Module):
    """
    带有DoRA适配器的ResUNet模型 - 单模型版本
    支持为不同层设置不同的rank和alpha值
    """
    def __init__(self, base_model, rank=16, alpha=1.0, **layer_params):
        super().__init__()
        self.model = apply_dora_to_resunet(base_model, rank, alpha, **layer_params)
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
    
 

class DoRAConvTranspose2d(nn.Module):
    """
    带有LoRA适配器的ConvTranspose2d层
    """
    def __init__(self, conv_layer, rank=8, alpha=1.0):
        super().__init__()
        # 检查是否已经是DoRAConvTranspose2d实例，避免嵌套问题
        if isinstance(conv_layer, DoRAConvTranspose2d):
            # 如果传入的是DoRAConvTranspose2d实例，则获取其内部的原始反卷积层
            self.conv = conv_layer.conv
        else:
            # 正常情况，传入的是普通反卷积层
            self.conv = conv_layer
            
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.output_padding = getattr(self.conv, 'output_padding', 0)
        self.rank = rank
        self.alpha = alpha
        
        # 冻结原始反卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 创建LoRA适配器
        self.lora_adapter = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=False
        )
        
        # 初始化LoRA适配器权重
        nn.init.zeros_(self.lora_adapter.weight)
        
        # 计算缩放因子
        if isinstance(self.kernel_size, tuple):
            kernel_size_product = self.kernel_size[0] * self.kernel_size[1]
        else:
            kernel_size_product = self.kernel_size * self.kernel_size
            
        self.scaling = self.alpha / (kernel_size_product * self.rank)
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def forward(self, x):
        # 训练和推理使用相同的逻辑
        # 原始反卷积输出
        original_output = self.conv(x)
        
        # LoRA适配器输出
        lora_output = self.lora_adapter(x) * self.scaling
        
        # 组合原始输出和LoRA输出
        return original_output + lora_output


class DoRAAfterConv(nn.Module):
    """
    带有LoRA适配器的after_conv层
    """
    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        # 检查是否已经是DoRAAfterConv实例，避免嵌套问题
        if isinstance(conv_layer, DoRAAfterConv):
            # 如果传入的是DoRAAfterConv实例，则获取其内部的原始卷积层
            self.conv = conv_layer.conv
        else:
            # 正常情况，传入的是普通卷积层
            self.conv = conv_layer
            
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.rank = rank
        self.alpha = alpha
        
        # 冻结原始卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 创建LoRA适配器
        self.lora_adapter = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        
        # 初始化LoRA适配器权重
        nn.init.zeros_(self.lora_adapter.weight)
        
        # 计算缩放因子
        if isinstance(self.kernel_size, tuple):
            kernel_size_product = self.kernel_size[0] * self.kernel_size[1]
        else:
            kernel_size_product = self.kernel_size * self.kernel_size
            
        self.scaling = self.alpha / (kernel_size_product * self.rank)
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def forward(self, x):
        device = x.device
        # 训练和推理使用相同的逻辑
        # 原始卷积输出
        original_output = self.conv(x)
        # 确保LoRA适配器在正确的设备上
        if next(self.lora_adapter.parameters()).device != device:
            self.lora_adapter = self.lora_adapter.to(device)
        # LoRA适配器输出
        lora_output = self.lora_adapter(x) * self.scaling
        # 确保所有输出在同一设备上
        original_output = original_output.to(device)
        lora_output = lora_output.to(device)
        # 组合原始输出和LoRA输出
        return original_output + lora_output


def apply_dora_to_after_conv(after_conv, rank=8, alpha=1.0):
    """
    将DoRA适配器应用到after_conv层
    
    参数:
        after_conv: 原始after_conv层
        rank: 秩值
        alpha: 缩放因子
    
    返回:
        应用了DoRA适配器的after_conv层
    """
    return DoRAAfterConv(after_conv, rank=rank, alpha=alpha)


def apply_dora_to_conv_transpose(conv_transpose, rank=8, alpha=1.0):
    """
    将DoRA适配器应用到ConvTranspose2d层
    
    参数:
        conv_transpose: 原始ConvTranspose2d层
        rank: 秩值
        alpha: 缩放因子
    
    返回:
        应用了DoRA适配器的ConvTranspose2d层
    """
    return DoRAConvTranspose2d(conv_transpose, rank=rank, alpha=alpha)


class DoRAManager:
    """
    DoRA模型管理器 - 单模型版本
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
            'film_rank':4,
            'film_alpha': 1,
            'use_bottleneck_dora': True,
            'bottleneck_rank': rank,
            'bottleneck_alpha': 16,
            'use_encoder1_dora': True,
            'encoder1_rank': rank,
            'encoder1_alpha': 1,
            'use_encoder2_dora':True,
            'encoder2_rank': rank,
            'encoder2_alpha': 2,
            'use_encoder3_dora': True,
            'encoder3_rank': rank,
            'encoder3_alpha': 4,
            'use_encoder4_dora': True,
            'encoder4_rank': rank,
            'encoder4_alpha': 8,
            'use_encoder5_dora': True,
            'encoder5_rank': rank,
            'encoder5_alpha': 12,
            'use_encoder6_dora':True,
            'encoder6_rank': rank,
            'encoder6_alpha': 16,
            'use_decoder1_dora': True,
            'decoder1_rank': rank,
            'decoder1_alpha': 16,
            'use_decoder2_dora':True,
            'decoder2_rank': rank,
            'decoder2_alpha': 16,
            'use_decoder3_dora':True,
            'decoder3_rank': rank,
            'decoder3_alpha': 12,
            'use_decoder4_dora': True,
            'decoder4_rank': rank,
            'decoder4_alpha': 8,
            'use_decoder5_dora': True,
            'decoder5_rank': rank,
            'decoder5_alpha': 4,
            'use_decoder6_dora': True,
            'decoder6_rank': rank,
            'decoder6_alpha': 2,
            'use_after_conv': False,
            'after_conv_rank':rank,
            'after_conv_alpha': 1,
            'use_conv_transpose':False,
            'conv_transpose_rank': rank,
            'conv_transpose_alpha': 1
        }
        # 更新默认参数
        for key, value in layer_params.items():
            default_params[key] = value

        # 创建DoRA模型
        self.dora_model = DoRAResUNet(base_model, rank, alpha, **default_params)
        # 确保DoRA模型在正确的设备上
        self.dora_model = self.dora_model.to(self.device)
    
    def get_dora_model(self):
        return self.dora_model

    def forward(self, input_dict):
        # 确保输入在正确的设备上
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and value.device != self.device:
                input_dict[key] = value.to(self.device)
        return self.dora_model(input_dict)
