import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class ConvAdapterLinear(nn.Module):
    """
    带有Conv-Adapter的Linear层 - 单模型版本
    使用线性层的Conv-Adapter实现
    """
    def __init__(self, linear_layer, gamma=1, alpha=1):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.out_features = linear_layer.out_features
        
        # 冻结原始线性层参数
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # 超参数：通道下采样因子γ和缩放因子α（不可训练）
        self.gamma = gamma
        self.alpha = alpha
        self.mid_features = in_features // gamma
        
        # 非线性激活函数统一为LeakyReLU
        self.activation = nn.LeakyReLU(inplace=False)
        
        # Conv-Adapter组件
        # 1. 输入降维层（通道下采样）
        self.down_projection = nn.Linear(in_features, self.mid_features, bias=False)
        
        # 2. 输出升维层
        self.up_projection = nn.Linear(self.mid_features, out_features, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化Conv-Adapter的权重"""
        nn.init.kaiming_normal_(self.down_projection.weight,nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.up_projection.weight, nonlinearity='leaky_relu')
    
    def forward(self, x):
        # 原始线性层输出
        original_output = self.linear(x)
        
        # Conv-Adapter路径
        # 1. 通道下采样（降维）
        down_projected = self.down_projection(x)
        
        # 2. 非线性激活（LeakyReLU）
        activated = self.activation(down_projected)
        
        # 3. 升维
        adapter_output = self.up_projection(activated)
        
        # 4. 应用超参数缩放因子α
        adapter_output = adapter_output * self.alpha
        
        # 残差连接：原始输出 + Conv-Adapter输出
        return original_output + adapter_output


class ConvAdapter2d(nn.Module):
    """
    带有Conv-Adapter的Conv2d层
    Conv-Adapter包含：
    1. 深度可分离卷积进行通道下采样（下采样因子为γ）
    2. 非线性激活函数（LeakyReLU）
    3. 逐点卷积恢复通道维度
    """
    def __init__(self, conv_layer, gamma=1, alpha=1):
        super().__init__()
        # 检查是否已经是ConvAdapter2d实例，避免嵌套问题
        if isinstance(conv_layer, ConvAdapter2d):
            # 如果传入的是ConvAdapter2d实例，则获取其内部的原始卷积层
            self.conv = conv_layer.conv
        else:
            # 正常情况，传入的是普通卷积层
            self.conv = conv_layer
            
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        
        # 超参数：通道下采样因子γ和缩放因子α（不可训练）
        self.gamma = gamma
        self.alpha = alpha
        
        # 冻结原始卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 计算中间通道数（通道下采样）
        self.mid_channels = self.in_channels // self.gamma
        
        # 1. 深度可分离卷积层 (W_down)
        # 使用与原始卷积相同的卷积核尺寸以保持感受野
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels,  # 深度卷积：每个通道独立卷积
            bias=False
        )
        
        # 2. 通道下采样层（1x1卷积）
        self.channel_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 3. 非线性激活函数统一为LeakyReLU
        self.activation = nn.LeakyReLU(inplace=False)
        
        # 4. 逐点卷积层 (W_up)
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 初始化权重
        self._init_weights()
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def _init_weights(self):
        """初始化Conv-Adapter的权重"""
        # 深度卷积权重初始化
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity='leaky_relu')
        
        # 通道下采样层权重初始化
        nn.init.kaiming_normal_(self.channel_down.weight,  nonlinearity='leaky_relu')
        
        # 逐点卷积权重初始化
        nn.init.kaiming_normal_(self.pointwise_conv.weight, nonlinearity='leaky_relu')
    
    def forward(self, x):
        # 获取输入张量的设备
        device = x.device
        
        # 原始卷积输出
        original_output = self.conv(x)
        
        # Conv-Adapter路径
        # 1. 深度可分离卷积
        depthwise_out = self.depthwise_conv(x)
        
        # 2. 通道下采样
        downsampled = self.channel_down(depthwise_out)
        
        # 3. 非线性激活（LeakyReLU）
        activated = self.activation(downsampled)
        
        # 4. 逐点卷积
        adapter_output = self.pointwise_conv(activated)
        
        # 5. 应用超参数缩放因子α
        adapter_output = adapter_output * self.alpha
        
        # 确保所有输出在同一设备上
        original_output = original_output.to(device)
        adapter_output = adapter_output.to(device)
        
        # 残差连接：原始输出 + Conv-Adapter输出
        return original_output + adapter_output

def apply_conv_adapter_to_conv_block(conv_block, gamma=1, alpha=1, gamma_conv1=None, gamma_conv2=None, alpha_conv1=None, alpha_conv2=None):
    """
    将Conv-Adapter应用到ConvBlock中的卷积层
    支持为不同卷积层设置不同的gamma和alpha值
    """

    # 替换第一个卷积层，使用特定参数或默认参数
    conv1_gamma = gamma_conv1 if gamma_conv1 is not None else gamma
    conv1_alpha = alpha_conv1 if alpha_conv1 is not None else alpha
    conv_block.conv1 = ConvAdapter2d(conv_block.conv1, gamma=conv1_gamma, alpha=conv1_alpha)
    
    # 替换第二个卷积层，使用特定参数或默认参数
    if hasattr(conv_block, 'conv2'):
        conv2_gamma = gamma_conv2 if gamma_conv2 is not None else gamma
        conv2_alpha = alpha_conv2 if alpha_conv2 is not None else alpha
        conv_block.conv2 = ConvAdapter2d(conv_block.conv2, gamma=conv2_gamma, alpha=conv2_alpha)
    
    return conv_block


def apply_conv_adapter_to_film(film_module, gamma=1, alpha=1):
    """
    将Conv-Adapter选择性地应用到FiLM模块中的重要线性层
    只对特定的重要线性层应用Conv-Adapter，而不是所有线性层
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

    # 选择性地应用Conv-Adapter
    def _apply_conv_adapter_selectively(module, path=''):
        for name, child in list(module.named_children()):
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Linear):
                # 只对重要的线性层应用Conv-Adapter
                if current_path in important_layers:
                    setattr(module, name, ConvAdapterLinear(child, gamma=gamma, alpha=alpha))
            else:
                # 递归处理子模块
                _apply_conv_adapter_selectively(child, current_path)

    # 开始选择性地处理FiLM模块
    _apply_conv_adapter_selectively(film_module)
    return film_module

def apply_conv_adapter_to_resunet(model, gamma=1, alpha=1, **layer_params):
    """
    将Conv-Adapter应用到ResUNet模型的关键层
    支持为不同层设置不同的gamma和alpha值，以及条件应用Conv-Adapter
    
    参数:
        model: 原始模型
        gamma: 默认的通道下采样因子
        alpha: 默认的缩放因子
        layer_params: 各层的特定参数，例如:
            use_encoder1_adapter: 是否在encoder_block1应用Conv-Adapter
            encoder1_gamma: encoder_block1的通道下采样因子
            encoder1_alpha: encoder_block1的缩放因子
            use_encoder2_adapter: 是否在encoder_block2应用Conv-Adapter
            encoder2_gamma: encoder_block2的通道下采样因子
            encoder2_alpha: encoder_block2的缩放因子
            ...
            use_film: 是否在FiLM层应用Conv-Adapter
            film_gamma: FiLM层的通道下采样因子
            film_alpha: FiLM层的缩放因子
            use_after_conv: 是否在after_conv层应用Conv-Adapter
            after_conv_gamma: after_conv层的通道下采样因子
            after_conv_alpha: after_conv层的缩放因子
            use_conv_transpose: 是否在ConvTranspose2d层应用Conv-Adapter
            conv_transpose_gamma: ConvTranspose2d层的通道下采样因子
            conv_transpose_alpha: ConvTranspose2d层的缩放因子
    """
    # 编码器各层的Conv-Adapter应用
    # encoder_block1
    use_encoder1_adapter = layer_params.get('use_encoder1_adapter',True)
    if use_encoder1_adapter:
        encoder1_gamma = layer_params.get('encoder1_gamma', gamma)
        encoder1_alpha = layer_params.get('encoder1_alpha', alpha)
        model.base.encoder_block1.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block1.conv_block1, 
            encoder1_gamma, encoder1_alpha,
            gamma_conv1=layer_params.get('encoder1_gamma_conv1', encoder1_gamma),
            gamma_conv2=layer_params.get('encoder1_gamma_conv2', encoder1_gamma),
            alpha_conv1=layer_params.get('encoder1_alpha_conv1', encoder1_alpha),
            alpha_conv2=layer_params.get('encoder1_alpha_conv2', encoder1_alpha)
        )
    
    # encoder_block2
    use_encoder2_adapter = layer_params.get('use_encoder2_adapter',True)
    if use_encoder2_adapter:
        encoder2_gamma = layer_params.get('encoder2_gamma', gamma)
        encoder2_alpha = layer_params.get('encoder2_alpha', alpha)
        model.base.encoder_block2.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block2.conv_block1, 
            encoder2_gamma, encoder2_alpha,
            gamma_conv1=layer_params.get('encoder2_gamma_conv1', encoder2_gamma),
            gamma_conv2=layer_params.get('encoder2_gamma_conv2', encoder2_gamma),
            alpha_conv1=layer_params.get('encoder2_alpha_conv1', encoder2_alpha),
            alpha_conv2=layer_params.get('encoder2_alpha_conv2', encoder2_alpha)
        )
    
    # encoder_block3
    use_encoder3_adapter = layer_params.get('use_encoder3_adapter', True)
    if use_encoder3_adapter:
        encoder3_gamma = layer_params.get('encoder3_gamma', gamma)
        encoder3_alpha = layer_params.get('encoder3_alpha', alpha)
        model.base.encoder_block3.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block3.conv_block1, 
            encoder3_gamma, encoder3_alpha,
            gamma_conv1=layer_params.get('encoder3_gamma_conv1', encoder3_gamma),
            gamma_conv2=layer_params.get('encoder3_gamma_conv2', encoder3_gamma),
            alpha_conv1=layer_params.get('encoder3_alpha_conv1', encoder3_alpha),
            alpha_conv2=layer_params.get('encoder3_alpha_conv2', encoder3_alpha)
        )
    
    # encoder_block4
    use_encoder4_adapter = layer_params.get('use_encoder4_adapter', True)
    if use_encoder4_adapter:
        encoder4_gamma = layer_params.get('encoder4_gamma', gamma)
        encoder4_alpha = layer_params.get('encoder4_alpha', alpha)
        model.base.encoder_block4.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block4.conv_block1, 
            encoder4_gamma, encoder4_alpha,
            gamma_conv1=layer_params.get('encoder4_gamma_conv1', encoder4_gamma),
            gamma_conv2=layer_params.get('encoder4_gamma_conv2', encoder4_gamma),
            alpha_conv1=layer_params.get('encoder4_alpha_conv1', encoder4_alpha),
            alpha_conv2=layer_params.get('encoder4_alpha_conv2', encoder4_alpha)
        )
    
    # encoder_block5
    use_encoder5_adapter = layer_params.get('use_encoder5_adapter', True)  # 默认应用
    if use_encoder5_adapter:
        encoder5_gamma = layer_params.get('encoder5_gamma', gamma)
        encoder5_alpha = layer_params.get('encoder5_alpha', alpha)
        model.base.encoder_block5.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block5.conv_block1, 
            encoder5_gamma, encoder5_alpha,
            gamma_conv1=layer_params.get('encoder5_gamma_conv1', encoder5_gamma),
            gamma_conv2=layer_params.get('encoder5_gamma_conv2', encoder5_gamma),
            alpha_conv1=layer_params.get('encoder5_alpha_conv1', encoder5_alpha),
            alpha_conv2=layer_params.get('encoder5_alpha_conv2', encoder5_alpha)
        )
    
    # encoder_block6
    use_encoder6_adapter = layer_params.get('use_encoder6_adapter', True)  # 默认应用
    if use_encoder6_adapter:
        encoder6_gamma = layer_params.get('encoder6_gamma', gamma)
        encoder6_alpha = layer_params.get('encoder6_alpha', alpha)
        model.base.encoder_block6.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.encoder_block6.conv_block1, 
            encoder6_gamma, encoder6_alpha,
            gamma_conv1=layer_params.get('encoder6_gamma_conv1', encoder6_gamma),
            gamma_conv2=layer_params.get('encoder6_gamma_conv2', encoder6_gamma),
            alpha_conv1=layer_params.get('encoder6_alpha_conv1', encoder6_alpha),
            alpha_conv2=layer_params.get('encoder6_alpha_conv2', encoder6_alpha)
        )
    
    # 瓶颈层的Conv-Adapter应用
    use_bottleneck_adapter = layer_params.get('use_bottleneck_adapter', True)  # 默认应用
    if use_bottleneck_adapter:
        bottleneck_gamma = layer_params.get('bottleneck_gamma', gamma)
        bottleneck_alpha = layer_params.get('bottleneck_alpha', alpha)
        model.base.conv_block7a.conv_block1 = apply_conv_adapter_to_conv_block(
            model.base.conv_block7a.conv_block1, 
            bottleneck_gamma, bottleneck_alpha,
            gamma_conv1=layer_params.get('bottleneck_gamma_conv1', bottleneck_gamma),
            gamma_conv2=layer_params.get('bottleneck_gamma_conv2', bottleneck_gamma),
            alpha_conv1=layer_params.get('bottleneck_alpha_conv1', bottleneck_alpha),
            alpha_conv2=layer_params.get('bottleneck_alpha_conv2', bottleneck_alpha)
        )
    
    # 解码器各层的Conv-Adapter应用
    # decoder_block1
    use_decoder1_adapter = layer_params.get('use_decoder1_adapter', True)  # 默认应用
    if use_decoder1_adapter:
        decoder1_gamma = layer_params.get('decoder1_gamma', gamma)
        decoder1_alpha = layer_params.get('decoder1_alpha', alpha)
        model.base.decoder_block1.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block1.conv_block2, 
            decoder1_gamma, decoder1_alpha,
            gamma_conv1=layer_params.get('decoder1_gamma_conv1', decoder1_gamma),
            gamma_conv2=layer_params.get('decoder1_gamma_conv2', decoder1_gamma),
            alpha_conv1=layer_params.get('decoder1_alpha_conv1', decoder1_alpha),
            alpha_conv2=layer_params.get('decoder1_alpha_conv2', decoder1_alpha)
        )
    
    # decoder_block2
    use_decoder2_adapter = layer_params.get('use_decoder2_adapter', True)  # 默认应用
    if use_decoder2_adapter:
        decoder2_gamma = layer_params.get('decoder2_gamma', gamma)
        decoder2_alpha = layer_params.get('decoder2_alpha', alpha)
        model.base.decoder_block2.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block2.conv_block2, 
            decoder2_gamma, decoder2_alpha,
            gamma_conv1=layer_params.get('decoder2_gamma_conv1', decoder2_gamma),
            gamma_conv2=layer_params.get('decoder2_gamma_conv2', decoder2_gamma),
            alpha_conv1=layer_params.get('decoder2_alpha_conv1', decoder2_alpha),
            alpha_conv2=layer_params.get('decoder2_alpha_conv2', decoder2_alpha)
        )
    
    # decoder_block3
    use_decoder3_adapter = layer_params.get('use_decoder3_adapter', True)
    if use_decoder3_adapter:
        decoder3_gamma = layer_params.get('decoder3_gamma', gamma)
        decoder3_alpha = layer_params.get('decoder3_alpha', alpha)
        model.base.decoder_block3.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block3.conv_block2, 
            decoder3_gamma, decoder3_alpha,
            gamma_conv1=layer_params.get('decoder3_gamma_conv1', decoder3_gamma),
            gamma_conv2=layer_params.get('decoder3_gamma_conv2', decoder3_gamma),
            alpha_conv1=layer_params.get('decoder3_alpha_conv1', decoder3_alpha),
            alpha_conv2=layer_params.get('decoder3_alpha_conv2', decoder3_alpha)
        )
    
    # decoder_block4
    use_decoder4_adapter = layer_params.get('use_decoder4_adapter', True)
    if use_decoder4_adapter:
        decoder4_gamma = layer_params.get('decoder4_gamma', gamma)
        decoder4_alpha = layer_params.get('decoder4_alpha', alpha)
        model.base.decoder_block4.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block4.conv_block2, 
            decoder4_gamma, decoder4_alpha,
            gamma_conv1=layer_params.get('decoder4_gamma_conv1', decoder4_gamma),
            gamma_conv2=layer_params.get('decoder4_gamma_conv2', decoder4_gamma),
            alpha_conv1=layer_params.get('decoder4_alpha_conv1', decoder4_alpha),
            alpha_conv2=layer_params.get('decoder4_alpha_conv2', decoder4_alpha)
        )
    
    # decoder_block5
    use_decoder5_adapter = layer_params.get('use_decoder5_adapter', True)
    if use_decoder5_adapter:
        decoder5_gamma = layer_params.get('decoder5_gamma', gamma)
        decoder5_alpha = layer_params.get('decoder5_alpha', alpha)
        model.base.decoder_block5.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block5.conv_block2, 
            decoder5_gamma, decoder5_alpha,
            gamma_conv1=layer_params.get('decoder5_gamma_conv1', decoder5_gamma),
            gamma_conv2=layer_params.get('decoder5_gamma_conv2', decoder5_gamma),
            alpha_conv1=layer_params.get('decoder5_alpha_conv1', decoder5_alpha),
            alpha_conv2=layer_params.get('decoder5_alpha_conv2', decoder5_alpha)
        )
    
    # decoder_block6
    use_decoder6_adapter = layer_params.get('use_decoder6_adapter',True)
    if use_decoder6_adapter:
        decoder6_gamma = layer_params.get('decoder6_gamma', gamma)
        decoder6_alpha = layer_params.get('decoder6_alpha', alpha)
        model.base.decoder_block6.conv_block2 = apply_conv_adapter_to_conv_block(
            model.base.decoder_block6.conv_block2, 
            decoder6_gamma, decoder6_alpha,
            gamma_conv1=layer_params.get('decoder6_gamma_conv1', decoder6_gamma),
            gamma_conv2=layer_params.get('decoder6_gamma_conv2', decoder6_gamma),
            alpha_conv1=layer_params.get('decoder6_alpha_conv1', decoder6_alpha),
            alpha_conv2=layer_params.get('decoder6_alpha_conv2', decoder6_alpha)
        )

    # 应用到FiLM层
    use_film = layer_params.get('use_film', False)  # 默认应用
    if use_film:
        film_gamma = layer_params.get('film_gamma', gamma)
        film_alpha = layer_params.get('film_alpha', alpha)
        model.film = apply_conv_adapter_to_film(model.film, film_gamma, film_alpha)
    
    # 应用到after_conv层
    use_after_conv = layer_params.get('use_after_conv', False)
    if use_after_conv and hasattr(model.base, 'after_conv'):
        after_conv_gamma = layer_params.get('after_conv_gamma', gamma)
        after_conv_alpha = layer_params.get('after_conv_alpha', alpha)
        model.base.after_conv = apply_conv_adapter_to_after_conv(
            model.base.after_conv,
            gamma=after_conv_gamma,
            alpha=after_conv_alpha
        )
    
    # 应用到ConvTranspose2d层
    use_conv_transpose = layer_params.get('use_conv_transpose', False)
    if use_conv_transpose:
        conv_transpose_gamma = layer_params.get('conv_transpose_gamma', gamma)
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
                
                # 应用Conv-Adapter
                setattr(parent, attr_name, apply_conv_adapter_to_conv_transpose(
                    module,
                    gamma=conv_transpose_gamma,
                    alpha=conv_transpose_alpha
                ))
    
    return model


class ConvAdapterResUNet(nn.Module):
    """
    带有Conv-Adapter的ResUNet模型 - 单模型版本
    支持为不同层设置不同的gamma和alpha值
    """
    def __init__(self, base_model, gamma=1, alpha=1, **layer_params):
        super().__init__()
        self.model = apply_conv_adapter_to_resunet(base_model, gamma, alpha, **layer_params)
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
    
 

class ConvAdapterConvTranspose2d(nn.Module):
    """
    带有Conv-Adapter的ConvTranspose2d层
    """
    def __init__(self, conv_layer, gamma=1, alpha=1):
        super().__init__()
        # 检查是否已经是ConvAdapterConvTranspose2d实例，避免嵌套问题
        if isinstance(conv_layer, ConvAdapterConvTranspose2d):
            # 如果传入的是ConvAdapterConvTranspose2d实例，则获取其内部的原始反卷积层
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
        
        # 超参数：通道下采样因子γ和缩放因子α（不可训练）
        self.gamma = gamma
        self.alpha = alpha
        
        # 冻结原始反卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 计算中间通道数（通道下采样）
        self.mid_channels = self.in_channels // self.gamma
        
        # 1. 深度可分离反卷积层
        self.depthwise_conv_transpose = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.in_channels,  # 深度反卷积：每个通道独立反卷积
            bias=False
        )
        
        # 2. 通道下采样层（1x1卷积）
        self.channel_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 3. 非线性激活函数统一为LeakyReLU
        self.activation = nn.LeakyReLU(inplace=False)
        
        # 4. 逐点卷积层
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 初始化权重
        self._init_weights()
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def _init_weights(self):
        """初始化Conv-Adapter的权重"""
        nn.init.kaiming_normal_(self.depthwise_conv_transpose.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.channel_down.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pointwise_conv.weight,  nonlinearity='leaky_relu')
    
    def forward(self, x):
        # 原始反卷积输出
        original_output = self.conv(x)
        
        # Conv-Adapter路径
        # 1. 深度可分离反卷积
        depthwise_out = self.depthwise_conv_transpose(x)
        
        # 2. 通道下采样
        downsampled = self.channel_down(depthwise_out)
        
        # 3. 非线性激活（LeakyReLU）
        activated = self.activation(downsampled)
        
        # 4. 逐点卷积
        adapter_output = self.pointwise_conv(activated)
        
        # 5. 应用超参数缩放因子α
        adapter_output = adapter_output * self.alpha
        
        # 组合原始输出和Conv-Adapter输出
        return original_output + adapter_output


class ConvAdapterAfterConv(nn.Module):
    """
    带有Conv-Adapter的after_conv层
    """
    def __init__(self, conv_layer, gamma=1, alpha=1):
        super().__init__()
        # 检查是否已经是ConvAdapterAfterConv实例，避免嵌套问题
        if isinstance(conv_layer, ConvAdapterAfterConv):
            # 如果传入的是ConvAdapterAfterConv实例，则获取其内部的原始卷积层
            self.conv = conv_layer.conv
        else:
            # 正常情况，传入的是普通卷积层
            self.conv = conv_layer
            
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        
        # 超参数：通道下采样因子γ和缩放因子α（不可训练）
        self.gamma = gamma
        self.alpha = alpha
        
        # 冻结原始卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # 计算中间通道数（通道下采样）
        self.mid_channels = self.in_channels // self.gamma
        
        # 1. 深度可分离卷积层
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels,  # 深度卷积：每个通道独立卷积
            bias=False
        )
        
        # 2. 通道下采样层（1x1卷积）
        self.channel_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 3. 非线性激活函数统一为LeakyReLU
        self.activation = nn.LeakyReLU(inplace=False)
        
        # 4. 逐点卷积层
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 初始化权重
        self._init_weights()
        
        # 检查是否在分布式环境中
        self.is_distributed = False
        try:
            self.is_distributed = torch.distributed.is_initialized()
        except:
            pass
    
    def _init_weights(self):
        """初始化Conv-Adapter的权重"""
        nn.init.kaiming_normal_(self.depthwise_conv.weight,nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.channel_down.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pointwise_conv.weight,  nonlinearity='leaky_relu')
    
    def forward(self, x):
        device = x.device
        # 原始卷积输出
        original_output = self.conv(x)
        
        # Conv-Adapter路径
        # 1. 深度可分离卷积
        depthwise_out = self.depthwise_conv(x)
        
        # 2. 通道下采样
        downsampled = self.channel_down(depthwise_out)
        
        # 3. 非线性激活（LeakyReLU）
        activated = self.activation(downsampled)
        
        # 4. 逐点卷积
        adapter_output = self.pointwise_conv(activated)
        
        # 5. 应用超参数缩放因子α
        adapter_output = adapter_output * self.alpha
        
        # 确保所有输出在同一设备上
        original_output = original_output.to(device)
        adapter_output = adapter_output.to(device)
        
        # 残差连接：原始输出 + Conv-Adapter输出
        return original_output + adapter_output


def apply_conv_adapter_to_after_conv(after_conv, gamma=1, alpha=1):
    """
    将Conv-Adapter应用到after_conv层
    
    参数:
        after_conv: 原始after_conv层
        gamma: 通道下采样因子
        alpha: 缩放因子
    
    返回:
        应用了Conv-Adapter的after_conv层
    """
    return ConvAdapterAfterConv(after_conv, gamma=gamma, alpha=alpha)


def apply_conv_adapter_to_conv_transpose(conv_transpose, gamma=1, alpha=1):
    """
    将Conv-Adapter应用到ConvTranspose2d层
    
    参数:
        conv_transpose: 原始ConvTranspose2d层
        gamma: 通道下采样因子
        alpha: 缩放因子
    
    返回:
        应用了Conv-Adapter的ConvTranspose2d层
    """
    return ConvAdapterConvTranspose2d(conv_transpose, gamma=gamma, alpha=alpha)


class ConvAdapterManager:
    """
    Conv-Adapter模型管理器 - 单模型版本
    支持为不同层设置不同的gamma和alpha值
    """
    def __init__(self, base_model, gamma=1, alpha=1, **layer_params):
        self.base_model = base_model
        self.gamma = gamma
        self.alpha = alpha
        self.layer_params = layer_params
        # 获取设备信息
        self.device = next(base_model.parameters()).device

        # 处理layer_params中的特殊参数
        # 确保所有必要的参数都存在，即使在JSON中没有明确指定
        # 这样可以避免模型加载时缺少键的问题
        default_params = {
            'use_film': False,
            'film_gamma': 1,
            'film_alpha': alpha,
            'use_bottleneck_adapter': True,
            'bottleneck_gamma': gamma,
            'bottleneck_alpha': alpha,
            'use_encoder1_adapter': True,
            'encoder1_gamma': gamma,
            'encoder1_alpha': alpha,
            'use_encoder2_adapter': True,
            'encoder2_gamma': gamma,
            'encoder2_alpha': alpha,
            'use_encoder3_adapter': True,
            'encoder3_gamma': gamma,
            'encoder3_alpha': alpha,
            'use_encoder4_adapter': True,
            'encoder4_gamma': gamma,
            'encoder4_alpha': alpha,
            'use_encoder5_adapter': True,
            'encoder5_gamma': gamma,
            'encoder5_alpha': alpha,
            'use_encoder6_adapter': True,
            'encoder6_gamma': gamma,
            'encoder6_alpha': alpha,
            'use_decoder1_adapter': True,
            'decoder1_gamma': gamma,
            'decoder1_alpha': alpha,
            'use_decoder2_adapter': True,
            'decoder2_gamma': gamma,
            'decoder2_alpha': alpha,
            'use_decoder3_adapter': True,
            'decoder3_gamma': gamma,
            'decoder3_alpha': alpha,
            'use_decoder4_adapter': True,
            'decoder4_gamma': gamma,
            'decoder4_alpha': alpha,
            'use_decoder5_adapter': True,
            'decoder5_gamma': gamma,
            'decoder5_alpha': alpha,
            'use_decoder6_adapter': True,
            'decoder6_gamma': gamma,
            'decoder6_alpha': alpha,
            'use_after_conv': False,
            'after_conv_gamma': gamma,
            'after_conv_alpha': alpha,
            'use_conv_transpose': False,
            'conv_transpose_gamma': gamma,
            'conv_transpose_alpha': alpha
        }
        # 更新默认参数
        for key, value in layer_params.items():
            default_params[key] = value

        # 创建Conv-Adapter模型
        self.conv_adapter_model = ConvAdapterResUNet(base_model, gamma, alpha, **default_params)
        # 确保Conv-Adapter模型在正确的设备上
        self.conv_adapter_model = self.conv_adapter_model.to(self.device)
    
    def get_conv_adapter_model(self):
        return self.conv_adapter_model

    def forward(self, input_dict):
        # 确保输入在正确的设备上
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and value.device != self.device:
                input_dict[key] = value.to(self.device)
        return self.conv_adapter_model(input_dict)
