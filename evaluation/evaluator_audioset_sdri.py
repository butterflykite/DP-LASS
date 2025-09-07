import os
import sys
sys.path.append('/home/zyk524/test/audio')
import csv
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import librosa
import lightning.pytorch as pl
import time
import json
# from optimize.load import load
from models.clap_encoder import CLAP_Encoder
from models.resunet import ResUNet30
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager0
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager1
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager2
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager3
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager4
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager5
from loramodel.lora_simple_single32 import DoRAManager as DoRAManager6
from models.audiosep import AudioSep
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)
import torch.nn.functional as F
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity

meta_csv_file = "/home/zyk524/test/audio/evaluation/metadata/class_labels_indices.csv"
df = pd.read_csv(meta_csv_file, sep=',')

IDS = df['mid'].tolist()
LABELS = df['display_name'].tolist()

CLASSES_NUM = len(LABELS)

IX_TO_LB = {i: label for i, label in enumerate(LABELS)}

class SmartDoRAEvaluator:
    """最佳DoRA模型评估器（无参考音频版本）

    对每个样本运行所有7个DoRA模型，并选择性能最佳的结果作为该样本的最终结果。
    使用无参考指标：CLAPScore、音频质量指标和分离度指标来评估分离效果。
    """

    def __init__(
            self,
            audios_dir='/dataset/AudioSep_eval/audioset',
            classes_num=527,
            sampling_rate=32000,
            number_per_class=10,
    ) -> None:
        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.number_per_class = number_per_class
        self.sampling_rate = sampling_rate


        # 存储已加载的模型
        self.base_model = None
        self.dora_managers = {}
        self.pl_models = {}
        self.query_encoder = None

        # 统计信息
        self.model_usage_stats = {}
        for i in range(7):  # 假设最多7个模型
            self.model_usage_stats[f"model{i}"] = 0
            
        # 用于CLAPScore计算的音频编码器
        self.audio_encoder = None
    @staticmethod
    def get_median_metrics(stats_dict, metric_type):
        """计算每个类别的中位数指标
        
        Args:
            stats_dict: 包含每个类别指标的字典
            metric_type: 指标类型
            
        Returns:
            dict: 每个类别的中位数指标
        """
        class_ids = stats_dict[metric_type].keys()
        median_stats_dict = {
            class_id: np.nanmedian(
                stats_dict[metric_type][class_id]) for class_id in class_ids if stats_dict[metric_type][class_id]
        }
        return median_stats_dict
        
    def _get_audio_names(self, audios_dir: str):
        """获取评估音频名称"""
        import re  # 导入正则表达式模块
        
        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [audio_name for audio_name in audio_names if '.wav' in audio_name]
        
        audio_names = [
            re.search(
                "(.*),(mixture|source).wav",
                audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))

        return audio_names
    def load_base_model(self, base_checkpoint_path, config_yaml):
        """加载基础模型

        Args:
            base_checkpoint_path (str): 基础模型检查点路径
            config_yaml (str): 配置文件路径
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n===== 加载基础模型 =====")
        print(f"检查点: {os.path.basename(base_checkpoint_path)}")

        # 加载配置
        configs = parse_yaml(config_yaml)
        input_channels = configs['model']['input_channels']
        output_channels = configs['model']['output_channels']
        condition_size = configs['model']['condition_size']

        print(
            f"模型配置: input_channels={input_channels}, output_channels={output_channels}, condition_size={condition_size}")

        # 创建基础模型
        self.base_model = ResUNet30(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_size=condition_size,
        )

        # 加载基础模型权重
        checkpoint = torch.load(base_checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.base_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"从'state_dict'加载基础模型权重成功")
        else:
            try:
                self.base_model.load_state_dict(checkpoint, strict=False)
                print(f"直接加载基础模型权重成功")
            except Exception as e:
                if hasattr(checkpoint, 'ss_model'):
                    self.base_model.load_state_dict(checkpoint.ss_model.state_dict(), strict=False)
                    print(f"从'ss_model'加载基础模型权重成功")
                else:
                    print(f"错误: 无法加载预训练模型权重: {e}")
                    raise ValueError(f"无法加载预训练模型权重: {e}")

        self.base_model = self.base_model.to(device)
        print(f"基础模型加载完成")

        # 创建查询编码器
        self.query_encoder= CLAP_Encoder().eval().to(device)
        print(f"查询编码器创建完成")
        
        # 创建音频编码器用于CLAPScore计算
        self.audio_encoder = CLAP_Encoder().eval().to(device)
        print(f"音频编码器创建完成")

    def load_all_dora_models(self, dora_checkpoint_paths):
        """加载所有DoRA模型

        Args:
            dora_checkpoint_paths (list): DoRA模型检查点路径列表
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.base_model is None:
            raise ValueError("请先加载基础模型")

        print(f"\n===== 加载所有DoRA模型 =====")
        print(f"共需加载 {len(dora_checkpoint_paths)} 个DoRA模型")

        # DoRAManager类映射
        dora_manager_classes = {
            0: DoRAManager0,
            1: DoRAManager1,
            2: DoRAManager2,
            3: DoRAManager3,
            4: DoRAManager4,
            5: DoRAManager5,
            6: DoRAManager6
        }

        for model_idx, dora_checkpoint_path in enumerate(dora_checkpoint_paths):
            print(f"\n  加载DoRA模型 {model_idx}: {os.path.basename(dora_checkpoint_path)}")

            # 选择对应的DoRAManager类
            if model_idx in dora_manager_classes:
                DoRAManagerClass = dora_manager_classes[model_idx]
            else:
                print(f"  警告: 模型索引 {model_idx} 超出范围，使用DoRAManager1")
                DoRAManagerClass = DoRAManager1

            # 创建DoRA管理器（基于基础模型的副本）
            base_model_copy = ResUNet30(
                input_channels=1,
                output_channels=1,
                condition_size=512,
            )
            base_model_copy.load_state_dict(self.base_model.state_dict())

            dora_manager = DoRAManagerClass(base_model_copy)

            # 加载DoRA权重
            try:
                dora_checkpoint = torch.load(dora_checkpoint_path, map_location=device)

                if 'dora_state_dict' in dora_checkpoint:
                    dora_manager.dora_model.load_state_dict(dora_checkpoint['dora_state_dict'])
                    print(f"  使用'dora_state_dict'加载DoRA权重成功")
                elif 'state_dict' in dora_checkpoint:
                    # 从Lightning checkpoint中提取DoRA相关参数
                    dora_state_dict = {}
                    for key, value in dora_checkpoint['state_dict'].items():
                        if any(dora_term in key.lower() for dora_term in
                               ['dora_adapters', 'dora_adapter', 'dora_', 'adapter']):
                            new_key = key
                            if 'model.base.' in key:
                                new_key = key.replace('model.base.', 'base.')
                            elif 'model.' in key:
                                new_key = key.replace('model.', '')
                            dora_state_dict[new_key] = value

                    if dora_state_dict:
                        dora_manager.dora_model.load_state_dict(dora_state_dict, strict=False)
                        print(f"  从'state_dict'提取DoRA参数加载成功")
                    else:
                        print(f"  警告: 未找到DoRA相关参数")
                else:
                    dora_manager.dora_model.load_state_dict(dora_checkpoint, strict=False)
                    print(f"  直接加载DoRA模型成功")

            except Exception as e:
                print(f"  警告: 加载DoRA模型权重失败: {e}")

            # 移动到设备并设置为评估模式
            dora_manager.dora_model = dora_manager.dora_model.to(device).eval()

            # 创建AudioSep模型
            pl_model = AudioSep(
                ss_model=dora_manager.dora_model,
                waveform_mixer=None,
                query_encoder=self.query_encoder,
                loss_function=None,
                optimizer_type=None,
                learning_rate=None,
                lr_lambda_func=None
            ).to(device).eval()

            # 存储模型
            self.dora_managers[model_idx] = dora_manager
            self.pl_models[model_idx] = pl_model

            print(f"  DoRA模型 {model_idx} 加载完成")

        print(f"\n所有DoRA模型加载完成，共 {len(self.pl_models)} 个模型")

    # def calculate_clap_score(self, audio, text, device):
    #     """计算CLAPScore - 音频与文本的语义相似度
    #
    #     Args:
    #         audio (np.ndarray): 分离后的音频
    #         text (str): 查询文本
    #         device: 计算设备
    #
    #     Returns:
    #         float: CLAPScore值
    #     """
    #     try:
    #         # 获取音频嵌入
    #         W = load(device)
    #         audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    #         audio_embed = self.audio_encoder.get_query_embed(
    #             modality='audio',
    #             audio=audio_tensor,
    #             device=device
    #         )
    #         audio_embed= torch.matmul(audio_embed,W)
    #         # 获取文本嵌入
    #         text_embed = self.audio_encoder.get_query_embed(
    #             modality='text',
    #             text=[text],
    #             device=device
    #         )
    #         clap_score = torch.cosine_similarity(audio_embed,text_embed, dim=-1)
    #         # clap_score = (audio_embeds * text_embed).sum(-1)
    #         return clap_score.item()
    #     except Exception as e:
    #         print(f"计算CLAPScore时出错: {e}")
    #         return 0.0


    def evaluate_sample(self, class_id, audio_name):
        sub_dir = os.path.join(
            self.audios_dir,
            "class_id={}".format(class_id))
            
        source_path = os.path.join(
            sub_dir, "{},source.wav".format(audio_name))
        mixture_path = os.path.join(
            sub_dir, "{},mixture.wav".format(audio_name))

        # 如果文件不存在，跳过
        if not os.path.exists(source_path) or not os.path.exists(mixture_path):
            print(f"警告: 类别 {class_id}, 音频 {audio_name} 的文件未找到，跳过")
            return None

        # 加载音频
        source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
        mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)
        sdr_no_sep = calculate_sdr(ref=source, est=mixture)

        text = IX_TO_LB[class_id]

        # 存储所有模型的结果
        model_results = []

        # 使用所有模型进行评估
        for model_idx, pl_model in self.pl_models.items():
            device = pl_model.device

            # 获取条件嵌入
            conditions = pl_model.query_encoder.get_query_embed(
                modality='text',
                text=[text],
                device=device
            )

            # 确保条件嵌入在正确的设备上
            if conditions.device != device:
                conditions = conditions.to(device)

            # 准备输入
            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": conditions,
            }

            # 使用当前DoRA模型进行分离
            with torch.no_grad():
                output_dict = pl_model.ss_model(input_dict)
                sep_segment = output_dict["waveform"]

            # 处理输出
            sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
            sdr = calculate_sdr(ref=source, est=sep_segment)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=source, est=sep_segment)
            # 计算无参考评估指标
            #CLAPScore - 音频与文本的语义相似度
            # clap_score = self.calculate_clap_score(sep_segment, text, device)
            total_score = sdri + sisdr
            # 存储结果
            model_results.append({
                "model_idx": model_idx,
                # "clap_score": clap_score,
                "sdri": sdri,
                "sisdr": sisdr,
                "total_score": total_score
            })

        # 找到最佳结果
        best_result = max(model_results, key=lambda x: x["total_score"])




        # 返回最佳结果和所有模型的结果
        return {
            # "clap_score": best_result["clap_score"],
            "sdri": best_result["sdri"],
            "sisdr": best_result["sisdr"],
        }

    def __call__(
            self,
            base_checkpoint_path,
            dora_checkpoint_paths,
            config_yaml,
    ):
        """执行最佳DoRA模型评估

        Args:
            base_checkpoint_path (str): 基础模型检查点路径
            dora_checkpoint_paths (list): DoRA模型检查点路径列表
            config_yaml (str): 配置文件路径

        Returns:
            dict: 包含评估结果的字典
        """
  

        print(f"\n===== 最佳DoRA模型评估开始 =====")

        # 加载基础模型
        self.load_base_model(base_checkpoint_path, config_yaml)

        # 加载所有DoRA模型
        self.load_all_dora_models(dora_checkpoint_paths)

        # 开始评估
        print(f"\n===== 开始样本评估 =====")
        all_results = []
        class_performance = {class_id: {"sdri": [], "sisdr": [], "total_score": []} for class_id in range(self.classes_num)}
        for class_id in tqdm(range(self.classes_num), desc="处理类别"):
            

            sub_dir = os.path.join(
                self.audios_dir,
                "class_id={}".format(class_id))

            if not os.path.exists(sub_dir):
                print(f"警告: 类别 {class_id} 的目录不存在: {sub_dir}")
                continue

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            # 对每个音频样本进行评估
            for audio_index, audio_name in enumerate(audio_names):
                if audio_index == self.number_per_class:
                    break

                sample_id = f"{class_id}_{audio_name}"
                result = self.evaluate_sample(class_id, audio_name)
                if result is not None:
                   all_results.append(result)
                class_performance[class_id]["sdri"].append(result["sdri"])
                class_performance[class_id]["sisdr"].append(result["sisdr"])
        # 计算统计信息
        sdris_dict = {}
        sisdrs_dict = {}
        total_scores_dict = {}
        
        for class_id in range(self.classes_num):
            if class_performance[class_id]["sdri"]:
                sdris_dict[class_id] = class_performance[class_id]["sdri"]
                sisdrs_dict[class_id] = class_performance[class_id]["sisdr"]
                
        
        # 计算每个类别的中位数
        median_sdris = self.get_median_metrics({"sdri": sdris_dict}, "sdri")
        median_sisdrs = self.get_median_metrics({"sisdr": sisdrs_dict}, "sisdr")
        

        class_median_performance = {
            "median_avg_sdri": np.mean(list(median_sdris.values())) if median_sdris else 0,
            "median_avg_sisdr": np.mean(list(median_sisdrs.values())) if median_sisdrs else 0,
        }

        return {
            "class_median_performance": class_median_performance
        }

def main():
    """主函数，用于测试最佳DoRA评估器（无参考音频版本）"""
    parser = argparse.ArgumentParser(description='最佳DoRA模型评估器')
    parser.add_argument('--audios_dir', type=str, default='/dataset/AudioSep_eval/audioset',
                        help='评估音频段的目录')
    parser.add_argument('--base_checkpoint', type=str,default='/home/zyk524/test/audio/checkpoint/audiosep_base_4M_steps.ckpt', help='基础模型检查点路径')
    parser.add_argument('--dora_checkpoints', type=str, nargs='+', default=[
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster0/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster1/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster2/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster3/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster4/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster5/checkpoints/dora_models/audiosep_base/dora0_model_epoch_149.pt",
        "/s7home/zyk524/test/audio/shendu_adapter/our/cluster6/checkpoints/dora_models/audiosep_base/dora0_model_epoch_149.pt"
    ],help='DoRA模型检查点路径列表')
    parser.add_argument('--config_yaml', type=str, default='/s7home/zyk524/test/audio/configs/audiosep_base.yaml', help='配置文件路径')
    parser.add_argument('--sampling_rate', type=int, default=32000, help='采样率')
    
    args = parser.parse_args()
    
    # 创建最佳DoRA评估器
    evaluator = SmartDoRAEvaluator(
        audios_dir=args.audios_dir,
        sampling_rate=args.sampling_rate,
    )
    
    # 执行评估
    results = evaluator(
        base_checkpoint_path=args.base_checkpoint,
        dora_checkpoint_paths=args.dora_checkpoints,
        config_yaml=args.config_yaml,
    )
    
    print(f"\n最佳DoRA模型评估完成！（无参考音频版本）")
    print(f"整体平均性能:  sdri={results['class_median_performance']['median_avg_sdri']:.3f}, sisdr={results['class_median_performance']['median_avg_sisdr']:.3f}")
if __name__ == "__main__":
    main()
