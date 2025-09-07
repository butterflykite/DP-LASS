import os
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import matplotlib.patheffects as patheffects
# 导入AudioSep相关模块
from models.CLAP.open_clip.htsat import HTSAT_Swin_Transformer
from models.CLAP.open_clip.model import CLAP
from models.CLAP.training.data import get_audio_features
import csv
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from sklearn.cluster import AffinityPropagation
def load_clap_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载预训练的CLAP模型

    参数：
        checkpoint_path: 模型检查点路径
        device: 运行设备

    返回：
        model: 加载好的CLAP模型
    """
    # 使用CLAP_Encoder类加载模型
    from models.clap_encoder import CLAP_Encoder

    # 创建CLAP_Encoder实例
    clap_encoder = CLAP_Encoder(
        pretrained_path=checkpoint_path,
        sampling_rate=32000,  # 与load_audio函数中的target_sr保持一致
        amodel="HTSAT-base"
    )

    # 将模型移动到指定设备
    clap_encoder = clap_encoder.to(device)
    clap_encoder.model = clap_encoder.model.to(device)
    clap_encoder.eval()

    return clap_encoder.model


def load_dataset(json_path):
    """
    加载音频数据集和对应标签
    参数：
        json_path: 包含音频路径和类别信息的JSON文件路径
    返回：
        audio_paths: 音频路径列表 [(path, category), ...]
        category_set: 所有类别的集合
    """
    # 加载JSON数据
    with open(json_path) as f:
        data = json.load(f)
    
    # 获取数据项列表
    items = data["data"]
    
    # 构建音频路径列表和类别集合
    audio_paths = []
    category_set = set()
    
    for item in items:
        if "wav" in item and "caption" in item:
            audio_path = item["wav"]
            category = item["caption"]
            
            # 添加到路径列表，直接使用路径和类别
            audio_paths.append((audio_path, category))
            
            # 添加到类别集合
            category_set.add(category)
    
    print(f"从JSON加载了 {len(audio_paths)} 个音频文件和 {len(category_set)} 个不同类别")
    return audio_paths, category_set


def load_audio(audio_path, target_sr=32000, max_duration=10):
    """
    加载并预处理音频文件
    参数：
        audio_path: 音频文件路径
        target_sr: 目标采样率
        max_duration: 最大时长（秒），超长音频截断
    返回：
        waveform: 标准化后的音频波形
    """
    try:
        # 计算最大样本数
        max_samples = int(max_duration * target_sr)
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        # 检查波形是否为空
        if waveform.numel() == 0 or waveform.shape[1] == 0:
            print(f"Warning: Empty waveform detected for {audio_path}, skipping this file")
            return None
        waveform = waveform.to(torch.float32)

        # 重采样
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # 确保单声道
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)


        # 确保波形是一维的
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # 截断或填充到 max_samples
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]
        elif waveform.shape[0] < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))

        # 幅度归一化
        max_abs = waveform.abs().max()
        if max_abs > 0:
            waveform = waveform / max_abs
        else:
            waveform = torch.zeros_like(waveform)  # 全零波形保持为零
        if not torch.isfinite(waveform).all():
            print(f"数值异常 {audio_path}")
            return None
        # 验证输出形状
        assert waveform.shape == (max_samples,), f"Unexpected shape {waveform.shape} for {audio_path}"

        return waveform

    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        # 返回None而不是零向量，以便后续处理可以跳过这个文件
        return None


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files):
        self.audio_files = audio_files  # [(path, category), ...]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 尝试获取有效的音频文件，如果当前索引的文件无效，则尝试下一个
        original_idx = idx
        max_attempts = len(self.audio_files)
        attempts = 0

        while attempts < max_attempts:
            path, category = self.audio_files[idx]
            waveform = load_audio(path)

            if waveform is not None:
                # 找到有效的音频文件
                return waveform, category, path

            # 如果当前文件无效，尝试下一个
            print(f"Skipping invalid audio file at index {idx}: {path}")
            idx = (idx + 1) % len(self.audio_files)  # 循环到数据集开头
            attempts += 1

            # 如果已经尝试了所有文件还没找到有效的，返回一个零向量作为最后的备选
            if idx == original_idx or attempts >= max_attempts:
                print(f"Warning: Could not find valid audio file after {attempts} attempts, using zeros as fallback")
                # 使用原始索引的文件信息，但返回零向量
                path, category = self.audio_files[original_idx]
                return torch.zeros(320000, dtype=torch.float32), category, path

def extract_audio_embeddings(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用HTSAT模型提取音频嵌入向量

    参数：
        model: CLAP模型
        dataloader: 音频数据加载器
        device: 运行设备

    返回：
        embeddings: 嵌入向量列表
        categories: 对应的类别列表
        audio_paths: 对应的音频路径列表
    """
    embeddings = []
    categories = []
    audio_paths = []

    with torch.no_grad():
        for batch in dataloader:
            # 获取音频数据、类别和路径
            waveforms, batch_categories, batch_paths = batch
            waveforms = waveforms.to(device)

            # 创建音频特征字典列表
            audio_dict_list = []
            # 重采样到48kHz (CLAP模型需要)
            waveforms = torchaudio.functional.resample(waveforms, orig_freq=32000, new_freq=48000)

            # 为每个波形创建特征字典
            for i in range(waveforms.size(0)):
                audio_dict = {}
                # 将CLAPAudioCfp对象转换为字典，以便get_mel函数可以使用下标访问
                if hasattr(model, 'model_cfg'):
                    audio_config = model.model_cfg["audio_cfg"]
                else:
                    # 将CLAPAudioCfp对象转换为字典
                    audio_config = {
                        "sample_rate": model.audio_cfg.sample_rate,
                        "window_size": model.audio_cfg.window_size,
                        "hop_size": model.audio_cfg.hop_size,
                        "fmin": model.audio_cfg.fmin,
                        "fmax": model.audio_cfg.fmax,
                        "mel_bins": model.audio_cfg.mel_bins,
                        "clip_samples": model.audio_cfg.clip_samples
                    }

                audio_dict = get_audio_features(
                    audio_dict,
                    waveforms[i],
                    480000,  # 最大样本数 (10秒 * 48000Hz)
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=audio_config
                )
                audio_dict_list.append(audio_dict)

            # 提取嵌入向量
            batch_embeddings = model.get_audio_embedding(audio_dict_list).cpu().numpy()

            embeddings.extend(batch_embeddings)
            categories.extend(batch_categories)
            audio_paths.extend(batch_paths)

    return np.array(embeddings), categories, audio_paths


def extract_category_features(feature_matrix, categories):
    """
    提取每个类别的特征向量

    参数：
        feature_matrix: 音频特征矩阵
        categories: 类别列表
    返回：
        category_features: 类别到特征向量的映射
        unique_categories: 唯一类别列表
    """
    # 收集每个类别的特征
    category_to_features = defaultdict(list)
    for i, category in enumerate(categories):
        # 直接添加特征到对应类别
        category_to_features[category].append(feature_matrix[i])

    # 计算每个类别的平均特征向量
    category_features = {}
    for category, features in category_to_features.items():
        if features:  # 确保有特征可用
            features_array = np.array(features)
            mean_feature = np.mean(features_array, axis=0)
            category_features[category] = mean_feature

    # 获取所有有特征的类别
    unique_categories = list(category_features.keys())
    print(f"提取了 {len(unique_categories)} 个类别的特征向量")

    return category_features, unique_categories


def cluster_categories(category_features, categories, n_clusters=7):
    """
    对类别进行聚类

    参数：
        category_features: 类别到特征向量的映射
        categories: 类别列表
        n_clusters: 聚类数量
    返回：
        cluster_results: 聚类结果 {cluster_id: [categories]}
        category_feature_matrix: 类别特征矩阵
        cluster_labels: 聚类标签
    """
    # 准备特征矩阵
    category_feature_matrix = np.array([category_features[cat] for cat in categories])

    # 进行聚类
    # kmeans = AffinityPropagation(damping=0.5)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(category_feature_matrix)
    cluster_centers = kmeans.cluster_centers_
    # 整理聚类结果
    cluster_results = defaultdict(list)
    for category, label in zip(categories, cluster_labels):
        cluster_results[int(label)].append(category)

    # 打印聚类结果摘要
    print("\n类别聚类总结：")
    for cluster_id in range(n_clusters):
        categories_in_cluster = cluster_results[cluster_id]
        print(f"聚类 {cluster_id}：")
        print(f"  类别数量：{len(categories_in_cluster)}")
        print(f"  类别：{', '.join(sorted(categories_in_cluster[:5]))}" +
              ("..." if len(categories_in_cluster) > 5 else "") + "\n")

    return cluster_results, category_feature_matrix, cluster_labels,cluster_centers


def save_cluster_centers(cluster_centers, output_path):
    """
    保存聚类中心到文件

    参数：
        cluster_centers: 聚类中心矩阵
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为numpy格式
    np.save(output_path, cluster_centers)
    print(f'Cluster centers saved to {output_path}')

    # 同时保存为JSON格式（便于查看）
    json_path = output_path.replace('.npy', '.json')
    cluster_centers_dict = {
        'n_clusters': len(cluster_centers),
        'embedding_dim': cluster_centers.shape[1],
        'cluster_centers': cluster_centers.tolist()
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_centers_dict, f, indent=2)
    print(f'Cluster centers in JSON format saved to {json_path}')
def visualize_category_clusters(category_feature_matrix, cluster_labels, categories, save_path=None):
    """
       可视化类别聚类结果

       参数：
           category_feature_matrix: 类别特征矩阵
           cluster_labels: 聚类标签
           categories: 类别列表
           save_path: 保存路径
       """
    # 加载类别索引映射
    category_mapping = {}
    try:
        # 尝试从class_labels_indices.csv加载类别索引映射
        csv_path = "/home/zyk524/test/audio/evaluation/metadata/class_labels_indices.csv"
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category_mapping[row['display_name']] = int(row['index'])
    except FileNotFoundError:
        raise RuntimeError(f"找不到索引文件：{csv_path}")
    except Exception as e:
        raise RuntimeError(f"加载索引文件失败：{str(e)}")

    # 验证并转换类别名称到索引
    converted_indices = []
    missing_categories = []

    for name in categories:
        if name not in category_mapping:
            missing_categories.append(name)
            continue
        converted_indices.append(category_mapping[name])

    if missing_categories:
        error_msg = f"发现{len(missing_categories)}个未注册类别：\n{missing_categories[:5]}"
        if len(missing_categories) > 5:
            error_msg += "..."
        raise ValueError(error_msg)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(50, len(category_feature_matrix) // 3))
    embeddings_2d = tsne.fit_transform(category_feature_matrix)

    # 可视化设置
    plt.figure(figsize=(18, 18))
    ax = plt.gca()

    # 设置散点大小
    scatter_size = 1000

    # 创建散点图
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=cluster_labels, cmap='tab20', alpha=0.85,
                         s=scatter_size, edgecolor='black', linewidth=0.5)

    # 添加索引标签
    for i, (x, y) in enumerate(embeddings_2d):
        index = converted_indices[i]  # 使用转换后的索引列表
        color = scatter.to_rgba(cluster_labels[i])  # 获取点的颜色
        text_color = "white" if np.mean(color[:3]) < 0.6 else "black"  # 根据颜色亮度调整文字颜色

        ax.text(x, y, str(index),
                fontsize=14,
                ha='center', va='center',
                color=text_color,
                fontweight='bold',
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],  # 添加黑色描边
                alpha=0.9)

    plt.title(f"Audioembedding Category Clusters (k={len(set(cluster_labels))})", fontsize=16)

    # 紧凑布局
    plt.tight_layout(pad=2)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至：{save_path}")
    else:
        plt.show()
def save_clusters_to_json(cluster_results, output_path):
    """
    将聚类结果保存为JSON文件

    参数：
        cluster_results: 聚类结果字典 {cluster_id: [categories]}
        output_path: 输出JSON文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建聚类结果字典
    clusters = {}

    # 按聚类ID组织类别，确保没有重复
    for cluster_id, categories in cluster_results.items():
        cluster_key = f'cluster{cluster_id}'

        # 确保类别不重复
        unique_categories = sorted(list(set(categories)))

        # 初始化聚类数据
        clusters[cluster_key] = {
            "cluster_id": int(cluster_id) if hasattr(cluster_id, 'item') else cluster_id,
            "categories_count": len(unique_categories),
            "categories_in_cluster": unique_categories
        }

    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print(f'聚类结果已保存到 {output_path}')


def create_cluster_datafiles(cluster_results, output_dir, n_clusters):
    """
    为每个聚类创建单独的数据文件，只包含类别名称

    参数：
        cluster_results: 聚类结果字典 {cluster_id: [categories]}
        n_clusters: 聚类数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 为每个聚类创建数据文件
    for cluster_id in range(n_clusters):
        # 确保类别不重复
        unique_categories = sorted(list(set(cluster_results[cluster_id])))

        cluster_data = {
            "cluster_id": cluster_id,
            "categories": unique_categories
        }
        output_path = os.path.join(output_dir, f'cluster{cluster_id}_data.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)

        print(f'聚类 {cluster_id} 的数据文件已保存到 {output_path}')


def main(label_json_path, checkpoint_path, output_dir, n_clusters=7, batch_size=16):
    """
    主函数

    参数：
        label_json_path: 标签JSON文件路径
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        n_clusters: 聚类数量
        batch_size: 批处理大小
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载数据集
    print('正在加载数据集...')
    audio_files, category_set = load_dataset(label_json_path)
    print(f'加载了 {len(audio_files)} 个音频文件')

    # 创建数据集
    dataset = AudioDataset(audio_files)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: (torch.stack([item[0] for item in x]),
                                                  [item[1] for item in x],
                                                  [item[2] for item in x]))

    # 加载模型
    model = load_clap_model(checkpoint_path, device)

    # 提取嵌入向量
    print('正在提取音频嵌入向量...')
    embeddings, categories, audio_paths = extract_audio_embeddings(model, dataloader, device)

    # 提取类别特征
    print("正在提取类别特征...")
    category_features, unique_categories = extract_category_features(embeddings, categories)

    # 聚类类别
    print(f"正在对 {len(unique_categories)} 个类别进行聚类，分为 {n_clusters} 类...")
    cluster_results, category_feature_matrix, cluster_labels,cluster_centers = cluster_categories(
        category_features, unique_categories, n_clusters)
    # 保存聚类中心
    cluster_centers_path = os.path.join(output_dir, 'cluster_centers.npy')
    save_cluster_centers(cluster_centers, cluster_centers_path)
    # 可视化聚类结果
    print("正在可视化聚类结果...")
    visualize_category_clusters(
        category_feature_matrix, cluster_labels, unique_categories,
        save_path=os.path.join(output_dir, 'category_cluster_visualization.png'))

    # 保存聚类结果
    save_clusters_to_json(cluster_results, os.path.join(output_dir, 'category_clusters.json'))

    # 为每个聚类创建数据文件
    create_cluster_datafiles(cluster_results, output_dir, n_clusters)

    print("聚类完成！")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='使用CLAP的HTSAT模型对音频类别进行聚类')
    parser.add_argument('--checkpoint', type=str,
                        default="/home/zyk524/test/audio/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
                        help='模型检查点路径')
    parser.add_argument('--label_json', type=str,
                        default="/home/zyk524/MSCLAP/AudioSet_Auido/audioset_eval_json/output.json", required=False,
                        help='包含音频路径和类别信息的JSON文件路径，格式：{"data": [{"wav": "路径", "caption": "类别"},...]}')
    parser.add_argument('--output_dir', type=str, default='/home/zyk524/MSCLAP/audioset_eval_best5/alleval/no100',
                        help='输出目录')
    parser.add_argument('--clusters', type=int, default=7,
                        help='聚类数量')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='批处理大小')

    args = parser.parse_args()

    main(args.label_json, args.checkpoint, args.output_dir, args.clusters, args.batch_size)
