"""
图像分类示例：使用Vision Transformers进行图像分类
此示例展示如何使用Hugging Face Transformers进行图像分类任务
"""

import os
import warnings

# 设置环境变量
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 禁用符号链接警告
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # 设置下载超时时间为120秒

# 过滤掉特定警告
warnings.filterwarnings("ignore", message=".*Xet Storage is enabled for this repo.*")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*cache-system uses symlinks.*")

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    pipeline
)
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional
import logging
import os
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    图像分类器类，使用Vision Transformer模型
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """初始化图像分类器"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        try:
            # 初始化处理器和模型
            logger.info(f"正在下载和加载模型: {model_name}...")
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
            logger.info("模型加载成功！")

            # 创建分类pipeline
            logger.info("正在创建分类pipeline...")
            self.classifier = pipeline(
                "image-classification",
                model=self.model,
                feature_extractor=self.processor,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("分类pipeline创建成功！")
            
        except Exception as e:
            logger.error(f"初始化图像分类器时出错: {e}")
            print("\n模型加载失败。可能的原因包括:")
            print("1. 网络连接问题 - 请检查您的网络连接")
            print("2. 模型下载失败 - 可以尝试:")
            print("   - 使用代理服务器")
            print("   - 增加超时时间（已设置为120秒）")
            print("   - 使用离线模式（需要预先下载模型）")
            print("3. 内存不足 - 如果使用GPU，可以尝试使用CPU版本")
            print("\n您可以参考以下文档:")
            print("- 离线模式: https://huggingface.co/docs/transformers/installation#offline-mode")
            print("- 常见问题: https://huggingface.co/docs/transformers/installation#troubleshooting")
            raise RuntimeError("模型初始化失败") from e

    def classify_image_from_url(self, image_url: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """从URL加载图像并进行分类"""
        logger.info(f"从URL加载图像: {image_url}")
        
        try:
            # 下载图像，设置超时
            response = requests.get(image_url, timeout=15)
            if response.status_code != 200:
                error_msg = f"下载图像失败，HTTP状态码: {response.status_code}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            # 打开图像
            try:
                image = Image.open(BytesIO(response.content))
                image = image.convert("RGB")  # 确保图像是RGB格式
            except Exception as e:
                logger.error(f"无法打开或处理图像: {e}")
                raise RuntimeError(f"无法处理图像: {e}")
                
            # 使用pipeline进行分类
            logger.info("开始图像分类...")
            predictions = self.classifier(image, top_k=top_k)
            logger.info(f"分类完成，找到 {len(predictions)} 个预测结果")
            
            return predictions
            
        except requests.exceptions.Timeout:
            logger.error("请求超时")
            raise RuntimeError("下载图像超时，请检查您的网络连接或尝试其他图像URL")
        except requests.exceptions.ConnectionError:
            logger.error("网络连接错误")
            raise RuntimeError("网络连接错误，请检查您的网络连接")
        except Exception as e:
            logger.error(f"分类过程中出错: {e}")
            raise RuntimeError(f"图像分类失败: {e}")

    def classify_image_from_path(self, image_path: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """从本地路径加载图像并进行分类"""
        logger.info(f"从本地路径加载图像: {image_path}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                error_msg = f"图像文件不存在: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            # 加载图像
            try:
                image = Image.open(image_path).convert("RGB")
                logger.info("图像加载成功")
            except Exception as e:
                logger.error(f"无法打开或处理图像: {e}")
                raise RuntimeError(f"无法处理图像: {e}")
                
            # 使用pipeline进行分类
            logger.info("开始图像分类...")
            predictions = self.classifier(image, top_k=top_k)
            logger.info(f"分类完成，找到 {len(predictions)} 个预测结果")
            
            return predictions
            
        except Exception as e:
            logger.error(f"分类过程中出错: {e}")
            raise RuntimeError(f"图像分类失败: {e}")

    def visualize_predictions(self, image: Image.Image, predictions: List[Dict[str, Union[str, float]]]):
        """可视化图像和预测结果"""
        try:
            logger.info("开始可视化预测结果...")
            
            # 检查输入
            if not isinstance(image, Image.Image):
                error_msg = f"image参数必须是PIL图像对象，而不是{type(image)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not predictions or not isinstance(predictions, list):
                error_msg = f"predictions参数必须是非空列表，而不是{type(predictions)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 创建图形
            plt.figure(figsize=(12, 6))

            # 显示图像
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('输入图像')

            # 显示预测结果
            plt.subplot(1, 2, 2)
            
            # 安全地获取标签和分数
            labels = []
            scores = []
            for i, p in enumerate(predictions):
                if 'label' not in p or 'score' not in p:
                    logger.warning(f"预测结果 #{i} 缺少label或score字段: {p}")
                    continue
                labels.append(p['label'])
                scores.append(p['score'])
            
            if not labels:
                logger.warning("没有有效的预测结果可以显示")
                plt.text(0.5, 0.5, '没有有效的预测结果', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=plt.gca().transAxes)
            else:
                y_pos = np.arange(len(labels))
                plt.barh(y_pos, scores, align='center')
                plt.yticks(y_pos, labels)
                plt.xlabel('置信度')
                plt.title('预测结果')

            plt.tight_layout()
            
            # 尝试显示图像
            try:
                plt.show()
                logger.info("可视化显示成功")
            except Exception as e:
                logger.warning(f"无法显示图像: {e}")
                print("注意: 无法在当前环境中显示图像。这在无GUI的环境中是正常的。")
                print("您可以尝试:")
                print("1. 在支持GUI的环境中运行")
                print("2. 使用matplotlib的非交互式后端")
                print("3. 将图像保存为文件而不是显示")
                
        except Exception as e:
            logger.error(f"可视化过程中出错: {e}")
            print(f"可视化失败: {e}")
            print("继续执行其余代码...")


def fine_tune_image_classifier(
    model_name: str = "google/vit-base-patch16-224",
    dataset_name: str = "cifar10",
    num_epochs: int = 3,
    batch_size: int = 8,
    output_dir: str = "./fine_tuned_vit"
):
    """微调图像分类模型的简化函数"""
    logger.info(f"开始微调模型 {model_name} 在 {dataset_name} 数据集上")

    try:
        # 加载数据集
        logger.info(f"正在加载数据集 {dataset_name}...")
        dataset = load_dataset(dataset_name)
        logger.info(f"数据集加载成功，包含 {len(dataset['train'])} 个训练样本和 {len(dataset['test'])} 个测试样本")

        # 加载特征提取器和模型
        logger.info(f"正在加载特征提取器 {model_name}...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # 获取标签映射
        label2id = {label: i for i, label in enumerate(dataset["train"].features["label"].names)}
        id2label = {i: label for label, i in label2id.items()}
        logger.info(f"标签映射创建成功，共 {len(label2id)} 个类别")

        # 加载模型
        logger.info(f"正在加载模型 {model_name}...")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载数据集或模型时出错: {e}")
        print("\n如果是网络连接问题，您可以尝试以下解决方案:")
        print("1. 检查您的网络连接")
        print("2. 使用代理服务器")
        print("3. 增加超时时间: 已设置环境变量 HF_HUB_DOWNLOAD_TIMEOUT=120")
        print("4. 使用离线模式 (参考: https://huggingface.co/docs/transformers/installation#offline-mode)")
        print("\n对于CIFAR-10数据集，您也可以尝试直接从官方网站下载: https://www.cs.toronto.edu/~kriz/cifar.html")
        
        # 提供一个简单的替代方案
        print("\n由于无法加载模型或数据集，这里提供一个简单的图像分类示例:")
        print("- CIFAR-10数据集包含10个类别: 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车")
        print("- Vision Transformer (ViT) 是一种基于Transformer架构的图像分类模型")
        print("- 微调过程通常包括: 加载预训练模型、准备数据集、设置训练参数、训练模型、评估模型")
        return None, None

    try:
        # 定义图像处理函数
        logger.info("定义数据预处理函数...")
        def preprocess_function(examples):
            images = [Image.fromarray(img).convert("RGB") for img in examples["img"]]
            inputs = feature_extractor(images=images, return_tensors="pt")
            inputs["labels"] = [label2id[label] for label in examples["label"]]
            return inputs

        # 处理数据集
        logger.info("处理训练和评估数据集...")
        train_dataset = dataset["train"].select(range(1000))  # 为了演示，只使用部分数据
        eval_dataset = dataset["test"].select(range(200))

        logger.info("应用数据预处理...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["img", "label"]
        )

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["img", "label"]
        )
        logger.info("数据预处理完成")

        # 使用Trainer进行训练
        from transformers import TrainingArguments, Trainer

        logger.info("设置训练参数...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        logger.info("创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # 训练模型
        logger.info("开始训练模型...")
        trainer.train()
        logger.info("模型训练完成")

        # 保存模型
        logger.info(f"正在保存模型到 {output_dir}...")
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        logger.info(f"模型已成功保存到 {output_dir}")

        return model, feature_extractor
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        print(f"\n训练过程中出现错误: {e}")
        print("可能的原因包括:")
        print("1. 内存不足 - 尝试减小batch_size或使用更小的模型")
        print("2. GPU内存不足 - 如果使用GPU，尝试减小batch_size或使用CPU训练")
        print("3. 数据集格式问题 - 检查数据集是否正确加载")
        print("4. 网络连接问题 - 如果在训练过程中需要下载额外资源")
        
        return None, None


def main():
    """主函数：展示不同的图像分类方法"""
    print("\n=== 使用预训练ViT模型进行图像分类 ===")
    
    try:
        # 1. 创建图像分类器
        print("正在初始化图像分类器...")
        classifier = ImageClassifier()
        print("图像分类器初始化成功！")

        # 示例图像URL
        image_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
            "https://farm1.staticflickr.com/29/57154382_38a4f94a01_z.jpg",  # 狗
        ]

        # 2. 处理每个示例图像
        for i, url in enumerate(image_urls):
            print(f"\n处理示例图像 {i+1}/{len(image_urls)}: {url}")
            try:
                # 设置超时，避免长时间等待
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    print(f"下载图像失败，HTTP状态码: {response.status_code}")
                    continue
                    
                # 下载图像
                image = Image.open(BytesIO(response.content))
                print("图像下载成功")

                # 分类
                print("正在进行图像分类...")
                predictions = classifier.classify_image_from_url(url)

                # 打印结果
                print("预测结果:")
                for pred in predictions:
                    print(f"- {pred['label']}: {pred['score']:.4f}")

                # 可视化（如果在支持显示的环境中）
                try:
                    print("尝试显示可视化结果...")
                    classifier.visualize_predictions(image, predictions)
                except Exception as viz_error:
                    print(f"无法显示可视化结果: {viz_error}")
                    print("提示: 确保您的环境支持matplotlib图形显示")

            except requests.exceptions.Timeout:
                print(f"下载图像超时，请检查您的网络连接或尝试其他图像URL")
            except requests.exceptions.ConnectionError:
                print(f"网络连接错误，请检查您的网络连接")
            except Exception as e:
                print(f"处理图像时出错: {e}")
                print("尝试使用本地图像可能会更可靠")

        # 3. 提示用户尝试自己的图像
        print("\n您也可以尝试使用自己的图像:")
        print("1. 将图像放在examples目录下")
        print("2. 修改代码中的image_path变量")
        print("3. 或者提供一个有效的图像URL")

    except RuntimeError as e:
        print(f"\n初始化图像分类器失败: {e}")
        print("请检查网络连接和系统资源")
        print("您可以尝试:")
        print("1. 检查网络连接")
        print("2. 使用较小的模型")
        print("3. 确保您有足够的系统内存")
        print("4. 如果使用GPU，检查GPU内存是否足够")

    # 4. 微调模型（可选，因为训练可能需要较长时间）
    print("\n=== 微调图像分类模型 ===")
    print("注意: 微调过程可能需要较长时间和较大的计算资源")
    print("所需资源:")
    print("- 良好的网络连接 (用于下载数据集和模型)")
    print("- 足够的RAM (至少8GB)")
    print("- 推荐使用GPU (用于加速训练)")
    print("- 足够的存储空间 (用于保存模型和数据集)")
    
    try:
        user_input = input("是否要运行微调示例？(y/n): ")

        if user_input.lower() == 'y':
            print("\n开始微调过程...")
            model, feature_extractor = fine_tune_image_classifier(
                model_name="google/vit-base-patch16-224",
                dataset_name="cifar10",
                num_epochs=1,  # 为了演示，只训练1个epoch
                batch_size=8,
                output_dir="./fine_tuned_vit"
            )
            
            if model is not None and feature_extractor is not None:
                print("\n微调成功完成！")
                print(f"微调后的模型已保存到 ./fine_tuned_vit")
                print("您可以使用这个微调后的模型进行推理")
            else:
                print("\n微调过程未成功完成")
        else:
            print("\n已跳过微调过程")
            print("您可以在准备好资源后再尝试微调")
    except Exception as e:
        print(f"\n微调过程中出现未预期的错误: {e}")


if __name__ == "__main__":
    main()
