"""
图像分类示例：使用Vision Transformers进行图像分类
此示例展示如何使用Hugging Face Transformers进行图像分类任务
"""

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
        
        # 初始化处理器和模型
        logger.info(f"加载模型: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)

        # 创建分类pipeline
        self.classifier = pipeline(
            "image-classification",
            model=self.model,
            feature_extractor=self.processor,
            device=0 if self.device == "cuda" else -1
        )

    def classify_image_from_url(self, image_url: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """从URL加载图像并进行分类"""
        logger.info(f"从URL加载图像: {image_url}")

        # 下载图像
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # 使用pipeline进行分类
        predictions = self.classifier(image, top_k=top_k)

        return predictions

    def classify_image_from_path(self, image_path: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """从本地路径加载图像并进行分类"""
        logger.info(f"从本地路径加载图像: {image_path}")

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 使用pipeline进行分类
        predictions = self.classifier(image, top_k=top_k)

        return predictions

    def visualize_predictions(self, image: Image.Image, predictions: List[Dict[str, Union[str, float]]]):
        """可视化图像和预测结果"""
        # 创建图形
        plt.figure(figsize=(12, 6))

        # 显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('输入图像')

        # 显示预测结果
        plt.subplot(1, 2, 2)
        labels = [p['label'] for p in predictions]
        scores = [p['score'] for p in predictions]

        y_pos = np.arange(len(labels))

        plt.barh(y_pos, scores, align='center')
        plt.yticks(y_pos, labels)
        plt.xlabel('置信度')
        plt.title('预测结果')

        plt.tight_layout()
        plt.show()


def fine_tune_image_classifier(
    model_name: str = "google/vit-base-patch16-224",
    dataset_name: str = "cifar10",
    num_epochs: int = 3,
    batch_size: int = 8,
    output_dir: str = "./fine_tuned_vit"
):
    """微调图像分类模型的简化函数"""
    logger.info(f"开始微调模型 {model_name} 在 {dataset_name} 数据集上")

    # 加载数据集
    dataset = load_dataset(dataset_name)

    # 加载特征提取器和模型
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # 获取标签映射
    label2id = {label: i for i, label in enumerate(dataset["train"].features["label"].names)}
    id2label = {i: label for label, i in label2id.items()}

    # 加载模型
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # 定义图像处理函数
    def preprocess_function(examples):
        images = [Image.fromarray(img).convert("RGB") for img in examples["img"]]
        inputs = feature_extractor(images=images, return_tensors="pt")
        inputs["labels"] = [label2id[label] for label in examples["label"]]
        return inputs

    # 处理数据集
    train_dataset = dataset["train"].select(range(1000))  # 为了演示，只使用部分数据
    eval_dataset = dataset["test"].select(range(200))

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

    # 使用Trainer进行训练
    from transformers import TrainingArguments, Trainer

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 训练模型
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)

    logger.info(f"模型已保存到 {output_dir}")

    return model, feature_extractor


def main():
    """主函数：展示不同的图像分类方法"""
    # 1. 简单的图像分类示例
    print("\n=== 使用预训练ViT模型进行图像分类 ===")
    classifier = ImageClassifier()

    # 示例图像URL
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
        "https://farm1.staticflickr.com/29/57154382_38a4f94a01_z.jpg",  # 狗
    ]

    for url in image_urls:
        try:
            # 下载图像
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))

            # 分类
            predictions = classifier.classify_image_from_url(url)

            # 打印结果
            print(f"\n图像URL: {url}")
            print("预测结果:")
            for pred in predictions:
                print(f"- {pred['label']}: {pred['score']:.4f}")

            # 可视化（如果在支持显示的环境中）
            try:
                classifier.visualize_predictions(image, predictions)
            except:
                print("无法显示可视化结果")

        except Exception as e:
            print(f"处理图像时出错: {e}")

    # 2. 微调模型（可选，因为训练可能需要较长时间）
    print("\n=== 微调图像分类模型 ===")
    print("注意: 微调过程可能需要较长时间和较大的计算资源")
    user_input = input("是否要运行微调示例？(y/n): ")

    if user_input.lower() == 'y':
        fine_tune_image_classifier(
            model_name="google/vit-base-patch16-224",
            dataset_name="cifar10",
            num_epochs=1,  # 为了演示，只训练1个epoch
            batch_size=8,
            output_dir="./fine_tuned_vit"
        )


if __name__ == "__main__":
    main()
