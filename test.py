from croppingDataset import GAICD
import os
import sys
import time
import math
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import argparse
import numpy as np
import random
from scipy.stats import spearmanr, pearsonr
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # 引入 tqdm 库
import wandb 
import torch.nn.functional as F
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1,2,3"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 扩展支持的backbone字典
BACKBONES = {
    # ResNet系列
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    
    # ResNeXt系列
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    
    # DenseNet系列
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    
    # ConvNeXt系列
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    
    # MNASNet系列
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet1_0': models.mnasnet1_0,
    
    # MobileNet系列
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    
    # RegNet系列
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    
    # ShuffleNet系列
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    
    # EfficientNet系列
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    
    # VGG系列
    'vgg16': models.vgg16,
}

parser = argparse.ArgumentParser(description='Grid anchor based image cropping')
parser.add_argument('--dataset_root', default='dataset/GAIC/', help='Dataset root directory path')
# parser.add_argument('--base_model', default='mobilenetv2', help='Pretrained base model')
# parser.add_argument('--downsample', default=4, type=int, help='downsample time')
parser.add_argument('--image_size', default=256, type=int, help='Batch size for training')
# parser.add_argument('--align_size', default=9, type=int, help='Spatial size of RoIAlign and RoDAlign')
# parser.add_argument('--reduced_dim', default=8, type=int, help='Spatial size of RoIAlign and RoDAlign')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--augmentation', default=1, type=int, help='choose single or multi scale')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--save_folder', default='weights/ablation/cropping/', help='Directory for saving checkpoint models')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--device', default='cuda', type=str, help='Device to train on')
parser.add_argument('--backbone', default='resnet50', type=str, 
                    choices=list(BACKBONES.keys()),
                    help='Backbone network architecture')
parser.add_argument('--output_dir', default='results/ablation/cropping/', help='Directory for saving results')
parser.add_argument('--test', default=False, type=bool, help='test or not')
parser.add_argument('--load_model', default=None, type=str, help='load model')
parser.add_argument('--model_epoch', default=None, type=int, help='model epoch')
args = parser.parse_args()

args.save_folder = args.save_folder + '/' + args.backbone + '/lr_' + str(args.lr) + '_' + time.strftime('%Y%m%d_%H%M%S')
args.output_dir = args.output_dir + '/' + args.backbone + '/' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(args.model_epoch)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

cuda = True if torch.cuda.is_available() else False



# if cuda:
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

def custom_collate(batch):
    images = []
    targets = []
    imgpaths = []
    for sample in batch:
        image = torch.from_numpy(sample['image'])
        ori_h = image.shape[1]
        ori_w = image.shape[2]
        print(f"  Original image size: {ori_h}x{ori_w}")
        bbox = sample['bbox']
        mos_scores = sample['MOS']
        # 调整图像大小为固定尺寸 (3, 256, 256)
        if image.shape[1] != 256 or image.shape[2] != 256:
            m = nn.AdaptiveAvgPool2d((256, 256))
            image = m(image.unsqueeze(0)).squeeze(0)
        # 使用多个高质量边界框而不是只选最好的
        top_k = 5
        top_indices = np.argsort(mos_scores)[-top_k:]
        for idx in top_indices:

            # print("===========")
            #         # 打印信息
            # print(f"\nImage path: {sample['imgpath']}")
            # print(f"Best MOS score: {mos_scores[idx]:.4f} (index: {idx})")
            # print(f"Best bbox coordinates (normalized):")
            # print(f"  xmin: {bbox['xmin'][idx]:.4f}")
            # print(f"  ymin: {bbox['ymin'][idx]:.4f}")
            # print(f"  xmax: {bbox['xmax'][idx]:.4f}")
            # print(f"  ymax: {bbox['ymax'][idx]:.4f}")
            #         # 检查边界框是否合理
            # width = bbox['xmin'][idx] - bbox['xmax'][idx]
            # height = bbox['ymax'][idx] - bbox['ymin'][idx]
            # if width > 0.9 or height > 0.9:
            #     print(f"Warning: Unusually large bbox in {sample['imgpath']}")
            # else:
            #     print("===========")
            # # print(f"All MOS scores: {[f'{score:.4f}' for score in mos_scores]}")
            # print("===========")
            # print(image.shape)
            images.append(image)
            target = torch.tensor([
                float(bbox['xmin'][idx]) / ori_w, 
                float(bbox['ymin'][idx]) / ori_h,
                float(bbox['xmax'][idx]) / ori_w,
                float(bbox['ymax'][idx]) / ori_h
            ], dtype=torch.float32)
            target = torch.stack([
                torch.min(target[0], target[2]),  # xmin
                torch.min(target[1], target[3]),  # ymin
                torch.max(target[0], target[2]),  # xmax
                torch.max(target[1], target[3])   # ymax
            ])
            targets.append(target)

    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)

    merged_sample = {
        'image': images,
        'bbox': targets,
        'imgpath': imgpaths
    }
    return merged_sample

def custom_collate_best(batch):
    images = []
    targets = []
    imgpaths = []
    for sample in batch:
        image = torch.from_numpy(sample['image'])
        ori_h = image.shape[1]
        ori_w = image.shape[2]
        bbox = sample['bbox']
        mos_scores = sample['MOS']
        
        # 调整图像大小为固定尺寸 (3, 256, 256)
        if image.shape[1] != 256 or image.shape[2] != 256:
            m = nn.AdaptiveAvgPool2d((256, 256))
            image = m(image.unsqueeze(0)).squeeze(0)
            
        # 找到MOS最高的边界框索引
        best_idx = np.argmax(mos_scores)
        imgpaths.append(sample['imgpath'])
        images.append(image)
        target = torch.tensor([
            float(bbox['xmin'][best_idx]) / ori_w, 
            float(bbox['ymin'][best_idx]) / ori_h,
            float(bbox['xmax'][best_idx]) / ori_w,
            float(bbox['ymax'][best_idx]) / ori_h
        ], dtype=torch.float32)
        
        # 确保坐标的顺序正确（xmin < xmax, ymin < ymax）
        target = torch.stack([
            torch.min(target[0], target[2]),  # xmin
            torch.min(target[1], target[3]),  # ymin
            torch.max(target[0], target[2]),  # xmax
            torch.max(target[1], target[3])   # ymax
        ])
        targets.append(target)

    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)

    merged_sample = {
        'image': images,
        'targets': targets,
        'imgpath': imgpaths
    }
    # print(f"merged_sample: {merged_sample}")
    return merged_sample


train_loader = data.DataLoader(
    GAICD(
    image_size=args.image_size, 
    dataset_dir=args.dataset_root, 
    set='train', 
    augmentation=args.augmentation
),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=custom_collate_best
)
val_loader = data.DataLoader(
    GAICD(
    image_size=args.image_size, 
    dataset_dir=args.dataset_root, 
    set='val', 
    augmentation=args.augmentation
),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=custom_collate_best
)

test_loader = data.DataLoader(
    GAICD(
    image_size=args.image_size, 
    dataset_dir=args.dataset_root, 
    set='test', 
    augmentation=args.augmentation
),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=custom_collate_best
)

# 修改模型定义
class CropModel(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        if backbone_name not in BACKBONES:
            raise ValueError(f"Backbone {backbone_name} not supported. Choose from {list(BACKBONES.keys())}")
        
        # 获取backbone模型
        backbone = BACKBONES[backbone_name](pretrained=True)
        
        # 根据不同backbone调整最后的全连接层
        if backbone_name.startswith('resnet') or backbone_name.startswith('resnext'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('densenet'):
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('convnext'):
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('mnasnet'):
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('mobilenet'):
            if backbone_name == 'mobilenet_v2':
                in_features = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_features, 4)
            else:  # v3
                in_features = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_features, 4)
                
        elif backbone_name.startswith('regnet'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('shufflenet'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, 4)
            
        elif backbone_name.startswith('efficientnet'):
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, 4)
            
        elif backbone_name == 'vgg16':
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, 4)
            
        self.backbone = backbone
        # 添加dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        # 添加额外的全连接层
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 修改模型实例化
net = CropModel(args.backbone).to(args.device)

if args.test:
    # 加载模型状态字典
    checkpoint = torch.load(args.load_model)
    if 'model_state_dict' in checkpoint:
        net = torch.nn.DataParallel(net, device_ids=[0])
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)  # 如果直接保存的是状态字典
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = net.cuda()
    print(f"Loaded model from {args.load_model}")

if cuda and not args.test:
    net = torch.nn.DataParallel(net, device_ids=[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = net.cuda()

# 添加验证函数
@torch.no_grad()
def validate(model, val_loader, device):
    """
    在验证集上评估模型
    """
    model.eval()
    val_loss = 0
    val_metrics = {
        'iou': 0,
        'center_distance': 0,
        'aspect_ratio_error': 0
    }
    
    for id, merged_sample in enumerate(tqdm(val_loader, desc="Processing validation images")):
        inputs = merged_sample['image'].to(device)
        targets = merged_sample['targets'].to(device)
        imgpaths = merged_sample['imgpath']
        
        # 前向传播
        outputs = model(inputs)
        # for i in range(len(outputs)):
            # print(f"{imgpaths[i]} outputs: {outputs[i]} targets: {targets[i]}")
        
        # 计算损失
        loss = combined_loss(outputs, targets)
        val_loss += loss.item() * inputs.size(0)
        
        # 计算评估指标
        metrics = calculate_metrics(outputs, targets)
        for k in metrics:
            val_metrics[k] += metrics[k] * inputs.size(0)
    
    # 计算平均值
    val_loss /= len(val_loader.dataset)
    for k in val_metrics:
        val_metrics[k] /= len(val_loader.dataset)
    
    return val_loss, val_metrics


def test():
    net.eval()

    # 创建输出目录
    bbox_dir = os.path.join(args.output_dir, 'bbox_coordinates')
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, merged_sample in enumerate(tqdm(test_loader, desc="Processing test images")):
            # 获取batch数据
            inputs = merged_sample['image'].to(args.device)
            targets = merged_sample['targets'].to(args.device)
            imgpaths = merged_sample['imgpath']
            
            # 获取模型预测
            outputs = net(inputs)
            
            # 将tensors移到CPU并转换为numpy数组以便处理
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # 处理batch中的每张图片
            for i in range(len(imgpaths)):
                # 获取图片名称
                imgname = os.path.splitext(os.path.basename(imgpaths[i]))[0]
                
                # 读取原始图像以获取尺寸
                orig_image = cv2.imread(imgpaths[i])
                if orig_image is None:
                    print(f"Warning: Could not read image {imgpaths[i]}")
                    continue
                    
                orig_h, orig_w = orig_image.shape[:2]
                
                # 获取当前图片的预测和目标
                pred = outputs[i]  # 预测坐标 [x1, y1, x2, y2]
                target = targets[i]  # 目标坐标 [x1, y1, x2, y2]
                
                # 转换为原始图像坐标
                pred_x1, pred_y1 = int(pred[0] * orig_w), int(pred[1] * orig_h)
                pred_x2, pred_y2 = int(pred[2] * orig_w), int(pred[3] * orig_h)
                
                gt_x1, gt_y1 = int(target[0] * orig_w), int(target[1] * orig_h)
                gt_x2, gt_y2 = int(target[2] * orig_w), int(target[3] * orig_h)
                
                # 保存预测的坐标
                txt_path = os.path.join(bbox_dir, f"{imgname}.txt")
                with open(txt_path, 'w') as f:
                    # 保存归一化坐标
                    f.write(f"{pred[0]:.6f},{pred[1]:.6f},{pred[2]:.6f},{pred[3]:.6f}\n")
                    # f.write(f"Target: {target[0]:.6f},{target[1]:.6f},{target[2]:.6f},{target[3]:.6f}\n")
                    # 保存原始坐标
                    # f.write(f"Predicted original: {pred_x1},{pred_y1},{pred_x2},{pred_y2}\n")
                    # f.write(f"Target original: {gt_x1},{gt_y1},{gt_x2},{gt_y2}\n")
                
                # 可视化结果
                vis_image = orig_image.copy()
                
                # 绘制预测框（红色）
                cv2.rectangle(vis_image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2)
                # 绘制真实框（绿色）
                cv2.rectangle(vis_image, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)
                
                # 计算并显示IoU
                # iou = calculate_iou(
                #     torch.tensor(pred).unsqueeze(0), 
                #     torch.tensor(target).unsqueeze(0)
                # ).item()
                
                # # 在图像上添加IoU文本
                # cv2.putText(vis_image, f'IoU: {iou:.4f}', 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                #            1, (255, 255, 255), 2)
                
                # 保存可视化结果
                vis_path = os.path.join(vis_dir, f"{imgname}.jpg")
                cv2.imwrite(vis_path, vis_image)



        
# 修改训练函数
def train():
    # 创建数据加载器
    # train_loader, val_loader = create_data_loaders(args)
    
    # 初始化模型
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        net.train()
        train_loss = 0.0
        batch_losses = []
        
        train_loader_with_progress = tqdm(train_loader, 
                                        desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", 
                                        unit="batch",
                                        total=len(train_loader))
        
        for batch_idx, merged_sample in enumerate(train_loader):
            inputs = merged_sample['image'].to(args.device)
            targets = merged_sample['targets'].to(args.device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            train_loss += current_loss * inputs.size(0)
            batch_losses.append(current_loss)
            
            # 记录每个batch的loss到wandb
            wandb.log({
                "batch_loss": current_loss,
                "batch": epoch * len(train_loader) + batch_idx
            })
            # 更新进度条信息
            train_loader_with_progress.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "Avg Loss": f"{np.mean(batch_losses):.4f}"
            })
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        print("=====validation======")
        val_loss, val_metrics = validate(net, val_loader, args.device)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_metrics['iou'],
            "val_center_distance": val_metrics['center_distance'],
            "val_aspect_ratio_error": val_metrics['aspect_ratio_error'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val IoU: {val_metrics["iou"]:.4f}')

        checkpoint_path = os.path.join(args.save_folder, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
        # wandb.save(checkpoint_path)

        # 保存最佳模型
        if val_loss < best_val_loss:
            # input("是否保存最佳模型？")
            best_val_loss = val_loss
            best_model_path = os.path.join(args.save_folder, f'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            print(f"epoch: {epoch+1}")
            # wandb.save(best_model_path)
            # print("=====test======")
            # test()

def calculate_iou(pred_boxes, target_boxes):
    """
    计算预测框和目标框之间的IoU
    
    Args:
        pred_boxes: 预测框 tensor, shape [N, 4] (x1, y1, x2, y2)
        target_boxes: 目标框 tensor, shape [N, 4] (x1, y1, x2, y2)
    
    Returns:
        IoU scores tensor, shape [N]
    """
    # 确保输入是正确的维度
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if target_boxes.dim() == 1:
        target_boxes = target_boxes.unsqueeze(0)

    # 计算交集的坐标
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    # 计算交集面积
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算预测框和目标框的面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    # 计算并集面积
    union = pred_area + target_area - intersection

    # 计算IoU
    iou = intersection / (union + 1e-6)  # 添加小值避免除零

    return iou

def calculate_metrics(pred_boxes, target_boxes):
    """
    计算多个评估指标
    """
    # 计算IoU
    iou = calculate_iou(pred_boxes, target_boxes)
    
    # 计算边界框中心点距离
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_distance = torch.sqrt(
            (pred_center_x - target_center_x) ** 2 + 
            (pred_center_y - target_center_y) ** 2
        ).mean()
    
    
    # 计算宽高比误差
    pred_width = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_height = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_width = target_boxes[:, 2] - target_boxes[:, 0]
    target_height = target_boxes[:, 3] - target_boxes[:, 1]
    
    pred_aspect_ratio = pred_width / (pred_height + 1e-6)
    target_aspect_ratio = target_width / (target_height + 1e-6)
    aspect_ratio_error = torch.abs(pred_aspect_ratio - target_aspect_ratio)

    print(f"IoU: {iou.mean().item():.4f}")
    print(f"Center Distance: {center_distance.mean().item():.4f}")
    print(f"aspect_ratio_error: {aspect_ratio_error.mean().item():.4f}")
        # 打印预测框和目标框的统计信息
    print("\n边界框统计:")
    print(f"  预测框: 宽度={pred_width.mean().item():.4f}, 高度={pred_height.mean().item():.4f}")
    print(f"  目标框: 宽度={target_width.mean().item():.4f}, 高度={target_height.mean().item():.4f}")
    print(f"  宽高比: 预测={pred_aspect_ratio.mean().item():.4f}, 目标={target_aspect_ratio.mean().item():.4f}")

    return {
        'iou': iou.mean().item(),
        'center_distance': center_distance.mean().item(),
        'aspect_ratio_error': aspect_ratio_error.mean().item()
    }

# 修改combined_loss函数
def combined_loss(pred, target):
    """
    使用SmoothL1Loss作为损失函数，但仍然计算其他指标用于监控
    """
    # 主损失函数
    criterion = nn.SmoothL1Loss(beta=1.0)
    loss = criterion(pred, target)
    
    # 计算其他指标（仅用于监控，不参与反向传播）
    with torch.no_grad():
        # IoU
        iou = calculate_iou(pred, target).mean()
        
        # 中心点距离
        pred_center_x = (pred[:, 0] + pred[:, 2]) / 2
        pred_center_y = (pred[:, 1] + pred[:, 3]) / 2
        target_center_x = (target[:, 0] + target[:, 2]) / 2
        target_center_y = (target[:, 1] + target[:, 3]) / 2
        center_distance = torch.sqrt(
            (pred_center_x - target_center_x) ** 2 + 
            (pred_center_y - target_center_y) ** 2
        ).mean()
        
        # 打印所有指标
        # print(f"SmoothL1 Loss: {loss:.4f}")
        # print(f"IoU: {iou:.4f}")
        # print(f"Center Distance: {center_distance:.4f}")
    
    return loss

if __name__ == '__main__':
    if not args.test:
        wandb.init(
            project="image-cropping",  # 项目名称
            name=f"train_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}",  # 添加时间戳
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "backbone": args.backbone,  # 添加backbone到wandb配置
                "image_size": args.image_size,
                "optimizer": "Adam"
            }
        )

        # 记录模型架构图
        wandb.watch(net, log="all", log_freq=100)
        train()
        wandb.finish()
# 测试模型
    if args.test:
        test()
    # 示例推理
    # inference('crop_model.pth', 'test_image.jpg', 'cropped_result.jpg')