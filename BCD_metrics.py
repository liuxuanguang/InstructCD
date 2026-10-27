# import numpy as np
# import os
# import re
# from skimage import io
# from sklearn.metrics import cohen_kappa_score
# from collections import defaultdict
# from PIL import Image
#
#
# def calculate_change_metrics(y_true, y_pred):
#     """
#     计算建筑物变化检测的精度指标。
#
#     参数:
#         y_true: 真实标签，单通道灰度图，变化建筑物为1，背景为0
#         y_pred: 预测标签，PIL三通道图像，通道含义：
#                通道0: 新增建筑物（从非建筑物变为建筑物）
#                通道1: 消失建筑物（从建筑物变为非建筑物）
#                通道2: 未变化区域
#                前两个通道合并为总体变化
#     返回:
#         字典，包含总体变化的精度指标
#     """
#     # 确保输入为整数类型
#     y_true = y_true.astype(int)
#
#     # 处理预测结果：从PIL三通道中提取总体变化
#     if isinstance(y_pred, Image.Image):
#         # 将PIL图像转换为numpy数组
#         y_pred_array = np.array(y_pred)
#     else:
#         y_pred_array = y_pred
#
#     # 确保是三通道
#     if len(y_pred_array.shape) == 2:
#         # 如果是单通道，复制到3个通道
#         y_pred_array = np.stack([y_pred_array, y_pred_array, y_pred_array], axis=-1)
#     elif y_pred_array.shape[-1] > 3:
#         # 如果通道数大于3，取前三个通道
#         y_pred_array = y_pred_array[:, :, :3]
#
#     # 从三通道中提取前两个通道，并合并为总体变化
#     # 注意：PIL图像可能是RGB顺序，但我们需要将其视为三个独立的通道
#     y_pred_changes = (y_pred_array[:, :, 0] + y_pred_array[:, :, 1]) > 0
#
#     # 确保标签是单通道
#     if len(y_true.shape) == 3:
#         if y_true.shape[-1] == 3:
#             # 如果是三通道，取第一通道作为变化区域
#             y_true_changes = y_true[:, :, 0] > 0
#         else:
#             # 如果通道数不是3，直接取第一个通道
#             y_true_changes = y_true[:, :, 0] > 0
#     else:
#         y_true_changes = y_true > 0
#
#     # 展平数组
#     true_flat = y_true_changes.flatten().astype(int)
#     pred_flat = y_pred_changes.flatten().astype(int)
#
#     # 计算混淆矩阵元素
#     tp = np.sum((true_flat == 1) & (pred_flat == 1))
#     fp = np.sum((true_flat == 0) & (pred_flat == 1))
#     fn = np.sum((true_flat == 1) & (pred_flat == 0))
#     tn = np.sum((true_flat == 0) & (pred_flat == 0))
#
#     # 计算指标
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#     iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
#
#     # 计算kappa系数
#     kappa = cohen_kappa_score(true_flat, pred_flat)
#
#     # 计算总体准确率
#     accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
#
#     metrics = {
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'iou': iou,
#         'kappa': kappa,
#         'accuracy': accuracy,
#         'tp': tp,
#         'fp': fp,
#         'fn': fn,
#         'tn': tn
#     }
#
#     return metrics
#
#
# def find_matching_true_file(pred_filename, true_dir, true_extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
#     """
#     根据预测文件名查找对应的真实标签文件。
#
#     参数:
#         pred_filename: 预测文件名
#         true_dir: 真实标签目录
#         true_extensions: 真实标签文件可能的扩展名
#
#     返回:
#         匹配的真实标签文件路径，如果未找到则返回None
#     """
#     # 从预测文件名中提取基本名称（去掉扩展名）
#     base_name = os.path.splitext(pred_filename)[0]
#
#     # 常见的命名模式
#     patterns_to_try = []
#
#     # 1. 尝试直接匹配（相同文件名）
#     patterns_to_try.append(base_name)
#
#     # 2. 尝试移除常见的预测文件后缀
#     for suffix in ['_pred', '_prediction', '_change', '_result', '_mask', '_output']:
#         if base_name.endswith(suffix):
#             patterns_to_try.append(base_name[:-len(suffix)])
#
#     # 3. 尝试匹配包含特定模式的名称
#     # 例如，预测文件名为"area1_pred.png"，标签文件名为"area1.png"
#     if '_pred' in base_name:
#         patterns_to_try.append(base_name.replace('_pred', ''))
#     if '_change' in base_name:
#         patterns_to_try.append(base_name.replace('_change', ''))
#     if '_result' in base_name:
#         patterns_to_try.append(base_name.replace('_result', ''))
#
#     # 4. 尝试从文件名中提取数字或字母序列
#     # 例如，预测文件名为"pred_001.png"，标签文件名为"gt_001.png"
#     # 查找文件名中的数字
#     numbers = re.findall(r'\d+', base_name)
#     if numbers:
#         patterns_to_try.append(''.join(numbers))  # 纯数字模式
#
#     # 5. 尝试常见的前缀/后缀组合
#     for prefix in ['gt_', 'label_', 'mask_', 'true_', 'ground_truth_']:
#         patterns_to_try.append(f"{prefix}{base_name}")
#         # 如果base_name已经有前缀，尝试替换
#         if base_name.startswith('pred_'):
#             patterns_to_try.append(base_name.replace('pred_', prefix))
#         if base_name.startswith('result_'):
#             patterns_to_try.append(base_name.replace('result_', prefix))
#
#     # 在真实标签目录中查找匹配的文件
#     for true_file in os.listdir(true_dir):
#         true_name, true_ext = os.path.splitext(true_file)
#
#         # 检查文件扩展名
#         if true_ext.lower() not in true_extensions:
#             continue
#
#         # 尝试所有模式
#         for pattern in patterns_to_try:
#             if true_name == pattern or true_name.startswith(pattern) or true_name.endswith(pattern):
#                 return os.path.join(true_dir, true_file)
#
#         # 额外检查：如果真实文件名包含预测文件名的基本部分
#         if base_name in true_name or true_name in base_name:
#             return os.path.join(true_dir, true_file)
#
#     return None
#
#
# def evaluate_batch_change_detection(pred_dir, true_dir, threshold=128, match_by_prefix=False):
#     """
#     批量评估建筑物变化检测结果。
#
#     参数:
#         pred_dir: 预测掩码图像目录路径（PIL三通道）
#         true_dir: 真实掩码图像目录路径（单通道灰度图）
#         threshold: 二值化阈值（仅当图像非二值时使用）
#         match_by_prefix: 如果为True，则通过文件名前缀匹配预测和标签文件
#
#     返回:
#         metrics_list: 每个图像的指标字典列表
#         avg_metrics: 平均指标字典
#         confusion_matrix_sum: 总体混淆矩阵字典
#     """
#     # 获取目录中的图像文件名
#     pred_files = sorted([f for f in os.listdir(pred_dir)
#                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
#
#     if not pred_files:
#         raise ValueError(f"预测目录中没有找到图像文件: {pred_dir}")
#
#     metrics_list = []
#     matched_pairs = []
#     unmatched_pred_files = []
#
#     print(f"找到 {len(pred_files)} 个预测文件")
#
#     for pred_file in pred_files:
#         pred_path = os.path.join(pred_dir, pred_file)
#
#         # 根据不同的匹配策略查找对应的真实标签文件
#         if match_by_prefix:
#             # 通过文件名前缀匹配
#             base_name = os.path.splitext(pred_file)[0]
#             # 尝试找到对应的真实标签文件
#             true_files = [f for f in os.listdir(true_dir)
#                           if os.path.splitext(f)[0].startswith(base_name) or
#                           base_name.startswith(os.path.splitext(f)[0])]
#
#             if not true_files:
#                 print(f"警告: 无法为预测文件 {pred_file} 找到匹配的真实标签文件")
#                 continue
#
#             # 如果有多个匹配，选择第一个
#             true_file = true_files[0]
#             true_path = os.path.join(true_dir, true_file)
#         else:
#             # 使用智能匹配函数
#             true_path = find_matching_true_file(pred_file, true_dir)
#
#             if true_path is None:
#                 print(f"警告: 无法为预测文件 {pred_file} 找到匹配的真实标签文件")
#                 unmatched_pred_files.append(pred_file)
#                 continue
#
#         if not os.path.exists(true_path):
#             print(f"警告: 真实标签文件不存在: {true_path}")
#             continue
#
#         print(f"匹配: {pred_file} -> {os.path.basename(true_path)}")
#         matched_pairs.append((pred_file, os.path.basename(true_path)))
#
#         try:
#             # 读取图像
#             # 预测图像使用PIL读取（假设是RGB三通道）
#             pred = Image.open(pred_path)
#             # 真实标签使用skimage读取（单通道灰度图）
#             true = io.imread(true_path)
#
#             # 处理真实标签：如果最大值大于1，进行阈值处理转换为二值
#             if true.max() > 1:
#                 true = (true > threshold).astype(int)
#
#             # 处理预测结果：将PIL图像转换为numpy数组
#             pred_array = np.array(pred)
#
#             # 检查图像尺寸是否一致
#             if pred_array.shape[:2] != true.shape[:2]:
#                 print(f"警告: 图像 {pred_file} 和 {os.path.basename(true_path)} 尺寸不一致!")
#                 # 调整尺寸以匹配
#                 if true.ndim == 2:
#                     true_resized = np.array(Image.fromarray(true).resize(
#                         (pred_array.shape[1], pred_array.shape[0]),
#                         resample=Image.NEAREST
#                     ))
#                 else:
#                     true_resized = np.array(Image.fromarray(true).resize(
#                         (pred_array.shape[1], pred_array.shape[0]),
#                         resample=Image.NEAREST
#                     ))[:, :, 0] if true.shape[2] >= 3 else true[:, :, 0]
#                 true = true_resized
#
#             # 计算指标
#             metrics = calculate_change_metrics(true, pred)
#             metrics['pred_file'] = pred_file
#             metrics['true_file'] = os.path.basename(true_path)
#             metrics_list.append(metrics)
#
#         except Exception as e:
#             print(f"处理文件 {pred_file} 时出错: {e}")
#
#     if not metrics_list:
#         raise ValueError("没有成功匹配和评估任何文件对!")
#
#     print(f"\n成功匹配 {len(metrics_list)} 对文件")
#     if unmatched_pred_files:
#         print(f"未匹配的预测文件: {unmatched_pred_files}")
#
#     # 计算平均指标
#     avg_metrics = defaultdict(float)
#     # 对于混淆矩阵元素，计算总和而不是平均值
#     confusion_matrix_sum = defaultdict(int)
#     for metric_name in metrics_list[0].keys():
#         if metric_name in ['tp', 'fp', 'fn', 'tn']:
#             for metrics in metrics_list:
#                 confusion_matrix_sum[metric_name] += metrics[metric_name]
#         elif metric_name not in ['pred_file', 'true_file']:  # 跳过文件名字段
#             values = [metrics[metric_name] for metrics in metrics_list]
#             avg_metrics[metric_name] = np.mean(values)
#
#     # 根据总体混淆矩阵重新计算宏观平均指标
#     tp_total = confusion_matrix_sum['tp']
#     fp_total = confusion_matrix_sum['fp']
#     fn_total = confusion_matrix_sum['fn']
#     tn_total = confusion_matrix_sum['tn']
#
#     precision_macro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
#     recall_macro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
#     f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro) if (precision_macro + recall_macro) > 0 else 0.0
#     iou_macro = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
#     accuracy_macro = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total) if (tp_total + fp_total + fn_total + tn_total) > 0 else 0.0
#
#     avg_metrics['precision_macro'] = precision_macro
#     avg_metrics['recall_macro'] = recall_macro
#     avg_metrics['f1_macro'] = f1_macro
#     avg_metrics['iou_macro'] = iou_macro
#     avg_metrics['accuracy_macro'] = accuracy_macro
#
#     return metrics_list, dict(avg_metrics), confusion_matrix_sum, matched_pairs
#
#
# def save_change_metrics_to_txt(metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs=None,
#                                output_file="change_detection_metrics.txt"):
#     """
#     将建筑物变化检测精度指标保存到txt文件。
#
#     参数:
#         metrics_list: 每个图像的指标字典列表
#         avg_metrics: 平均指标字典
#         confusion_matrix_sum: 总体混淆矩阵字典
#         matched_pairs: 匹配的文件对列表
#         output_file: 输出文件路径
#     """
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write("建筑物变化检测结果精度评价\n")
#         f.write("=" * 60 + "\n")
#
#         # 记录匹配的文件对
#         if matched_pairs:
#             f.write(f"文件匹配对 (共 {len(matched_pairs)} 对):\n")
#             for i, (pred_file, true_file) in enumerate(matched_pairs):
#                 f.write(f"  {i + 1:3d}. {pred_file} -> {true_file}\n")
#             f.write("\n")
#
#         f.write("每个图像的指标:\n")
#         f.write("-" * 100 + "\n")
#
#         # 表头
#         headers = ["序号", "预测文件", "标签文件", "精度", "召回率", "F1", "IOU", "Kappa", "准确率", "TP", "FP", "FN",
#                    "TN"]
#         f.write("{:<5} {:<30} {:<30} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6} {:<6} {:<6} {:<6}\n".format(*headers))
#         f.write("-" * 100 + "\n")
#
#         # 每个图像的数据
#         for i, metrics in enumerate(metrics_list):
#             row = [
#                 f"{i + 1}",
#                 metrics.get('pred_file', 'N/A')[:28],
#                 metrics.get('true_file', 'N/A')[:28],
#                 f"{metrics['precision']:.4f}",
#                 f"{metrics['recall']:.4f}",
#                 f"{metrics['f1']:.4f}",
#                 f"{metrics['iou']:.4f}",
#                 f"{metrics['kappa']:.4f}",
#                 f"{metrics['accuracy']:.4f}",
#                 f"{int(metrics['tp'])}",
#                 f"{int(metrics['fp'])}",
#                 f"{int(metrics['fn'])}",
#                 f"{int(metrics['tn'])}"
#             ]
#             f.write("{:<5} {:<30} {:<30} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6} {:<6} {:<6} {:<6}\n".format(*row))
#
#         f.write("\n" + "=" * 60 + "\n")
#         f.write("宏观平均指标（基于总体混淆矩阵）:\n")
#         f.write("=" * 60 + "\n")
#
#         # 宏观平均指标（基于总体混淆矩阵计算）
#         f.write(f"精确度: {avg_metrics['precision_macro']:.4f}\n")
#         f.write(f"召回率: {avg_metrics['recall_macro']:.4f}\n")
#         f.write(f"F1分数: {avg_metrics['f1_macro']:.4f}\n")
#         f.write(f"IOU: {avg_metrics['iou_macro']:.4f}\n")
#         f.write(f"准确率: {avg_metrics['accuracy_macro']:.4f}\n")
#
#         # 总体混淆矩阵统计
#         f.write(f"\n总体混淆矩阵统计:\n")
#         f.write(f"  TP（真阳性）: {confusion_matrix_sum['tp']}\n")
#         f.write(f"  FP（假阳性）: {confusion_matrix_sum['fp']}\n")
#         f.write(f"  FN（假阴性）: {confusion_matrix_sum['fn']}\n")
#         f.write(f"  TN（真阴性）: {confusion_matrix_sum['tn']}\n")
#
#         # 计算总体统计
#         total = confusion_matrix_sum['tp'] + confusion_matrix_sum['fp'] + confusion_matrix_sum['fn'] + \
#                 confusion_matrix_sum['tn']
#         f.write(f"\n总体统计:\n")
#         f.write(f"  总像素数: {total}\n")
#         f.write(f"  变化像素比例: {(confusion_matrix_sum['tp'] + confusion_matrix_sum['fn']) / total * 100:.2f}%\n")
#         f.write(f"  检测变化像素比例: {(confusion_matrix_sum['tp'] + confusion_matrix_sum['fp']) / total * 100:.2f}%\n")
#
#     print(f"建筑物变化检测精度结果已保存到 {output_file}")
#
#
# # 主程序示例
# if __name__ == "__main__":
#     # 设置预测掩码和真实掩码的目录路径（请根据实际路径修改）
#     pred_dir = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/LEVIR-CD/LEVIR_CD_test/binary_results/final_changes"
#     true_dir = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/test/label"
#
#     # 如果预测文件和标签文件有相同的前缀，可以设置 match_by_prefix=True
#     # 否则，代码会自动尝试匹配
#     match_by_prefix = False  # 根据实际情况调整
#
#     # 评估批量变化检测结果
#     try:
#         metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs = evaluate_batch_change_detection(
#             pred_dir, true_dir, match_by_prefix=match_by_prefix
#         )
#
#         # 保存结果到txt文件
#         output_path = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/LEVIR-CD/LEVIR_CD_test/binary_results/change_detection_metrics.txt"
#         save_change_metrics_to_txt(metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs, output_path)
#
#         print("\n宏观平均指标:")
#         print(f"精确度: {avg_metrics['precision_macro']:.4f}")
#         print(f"召回率: {avg_metrics['recall_macro']:.4f}")
#         print(f"F1分数: {avg_metrics['f1_macro']:.4f}")
#         print(f"IOU: {avg_metrics['iou_macro']:.4f}")
#         print(f"准确率: {avg_metrics['accuracy_macro']:.4f}")
#
#         print("\n总体混淆矩阵统计:")
#         print(f"TP（真阳性）: {confusion_matrix_sum['tp']}")
#         print(f"FP（假阳性）: {confusion_matrix_sum['fp']}")
#         print(f"FN（假阴性）: {confusion_matrix_sum['fn']}")
#         print(f"TN（真阴性）: {confusion_matrix_sum['tn']}")
#
#     except Exception as e:
#         print(f"评估过程中出错: {e}")
#         import traceback
#
#         traceback.print_exc()

import numpy as np
import os
import re
from skimage import io
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
from PIL import Image


def calculate_change_metrics(y_true, y_pred):
    """
    计算建筑物变化检测的精度指标。

    参数:
        y_true: 真实标签，单通道灰度图，变化建筑物为1，背景为0
        y_pred: 预测标签，PIL三通道图像，通道含义：
               通道0: 新增建筑物（从非建筑物变为建筑物）
               通道1: 消失建筑物（从建筑物变为非建筑物）
               通道2: 未变化区域
               前两个通道合并为总体变化
    返回:
        字典，包含总体变化的精度指标
    """
    # 确保输入为整数类型
    y_true = y_true.astype(int)

    # 处理预测结果：从PIL三通道中提取总体变化
    if isinstance(y_pred, Image.Image):
        # 将PIL图像转换为numpy数组
        y_pred_array = np.array(y_pred)
    else:
        y_pred_array = y_pred

    # 确保是三通道
    if len(y_pred_array.shape) == 2:
        # 如果是单通道，复制到3个通道
        y_pred_array = np.stack([y_pred_array, y_pred_array, y_pred_array], axis=-1)
    elif y_pred_array.shape[-1] > 3:
        # 如果通道数大于3，取前三个通道
        y_pred_array = y_pred_array[:, :, :3]

    # 从三通道中提取前两个通道，并合并为总体变化
    # 注意：PIL图像可能是RGB顺序，但我们需要将其视为三个独立的通道
    y_pred_changes = (y_pred_array[:, :, 0] + y_pred_array[:, :, 1]) > 0

    # 确保标签是单通道
    if len(y_true.shape) == 3:
        if y_true.shape[-1] == 3:
            # 如果是三通道，取第一通道作为变化区域
            y_true_changes = y_true[:, :, 0] > 0
        else:
            # 如果通道数不是3，直接取第一个通道
            y_true_changes = y_true[:, :, 0] > 0
    else:
        y_true_changes = y_true > 0

    # 展平数组
    true_flat = y_true_changes.flatten().astype(int)
    pred_flat = y_pred_changes.flatten().astype(int)

    # 计算混淆矩阵元素
    tp = np.sum((true_flat == 1) & (pred_flat == 1))
    fp = np.sum((true_flat == 0) & (pred_flat == 1))
    fn = np.sum((true_flat == 1) & (pred_flat == 0))
    tn = np.sum((true_flat == 0) & (pred_flat == 0))

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # 计算kappa系数
    kappa = cohen_kappa_score(true_flat, pred_flat)

    # 计算总体准确率
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'kappa': kappa,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    return metrics


def find_matching_true_file(pred_filename, true_dir, true_extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    """
    根据预测文件名查找对应的真实标签文件。
    专门针对格式: test_数字_final.png -> test_数字.png

    参数:
        pred_filename: 预测文件名
        true_dir: 真实标签目录
        true_extensions: 真实标签文件可能的扩展名

    返回:
        匹配的真实标签文件路径，如果未找到则返回None
    """
    # 从预测文件名中提取基本名称（去掉扩展名）
    base_name = os.path.splitext(pred_filename)[0]

    # 尝试提取数字
    # 模式1: test_数字_final
    match1 = re.match(r'test_(\d+)_final$', base_name)
    if match1:
        number = match1.group(1)
        # 直接构造真实文件名: test_数字.png
        for ext in true_extensions:
            true_path = os.path.join(true_dir, f"test_{number}{ext}")
            if os.path.exists(true_path):
                return true_path

        # 如果没找到，尝试test_数字.tif等
        for true_file in os.listdir(true_dir):
            true_name, true_ext = os.path.splitext(true_file)
            if true_ext.lower() in true_extensions:
                # 尝试匹配test_数字
                if re.match(fr'test_{number}$', true_name):
                    return os.path.join(true_dir, true_file)

    # 模式2: 文件名包含数字
    numbers = re.findall(r'\d+', base_name)
    if numbers:
        # 取最后一个数字（通常是最相关的）
        number = numbers[-1]
        # 尝试直接匹配数字
        for true_file in os.listdir(true_dir):
            true_name, true_ext = os.path.splitext(true_file)
            if true_ext.lower() in true_extensions:
                # 检查文件名是否以test_数字开头或以数字结尾
                if true_name == f"test_{number}" or true_name.endswith(f"_{number}") or f"_{number}." in true_file:
                    return os.path.join(true_dir, true_file)

    # 模式3: 尝试移除_final后缀
    if base_name.endswith('_final'):
        base_without_final = base_name[:-6]  # 移除'_final'
        for ext in true_extensions:
            true_path = os.path.join(true_dir, f"{base_without_final}{ext}")
            if os.path.exists(true_path):
                return true_path

    # 模式4: 尝试其他常见模式
    patterns_to_try = [
        base_name.replace('_final', ''),
        base_name.replace('_pred', ''),
        base_name.replace('_change', ''),
        base_name.replace('_result', ''),
        base_name.replace('_mask', ''),
        base_name.replace('_output', ''),
    ]

    for pattern in patterns_to_try:
        for ext in true_extensions:
            true_path = os.path.join(true_dir, f"{pattern}{ext}")
            if os.path.exists(true_path):
                return true_path

    # 如果以上都没找到，尝试文件名包含关系
    for true_file in os.listdir(true_dir):
        true_name, true_ext = os.path.splitext(true_file)
        if true_ext.lower() in true_extensions:
            # 检查文件名是否包含预测文件名的基本部分
            if base_name.replace('_final', '') in true_name or true_name in base_name:
                return os.path.join(true_dir, true_file)

    return None


def extract_number_from_filename(filename):
    """
    从文件名中提取数字

    参数:
        filename: 文件名

    返回:
        提取到的数字列表
    """
    return re.findall(r'\d+', filename)


def find_matching_by_number(pred_filename, true_dir, true_extensions):
    """
    通过数字匹配预测文件和真实文件
    """
    pred_numbers = extract_number_from_filename(pred_filename)
    if not pred_numbers:
        return None

    for true_file in os.listdir(true_dir):
        true_ext = os.path.splitext(true_file)[1].lower()
        if true_ext in true_extensions:
            true_numbers = extract_number_from_filename(true_file)
            if true_numbers and pred_numbers == true_numbers:
                return os.path.join(true_dir, true_file)

    return None


# def evaluate_batch_change_detection(pred_dir, true_dir, threshold=128, match_by_prefix=False):
#     """
#     批量评估建筑物变化检测结果。
#
#     参数:
#         pred_dir: 预测掩码图像目录路径（PIL三通道）
#         true_dir: 真实掩码图像目录路径（单通道灰度图）
#         threshold: 二值化阈值（仅当图像非二值时使用）
#         match_by_prefix: 如果为True，则通过文件名前缀匹配预测和标签文件
#
#     返回:
#         metrics_list: 每个图像的指标字典列表
#         avg_metrics: 平均指标字典
#         confusion_matrix_sum: 总体混淆矩阵字典
#     """
#     # 获取目录中的图像文件名
#     pred_files = sorted([f for f in os.listdir(pred_dir)
#                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
#
#     if not pred_files:
#         raise ValueError(f"预测目录中没有找到图像文件: {pred_dir}")
#
#     # 获取真实标签文件列表
#     true_files = sorted([f for f in os.listdir(true_dir)
#                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
#
#     print(f"找到 {len(pred_files)} 个预测文件")
#     print(f"找到 {len(true_files)} 个真实标签文件")
#
#     # 先尝试通过数字匹配
#     print("\n尝试通过数字匹配文件...")
#
#     # 为每个预测文件提取数字
#     pred_files_with_numbers = []
#     for pred_file in pred_files:
#         numbers = extract_number_from_filename(pred_file)
#         if numbers:
#             pred_files_with_numbers.append((pred_file, numbers[0]))  # 取第一个数字
#         else:
#             pred_files_with_numbers.append((pred_file, None))
#
#     # 为每个真实文件提取数字
#     true_files_with_numbers = []
#     for true_file in true_files:
#         numbers = extract_number_from_filename(true_file)
#         if numbers:
#             true_files_with_numbers.append((true_file, numbers[0]))  # 取第一个数字
#         else:
#             true_files_with_numbers.append((true_file, None))
#
#     metrics_list = []
#     matched_pairs = []
#     unmatched_pred_files = []
#
#     # 首先尝试通过数字匹配
#     for pred_file, pred_number in pred_files_with_numbers:
#         if pred_number is None:
#             continue
#
#         # 查找相同数字的真实文件
#         matching_true_files = []
#         for true_file, true_number in true_files_with_numbers:
#             if true_number == pred_number:
#                 matching_true_files.append(true_file)
#
#         if len(matching_true_files) == 1:
#             # 找到唯一匹配
#             true_file = matching_true_files[0]
#             true_path = os.path.join(true_dir, true_file)
#             print(f"数字匹配: {pred_file} -> {true_file}")
#             matched_pairs.append((pred_file, true_file))
#
#             # 评估
#             try:
#                 metrics = evaluate_single_pair(pred_file, true_path, pred_dir, true_dir, threshold)
#                 if metrics:
#                     metrics_list.append(metrics)
#             except Exception as e:
#                 print(f"处理文件 {pred_file} 时出错: {e}")
#                 unmatched_pred_files.append(pred_file)
#         elif len(matching_true_files) > 1:
#             print(f"警告: 预测文件 {pred_file} 匹配到多个真实文件: {matching_true_files}")
#             unmatched_pred_files.append(pred_file)
#         else:
#             # 没有通过数字匹配到，使用原来的匹配函数
#             true_path = find_matching_true_file(pred_file, true_dir)
#             if true_path and os.path.exists(true_path):
#                 true_file = os.path.basename(true_path)
#                 print(f"智能匹配: {pred_file} -> {true_file}")
#                 matched_pairs.append((pred_file, true_file))
#
#                 try:
#                     metrics = evaluate_single_pair(pred_file, true_path, pred_dir, true_dir, threshold)
#                     if metrics:
#                         metrics_list.append(metrics)
#                 except Exception as e:
#                     print(f"处理文件 {pred_file} 时出错: {e}")
#                     unmatched_pred_files.append(pred_file)
#             else:
#                 print(f"警告: 无法为预测文件 {pred_file} 找到匹配的真实标签文件")
#                 unmatched_pred_files.append(pred_file)
#
#     if not metrics_list:
#         raise ValueError("没有成功匹配和评估任何文件对!")
#
#     print(f"\n成功匹配 {len(metrics_list)} 对文件")
#     if unmatched_pred_files:
#         print(f"未匹配的预测文件: {unmatched_pred_files}")
#
#     # 计算平均指标
#     avg_metrics = defaultdict(float)
#     # 对于混淆矩阵元素，计算总和而不是平均值
#     confusion_matrix_sum = defaultdict(int)
#     for metric_name in metrics_list[0].keys():
#         if metric_name in ['tp', 'fp', 'fn', 'tn']:
#             for metrics in metrics_list:
#                 confusion_matrix_sum[metric_name] += metrics[metric_name]
#         elif metric_name not in ['pred_file', 'true_file']:  # 跳过文件名字段
#             values = [metrics[metric_name] for metrics in metrics_list]
#             avg_metrics[metric_name] = np.mean(values)
#
#     # 根据总体混淆矩阵重新计算宏观平均指标
#     tp_total = confusion_matrix_sum['tp']
#     fp_total = confusion_matrix_sum['fp']
#     fn_total = confusion_matrix_sum['fn']
#     tn_total = confusion_matrix_sum['tn']
#
#     precision_macro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
#     recall_macro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
#     f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro) if (
#                                                                                                       precision_macro + recall_macro) > 0 else 0.0
#     iou_macro = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
#     accuracy_macro = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total) if (
#                                                                                                         tp_total + fp_total + fn_total + tn_total) > 0 else 0.0
#
#     avg_metrics['precision_macro'] = precision_macro
#     avg_metrics['recall_macro'] = recall_macro
#     avg_metrics['f1_macro'] = f1_macro
#     avg_metrics['iou_macro'] = iou_macro
#     avg_metrics['accuracy_macro'] = accuracy_macro
#
#     return metrics_list, dict(avg_metrics), confusion_matrix_sum, matched_pairs


def evaluate_batch_change_detection(pred_dir, true_dir, threshold=128, match_by_prefix=False,
                                   building_colors=None):
    """
    批量评估建筑物变化检测结果。

    参数:
        pred_dir: 预测掩码图像目录路径（PIL三通道）
        true_dir: 真实掩码图像目录路径（单通道灰度图 或 RGB三通道图）
        threshold: 二值化阈值（仅当图像非二值时使用）
        match_by_prefix: 如果为True，则通过文件名前缀匹配预测和标签文件
        building_colors: 可选, list of (R,G,B) tuples.
                        当GT为RGB三通道时，仅保留指定颜色的像素为"建筑物变化"，
                        其余颜色视为背景/非建筑变化。

    返回:
        metrics_list: 每个图像的指标字典列表
        avg_metrics: 平均指标字典
        confusion_matrix_sum: 总体混淆矩阵字典
    """
    # 获取目录中的图像文件名
    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not pred_files:
        raise ValueError(f"预测目录中没有找到图像文件: {pred_dir}")

    # 获取真实标签文件列表
    true_files = sorted([f for f in os.listdir(true_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    print(f"找到 {len(pred_files)} 个预测文件")
    print(f"找到 {len(true_files)} 个真实标签文件")

    # 创建真实文件的查找字典
    true_file_dict = {os.path.splitext(f)[0]: f for f in true_files}

    metrics_list = []
    matched_pairs = []
    unmatched_pred_files = []

    # 处理每个预测文件
    for pred_file in pred_files:
        # 获取基本文件名（不带扩展名）
        pred_base = os.path.splitext(pred_file)[0]

        # 移除_final后缀
        if pred_base.endswith('_final'):
            true_base = pred_base[:-6]  # 移除'_final'
        else:
            # 尝试其他可能的后缀
            for suffix in ['_pred', '_change', '_result', '_mask']:
                if pred_base.endswith(suffix):
                    true_base = pred_base[:-len(suffix)]
                    break
            else:
                true_base = pred_base

        # 尝试查找对应的真实文件
        true_file = None

        # 方法1: 直接匹配基本名称
        if true_base in true_file_dict:
            true_file = true_file_dict[true_base]

        # 方法2: 尝试移除下划线后的数字
        if not true_file and '_' in true_base:
            # 对于类似test_94_000的格式，先尝试整个字符串匹配
            # 如果不行，再尝试test_94_000
            parts = true_base.split('_')
            for i in range(len(parts), 0, -1):
                test_base = '_'.join(parts[:i])
                if test_base in true_file_dict:
                    true_file = true_file_dict[test_base]
                    break

        # 如果找到了匹配的真实文件
        if true_file:
            true_path = os.path.join(true_dir, true_file)
            print(f"精确匹配: {pred_file} -> {true_file}")
            matched_pairs.append((pred_file, true_file))

            try:
                # 评估单个文件对
                metrics = evaluate_single_pair(pred_file, true_path, pred_dir, true_dir,
                                              threshold, building_colors=building_colors)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"处理文件 {pred_file} 时出错: {e}")
                unmatched_pred_files.append(pred_file)
        else:
            print(f"警告: 无法为预测文件 {pred_file} 找到匹配的真实标签文件")
            unmatched_pred_files.append(pred_file)

    if not metrics_list:
        raise ValueError("没有成功匹配和评估任何文件对!")

    print(f"\n成功匹配 {len(metrics_list)} 对文件")
    if unmatched_pred_files:
        print(f"未匹配的预测文件: {unmatched_pred_files}")

    # 计算平均指标
    avg_metrics = defaultdict(float)
    # 对于混淆矩阵元素，计算总和而不是平均值
    confusion_matrix_sum = defaultdict(int)
    for metric_name in metrics_list[0].keys():
        if metric_name in ['tp', 'fp', 'fn', 'tn']:
            for metrics in metrics_list:
                confusion_matrix_sum[metric_name] += metrics[metric_name]
        elif metric_name not in ['pred_file', 'true_file']:  # 跳过文件名字段
            values = [metrics[metric_name] for metrics in metrics_list]
            avg_metrics[metric_name] = np.mean(values)

    # 根据总体混淆矩阵重新计算宏观平均指标
    tp_total = confusion_matrix_sum['tp']
    fp_total = confusion_matrix_sum['fp']
    fn_total = confusion_matrix_sum['fn']
    tn_total = confusion_matrix_sum['tn']

    precision_macro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall_macro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro) if (
                                                                                                      precision_macro + recall_macro) > 0 else 0.0
    iou_macro = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
    accuracy_macro = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total) if (
                                                                                                        tp_total + fp_total + fn_total + tn_total) > 0 else 0.0

    avg_metrics['precision_macro'] = precision_macro
    avg_metrics['recall_macro'] = recall_macro
    avg_metrics['f1_macro'] = f1_macro
    avg_metrics['iou_macro'] = iou_macro
    avg_metrics['accuracy_macro'] = accuracy_macro

    return metrics_list, dict(avg_metrics), confusion_matrix_sum, matched_pairs

def filter_change_labels_by_color(true_rgb, building_colors):
    """
    从RGB变化标签中仅保留指定建筑物颜色，将其他变化类别置为背景。

    参数:
        true_rgb: np.array (H, W, 3), RGB三通道真实标签
        building_colors: list of tuple, 建筑物变化对应的RGB颜色
                        例如: [(128, 0, 0), (0, 128, 0)] 表示红色和暗绿色

    返回:
        np.array (H, W), 二值掩码: 1=建筑物变化, 0=其他(背景/非建筑变化)
    """
    H, W = true_rgb.shape[:2]
    building_mask = np.zeros((H, W), dtype=np.uint8)

    for color in building_colors:
        color_array = np.array(color, dtype=np.uint8)
        match = np.all(true_rgb == color_array.reshape(1, 1, 3), axis=2)
        building_mask[match] = 1

    return building_mask


def evaluate_single_pair(pred_file, true_path, pred_dir, true_dir, threshold,
                         building_colors=None):
    """评估单个文件对

    参数:
        building_colors: 可选, list of (R,G,B) tuples, 仅保留指定颜色的变化为建筑物变化
    """
    pred_path = os.path.join(pred_dir, pred_file)

    # 读取图像
    pred = Image.open(pred_path)
    true = io.imread(true_path)

    # ---- 建筑物颜色过滤 (在二值化之前) ----
    if building_colors is not None and len(true.shape) == 3 and true.shape[-1] >= 3:
        # RGB标签: 仅保留指定建筑物颜色
        true = filter_change_labels_by_color(true[:, :, :3], building_colors)
    elif true.max() > 1:
        # 灰度/索引标签: 标准阈值处理
        true = (true > threshold).astype(int)

    # 处理预测结果：将PIL图像转换为numpy数组
    pred_array = np.array(pred)

    # 检查图像尺寸是否一致
    if pred_array.shape[:2] != true.shape[:2]:
        print(f"警告: 图像 {pred_file} 和 {os.path.basename(true_path)} 尺寸不一致!")
        # 调整尺寸以匹配
        if true.ndim == 2:
            true_resized = np.array(Image.fromarray(true).resize(
                (pred_array.shape[1], pred_array.shape[0]),
                resample=Image.NEAREST
            ))
        else:
            true_resized = np.array(Image.fromarray(true).resize(
                (pred_array.shape[1], pred_array.shape[0]),
                resample=Image.NEAREST
            ))[:, :, 0] if true.shape[2] >= 3 else true[:, :, 0]
        true = true_resized

    # 计算指标
    metrics = calculate_change_metrics(true, pred)
    metrics['pred_file'] = pred_file
    metrics['true_file'] = os.path.basename(true_path)

    return metrics


def save_change_metrics_to_txt(metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs=None,
                               output_file="change_detection_metrics.txt"):
    """
    将建筑物变化检测精度指标保存到txt文件。

    参数:
        metrics_list: 每个图像的指标字典列表
        avg_metrics: 平均指标字典
        confusion_matrix_sum: 总体混淆矩阵字典
        matched_pairs: 匹配的文件对列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("建筑物变化检测结果精度评价\n")
        f.write("=" * 60 + "\n")

        # 记录匹配的文件对
        if matched_pairs:
            f.write(f"文件匹配对 (共 {len(matched_pairs)} 对):\n")
            for i, (pred_file, true_file) in enumerate(matched_pairs):
                f.write(f"  {i + 1:3d}. {pred_file} -> {true_file}\n")
            f.write("\n")

        f.write("每个图像的指标:\n")
        f.write("-" * 100 + "\n")

        # 表头
        headers = ["序号", "预测文件", "标签文件", "精度", "召回率", "F1", "IOU", "Kappa", "准确率", "TP", "FP", "FN",
                   "TN"]
        f.write("{:<5} {:<30} {:<30} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6} {:<6} {:<6} {:<6}\n".format(*headers))
        f.write("-" * 100 + "\n")

        # 每个图像的数据
        for i, metrics in enumerate(metrics_list):
            row = [
                f"{i + 1}",
                metrics.get('pred_file', 'N/A')[:28],
                metrics.get('true_file', 'N/A')[:28],
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['iou']:.4f}",
                f"{metrics['kappa']:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{int(metrics['tp'])}",
                f"{int(metrics['fp'])}",
                f"{int(metrics['fn'])}",
                f"{int(metrics['tn'])}"
            ]
            f.write("{:<5} {:<30} {:<30} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6} {:<6} {:<6} {:<6}\n".format(*row))

        f.write("\n" + "=" * 60 + "\n")
        f.write("宏观平均指标（基于总体混淆矩阵）:\n")
        f.write("=" * 60 + "\n")

        # 宏观平均指标（基于总体混淆矩阵计算）
        f.write(f"精确度: {avg_metrics['precision_macro']:.4f}\n")
        f.write(f"召回率: {avg_metrics['recall_macro']:.4f}\n")
        f.write(f"F1分数: {avg_metrics['f1_macro']:.4f}\n")
        f.write(f"IOU: {avg_metrics['iou_macro']:.4f}\n")
        f.write(f"准确率: {avg_metrics['accuracy_macro']:.4f}\n")

        # 总体混淆矩阵统计
        f.write(f"\n总体混淆矩阵统计:\n")
        f.write(f"  TP（真阳性）: {confusion_matrix_sum['tp']}\n")
        f.write(f"  FP（假阳性）: {confusion_matrix_sum['fp']}\n")
        f.write(f"  FN（假阴性）: {confusion_matrix_sum['fn']}\n")
        f.write(f"  TN（真阴性）: {confusion_matrix_sum['tn']}\n")

        # 计算总体统计
        total = confusion_matrix_sum['tp'] + confusion_matrix_sum['fp'] + confusion_matrix_sum['fn'] + \
                confusion_matrix_sum['tn']
        f.write(f"\n总体统计:\n")
        f.write(f"  总像素数: {total}\n")
        f.write(f"  变化像素比例: {(confusion_matrix_sum['tp'] + confusion_matrix_sum['fn']) / total * 100:.2f}%\n")
        f.write(f"  检测变化像素比例: {(confusion_matrix_sum['tp'] + confusion_matrix_sum['fp']) / total * 100:.2f}%\n")

    print(f"建筑物变化检测精度结果已保存到 {output_file}")


def evaluate_with_exact_match(pred_dir, true_dir, threshold=128):
    """
    使用精确匹配模式评估建筑物变化检测结果
    专门针对格式: test_数字_final.png -> test_数字.png
    """
    # 获取预测文件列表
    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not pred_files:
        raise ValueError(f"预测目录中没有找到图像文件: {pred_dir}")

    metrics_list = []
    matched_pairs = []
    unmatched_files = []

    print(f"找到 {len(pred_files)} 个预测文件")

    for pred_file in pred_files:
        # 提取预测文件中的数字
        # pred_match = re.search(r'test_(\d+)_final', pred_file)
        pred_match = re.search(r'test_(\d+)_(\d+)_final', pred_file)
        if not pred_match:
            print(f"警告: 预测文件 {pred_file} 不符合 test_数字_final.png 格式")
            unmatched_files.append(pred_file)
            continue

        pred_number = pred_match.group(1)

        # 构造对应的真实文件名
        true_filename = f"test_{pred_number}.png"
        true_path = os.path.join(true_dir, true_filename)

        if not os.path.exists(true_path):
            # 尝试其他扩展名
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                true_path = os.path.join(true_dir, f"test_{pred_number}{ext}")
                if os.path.exists(true_path):
                    found = True
                    true_filename = f"test_{pred_number}{ext}"
                    break

            if not found:
                print(f"警告: 未找到预测文件 {pred_file} 对应的真实标签文件")
                unmatched_files.append(pred_file)
                continue

        print(f"匹配: {pred_file} -> {true_filename}")
        matched_pairs.append((pred_file, true_filename))

        try:
            # 评估单个文件对
            metrics = evaluate_single_pair(pred_file, true_path, pred_dir, true_dir, threshold)
            if metrics:
                metrics_list.append(metrics)
        except Exception as e:
            print(f"处理文件 {pred_file} 时出错: {e}")
            unmatched_files.append(pred_file)

    if not metrics_list:
        raise ValueError("没有成功匹配和评估任何文件对!")

    print(f"\n成功匹配 {len(metrics_list)} 对文件")
    if unmatched_files:
        print(f"未匹配的预测文件: {unmatched_files}")

    # 计算平均指标
    avg_metrics = defaultdict(float)
    confusion_matrix_sum = defaultdict(int)

    for metric_name in metrics_list[0].keys():
        if metric_name in ['tp', 'fp', 'fn', 'tn']:
            for metrics in metrics_list:
                confusion_matrix_sum[metric_name] += metrics[metric_name]
        elif metric_name not in ['pred_file', 'true_file']:
            values = [metrics[metric_name] for metrics in metrics_list]
            avg_metrics[metric_name] = np.mean(values)

    # 根据总体混淆矩阵重新计算宏观平均指标
    tp_total = confusion_matrix_sum['tp']
    fp_total = confusion_matrix_sum['fp']
    fn_total = confusion_matrix_sum['fn']
    tn_total = confusion_matrix_sum['tn']

    precision_macro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall_macro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro) if (
                                                                                                      precision_macro + recall_macro) > 0 else 0.0
    iou_macro = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
    accuracy_macro = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total) if (
                                                                                                        tp_total + fp_total + fn_total + tn_total) > 0 else 0.0

    avg_metrics['precision_macro'] = precision_macro
    avg_metrics['recall_macro'] = recall_macro
    avg_metrics['f1_macro'] = f1_macro
    avg_metrics['iou_macro'] = iou_macro
    avg_metrics['accuracy_macro'] = accuracy_macro

    return metrics_list, dict(avg_metrics), confusion_matrix_sum, matched_pairs


# 主程序示例
if __name__ == "__main__":
    # 设置预测掩码和真实掩码的目录路径（请根据实际路径修改）
    pred_dir = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/multi_class_perturbation/bare_land/lambda_p0p20/binary_results/initial_changes"
    true_dir = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/bareland_label"

    # 方法1: 使用精确匹配（针对 test_数字_final.png -> test_数字.png 格式）
    try:
        print("使用精确匹配模式...")
        metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs = evaluate_with_exact_match(
            pred_dir, true_dir, threshold=128
        )

        # 保存结果到txt文件
        output_path = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/multi_class_perturbation/bare_land/lambda_p0p20/binary_results/initial_changes/changes_metrics.txt"
        save_change_metrics_to_txt(metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs, output_path)

        print("\n宏观平均指标:")
        print(f"精确度: {avg_metrics['precision_macro']:.4f}")
        print(f"召回率: {avg_metrics['recall_macro']:.4f}")
        print(f"F1分数: {avg_metrics['f1_macro']:.4f}")
        print(f"IOU: {avg_metrics['iou_macro']:.4f}")
        print(f"准确率: {avg_metrics['accuracy_macro']:.4f}")

        print("\n总体混淆矩阵统计:")
        print(f"TP（真阳性）: {confusion_matrix_sum['tp']}")
        print(f"FP（假阳性）: {confusion_matrix_sum['fp']}")
        print(f"FN（假阴性）: {confusion_matrix_sum['fn']}")
        print(f"TN（真阴性）: {confusion_matrix_sum['tn']}")

    except Exception as e:
        print(f"精确匹配模式评估过程中出错: {e}")

        # 如果精确匹配失败，尝试使用智能匹配
        print("\n尝试使用智能匹配模式...")
        try:
            metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs = evaluate_batch_change_detection(
                pred_dir, true_dir, match_by_prefix=False
            )

            # 保存结果到txt文件
            output_path = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/multi_class_perturbation/bare_land/lambda_p0p20/binary_results/initial_changes/changes_metrics.txt"
            save_change_metrics_to_txt(metrics_list, avg_metrics, confusion_matrix_sum, matched_pairs, output_path)

            print("\n宏观平均指标:")
            print(f"精确度: {avg_metrics['precision_macro']:.4f}")
            print(f"召回率: {avg_metrics['recall_macro']:.4f}")
            print(f"F1分数: {avg_metrics['f1_macro']:.4f}")
            print(f"IOU: {avg_metrics['iou_macro']:.4f}")
            print(f"准确率: {avg_metrics['accuracy_macro']:.4f}")

            print("\n总体混淆矩阵统计:")
            print(f"TP（真阳性）: {confusion_matrix_sum['tp']}")
            print(f"FP（假阳性）: {confusion_matrix_sum['fp']}")
            print(f"FN（假阴性）: {confusion_matrix_sum['fn']}")
            print(f"TN（真阴性）: {confusion_matrix_sum['tn']}")

        except Exception as e2:
            print(f"智能匹配模式评估过程中出错: {e2}")
            import traceback

            traceback.print_exc()
