import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
from scipy import ndimage
import time
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from TextTransV3 import enhanced_CDTT
from SurperPixelSegmentation import dual_temporal_hierarchical_segmentation as FineSeg
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import mode
from skimage.morphology import skeletonize, thin
import math
import gc
import glob
import shutil
from scipy.ndimage import gaussian_filter
from AdaptiveThresholdManager import AdaptiveThresholdManagerV2

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def compute_overlap(mask1, mask2):
    return np.logical_and(mask1, mask2)


def refine_overlapping_masks(masks_data, class_scores, class_probs, h, w, iou_threshold=0.2):
    n_masks = len(masks_data)
    refined_masks = []
    refined_classes = []
    refined_scores = []
    if n_masks == 0:
        return refined_masks, refined_classes, refined_scores
    iou_matrix = np.zeros((n_masks, n_masks))
    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            mask_i = masks_data[i][1]
            mask_j = masks_data[j][1]
            iou_matrix[i, j] = compute_iou(mask_i, mask_j)
            iou_matrix[j, i] = iou_matrix[i, j]
    adjacency = {}
    for i in range(n_masks):
        overlapping_masks = []
        for j in range(n_masks):
            if i != j and iou_matrix[i, j] > iou_threshold:
                overlapping_masks.append(j)
        adjacency[i] = overlapping_masks
    processed = [False] * n_masks
    for i in range(n_masks):
        if processed[i]:
            continue
        class_id_i, mask_i, score_i, prob_map_i = masks_data[i]
        overlapping = adjacency[i]
        if not overlapping:
            refined_masks.append(mask_i)
            refined_classes.append(class_id_i)
            refined_scores.append(score_i)
            processed[i] = True
        else:
            overlapping_group = [i] + overlapping
            for idx in overlapping:
                if idx < len(processed):
                    processed[idx] = True

            group_masks = []
            group_classes = []
            group_scores = []
            group_probs = []

            for idx in overlapping_group:
                if idx < len(masks_data):
                    class_id, mask, score, prob_map = masks_data[idx]
                    group_masks.append(mask)
                    group_classes.append(class_id)
                    group_scores.append(score)
                    if prob_map is not None and prob_map.shape == (h, w):
                        group_probs.append(prob_map)
                    else:
                        group_probs.append(mask.astype(np.float32))

            if not group_probs:
                refined_masks.append(mask_i)
                refined_classes.append(class_id_i)
                refined_scores.append(score_i)
                processed[i] = True
                continue

            n_group = len(group_masks)
            if n_group == 0:
                continue
            elif n_group == 1:
                refined_masks.append(mask_i)
                refined_classes.append(class_id_i)
                refined_scores.append(score_i)
                processed[i] = True
                continue

            overlap_matrix = np.zeros((n_group, n_group))
            for a in range(n_group):
                for b in range(a + 1, n_group):
                    overlap = compute_overlap(group_masks[a], group_masks[b])
                    overlap_matrix[a, b] = overlap.any()
                    overlap_matrix[b, a] = overlap_matrix[a, b]

            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in group_masks:
                combined_mask = np.logical_or(combined_mask, mask)

            labeled, num_features = ndimage.label(combined_mask)

            for label_idx in range(1, num_features + 1):
                component_mask = (labeled == label_idx)

                overlapping_indices = []
                overlapping_scores = []

                for mask_idx, mask in enumerate(group_masks):
                    if compute_iou(mask, component_mask) > 0.1:  # 有显著重叠
                        overlapping_indices.append(mask_idx)
                        overlapping_scores.append(group_scores[mask_idx])

                if len(overlapping_indices) == 0:
                    continue
                elif len(overlapping_indices) == 1:
                    mask_idx = overlapping_indices[0]
                    refined_masks.append(component_mask)
                    refined_classes.append(group_classes[mask_idx])
                    refined_scores.append(group_scores[mask_idx])
                else:
                    best_score = -1
                    best_idx = -1
                    for i, (mask_idx, score) in enumerate(zip(overlapping_indices, overlapping_scores)):
                        if score > best_score:
                            best_score = score
                            best_idx = mask_idx

                    if best_idx >= 0:
                        refined_masks.append(component_mask)
                        refined_classes.append(group_classes[best_idx])
                        refined_scores.append(group_scores[best_idx])
                    else:
                        mask_idx = overlapping_indices[0]
                        refined_masks.append(component_mask)
                        refined_classes.append(group_classes[mask_idx])
                        refined_scores.append(group_scores[mask_idx])

    return refined_masks, refined_classes, refined_scores


def multi_class_semantic_segmentation(image, prompts, colors=None, output_dir=None,
                                      high_processor=None, low_processor=None, model=None,
                                      adaptive_threshold_manager=None):
    """
    多类别语义分割函数，使用自适应阈值策略

    新增参数:
        adaptive_threshold_manager: 自适应阈值管理器实例（如果提供则使用自适应阈值）
    """
    if colors is None:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(image, np.ndarray):
        image_np = image
        h, w = image_np.shape[:2]
    else:
        image_np = np.array(image)
        h, w = image_np.shape[:2]

    if model is None:
        model = build_sam3_image_model()

    # 初始化存储
    all_masks = []
    all_class_ids = []
    all_scores = []
    all_probs = []
    class_probs_maps = []
    class_max_probs = []
    class_total_masks = []
    class_instance_data = [[] for _ in range(len(prompts))]

    # 处理每个类别
    for class_id, (class_name, class_color) in enumerate(zip(prompts, colors)):
        print(f"处理类别 {class_id + 1}/{len(prompts)}: {class_name}")

        # ============ 自适应阈值确定 ============
        if adaptive_threshold_manager is not None:
            # 使用自适应阈值管理器
            thresholds = adaptive_threshold_manager.get_adaptive_threshold(
                image, class_name)
            class_high_threshold = thresholds['high_threshold']
            class_low_threshold = thresholds['low_threshold']

            print(f"  自适应阈值: high={class_high_threshold:.4f}, low={class_low_threshold:.4f}")


            # 创建类别专用处理器
            class_high_processor = adaptive_threshold_manager.create_processor_with_threshold(
                class_high_threshold
            )
            class_low_processor = adaptive_threshold_manager.create_processor_with_threshold(
                class_low_threshold
            )
        else:
            # 使用传入的处理器（原有逻辑）
            class_high_processor = high_processor
            class_low_processor = low_processor

            if class_high_processor is None:
                class_high_processor = Sam3Processor(model, confidence_threshold=0.2)
            if class_low_processor is None:
                class_low_processor = Sam3Processor(model, confidence_threshold=0.1)

        # 1. 先用高阈值处理器处理
        print(f"  使用高阈值处理器处理...")
        high_inference_state = class_high_processor.set_image(image)
        high_inference_state = class_high_processor.set_text_prompt(
            state=high_inference_state.copy(),
            prompt=class_name
        )

        # 检查高阈值处理器的结果
        high_masks = None
        high_scores = None
        high_masks_logits = None
        high_num_instances = 0

        if 'masks' in high_inference_state and high_inference_state['masks'] is not None:
            high_masks = high_inference_state['masks']
            if hasattr(high_masks, 'shape'):
                high_num_instances = high_masks.shape[0]

        if 'scores' in high_inference_state and high_inference_state['scores'] is not None:
            high_scores = high_inference_state['scores']

        if 'masks_logits' in high_inference_state and high_inference_state['masks_logits'] is not None:
            high_masks_logits = high_inference_state['masks_logits']

        print(f"  高阈值处理器检测到 {high_num_instances} 个实例")

        # 2. 根据检测数量决定使用哪个处理器的结果
        current_state = None
        masks = None
        scores = None
        masks_logits = None
        num_instances = 0
        processor_used = ""

        if high_num_instances > 0:
            print(f"  实例数量{high_num_instances} > 0，使用高阈值处理器结果")
            current_state = high_inference_state
            masks = high_masks
            scores = high_scores
            masks_logits = high_masks_logits
            num_instances = high_num_instances
            processor_used = f"高阈值({class_high_threshold if adaptive_threshold_manager else 0.2:.3f})"
        else:
            print(f"  实例数量{high_num_instances} ≤ 0，使用低阈值处理器重新处理...")
            low_inference_state = class_low_processor.set_image(image)
            low_inference_state = class_low_processor.set_text_prompt(
                state=low_inference_state.copy(),
                prompt=class_name
            )

            low_masks = None
            low_scores = None
            low_masks_logits = None
            low_num_instances = 0

            if 'masks' in low_inference_state and low_inference_state['masks'] is not None:
                low_masks = low_inference_state['masks']
                if hasattr(low_masks, 'shape'):
                    low_num_instances = low_masks.shape[0]

            if 'scores' in low_inference_state and low_inference_state['scores'] is not None:
                low_scores = low_inference_state['scores']

            if 'masks_logits' in low_inference_state and low_inference_state['masks_logits'] is not None:
                low_masks_logits = low_inference_state['masks_logits']

            print(f"  低阈值处理器检测到 {low_num_instances} 个实例")

            if low_num_instances > 0:
                current_state = low_inference_state
                masks = low_masks
                scores = low_scores
                masks_logits = low_masks_logits
                num_instances = low_num_instances
                processor_used = f"低阈值({class_low_threshold if adaptive_threshold_manager else 0.1:.3f})"
            else:
                print(f"  低阈值处理器也未检测到实例")
                current_state = high_inference_state
                masks = high_masks
                scores = high_scores
                masks_logits = high_masks_logits
                num_instances = high_num_instances
                processor_used = "回退"

        print(f"  类别'{class_name}'使用{processor_used}，检测到{num_instances}个实例")

        # 3. 处理分割结果
        if current_state is not None and 'masks' in current_state and current_state['masks'] is not None:
            masks = current_state['masks']
            if hasattr(masks, 'shape') and masks.shape[0] > 0:
                if 'scores' in current_state and current_state['scores'] is not None:
                    scores = current_state['scores']
                    if hasattr(scores, 'mean'):
                        avg_score = scores.mean().item()
                    else:
                        avg_score = 0.5
                else:
                    scores = torch.ones(masks.shape[0]) * 0.5
                    avg_score = 0.5

                if 'masks_logits' in current_state and current_state['masks_logits'] is not None:
                    masks_logits = current_state['masks_logits']
                    if masks_logits.dim() == 4:
                        masks_logits = masks_logits.squeeze(1)
                    masks_probs = masks_logits
                    max_instance_probs, _ = torch.max(masks_probs, dim=0)
                    if max_instance_probs.ndim > 2:
                        max_instance_probs = max_instance_probs.squeeze()
                    class_probs_maps.append(max_instance_probs.cpu().numpy())
                    class_max_probs.append(max_instance_probs.cpu().numpy())
                else:
                    masks_probs = None
                    class_probs_maps.append(np.zeros((h, w), dtype=np.float32))
                    class_max_probs.append(np.zeros((h, w), dtype=np.float32))

                for instance_idx in range(masks.shape[0]):
                    mask = masks[instance_idx]
                    if scores is not None and instance_idx < len(scores):
                        score = scores[instance_idx].item() if hasattr(scores[instance_idx], 'item') else scores[instance_idx]
                    else:
                        score = 0.5

                    if mask.dim() == 2:
                        mask_np = mask.cpu().numpy().astype(bool)
                    elif mask.dim() == 3:
                        mask_np = mask.squeeze().cpu().numpy().astype(bool)
                    elif mask.dim() == 4:
                        mask_np = mask.squeeze().cpu().numpy().astype(bool)
                    else:
                        continue

                    if mask_np.shape != (h, w):
                        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                        mask_resized = F.interpolate(
                            mask_tensor.float(), size=(h, w), mode='nearest'
                        ).squeeze().numpy().astype(bool)
                    else:
                        mask_resized = mask_np

                    if np.any(mask_resized):
                        all_masks.append(mask_resized)
                        all_class_ids.append(class_id)
                        all_scores.append(score)

                        if masks_probs is not None and instance_idx < masks_probs.shape[0]:
                            instance_prob = masks_probs[instance_idx].cpu().numpy()
                            if instance_prob.shape != (h, w):
                                if instance_prob.ndim == 2:
                                    instance_prob_tensor = torch.from_numpy(instance_prob).unsqueeze(0).unsqueeze(0)
                                elif instance_prob.ndim == 3:
                                    instance_prob_tensor = torch.from_numpy(instance_prob).unsqueeze(0)
                                else:
                                    instance_prob_tensor = torch.from_numpy(instance_prob)
                                instance_prob_resized = F.interpolate(
                                    instance_prob_tensor.float(), size=(h, w),
                                    mode='bilinear', align_corners=False
                                )
                                if instance_prob_resized.shape[0] == 1 and instance_prob_resized.shape[1] == 1:
                                    instance_prob_resized = instance_prob_resized.squeeze().numpy()
                                else:
                                    instance_prob_resized = instance_prob_resized[0, 0, :, :].numpy()
                            else:
                                instance_prob_resized = instance_prob
                            all_probs.append(instance_prob_resized)
                        else:
                            all_probs.append(mask_resized.astype(np.float32))

                        class_instance_data[class_id].append({
                            'mask': mask_resized,
                            'score': score,
                            'prob': instance_prob_resized if masks_probs is not None else mask_resized.astype(np.float32)
                        })

                if len(class_instance_data[class_id]) > 0:
                    class_combined_mask = np.zeros((h, w), dtype=bool)
                    for inst_data in class_instance_data[class_id]:
                        class_combined_mask = np.logical_or(class_combined_mask, inst_data['mask'])
                    class_total_masks.append(class_combined_mask)
                else:
                    class_total_masks.append(np.zeros((h, w), dtype=bool))
            else:
                class_probs_maps.append(np.zeros((h, w), dtype=np.float32))
                class_max_probs.append(np.zeros((h, w), dtype=np.float32))
                class_total_masks.append(np.zeros((h, w), dtype=bool))
        else:
            class_probs_maps.append(np.zeros((h, w), dtype=np.float32))
            class_max_probs.append(np.zeros((h, w), dtype=np.float32))
            class_total_masks.append(np.zeros((h, w), dtype=bool))

    # 计算每个类别的平均分数
    class_avg_scores = []
    for class_id in range(len(prompts)):
        if len(class_instance_data[class_id]) > 0:
            scores_list = [inst['score'] for inst in class_instance_data[class_id]]
            class_avg_scores.append(np.mean(scores_list))
        else:
            class_avg_scores.append(0.0)

    masks_data_for_refinement = []
    for i, (mask, class_id, score, prob) in enumerate(zip(all_masks, all_class_ids, all_scores, all_probs)):
        masks_data_for_refinement.append((class_id, mask, score, prob))

    refined_masks, refined_classes, refined_scores = refine_overlapping_masks(
        masks_data_for_refinement,
        class_avg_scores,
        np.stack(class_probs_maps) if len(class_probs_maps) > 0 else None,
        h, w,
        iou_threshold=0.3
    )

    # 重新组织实例（与原代码相同）
    final_class_masks = []
    final_instance_masks_per_class = [[] for _ in range(len(prompts))]
    final_scores_per_class = [[] for _ in range(len(prompts))]
    detection_counts = [0] * len(prompts)
    confidence_scores = [0.0] * len(prompts)

    semantic_map = np.zeros((h, w), dtype=np.uint8)
    instance_map = np.zeros((h, w), dtype=np.uint32)
    instance_counter = 1

    for i, (mask, class_id, score) in enumerate(zip(refined_masks, refined_classes, refined_scores)):
        instance_id = instance_counter
        instance_counter += 1
        mask_indices = np.where(mask)
        semantic_map[mask_indices] = class_id + 1
        instance_map[mask_indices] = instance_id
        final_instance_masks_per_class[class_id].append(mask)
        final_scores_per_class[class_id].append(score)

    for class_id in range(len(prompts)):
        if final_instance_masks_per_class[class_id]:
            class_mask = np.zeros((h, w), dtype=bool)
            for mask in final_instance_masks_per_class[class_id]:
                class_mask = np.logical_or(class_mask, mask)
            final_class_masks.append(class_mask)
            detection_counts[class_id] = len(final_instance_masks_per_class[class_id])
            confidence_scores[class_id] = np.mean(final_scores_per_class[class_id]) if final_scores_per_class[class_id] else 0.0
        else:
            final_class_masks.append(np.zeros((h, w), dtype=bool))

    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(prompts)):
        if detection_counts[class_id] > 0:
            mask = final_class_masks[class_id]
            if class_id < len(colors):
                colored_seg[mask] = colors[class_id]

    # ── 清理所有 GPU 变量，防止显存泄漏 ──
    try: del high_inference_state, low_inference_state
    except NameError: pass
    try: del class_high_processor, class_low_processor
    except NameError: pass
    try: del all_masks, all_class_ids, all_scores, all_probs, class_instance_data
    except NameError: pass
    try: del final_instance_masks_per_class, final_scores_per_class
    except NameError: pass
    try: del masks_data_for_refinement
    except NameError: pass
    torch.cuda.empty_cache()
    gc.collect()

    return semantic_map, colored_seg, final_class_masks, class_probs_maps, class_max_probs


class SequentialMultiClassChangeDetection:
    """按类别顺序处理的多类别变化检测"""

    # 定义固定的类别索引和颜色
    CLASS_INDEX = {
        'bare land': 1,
        'low vegetation': 2,
        'building': 3,
        'water': 4,
        'tree': 5,
        'sports ground': 6
    }

    # 定义类别RGB颜色
    CLASS_COLORS = {
        'bare land': [128, 128, 128],
        'low vegetation': [0, 128, 0],
        'building': [128, 0, 0],
        'water': [0, 0, 255],
        'tree': [0, 255, 0],
        'sports ground': [255, 0, 0]
    }

    def __init__(self, image1_path, image2_path, instruction_text=None,
                 output_dir="sequential_change_results",
                 use_adaptive_threshold=True,
                 model=None):

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 加载SAM3模型（支持外部复用）
        if model is not None:
            self.model = model
        else:
            print("=" * 60)
            print("1. 加载SAM3模型...")
            self.model = build_sam3_image_model()

        # 创建自适应阈值管理器
        self.use_adaptive_threshold = use_adaptive_threshold
        if use_adaptive_threshold:
            print("  ✓ 启用自适应阈值管理器")
            self.adaptive_threshold_manager = AdaptiveThresholdManagerV2(
                model=self.model,
                processor_class=Sam3Processor,
                probe_threshold=0.01,  # 极低试探阈值（尽可能捕获全部候选）
                anti_miss_bias=0.7,  # 抗漏检偏置（0.7表示强烈倾向于不漏检）
                min_coverage_ratio=0.01,  # 最小覆盖率期望
                max_retry=3,  # 最大重试次数
                enable_feedback=True,  # 启用反馈验证
                verbose=True
            )
            # 不再需要固定阈值的处理器
            self.processor_high = None
            self.processor_low = None
        else:
            print("  使用固定阈值处理器")
            self.adaptive_threshold_manager = None
            self.processor_high = Sam3Processor(self.model, confidence_threshold=0.2)
            self.processor_low = Sam3Processor(self.model, confidence_threshold=0.1)

        print("✓ SAM3模型加载完成")

        # 2. 加载影像
        print("\n2. 加载双时相影像...")
        self.image1 = Image.open(image1_path).convert('RGB')
        self.image2 = Image.open(image2_path).convert('RGB')
        self.image1_np = np.array(self.image1)
        self.image2_np = np.array(self.image2)

        # 调整尺寸一致
        if self.image1.size != self.image2.size:
            print(f"  调整影像尺寸一致: 时相1={self.image1.size}, 时相2={self.image2.size}")
            h, w = min(self.image1.size[1], self.image2.size[1]), min(self.image1.size[0], self.image2.size[0])
            self.image1 = self.image1.resize((w, h))
            self.image2 = self.image2.resize((w, h))
            self.image1_np = np.array(self.image1)
            self.image2_np = np.array(self.image2)

        self.height, self.width = self.image1_np.shape[:2]
        print(f"✓ 影像加载完成: 尺寸={self.width}x{self.height}")

        # 3. 指令转换
        print("\n3. 进行指令转换...")
        if instruction_text is None:
            instruction_text = ['building change']  # 默认指令

        # 初始化翻译专家（仅离线模式，不需要API Key）
        # expert = InstructionTranslationExpert(
        #     llm_api_key='sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau',  # 不使用LLM
        #     enable_online=True )
        # trans_results = expert.translate(instruction_text)

        trans_results = enhanced_CDTT(instruction_text)
        self.trans_results = trans_results  # 保存指令转换结果

        # 提取变化语义信息
        self.change_semantics = None
        self.change_direction = None
        self.transitions = []
        self.involved_classes = []

        if trans_results and len(trans_results) > 0:
            trans_item = trans_results[0]
            # 优先使用 semantics 中的信息
            if 'semantics' in trans_item:
                semantics_info = trans_item['semantics']
                self.change_semantics = semantics_info.get('change_semantics', 'all_changes')
                self.change_direction = semantics_info.get('change_direction', 'bidirectional')
                self.transitions = semantics_info.get('transitions', [])
                self.involved_classes = semantics_info.get('involved_classes', [])

        # 如果没有involved_classes，从指令中提取
        if not self.involved_classes:
            self.involved_classes = []
            for trans_item in trans_results:
                phase1_instr = trans_item['data'].get("phase1_instruction", "")
                phase2_instr = trans_item['data'].get("phase2_instruction", "")

                # 解析指令获取类别
                classes_t1 = self._parse_classes(phase1_instr)
                classes_t2 = self._parse_classes(phase2_instr)

                # 合并所有类别
                for cls in classes_t1 + classes_t2:
                    if cls and cls not in self.involved_classes:
                        self.involved_classes.append(cls)

        # 动态生成CLASS_INDEX
        self.CLASS_INDEX = {}
        for i, class_name in enumerate(self.involved_classes, 1):
            self.CLASS_INDEX[class_name] = i

        # 动态生成CLASS_COLORS
        self.CLASS_COLORS = {}
        # 预设一些基础颜色
        base_colors = [
            [128, 128, 128],  # 灰色 - bare land
            [0, 128, 0],  # 深绿色 - low vegetation
            [128, 0, 0],  # 深红色 - building
            [0, 0, 255],  # 蓝色 - water
            [0, 255, 0],  # 绿色 - tree
            [255, 0, 0],  # 红色 - sports ground
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 紫色
            [0, 255, 255],  # 青色
            [128, 0, 128],  # 紫色
        ]

        # 为每个类别分配颜色
        for i, class_name in enumerate(self.involved_classes):
            if i < len(base_colors):
                self.CLASS_COLORS[class_name] = base_colors[i]
            else:
                # 如果没有足够的预设颜色，生成随机颜色
                np.random.seed(i + 42)  # 固定种子以确保一致性
                self.CLASS_COLORS[class_name] = [int(x) for x in np.random.randint(0, 256, 3)]

        print(f"✓ 动态生成了 {len(self.involved_classes)} 个类别:")
        for i, class_name in enumerate(self.involved_classes, 1):
            class_idx = self.CLASS_INDEX[class_name]
            color = self.CLASS_COLORS[class_name]
            print(f"  类别{i}: {class_name} (索引={class_idx}, 颜色={color})")

        # 设置all_classes为involved_classes
        self.all_classes = self.involved_classes.copy()

        # 4. 为每个类别准备颜色
        self.colors = []
        for cls in self.all_classes:
            if cls in self.CLASS_COLORS:
                self.colors.append(self.CLASS_COLORS[cls])
            else:
                # 如果没有颜色，使用随机颜色
                np.random.seed(hash(cls) % 1000)
                self.colors.append([int(x) for x in np.random.randint(0, 256, 3)])

    def _parse_classes(self, instruction):
        """解析指令中的类别"""
        if not instruction or instruction.strip() == "":
            return []

        # 去除多余空格
        instruction = instruction.strip()

        # 使用多种分隔符分割
        separators = [';', ',', '、', '和', '与']

        for sep in separators:
            if sep in instruction:
                parts = [p.strip() for p in instruction.split(sep) if p.strip()]
                if len(parts) > 0:
                    return parts

        # 如果没有分隔符，返回整个指令
        return [instruction]

    def convex_hull_separation(self, binary_image, min_defect_depth=20, min_area=50):
        """
        使用凸包检测算法分割粘连对象

        Args:
            binary_image: 二值图像
            min_defect_depth: 凸缺陷最小深度阈值
            min_area: 最小面积阈值
        """
        # 确保输入是uint8类型
        if binary_image.dtype != np.uint8:
            if binary_image.dtype == bool:
                binary_image = (binary_image * 255).astype(np.uint8)
            else:
                binary_image = binary_image.astype(np.uint8)

        # 确保是0/255格式
        if np.max(binary_image) <= 1:
            binary_image = (binary_image * 255).astype(np.uint8)

        # 复制原图
        separated = binary_image.copy()

        # 1. 查找轮廓
        contours, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return separated

        print(f"找到 {len(contours)} 个轮廓")

        # 2. 处理每个轮廓
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)

            if area < min_area:  # 过滤小面积轮廓
                continue

            # 计算凸包
            hull = cv2.convexHull(contour, returnPoints=False)

            if hull is None or len(hull) < 3:
                continue

            # 计算凸缺陷
            defects = cv2.convexityDefects(contour, hull)

            if defects is None:
                continue

            # 过滤凸缺陷
            deep_defects = []
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                depth = d / 256.0

                if depth > min_defect_depth:
                    deep_defects.append({
                        'start': start,
                        'end': end,
                        'far': far,
                        'depth': depth
                    })

            if len(deep_defects) < 2:
                continue

            print(f"轮廓 {i}: 面积={area:.1f}, 凸缺陷数量={len(deep_defects)}")

            # 3. 根据凸缺陷进行分割
            if self.separate_by_convex_defects(separated, contour, deep_defects, min_defect_depth):
                print(f"  成功分割轮廓 {i}")

        return separated

    def separate_by_convex_defects(self, image, contour, defects, min_depth):
        if len(defects) < 2:
            return False

        # 1. 对凸缺陷按深度排序
        defects.sort(key=lambda x: x['depth'], reverse=True)

        # 2. 选择前2个最深的凸缺陷
        selected_defects = defects[:2]

        # 3. 在凸缺陷之间画分割线
        for i in range(len(selected_defects)):
            for j in range(i + 1, len(selected_defects)):
                defect1 = selected_defects[i]
                defect2 = selected_defects[j]

                # 获取凹点
                point1 = defect1['far']
                point2 = defect2['far']

                # 计算两点间距离
                distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

                # 如果距离适中，则画分割线
                if 10 < distance < 200:  # 距离阈值可根据实际情况调整
                    # 在两点之间画线
                    cv2.line(image, point1, point2, 0, 2)
                    return True

        return False

    def morphology_based_separation(self, image_binary, kernel_size=3, iterations=2):
        """
        基于形态学操作分离粘连实例

        Args:
            image_binary: 二值图像
            kernel_size: 结构元素大小
            iterations: 迭代次数
        """
        # 确保输入是uint8类型
        if image_binary.dtype != np.uint8:
            if image_binary.dtype == bool:
                image_binary = (image_binary * 255).astype(np.uint8)
            else:
                image_binary = image_binary.astype(np.uint8)

        # 确保是0/255格式
        if np.max(image_binary) <= 1:
            image_binary = (image_binary * 255).astype(np.uint8)

        # 使用矩形结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 先腐蚀分离粘连
        eroded = cv2.erode(image_binary, kernel, iterations=iterations)

        # 再膨胀恢复大致大小
        dilated = cv2.dilate(eroded, kernel, iterations=iterations)

        return dilated

    def separate_adhesive_buildings(self, binary_image, use_method='combined'):
        if binary_image is None or np.sum(binary_image) == 0:
            return binary_image

        # 转换为uint8
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)

        if use_method == 'convex_hull':
            return self.convex_hull_separation(binary_image, min_defect_depth=15, min_area=100)

        elif use_method == 'morphology':
            return self.morphology_based_separation(binary_image, kernel_size=3, iterations=2)

        elif use_method == 'combined':
            # 先使用形态学，再使用凸包
            img1 = self.morphology_based_separation(binary_image, kernel_size=3, iterations=1)
            return self.convex_hull_separation(img1, min_defect_depth=10, min_area=50)

        else:
            return binary_image

    def find_instances(self, binary_img):
        """从二值图像中找到所有实例"""
        # 使用连通组件标记找到所有实例
        num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))

        instances = {}

        for label in range(1, num_labels):  # 跳过背景(0)
            instance_mask = (labels == label)

            if np.sum(instance_mask) > 0:  # 只保留有像素的实例
                # 计算质心
                coords = np.where(instance_mask)
                cy = np.mean(coords[0])
                cx = np.mean(coords[1])

                instances[label] = {
                    'mask': instance_mask,
                    'centroid': (cy, cx)
                }

        return instances, labels

    def compute_shape_signature(self, instance_mask, centroid, num_angles=72):
        """计算实例的形状签名（质心到边界的距离特征）"""
        height, width = instance_mask.shape
        cy, cx = centroid

        # 获取实例边界
        contours, _ = cv2.findContours(instance_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 创建边界距离数组
        distances = np.zeros(num_angles)

        # 每5度一个角度
        for i in range(num_angles):
            angle = math.radians(i * 5)  # 修正：每5度，而不是每1度

            # 创建射线方向
            dx = math.cos(angle)
            dy = math.sin(angle)

            # 沿着射线方向寻找边界
            for d in range(1, int(math.sqrt(height ** 2 + width ** 2))):
                x = int(cx + d * dx)
                y = int(cy + d * dy)

                # 检查是否到达边界或超出图像范围
                if x < 0 or x >= width or y < 0 or y >= height or not instance_mask[y, x]:
                    distances[i] = d
                    break

        # 归一化
        if np.max(distances) > 0:
            distances = distances / np.max(distances)

        return distances

    def compute_similarity(self, sig1, sig2):
        """计算两个形状签名的相似度"""
        if sig1 is None or sig2 is None:
            return 0

        # 计算余弦相似度
        dot_product = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)

        if norm1 == 0 or norm2 == 0:
            return 0

        cosine_sim = dot_product / (norm1 * norm2)

        # 转换为百分比
        similarity = (cosine_sim + 1) / 2 * 100

        return similarity

    def skeleton_based_separation(self, image_binary, min_branch_length=20):
        """
        基于骨架分析分离粘连实例

        Args:
            image_binary: 二值图像
            min_branch_length: 最小分支长度
        """
        # 1. 确保输入是uint8类型
        if image_binary.dtype == bool:
            image_binary = (image_binary * 255).astype(np.uint8)
        elif image_binary.dtype != np.uint8:
            image_binary = image_binary.astype(np.uint8)

        # 确保图像是0/255的格式
        if np.max(image_binary) <= 1:
            image_binary = (image_binary * 255).astype(np.uint8)

        # 2. 骨架化
        skeleton = skeletonize(image_binary > 0)
        skeleton = skeleton.astype(np.uint8) * 255

        # 3. 找到端点（度为1的点）和交叉点（度>=3的点）
        kernel = np.ones((3, 3), dtype=np.uint8)

        # 4. 找到端点
        endpoints = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    # 计算3x3邻域
                    neighbors = skeleton[i - 1:i + 2, j - 1:j + 2]
                    if np.sum(neighbors) == 2:  # 端点
                        endpoints.append((i, j))

        # 5. 在端点处切割
        # 复制图像，确保是uint8类型
        separated = image_binary.copy().astype(np.uint8)

        for y, x in endpoints:
            cv2.circle(separated, (x, y), 3, 0, -1)  # 在端点处画黑点断开连接

        return separated

    def get_multi_class_segmentation(self, image, phase_name=""):
        print(f"  ✓ 进行多类别语义分割{phase_name}...")

        # 调用多类别语义分割函数，传入自适应阈值管理器
        semantic_map, colored_seg, class_masks, class_probs, each_class_max_probs = \
            multi_class_semantic_segmentation(
                image=image,
                prompts=self.all_classes,
                colors=self.colors,
                output_dir=os.path.join(self.output_dir, f"semantic_{phase_name}") if phase_name else None,
                high_processor=self.processor_high,
                low_processor=self.processor_low,
                model=self.model,
                adaptive_threshold_manager=self.adaptive_threshold_manager  # 新增参数
            )

        return semantic_map, colored_seg, class_masks, class_probs, each_class_max_probs

    def detect_initial_changes(self, semantic_map1, semantic_map2, similarity_threshold=90):
        """基于重叠、形状相似度和面积相似度的变化检测"""
        print("\n" + "=" * 60)
        print("2. 检测初始变化（使用凸包分割粘连对象）...")

        # 1. 加载和预处理
        building_t1 = semantic_map1.astype(np.uint8)
        building_t2 = semantic_map2.astype(np.uint8)

        # 确保是二值图像
        building_t1 = (building_t1 > 0).astype(np.uint8) * 255
        building_t2 = (building_t2 > 0).astype(np.uint8) * 255
        # building_t1 = self.separate_adhesive_buildings(building_t1, use_method='combined')
        # building_t2 = self.separate_adhesive_buildings(building_t2, use_method='combined')

        # 凸包分割粘连对象
        # self.separated_building_t1 = building_t1
        # self.separated_building_t2 = building_t2

        # 找到所有实例
        instances_t1, labels_t1 = self.find_instances(building_t1)
        instances_t2, labels_t2 = self.find_instances(building_t2)

        # 3. 计算形状签名
        for label, instance in instances_t1.items():
            instance['shape_sig'] = self.compute_shape_signature(instance['mask'], instance['centroid'])

        for label, instance in instances_t2.items():
            instance['shape_sig'] = self.compute_shape_signature(instance['mask'], instance['centroid'])

        # 4. 找到重叠实例对并比较形状
        changed_instances = set()  # 变化实例集合
        unchanged_pairs = []  # 不变化实例对
        changed_pairs = []  # 变化实例对

        # 初始化：将所有实例标记为变化
        for label in instances_t1:
            changed_instances.add(('t1', label))

        for label in instances_t2:
            changed_instances.add(('t2', label))

        # 计算所有实例的面积
        for label, instance in instances_t1.items():
            instance['area'] = np.sum(instance['mask'])

        for label, instance in instances_t2.items():
            instance['area'] = np.sum(instance['mask'])

        # 寻找重叠的实例对
        # 创建重叠变化图
        overlap_change = np.zeros_like(building_t1, dtype=np.uint8)
        for label1, instance1 in instances_t1.items():
            for label2, instance2 in instances_t2.items():
                # 计算重叠面积
                overlap = np.sum(instance1['mask'] & instance2['mask'])

                if overlap > 0:  # 有重叠
                    # 计算形状相似度
                    sim = self.compute_similarity(instance1['shape_sig'], instance2['shape_sig'])

                    # 计算面积相似度
                    area1 = instance1['area']
                    area2 = instance2['area']
                    area_sim = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

                    # 计算面积差异百分比
                    area_ratio = area1 / area2 if area2 > 0 else float('inf')
                    area_diff = abs(area1 - area2) / ((area1 + area2) / 2) if (area1 + area2) > 0 else float('inf')

                    print(f"实例对 T1-{label1} 与 T2-{label2}: ")
                    print(f"  重叠面积: {overlap}, 形状相似度: {sim:.2f}%")
                    print(f"  时相1面积: {area1}, 时相2面积: {area2}")
                    print(f"  面积相似度: {area_sim:.2f}, 面积差异百分比: {area_diff:.2f}")

                    # 判断条件：面积相似（面积相似度>0.8）且形状相似度大于阈值
                    area_threshold = 0.6  # 面积相似度阈值
                    shape_sim_threshold = similarity_threshold

                    is_unchanged = False

                    if area_sim >= area_threshold and sim >= shape_sim_threshold:
                        # 条件1：面积相似且形状相似 -> 不变化
                        is_unchanged = True
                        print(
                            f"  ✓ 不变化判定: 面积相似({area_sim:.2f}≥{area_threshold})且形状相似({sim:.2f}%≥{shape_sim_threshold}%)")
                    else:
                        # 不满足条件 -> 变化
                        is_unchanged = False
                        print(f"  ✗ 变化判定")

                    if is_unchanged:  # 满足不变化条件
                        if ('t1', label1) in changed_instances:
                            changed_instances.remove(('t1', label1))
                        if ('t2', label2) in changed_instances:
                            changed_instances.remove(('t2', label2))
                        unchanged_pairs.append((label1, label2, sim, area_sim))
                    else:  # 视为变化
                        changed_pairs.append((label1, label2, sim, area_sim))

                        # 将两个重叠且变化的实例合并到overlap_change图上
                        mask1 = instance1['mask']
                        overlap_change[mask1] = 255

                        mask2 = instance2['mask']
                        overlap_change[mask2] = 255

        print(f"\n结果统计:")
        print(f"不变化实例对数: {len(unchanged_pairs)}")
        for label1, label2, sim, area_sim in unchanged_pairs:
            print(f"  T1-{label1} ↔ T2-{label2}: 形状相似度={sim:.2f}%, 面积相似度={area_sim:.2f}")

        print(f"变化实例对数: {len(changed_pairs)}")
        for label1, label2, sim, area_sim in changed_pairs:
            print(f"  T1-{label1} ↔ T2-{label2}: 形状相似度={sim:.2f}%, 面积相似度={area_sim:.2f}")

        print(f"总变化实例数: {len(changed_instances)}")

        # 5. 创建变化图
        change_map = np.zeros_like(building_t1, dtype=np.uint8)
        # 创建不重叠变化图
        non_overlap_change = np.zeros_like(building_t1, dtype=np.uint8)
        for phase, label in changed_instances:
            if phase == 't1':
                mask = instances_t1[label]['mask']
            else:
                mask = instances_t2[label]['mask']

            change_map[mask] = 255

            overlap_area = np.sum(mask & (overlap_change > 0))
            if overlap_area == 0:  # 没有重叠
                non_overlap_change[mask] = 255
            else:
                # 如果与已有重叠实例有重叠，移除这部分
                non_overlap_change[mask] = 0
                # 确保重叠部分完整
                overlap_change[mask] = 255

        overlap_change = change_map - non_overlap_change

        return overlap_change, non_overlap_change, change_map

    def analyze_overlap_objects(self, superpixel_labels, overlapping_changes_mask, semantic_map1, semantic_map2):
        # 如果没有重叠变化区域，直接返回
        if np.sum(overlapping_changes_mask) == 0:
            print("  ✓ 无重叠变化区域，跳过精细判定")
            return np.zeros(1, dtype=bool), np.zeros((self.height, self.width), dtype=bool)

        # 确保 class_prob_maps_t1 和 class_prob_maps_t2 是字典格式
        if not isinstance(semantic_map1, dict) or not isinstance(semantic_map2, dict):
            print("  ⚠ 类别概率图不是字典格式，尝试转换...")

            # 尝试从语义图中提取概率信息
            try:
                if isinstance(semantic_map1, np.ndarray):
                    # 假设是建筑物概率图
                    if len(semantic_map1.shape) == 2:  # 单通道
                        semantic_map1 = {'building': semantic_map1}
                    elif len(semantic_map1.shape) == 3:  # 多通道
                        # 取第一个通道作为建筑物概率
                        semantic_map1 = {'building': semantic_map1[:, :, 0]}
                else:
                    # 创建默认概率图
                    building_mask_t1 = semantic_map1 > 0
                    semantic_map1 = {'building': building_mask_t1.astype(np.float32)}

                if isinstance(semantic_map2, np.ndarray):
                    if len(semantic_map2.shape) == 2:  # 单通道
                        semantic_map2 = {'building': semantic_map2}
                    elif len(semantic_map2.shape) == 3:  # 多通道
                        semantic_map2 = {'building': semantic_map2[:, :, 0]}
                else:
                    building_mask_t2 = semantic_map2 > 0
                    semantic_map2 = {'building': building_mask_t2.astype(np.float32)}

                print(f"  ✓ 已创建类别概率图字典: {list(semantic_map1.keys())}")

            except Exception as e:
                print(f"  ✗ 无法创建类别概率图字典: {e}")
                return np.zeros(1, dtype=bool), np.zeros((self.height, self.width), dtype=bool)

        # 创建二值掩码
        overlap_binary = overlapping_changes_mask.astype(np.uint8) * 255

        # 获取连通组件
        num_initial_objects, labels_initial, stats_initial, _ = cv2.connectedComponentsWithStats(
            overlap_binary, connectivity=8
        )

        if num_initial_objects == 1:  # 只有背景
            return np.zeros(1, dtype=bool), np.zeros((self.height, self.width), dtype=bool)

        # 初始化结果
        overlap_objects_mask = np.zeros((self.height, self.width), dtype=bool)
        refined_overlap_changes = np.zeros((self.height, self.width), dtype=bool)

        # 处理每个初始重叠变化对象
        for obj_id in range(1, num_initial_objects):  # 跳过背景
            # 获取初始对象的掩码
            initial_obj_mask = (labels_initial == obj_id)

            if np.sum(initial_obj_mask) == 0:
                continue

            # 获取该初始对象内部的精细分割标签
            obj_superpixel_labels = superpixel_labels.copy()
            # 只保留在该初始对象内的精细分割
            obj_superpixel_labels[~initial_obj_mask] = 0

            # 获取该对象内所有的精细分割标签
            unique_labels = np.unique(obj_superpixel_labels)
            unique_labels = unique_labels[unique_labels > 0]  # 去掉背景0

            if len(unique_labels) == 0:
                # 如果没有精细分割，使用初始对象本身
                refined_objects = [initial_obj_mask]
            else:
                # 创建精细分割对象的列表
                refined_objects = []
                for label in unique_labels:
                    refined_obj_mask = (obj_superpixel_labels == label)
                    if np.sum(refined_obj_mask) > 0:
                        refined_objects.append(refined_obj_mask)

            # 3. 对每个精细分割对象进行变化概率计算
            print(f"初始对象{obj_id}: 精细分割为 {len(refined_objects)} 个子对象")

            for i, refined_mask in enumerate(refined_objects):
                obj_probs_t1 = []
                obj_probs_t2 = []

                for class_name, prob_map_t1 in semantic_map1.items():
                    if class_name in semantic_map2:
                        prob_values_t1 = prob_map_t1[refined_mask]
                        prob_values_t2 = semantic_map2[class_name][refined_mask]

                        if len(prob_values_t1) > 0 and len(prob_values_t2) > 0:
                            # 计算平均概率
                            avg_t1 = np.mean(prob_values_t1)
                            avg_t2 = np.mean(prob_values_t2)

                            obj_probs_t1.append(avg_t1)
                            obj_probs_t2.append(avg_t2)

                if obj_probs_t1 and obj_probs_t2:
                    # 计算变化概率
                    obj_probs_t1_array = np.array(obj_probs_t1)
                    obj_probs_t2_array = np.array(obj_probs_t2)

                    # 使用最大概率差异作为变化概率
                    change_prob = np.max(np.abs(obj_probs_t1_array - obj_probs_t2_array))

                    # 判断是否变化
                    if change_prob > 0.6:  # 变化阈值
                        refined_overlap_changes[refined_mask] = True
                        overlap_objects_mask[refined_mask] = True

        return overlap_objects_mask, refined_overlap_changes

    def get_class_probability_maps(self, generate_coarse_seg=True):
        self.class_prob_maps_t1 = self._get_probability_maps_for_image(self.image1_np, "时相1")
        self.class_prob_maps_t2 = self._get_probability_maps_for_image(self.image2_np, "时相2")

        return self.class_prob_maps_t1, self.class_prob_maps_t2

    def _get_probability_maps_for_image(self, image_np, phase_name):
        """获取单个影像的概率图"""
        prob_maps = {}

        for class_name in self.all_classes:
            print(f"    处理{phase_name}类别 '{class_name}'...")

            height, width = image_np.shape[:2]

            # 创建灰度图像
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # 使用自适应阈值
            thresh = cv2.adaptiveThreshold(gray_image, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            # 形态学处理
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # 转换为概率
            prob_map = thresh.astype(np.float32) / 255.0

            # 高斯平滑
            prob_map = gaussian_filter(prob_map, sigma=1)

            prob_maps[class_name] = prob_map

        return prob_maps

    def combine_changes(self, initial_changes, overlapping_changes_mask, refined_overlap_changes, non_overlapping_changes_mask):
        refined_overlap_changes_int = refined_overlap_changes.astype(np.uint8)
        final_changes = refined_overlap_changes_int + non_overlapping_changes_mask
        final_changes[final_changes > 0] = 1
        return final_changes

    def process_single_class(self, class_name, refine_factor=4, change_threshold=0.3):
        # 获取语义分割结果
        semantic_map1, colored_seg1, class_masks1, class_probs1, each_class_max_probs1 = self.get_multi_class_segmentation(
            self.image1, phase_name="t1"
        )
        semantic_map2, colored_seg2, class_masks2, class_probs2, each_class_max_probs2 = self.get_multi_class_segmentation(
            self.image2, phase_name="t2"
        )

        # 保存粗分割结果
        seg_dir = os.path.join(self.output_dir, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)

        seg1_path = os.path.join(seg_dir, "seg1.png")
        seg2_path = os.path.join(seg_dir, "seg2.png")
        segRGB1_path = os.path.join(seg_dir, "segRGB1.png")
        segRGB2_path = os.path.join(seg_dir, "segRGB2.png")

        # 保存分割图
        plt.imsave(seg1_path, semantic_map1, cmap='gray')
        plt.imsave(seg2_path, semantic_map2, cmap='gray')
        plt.imsave(segRGB1_path, colored_seg1)
        plt.imsave(segRGB2_path, colored_seg2)

        print(f"  ✓ 粗分割图已保存: {seg1_path}, {seg2_path}")

        # 动态定义类别配置，基于self.all_classes
        class_configs = []
        for i, class_name in enumerate(self.all_classes):
            class_configs.append({
                'name': class_name.replace(' ', '_'),  # 将空格替换为下划线
                'index': i + 1,  # 类别索引从1开始
                'prob_idx': i,  # 概率图索引
                'original_name': class_name
            })

        print(f"  ✓ 处理 {len(class_configs)} 个类别: {[config['name'] for config in class_configs]}")

        # 存储结果的字典
        results = {}
        class_maps = {}

        # 批量处理所有类别
        for config in class_configs:
            class_name = config['name']
            original_name = config['original_name']
            class_idx = config['index']
            prob_idx = config['prob_idx']

            # 检查索引是否在范围内
            if prob_idx >= len(each_class_max_probs1) or prob_idx >= len(each_class_max_probs2):
                print(f"  ⚠ 警告: 类别 {class_name} 的概率图索引 {prob_idx} 超出范围，跳过")
                continue

            class_prob_map1 = each_class_max_probs1[prob_idx]
            class_prob_map2 = each_class_max_probs2[prob_idx]

            # 创建单类别分割图
            sem_map1 = (semantic_map1 == class_idx).astype(np.uint8)
            sem_map2 = (semantic_map2 == class_idx).astype(np.uint8)

            # 保存路径
            sem_map1_path = os.path.join(seg_dir, f"{class_name}_semantic_map1.png")
            sem_map2_path = os.path.join(seg_dir, f"{class_name}_semantic_map2.png")

            # 保存分割图
            plt.imsave(sem_map1_path, sem_map1, cmap='gray')
            plt.imsave(sem_map2_path, sem_map2, cmap='gray')

            # 存储类别地图
            class_maps[f'{class_name}_semantic_map1'] = sem_map1
            class_maps[f'{class_name}_semantic_map2'] = sem_map2

            # 超像素分割
            superpixel_labels = self._get_class_superpixels(sem_map1_path, sem_map2_path, refine_factor=refine_factor)

            # 获取初始变化实例（使用凸包分割）
            overlap_change, non_overlap_change, change_map = self.detect_initial_changes(
                sem_map1.squeeze(), sem_map2.squeeze(), similarity_threshold=98)

            # 对重叠变化对象进行精细变化判定
            if np.sum(overlap_change) > 0:
                overlap_object_changed, refined_overlap_changes = self.analyze_overlap_objects(
                    superpixel_labels, overlap_change, class_prob_map1.squeeze(), class_prob_map2.squeeze())
            else:
                refined_overlap_changes = np.zeros((self.height, self.width), dtype=bool)
                overlap_objects_mask = np.zeros((self.height, self.width), dtype=bool)
                print(f"  ✓ 类别 {class_name} 无重叠变化对象需要分析")

            # 合并变化结果
            final_changes = self.combine_changes(change_map, overlap_change, refined_overlap_changes,
                                                 non_overlap_change)

            # 保存结果
            change_map_path = os.path.join(seg_dir, f"{class_name}_initial_change.png")
            binary_change_path = os.path.join(seg_dir, f"{class_name}_final_change.png")

            plt.imsave(change_map_path, change_map, cmap='gray')
            Image.fromarray((final_changes * 255).astype(np.uint8)).save(binary_change_path)

            # 存储结果
            results[class_name] = {
                'semantic_map1': sem_map1,
                'semantic_map2': sem_map2,
                'superpixel_labels': superpixel_labels,
                'change_map': change_map,
                'binary_change_map': final_changes
            }

            print(f"✓ 处理类别 {original_name} 完成")

        # 创建合并结果
        all_binary_maps = []
        all_initial_map = []

        for config in class_configs:
            class_name = config['name']
            if class_name in results:
                # 合并final_changes
                all_binary_maps.append(results[class_name]['binary_change_map'].astype(np.uint8))
                all_initial_map.append(results[class_name]['change_map'].astype(np.uint8))
            else:
                # 创建空的占位图
                all_binary_maps.append(np.zeros((self.height, self.width), dtype=np.uint8))
                all_initial_map.append(np.zeros((self.height, self.width), dtype=np.uint8))

        # 确保所有的变化图都是相同的数据类型和形状
        all_binary_change_maps = [map.astype(np.uint8) for map in all_binary_maps]
        all_initial_map = [map.astype(np.uint8) for map in all_initial_map]

        # 将所有的二值变化图堆叠成一个多通道的tensor
        if all_binary_change_maps:
            combined_change_tensor = np.stack(all_binary_change_maps, axis=0)
            combined_initial_change_tensor = np.stack(all_initial_map, axis=0)
        else:
            # 如果没有处理任何类别，创建空的tensor
            combined_change_tensor = np.zeros((len(class_configs), self.height, self.width), dtype=np.uint8)
            combined_initial_change_tensor = np.zeros((len(class_configs), self.height, self.width), dtype=np.uint8)

        # 转换为PyTorch tensor
        combined_change_tensor = torch.from_numpy(combined_change_tensor).float()
        combined_initial_change_tensor = torch.from_numpy(combined_initial_change_tensor).float()

        # 创建一个合并的二值变化图（所有类别的变化叠加）
        combined_binary_map = np.zeros((self.height, self.width), dtype=np.uint8)
        combined_initial_binary_map = np.zeros((self.height, self.width), dtype=np.uint8)

        for i, config in enumerate(class_configs):
            if i < len(all_binary_change_maps):
                combined_binary_map = np.logical_or(combined_binary_map, all_binary_change_maps[i]).astype(np.uint8)
                combined_initial_binary_map = np.logical_or(combined_initial_binary_map, all_initial_map[i]).astype(
                    np.uint8)

        # 保存合并的二值变化图
        combined_binary_path = os.path.join(seg_dir, "combined_binary_change.png")
        Image.fromarray((combined_binary_map * 255).astype(np.uint8)).save(combined_binary_path)
        print(f"✓ 保存合并的二值变化图: {combined_binary_path}")

        combined_initial_binary_path = os.path.join(seg_dir, "combined_initial_binary_change.png")
        Image.fromarray((combined_initial_binary_map * 255).astype(np.uint8)).save(combined_initial_binary_path)
        print(f"✓ 保存合并的初始二值变化图: {combined_initial_binary_path}")

        # 返回合并的tensor，供后续使用
        return {
            'class_names': [config['name'] for config in class_configs],
            'change_tensor': combined_change_tensor,
            'combined_binary_map': combined_binary_map,
            'combined_initial_binary_map': combined_initial_binary_map,
            'semantic_map1': semantic_map1,
            'semantic_map2': semantic_map2,
            'individual_maps': {config['name']: results.get(config['name'], {}).get('binary_change_map', np.zeros((self.height, self.width),
            dtype=np.uint8)) for config in class_configs},
            'change_maps': {config['name']: (
                results.get(config['name'], {}).get('change_map', np.zeros((self.height, self.width), dtype=np.uint8)),
                results.get(config['name'], {}).get('binary_change_map',
                np.zeros((self.height, self.width), dtype=np.uint8)))
                for config in class_configs
            }
        }

    def _get_class_superpixels(self, seg1_path, seg2_path, refine_factor=4):
        """超像素分割"""
        # 修改这行：使用 os.path.join 而不是 / 操作符
        temp_dir = os.path.join(self.output_dir, "temp_superpixels")
        os.makedirs(temp_dir, exist_ok=True)
        # 修改这两行：使用 os.path.join
        temp_img1_path = os.path.join(temp_dir, "temp_phase1.png")
        temp_img2_path = os.path.join(temp_dir, "temp_phase2.png")
        # 保存图片
        Image.fromarray(self.image1_np).save(temp_img1_path)
        Image.fromarray(self.image2_np).save(temp_img2_path)
        try:
            superpixel_labels = FineSeg(
                image1_path=temp_img1_path,  # 已经是字符串，无需转换
                image2_path=temp_img2_path,  # 已经是字符串，无需转换
                coarse_seg1_path=seg1_path,
                coarse_seg2_path=seg2_path,
                refine_factor=refine_factor,
                output_dir=temp_dir
            )
        except Exception as e:
            print(f"  超像素分割失败，使用SLIC: {e}")
            from skimage.segmentation import slic
            avg_image = (self.image1_np.astype(np.float32) + self.image2_np.astype(np.float32)) / 2
            avg_image = avg_image.astype(np.uint8)
            n_segments = (self.height * self.width) // 100
            superpixel_labels = slic(avg_image, n_segments=n_segments, compactness=30, sigma=2, start_label=0)

        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return superpixel_labels

    def visualize_semantic_change_maps(self, all_results, output_dir=None):
        """
        可视化语义变化检测结果 - 增强复合变化处理
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        if len(all_results) == 0:
            print("没有结果可以可视化")
            return None

        result = all_results[0]

        # 获取语义分割图
        semantic_map1 = result['semantic_map1']
        semantic_map2 = result['semantic_map2']

        # 获取每个类别的变化图
        individual_maps = result['individual_maps']
        instance_change_map = result['combined_binary_map']

        height, width = semantic_map1.shape

        # 创建最终的彩色变化图
        final_adaptive_change_map = np.zeros((height, width, 3), dtype=np.uint8)

        # 存储图例信息
        legend_info = []
        color_idx = 0

        # 生成随机颜色的函数
        def get_random_color(seed):
            np.random.seed(seed + 42)  # 加偏移确保颜色多样性
            return [int(x) for x in np.random.randint(50, 256, 3)]

        print("\n" + "=" * 60)
        print("自适应变化信息输出")
        print("=" * 60)

        # 调试信息：打印CLASS_INDEX和CLASS_COLORS
        print(f"动态CLASS_INDEX: {self.CLASS_INDEX}")
        print(f"动态CLASS_COLORS: {self.CLASS_COLORS}")

        # 处理每个变化指令
        for trans_idx, trans_item in enumerate(self.trans_results):
            print(f"\n处理指令 {trans_idx + 1}: {trans_item.get('original_instruction', '')}")

            semantics = trans_item.get('semantics', {})
            change_type = semantics.get('change_semantics', 'all_changes')
            change_direction = semantics.get('change_direction', 'bidirectional')
            transitions = semantics.get('transitions', [])
            involved_classes = semantics.get('involved_classes', [])

            print(f"  变化类型: {change_type}")
            print(f"  变化方向: {change_direction}")
            print(f"  转换关系: {transitions}")
            print(f"  涉及类别: {involved_classes}")

            # ==================== 新增：复合变化处理 ====================
            if change_type == 'composite' and transitions and len(transitions) > 0:
                print(f"  → 检测到复合变化指令")

                # 处理每个转换关系
                for transition in transitions:
                    # 解析转换字符串
                    if isinstance(transition, str):
                        if '→' in transition:
                            # 单向转换格式: "background→building" 或 "low vegetation→background"
                            parts = transition.split('→')
                            if len(parts) == 2:
                                from_class = parts[0].strip()
                                to_class = parts[1].strip()
                            else:
                                print(f"    跳过无效的转换格式: {transition}")
                                continue
                        elif '↔' in transition:
                            # 双向转换格式: "building↔bare land"
                            parts = transition.split('↔')
                            if len(parts) == 2:
                                class1 = parts[0].strip()
                                class2 = parts[1].strip()

                                # 双向转换需要处理两个方向
                                # 处理方向1: class1 -> class2
                                print(f"    处理双向转换方向1: {class1} → {class2}")
                                self._process_conversion(class1, class2, semantic_map1, semantic_map2, individual_maps,
                                                         final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1

                                # 处理方向2: class2 -> class1
                                print(f"    处理双向转换方向2: {class2} → {class1}")
                                self._process_conversion(class2, class1, semantic_map1, semantic_map2, individual_maps,
                                                         final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1

                                # 跳过后续处理
                                continue
                            else:
                                print(f"    跳过无效的双向转换格式: {transition}")
                                continue
                        else:
                            print(f"    跳过不支持的转换格式: {transition}")
                            continue
                    elif isinstance(transition, dict):
                        # 如果是字典格式，保持原处理方式
                        from_class = transition.get('from', '')
                        to_class = transition.get('to', '')
                    else:
                        print(f"    跳过不支持的转换格式: {type(transition)}")
                        continue

                    print(f"    处理复合转换: {from_class} → {to_class}")

                    # 获取类别索引
                    from_idx = self.CLASS_INDEX.get(from_class, 0)  # background的索引是0
                    to_idx = self.CLASS_INDEX.get(to_class, 0)  # background的索引是0

                    # 处理特殊情况：background不是CLASS_INDEX中的键，但索引为0
                    if from_class == "background":
                        from_idx = 0
                    if to_class == "background":
                        to_idx = 0

                    if from_idx >= 0 and to_idx >= 0:  # 允许background索引0
                        # 获取变化图（处理类别名称中的空格）
                        from_class_name = from_class.replace(' ', '_')
                        to_class_name = to_class.replace(' ', '_')

                        from_change = individual_maps.get(from_class_name,
                                                          np.zeros((height, width), dtype=np.uint8))
                        to_change = individual_maps.get(to_class_name, np.zeros((height, width), dtype=np.uint8))

                        # 找到T1是from_class的区域
                        t1_is_from = (semantic_map1 == from_idx)
                        # 找到T2是to_class的区域
                        t2_is_to = (semantic_map2 == to_idx)

                        # 同时要在变化区域内（两个类别的变化区域的并集）
                        combined_change_region = ((from_change > 0) | (to_change > 0))

                        # 最终变化区域：T1是A类 且 T2是B类 且 在变化区域内
                        transition_region = t1_is_from & t2_is_to & combined_change_region

                        pixel_count = np.sum(transition_region)
                        print(f"    发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

                        if np.any(transition_region):
                            color = get_random_color(color_idx)
                            color_idx += 1
                            final_adaptive_change_map[transition_region] = color
                            legend_info.append({
                                'color': color,
                                'label': f'{from_class} → {to_class}'
                            })
                    else:
                        print(f"    ⚠ 类别索引无效: {from_class}={from_idx}, {to_class}={to_idx}")

            # ==================== 情况1：类别转换 (A -> B) ====================
            elif transitions and len(transitions) > 0 and change_type != 'composite':
                print(f"  → 检测到类别转换指令")
                for transition in transitions:
                    # 解析转换字符串
                    if isinstance(transition, str):
                        if '→' in transition:
                            # 单向转换格式: "bare land→building"
                            parts = transition.split('→')
                            if len(parts) == 2:
                                from_class = parts[0].strip()
                                to_class = parts[1].strip()
                            else:
                                print(f"    跳过无效的转换格式: {transition}")
                                continue
                        elif '↔' in transition:
                            # 双向转换格式: "building↔bare land"
                            parts = transition.split('↔')
                            if len(parts) == 2:
                                class1 = parts[0].strip()
                                class2 = parts[1].strip()

                                # 双向转换需要处理两个方向
                                # 处理方向1: class1 -> class2
                                print(f"    处理双向转换方向1: {class1} → {class2}")
                                self._process_conversion(class1, class2, semantic_map1, semantic_map2, individual_maps,
                                                         final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1

                                # 处理方向2: class2 -> class1
                                print(f"    处理双向转换方向2: {class2} → {class1}")
                                self._process_conversion(class2, class1, semantic_map1, semantic_map2, individual_maps,
                                                         final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1

                                # 跳过后续处理
                                continue
                            else:
                                print(f"    跳过无效的双向转换格式: {transition}")
                                continue
                        else:
                            print(f"    跳过不支持的转换格式: {transition}")
                            continue
                    elif isinstance(transition, dict):
                        # 如果是字典格式，保持原处理方式
                        from_class = transition.get('from', '')
                        to_class = transition.get('to', '')
                    else:
                        print(f"    跳过不支持的转换格式: {type(transition)}")
                        continue

                    print(f"    处理转换: {from_class} → {to_class}")

                    # 获取类别索引
                    from_idx = self.CLASS_INDEX.get(from_class, 0)  # background的索引是0
                    to_idx = self.CLASS_INDEX.get(to_class, 0)  # background的索引是0

                    # 处理特殊情况：background不是CLASS_INDEX中的键，但索引为0
                    if from_class == "background":
                        from_idx = 0
                    if to_class == "background":
                        to_idx = 0

                    if from_idx >= 0 and to_idx >= 0:  # 允许background索引0
                        # 获取变化图（处理类别名称中的空格）
                        from_class_name = from_class.replace(' ', '_')
                        to_class_name = to_class.replace(' ', '_')

                        from_change = individual_maps.get(from_class_name,
                                                          np.zeros((height, width), dtype=np.uint8))
                        to_change = individual_maps.get(to_class_name, np.zeros((height, width), dtype=np.uint8))

                        # 找到T1是from_class的区域
                        t1_is_from = (semantic_map1 == from_idx)
                        # 找到T2是to_class的区域
                        t2_is_to = (semantic_map2 == to_idx)

                        # 同时要在变化区域内（两个类别的变化区域的并集）
                        combined_change_region = ((from_change > 0) | (to_change > 0))

                        # 最终变化区域：T1是A类 且 T2是B类 且 在变化区域内
                        transition_region = t1_is_from & t2_is_to & combined_change_region

                        pixel_count = np.sum(transition_region)
                        print(f"    发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

                        if np.any(transition_region):
                            color = get_random_color(color_idx)
                            color_idx += 1
                            final_adaptive_change_map[transition_region] = color
                            legend_info.append({
                                'color': color,
                                'label': f'{from_class} → {to_class}'
                            })
                    else:
                        print(f"    ⚠ 类别索引无效: {from_class}={from_idx}, {to_class}={to_idx}")

            # ==================== 情况2：单类别出现或消失 ====================
            elif change_direction in ['appear', 'disappear', 'increase', 'decrease', 'unidirectional']:
                print(f"  → 检测到单向变化指令: {change_direction}")

                for cls in involved_classes:
                    cls_idx = self.CLASS_INDEX.get(cls, -1)
                    if cls_idx <= 0 and cls != "background":  # background的索引是0
                        print(f"    ⚠ 类别 '{cls}' 未找到对应索引，跳过")
                        continue

                    cls_name = cls.replace(' ', '_')
                    cls_change = individual_maps.get(cls_name, np.zeros((height, width), dtype=np.uint8))

                    # 根据变化方向确定检测逻辑
                    if change_direction in ['disappear', 'decrease']:
                        # 地物消失：T1有该地物，T2是background
                        t1_has = (semantic_map1 == cls_idx)
                        t2_is_background = (semantic_map2 == 0)  # background索引为0
                        change_region = t1_has & t2_is_background & (cls_change > 0)
                        label = f'{cls} disappeared'
                        print(f"    检测 {cls} 消失区域 (T1={cls_idx}, T2=0)")
                    else:  # appear, increase
                        # 地物出现：T1是background，T2有该地物
                        t1_is_background = (semantic_map1 == 0)  # background索引为0
                        t2_has = (semantic_map2 == cls_idx)
                        change_region = t1_is_background & t2_has & (cls_change > 0)
                        label = f'{cls} appeared'
                        print(f"    检测 {cls} 出现区域 (T1=0, T2={cls_idx})")

                    pixel_count = np.sum(change_region)
                    print(f"    发现 {pixel_count} 个像素")

                    if np.any(change_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[change_region] = color
                        legend_info.append({
                            'color': color,
                            'label': label
                        })

            # ==================== 情况3：双向变化或所有变化 ====================
            elif change_direction == 'bidirectional' or change_type == 'all_changes':
                print(f"  → 检测到双向变化指令")

                for cls in involved_classes:
                    cls_idx = self.CLASS_INDEX.get(cls, -1)
                    if cls_idx <= 0 and cls != "background":
                        print(f"    ⚠ 类别 '{cls}' 未找到对应索引，跳过")
                        continue

                    cls_name = cls.replace(' ', '_')
                    cls_change = individual_maps.get(cls_name, np.zeros((height, width), dtype=np.uint8))

                    # 分别检测出现和消失
                    # 消失区域：T1有，T2是background
                    t1_has = (semantic_map1 == cls_idx)
                    t2_is_background = (semantic_map2 == 0)  # background索引为0
                    disappear_region = t1_has & t2_is_background & (cls_change > 0)

                    # 出现区域：T1是background，T2有
                    t1_is_background = (semantic_map1 == 0)  # background索引为0
                    t2_has = (semantic_map2 == cls_idx)
                    appear_region = t1_is_background & t2_has & (cls_change > 0)

                    # 处理消失区域
                    if np.any(disappear_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[disappear_region] = color
                        legend_info.append({
                            'color': color,
                            'label': f'{cls} disappeared'
                        })
                        print(f"    {cls} 消失: {np.sum(disappear_region)} 像素")

                    # 处理出现区域
                    if np.any(appear_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[appear_region] = color
                        legend_info.append({
                            'color': color,
                            'label': f'{cls} appeared'
                        })
                        print(f"    {cls} 出现: {np.sum(appear_region)} 像素")

            # ==================== 情况4：未知变化类型，使用默认处理 ====================
            else:
                print(f"  → 使用默认变化检测")
                for cls in involved_classes:
                    cls_idx = self.CLASS_INDEX.get(cls, -1)
                    if cls_idx <= 0 and cls != "background":
                        continue

                    cls_name = cls.replace(' ', '_')
                    cls_change = individual_maps.get(cls_name, np.zeros((height, width), dtype=np.uint8))

                    # 所有变化区域
                    change_region = (cls_change > 0)

                    if np.any(change_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[change_region] = color
                        legend_info.append({
                            'color': color,
                            'label': f'{cls} change'
                        })
                        print(f"    {cls} 变化: {np.sum(change_region)} 像素")

        # ==================== 保存结果 ====================
        print("\n" + "-" * 40)
        print("保存自适应变化检测结果...")

        # 1. 保存带图例的自适应变化图
        adaptive_change_path = os.path.join(output_dir, "adaptive_semantic_change.png")
        self._save_change_map_with_legend(final_adaptive_change_map, legend_info, adaptive_change_path)
        print(f"  ✓ 保存自适应变化图（带图例）: {adaptive_change_path}")

        # 2. 保存纯变化图（无图例）
        adaptive_change_raw_path = os.path.join(output_dir, "adaptive_semantic_change_raw.png")
        Image.fromarray(final_adaptive_change_map).save(adaptive_change_raw_path)
        print(f"  ✓ 保存自适应变化图（无图例）: {adaptive_change_raw_path}")

        # ==================== 原有可视化逻辑 ====================
        # 用变化图掩膜语义分割图
        semantic_change_t1 = semantic_map1 * instance_change_map
        semantic_change_t2 = semantic_map2 * instance_change_map

        # 创建彩色图
        colored_change_t1 = np.zeros((height, width, 3), dtype=np.uint8)
        colored_change_t2 = np.zeros((height, width, 3), dtype=np.uint8)

        # 使用动态生成的CLASS_INDEX和CLASS_COLORS
        for class_name, class_idx in self.CLASS_INDEX.items():
            color = self.CLASS_COLORS.get(class_name, [128, 128, 128])  # 默认灰色

            mask_t1 = (semantic_change_t1 == class_idx)
            colored_change_t1[mask_t1] = color

            mask_t2 = (semantic_change_t2 == class_idx)
            colored_change_t2[mask_t2] = color

        # 保存原有结果
        change_t1_path = os.path.join(output_dir, "semantic_change_t1.png")
        if len(colored_change_t1.shape) == 3 and colored_change_t1.shape[2] == 3:
            plt.imsave(change_t1_path, colored_change_t1)
        else:
            plt.imsave(change_t1_path, colored_change_t1, cmap='gray')
        print(f"  ✓ 保存时相1语义变化检测图: {change_t1_path}")

        change_t2_path = os.path.join(output_dir, "semantic_change_t2.png")
        if len(colored_change_t2.shape) == 3 and colored_change_t2.shape[2] == 3:
            plt.imsave(change_t2_path, colored_change_t2)
        else:
            plt.imsave(change_t2_path, colored_change_t2, cmap='gray')
        print(f"  ✓ 保存时相2语义变化检测图: {change_t2_path}")

        binary_change_path = os.path.join(output_dir, "binary_change_map.png")
        binary_uint8 = (instance_change_map * 255).astype(np.uint8)
        Image.fromarray(binary_uint8).save(binary_change_path)
        print(f"  ✓ 保存二值变化图: {binary_change_path}")

        # 创建综合可视化图
        self._create_visualization_figure(
            self.image1_np, self.image2_np,
            colored_change_t1, colored_change_t2,
            instance_change_map,
            output_dir
        )

        # 创建自适应变化可视化图
        self._create_adaptive_visualization_figure(
            self.image1_np, self.image2_np,
            final_adaptive_change_map, legend_info,
            output_dir
        )

        return {
            'semantic_change_t1': colored_change_t1,
            'semantic_change_t2': colored_change_t2,
            'binary_change_map': instance_change_map,
            'adaptive_change_map': final_adaptive_change_map,
            'legend_info': legend_info
        }

    # 在SequentialMultiClassChangeDetection类中添加_process_conversion方法
    def _process_conversion(self, from_class, to_class, semantic_map1, semantic_map2,
                            individual_maps, final_adaptive_change_map,
                            legend_info, get_random_color_func, color_idx):
        """辅助方法：处理单个方向的变化转换"""
        # 获取类别索引
        from_idx = self.CLASS_INDEX.get(from_class, 0)  # background的索引是0
        to_idx = self.CLASS_INDEX.get(to_class, 0)  # background的索引是0

        # 处理特殊情况：background不是CLASS_INDEX中的键，但索引为0
        if from_class == "background":
            from_idx = 0
        if to_class == "background":
            to_idx = 0

        if from_idx >= 0 and to_idx >= 0:  # 允许background索引0
            # 获取变化图（处理类别名称中的空格）
            from_class_name = from_class.replace(' ', '_')
            to_class_name = to_class.replace(' ', '_')

            from_change = individual_maps.get(from_class_name,
                                              np.zeros((semantic_map1.shape[0], semantic_map1.shape[1]),
                                                       dtype=np.uint8))
            to_change = individual_maps.get(to_class_name,
                                            np.zeros((semantic_map1.shape[0], semantic_map1.shape[1]), dtype=np.uint8))

            # 找到T1是from_class的区域
            t1_is_from = (semantic_map1 == from_idx)
            # 找到T2是to_class的区域
            t2_is_to = (semantic_map2 == to_idx)

            # 同时要在变化区域内（两个类别的变化区域的并集）
            combined_change_region = ((from_change > 0) | (to_change > 0))

            # 最终变化区域：T1是A类 且 T2是B类 且 在变化区域内
            transition_region = t1_is_from & t2_is_to & combined_change_region

            pixel_count = np.sum(transition_region)
            print(f"      发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

            if np.any(transition_region):
                color = get_random_color_func(color_idx)
                final_adaptive_change_map[transition_region] = color
                legend_info.append({
                    'color': color,
                    'label': f'{from_class} → {to_class}'
                })

    def _save_change_map_with_legend(self, change_map, legend_info, save_path):
        """
        保存带图例的变化图

        Args:
            change_map: 彩色变化图 (H, W, 3)
            legend_info: 图例信息列表 [{'color': [r,g,b], 'label': 'xxx'}, ...]
            save_path: 保存路径
        """
        # 计算图例所需的额外高度
        legend_height = max(60, len(legend_info) * 30 + 40)

        height, width = change_map.shape[:2]

        # 创建带图例的画布
        canvas_height = height + legend_height
        canvas = np.ones((canvas_height, width, 3), dtype=np.uint8) * 255  # 白色背景

        # 放置变化图
        canvas[:height, :, :] = change_map

        # 使用matplotlib绘制图例
        fig, ax = plt.subplots(figsize=(width / 100, canvas_height / 100), dpi=100)
        ax.imshow(canvas)
        ax.axis('off')

        # 绘制图例
        if legend_info:
            legend_start_y = height + 20
            legend_x = 20
            box_size = 20
            spacing = 30

            for i, item in enumerate(legend_info):
                color = item['color']
                label = item['label']

                # 归一化颜色值到0-1范围
                color_normalized = [c / 255.0 for c in color]

                # 绘制颜色方块
                rect_y = legend_start_y + i * spacing
                rect = plt.Rectangle((legend_x, rect_y), box_size, box_size,
                                     facecolor=color_normalized, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

                # 绘制标签文本
                ax.text(legend_x + box_size + 10, rect_y + box_size / 2, label,
                        fontsize=10, verticalalignment='center', fontfamily='sans-serif')

        # 添加标题
        ax.text(width / 2, height + 10, 'Change Detection Legend',
                fontsize=12, fontweight='bold', ha='center', va='top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _create_adaptive_visualization_figure(self, img1, img2, adaptive_change_map, legend_info, output_dir):
        """
        创建自适应变化检测的综合可视化图

        Args:
            img1: 时相1原图
            img2: 时相2原图
            adaptive_change_map: 自适应彩色变化图
            legend_info: 图例信息
            output_dir: 输出目录
        """
        # 创建图形
        fig = plt.figure(figsize=(20, 12))

        # 使用GridSpec进行更灵活的布局
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                              hspace=0.15, wspace=0.1)

        # 1. 时相1原图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.set_title("Phase 1 (Before)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. 时相2原图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.set_title("Phase 2 (After)", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # 3. 自适应变化图
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(adaptive_change_map)
        ax3.set_title("Adaptive Semantic Change", fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 4. 时相1叠加变化区域
        ax4 = fig.add_subplot(gs[1, 0])
        overlay1 = img1.copy()
        change_mask = np.any(adaptive_change_map > 0, axis=2)
        overlay1_float = overlay1.astype(np.float32)
        overlay1_float[change_mask] = overlay1_float[change_mask] * 0.5 + adaptive_change_map[change_mask].astype(
            np.float32) * 0.5
        ax4.imshow(overlay1_float.astype(np.uint8))
        ax4.set_title("Phase 1 with Changes", fontsize=14, fontweight='bold')
        ax4.axis('off')

        # 5. 时相2叠加变化区域
        ax5 = fig.add_subplot(gs[1, 1])
        overlay2 = img2.copy()
        overlay2_float = overlay2.astype(np.float32)
        overlay2_float[change_mask] = overlay2_float[change_mask] * 0.5 + adaptive_change_map[change_mask].astype(
            np.float32) * 0.5
        ax5.imshow(overlay2_float.astype(np.uint8))
        ax5.set_title("Phase 2 with Changes", fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 6. 图例区域
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title("Legend", fontsize=14, fontweight='bold')

        if legend_info:
            n_items = len(legend_info)
            start_y = 0.9
            step_y = min(0.15, 0.8 / max(n_items, 1))

            for i, item in enumerate(legend_info):
                color = item['color']
                label = item['label']

                # 归一化颜色
                color_normalized = [c / 255.0 for c in color]

                y_pos = start_y - i * step_y

                # 绘制颜色方块
                rect = plt.Rectangle((0.1, y_pos - 0.05), 0.15, 0.08,
                                     facecolor=color_normalized, edgecolor='black', linewidth=1,
                                     transform=ax6.transAxes)
                ax6.add_patch(rect)

                # 绘制标签
                ax6.text(0.3, y_pos, label, fontsize=11, va='center',
                         transform=ax6.transAxes, fontfamily='sans-serif')
        else:
            ax6.text(0.5, 0.5, 'No changes detected', fontsize=12, ha='center', va='center',
                     transform=ax6.transAxes, style='italic', color='gray')

        # 添加总标题
        fig.suptitle('Adaptive Semantic Change Detection Results', fontsize=16, fontweight='bold', y=0.98)

        # 保存
        vis_path = os.path.join(output_dir, "adaptive_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  ✓ 保存自适应变化可视化图: {vis_path}")

    def _create_visualization_figure(self, img1, img2,
                                         semantic_change1, semantic_change2,
                                         binary_change, output_dir):
        """
        创建可视化图表

        Args:
            img1: 时相1原图
            img2: 时相2原图
            semantic_change1: 时相1语义变化图
            semantic_change2: 时相2语义变化图
            binary_change: 二值变化图
            output_dir: 输出目录
        """
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 时相1原图
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title("Phase 1 Original")
        axes[0, 0].axis('off')

        # 2. 时相2原图
        axes[0, 1].imshow(img2)
        axes[0, 1].set_title("Phase 2 Original")
        axes[0, 1].axis('off')

        # 3. 时相1语义变化检测图
        # 检查semantic_change1的维度
        if len(semantic_change1.shape) == 3 and semantic_change1.shape[2] == 3:  # RGB
            axes[0, 2].imshow(semantic_change1)
        else:  # 灰度
            # 如果是二维数组，则使用伪彩色显示
            colored_change1 = self._gray_to_color(semantic_change1)
            axes[0, 2].imshow(colored_change1)
        axes[0, 2].set_title("Phase 1 Semantic Change")
        axes[0, 2].axis('off')

        # 4. 时相2语义变化检测图
        if len(semantic_change2.shape) == 3 and semantic_change2.shape[2] == 3:  # RGB
            axes[1, 0].imshow(semantic_change2)
        else:  # 灰度
            colored_change2 = self._gray_to_color(semantic_change2)
            axes[1, 0].imshow(colored_change2)
        axes[1, 0].set_title("Phase 2 Semantic Change")
        axes[1, 0].axis('off')

        # 5. 二值变化图
        axes[1, 1].imshow(binary_change, cmap='gray')
        axes[1, 1].set_title("Binary Change Map")
        axes[1, 1].axis('off')

        # 6. 叠加可视化
        overlay = img1.copy()
        # 将变化区域标记为红色
        change_mask = binary_change.astype(bool)
        if len(overlay.shape) == 3:  # RGB
            overlay[change_mask] = [255, 0, 0]  # 红色
        else:  # 灰度
            # 转换为三通道
            overlay_rgb = np.stack([overlay, overlay, overlay], axis=2)
            overlay_rgb[change_mask] = [255, 0, 0]
            overlay = overlay_rgb
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Change Overlay (Red)")
        axes[1, 2].axis('off')

        plt.tight_layout()

        # 保存
        vis_path = os.path.join(output_dir, "visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存可视化图: {vis_path}")

    def _gray_to_color(self, gray_image):
        if len(gray_image.shape) == 2:  # 二维灰度图
            # 创建彩色图像
            colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

            # 获取唯一类别
            unique_classes = np.unique(gray_image)

            for class_value in unique_classes:
                if class_value > 0:  # 跳过背景
                    # 从CLASS_INDEX反向查找类别名称
                    class_name = None
                    for name, idx in self.CLASS_INDEX.items():
                        if idx == class_value:
                            class_name = name
                            break

                    if class_name and class_name in self.CLASS_COLORS:
                        color = self.CLASS_COLORS[class_name]
                    else:
                        # 如果没有找到颜色，生成一个
                        np.random.seed(class_value + 42)
                        color = [int(x) for x in np.random.randint(0, 256, 3)]

                    mask = gray_image == class_value
                    colored_image[mask] = color

            return colored_image
        else:
            return gray_image

    def run_sequential_pipeline(self, refine_factor=6, change_threshold=0.3):
        """运行顺序多类别变化检测流程"""
        print("=" * 60)
        print("开始运行顺序多类别变化检测流程")
        print("=" * 60)

        all_results = []

        result = self.process_single_class(
            class_name=self.all_classes,  # 这里传递的是列表，但process_single_class期望单个字符串
            refine_factor=refine_factor,
            change_threshold=change_threshold
        )
        if result is not None:
            all_results.append(result)

        if len(all_results) == 0:
            print("没有检测到任何类别的变化")
            return None

        change_results = self.visualize_semantic_change_maps(all_results)
        print("\n" + "=" * 60)
        print("✓ 顺序多类别变化检测流程完成!")
        print(f"结果保存到: {self.output_dir}")
        print("=" * 60)

        return all_results, change_results

# 主程序
if __name__ == "__main__":
    from tqdm import tqdm
    from contextlib import redirect_stdout, redirect_stderr

    IMAGE1_FOLDER = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/im1"
    IMAGE2_FOLDER = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/im2"
    OUTPUT_DIR = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/Instruct4/final_change"
    INSTRUCTION_TEXT = ['new building and disappeared sports ground']

    pairs = []
    t1_files = {}
    for ext in ("*.png", "*.jpg", "*.tif", "*.tiff"):
        for p in glob.glob(os.path.join(IMAGE1_FOLDER, ext)):
            t1_files[os.path.basename(p)] = p
    for name, p1 in t1_files.items():
        p2 = os.path.join(IMAGE2_FOLDER, name)
        if os.path.exists(p2):
            pairs.append((p1, p2, name))
    print(f"找到 {len(pairs)} 对影像")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            shared_model = build_sam3_image_model()

    try:
        dummy_p = Sam3Processor(shared_model, confidence_threshold=0.15)
        dummy = Image.new('RGB', (1008, 1008), color=(128, 128, 128))
        s = dummy_p.set_image(dummy)
        dummy_p.reset_all_prompts(s)
        dummy_p.set_text_prompt(state=s, prompt="building")
        del s, dummy_p
        torch.cuda.empty_cache()
    except Exception:
        pass

    success, failed, skipped = 0, 0, 0
    pbar = tqdm(total=len(pairs), desc="处理进度", unit="对", ncols=100)

    for p1, p2, fname in pairs:
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.png")

        # 跳过已处理的
        if os.path.exists(out_path):
            skipped += 1
            pbar.set_postfix({'成功': success, '失败': failed, '跳过': skipped})
            pbar.update(1)
            continue

        try:
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    detector = SequentialMultiClassChangeDetection(
                        image1_path=p1, image2_path=p2,
                        instruction_text=INSTRUCTION_TEXT,
                        output_dir="/tmp/_batch_inst5",
                        use_adaptive_threshold=True,
                        model=shared_model
                    )
                    detector.run_sequential_pipeline(
                        refine_factor=2, change_threshold=0.6)

            src = "/tmp/_batch_inst5/adaptive_semantic_change_raw.png"
            if os.path.exists(src):
                shutil.copy2(src, out_path)
            shutil.rmtree("/tmp/_batch_inst5", ignore_errors=True)
            success += 1
        except Exception:
            failed += 1
            shutil.rmtree("/tmp/_batch_inst5", ignore_errors=True)

        try: del detector
        except NameError: pass
        plt.close('all')
        torch.cuda.empty_cache()
        gc.collect()

        pbar.set_postfix({'成功': success, '失败': failed, '跳过': skipped})
        pbar.update(1)

    pbar.close()
    print(f"\n完成: 成功={success}, 失败={failed}, 跳过={skipped}")
