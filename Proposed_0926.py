"""
Proposed_0926.py —— 多提示词竞争语义分割 + 自适应阈值的变化检测管线
（基于 Proposed_multiclass_自动确定置信度阈值_消失类别.py 改进）

核心改进：
1. 语义分割阶段采用 水体提取_批量_原始检测代码.py 中的多提示词竞争机制：
   - 每个类别构建一组同义词提示词（如 water: river,lake,water,pond,sea,...）
   - 同时构建一个"干扰词组"（所有非目标地物词条 + 通用干扰词）
   - 逐词求 SAM3 的掩码 logits × 得分，按类别聚合后做 softmax/argmax 竞争
   - 只有目标类在竞争中"赢过"干扰类且概率超过阈值的像素才被保留，
     从而抑制"背景被误判为目标类"的误检（旧版本误检根源）
2. 竞争后的 softmax 概率作为每类概率图（修复了旧版本把原始 logits 当概率的问题）
3. 自适应阈值管理器（AdaptiveThresholdManagerV2）作用于竞争概率图，
   沿用"高阈值优先、无检测回退低阈值"的策略
4. 下游流程（实例级变化检测、超像素精细判定、指令语义可视化）与原管线一致

用法：
    python Proposed_0926.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
from scipy import ndimage
import torch.nn.functional as F
import math
import gc
import glob
import shutil

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from TextTransV3 import enhanced_CDTT
from SurperPixelSegmentation import dual_temporal_hierarchical_segmentation as FineSeg
from AdaptiveThresholdManager import AdaptiveThresholdManagerV2

# ============================================================================
# 类别同义词词典（SECOND 数据集的 6 个类别）
# 目标类的提示词组 = 该类的同义词；干扰词组 = 其他所有类别的同义词 + 通用干扰词
# ============================================================================
CLASS_SYNONYMS = {
    'bare land': 'bare land,soil,land,ground,unpaved,dirt',
    'low vegetation': 'low vegetation,vegetation,grass,shrub,bush,lawn,crop',
    'building': 'building,roof,house,construction',
    'water': 'river,lake,water,pond,sea,ocean,canal,reservoir,stream,pool,wetland',
    'tree': 'tree,forest,woods',
    'sports ground': 'sports ground,playground,court,stadium,track',
}

# 通用干扰词（不属于任何目标类，负责"吃掉"背景/阴影/道路等易混区域）
GENERIC_DISTRACTOR_TERMS = 'ground,road,shadow,cloud'


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def refine_overlapping_masks(masks_data, h, w, iou_threshold=0.3):
    """处理类别间重叠：将重叠组按连通域拆分，每个连通域归属得分最高的实例"""
    n_masks = len(masks_data)
    refined_masks = []
    refined_classes = []
    refined_scores = []
    if n_masks == 0:
        return refined_masks, refined_classes, refined_scores

    iou_matrix = np.zeros((n_masks, n_masks))
    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            iou = compute_iou(masks_data[i][1], masks_data[j][1])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    adjacency = {}
    for i in range(n_masks):
        adjacency[i] = [j for j in range(n_masks) if i != j and iou_matrix[i, j] > iou_threshold]

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
            continue

        overlapping_group = [i] + overlapping
        for idx in overlapping_group:
            processed[idx] = True

        group_masks = []
        group_classes = []
        group_scores = []
        for idx in overlapping_group:
            class_id, mask, score, prob_map = masks_data[idx]
            group_masks.append(mask)
            group_classes.append(class_id)
            group_scores.append(score)

        combined_mask = np.zeros((h, w), dtype=bool)
        for mask in group_masks:
            combined_mask = np.logical_or(combined_mask, mask)

        labeled, num_features = ndimage.label(combined_mask)

        for label_idx in range(1, num_features + 1):
            component_mask = (labeled == label_idx)

            overlapping_indices = []
            overlapping_scores = []
            for mask_idx, mask in enumerate(group_masks):
                if compute_iou(mask, component_mask) > 0.1:
                    overlapping_indices.append(mask_idx)
                    overlapping_scores.append(group_scores[mask_idx])

            if len(overlapping_indices) == 0:
                continue
            if len(overlapping_indices) == 1:
                best_idx = overlapping_indices[0]
            else:
                best_score = -1
                best_idx = overlapping_indices[0]
                for mask_idx, score in zip(overlapping_indices, overlapping_scores):
                    if score > best_score:
                        best_score = score
                        best_idx = mask_idx

            refined_masks.append(component_mask)
            refined_classes.append(group_classes[best_idx])
            refined_scores.append(group_scores[best_idx])

    return refined_masks, refined_classes, refined_scores


# ============================================================================
# 多提示词竞争语义分割器（基于 OVCDSegmentor 思路，复用外部 SAM3 模型）
# ============================================================================
class CompetitiveSegmentor:
    """
    多提示词竞争分割器：
    class_names = [干扰词组, 目标类1词组, 目标类2词组, ...]
    每个同义词作为一个 query 求 SAM3 的掩码 logits × 得分，
    按类别聚合后对像素做 softmax/argmax 竞争，
    目标类必须"赢过"干扰组才会被保留 -> 抑制背景误检。
    """

    def __init__(self, model, class_names, device='cuda', confidence_threshold=0.1):
        self.device = device
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)

        self.query_words = []
        self.query_to_cls = []
        for cls_idx, name_group in enumerate(class_names):
            for syn in name_group.split(','):
                syn = syn.strip()
                if syn:
                    self.query_words.append(syn)
                    self.query_to_cls.append(cls_idx)

        self.num_classes = len(class_names)

    @torch.no_grad()
    def forward_single(self, image):
        """
        Args:
            image: PIL Image
        Returns:
            cls_logits: [num_classes, H, W] 类别级聚合 logits（含干扰组，索引0）
            probs:      [num_classes, H, W] softmax 竞争概率
        """
        w, h = image.size
        dev = self.device
        seg_logits = torch.zeros((len(self.query_words), h, w), device=dev)

        state = self.processor.set_image(image)

        for q_idx, query_word in enumerate(self.query_words):
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=query_word, state=state)

            # 实例头：每个实例掩码概率 × 实例得分，取最大
            m = state.get('masks_logits')
            if m is not None and m.shape[0] > 0:
                s = state.get('scores')
                if s is None:
                    s = torch.ones(m.shape[0], device=dev)
                for inst_id in range(m.shape[0]):
                    inst_logits = m[inst_id].squeeze()
                    if inst_logits.shape != (h, w):
                        inst_logits = F.interpolate(
                            inst_logits.view(1, 1, *inst_logits.shape),
                            size=(h, w), mode='bilinear', align_corners=False,
                        ).squeeze()
                    seg_logits[q_idx] = torch.maximum(seg_logits[q_idx], inst_logits * s[inst_id])

            # 语义头（若模型提供 semantic_mask_logits，则一并融合）
            sem = state.get('semantic_mask_logits')
            if sem is not None:
                if sem.shape[-2:] != (h, w):
                    sem = F.interpolate(
                        sem if sem.ndim == 4 else sem.unsqueeze(0).unsqueeze(0),
                        size=(h, w), mode='bilinear', align_corners=False,
                    ).squeeze()
                seg_logits[q_idx] = torch.maximum(seg_logits[q_idx], sem)

        # 同义词 -> 类别级聚合（max over synonyms）
        cls_logits = torch.zeros((self.num_classes, h, w), device=dev)
        for q_idx, cls_idx in enumerate(self.query_to_cls):
            cls_logits[cls_idx] = torch.maximum(cls_logits[cls_idx], seg_logits[q_idx])

        probs = F.softmax(cls_logits, dim=0)
        return cls_logits, probs


# ============================================================================
# 多类别语义分割（竞争版）：一次前向覆盖全部目标类，逐类应用自适应阈值
# ============================================================================
def multi_class_semantic_segmentation_competitive(
        image, prompts, colors=None, output_dir=None, model=None,
        adaptive_threshold_manager=None,
        prob_thd_high=0.5, prob_thd_low=0.3,
        min_instance_area=100, device='cuda'):
    """
    多类别语义分割（多提示词竞争 + 自适应阈值）

    Args:
        image: PIL Image 或 np.ndarray
        prompts: 类别名列表（如 ['water']）
        adaptive_threshold_manager: 自适应阈值管理器（为 None 时用固定概率阈值）
        prob_thd_high / prob_thd_low: 固定模式下的高/低概率阈值
    Returns:
        semantic_map, colored_seg, final_class_masks, class_probs_maps, class_max_probs
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
        image_pil = Image.fromarray(image_np)
    else:
        image_pil = image
        image_np = np.array(image)
    h, w = image_np.shape[:2]

    if model is None:
        model = build_sam3_image_model()

    # ---- 构建竞争提示词组: [干扰组] + [每类目标词组] ----
    target_groups = [CLASS_SYNONYMS.get(cls, cls) for cls in prompts]
    distractor_terms = []
    for cls, syn in CLASS_SYNONYMS.items():
        if cls in prompts:
            continue
        distractor_terms.append(syn)
    distractor_terms.append(GENERIC_DISTRACTOR_TERMS)
    distractor_group = ','.join(distractor_terms)
    class_names = [distractor_group] + target_groups

    print(f"  竞争提示词: 干扰组({len(distractor_group.split(','))}词) + {len(target_groups)}个目标类")

    segmentor = CompetitiveSegmentor(model, class_names, device=device, confidence_threshold=0.1)
    cls_logits, probs = segmentor.forward_single(image_pil)

    pred_label = probs.argmax(dim=0).cpu().numpy()  # 0 = 干扰/背景

    all_masks = []
    all_class_ids = []
    all_scores = []
    all_probs = []
    class_probs_maps = []
    class_max_probs = []
    class_instance_data = [[] for _ in range(len(prompts))]

    for class_id, class_name in enumerate(prompts):
        print(f"处理类别 {class_id + 1}/{len(prompts)}: {class_name}")

        prob_map = probs[class_id + 1].cpu().numpy()          # 竞争后的软概率
        raw_map = cls_logits[class_id + 1].cpu().numpy()      # 聚合 logits（用于激活下限）

        # ---- 自适应/固定阈值确定 ----
        if adaptive_threshold_manager is not None:
            thresholds = adaptive_threshold_manager.get_adaptive_threshold(image_pil, class_name)
            class_high = float(thresholds['high_threshold'])
            class_low = float(thresholds['low_threshold'])
        else:
            class_high = prob_thd_high
            class_low = prob_thd_low

        # 高阈值优先；无检测时回退低阈值（与原管线策略一致）
        activated = raw_map > 0.05  # 原始激活下限，防止近零区域在低阈值下误入
        class_mask = None
        used_threshold = class_high
        used_tag = '高'
        for thd, tag in ((class_high, '高'), (class_low, '低')):
            cand = (pred_label == class_id + 1) & (prob_map > thd) & activated
            if cand.any():
                class_mask = cand
                used_threshold = thd
                used_tag = tag
                break
        if class_mask is None:
            class_mask = np.zeros((h, w), dtype=bool)
            used_tag = '无检测'

        print(f"  {used_tag}阈值={used_threshold:.4f}, 检测像素={int(class_mask.sum())}")

        class_probs_maps.append(prob_map)
        class_max_probs.append(prob_map)

        # ---- 实例提取：连通域 + 最小面积过滤 ----
        if class_mask.any():
            labeled, num = ndimage.label(class_mask)
            for lab in range(1, num + 1):
                comp = (labeled == lab)
                if comp.sum() < min_instance_area:
                    continue
                score = float(prob_map[comp].mean())
                all_masks.append(comp)
                all_class_ids.append(class_id)
                all_scores.append(score)
                all_probs.append(prob_map)
                class_instance_data[class_id].append({
                    'mask': comp, 'score': score, 'prob': prob_map
                })
        else:
            class_instance_data[class_id] = []

    # ---- 类别间重叠精炼（与原管线一致）----
    masks_data_for_refinement = [
        (cid, m, s, p) for m, cid, s, p in zip(all_masks, all_class_ids, all_scores, all_probs)
    ]
    refined_masks, refined_classes, refined_scores = refine_overlapping_masks(
        masks_data_for_refinement, h, w, iou_threshold=0.3
    )

    # ---- 组织最终结果 ----
    final_class_masks = []
    final_instance_masks_per_class = [[] for _ in range(len(prompts))]
    final_scores_per_class = [[] for _ in range(len(prompts))]
    detection_counts = [0] * len(prompts)
    confidence_scores = [0.0] * len(prompts)

    semantic_map = np.zeros((h, w), dtype=np.uint8)
    instance_map = np.zeros((h, w), dtype=np.uint32)
    instance_counter = 1

    for mask, class_id, score in zip(refined_masks, refined_classes, refined_scores):
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
            confidence_scores[class_id] = float(np.mean(final_scores_per_class[class_id]))
        else:
            final_class_masks.append(np.zeros((h, w), dtype=bool))

    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(prompts)):
        if detection_counts[class_id] > 0:
            mask = final_class_masks[class_id]
            if class_id < len(colors):
                colored_seg[mask] = colors[class_id]

    # ---- 清理 GPU 变量 ----
    try:
        del cls_logits, probs, pred_label
    except NameError:
        pass
    try:
        del all_masks, all_class_ids, all_scores, all_probs, class_instance_data
    except NameError:
        pass
    try:
        del final_instance_masks_per_class, final_scores_per_class
    except NameError:
        pass
    try:
        del masks_data_for_refinement
    except NameError:
        pass
    torch.cuda.empty_cache()
    gc.collect()

    return semantic_map, colored_seg, final_class_masks, class_probs_maps, class_max_probs


# ============================================================================
# 顺序多类别变化检测主类（下游流程与原管线一致）
# ============================================================================
class SequentialMultiClassChangeDetection:
    """按类别顺序处理的多类别变化检测（竞争分割版）"""

    CLASS_INDEX = {
        'bare land': 1,
        'low vegetation': 2,
        'building': 3,
        'water': 4,
        'tree': 5,
        'sports ground': 6
    }

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

        # 创建自适应阈值管理器（竞争分割模式下阈值作用于竞争概率图）
        self.use_adaptive_threshold = use_adaptive_threshold
        if use_adaptive_threshold:
            print("  ✓ 启用自适应阈值管理器（作用于竞争概率图）")
            self.adaptive_threshold_manager = AdaptiveThresholdManagerV2(
                model=self.model,
                processor_class=Sam3Processor,
                probe_threshold=0.01,
                anti_miss_bias=0.7,
                min_coverage_ratio=0.01,
                max_retry=3,
                enable_feedback=True,
                verbose=True
            )
        else:
            print("  使用固定概率阈值 (高=0.5, 低=0.3)")
            self.adaptive_threshold_manager = None
        self.prob_thd_high = 0.5
        self.prob_thd_low = 0.3

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
            instruction_text = ['building change']

        trans_results = enhanced_CDTT(instruction_text)
        self.trans_results = trans_results

        self.change_semantics = None
        self.change_direction = None
        self.transitions = []
        self.involved_classes = []

        if trans_results and len(trans_results) > 0:
            trans_item = trans_results[0]
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
                if 'data' not in trans_item:
                    continue
                phase1_instr = trans_item['data'].get("phase1_instruction", "")
                phase2_instr = trans_item['data'].get("phase2_instruction", "")

                classes_t1 = self._parse_classes(phase1_instr)
                classes_t2 = self._parse_classes(phase2_instr)

                for cls in classes_t1 + classes_t2:
                    if cls and cls not in self.involved_classes:
                        self.involved_classes.append(cls)

        # 动态生成CLASS_INDEX
        self.CLASS_INDEX = {}
        for i, class_name in enumerate(self.involved_classes, 1):
            self.CLASS_INDEX[class_name] = i

        # 动态生成CLASS_COLORS
        self.CLASS_COLORS = {}
        base_colors = [
            [128, 128, 128],  # 灰色
            [0, 128, 0],  # 深绿色
            [128, 0, 0],  # 深红色
            [0, 0, 255],  # 蓝色
            [0, 255, 0],  # 绿色
            [255, 0, 0],  # 红色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 紫色
            [0, 255, 255],  # 青色
            [128, 0, 128],  # 紫色
        ]

        for i, class_name in enumerate(self.involved_classes):
            if i < len(base_colors):
                self.CLASS_COLORS[class_name] = base_colors[i]
            else:
                np.random.seed(i + 42)
                self.CLASS_COLORS[class_name] = [int(x) for x in np.random.randint(0, 256, 3)]

        print(f"✓ 动态生成了 {len(self.involved_classes)} 个类别:")
        for i, class_name in enumerate(self.involved_classes, 1):
            class_idx = self.CLASS_INDEX[class_name]
            color = self.CLASS_COLORS[class_name]
            print(f"  类别{i}: {class_name} (索引={class_idx}, 颜色={color})")

        self.all_classes = self.involved_classes.copy()

        # 4. 为每个类别准备颜色
        self.colors = []
        for cls in self.all_classes:
            if cls in self.CLASS_COLORS:
                self.colors.append(self.CLASS_COLORS[cls])
            else:
                np.random.seed(hash(cls) % 1000)
                self.colors.append([int(x) for x in np.random.randint(0, 256, 3)])

    def _parse_classes(self, instruction):
        """解析指令中的类别"""
        if not instruction or instruction.strip() == "":
            return []

        instruction = instruction.strip()
        separators = [';', ',', '、', '和', '与']
        for sep in separators:
            if sep in instruction:
                parts = [p.strip() for p in instruction.split(sep) if p.strip()]
                if len(parts) > 0:
                    return parts
        return [instruction]

    def find_instances(self, binary_img):
        """从二值图像中找到所有实例"""
        num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))

        instances = {}
        for label in range(1, num_labels):
            instance_mask = (labels == label)
            if np.sum(instance_mask) > 0:
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

        contours, _ = cv2.findContours(instance_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        distances = np.zeros(num_angles)
        for i in range(num_angles):
            angle = math.radians(i * 5)
            dx = math.cos(angle)
            dy = math.sin(angle)
            for d in range(1, int(math.sqrt(height ** 2 + width ** 2))):
                x = int(cx + d * dx)
                y = int(cy + d * dy)
                if x < 0 or x >= width or y < 0 or y >= height or not instance_mask[y, x]:
                    distances[i] = d
                    break
        if np.max(distances) > 0:
            distances = distances / np.max(distances)
        return distances

    def compute_similarity(self, sig1, sig2):
        """计算两个形状签名的相似度"""
        if sig1 is None or sig2 is None:
            return 0
        dot_product = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cosine_sim = dot_product / (norm1 * norm2)
        return (cosine_sim + 1) / 2 * 100

    def get_multi_class_segmentation(self, image, phase_name=""):
        print(f"  ✓ 进行多类别语义分割（竞争模式）{phase_name}...")

        semantic_map, colored_seg, class_masks, class_probs, each_class_max_probs = \
            multi_class_semantic_segmentation_competitive(
                image=image,
                prompts=self.all_classes,
                colors=self.colors,
                output_dir=os.path.join(self.output_dir, f"semantic_{phase_name}") if phase_name else None,
                model=self.model,
                adaptive_threshold_manager=self.adaptive_threshold_manager
                if self.use_adaptive_threshold else None,
                prob_thd_high=self.prob_thd_high,
                prob_thd_low=self.prob_thd_low,
            )

        return semantic_map, colored_seg, class_masks, class_probs, each_class_max_probs

    def detect_initial_changes(self, semantic_map1, semantic_map2, similarity_threshold=90):
        """基于重叠、形状相似度和面积相似度的变化检测"""
        print("\n" + "=" * 60)
        print("2. 检测初始变化...")

        building_t1 = semantic_map1.astype(np.uint8)
        building_t2 = semantic_map2.astype(np.uint8)

        building_t1 = (building_t1 > 0).astype(np.uint8) * 255
        building_t2 = (building_t2 > 0).astype(np.uint8) * 255

        instances_t1, labels_t1 = self.find_instances(building_t1)
        instances_t2, labels_t2 = self.find_instances(building_t2)

        for label, instance in instances_t1.items():
            instance['shape_sig'] = self.compute_shape_signature(instance['mask'], instance['centroid'])
        for label, instance in instances_t2.items():
            instance['shape_sig'] = self.compute_shape_signature(instance['mask'], instance['centroid'])

        changed_instances = set()
        unchanged_pairs = []
        changed_pairs = []

        for label in instances_t1:
            changed_instances.add(('t1', label))
        for label in instances_t2:
            changed_instances.add(('t2', label))

        for label, instance in instances_t1.items():
            instance['area'] = np.sum(instance['mask'])
        for label, instance in instances_t2.items():
            instance['area'] = np.sum(instance['mask'])

        overlap_change = np.zeros_like(building_t1, dtype=np.uint8)
        for label1, instance1 in instances_t1.items():
            for label2, instance2 in instances_t2.items():
                overlap = np.sum(instance1['mask'] & instance2['mask'])
                if overlap > 0:
                    sim = self.compute_similarity(instance1['shape_sig'], instance2['shape_sig'])
                    area1 = instance1['area']
                    area2 = instance2['area']
                    area_sim = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
                    area_diff = abs(area1 - area2) / ((area1 + area2) / 2) if (area1 + area2) > 0 else float('inf')

                    print(f"实例对 T1-{label1} 与 T2-{label2}: ")
                    print(f"  重叠面积: {overlap}, 形状相似度: {sim:.2f}%")
                    print(f"  时相1面积: {area1}, 时相2面积: {area2}")
                    print(f"  面积相似度: {area_sim:.2f}, 面积差异百分比: {area_diff:.2f}")

                    area_threshold = 0.6
                    shape_sim_threshold = similarity_threshold

                    if area_sim >= area_threshold and sim >= shape_sim_threshold:
                        print(f"  ✓ 不变化判定: 面积相似({area_sim:.2f}≥{area_threshold})且形状相似({sim:.2f}%≥{shape_sim_threshold}%)")
                        is_unchanged = True
                    else:
                        print(f"  ✗ 变化判定")
                        is_unchanged = False

                    if is_unchanged:
                        if ('t1', label1) in changed_instances:
                            changed_instances.remove(('t1', label1))
                        if ('t2', label2) in changed_instances:
                            changed_instances.remove(('t2', label2))
                        unchanged_pairs.append((label1, label2, sim, area_sim))
                    else:
                        changed_pairs.append((label1, label2, sim, area_sim))
                        overlap_change[instance1['mask']] = 255
                        overlap_change[instance2['mask']] = 255

        print(f"\n结果统计:")
        print(f"不变化实例对数: {len(unchanged_pairs)}")
        print(f"变化实例对数: {len(changed_pairs)}")
        print(f"总变化实例数: {len(changed_instances)}")

        change_map = np.zeros_like(building_t1, dtype=np.uint8)
        non_overlap_change = np.zeros_like(building_t1, dtype=np.uint8)
        for phase, label in changed_instances:
            if phase == 't1':
                mask = instances_t1[label]['mask']
            else:
                mask = instances_t2[label]['mask']

            change_map[mask] = 255
            overlap_area = np.sum(mask & (overlap_change > 0))
            if overlap_area == 0:
                non_overlap_change[mask] = 255
            else:
                non_overlap_change[mask] = 0
                overlap_change[mask] = 255

        overlap_change = change_map - non_overlap_change
        return overlap_change, non_overlap_change, change_map

    def analyze_overlap_objects(self, superpixel_labels, overlapping_changes_mask,
                                semantic_map1, semantic_map2):
        if np.sum(overlapping_changes_mask) == 0:
            print("  ✓ 无重叠变化区域，跳过精细判定")
            return np.zeros((self.height, self.width), dtype=bool), \
                   np.zeros((self.height, self.width), dtype=bool)

        if not isinstance(semantic_map1, dict) or not isinstance(semantic_map2, dict):
            print("  ⚠ 类别概率图不是字典格式，尝试转换...")
            try:
                if isinstance(semantic_map1, np.ndarray):
                    if len(semantic_map1.shape) == 2:
                        semantic_map1 = {'building': semantic_map1}
                    elif len(semantic_map1.shape) == 3:
                        semantic_map1 = {'building': semantic_map1[:, :, 0]}
                else:
                    building_mask_t1 = semantic_map1 > 0
                    semantic_map1 = {'building': building_mask_t1.astype(np.float32)}

                if isinstance(semantic_map2, np.ndarray):
                    if len(semantic_map2.shape) == 2:
                        semantic_map2 = {'building': semantic_map2}
                    elif len(semantic_map2.shape) == 3:
                        semantic_map2 = {'building': semantic_map2[:, :, 0]}
                else:
                    building_mask_t2 = semantic_map2 > 0
                    semantic_map2 = {'building': building_mask_t2.astype(np.float32)}

                print(f"  ✓ 已创建类别概率图字典: {list(semantic_map1.keys())}")
            except Exception as e:
                print(f"  ✗ 无法创建类别概率图字典: {e}")
                return np.zeros((self.height, self.width), dtype=bool), \
                       np.zeros((self.height, self.width), dtype=bool)

        overlap_binary = (overlapping_changes_mask > 0).astype(np.uint8) * 255

        num_initial_objects, labels_initial, stats_initial, _ = cv2.connectedComponentsWithStats(
            overlap_binary, connectivity=8
        )

        if num_initial_objects == 1:
            return np.zeros((self.height, self.width), dtype=bool), \
                   np.zeros((self.height, self.width), dtype=bool)

        overlap_objects_mask = np.zeros((self.height, self.width), dtype=bool)
        refined_overlap_changes = np.zeros((self.height, self.width), dtype=bool)

        for obj_id in range(1, num_initial_objects):
            initial_obj_mask = (labels_initial == obj_id)
            if np.sum(initial_obj_mask) == 0:
                continue

            obj_superpixel_labels = superpixel_labels.copy()
            obj_superpixel_labels[~initial_obj_mask] = 0

            unique_labels = np.unique(obj_superpixel_labels)
            unique_labels = unique_labels[unique_labels > 0]

            if len(unique_labels) == 0:
                refined_objects = [initial_obj_mask]
            else:
                refined_objects = []
                for label in unique_labels:
                    refined_obj_mask = (obj_superpixel_labels == label)
                    if np.sum(refined_obj_mask) > 0:
                        refined_objects.append(refined_obj_mask)

            print(f"初始对象{obj_id}: 精细分割为 {len(refined_objects)} 个子对象")

            for refined_mask in refined_objects:
                obj_probs_t1 = []
                obj_probs_t2 = []

                for class_name, prob_map_t1 in semantic_map1.items():
                    if class_name in semantic_map2:
                        prob_values_t1 = prob_map_t1[refined_mask]
                        prob_values_t2 = semantic_map2[class_name][refined_mask]
                        if len(prob_values_t1) > 0 and len(prob_values_t2) > 0:
                            obj_probs_t1.append(np.mean(prob_values_t1))
                            obj_probs_t2.append(np.mean(prob_values_t2))

                if obj_probs_t1 and obj_probs_t2:
                    obj_probs_t1_array = np.array(obj_probs_t1)
                    obj_probs_t2_array = np.array(obj_probs_t2)
                    change_prob = np.max(np.abs(obj_probs_t1_array - obj_probs_t2_array))

                    if change_prob > 0.6:
                        refined_overlap_changes[refined_mask] = True
                        overlap_objects_mask[refined_mask] = True

        return overlap_objects_mask, refined_overlap_changes

    def _get_class_superpixels(self, seg1_path, seg2_path, refine_factor=4):
        """超像素分割"""
        temp_dir = os.path.join(self.output_dir, "temp_superpixels")
        os.makedirs(temp_dir, exist_ok=True)
        temp_img1_path = os.path.join(temp_dir, "temp_phase1.png")
        temp_img2_path = os.path.join(temp_dir, "temp_phase2.png")
        Image.fromarray(self.image1_np).save(temp_img1_path)
        Image.fromarray(self.image2_np).save(temp_img2_path)
        try:
            superpixel_labels = FineSeg(
                image1_path=temp_img1_path,
                image2_path=temp_img2_path,
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

        shutil.rmtree(temp_dir, ignore_errors=True)
        return superpixel_labels

    def combine_changes(self, initial_changes, overlapping_changes_mask,
                        refined_overlap_changes, non_overlapping_changes_mask):
        refined_overlap_changes_int = refined_overlap_changes.astype(np.uint8)
        final_changes = refined_overlap_changes_int + non_overlapping_changes_mask
        final_changes[final_changes > 0] = 1
        return final_changes

    def process_single_class(self, class_name=None, refine_factor=4, change_threshold=0.3):
        # 获取语义分割结果
        semantic_map1, colored_seg1, class_masks1, class_probs1, each_class_max_probs1 = self.get_multi_class_segmentation(
            self.image1, phase_name="t1"
        )
        semantic_map2, colored_seg2, class_masks2, class_probs2, each_class_max_probs2 = self.get_multi_class_segmentation(
            self.image2, phase_name="t2"
        )

        seg_dir = os.path.join(self.output_dir, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)

        seg1_path = os.path.join(seg_dir, "seg1.png")
        seg2_path = os.path.join(seg_dir, "seg2.png")
        segRGB1_path = os.path.join(seg_dir, "segRGB1.png")
        segRGB2_path = os.path.join(seg_dir, "segRGB2.png")

        plt.imsave(seg1_path, semantic_map1, cmap='gray')
        plt.imsave(seg2_path, semantic_map2, cmap='gray')
        plt.imsave(segRGB1_path, colored_seg1)
        plt.imsave(segRGB2_path, colored_seg2)

        print(f"  ✓ 粗分割图已保存: {seg1_path}, {seg2_path}")

        class_configs = []
        for i, cls in enumerate(self.all_classes):
            class_configs.append({
                'name': cls.replace(' ', '_'),
                'index': i + 1,
                'prob_idx': i,
                'original_name': cls
            })

        print(f"  ✓ 处理 {len(class_configs)} 个类别: {[config['name'] for config in class_configs]}")

        results = {}
        class_maps = {}

        for config in class_configs:
            class_name = config['name']
            original_name = config['original_name']
            class_idx = config['index']
            prob_idx = config['prob_idx']

            if prob_idx >= len(each_class_max_probs1) or prob_idx >= len(each_class_max_probs2):
                print(f"  ⚠ 警告: 类别 {class_name} 的概率图索引 {prob_idx} 超出范围，跳过")
                continue

            class_prob_map1 = each_class_max_probs1[prob_idx]
            class_prob_map2 = each_class_max_probs2[prob_idx]

            sem_map1 = (semantic_map1 == class_idx).astype(np.uint8)
            sem_map2 = (semantic_map2 == class_idx).astype(np.uint8)

            sem_map1_path = os.path.join(seg_dir, f"{class_name}_semantic_map1.png")
            sem_map2_path = os.path.join(seg_dir, f"{class_name}_semantic_map2.png")

            plt.imsave(sem_map1_path, sem_map1, cmap='gray')
            plt.imsave(sem_map2_path, sem_map2, cmap='gray')

            class_maps[f'{class_name}_semantic_map1'] = sem_map1
            class_maps[f'{class_name}_semantic_map2'] = sem_map2

            superpixel_labels = self._get_class_superpixels(sem_map1_path, sem_map2_path, refine_factor=refine_factor)

            overlap_change, non_overlap_change, change_map = self.detect_initial_changes(
                sem_map1.squeeze(), sem_map2.squeeze(), similarity_threshold=98)

            if np.sum(overlap_change) > 0:
                overlap_object_changed, refined_overlap_changes = self.analyze_overlap_objects(
                    superpixel_labels, overlap_change, class_prob_map1.squeeze(), class_prob_map2.squeeze())
            else:
                refined_overlap_changes = np.zeros((self.height, self.width), dtype=bool)
                overlap_objects_mask = np.zeros((self.height, self.width), dtype=bool)
                print(f"  ✓ 类别 {class_name} 无重叠变化对象需要分析")

            final_changes = self.combine_changes(change_map, overlap_change,
                                                 refined_overlap_changes, non_overlap_change)

            change_map_path = os.path.join(seg_dir, f"{class_name}_initial_change.png")
            binary_change_path = os.path.join(seg_dir, f"{class_name}_final_change.png")

            plt.imsave(change_map_path, change_map, cmap='gray')
            Image.fromarray((final_changes * 255).astype(np.uint8)).save(binary_change_path)

            results[class_name] = {
                'semantic_map1': sem_map1,
                'semantic_map2': sem_map2,
                'superpixel_labels': superpixel_labels,
                'change_map': change_map,
                'binary_change_map': final_changes
            }

            print(f"✓ 处理类别 {original_name} 完成")

        all_binary_maps = []
        all_initial_map = []

        for config in class_configs:
            class_name = config['name']
            if class_name in results:
                all_binary_maps.append(results[class_name]['binary_change_map'].astype(np.uint8))
                all_initial_map.append(results[class_name]['change_map'].astype(np.uint8))
            else:
                all_binary_maps.append(np.zeros((self.height, self.width), dtype=np.uint8))
                all_initial_map.append(np.zeros((self.height, self.width), dtype=np.uint8))

        all_binary_change_maps = [map.astype(np.uint8) for map in all_binary_maps]
        all_initial_map = [map.astype(np.uint8) for map in all_initial_map]

        if all_binary_change_maps:
            combined_change_tensor = np.stack(all_binary_change_maps, axis=0)
            combined_initial_change_tensor = np.stack(all_initial_map, axis=0)
        else:
            combined_change_tensor = np.zeros((len(class_configs), self.height, self.width), dtype=np.uint8)
            combined_initial_change_tensor = np.zeros((len(class_configs), self.height, self.width), dtype=np.uint8)

        combined_change_tensor = torch.from_numpy(combined_change_tensor).float()
        combined_initial_change_tensor = torch.from_numpy(combined_initial_change_tensor).float()

        combined_binary_map = np.zeros((self.height, self.width), dtype=np.uint8)
        combined_initial_binary_map = np.zeros((self.height, self.width), dtype=np.uint8)

        for i, config in enumerate(class_configs):
            if i < len(all_binary_change_maps):
                combined_binary_map = np.logical_or(combined_binary_map, all_binary_change_maps[i]).astype(np.uint8)
                combined_initial_binary_map = np.logical_or(combined_initial_binary_map, all_initial_map[i]).astype(
                    np.uint8)

        combined_binary_path = os.path.join(seg_dir, "combined_binary_change.png")
        Image.fromarray((combined_binary_map * 255).astype(np.uint8)).save(combined_binary_path)
        print(f"✓ 保存合并的二值变化图: {combined_binary_path}")

        combined_initial_binary_path = os.path.join(seg_dir, "combined_initial_binary_change.png")
        Image.fromarray((combined_initial_binary_map * 255).astype(np.uint8)).save(combined_initial_binary_path)
        print(f"✓ 保存合并的初始二值变化图: {combined_initial_binary_path}")

        return {
            'class_names': [config['name'] for config in class_configs],
            'change_tensor': combined_change_tensor,
            'combined_binary_map': combined_binary_map,
            'combined_initial_binary_map': combined_initial_binary_map,
            'semantic_map1': semantic_map1,
            'semantic_map2': semantic_map2,
            'individual_maps': {config['name']: results.get(config['name'], {}).get('binary_change_map',
                                np.zeros((self.height, self.width), dtype=np.uint8))
                                for config in class_configs},
            'change_maps': {config['name']: (
                results.get(config['name'], {}).get('change_map', np.zeros((self.height, self.width), dtype=np.uint8)),
                results.get(config['name'], {}).get('binary_change_map',
                np.zeros((self.height, self.width), dtype=np.uint8)))
                for config in class_configs
            }
        }

    def visualize_semantic_change_maps(self, all_results, output_dir=None):
        """可视化语义变化检测结果 - 增强复合变化处理"""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if len(all_results) == 0:
            print("没有结果可以可视化")
            return None

        result = all_results[0]
        semantic_map1 = result['semantic_map1']
        semantic_map2 = result['semantic_map2']
        individual_maps = result['individual_maps']
        instance_change_map = result['combined_binary_map']

        height, width = semantic_map1.shape
        final_adaptive_change_map = np.zeros((height, width, 3), dtype=np.uint8)
        legend_info = []
        color_idx = 0

        def get_random_color(seed):
            np.random.seed(seed + 42)
            return [int(x) for x in np.random.randint(50, 256, 3)]

        print("\n" + "=" * 60)
        print("自适应变化信息输出")
        print("=" * 60)

        print(f"动态CLASS_INDEX: {self.CLASS_INDEX}")
        print(f"动态CLASS_COLORS: {self.CLASS_COLORS}")

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

            # ==================== 复合变化处理 ====================
            if change_type == 'composite' and transitions and len(transitions) > 0:
                print(f"  → 检测到复合变化指令")
                for transition in transitions:
                    if isinstance(transition, str):
                        if '→' in transition:
                            parts = transition.split('→')
                            if len(parts) == 2:
                                from_class = parts[0].strip()
                                to_class = parts[1].strip()
                            else:
                                print(f"    跳过无效的转换格式: {transition}")
                                continue
                        elif '↔' in transition:
                            parts = transition.split('↔')
                            if len(parts) == 2:
                                class1 = parts[0].strip()
                                class2 = parts[1].strip()
                                print(f"    处理双向转换方向1: {class1} → {class2}")
                                self._process_conversion(class1, class2, semantic_map1, semantic_map2,
                                                         individual_maps, final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1
                                print(f"    处理双向转换方向2: {class2} → {class1}")
                                self._process_conversion(class2, class1, semantic_map1, semantic_map2,
                                                         individual_maps, final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1
                                continue
                            else:
                                print(f"    跳过无效的双向转换格式: {transition}")
                                continue
                        else:
                            print(f"    跳过不支持的转换格式: {transition}")
                            continue
                    elif isinstance(transition, dict):
                        from_class = transition.get('from', '')
                        to_class = transition.get('to', '')
                    else:
                        print(f"    跳过不支持的转换格式: {type(transition)}")
                        continue

                    print(f"    处理复合转换: {from_class} → {to_class}")

                    from_idx = self.CLASS_INDEX.get(from_class, 0)
                    to_idx = self.CLASS_INDEX.get(to_class, 0)
                    if from_class == "background":
                        from_idx = 0
                    if to_class == "background":
                        to_idx = 0

                    if from_idx >= 0 and to_idx >= 0:
                        from_class_name = from_class.replace(' ', '_')
                        to_class_name = to_class.replace(' ', '_')
                        from_change = individual_maps.get(from_class_name,
                                                          np.zeros((height, width), dtype=np.uint8))
                        to_change = individual_maps.get(to_class_name, np.zeros((height, width), dtype=np.uint8))
                        t1_is_from = (semantic_map1 == from_idx)
                        t2_is_to = (semantic_map2 == to_idx)
                        combined_change_region = ((from_change > 0) | (to_change > 0))
                        transition_region = t1_is_from & t2_is_to & combined_change_region

                        pixel_count = np.sum(transition_region)
                        print(f"    发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

                        if np.any(transition_region):
                            color = get_random_color(color_idx)
                            color_idx += 1
                            final_adaptive_change_map[transition_region] = color
                            legend_info.append({'color': color, 'label': f'{from_class} → {to_class}'})
                    else:
                        print(f"    ⚠ 类别索引无效: {from_class}={from_idx}, {to_class}={to_idx}")

            # ==================== 情况1：类别转换 (A -> B) ====================
            elif transitions and len(transitions) > 0 and change_type != 'composite':
                print(f"  → 检测到类别转换指令")
                for transition in transitions:
                    if isinstance(transition, str):
                        if '→' in transition:
                            parts = transition.split('→')
                            if len(parts) == 2:
                                from_class = parts[0].strip()
                                to_class = parts[1].strip()
                            else:
                                print(f"    跳过无效的转换格式: {transition}")
                                continue
                        elif '↔' in transition:
                            parts = transition.split('↔')
                            if len(parts) == 2:
                                class1 = parts[0].strip()
                                class2 = parts[1].strip()
                                print(f"    处理双向转换方向1: {class1} → {class2}")
                                self._process_conversion(class1, class2, semantic_map1, semantic_map2,
                                                         individual_maps, final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1
                                print(f"    处理双向转换方向2: {class2} → {class1}")
                                self._process_conversion(class2, class1, semantic_map1, semantic_map2,
                                                         individual_maps, final_adaptive_change_map,
                                                         legend_info, get_random_color, color_idx)
                                color_idx += 1
                                continue
                            else:
                                print(f"    跳过无效的双向转换格式: {transition}")
                                continue
                        else:
                            print(f"    跳过不支持的转换格式: {transition}")
                            continue
                    elif isinstance(transition, dict):
                        from_class = transition.get('from', '')
                        to_class = transition.get('to', '')
                    else:
                        print(f"    跳过不支持的转换格式: {type(transition)}")
                        continue

                    print(f"    处理转换: {from_class} → {to_class}")

                    from_idx = self.CLASS_INDEX.get(from_class, 0)
                    to_idx = self.CLASS_INDEX.get(to_class, 0)
                    if from_class == "background":
                        from_idx = 0
                    if to_class == "background":
                        to_idx = 0

                    if from_idx >= 0 and to_idx >= 0:
                        from_class_name = from_class.replace(' ', '_')
                        to_class_name = to_class.replace(' ', '_')
                        from_change = individual_maps.get(from_class_name,
                                                          np.zeros((height, width), dtype=np.uint8))
                        to_change = individual_maps.get(to_class_name, np.zeros((height, width), dtype=np.uint8))
                        t1_is_from = (semantic_map1 == from_idx)
                        t2_is_to = (semantic_map2 == to_idx)
                        combined_change_region = ((from_change > 0) | (to_change > 0))
                        transition_region = t1_is_from & t2_is_to & combined_change_region

                        pixel_count = np.sum(transition_region)
                        print(f"    发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

                        if np.any(transition_region):
                            color = get_random_color(color_idx)
                            color_idx += 1
                            final_adaptive_change_map[transition_region] = color
                            legend_info.append({'color': color, 'label': f'{from_class} → {to_class}'})
                    else:
                        print(f"    ⚠ 类别索引无效: {from_class}={from_idx}, {to_class}={to_idx}")

            # ==================== 情况2：单类别出现或消失 ====================
            elif change_direction in ['appear', 'disappear', 'increase', 'decrease', 'unidirectional']:
                print(f"  → 检测到单向变化指令: {change_direction}")

                for cls in involved_classes:
                    cls_idx = self.CLASS_INDEX.get(cls, -1)
                    if cls_idx <= 0 and cls != "background":
                        print(f"    ⚠ 类别 '{cls}' 未找到对应索引，跳过")
                        continue

                    cls_name = cls.replace(' ', '_')
                    cls_change = individual_maps.get(cls_name, np.zeros((height, width), dtype=np.uint8))

                    if change_direction in ['disappear', 'decrease']:
                        t1_has = (semantic_map1 == cls_idx)
                        t2_is_background = (semantic_map2 == 0)
                        change_region = t1_has & t2_is_background & (cls_change > 0)
                        label = f'{cls} disappeared'
                        print(f"    检测 {cls} 消失区域 (T1={cls_idx}, T2=0)")
                    else:
                        t1_is_background = (semantic_map1 == 0)
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
                        legend_info.append({'color': color, 'label': label})

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

                    t1_has = (semantic_map1 == cls_idx)
                    t2_is_background = (semantic_map2 == 0)
                    disappear_region = t1_has & t2_is_background & (cls_change > 0)

                    t1_is_background = (semantic_map1 == 0)
                    t2_has = (semantic_map2 == cls_idx)
                    appear_region = t1_is_background & t2_has & (cls_change > 0)

                    if np.any(disappear_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[disappear_region] = color
                        legend_info.append({'color': color, 'label': f'{cls} disappeared'})
                        print(f"    {cls} 消失: {np.sum(disappear_region)} 像素")

                    if np.any(appear_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[appear_region] = color
                        legend_info.append({'color': color, 'label': f'{cls} appeared'})
                        print(f"    {cls} 出现: {np.sum(appear_region)} 像素")

            # ==================== 情况4：未知变化类型 ====================
            else:
                print(f"  → 使用默认变化检测")
                for cls in involved_classes:
                    cls_idx = self.CLASS_INDEX.get(cls, -1)
                    if cls_idx <= 0 and cls != "background":
                        continue

                    cls_name = cls.replace(' ', '_')
                    cls_change = individual_maps.get(cls_name, np.zeros((height, width), dtype=np.uint8))
                    change_region = (cls_change > 0)

                    if np.any(change_region):
                        color = get_random_color(color_idx)
                        color_idx += 1
                        final_adaptive_change_map[change_region] = color
                        legend_info.append({'color': color, 'label': f'{cls} change'})
                        print(f"    {cls} 变化: {np.sum(change_region)} 像素")

        # ==================== 保存结果 ====================
        print("\n" + "-" * 40)
        print("保存自适应变化检测结果...")

        adaptive_change_path = os.path.join(output_dir, "adaptive_semantic_change.png")
        self._save_change_map_with_legend(final_adaptive_change_map, legend_info, adaptive_change_path)
        print(f"  ✓ 保存自适应变化图（带图例）: {adaptive_change_path}")

        adaptive_change_raw_path = os.path.join(output_dir, "adaptive_semantic_change_raw.png")
        Image.fromarray(final_adaptive_change_map).save(adaptive_change_raw_path)
        print(f"  ✓ 保存自适应变化图（无图例）: {adaptive_change_raw_path}")

        # ==================== 原有可视化逻辑 ====================
        semantic_change_t1 = semantic_map1 * instance_change_map
        semantic_change_t2 = semantic_map2 * instance_change_map

        colored_change_t1 = np.zeros((height, width, 3), dtype=np.uint8)
        colored_change_t2 = np.zeros((height, width, 3), dtype=np.uint8)

        for class_name, class_idx in self.CLASS_INDEX.items():
            color = self.CLASS_COLORS.get(class_name, [128, 128, 128])
            mask_t1 = (semantic_change_t1 == class_idx)
            colored_change_t1[mask_t1] = color
            mask_t2 = (semantic_change_t2 == class_idx)
            colored_change_t2[mask_t2] = color

        change_t1_path = os.path.join(output_dir, "semantic_change_t1.png")
        plt.imsave(change_t1_path, colored_change_t1)
        print(f"  ✓ 保存时相1语义变化检测图: {change_t1_path}")

        change_t2_path = os.path.join(output_dir, "semantic_change_t2.png")
        plt.imsave(change_t2_path, colored_change_t2)
        print(f"  ✓ 保存时相2语义变化检测图: {change_t2_path}")

        binary_change_path = os.path.join(output_dir, "binary_change_map.png")
        binary_uint8 = (instance_change_map * 255).astype(np.uint8)
        Image.fromarray(binary_uint8).save(binary_change_path)
        print(f"  ✓ 保存二值变化图: {binary_change_path}")

        self._create_visualization_figure(
            self.image1_np, self.image2_np,
            colored_change_t1, colored_change_t2,
            instance_change_map,
            output_dir
        )

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

    def _process_conversion(self, from_class, to_class, semantic_map1, semantic_map2,
                            individual_maps, final_adaptive_change_map,
                            legend_info, get_random_color_func, color_idx):
        """辅助方法：处理单个方向的变化转换"""
        from_idx = self.CLASS_INDEX.get(from_class, 0)
        to_idx = self.CLASS_INDEX.get(to_class, 0)
        if from_class == "background":
            from_idx = 0
        if to_class == "background":
            to_idx = 0

        if from_idx >= 0 and to_idx >= 0:
            from_class_name = from_class.replace(' ', '_')
            to_class_name = to_class.replace(' ', '_')
            from_change = individual_maps.get(from_class_name,
                                              np.zeros((semantic_map1.shape[0], semantic_map1.shape[1]),
                                                       dtype=np.uint8))
            to_change = individual_maps.get(to_class_name,
                                            np.zeros((semantic_map1.shape[0], semantic_map1.shape[1]),
                                                     dtype=np.uint8))
            t1_is_from = (semantic_map1 == from_idx)
            t2_is_to = (semantic_map2 == to_idx)
            combined_change_region = ((from_change > 0) | (to_change > 0))
            transition_region = t1_is_from & t2_is_to & combined_change_region

            pixel_count = np.sum(transition_region)
            print(f"      发现 {pixel_count} 个像素满足 {from_class} → {to_class}")

            if np.any(transition_region):
                color = get_random_color_func(color_idx)
                final_adaptive_change_map[transition_region] = color
                legend_info.append({'color': color, 'label': f'{from_class} → {to_class}'})

    def _save_change_map_with_legend(self, change_map, legend_info, save_path):
        legend_height = max(60, len(legend_info) * 30 + 40)
        height, width = change_map.shape[:2]

        canvas_height = height + legend_height
        canvas = np.ones((canvas_height, width, 3), dtype=np.uint8) * 255
        canvas[:height, :, :] = change_map

        fig, ax = plt.subplots(figsize=(width / 100, canvas_height / 100), dpi=100)
        ax.imshow(canvas)
        ax.axis('off')

        if legend_info:
            legend_start_y = height + 20
            legend_x = 20
            box_size = 20
            spacing = 30
            for i, item in enumerate(legend_info):
                color = item['color']
                label = item['label']
                color_normalized = [c / 255.0 for c in color]
                rect_y = legend_start_y + i * spacing
                rect = plt.Rectangle((legend_x, rect_y), box_size, box_size,
                                     facecolor=color_normalized, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(legend_x + box_size + 10, rect_y + box_size / 2, label,
                        fontsize=10, verticalalignment='center', fontfamily='sans-serif')

        ax.text(width / 2, height + 10, 'Change Detection Legend',
                fontsize=12, fontweight='bold', ha='center', va='top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _create_adaptive_visualization_figure(self, img1, img2, adaptive_change_map, legend_info, output_dir):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                              hspace=0.15, wspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.set_title("Phase 1 (Before)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.set_title("Phase 2 (After)", fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(adaptive_change_map)
        ax3.set_title("Adaptive Semantic Change", fontsize=14, fontweight='bold')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 0])
        overlay1 = img1.copy()
        change_mask = np.any(adaptive_change_map > 0, axis=2)
        overlay1_float = overlay1.astype(np.float32)
        overlay1_float[change_mask] = overlay1_float[change_mask] * 0.5 + adaptive_change_map[change_mask].astype(
            np.float32) * 0.5
        ax4.imshow(overlay1_float.astype(np.uint8))
        ax4.set_title("Phase 1 with Changes", fontsize=14, fontweight='bold')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        overlay2 = img2.copy()
        overlay2_float = overlay2.astype(np.float32)
        overlay2_float[change_mask] = overlay2_float[change_mask] * 0.5 + adaptive_change_map[change_mask].astype(
            np.float32) * 0.5
        ax5.imshow(overlay2_float.astype(np.uint8))
        ax5.set_title("Phase 2 with Changes", fontsize=14, fontweight='bold')
        ax5.axis('off')

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
                color_normalized = [c / 255.0 for c in color]
                y_pos = start_y - i * step_y
                rect = plt.Rectangle((0.1, y_pos - 0.05), 0.15, 0.08,
                                     facecolor=color_normalized, edgecolor='black', linewidth=1,
                                     transform=ax6.transAxes)
                ax6.add_patch(rect)
                ax6.text(0.3, y_pos, label, fontsize=11, va='center',
                         transform=ax6.transAxes, fontfamily='sans-serif')
        else:
            ax6.text(0.5, 0.5, 'No changes detected', fontsize=12, ha='center', va='center',
                     transform=ax6.transAxes, style='italic', color='gray')

        fig.suptitle('Adaptive Semantic Change Detection Results', fontsize=16, fontweight='bold', y=0.98)

        vis_path = os.path.join(output_dir, "adaptive_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  ✓ 保存自适应变化可视化图: {vis_path}")

    def _create_visualization_figure(self, img1, img2,
                                     semantic_change1, semantic_change2,
                                     binary_change, output_dir):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(img1)
        axes[0, 0].set_title("Phase 1 Original")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img2)
        axes[0, 1].set_title("Phase 2 Original")
        axes[0, 1].axis('off')

        if len(semantic_change1.shape) == 3 and semantic_change1.shape[2] == 3:
            axes[0, 2].imshow(semantic_change1)
        else:
            axes[0, 2].imshow(self._gray_to_color(semantic_change1))
        axes[0, 2].set_title("Phase 1 Semantic Change")
        axes[0, 2].axis('off')

        if len(semantic_change2.shape) == 3 and semantic_change2.shape[2] == 3:
            axes[1, 0].imshow(semantic_change2)
        else:
            axes[1, 0].imshow(self._gray_to_color(semantic_change2))
        axes[1, 0].set_title("Phase 2 Semantic Change")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(binary_change, cmap='gray')
        axes[1, 1].set_title("Binary Change Map")
        axes[1, 1].axis('off')

        overlay = img1.copy()
        change_mask = binary_change.astype(bool)
        if len(overlay.shape) == 3:
            overlay[change_mask] = [255, 0, 0]
        else:
            overlay_rgb = np.stack([overlay, overlay, overlay], axis=2)
            overlay_rgb[change_mask] = [255, 0, 0]
            overlay = overlay_rgb
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Change Overlay (Red)")
        axes[1, 2].axis('off')

        plt.tight_layout()
        vis_path = os.path.join(output_dir, "visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存可视化图: {vis_path}")

    def _gray_to_color(self, gray_image):
        if len(gray_image.shape) == 2:
            colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
            unique_classes = np.unique(gray_image)
            for class_value in unique_classes:
                if class_value > 0:
                    class_name = None
                    for name, idx in self.CLASS_INDEX.items():
                        if idx == class_value:
                            class_name = name
                            break
                    if class_name and class_name in self.CLASS_COLORS:
                        color = self.CLASS_COLORS[class_name]
                    else:
                        np.random.seed(class_value + 42)
                        color = [int(x) for x in np.random.randint(0, 256, 3)]
                    colored_image[gray_image == class_value] = color
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


# ============================================================================
# 精度评估（SECOND 数据集：water_label 二值变化真值）
# ============================================================================
def evaluate_results(pred_dir, gt_dir, label1_dir, label2_dir, water_cls=1):
    """
    对最终变化图与真值做像素级精度评估。
    SECOND 的类别索引: 0=背景 1=水 2=裸地 3=低植被 4=树 5=建筑 6=运动场
    同时给出"完整GT（含出现+消失）"和"仅消失方向GT"两组指标。
    """
    gt_files = {os.path.basename(f): f for f in glob.glob(os.path.join(gt_dir, '*.png'))}
    pred_files = {os.path.basename(f): f for f in glob.glob(os.path.join(pred_dir, '*.png'))}
    common = sorted(set(gt_files) & set(pred_files))
    if len(common) == 0:
        print("没有可评估的预测/真值对")
        return

    TP = FP = FN = TN = 0
    TP_dis = FP_dis = FN_dis = 0
    gt_has = pred_has = tp_img = 0

    for name in common:
        gt = np.array(Image.open(gt_files[name])) > 0
        pr = np.array(Image.open(pred_files[name]))
        pr = (pr.max(axis=2) > 0) if pr.ndim == 3 else (pr > 0)

        l1 = np.array(Image.open(os.path.join(label1_dir, name)))
        l2 = np.array(Image.open(os.path.join(label2_dir, name)))
        gt_dis = gt & (l1 == water_cls) & (l2 != water_cls)

        TP += int((gt & pr).sum())
        FP += int((~gt & pr).sum())
        FN += int((gt & ~pr).sum())
        TN += int((~gt & ~pr).sum())

        TP_dis += int((gt_dis & pr).sum())
        FP_dis += int((~gt_dis & pr).sum())
        FN_dis += int((gt_dis & ~pr).sum())

        if gt.any():
            gt_has += 1
        if pr.any():
            pred_has += 1
        if (gt & pr).any():
            tp_img += 1

    def metrics(tp, fp, fn, tn=None):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        oa = (tp + tn) / (tp + fp + fn + tn) if tn is not None else None
        return p, r, f1, iou, oa

    print("\n" + "=" * 60)
    print(f"精度评估: {pred_dir}")
    print(f"评估图像数: {len(common)}")
    print("=" * 60)

    p, r, f1, iou, oa = metrics(TP, FP, FN, TN)
    print("【完整 water_label GT（含出现+消失）】")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  Precision = {p*100:.2f}%   Recall = {r*100:.2f}%   F1 = {f1*100:.2f}%")
    print(f"  IoU = {iou*100:.2f}%   OA = {oa*100:.2f}%")

    p, r, f1, iou, _ = metrics(TP_dis, FP_dis, FN_dis)
    print("【仅『消失』方向 GT (water->非water)】")
    print(f"  TP={TP_dis}, FP={FP_dis}, FN={FN_dis}")
    print(f"  Precision = {p*100:.2f}%   Recall = {r*100:.2f}%   F1 = {f1*100:.2f}%   IoU = {iou*100:.2f}%")

    print(f"GT有变化的图: {gt_has}, 预测有变化的图: {pred_has}, 正确检出的图(TP>0): {tp_img}")


# ============================================================================
# 主程序：SECOND 数据集 "disappeared water" 批量处理 + 精度评估
# ============================================================================
if __name__ == "__main__":
    from tqdm import tqdm
    from contextlib import redirect_stdout, redirect_stderr

    IMAGE1_FOLDER = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/im1"
    IMAGE2_FOLDER = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/im2"
    TEST_ROOT = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test"
    OUTPUT_DIR = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/Instruct_added_water/Proposed_0926"
    INSTRUCTION_TEXT = ['added water']

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
                        output_dir="/tmp/_batch_0927",
                        use_adaptive_threshold=True,
                        model=shared_model
                    )
                    detector.run_sequential_pipeline(
                        refine_factor=2, change_threshold=0.6)

            src = "/tmp/_batch_0927/adaptive_semantic_change_raw.png"
            if os.path.exists(src):
                shutil.copy2(src, out_path)
            shutil.rmtree("/tmp/_batch_0927", ignore_errors=True)
            success += 1
        except Exception:
            failed += 1
            shutil.rmtree("/tmp/_batch_0927", ignore_errors=True)

        try:
            del detector
        except NameError:
            pass
        plt.close('all')
        torch.cuda.empty_cache()
        gc.collect()

        pbar.set_postfix({'成功': success, '失败': failed, '跳过': skipped})
        pbar.update(1)

    pbar.close()
    print(f"\n完成: 成功={success}, 失败={failed}, 跳过={skipped}")

    # ---- 精度评估 ----
    evaluate_results(
        pred_dir=OUTPUT_DIR,
        gt_dir=os.path.join(TEST_ROOT, "water_label"),
        label1_dir=os.path.join(TEST_ROOT, "label1"),
        label2_dir=os.path.join(TEST_ROOT, "label2"),
        water_cls=1,
    )
