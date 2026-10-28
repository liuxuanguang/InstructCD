# file_name: TextTransV3.py

import json
import re
import time
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import requests
from collections import defaultdict
from datetime import datetime


class LLMClient:
    """
    大语言模型客户端，用于在线指令转换
    """

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def convert_instruction(self, user_instruction: str,
                            model_name: str = "Pro/MiniMaxAI/MiniMax-M2.5",
                            temperature: float = 0.7,
                            max_tokens: int = 10000) -> Optional[Dict]:
        prompt = self._build_conversion_prompt(user_instruction)
        endpoint = f"{self.base_url}/chat/completions"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(endpoint, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return self._parse_llm_response(content, user_instruction)
        except requests.exceptions.RequestException as e:
            print(f"大语言模型API调用失败: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
        return None

    def _build_conversion_prompt(self, user_instruction: str) -> str:
        return f"""你是一个遥感图像处理专家，需要将自然语言变化检测指令转换为结构化格式。

    转换规则：
    1. 输入格式有多种：
       - 新增类：如"new building"或"building appears"
       - 消失类：如"building disappeared"或"building disappears"或"building disappear"
       - 转换类：如"from bare land to building"
       - 复合类：如"new building and disappear low vegetation"（同时包含多个变化）
       - 多个连续转换可以用and、分号;分隔，每个转换都要独立处理
    2. 输出格式为结构化数据：
       - 用户指令：原始输入
       - 变化类型：TYPE_CONVERSION（对于转换类）、APPEARANCE（对于新增类）、DISAPPEARANCE（对于消失类）、COMPOSITE（对于复合类）
       - 时相1指令：涉及的类别用分号分隔，如"building;low vegetation"
       - 时相2指令：同phase1_instruction
       - 置信度：0.9
       - 描述：可选的描述文本
       - 标签：[相关标签列表]
       - 来源："core"
       - 变化语义：UNIDIRECTIONAL_CONVERSION（对于转换类）、APPEARANCE（对于新增类）、DISAPPEARANCE（对于消失类）、COMPOSITE（对于复合类）
       - 变化方向：FORWARD（对于转换类和新增类）、BACKWARD（对于消失类）、BIDIRECTIONAL（对于同时包含新增和消失的复合类）
       - 转换关系：[源→目标]，使用background表示背景，[每个转换独立生成一个条目，多个转换保留多个条目]

    重要转换关系规则：
    - 新增类：使用"background→目标地物"，如"background→building"
    - 消失类：使用"源地物→background"，如"building→background"
    - 转换类：使用"源地物→目标地物"，如"bare land→building"
    - 复合类：使用多个转换关系，用逗号分隔
    - 地物增加：从背景变为目标地物，转换方向为FORWARD
    - 地物消失：从目标地物变为背景，转换方向为BACKWARD
    - 复合类中同时包含新增和消失：转换方向为BIDIRECTIONAL
    - 多个连续转换类：每个from X to Y生成一个独立转换关系，不要合并，比如两个转换就生成两个条目
    重要变化方向规则：
    - 只包含新增：FORWARD
    - 只包含消失：BACKWARD
    - 同时包含新增和消失：BIDIRECTIONAL

    示例1（新增类）：
    输入："new building"
    输出：
    InstructionExample(
        "new building", "APPEARANCE",
        "building", "building", 0.95,
        "新建筑物出现", ["appearance", "new"], source="core",
        change_semantics="APPEARANCE",
        change_direction="FORWARD",
        transitions=["background→building"]
    )

    示例2（消失类）：
    输入："building disappeared"
    输出：
    InstructionExample(
        "building disappeared", "DISAPPEARANCE",
        "building", "building", 0.95,
        "建筑物消失", ["disappearance", "demolished"], source="core",
        change_semantics="DISAPPEARANCE",
        change_direction="BACKWARD",
        transitions=["building→background"]
    )

    示例3（转换类）：
    输入："from vegetation to building"
    输出：
    InstructionExample(
        "from vegetation to building", "TYPE_CONVERSION",
        "vegetation;building", "vegetation;building", 0.93,
        "植被到建筑物的转换", ["vegetation_to_building", "deforestation"], source="core",
        change_semantics="UNIDIRECTIONAL_CONVERSION",
        change_direction="FORWARD",
        transitions=["vegetation→building"]
    )

    示例4（复合类-新增和消失）：
    输入："new building and disappear low vegetation"
    输出：
    InstructionExample(
        "new building and disappear low vegetation", "COMPOSITE",
        "building;low vegetation", "building;low vegetation", 0.9,
        "建筑物新增和低矮植被消失的复合变化", ["composite", "appearance", "disappearance"], source="core",
        change_semantics="COMPOSITE",
        change_direction="BIDIRECTIONAL",
        transitions=["background→building", "low vegetation→background"]
    )

    示例5（复合类-两个新增）：
    输入："new building and new vegetation"
    输出：
    InstructionExample(
        "new building and new vegetation", "COMPOSITE",
        "building;vegetation", "building;vegetation", 0.85,
        "建筑物和植被同时新增", ["composite", "appearance"], source="core",
        change_semantics="COMPOSITE",
        change_direction="FORWARD",
        transitions=["background→building", "background→vegetation"]
    )

    示例6（复合类-两个消失）：
    输入："building disappear and vegetation disappear"
    输出：
    InstructionExample(
        "building disappear and vegetation disappear", "COMPOSITE",
        "building;vegetation", "building;vegetation", 0.85,
        "建筑物和植被同时消失", ["composite", "disappearance"], source="core",
        change_semantics="COMPOSITE",
        change_direction="BACKWARD",
        transitions=["building→background", "vegetation→background"]
    )

   示例7（多个连续转换类，and分隔）：
   输入："from water to bare land and from bare land to building"
   输出：
   InstructionExample(
     "from water to bare land and from bare land to building", "COMPOSITE",
     "water;bare land;building", "water;bare land;building", 0.9,
     "水转裸地、裸地转建筑物的连续转换", ["composite", "unidirectional_conversion"], source="core",
     change_semantics="COMPOSITE",
     change_direction="FORWARD",
     transitions=["water→bare land", "bare land→building"]
   )

   示例8（多个连续转换类，分号分隔）：
    输入："from tree to bare land; from bare land to building"
    输出：
    InstructionExample(
        "from tree to bare land; from bare land to building", "COMPOSITE",
        "tree;bare land;building", "tree;bare land;building", 0.9,
        "树木转裸地、裸地转建筑物的连续转换", ["composite", "unidirectional_conversion"], source="core",
        change_semantics="COMPOSITE",
        change_direction="FORWARD",
        transitions=["tree→bare land", "bare land→building"]
    )

请严格按照以上规则，将"{user_instruction}"转换为相同的结构格式。特别注意：
1. 如果是多个from X to Y的连续转换，每个转换必须生成一个独立的转换关系条目，绝对不能合并成一个
2. 仔细分析指令中每个地物的变化类型（新增还是消失）
3. 对于复合指令（包含"and"的指令），必须分别处理每个部分
4. 正确设置每个转换关系（新增用background→地物，消失用地物→background）
5. 正确设置变化方向：只有新增=FORWARD，只有消失=BACKWARD，新增和消失同时存在=BIDIRECTIONAL
6. 时相1和时相2指令要包含所有涉及的类别，用分号分隔
7. 不要使用双向箭头↔，除非确实是双向转换
8. 不要生成类似"building↔building"的错误转换关系"""

    def _parse_llm_response(self, content: str, original_instruction: str) -> Optional[Dict]:
        try:
            user_instruction_match = re.search(r'InstructionExample\(\s*"([^"]+)"', content)
            user_instruction = user_instruction_match.group(1) if user_instruction_match else original_instruction

            change_type_match = re.search(r'change_type.*?"([^"]+)"', content)
            if change_type_match:
                change_type = change_type_match.group(1)
            else:
                pattern = r'InstructionExample\([^,]+\s*,\s*"([^"]+)"'
                change_type_match = re.search(pattern, content)
                change_type = change_type_match.group(1) if change_type_match else (
                    "COMPOSITE" if "and" in original_instruction.lower() or ";" in original_instruction.lower() else "TYPE_CONVERSION")

            phase1_match = re.search(r'phase1_instruction.*?"([^"]+)"', content)
            phase2_match = re.search(r'phase2_instruction.*?"([^"]+)"', content)
            if phase1_match and phase2_match:
                phase1_instruction = phase1_match.group(1)
                phase2_instruction = phase2_match.group(1)
            else:
                pattern = r'InstructionExample\([^,]+\s*,\s*[^,]+\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"'
                matches = re.search(pattern, content)
                if matches:
                    phase1_instruction = matches.group(1)
                    phase2_instruction = matches.group(2)
                else:
                    phase1_instruction = phase2_instruction = "change_area"

            confidence_match = re.search(r'confidence.*?([0-9]+\.[0-9]+)', content)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.9

            description_match = re.search(r'description.*?"([^"]*)"', content)
            description = description_match.group(1) if description_match else ""

            tags_match = re.search(r'tags.*?\[(.*?)\]', content)
            if tags_match:
                tags_str = tags_match.group(1)
                tags = [t.strip().strip('"') for t in tags_str.split(',') if t.strip()]
            else:
                tags = []

            transitions_match = re.search(r'transitions.*?\[(.*?)\]', content)
            transitions = []
            if transitions_match:
                transitions_str = transitions_match.group(1)
                for t in transitions_str.split(','):
                    t_clean = t.strip().strip('"').strip("'").strip()
                    if t_clean and '→' in t_clean:
                        transitions.append(t_clean)

                corrected_transitions = []
                for transition in transitions:
                    from_part, to_part = transition.split('→')
                    from_part = from_part.strip()
                    to_part = to_part.strip()
                    if from_part.startswith("no ") or from_part == "no":
                        from_part = "background"
                    if to_part.startswith("no ") or to_part == "no":
                        to_part = "background"
                    corrected_transitions.append(f"{from_part}→{to_part}")
                transitions = corrected_transitions

            # ===== [修复] 多from-to自动提取修正逻辑：使用更精确的正则 =====
            instruction_lower = original_instruction.lower()
            # 使用带边界的正则，避免跨from-to捕获
            from_to_pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=\s+and\s+|\s*[;；]\s*|$)'
            all_from_to = re.findall(from_to_pattern, instruction_lower, re.DOTALL)
            if len(all_from_to) >= 2:
                corrected_transitions = []
                all_classes = set()
                for (from_cls, to_cls) in all_from_to:
                    from_cls = self._normalize_classes(from_cls.strip())
                    to_cls = self._normalize_classes(to_cls.strip())
                    corrected_transitions.append(f"{from_cls}→{to_cls}")
                    all_classes.add(from_cls)
                    all_classes.add(to_cls)
                transitions = corrected_transitions
                phase1_instruction = ";".join(sorted(all_classes))
                phase2_instruction = phase1_instruction
                change_type = "COMPOSITE"

            elif " and " in instruction_lower or ";" in instruction_lower or "；" in instruction_lower:
                parts = re.split(r'\s+and\s+|\s*[;；]\s*', instruction_lower)
                corrected_transitions = []
                common_classes = ["building", "vegetation", "tree", "water", "sports ground", "bare land",
                                  "low vegetation"]
                for part in parts:
                    part = part.strip()
                    # [修复] 先检查disappear，避免被appear子串干扰
                    is_disappearance = "disappear" in part or "loss" in part
                    is_appearance = ("new" in part or ("appear" in part and "disappear" not in part))
                    found_class = None
                    sorted_classes = sorted(common_classes, key=len, reverse=True)
                    for cls in sorted_classes:
                        if cls in part:
                            found_class = cls
                            break
                    if found_class:
                        if is_disappearance:
                            corrected_transitions.append(f"{found_class}→background")
                        elif is_appearance:
                            corrected_transitions.append(f"background→{found_class}")
                if corrected_transitions:
                    transitions = list(dict.fromkeys(corrected_transitions))

            change_semantics_match = re.search(r'change_semantics.*?"([^"]+)"', content)
            if change_semantics_match:
                change_semantics = change_semantics_match.group(1)
            else:
                instruction_lower = original_instruction.lower()
                has_appearance = ("new" in instruction_lower or
                                 ("appear" in instruction_lower and "disappear" not in instruction_lower))
                has_disappearance = "disappear" in instruction_lower or "loss" in instruction_lower
                has_multi_conversion = len(re.findall(r'from\s+.*?\s+to\b', instruction_lower)) >= 2
                if has_multi_conversion or (has_appearance and has_disappearance):
                    change_semantics = "COMPOSITE"
                elif has_appearance:
                    change_semantics = "APPEARANCE"
                elif has_disappearance:
                    change_semantics = "DISAPPEARANCE"
                elif "and" in instruction_lower:
                    change_semantics = "COMPOSITE"
                else:
                    change_semantics = "UNIDIRECTIONAL_CONVERSION"

            change_direction_match = re.search(r'change_direction.*?"([^"]+)"', content)
            if change_direction_match:
                change_direction = change_direction_match.group(1)
            else:
                instruction_lower = original_instruction.lower()
                has_appearance = ("new" in instruction_lower or
                                 ("appear" in instruction_lower and "disappear" not in instruction_lower))
                has_disappearance = "disappear" in instruction_lower or "loss" in instruction_lower
                has_multi_conversion = len(re.findall(r'from\s+.*?\s+to\b', instruction_lower)) >= 2
                if change_semantics in ("COMPOSITE", "composite"):
                    if has_multi_conversion:
                        change_direction = "FORWARD"
                    elif has_appearance and has_disappearance:
                        change_direction = "BIDIRECTIONAL"
                    elif has_appearance:
                        change_direction = "FORWARD"
                    elif has_disappearance:
                        change_direction = "BACKWARD"
                    else:
                        change_direction = "BIDIRECTIONAL"
                elif change_semantics in ("APPEARANCE", "appearance"):
                    change_direction = "FORWARD"
                elif change_semantics in ("DISAPPEARANCE", "disappearance"):
                    change_direction = "BACKWARD"
                else:
                    change_direction = "FORWARD"

            phase1_instruction = self._normalize_classes(phase1_instruction)
            phase2_instruction = self._normalize_classes(phase2_instruction)

            return {
                "user_instruction": user_instruction,
                "change_type": change_type,
                "phase1_instruction": phase1_instruction,
                "phase2_instruction": phase2_instruction,
                "confidence": confidence,
                "description": description,
                "tags": tags,
                "source": "llm_online",
                "change_semantics": change_semantics,
                "change_direction": change_direction,
                "transitions": transitions
            }
        except Exception as e:
            print(f"解析大语言模型响应失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _normalize_classes(self, class_str: str) -> str:
        if not class_str:
            return ""
        class_mapping = {
            "bare ground": "bare land",
            "bareground": "bare land",
            "bare soil": "bare land",
            "ground": "bare land",
            "soil": "bare land",
            "veg": "vegetation",
            "low_vegetation": "low vegetation",
            "sportsground": "sports ground",
            "playground": "sports ground",
            "waterbody": "water",
            "forest": "tree",
            "farmland": "bare land",
            "farm": "bare land",
            "grass": "vegetation"
        }
        classes = [c.strip() for c in class_str.split(';')]
        normalized_classes = []
        for cls in classes:
            cls_lower = cls.lower()
            if cls_lower in class_mapping:
                normalized_classes.append(class_mapping[cls_lower])
            else:
                matched = False
                for key, value in class_mapping.items():
                    if key in cls_lower:
                        normalized_classes.append(value)
                        matched = True
                        break
                if not matched:
                    normalized_classes.append(cls_lower)
        return ";".join(normalized_classes)


# ==================== 增强的变化语义定义 ====================
class ChangeSemantics(Enum):
    """变化语义类型"""
    APPEARANCE = "appearance"
    DISAPPEARANCE = "disappearance"
    UNIDIRECTIONAL_CONVERSION = "unidirectional_conversion"
    BIDIRECTIONAL_CONVERSION = "bidirectional_conversion"
    AREA_CHANGE = "area_change"
    COMPOSITE = "composite"
    ALL_CHANGES = "all_changes"


class ChangeDirection(Enum):
    """变化方向"""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


# ==================== 增强的数据类定义 ====================
@dataclass
class ChangeTransition:
    """变化转换定义"""
    from_class: str
    to_class: str
    direction: ChangeDirection

    def __str__(self):
        if self.direction == ChangeDirection.BIDIRECTIONAL:
            return f"{self.from_class}↔{self.to_class}"
        else:
            return f"{self.from_class}→{self.to_class}"


@dataclass
class InstructionSemantics:
    """指令语义解析结果"""
    change_semantics: str
    target_objects: List[str]
    output_requirements: Dict[str, Any]
    spatial_constraints: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    original_instruction: Optional[str] = None
    change_direction: Optional[str] = None
    involved_classes: Optional[List[str]] = None
    transitions: Optional[List[Any]] = None
    phase1_instruction: Optional[str] = None
    phase2_instruction: Optional[str] = None
    visualization_hint: Optional[Dict[str, Any]] = None

    def get_change_type(self) -> str:
        return self.change_semantics

    def is_appearance(self) -> bool:
        return self.change_semantics == ChangeSemantics.APPEARANCE.value

    def is_disappearance(self) -> bool:
        return self.change_semantics == ChangeSemantics.DISAPPEARANCE.value

    def is_conversion(self) -> bool:
        return self.change_semantics in [
            ChangeSemantics.UNIDIRECTIONAL_CONVERSION.value,
            ChangeSemantics.BIDIRECTIONAL_CONVERSION.value
        ]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InstructionExample:
    """指令示例数据类（增强版）"""
    user_instruction: str
    change_type: str
    phase1_instruction: str
    phase2_instruction: str
    confidence: float
    description: str = ""
    tags: List[str] = field(default_factory=list)
    frequency: int = 1
    source: str = "core"
    created_at: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    last_used: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    keywords: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    change_semantics: str = ChangeSemantics.ALL_CHANGES.value
    change_direction: str = ChangeDirection.BIDIRECTIONAL.value
    transitions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tags:
            self.tags = []
        if not self.keywords:
            self.keywords = self._extract_keywords()
        if not self.objects:
            self.objects = self._extract_objects()

    def _extract_keywords(self) -> List[str]:
        if not self.user_instruction:
            return []
        words = re.findall(r'\b\w+\b', self.user_instruction.lower())
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'of', 'for'}
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _extract_objects(self) -> List[str]:
        objects = []
        common_objects = [
            "building", "road", "vegetation", "forest", "farmland", "water",
            "lake", "river", "land", "bare land", "soil", "structure",
            "house", "street", "highway", "plants", "trees", "crop",
            "tree", "low vegetation", "urban", "city", "town", "settlement",
            "sports ground", "playground"
        ]
        instruction_lower = self.user_instruction.lower()
        for obj in common_objects:
            if obj in instruction_lower:
                objects.append(obj)
        return objects

    def to_dict(self) -> Dict:
        return asdict(self)

    def update_usage(self):
        self.frequency += 1
        self.last_used = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ChangeType(Enum):
    ALL_CHANGE = "all_change"
    SIMPLE_EXISTENCE = "simple_existence_change"
    TYPE_CONVERSION = "type_conversion_change"
    APPEARANCE = "appearance_change"
    DISAPPEARANCE = "disappearance_change"
    COMPOSITE = "composite_change"
    UNKNOWN = "unknown_change"


# ==================== 语义解析器 ====================
class SemanticParser:
    """语义解析器 - 将指令转换为详细的语义表示"""

    def __init__(self):
        self.appearance_keywords = [
            r'\bnew\b', r'\bappear(?:ed|ance)?\b', r'\bemerge(?:d|nce)?\b',
            r'\bform(?:ed)?\b', r'\bconstruct(?:ed)?\b', r'\bbuilt\b',
            r'\bcreate(?:d)?\b', r'\badd(?:ed)?\b', r'\bestablish(?:ed)?\b'
        ]
        self.disappearance_keywords = [
            r'\bdisappear(?:ed|ance)?\b', r'\bvanish(?:ed)?\b', r'\bremove(?:d)?\b',
            r'\bdemolish(?:ed)?\b', r'\bdestroy(?:ed)?\b', r'\bloss\b',
            r'\bdelete(?:d)?\b', r'\bclear(?:ed)?\b'
        ]
        self.conversion_keywords = [
            r'\bfrom\s+.+\s+to\b', r'\bconvert(?:ed)?\b', r'\btransform(?:ed)?\b',
            r'\bchange(?:d)?\s+(?:from|to)\b', r'\bturn(?:ed)?\s+into\b',
            r'\bbecome\b', r'\bbetween\b'
        ]
        self.area_change_keywords = {
            'increase': [r'\bincrease(?:d)?\b', r'\bexpand(?:ed)?\b', r'\bgrow(?:th)?\b',
                         r'\benlarge(?:d|ment)?\b'],
            'decrease': [r'\bdecrease(?:d)?\b', r'\breduce(?:d)?\b', r'\bdecline(?:d)?\b',
                         r'\bshrink(?:age)?\b']
        }
        self.bidirectional_keywords = [
            r'\bbetween\b', r'\band\b.*\band\b', r'\b(?:mutual|reciprocal)\b',
            r'\brelationship\b', r'\binteraction\b'
        ]

    def parse(self, user_instruction: str, base_result: Dict) -> InstructionSemantics:
        """
        [核心修复] 优先使用base_result中已有的transitions/semantics/direction，
        仅在缺失时才重新推断。
        """
        instruction_lower = user_instruction.lower()

        # ===== [修复1] 优先从base_result提取已有的可靠结果 =====
        base_data = base_result.get('data', {})
        base_transitions_raw = base_data.get('transitions', [])
        base_change_semantics = base_data.get('change_semantics', '')
        base_change_direction = base_data.get('change_direction', '')

        involved_classes = self._extract_involved_classes(instruction_lower, base_result)

        # 如果base_result已经有有效的transitions（来自精确匹配或LLM），直接使用
        if base_transitions_raw and base_change_semantics:
            change_semantics_str = base_change_semantics
            change_direction_str = base_change_direction if base_change_direction else "bidirectional"
            transitions_str = base_transitions_raw

            # 转enum用于可视化提示
            try:
                change_semantics_enum = ChangeSemantics(change_semantics_str)
            except ValueError:
                change_semantics_enum = ChangeSemantics.ALL_CHANGES

            # 从transitions字符串构建ChangeTransition对象用于可视化
            transition_objects = self._parse_transition_strings(transitions_str)
            visualization_hint = self._generate_visualization_hint(
                change_semantics_enum, transition_objects, involved_classes
            )
        else:
            # ===== 回退到自主推断 =====
            change_semantics = self._identify_change_semantics(instruction_lower, base_result)
            change_direction = self._identify_change_direction(instruction_lower, change_semantics)
            transitions = self._build_transitions(
                instruction_lower, base_result, change_semantics,
                change_direction, involved_classes
            )
            change_semantics_str = change_semantics.value
            change_direction_str = change_direction.value
            transitions_str = [str(t) for t in transitions] if transitions else []
            visualization_hint = self._generate_visualization_hint(
                change_semantics, transitions, involved_classes
            )

        return InstructionSemantics(
            change_semantics=change_semantics_str,
            target_objects=involved_classes,
            output_requirements={
                "visualization": True,
                "confidence": base_data.get('confidence', 0.5)
            },
            spatial_constraints=None,
            confidence=base_data.get('confidence', 0.5),
            original_instruction=user_instruction,
            change_direction=change_direction_str,
            involved_classes=involved_classes,
            transitions=transitions_str,
            phase1_instruction=base_data.get('phase1_instruction', ''),
            phase2_instruction=base_data.get('phase2_instruction', ''),
            visualization_hint=visualization_hint
        )

    def _parse_transition_strings(self, transitions_raw: List[str]) -> List[ChangeTransition]:
        """将字符串形式的transitions解析为ChangeTransition对象"""
        result = []
        for t in transitions_raw:
            if '↔' in t:
                parts = t.split('↔')
                if len(parts) == 2:
                    result.append(ChangeTransition(
                        from_class=parts[0].strip(), to_class=parts[1].strip(),
                        direction=ChangeDirection.BIDIRECTIONAL
                    ))
            elif '→' in t:
                parts = t.split('→')
                if len(parts) == 2:
                    from_cls = parts[0].strip()
                    to_cls = parts[1].strip()
                    if to_cls == "background":
                        direction = ChangeDirection.BACKWARD
                    else:
                        direction = ChangeDirection.FORWARD
                    result.append(ChangeTransition(
                        from_class=from_cls, to_class=to_cls, direction=direction
                    ))
        return result

    def _identify_change_semantics(self, instruction: str, base_result: Dict) -> ChangeSemantics:
        change_type = base_result.get('data', {}).get('change_type', '')
        instruction_lower = instruction.lower()

        # ===== [修复2] 最先检测多from-to模式，直接返回COMPOSITE =====
        multi_from_to_count = len(re.findall(r'\bfrom\s+.*?\s+to\b', instruction_lower))
        if multi_from_to_count >= 2:
            return ChangeSemantics.COMPOSITE

        # 通用变化检测模式
        change_patterns = [
            r'detecting\s+(\w+\s+)*change',
            r'(\w+\s+)*change\s+detection',
            r'find\s+changes\s+in',
            r'identify\s+(\w+\s+)*changes',
            r'detect\s+(\w+\s+)*changes'
        ]
        is_generic_change = any(re.search(pattern, instruction_lower) for pattern in change_patterns)
        if is_generic_change:
            common_classes = ["building", "vegetation", "tree", "water", "bare land", "low vegetation",
                              "sports ground"]
            mentioned_classes = [cls for cls in common_classes if cls in instruction_lower]
            if mentioned_classes:
                return ChangeSemantics.ALL_CHANGES

        # ===== [修复3] 先检测disappear再检测appear，避免子串干扰 =====
        has_disappearance = any(re.search(kw, instruction_lower) for kw in self.disappearance_keywords)
        has_appearance = any(re.search(kw, instruction_lower) for kw in self.appearance_keywords)
        # 如果同时匹配到appear和disappear，以disappear为准来判断是否有消失
        if has_appearance and has_disappearance:
            # 双重检查：appear是否独立于disappear
            # 移除所有disappear*后检查是否还有appear
            cleaned = re.sub(r'\bdisappear(?:ed|ance|s)?\b', '', instruction_lower)
            has_independent_appearance = any(re.search(kw, cleaned) for kw in self.appearance_keywords)
            if has_independent_appearance:
                return ChangeSemantics.COMPOSITE
            else:
                # 只有disappear没有独立的appear
                has_appearance = False

        if " and " in instruction_lower or ";" in instruction_lower or "；" in instruction_lower:
            if has_appearance or has_disappearance:
                return ChangeSemantics.COMPOSITE
            # 检查是否有from-to
            if re.search(r'\bfrom\s+.+\s+to\b', instruction_lower):
                return ChangeSemantics.COMPOSITE

        if has_appearance and not has_disappearance:
            return ChangeSemantics.APPEARANCE
        if has_disappearance and not has_appearance:
            return ChangeSemantics.DISAPPEARANCE
        if any(re.search(kw, instruction_lower) for kw in self.conversion_keywords):
            if any(re.search(kw, instruction_lower) for kw in self.bidirectional_keywords):
                return ChangeSemantics.BIDIRECTIONAL_CONVERSION
            else:
                return ChangeSemantics.UNIDIRECTIONAL_CONVERSION

        for direction, keywords in self.area_change_keywords.items():
            if any(re.search(kw, instruction_lower) for kw in keywords):
                return ChangeSemantics.AREA_CHANGE

        if 'all' in instruction_lower and 'change' in instruction_lower:
            return ChangeSemantics.ALL_CHANGES
        if change_type == ChangeType.COMPOSITE.value:
            return ChangeSemantics.COMPOSITE

        if 'change' in instruction_lower:
            words = instruction_lower.split()
            for word in words:
                if word in ["building", "vegetation", "tree", "water", "land", "ground"]:
                    return ChangeSemantics.ALL_CHANGES
        return ChangeSemantics.ALL_CHANGES

    def _identify_change_direction(self, instruction: str, semantics: ChangeSemantics) -> ChangeDirection:
        if semantics == ChangeSemantics.BIDIRECTIONAL_CONVERSION:
            return ChangeDirection.BIDIRECTIONAL
        if semantics in [ChangeSemantics.APPEARANCE, ChangeSemantics.UNIDIRECTIONAL_CONVERSION]:
            return ChangeDirection.FORWARD
        if semantics == ChangeSemantics.DISAPPEARANCE:
            return ChangeDirection.BACKWARD

        if semantics == ChangeSemantics.COMPOSITE:
            instruction_lower = instruction.lower()
            # [修复] 更精确地检测独立的appear
            has_disappearance = any(re.search(kw, instruction_lower) for kw in self.disappearance_keywords)
            cleaned = re.sub(r'\bdisappear(?:ed|ance|s)?\b', '', instruction_lower)
            has_independent_appearance = any(re.search(kw, cleaned) for kw in self.appearance_keywords)
            has_multi_conversion = len(re.findall(r'\bfrom\s+.*?\s+to\b', instruction_lower)) >= 2

            if has_multi_conversion:
                return ChangeDirection.FORWARD
            if has_independent_appearance and has_disappearance:
                return ChangeDirection.BIDIRECTIONAL
            elif has_independent_appearance:
                return ChangeDirection.FORWARD
            elif has_disappearance:
                return ChangeDirection.BACKWARD
            else:
                return ChangeDirection.BIDIRECTIONAL

        if semantics == ChangeSemantics.ALL_CHANGES:
            return ChangeDirection.BIDIRECTIONAL

        return ChangeDirection.BIDIRECTIONAL

    def _extract_involved_classes(self, instruction: str, base_result: Dict) -> List[str]:
        phase1 = base_result.get('data', {}).get('phase1_instruction', '')
        phase2 = base_result.get('data', {}).get('phase2_instruction', '')
        classes = set()
        for phase in [phase1, phase2]:
            if phase:
                classes.update([c.strip() for c in phase.split(';') if c.strip()])
        return sorted(list(classes))

    def _build_transitions(self, instruction: str, base_result: Dict,
                           semantics: ChangeSemantics, direction: ChangeDirection,
                           involved_classes: List[str]) -> List[ChangeTransition]:
        """[修复] 改进transition构建逻辑，特别是复合和转换类"""
        transitions = []

        if semantics == ChangeSemantics.ALL_CHANGES:
            for cls in involved_classes:
                transitions.append(
                    ChangeTransition(from_class="background", to_class=cls, direction=ChangeDirection.FORWARD))
                transitions.append(
                    ChangeTransition(from_class=cls, to_class="background", direction=ChangeDirection.BACKWARD))
            return transitions

        if semantics == ChangeSemantics.APPEARANCE:
            for cls in involved_classes:
                transitions.append(
                    ChangeTransition(from_class="background", to_class=cls, direction=ChangeDirection.FORWARD))
            return transitions

        if semantics == ChangeSemantics.DISAPPEARANCE:
            for cls in involved_classes:
                transitions.append(
                    ChangeTransition(from_class=cls, to_class="background", direction=ChangeDirection.BACKWARD))
            return transitions

        if semantics == ChangeSemantics.COMPOSITE:
            instruction_lower = instruction.lower()

            # ===== [修复4] 先处理多from-to复合转换 =====
            from_to_pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=\s+and\s+|\s*[;；]\s*|$)'
            from_to_matches = re.findall(from_to_pattern, instruction_lower, re.DOTALL)
            if len(from_to_matches) >= 2:
                for from_cls_raw, to_cls_raw in from_to_matches:
                    from_cls = self._clean_object_name(from_cls_raw.strip())
                    to_cls = self._clean_object_name(to_cls_raw.strip())
                    transitions.append(
                        ChangeTransition(from_class=from_cls, to_class=to_cls,
                                         direction=ChangeDirection.FORWARD))
                return transitions

            # 处理 appear/disappear 混合复合
            parts = re.split(r'\s+and\s+|\s*[;；]\s*', instruction_lower)
            for part in parts:
                part = part.strip()
                is_disappearance = any(re.search(kw, part) for kw in self.disappearance_keywords)
                # [修复] 检查独立的appear
                cleaned_part = re.sub(r'\bdisappear(?:ed|ance|s)?\b', '', part)
                is_appearance = (not is_disappearance and
                                 any(re.search(kw, cleaned_part) for kw in self.appearance_keywords))

                # 查找from-to模式（单个）
                single_from_to = re.search(r'from\s+(.*?)\s+to\s+(.*?)$', part)
                if single_from_to:
                    from_cls = self._clean_object_name(single_from_to.group(1).strip())
                    to_cls = self._clean_object_name(single_from_to.group(2).strip())
                    transitions.append(
                        ChangeTransition(from_class=from_cls, to_class=to_cls,
                                         direction=ChangeDirection.FORWARD))
                    continue

                # 查找涉及的类别
                found_class = None
                common_classes = ["low vegetation", "sports ground", "bare land",
                                  "building", "vegetation", "tree", "water"]
                for cls in common_classes:
                    if cls in part:
                        found_class = cls
                        break

                if found_class:
                    if is_disappearance:
                        transitions.append(ChangeTransition(from_class=found_class, to_class="background",
                                                            direction=ChangeDirection.BACKWARD))
                    elif is_appearance:
                        transitions.append(ChangeTransition(from_class="background", to_class=found_class,
                                                            direction=ChangeDirection.FORWARD))
            return transitions

        if semantics in [ChangeSemantics.UNIDIRECTIONAL_CONVERSION, ChangeSemantics.BIDIRECTIONAL_CONVERSION]:
            instruction_lower = instruction.lower()

            # ===== [修复5] 使用带边界的正则提取from-to =====
            from_to_pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=\s+and\s+|\s*[;；]\s*|$)'
            from_to_matches = re.findall(from_to_pattern, instruction_lower, re.DOTALL)
            if from_to_matches:
                for from_cls_raw, to_cls_raw in from_to_matches:
                    from_cls = self._clean_object_name(from_cls_raw.strip())
                    to_cls = self._clean_object_name(to_cls_raw.strip())
                    transitions.append(
                        ChangeTransition(from_class=from_cls, to_class=to_cls, direction=direction))
                    if semantics == ChangeSemantics.BIDIRECTIONAL_CONVERSION:
                        transitions.append(
                            ChangeTransition(from_class=to_cls, to_class=from_cls, direction=direction))
            else:
                for from_cls in involved_classes:
                    for to_cls in involved_classes:
                        if from_cls != to_cls:
                            transitions.append(
                                ChangeTransition(from_class=from_cls, to_class=to_cls, direction=direction))
                            if semantics == ChangeSemantics.BIDIRECTIONAL_CONVERSION:
                                transitions.append(
                                    ChangeTransition(from_class=to_cls, to_class=from_cls, direction=direction))
            return transitions

        if semantics == ChangeSemantics.AREA_CHANGE:
            for cls in involved_classes:
                transitions.append(
                    ChangeTransition(from_class=cls, to_class=cls, direction=ChangeDirection.BIDIRECTIONAL))

        return transitions

    def _clean_object_name(self, obj_name: str) -> str:
        """[修复6] 使用精确匹配而非子串匹配"""
        obj_name = obj_name.lower().strip()
        obj_name = re.sub(r'^(a|an|the)\s+', '', obj_name)
        # 去除尾部多余符号
        obj_name = obj_name.strip(' ,;.；，')

        # 精确映射表（优先完全匹配）
        direct_mapping = {
            "building": "building", "buildings": "building",
            "bare land": "bare land", "bare ground": "bare land",
            "bare soil": "bare land", "bareland": "bare land",
            "vegetation": "vegetation", "water": "water",
            "tree": "tree", "trees": "tree",
            "low vegetation": "low vegetation",
            "sports ground": "sports ground", "playground": "sports ground",
            "farmland": "bare land", "forest": "tree",
        }
        if obj_name in direct_mapping:
            return direct_mapping[obj_name]

        # 正则精确匹配（使用fullmatch而非search）
        obj_mapping = {
            r'building(s?)': "building", r'bare\s+land': "bare land",
            r'vegetation': "vegetation", r'water': "water",
            r'tree(s?)': "tree", r'low\s+vegetation': "low vegetation",
            r'sports\s+ground': "sports ground", r'playground': "sports ground"
        }
        for pattern, replacement in obj_mapping.items():
            if re.fullmatch(pattern, obj_name, re.IGNORECASE):
                return replacement
        return obj_name

    def _generate_visualization_hint(self, semantics: ChangeSemantics,
                                     transitions: List[ChangeTransition],
                                     involved_classes: List[str]) -> Dict[str, Any]:
        hint = {
            "semantics_type": semantics.value,
            "color_strategy": "default",
            "legend_items": [],
            "title_suggestion": ""
        }
        if semantics == ChangeSemantics.APPEARANCE:
            hint["color_strategy"] = "appearance"
            hint["title_suggestion"] = f"New {', '.join(involved_classes)}"
            for cls in involved_classes:
                if cls != 'background':
                    hint["legend_items"].append(
                        {"label": f"New {cls}", "description": f"Newly appeared {cls} areas"})
        elif semantics == ChangeSemantics.DISAPPEARANCE:
            hint["color_strategy"] = "disappearance"
            hint["title_suggestion"] = f"Disappeared {', '.join(involved_classes)}"
            for cls in involved_classes:
                if cls != 'background':
                    hint["legend_items"].append(
                        {"label": f"Lost {cls}", "description": f"Disappeared {cls} areas"})
        elif semantics in [ChangeSemantics.UNIDIRECTIONAL_CONVERSION, ChangeSemantics.BIDIRECTIONAL_CONVERSION]:
            hint["color_strategy"] = "conversion"
            hint["title_suggestion"] = "Class Conversion"
            for trans in transitions:
                hint["legend_items"].append(
                    {"label": str(trans),
                     "description": f"Conversion from {trans.from_class} to {trans.to_class}"})
        elif semantics == ChangeSemantics.AREA_CHANGE:
            hint["color_strategy"] = "area_change"
            hint["title_suggestion"] = f"Area Change in {', '.join(involved_classes)}"
            for cls in involved_classes:
                hint["legend_items"].append({"label": f"{cls} change", "description": f"Area change in {cls}"})
        return hint


# ==================== 指令库管理器 ====================
class InstructionLibraryManager:
    """指令库管理器（依赖外部JSON）"""

    def __init__(self, library_dir: str = "./instruction_library"):
        self.library_dir = library_dir
        os.makedirs(library_dir, exist_ok=True)

        self.core_file = os.path.join(library_dir, "core_library.json")
        self.learned_file = os.path.join(library_dir, "learned_library.json")
        self.pattern_file = os.path.join(library_dir, "patterns.json")
        self.synonym_file = os.path.join(library_dir, "synonyms.json")

        self.core_library: Dict[str, InstructionExample] = self._load_core_library()
        self.learned_library: List[InstructionExample] = self._load_learned_library()
        self.synonyms: Dict[str, List[str]] = self._load_synonyms()

        self.stats = {
            "total_instructions": len(self.core_library) + len(self.learned_library),
            "core_count": len(self.core_library),
            "learned_count": len(self.learned_library),
            "updates": 0
        }

    def _load_core_library(self) -> Dict[str, InstructionExample]:
        """
        [修复] 从JSON文件加载核心指令库，构建多级索引确保匹配成功
        """
        library = {}
        if not os.path.exists(self.core_file):
            print(f"警告: 核心指令库JSON文件不存在于 {self.core_file}。请提供该文件。")
            return library

        try:
            with open(self.core_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            data = json.loads(raw_text)
            print(f"[加载] 从JSON读取到 {len(data)} 条核心指令")

            loaded_count = 0
            for k, v in data.items():
                # 确保所有必要字段
                defaults = {
                    "keywords": [], "objects": [],
                    "change_semantics": ChangeSemantics.ALL_CHANGES.value,
                    "change_direction": ChangeDirection.BIDIRECTIONAL.value,
                    "transitions": [], "frequency": 1,
                    "source": "core", "description": "", "tags": []
                }
                for dk, dv in defaults.items():
                    if dk not in v:
                        v[dk] = dv

                # 仅对单转换类指令修正（不修正复合类）
                if (v["change_type"] == "type_conversion_change"
                        and "from" in k.lower() and "to" in k.lower()):
                    # 检查是否为单个from-to（非复合）
                    from_to_count = len(re.findall(r'\bfrom\s+', k.lower()))
                    if from_to_count == 1:
                        v = self._fix_conversion_instruction(k, v)

                # 实例化
                try:
                    example = InstructionExample(**v)
                except TypeError as e:
                    print(f"[加载警告] 指令 '{k}' 字段不匹配: {e}")
                    # 移除多余字段
                    valid_fields = {f.name for f in InstructionExample.__dataclass_fields__.values()}
                    v_clean = {fk: fv for fk, fv in v.items() if fk in valid_fields}
                    example = InstructionExample(**v_clean)

                # ===== [修复7] 构建多级索引 =====
                # 索引1：标准归一化（小写+去标点+合并空格）
                normalized = self._normalize_text(example.user_instruction)
                library[normalized] = example

                # 索引2：无空格版本
                no_space_key = re.sub(r'\s+', '', normalized)
                if no_space_key and no_space_key != normalized:
                    library[no_space_key] = example

                # 索引3：原始JSON键的归一化
                normalized_key = self._normalize_text(k)
                if normalized_key not in library:
                    library[normalized_key] = example

                # 索引4：对中文标点做额外处理
                alt_key = k.replace('；', ';').replace('，', ',')
                alt_normalized = self._normalize_text(alt_key)
                if alt_normalized not in library:
                    library[alt_normalized] = example

                loaded_count += 1

            print(f"[加载] 成功加载 {loaded_count} 条核心指令，索引总数 {len(library)}")
            return library

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            # 尝试修复常见JSON问题
            try:
                raw_text_fixed = raw_text.replace('\n', '').replace('\r', '')
                data = json.loads(raw_text_fixed)
                print(f"[加载] JSON修复后读取到 {len(data)} 条")
                # 简化加载
                for k, v in data.items():
                    for dk in ["keywords", "objects", "transitions"]:
                        if dk not in v:
                            v[dk] = []
                    for dk, dv in [("change_semantics", "all_changes"),
                                   ("change_direction", "bidirectional"),
                                   ("frequency", 1), ("source", "core")]:
                        if dk not in v:
                            v[dk] = dv
                    example = InstructionExample(**v)
                    normalized = self._normalize_text(example.user_instruction)
                    library[normalized] = example
            except Exception as e2:
                print(f"JSON修复也失败: {e2}")
        except Exception as e:
            print(f"加载核心指令库失败: {e}")
            import traceback
            traceback.print_exc()
        return library

    def _fix_conversion_instruction(self, instruction_key: str, instruction_data: Dict) -> Dict:
        """自动修正单个转换类指令"""
        try:
            pattern = r'from\s+(.+?)\s+to\s+(.+?)(?:\s+and\s+|$)'
            match = re.search(pattern, instruction_key.lower())
            if not match:
                # 回退到原始模式
                pattern = r'from\s+(.+?)\s+to\s+(.+)'
                match = re.search(pattern, instruction_key.lower())
            if match:
                from_class = self._clean_object_name(match.group(1).strip())
                to_class = self._clean_object_name(match.group(2).strip())
                from_class = self._normalize_class_name(from_class)
                to_class = self._normalize_class_name(to_class)
                new_phase_instruction = f"{from_class};{to_class}"
                instruction_data["phase1_instruction"] = new_phase_instruction
                instruction_data["phase2_instruction"] = new_phase_instruction
                if "tags" in instruction_data:
                    tags = instruction_data["tags"]
                    new_tag = f"{from_class}_to_{to_class}".replace(" ", "_")
                    if new_tag not in tags:
                        tags.append(new_tag)
                if "objects" in instruction_data:
                    objects = instruction_data["objects"]
                    if from_class not in objects:
                        objects.append(from_class)
                    if to_class not in objects:
                        objects.append(to_class)
        except Exception as e:
            print(f"修正指令 {instruction_key} 失败: {e}")
        return instruction_data

    def _clean_object_name(self, obj_name: str) -> str:
        obj_name = obj_name.lower().strip()
        obj_name = re.sub(r'^(a|an|the)\s+', '', obj_name)
        obj_name = obj_name.strip(' ,;.；，')
        return obj_name

    def _normalize_class_name(self, class_name: str) -> str:
        class_mapping = {
            "bare ground": "bare land", "bareground": "bare land", "bare soil": "bare land",
            "ground": "bare land", "soil": "bare land", "veg": "vegetation",
            "low_vegetation": "low vegetation", "sportsground": "sports ground",
            "playground": "sports ground", "waterbody": "water", "forest": "tree",
            "farmland": "bare land", "farm": "bare land"
        }
        if class_name in class_mapping:
            return class_mapping[class_name]
        for key, value in class_mapping.items():
            if key in class_name:
                return value
        return class_name

    def _normalize_text(self, text: str) -> str:
        """归一化文本用于索引匹配"""
        if not text:
            return ""
        text = text.lower()
        # 统一中文标点为英文
        chinese_punctuation = {
            '；': ';', '，': ',', '。': '.', '？': '?', '！': '!',
            '：': ':', '、': ',', '（': '(', '）': ')'
        }
        for ch, en in chinese_punctuation.items():
            text = text.replace(ch, en)
        # 去除所有标点
        text = re.sub(r'[^\w\s]', ' ', text)
        # 合并多空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _load_learned_library(self) -> List[InstructionExample]:
        if os.path.exists(self.learned_file):
            try:
                with open(self.learned_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                library = []
                for item in data:
                    defaults = {
                        "keywords": [], "objects": [],
                        "change_semantics": ChangeSemantics.ALL_CHANGES.value,
                        "change_direction": ChangeDirection.BIDIRECTIONAL.value,
                        "transitions": []
                    }
                    for dk, dv in defaults.items():
                        if dk not in item:
                            item[dk] = dv
                    library.append(InstructionExample(**item))
                return library
            except Exception as e:
                print(f"加载学习指令库失败: {e}")
        return []

    def _load_synonyms(self) -> Dict[str, List[str]]:
        if os.path.exists(self.synonym_file):
            try:
                with open(self.synonym_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载同义词库失败: {e}")
        return self._create_default_synonyms()

    def _create_default_synonyms(self) -> Dict[str, List[str]]:
        return {
            "building": ["structure", "house", "construction", "edifice"],
            "vegetation": ["plants", "greenery", "flora", "green area"],
            "bare land": ["exposed land", "unvegetated land", "bare ground"],
            "water": ["lake", "river", "pond", "waterbody"],
        }

    def search_exact(self, user_instruction: str) -> Optional[InstructionExample]:
        """
        [修复] 增强精确匹配，支持多种归一化变体
        """
        # 生成多种归一化变体
        normalized_variants = set()

        # 变体1：标准归一化
        n1 = self._normalize_text(user_instruction)
        normalized_variants.add(n1)

        # 变体2：无空格版本
        n2 = re.sub(r'\s+', '', n1)
        normalized_variants.add(n2)

        # 变体3：仅去标点保留空格
        n3 = re.sub(r'[^\w\s]', '', user_instruction.lower())
        n3 = re.sub(r'\s+', ' ', n3).strip()
        normalized_variants.add(n3)

        # 变体4：中文标点转英文后归一化
        n4 = user_instruction.replace('；', ';').replace('，', ',')
        n4 = self._normalize_text(n4)
        normalized_variants.add(n4)

        # 变体5：对多from-to指令，尝试归一化类名后重建
        if "from" in user_instruction.lower() and "to" in user_instruction.lower():
            from_to_pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=(?:\s+and\s+|\s*[;；]\s*|$))'
            matches = re.findall(from_to_pattern, user_instruction.lower(), re.DOTALL)
            if matches:
                simplified_parts = []
                for from_part, to_part in matches:
                    from_cls = self._normalize_class_name(from_part.strip())
                    to_cls = self._normalize_class_name(to_part.strip())
                    simplified_parts.append(f"from {from_cls} to {to_cls}")
                # 用 and 连接
                simplified_and = " and ".join(simplified_parts)
                normalized_variants.add(self._normalize_text(simplified_and))
                # 用分号连接
                simplified_semi = "; ".join(simplified_parts)
                normalized_variants.add(self._normalize_text(simplified_semi))

        # 在core_library中查找
        for normalized in normalized_variants:
            if normalized in self.core_library:
                return self.core_library[normalized]

        # 在learned_library中查找
        for normalized in normalized_variants:
            for example in self.learned_library:
                if self._normalize_text(example.user_instruction) == normalized:
                    return example

        # ===== [修复8] 最后尝试无标点无空格完全匹配 =====
        user_clean = re.sub(r'[^\w]', '', user_instruction.lower())
        for key, example in self.core_library.items():
            key_clean = re.sub(r'[^\w]', '', key.lower())
            if user_clean == key_clean:
                return example

        for example in self.learned_library:
            example_clean = re.sub(r'[^\w]', '', example.user_instruction.lower())
            if user_clean == example_clean:
                return example

        return None

    def get_all_instructions(self) -> List[InstructionExample]:
        unique_core = list({id(v): v for v in self.core_library.values()}.values())
        return unique_core + self.learned_library

    def add_learned_example(self, example: InstructionExample) -> bool:
        for existing in self.learned_library:
            if existing.user_instruction == example.user_instruction:
                existing.frequency += 1
                existing.last_used = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return False
        self.learned_library.append(example)
        self.stats["total_instructions"] += 1
        self.stats["learned_count"] += 1
        self.stats["updates"] += 1
        return True

    def get_stats(self) -> Dict:
        return self.stats

    def save_all(self):
        unique_core_data = {}
        for k, v in self.core_library.items():
            normalized = self._normalize_text(v.user_instruction)
            if normalized not in unique_core_data:
                unique_core_data[normalized] = v.to_dict()

        with open(self.core_file, 'w', encoding='utf-8') as f:
            json.dump(unique_core_data, f, ensure_ascii=False, indent=2)

        data = [ex.to_dict() for ex in self.learned_library]
        with open(self.learned_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("所有数据已保存")


# ==================== 翻译专家 ====================
class TranslatorExpert:
    """翻译专家（增强版）- 集成离线知识库与在线大语言模型"""

    def __init__(self, enable_llm: bool = True,
                 llm_api_key: Optional[str] = None,
                 enable_learning: bool = True,
                 library_dir: str = "./instruction_library"):
        self.library_manager = InstructionLibraryManager(library_dir)
        self.semantic_parser = SemanticParser()
        self.enable_learning = enable_learning

        self.enable_llm = enable_llm
        self.llm_client = None
        if enable_llm and llm_api_key:
            self.llm_client = LLMClient(api_key=llm_api_key)

        self.stats = {
            "total_requests": 0,
            "exact_matches": 0,
            "llm_conversions": 0,
            "heuristic_fallbacks": 0,
            "learning_updates": 0
        }

    def translate(self, user_instruction: str) -> Dict:
        self.stats["total_requests"] += 1
        print(f"\n转换指令: '{user_instruction}'")

        exact_match = self.library_manager.search_exact(user_instruction)
        if exact_match:
            self.stats["exact_matches"] += 1
            exact_match.update_usage()
            base_result = self._format_result(exact_match, "exact_match")
            print(f"  ✓ 一级转换成功：精确匹配")
        else:
            if self.enable_llm and self.llm_client:
                llm_result = self._online_llm_conversion(user_instruction)
                if llm_result:
                    self.stats["llm_conversions"] += 1
                    base_result = llm_result
                    print(f"  ✓ 二级转换成功：在线大语言模型转换")
                    if self.enable_learning:
                        self._add_to_learning_library(base_result)
                else:
                    self.stats["heuristic_fallbacks"] += 1
                    base_result = self._heuristic_translation(user_instruction)
                    print(f"  ⚠ 三级转换：启发式回退")
            else:
                self.stats["heuristic_fallbacks"] += 1
                base_result = self._heuristic_translation(user_instruction)
                print(f"  ⚠ 三级转换：启发式回退")

        semantics = self.semantic_parser.parse(user_instruction, base_result)
        base_result["semantics"] = semantics.to_dict()
        return base_result

    def _online_llm_conversion(self, user_instruction: str) -> Optional[Dict]:
        if not self.llm_client:
            return None
        result = self.llm_client.convert_instruction(user_instruction)
        if result:
            return {
                "success": True,
                "data": {
                    "original_instruction": result["user_instruction"],
                    "change_type": result["change_type"],
                    "phase1_instruction": result["phase1_instruction"],
                    "phase2_instruction": result["phase2_instruction"],
                    "confidence": result["confidence"],
                    "change_semantics": result.get("change_semantics", "unidirectional_conversion"),
                    "change_direction": result.get("change_direction", "forward"),
                    "transitions": result.get("transitions", [])
                },
                "source": "llm_online"
            }
        return None

    def _add_to_learning_library(self, llm_result: Dict) -> bool:
        try:
            data = llm_result["data"]
            example = InstructionExample(
                user_instruction=data["original_instruction"],
                change_type=data["change_type"],
                phase1_instruction=data["phase1_instruction"],
                phase2_instruction=data["phase2_instruction"],
                confidence=data["confidence"],
                description="",
                tags=[],
                source="llm_online",
                change_semantics=data.get("change_semantics", "unidirectional_conversion"),
                change_direction=data.get("change_direction", "forward"),
                transitions=data.get("transitions", [])
            )
            success = self.library_manager.add_learned_example(example)
            if success:
                self.stats["learning_updates"] += 1
                print(f"  ✓ 已添加到学习库")
            return success
        except Exception as e:
            print(f"添加到学习库失败: {e}")
            return False

    def _format_result(self, example: InstructionExample, source: str) -> Dict:
        return {
            "success": True,
            "data": {
                "original_instruction": example.user_instruction,
                "change_type": example.change_type,
                "phase1_instruction": example.phase1_instruction,
                "phase2_instruction": example.phase2_instruction,
                "confidence": example.confidence,
                "change_semantics": example.change_semantics,
                "change_direction": example.change_direction,
                "transitions": example.transitions
            },
            "source": source
        }

    def _heuristic_translation(self, user_instruction: str) -> Dict:
        """[修复] 增强启发式翻译，特别是多from-to复合指令"""
        instruction_lower = user_instruction.lower()

        # ===== [修复9] 最先检测多from-to模式 =====
        from_to_pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=\s+and\s+|\s*[;；]\s*|$)'
        from_to_matches = re.findall(from_to_pattern, instruction_lower, re.DOTALL)
        if len(from_to_matches) >= 2:
            all_classes = set()
            transitions = []
            for from_cls_raw, to_cls_raw in from_to_matches:
                from_cls = self._clean_object_name(from_cls_raw.strip())
                to_cls = self._clean_object_name(to_cls_raw.strip())
                transitions.append(f"{from_cls}→{to_cls}")
                all_classes.add(from_cls)
                all_classes.add(to_cls)
            phase_instruction = ";".join(sorted(all_classes))
            return {
                "success": True,
                "data": {
                    "original_instruction": user_instruction,
                    "change_type": ChangeType.COMPOSITE.value,
                    "phase1_instruction": phase_instruction,
                    "phase2_instruction": phase_instruction,
                    "confidence": 0.7,
                    "change_semantics": ChangeSemantics.COMPOSITE.value,
                    "change_direction": ChangeDirection.FORWARD.value,
                    "transitions": transitions
                },
                "source": "heuristic_fallback"
            }

        is_composite = (" and " in instruction_lower or
                        ";" in instruction_lower or
                        "；" in instruction_lower)
        # [修复] 精确检测appearance和disappearance
        has_disappearance = bool(re.search(r'\bdisappear', instruction_lower)) or "loss" in instruction_lower
        has_appearance = (("new" in instruction_lower or
                          (bool(re.search(r'\bappear', instruction_lower)) and not has_disappearance)) or
                         bool(re.search(r'\bemerge', instruction_lower)))

        if is_composite:
            change_type = ChangeType.COMPOSITE.value
            change_semantics = ChangeSemantics.COMPOSITE.value
            if has_appearance and has_disappearance:
                change_direction = ChangeDirection.BIDIRECTIONAL.value
            elif has_appearance:
                change_direction = ChangeDirection.FORWARD.value
            elif has_disappearance:
                change_direction = ChangeDirection.BACKWARD.value
            else:
                change_direction = ChangeDirection.BIDIRECTIONAL.value
        elif "from" in instruction_lower and "to" in instruction_lower:
            change_type = ChangeType.TYPE_CONVERSION.value
            change_semantics = ChangeSemantics.UNIDIRECTIONAL_CONVERSION.value
            change_direction = ChangeDirection.FORWARD.value
        elif has_appearance:
            change_type = ChangeType.APPEARANCE.value
            change_semantics = ChangeSemantics.APPEARANCE.value
            change_direction = ChangeDirection.FORWARD.value
        elif has_disappearance:
            change_type = ChangeType.DISAPPEARANCE.value
            change_semantics = ChangeSemantics.DISAPPEARANCE.value
            change_direction = ChangeDirection.BACKWARD.value
        elif "between" in instruction_lower:
            change_type = ChangeType.TYPE_CONVERSION.value
            change_semantics = ChangeSemantics.BIDIRECTIONAL_CONVERSION.value
            change_direction = ChangeDirection.BIDIRECTIONAL.value
        else:
            change_type = ChangeType.SIMPLE_EXISTENCE.value
            change_semantics = ChangeSemantics.ALL_CHANGES.value
            change_direction = ChangeDirection.BIDIRECTIONAL.value

        all_classes = []
        # [修复] 长类名优先匹配
        common_classes = ["low vegetation", "sports ground", "bare land",
                          "building", "vegetation", "tree", "water"]
        for cls in common_classes:
            if cls in instruction_lower:
                all_classes.append(cls)

        if all_classes:
            phase_instruction = ";".join(all_classes)
        else:
            phase_instruction = "change_area"
        phase1 = phase2 = phase_instruction

        transitions = []
        if is_composite:
            parts = re.split(r'\s+and\s+|\s*[;；]\s*', instruction_lower)
            for part in parts:
                part = part.strip()
                # 检查单个from-to
                single_from_to = re.search(r'from\s+(.*?)\s+to\s+(.*?)$', part)
                if single_from_to:
                    from_cls = self._clean_object_name(single_from_to.group(1).strip())
                    to_cls = self._clean_object_name(single_from_to.group(2).strip())
                    transitions.append(f"{from_cls}→{to_cls}")
                    continue

                is_part_disappearance = bool(re.search(r'\bdisappear', part)) or "loss" in part
                is_part_appearance = (("new" in part or
                                      (bool(re.search(r'\bappear', part)) and not is_part_disappearance)) or
                                     bool(re.search(r'\bemerge', part)))
                found_class = None
                for cls in common_classes:
                    if cls in part:
                        found_class = cls
                        break
                if found_class:
                    if is_part_disappearance:
                        transitions.append(f"{found_class}→background")
                    elif is_part_appearance:
                        transitions.append(f"background→{found_class}")
        elif change_semantics == ChangeSemantics.UNIDIRECTIONAL_CONVERSION.value:
            # [修复] 使用带边界的正则
            pattern = r'from\s+(.*?)\s+to\s+(.*?)(?=\s+and\s+|\s*[;；]\s*|$)'
            match = re.search(pattern, instruction_lower)
            if match:
                from_class = self._clean_object_name(match.group(1).strip())
                to_class = self._clean_object_name(match.group(2).strip())
                transitions = [f"{from_class}→{to_class}"]
        elif change_semantics == ChangeSemantics.APPEARANCE.value:
            for cls in all_classes:
                transitions.append(f"background→{cls}")
        elif change_semantics == ChangeSemantics.DISAPPEARANCE.value:
            for cls in all_classes:
                transitions.append(f"{cls}→background")

        return {
            "success": True,
            "data": {
                "original_instruction": user_instruction,
                "change_type": change_type,
                "phase1_instruction": phase1,
                "phase2_instruction": phase2,
                "confidence": 0.6,
                "change_semantics": change_semantics,
                "change_direction": change_direction,
                "transitions": transitions
            },
            "source": "heuristic_fallback"
        }

    def _clean_object_name(self, obj_name: str) -> str:
        """[修复] 精确匹配地物名称"""
        obj_name = obj_name.lower().strip()
        obj_name = re.sub(r'^(a|an|the)\s+', '', obj_name)
        obj_name = obj_name.strip(' ,;.；，')

        direct_mapping = {
            "building": "building", "buildings": "building",
            "bare land": "bare land", "bare ground": "bare land",
            "bare soil": "bare land", "bareland": "bare land",
            "vegetation": "vegetation", "water": "water",
            "tree": "tree", "trees": "tree",
            "low vegetation": "low vegetation",
            "sports ground": "sports ground", "playground": "sports ground",
            "farmland": "bare land", "forest": "tree",
        }
        if obj_name in direct_mapping:
            return direct_mapping[obj_name]

        obj_mapping = {
            r'building(s?)': "building", r'bare\s+land': "bare land",
            r'vegetation': "vegetation", r'water': "water",
            r'tree(s?)': "tree", r'low\s+vegetation': "low vegetation",
            r'sports\s+ground': "sports ground", r'playground': "sports ground"
        }
        for pattern, replacement in obj_mapping.items():
            if re.fullmatch(pattern, obj_name, re.IGNORECASE):
                return replacement
        return obj_name

    def get_stats(self) -> Dict:
        total = max(1, self.stats["total_requests"])
        return {
            "total_requests": self.stats["total_requests"],
            "exact_match_rate": self.stats["exact_matches"] / total * 100,
            "llm_conversion_rate": self.stats["llm_conversions"] / total * 100,
            "heuristic_fallback_rate": self.stats["heuristic_fallbacks"] / total * 100,
            "learning_updates": self.stats["learning_updates"]
        }

    def save_library(self):
        self.library_manager.save_all()


# ==================== 主接口函数 ====================
def enhanced_CDTT(test_cases: List[str],
                  verbose: bool = True,
                  save_results: bool = False,
                  enable_llm: bool = True,
                  llm_api_key: Optional[str] = "sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau") -> List[Dict]:
    if verbose:
        print("=" * 60)
        print("增强版指令转换系统（简化版）")
        print("分级转换策略：")
        print("  1. 一级：精确匹配（离线知识库）")
        print("  2. 二级：在线大语言模型转换（跳过模式匹配）")
        print("  3. 三级：启发式回退")
        print("=" * 60)
        print(f"处理 {len(test_cases)} 个指令...")

    start_time = time.time()
    translator = TranslatorExpert(
        enable_llm=enable_llm,
        llm_api_key=llm_api_key,
        enable_learning=True
    )

    results = []
    for i, instruction in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] 处理指令: '{instruction}'")

        result = translator.translate(instruction)
        result["index"] = i
        results.append(result)

        if verbose and result.get("success", False):
            data = result.get("data", {})
            semantics = result.get("semantics", {})
            print(f"  来源: {result.get('source', 'unknown')}")
            print(f"  变化语义: {semantics.get('change_semantics', 'unknown')}")
            print(f"  变化方向: {semantics.get('change_direction', 'unknown')}")
            print(f"  涉及类别: {', '.join(semantics.get('involved_classes', []))}")
            print(f"  转换关系: {', '.join(semantics.get('transitions', []))}")
            print(f"  时相1指令: {data.get('phase1_instruction', 'N/A')}")
            print(f"  时相2指令: {data.get('phase2_instruction', 'N/A')}")
            print(f"  置信度: {data.get('confidence', 0):.2f}")

    end_time = time.time()

    if verbose:
        print("\n" + "=" * 60)
        print(f"处理完成，耗时: {end_time - start_time:.2f}秒")
        stats = translator.get_stats()
        print(f"总计转换: {stats['total_requests']} 个指令")
        print(f"精确匹配率: {stats['exact_match_rate']:.1f}%")
        print(f"大语言模型转换率: {stats['llm_conversion_rate']:.1f}%")
        print(f"启发式回退率: {stats['heuristic_fallback_rate']:.1f}%")
        print(f"学习库更新: {stats['learning_updates']} 次")
        print("=" * 60)

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./translation_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "conversion_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        translator.save_library()
        print(f"\n结果已保存到: {output_dir}")

    return results


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_instructions = [
        # 复合转换类（核心修复目标）
        # "transform rock into ice and from bare land to building",
        # "from tree to bare land; from bare land to building",
        # 精确匹配类
        "Detect changes between 2015 and 2020: forest to urban, wetland loss, and cropland expansion for environmental assessment",
        # "building disappeared",
        # "from bare land to building",
        # "detecting building change",
        # "new building and new vegetation",
        # "please find new low vegetation",
        # # 复合类
        # "new building and disappear low vegetation",
        # "disappear building and new low vegetation",
        # "new building and new vegetation",
        # "building disappear and vegetation disappear",
        # # LLM/启发式类
        # "appearance of farmland",
        # "emerge water",
        # "appear low vegetation",
        # "from low vegetation to building and from building to water",
    ]

    LLM_API_KEY = "sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau"

    results = enhanced_CDTT(
        test_instructions,
        verbose=True,
        save_results=False,
        enable_llm=True,
        llm_api_key=LLM_API_KEY
    )

    print("\n" + "=" * 60)
    print("详细转换结果：")
    print("=" * 60)

    for result in results:
        print(f"\n指令: {result.get('data', {}).get('original_instruction', 'N/A')}")
        print(f"  来源: {result.get('source', 'unknown')}")
        print(f"  变化类型: {result.get('data', {}).get('change_type', 'N/A')}")
        print(f"  变化语义: {result.get('semantics', {}).get('change_semantics', 'N/A')}")
        print(f"  变化方向: {result.get('semantics', {}).get('change_direction', 'N/A')}")
        print(f"  时相1指令: {result.get('data', {}).get('phase1_instruction', 'N/A')}")
        print(f"  时相2指令: {result.get('data', {}).get('phase2_instruction', 'N/A')}")
        print(f"  转换关系(data): {result.get('data', {}).get('transitions', [])}")
        print(f"  转换关系(semantics): {result.get('semantics', {}).get('transitions', [])}")
        print(f"  置信度: {result.get('data', {}).get('confidence', 0):.2f}")
