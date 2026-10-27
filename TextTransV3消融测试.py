# experiment_runner.py
import json
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Tuple
import sys

# 导入TextTransV3中的函数
sys.path.append('.')
from TextTransV3 import enhanced_CDTT, InstructionExample, ChangeSemantics, ChangeDirection, ChangeType

# 设置随机种子以确保可重复性
random.seed(42)


def generate_test_instructions(num_each: int = 60) -> List[Dict[str, str]]:
    """
    为五类变化各生成 num_each 条指令，返回列表每个元素包含指令文本和类别。
    使用25类常见土地覆盖土地利用类别，确保指令不重复。
    """
    instructions = []
    used_instructions = set()  # 用于跟踪已使用的指令，避免重复

    # 25类常见土地覆盖土地利用类别
    objects = [
        # 建筑与基础设施
        "building", "residential", "commercial", "industrial", "transportation",
        # 植被
        "forest", "grassland", "cropland", "orchard", "vineyard",
        # 水体
        "water", "river", "lake", "wetland", "ocean",
        # 裸地与人工地表
        "bare land", "rock", "sand", "impervious", "mining",
        # 其他
        "snow", "ice", "shrubland", "savanna", "mangrove"
    ]

    # 各类的模板
    templates = {
        "appearance": [
            "new {obj}",
            "appearance of {obj}",
            "appear {obj}",
            "emerge {obj}"
        ],
        "disappearance": [
            "{obj} disappeared",
            "disappearance of {obj}",
            "loss of {obj}",
            "{obj} vanish"
        ],
        "transformation": [
            "from {src} to {dst}",
            "conversion from {src} to {dst}",
            "transform {src} into {dst}",
            "change from {src} to {dst}"
        ],
        "bidirectional": [
            "detecting {obj} change",
            "find changes in {obj}",
            "{obj} change detection",
            "identify {obj} changes"
        ],
        "composite": [
            "new {obj1} and disappear {obj2}",
            "disappear {obj1} and new {obj2}",
            "new {obj1} and new {obj2}",
            "disappear {obj1} and disappear {obj2}"
        ]
    }

    # 为每类生成指令
    category_names = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]

    for cat in category_names:
        count = 0
        attempts = 0
        max_attempts = num_each * 10  # 防止无限循环

        while count < num_each and attempts < max_attempts:
            attempts += 1

            if cat in ["appearance", "disappearance", "bidirectional"]:
                obj = random.choice(objects)
                template = random.choice(templates[cat])
                instruction = template.format(obj=obj)
            elif cat == "transformation":
                src, dst = random.sample(objects, 2)  # 确保源和目标不同
                template = random.choice(templates[cat])
                instruction = template.format(src=src, dst=dst)
            elif cat == "composite":
                obj1, obj2 = random.sample(objects, 2)  # 确保两个对象不同
                template = random.choice(templates[cat])
                instruction = template.format(obj1=obj1, obj2=obj2)

            # 检查是否已存在
            if instruction not in used_instructions:
                used_instructions.add(instruction)
                instructions.append({"instruction": instruction, "category": cat})
                count += 1

        if count < num_each:
            print(f"警告: 类别 {cat} 只生成了 {count}/{num_each} 条不重复指令")

    return instructions


def create_offline_library(instructions: List[Dict[str, str]], output_file: str = "core_library.json"):
    """创建离线指令库JSON文件"""
    library = {}

    for item in instructions:
        instruction = item["instruction"]
        category = item["category"]

        # 推断变化类型和语义
        if category == "appearance":
            change_type = "appearance_change"
            change_semantics = "appearance"
            change_direction = "forward"
            transitions = ["background→" + extract_object(instruction)]
        elif category == "disappearance":
            change_type = "disappearance_change"
            change_semantics = "disappearance"
            change_direction = "backward"
            transitions = [extract_object(instruction) + "→background"]
        elif category == "transformation":
            change_type = "type_conversion_change"
            change_semantics = "unidirectional_conversion"
            change_direction = "forward"
            # 提取from和to
            from_to = extract_from_to(instruction)
            transitions = [f"{from_to[0]}→{from_to[1]}"]
        elif category == "bidirectional":
            change_type = "all_change"
            change_semantics = "all_changes"
            change_direction = "bidirectional"
            obj = extract_object(instruction)
            transitions = [f"background→{obj}", f"{obj}→background"]
        elif category == "composite":
            change_type = "composite_change"
            change_semantics = "composite"
            # 分析复合指令
            if "new" in instruction.lower() and "disappear" in instruction.lower():
                change_direction = "bidirectional"
                transitions = analyze_composite_transitions(instruction)
            elif "new" in instruction.lower():
                change_direction = "forward"
                transitions = analyze_composite_transitions(instruction)
            elif "disappear" in instruction.lower():
                change_direction = "backward"
                transitions = analyze_composite_transitions(instruction)
            else:
                change_direction = "bidirectional"
                transitions = analyze_composite_transitions(instruction)

        # 提取对象
        objects = extract_objects_from_instruction(instruction)
        phase_instruction = ";".join(objects) if objects else "change_area"

        # 创建InstructionExample
        example = InstructionExample(
            user_instruction=instruction,
            change_type=change_type,
            phase1_instruction=phase_instruction,
            phase2_instruction=phase_instruction,
            confidence=0.9,
            description=f"{category}指令",
            tags=[category],
            source="core",
            change_semantics=change_semantics,
            change_direction=change_direction,
            transitions=transitions
        )

        # 使用归一化的指令作为键
        normalized_key = normalize_instruction(instruction)
        library[normalized_key] = example.to_dict()

    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    print(f"已创建离线库，包含 {len(library)} 条指令，保存到 {output_file}")
    return output_file


def extract_object(instruction: str) -> str:
    """从指令中提取主要对象"""
    objects = [
        "building", "residential", "commercial", "industrial", "transportation",
        "forest", "grassland", "cropland", "orchard", "vineyard",
        "water", "river", "lake", "wetland", "ocean",
        "bare land", "rock", "sand", "impervious", "mining",
        "snow", "ice", "shrubland", "savanna", "mangrove"
    ]
    instruction_lower = instruction.lower()
    for obj in objects:
        if obj in instruction_lower:
            return obj
    return "unknown"


def extract_from_to(instruction: str) -> tuple:
    """从转换指令中提取源和目标"""
    import re
    pattern = r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)'
    match = re.search(pattern, instruction.lower())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "unknown", "unknown"


def analyze_composite_transitions(instruction: str) -> List[str]:
    """分析复合指令的转换关系"""
    transitions = []
    instruction_lower = instruction.lower()

    # 检查多个from-to
    import re
    from_to_pattern = r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)'
    matches = re.findall(from_to_pattern, instruction_lower)
    if matches:
        for from_obj, to_obj in matches:
            transitions.append(f"{from_obj}→{to_obj}")
    else:
        # 处理new/disappear
        objects = [
            "building", "residential", "commercial", "industrial", "transportation",
            "forest", "grassland", "cropland", "orchard", "vineyard",
            "water", "river", "lake", "wetland", "ocean",
            "bare land", "rock", "sand", "impervious", "mining",
            "snow", "ice", "shrubland", "savanna", "mangrove"
        ]
        if "new" in instruction.lower():
            for obj in objects:
                if obj in instruction_lower:
                    transitions.append(f"background→{obj}")
        if "disappear" in instruction.lower():
            for obj in objects:
                if obj in instruction_lower:
                    transitions.append(f"{obj}→background")

    return transitions


def extract_objects_from_instruction(instruction: str) -> List[str]:
    """从指令中提取所有对象"""
    objects = []
    common_objects = [
        "building", "residential", "commercial", "industrial", "transportation",
        "forest", "grassland", "cropland", "orchard", "vineyard",
        "water", "river", "lake", "wetland", "ocean",
        "bare land", "rock", "sand", "impervious", "mining",
        "snow", "ice", "shrubland", "savanna", "mangrove"
    ]
    instruction_lower = instruction.lower()
    for obj in common_objects:
        if obj in instruction_lower:
            objects.append(obj)
    return objects


def normalize_instruction(instruction: str) -> str:
    """归一化指令用于作为键"""
    import re
    text = instruction.lower()
    # 统一中文标点
    chinese_punctuation = {'；': ';', '，': ',', '。': '.', '？': '?', '！': '!', '：': ':', '、': ',', '（': '(', '）': ')'}
    for ch, en in chinese_punctuation.items():
        text = text.replace(ch, en)
    # 去除所有标点
    text = re.sub(r'[^\w\s]', ' ', text)
    # 合并多空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def run_experiment_method(method_name: str, test_instructions: List[str], enable_llm: bool, library_file: str,
                          output_file: str) -> Tuple[List, Dict[str, float]]:
    """运行一种实验方法，返回结果和每个指令的处理时间"""
    print(f"\n{'=' * 60}")
    print(f"运行方法: {method_name}")
    print(f"启用LLM: {enable_llm}")
    print(f"使用库文件: {library_file}")
    print(f"{'=' * 60}")

    # 备份原库文件
    original_lib_dir = "./instruction_library"
    backup_lib_dir = f"./instruction_library_backup_{method_name}"

    # 创建临时库目录
    if os.path.exists(original_lib_dir):
        import shutil
        if os.path.exists(backup_lib_dir):
            shutil.rmtree(backup_lib_dir)
        shutil.copytree(original_lib_dir, backup_lib_dir)

    # 复制指定的库文件
    os.makedirs(original_lib_dir, exist_ok=True)
    target_file = os.path.join(original_lib_dir, "core_library.json")
    if os.path.exists(library_file):
        import shutil
        shutil.copy(library_file, target_file)
    else:
        # 创建空库
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    # 运行转换并计时
    all_results = []
    instruction_times = {}  # 记录每个指令的处理时间
    total_start_time = time.time()

    for instruction in test_instructions:
        instruction_start_time = time.time()
        result = enhanced_CDTT(
            [instruction],  # 每次只处理一个指令
            verbose=False,
            save_results=False,
            enable_llm=enable_llm,
            llm_api_key="sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau"
        )
        instruction_end_time = time.time()
        instruction_time = instruction_end_time - instruction_start_time
        instruction_times[instruction] = instruction_time
        all_results.extend(result)

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # 恢复原库文件
    if os.path.exists(backup_lib_dir):
        import shutil
        shutil.rmtree(original_lib_dir)
        shutil.copytree(backup_lib_dir, original_lib_dir)
        shutil.rmtree(backup_lib_dir)

    # 保存结果到txt文件，按照指定格式
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入实验信息
        f.write(f"实验方法: {method_name}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总指令数: {len(test_instructions)}\n")
        f.write(f"总处理时间: {total_processing_time:.2f}秒\n")
        f.write(f"平均每个指令处理时间: {total_processing_time / len(test_instructions):.4f}秒\n")
        f.write(f"启用LLM: {enable_llm}\n")
        f.write(f"使用库文件: {library_file}\n")
        f.write("=" * 60 + "\n\n")

        # 统计信息
        success_count = sum(1 for r in all_results if r.get("success", False))
        f.write(f"成功转换: {success_count}/{len(test_instructions)}\n")
        f.write(f"成功率: {success_count / len(test_instructions) * 100:.1f}%\n\n")

        # 按类别统计
        categories = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]
        for category in categories:
            category_instructions = [inst for inst in test_instructions if
                                     any(cat in inst.lower() for cat in [category])]
            if category_instructions:
                category_success = sum(1 for r in all_results
                                       if r.get("success", False) and
                                       any(cat in r.get("data", {}).get("original_instruction", "").lower()
                                           for cat in [category]))
                f.write(
                    f"{category}: {category_success}/{len(category_instructions)} ({category_success / len(category_instructions) * 100:.1f}%)\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # 详细结果 - 按照指定格式
        for i, result in enumerate(all_results, 1):
            original_instruction = result.get('data', {}).get('original_instruction', 'N/A')
            source = result.get('source', 'unknown')
            change_semantics = result.get('semantics', {}).get('change_semantics', 'N/A')
            change_direction = result.get('semantics', {}).get('change_direction', 'N/A')
            involved_classes = result.get('semantics', {}).get('involved_classes', [])
            transitions = result.get('semantics', {}).get('transitions', [])
            phase1_instruction = result.get('data', {}).get('phase1_instruction', 'N/A')
            phase2_instruction = result.get('data', {}).get('phase2_instruction', 'N/A')
            confidence = result.get('data', {}).get('confidence', 0)

            # 获取该指令的处理时间
            instruction_time = instruction_times.get(original_instruction, 0.0)

            # 确定转换状态
            if source == "exact_match":
                status = "✓ 一级转换成功：精确匹配"
            elif source == "llm_online":
                status = "✓ 二级转换成功：在线大语言模型转换"
            elif source == "heuristic_fallback":
                status = "⚠ 三级转换：启发式回退"
            else:
                status = "未知转换状态"

            # 写入详细结果
            f.write(f"转换指令: '{original_instruction}'\n")
            f.write(f"  处理时间: {instruction_time:.4f}秒\n")
            f.write(f"  {status}\n")
            f.write(f"  来源: {source}\n")
            f.write(f"  变化语义: {change_semantics}\n")
            f.write(f"  变化方向: {change_direction}\n")
            f.write(f"  涉及类别: {', '.join(involved_classes)}\n")
            f.write(f"  转换关系: {', '.join(transitions)}\n")
            f.write(f"  时相1指令: {phase1_instruction}\n")
            f.write(f"  时相2指令: {phase2_instruction}\n")
            f.write(f"  置信度: {confidence:.2f}\n")
            f.write("-" * 60 + "\n")

    print(f"方法 {method_name} 完成，总处理时间: {total_processing_time:.2f}秒，结果保存到 {output_file}")
    return all_results, instruction_times


def main():
    """主实验函数"""
    print("开始实验：比较三种指令转换方法")
    print("=" * 60)

    # 1. 生成测试指令（每类60条，共300条）
    print("生成测试指令...")
    test_instructions_data = generate_test_instructions(num_each=60)
    test_instructions = [item["instruction"] for item in test_instructions_data]

    # 确保指令不重复（双重保险）
    test_instructions = list(set(test_instructions))
    print(f"生成了 {len(test_instructions)} 条不重复的测试指令")

    # 2. 创建离线库文件
    print("创建离线库文件...")

    # 方法1：完整离线库（包含所有指令）
    full_library_file = "full_core_library.json"
    create_offline_library(test_instructions_data, full_library_file)

    # 方法2：空离线库（用于仅在线转换）
    empty_library_file = "empty_core_library.json"
    with open(empty_library_file, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    # 3. 运行三种方法（使用相同的指令集）
    methods = [
        ("仅离线库处理", False, full_library_file, "results_offline_only.txt"),
        ("仅在线转换处理", True, empty_library_file, "results_online_only.txt"),
        ("结合离线和在线处理", True, full_library_file, "results_combined.txt")
    ]

    all_results = {}
    all_instruction_times = {}

    for method_name, enable_llm, library_file, output_file in methods:
        results, instruction_times = run_experiment_method(method_name, test_instructions, enable_llm, library_file,
                                                           output_file)
        all_results[method_name] = results
        all_instruction_times[method_name] = instruction_times

    # 4. 生成汇总报告
    summary_file = "experiment_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("实验汇总报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总指令数: {len(test_instructions)}\n")
        f.write(f"每类指令数: 60条\n")
        f.write(f"指令类别: appearance, disappearance, transformation, bidirectional, composite\n")
        f.write(f"地物类别数: 25类\n\n")

        f.write("方法比较:\n")
        f.write("-" * 60 + "\n")

        for method_name, enable_llm, library_file, output_file in methods:
            f.write(f"\n{method_name}:\n")
            f.write(f"  启用LLM: {enable_llm}\n")
            f.write(f"  使用库文件: {library_file}\n")
            f.write(f"  结果文件: {output_file}\n")

        f.write("\n" + "=" * 60 + "\n")

        # 添加成功率对比表
        f.write("\n成功率对比:\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'方法':<20} {'总成功率':<10} {'appearance':<10} {'disappearance':<12} {'transformation':<12} {'bidirectional':<12} {'composite':<10}\n")

        for method_name, _, _, _ in methods:
            if method_name in all_results:
                results = all_results[method_name]
                success_count = sum(1 for r in results if r.get("success", False))
                total_success_rate = f"{success_count / len(test_instructions) * 100:.1f}%"

                # 按类别统计
                categories = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]
                category_rates = []
                for category in categories:
                    category_instructions = [inst for inst in test_instructions if
                                             any(cat in inst.lower() for cat in [category])]
                    if category_instructions:
                        category_success = sum(1 for r in results
                                               if r.get("success", False) and
                                               any(cat in r.get("data", {}).get("original_instruction", "").lower()
                                                   for cat in [category]))
                        rate = f"{category_success / len(category_instructions) * 100:.1f}%"
                    else:
                        rate = "N/A"
                    category_rates.append(rate)

                f.write(
                    f"{method_name:<20} {total_success_rate:<10} {category_rates[0]:<10} {category_rates[1]:<12} {category_rates[2]:<12} {category_rates[3]:<12} {category_rates[4]:<10}\n")

        # 添加处理时间对比
        f.write("\n处理时间对比:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'方法':<20} {'总处理时间(秒)':<15} {'平均每个指令时间(秒)':<20}\n")

        for method_name, _, _, _ in methods:
            if method_name in all_instruction_times:
                instruction_times = all_instruction_times[method_name]
                total_time = sum(instruction_times.values())
                avg_time = total_time / len(instruction_times) if instruction_times else 0
                f.write(f"{method_name:<20} {total_time:<15.2f} {avg_time:<20.4f}\n")

    print(f"\n实验完成！汇总报告已保存到 {summary_file}")
    print("结果文件:")
    for _, _, _, output_file in methods:
        print(f"  - {output_file}")


if __name__ == "__main__":
    main()


# # experiment_runner.py
# import json
# import os
# import time
# import random
# from datetime import datetime
# from typing import List, Dict
# import sys
#
# # 导入TextTransV3中的函数
# sys.path.append('.')
# from TextTransV3 import enhanced_CDTT, InstructionExample, ChangeSemantics, ChangeDirection, ChangeType
#
# # 设置随机种子以确保可重复性
# random.seed(42)
#
#
# def generate_test_instructions(num_each: int = 60) -> List[Dict[str, str]]:
#     """
#     为五类变化各生成 num_each 条指令，返回列表每个元素包含指令文本和类别。
#     使用25类常见土地覆盖土地利用类别，确保指令不重复。
#     """
#     instructions = []
#     used_instructions = set()  # 用于跟踪已使用的指令，避免重复
#
#     # 25类常见土地覆盖土地利用类别
#     objects = [
#         # 建筑与基础设施
#         "building", "residential", "commercial", "industrial", "transportation",
#         # 植被
#         "forest", "grassland", "cropland", "orchard", "vineyard",
#         # 水体
#         "water", "river", "lake", "wetland", "ocean",
#         # 裸地与人工地表
#         "bare land", "rock", "sand", "impervious", "mining",
#         # 其他
#         "snow", "ice", "shrubland", "savanna", "mangrove"
#     ]
#
#     # 各类的模板
#     templates = {
#         "appearance": [
#             "new {obj}",
#             "appearance of {obj}",
#             "appear {obj}",
#             "emerge {obj}"
#         ],
#         "disappearance": [
#             "{obj} disappeared",
#             "disappearance of {obj}",
#             "loss of {obj}",
#             "{obj} vanish"
#         ],
#         "transformation": [
#             "from {src} to {dst}",
#             "conversion from {src} to {dst}",
#             "transform {src} into {dst}",
#             "change from {src} to {dst}"
#         ],
#         "bidirectional": [
#             "detecting {obj} change",
#             "find changes in {obj}",
#             "{obj} change detection",
#             "identify {obj} changes"
#         ],
#         "composite": [
#             "new {obj1} and disappear {obj2}",
#             "disappear {obj1} and new {obj2}",
#             "new {obj1} and new {obj2}",
#             "disappear {obj1} and disappear {obj2}"
#         ]
#     }
#
#     # 为每类生成指令
#     category_names = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]
#
#     for cat in category_names:
#         count = 0
#         attempts = 0
#         max_attempts = num_each * 10  # 防止无限循环
#
#         while count < num_each and attempts < max_attempts:
#             attempts += 1
#
#             if cat in ["appearance", "disappearance", "bidirectional"]:
#                 obj = random.choice(objects)
#                 template = random.choice(templates[cat])
#                 instruction = template.format(obj=obj)
#             elif cat == "transformation":
#                 src, dst = random.sample(objects, 2)  # 确保源和目标不同
#                 template = random.choice(templates[cat])
#                 instruction = template.format(src=src, dst=dst)
#             elif cat == "composite":
#                 obj1, obj2 = random.sample(objects, 2)  # 确保两个对象不同
#                 template = random.choice(templates[cat])
#                 instruction = template.format(obj1=obj1, obj2=obj2)
#
#             # 检查是否已存在
#             if instruction not in used_instructions:
#                 used_instructions.add(instruction)
#                 instructions.append({"instruction": instruction, "category": cat})
#                 count += 1
#
#         if count < num_each:
#             print(f"警告: 类别 {cat} 只生成了 {count}/{num_each} 条不重复指令")
#
#     return instructions
#
#
# def create_offline_library(instructions: List[Dict[str, str]], output_file: str = "core_library.json"):
#     """创建离线指令库JSON文件"""
#     library = {}
#
#     for item in instructions:
#         instruction = item["instruction"]
#         category = item["category"]
#
#         # 推断变化类型和语义
#         if category == "appearance":
#             change_type = "appearance_change"
#             change_semantics = "appearance"
#             change_direction = "forward"
#             transitions = ["background→" + extract_object(instruction)]
#         elif category == "disappearance":
#             change_type = "disappearance_change"
#             change_semantics = "disappearance"
#             change_direction = "backward"
#             transitions = [extract_object(instruction) + "→background"]
#         elif category == "transformation":
#             change_type = "type_conversion_change"
#             change_semantics = "unidirectional_conversion"
#             change_direction = "forward"
#             # 提取from和to
#             from_to = extract_from_to(instruction)
#             transitions = [f"{from_to[0]}→{from_to[1]}"]
#         elif category == "bidirectional":
#             change_type = "all_change"
#             change_semantics = "all_changes"
#             change_direction = "bidirectional"
#             obj = extract_object(instruction)
#             transitions = [f"background→{obj}", f"{obj}→background"]
#         elif category == "composite":
#             change_type = "composite_change"
#             change_semantics = "composite"
#             # 分析复合指令
#             if "new" in instruction.lower() and "disappear" in instruction.lower():
#                 change_direction = "bidirectional"
#                 transitions = analyze_composite_transitions(instruction)
#             elif "new" in instruction.lower():
#                 change_direction = "forward"
#                 transitions = analyze_composite_transitions(instruction)
#             elif "disappear" in instruction.lower():
#                 change_direction = "backward"
#                 transitions = analyze_composite_transitions(instruction)
#             else:
#                 change_direction = "bidirectional"
#                 transitions = analyze_composite_transitions(instruction)
#
#         # 提取对象
#         objects = extract_objects_from_instruction(instruction)
#         phase_instruction = ";".join(objects) if objects else "change_area"
#
#         # 创建InstructionExample
#         example = InstructionExample(
#             user_instruction=instruction,
#             change_type=change_type,
#             phase1_instruction=phase_instruction,
#             phase2_instruction=phase_instruction,
#             confidence=0.9,
#             description=f"{category}指令",
#             tags=[category],
#             source="core",
#             change_semantics=change_semantics,
#             change_direction=change_direction,
#             transitions=transitions
#         )
#
#         # 使用归一化的指令作为键
#         normalized_key = normalize_instruction(instruction)
#         library[normalized_key] = example.to_dict()
#
#     # 保存到JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(library, f, ensure_ascii=False, indent=2)
#
#     print(f"已创建离线库，包含 {len(library)} 条指令，保存到 {output_file}")
#     return output_file
#
#
# def extract_object(instruction: str) -> str:
#     """从指令中提取主要对象"""
#     objects = [
#         "building", "residential", "commercial", "industrial", "transportation",
#         "forest", "grassland", "cropland", "orchard", "vineyard",
#         "water", "river", "lake", "wetland", "ocean",
#         "bare land", "rock", "sand", "impervious", "mining",
#         "snow", "ice", "shrubland", "savanna", "mangrove"
#     ]
#     instruction_lower = instruction.lower()
#     for obj in objects:
#         if obj in instruction_lower:
#             return obj
#     return "unknown"
#
#
# def extract_from_to(instruction: str) -> tuple:
#     """从转换指令中提取源和目标"""
#     import re
#     pattern = r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)'
#     match = re.search(pattern, instruction.lower())
#     if match:
#         return match.group(1).strip(), match.group(2).strip()
#     return "unknown", "unknown"
#
#
# def analyze_composite_transitions(instruction: str) -> List[str]:
#     """分析复合指令的转换关系"""
#     transitions = []
#     instruction_lower = instruction.lower()
#
#     # 检查多个from-to
#     import re
#     from_to_pattern = r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)'
#     matches = re.findall(from_to_pattern, instruction_lower)
#     if matches:
#         for from_obj, to_obj in matches:
#             transitions.append(f"{from_obj}→{to_obj}")
#     else:
#         # 处理new/disappear
#         objects = [
#             "building", "residential", "commercial", "industrial", "transportation",
#             "forest", "grassland", "cropland", "orchard", "vineyard",
#             "water", "river", "lake", "wetland", "ocean",
#             "bare land", "rock", "sand", "impervious", "mining",
#             "snow", "ice", "shrubland", "savanna", "mangrove"
#         ]
#         if "new" in instruction_lower:
#             for obj in objects:
#                 if obj in instruction_lower:
#                     transitions.append(f"background→{obj}")
#         if "disappear" in instruction_lower:
#             for obj in objects:
#                 if obj in instruction_lower:
#                     transitions.append(f"{obj}→background")
#
#     return transitions
#
#
# def extract_objects_from_instruction(instruction: str) -> List[str]:
#     """从指令中提取所有对象"""
#     objects = []
#     common_objects = [
#         "building", "residential", "commercial", "industrial", "transportation",
#         "forest", "grassland", "cropland", "orchard", "vineyard",
#         "water", "river", "lake", "wetland", "ocean",
#         "bare land", "rock", "sand", "impervious", "mining",
#         "snow", "ice", "shrubland", "savanna", "mangrove"
#     ]
#     instruction_lower = instruction.lower()
#     for obj in common_objects:
#         if obj in instruction_lower:
#             objects.append(obj)
#     return objects
#
#
# def normalize_instruction(instruction: str) -> str:
#     """归一化指令用于作为键"""
#     import re
#     text = instruction.lower()
#     # 统一中文标点
#     chinese_punctuation = {'；': ';', '，': ',', '。': '.', '？': '?', '！': '!', '：': ':', '、': ',', '（': '(', '）': ')'}
#     for ch, en in chinese_punctuation.items():
#         text = text.replace(ch, en)
#     # 去除所有标点
#     text = re.sub(r'[^\w\s]', ' ', text)
#     # 合并多空格
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()
#
#
# def run_experiment_method(method_name: str, test_instructions: List[str], enable_llm: bool, library_file: str,
#                           output_file: str):
#     """运行一种实验方法"""
#     print(f"\n{'=' * 60}")
#     print(f"运行方法: {method_name}")
#     print(f"启用LLM: {enable_llm}")
#     print(f"使用库文件: {library_file}")
#     print(f"{'=' * 60}")
#
#     # 备份原库文件
#     original_lib_dir = "./instruction_library"
#     backup_lib_dir = f"./instruction_library_backup_{method_name}"
#
#     # 创建临时库目录
#     if os.path.exists(original_lib_dir):
#         import shutil
#         if os.path.exists(backup_lib_dir):
#             shutil.rmtree(backup_lib_dir)
#         shutil.copytree(original_lib_dir, backup_lib_dir)
#
#     # 复制指定的库文件
#     os.makedirs(original_lib_dir, exist_ok=True)
#     target_file = os.path.join(original_lib_dir, "core_library.json")
#     if os.path.exists(library_file):
#         import shutil
#         shutil.copy(library_file, target_file)
#     else:
#         # 创建空库
#         with open(target_file, 'w', encoding='utf-8') as f:
#             json.dump({}, f)
#
#     # 运行转换
#     start_time = time.time()
#     results = enhanced_CDTT(
#         test_instructions,
#         verbose=False,  # 关闭控制台输出，我们将在文件中保存详细结果
#         save_results=False,
#         enable_llm=enable_llm,
#         llm_api_key="sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau"
#     )
#     end_time = time.time()
#
#     # 恢复原库文件
#     if os.path.exists(backup_lib_dir):
#         import shutil
#         shutil.rmtree(original_lib_dir)
#         shutil.copytree(backup_lib_dir, original_lib_dir)
#         shutil.rmtree(backup_lib_dir)
#
#     # 保存结果到txt文件，按照指定格式
#     with open(output_file, 'w', encoding='utf-8') as f:
#         # 写入实验信息
#         f.write(f"实验方法: {method_name}\n")
#         f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"总指令数: {len(test_instructions)}\n")
#         f.write(f"处理时间: {end_time - start_time:.2f}秒\n")
#         f.write(f"启用LLM: {enable_llm}\n")
#         f.write(f"使用库文件: {library_file}\n")
#         f.write("=" * 60 + "\n\n")
#
#         # 统计信息
#         success_count = sum(1 for r in results if r.get("success", False))
#         f.write(f"成功转换: {success_count}/{len(test_instructions)}\n")
#         f.write(f"成功率: {success_count / len(test_instructions) * 100:.1f}%\n\n")
#
#         # 按类别统计
#         categories = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]
#         for category in categories:
#             category_instructions = [inst for inst in test_instructions if
#                                      any(cat in inst.lower() for cat in [category])]
#             if category_instructions:
#                 category_success = sum(1 for r in results
#                                        if r.get("success", False) and
#                                        any(cat in r.get("data", {}).get("original_instruction", "").lower()
#                                            for cat in [category]))
#                 f.write(
#                     f"{category}: {category_success}/{len(category_instructions)} ({category_success / len(category_instructions) * 100:.1f}%)\n")
#
#         f.write("\n" + "=" * 60 + "\n\n")
#
#         # 详细结果 - 按照指定格式
#         for i, result in enumerate(results, 1):
#             original_instruction = result.get('data', {}).get('original_instruction', 'N/A')
#             source = result.get('source', 'unknown')
#             change_semantics = result.get('semantics', {}).get('change_semantics', 'N/A')
#             change_direction = result.get('semantics', {}).get('change_direction', 'N/A')
#             involved_classes = result.get('semantics', {}).get('involved_classes', [])
#             transitions = result.get('semantics', {}).get('transitions', [])
#             phase1_instruction = result.get('data', {}).get('phase1_instruction', 'N/A')
#             phase2_instruction = result.get('data', {}).get('phase2_instruction', 'N/A')
#             confidence = result.get('data', {}).get('confidence', 0)
#
#             # 确定转换状态
#             if source == "exact_match":
#                 status = "✓ 一级转换成功：精确匹配"
#             elif source == "llm_online":
#                 status = "✓ 二级转换成功：在线大语言模型转换"
#             elif source == "heuristic_fallback":
#                 status = "⚠ 三级转换：启发式回退"
#             else:
#                 status = "未知转换状态"
#
#             # 写入详细结果
#             f.write(f"转换指令: '{original_instruction}'\n")
#             f.write(f"  {status}\n")
#             f.write(f"  来源: {source}\n")
#             f.write(f"  变化语义: {change_semantics}\n")
#             f.write(f"  变化方向: {change_direction}\n")
#             f.write(f"  涉及类别: {', '.join(involved_classes)}\n")
#             f.write(f"  转换关系: {', '.join(transitions)}\n")
#             f.write(f"  时相1指令: {phase1_instruction}\n")
#             f.write(f"  时相2指令: {phase2_instruction}\n")
#             f.write(f"  置信度: {confidence:.2f}\n")
#             f.write("-" * 60 + "\n")
#
#     print(f"方法 {method_name} 完成，结果保存到 {output_file}")
#     return results
#
#
# def main():
#     """主实验函数"""
#     print("开始实验：比较三种指令转换方法")
#     print("=" * 60)
#
#     # 1. 生成测试指令（每类60条，共300条）
#     print("生成测试指令...")
#     test_instructions_data = generate_test_instructions(num_each=60)
#     test_instructions = [item["instruction"] for item in test_instructions_data]
#
#     # 确保指令不重复（双重保险）
#     test_instructions = list(set(test_instructions))
#     print(f"生成了 {len(test_instructions)} 条不重复的测试指令")
#
#     # 2. 创建离线库文件
#     print("创建离线库文件...")
#
#     # 方法1：完整离线库（包含所有指令）
#     full_library_file = "full_core_library.json"
#     create_offline_library(test_instructions_data, full_library_file)
#
#     # 方法2：空离线库（用于仅在线转换）
#     empty_library_file = "empty_core_library.json"
#     with open(empty_library_file, 'w', encoding='utf-8') as f:
#         json.dump({}, f)
#
#     # 3. 运行三种方法（使用相同的指令集）
#     methods = [
#         ("仅离线库处理", False, full_library_file, "results_offline_only.txt"),
#         ("仅在线转换处理", True, empty_library_file, "results_online_only.txt"),
#         ("结合离线和在线处理", True, full_library_file, "results_combined.txt")
#     ]
#
#     all_results = {}
#     for method_name, enable_llm, library_file, output_file in methods:
#         results = run_experiment_method(method_name, test_instructions, enable_llm, library_file, output_file)
#         all_results[method_name] = results
#
#     # 4. 生成汇总报告
#     summary_file = "experiment_summary.txt"
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         f.write("实验汇总报告\n")
#         f.write("=" * 60 + "\n\n")
#         f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"总指令数: {len(test_instructions)}\n")
#         f.write(f"每类指令数: 60条\n")
#         f.write(f"指令类别: appearance, disappearance, transformation, bidirectional, composite\n")
#         f.write(f"地物类别数: 25类\n\n")
#
#         f.write("方法比较:\n")
#         f.write("-" * 60 + "\n")
#
#         for method_name, enable_llm, library_file, output_file in methods:
#             f.write(f"\n{method_name}:\n")
#             f.write(f"  启用LLM: {enable_llm}\n")
#             f.write(f"  使用库文件: {library_file}\n")
#             f.write(f"  结果文件: {output_file}\n")
#
#         f.write("\n" + "=" * 60 + "\n")
#
#         # 添加成功率对比表
#         f.write("\n成功率对比:\n")
#         f.write("-" * 60 + "\n")
#         f.write(
#             f"{'方法':<20} {'总成功率':<10} {'appearance':<10} {'disappearance':<12} {'transformation':<12} {'bidirectional':<12} {'composite':<10}\n")
#
#         for method_name, _, _, _ in methods:
#             if method_name in all_results:
#                 results = all_results[method_name]
#                 success_count = sum(1 for r in results if r.get("success", False))
#                 total_success_rate = f"{success_count / len(test_instructions) * 100:.1f}%"
#
#                 # 按类别统计
#                 categories = ["appearance", "disappearance", "transformation", "bidirectional", "composite"]
#                 category_rates = []
#                 for category in categories:
#                     category_instructions = [inst for inst in test_instructions if
#                                              any(cat in inst.lower() for cat in [category])]
#                     if category_instructions:
#                         category_success = sum(1 for r in results
#                                                if r.get("success", False) and
#                                                any(cat in r.get("data", {}).get("original_instruction", "").lower()
#                                                    for cat in [category]))
#                         rate = f"{category_success / len(category_instructions) * 100:.1f}%"
#                     else:
#                         rate = "N/A"
#                     category_rates.append(rate)
#
#                 f.write(
#                     f"{method_name:<20} {total_success_rate:<10} {category_rates[0]:<10} {category_rates[1]:<12} {category_rates[2]:<12} {category_rates[3]:<12} {category_rates[4]:<10}\n")
#
#     print(f"\n实验完成！汇总报告已保存到 {summary_file}")
#     print("结果文件:")
#     for _, _, _, output_file in methods:
#         print(f"  - {output_file}")
#
#
# if __name__ == "__main__":
#     main()