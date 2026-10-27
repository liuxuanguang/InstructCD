import requests
import json
import time


class SiliconFlowAPIClient:
    """
    硅基流动平台API客户端
    支持Pro/MiniMaxAI/MiniMax-M2.5模型的API调用
    """

    def __init__(self, api_key, base_url="https://api.siliconflow.cn/v1"):
        """
        初始化API客户端

        Args:
            api_key: 您的API密钥
            base_url: API基础URL，默认为硅基流动官方API地址
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def call_chat_completion(self, model, messages, temperature=0.7, max_tokens=1000,
                             stream=False, **kwargs):
        """
        调用聊天补全API

        Args:
            model: 模型名称，例如"Pro/MiniMaxAI/MiniMax-M2.5"
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大输出token数
            stream: 是否使用流式输出
            **kwargs: 其他可选参数

        Returns:
            API响应结果
        """
        # 构建API端点
        endpoint = f"{self.base_url}/chat/completions"

        # 构建请求数据
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        # 添加其他可选参数
        data.update(kwargs)

        try:
            # 发送POST请求
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=data,
                stream=stream
            )

            # 检查响应状态
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"错误详情: {e.response.text}")
            return None

    def _handle_stream_response(self, response):
        """
        处理流式响应

        Args:
            response: 原始响应对象

        Returns:
            完整的响应内容
        """
        full_content = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = line_str[6:]  # 移除"data: "前缀
                    if data == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            delta = data_json["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                full_content += content
                                # 可选：实时打印流式输出
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
        return full_content

    def call_embeddings(self, model, input_texts, **kwargs):
        """
        调用嵌入（Embeddings）API

        Args:
            model: 嵌入模型名称
            input_texts: 输入文本列表
            **kwargs: 其他可选参数

        Returns:
            嵌入向量结果
        """
        endpoint = f"{self.base_url}/embeddings"

        data = {
            "model": model,
            "input": input_texts
        }
        data.update(kwargs)

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"嵌入API调用失败: {e}")
            return None

    def list_models(self):
        """
        获取可用模型列表

        Returns:
            可用模型列表
        """
        endpoint = f"{self.base_url}/models"

        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取模型列表失败: {e}")
            return None


# 使用示例
def main():
    # 配置您的API密钥
    API_KEY = "sk-zzjvvunlljmswuvhhmlekbknvwvzqzmxndcthbzddihdsuau"

    # 初始化客户端
    client = SiliconFlowAPIClient(api_key=API_KEY)

    # 示例1: 获取可用模型列表
    print("正在获取可用模型列表...")
    models = client.list_models()
    if models and "data" in models:
        print("可用的模型:")
        for model in models["data"]:
            print(f"  - {model.get('id', 'Unknown')}")

    # 示例2: 调用聊天API
    print("\n正在调用聊天API...")

    # 注意：您指定的模型名称可能需要验证，以下是几种可能的情况
    # 如果模型名称不正确，请根据上一步获取的模型列表进行调整
    model_name = "Pro/MiniMaxAI/MiniMax-M2.5"  # 尝试这个模型名称
    # 或者使用完整的路径: "Pro/MiniMaxAI/MiniMax-M2.5"
    # 实际模型名称可能需要通过list_models()获取

    messages = [
        {
            "role": "user",
            "content": """你是遥感图像处理专家，需要将自然语言变化检测指令转换为结构化格式。

    转换规则：
    1. 输入格式：from [源类别] to [目标类别]
    2. 输出格式为结构化数据：
       - 用户指令：原始输入
       - 变化类型：TYPE_CONVERSION
       - 时相1指令：源类别;目标类别
       - 时相2指令：源类别;目标类别
       - 置信度：0.95
       - 描述：可选的描述文本
       - 标签：[相关标签列表]
       - 来源："core"
       - 变化语义：UNIDIRECTIONAL_CONVERSION
       - 变化方向：FORWARD
       - 转换关系：[源类别→目标类别]

    示例1：
    输入："from bare ground to building"
    输出：
    InstructionExample(
        "from bare ground to building", "TYPE_CONVERSION",
        "bare land;building", "bare land;building", 0.95,
        "", ["land_to_building"], source="core",
        change_semantics="UNIDIRECTIONAL_CONVERSION",
        change_direction="FORWARD",
        transitions=["bare land→building"]
    )

    示例2：
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

    示例3：
    输入："from tree to sports ground"
    输出：
    InstructionExample(
        "from tree to sports ground", "TYPE_CONVERSION",
        "tree;sports ground", "tree;sports ground", 0.93,
        "树木到操场的转换", ["tree_to_sports ground", "deforestation"], source="core",
        change_semantics="UNIDIRECTIONAL_CONVERSION",
        change_direction="FORWARD",
        transitions=["tree→sports ground"]
    )

    请根据以上规则，将"from bare land to sports ground"转换为相同的结构格式，结构要一模一样"""
        }
    ]

    # 非流式调用
    print("非流式调用:")
    response = client.call_chat_completion(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=5000
    )

    if response and "choices" in response:
        answer = response["choices"][0]["message"]["content"]
        print(f"AI回答: {answer}")
        print(f"使用token数: 输入{response.get('usage', {}).get('prompt_tokens', 0)}, "
              f"输出{response.get('usage', {}).get('completion_tokens', 0)}")




if __name__ == "__main__":
    # 运行基本示例
    main()