# Chapter 1 使用说明

本目录实现了一个简单的智能旅行助手示例。程序会调用兼容 OpenAI 接口的大语言模型进行推理，并根据模型给出的 `Action` 调用本地工具：先查询城市天气，再结合天气搜索合适的旅游景点推荐。

## 环境要求

- Python 3.9 或更高版本
- 可访问互联网的运行环境
- 一个兼容 OpenAI Chat Completions API 的大模型服务
- Tavily API Key，用于景点搜索

建议在虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openai python-dotenv requests tavily-python
```

## API 与环境变量

程序通过 `.env` 文件读取所需配置。请在项目根目录或 `Chapter_1` 目录下创建 `.env` 文件，并填写以下内容：

```env
LLM_API_KEY=你的大模型服务 API Key
LLM_BASE_URL=你的大模型服务 Base URL
LLM_MODEL=你要使用的模型名称
TAVILY_API_KEY=你的 Tavily API Key
```

变量说明：

| 变量名 | 作用 |
| --- | --- |
| `LLM_API_KEY` | 调用大语言模型服务所需的 API Key |
| `LLM_BASE_URL` | 兼容 OpenAI 接口的大模型服务地址 |
| `LLM_MODEL` | 使用的模型 ID |
| `TAVILY_API_KEY` | Tavily Search API Key，用于搜索景点推荐 |

本项目中使用到的外部 API：

| API | 文件 | 作用 |
| --- | --- | --- |
| OpenAI-compatible Chat Completions API | `llm.py` | 调用大语言模型，生成 Thought 和 Action |
| wttr.in | `tools/weather_tool.py` | 查询指定城市的实时天气 |
| Tavily Search API | `tools/attraction_tool.py` | 根据城市和天气搜索旅游景点推荐 |

## 文件夹结构

```text
Chapter_1/
├── execute.py
├── llm.py
├── system_prompt.py
└── tools/
    ├── __init__.py
    ├── attraction_tool.py
    └── weather_tool.py
```

文件说明：

| 文件 | 说明 |
| --- | --- |
| `execute.py` | 主程序入口，负责加载环境变量、初始化 LLM、循环执行推理和工具调用 |
| `llm.py` | 封装兼容 OpenAI 接口的大模型客户端 |
| `system_prompt.py` | 定义 Agent 的系统提示词、可用工具和输出格式 |
| `tools/__init__.py` | 注册可供 Agent 调用的工具函数 |
| `tools/weather_tool.py` | 提供 `get_weather(city)` 天气查询工具 |
| `tools/attraction_tool.py` | 提供 `get_attraction(city, weather)` 景点推荐搜索工具 |

## 使用方法

1. 安装依赖。

```powershell
pip install openai python-dotenv requests tavily-python
```

2. 配置 `.env` 文件。

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-api-base-url
LLM_MODEL=your_model_name
TAVILY_API_KEY=your_tavily_api_key
```

3. 运行主程序。

在项目根目录执行：

```powershell
python Chapter_1\execute.py
```

或者进入 `Chapter_1` 后执行：

```powershell
cd Chapter_1
python execute.py
```

程序默认的问题写在 `execute.py` 中：

```python
user_prompt = "Hello, please help me check today's weather in Beijing, and then recommend a suitable tourist attraction based on the weather."
```

如需更换城市或任务，可以直接修改这行文本。

## 运行流程

程序整体流程如下：

1. `execute.py` 加载 `.env` 中的模型和 API 配置。
2. 初始化 `OpenAICompatibleClient`。
3. 将用户问题和 `AGENT_SYSTEM_PROMPT` 一起发送给大语言模型。
4. 模型按固定格式输出：

```text
Thought: ...
Action: ...
```

5. 程序解析 `Action`：

- 如果是 `get_weather(city="...")`，调用天气工具。
- 如果是 `get_attraction(city="...", weather="...")`，调用 Tavily 搜索工具。
- 如果是 `Finish[...]`，结束任务并输出最终答案。

6. 工具返回结果会作为 `Observation` 追加进上下文，继续下一轮推理。

## 可用工具

### get_weather

```python
get_weather(city: str) -> str
```

查询指定城市当前天气，返回天气描述和摄氏温度。

示例：

```text
get_weather(city="Beijing")
```

### get_attraction

```python
get_attraction(city: str, weather: str) -> str
```

根据城市和天气情况搜索适合游览的景点推荐。

示例：

```text
get_attraction(city="Beijing", weather="Sunny, temperature 25 degrees Celsius")
```

## 注意事项

- `execute.py` 中主循环最多运行 5 轮，避免模型无限调用工具。
- 模型输出必须包含 `Action:`，否则程序会提示没有找到动作。
- 工具调用参数目前通过正则表达式解析，建议保持 `arg_name="arg_value"` 的格式。
- 天气查询依赖 `wttr.in`，景点推荐依赖 Tavily，运行时需要网络连接。
- 如果 Tavily Key 未配置，景点工具会返回 `TAVILY_API_KEY environment variable not configured` 错误。
