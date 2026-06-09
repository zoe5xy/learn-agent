# tools/__init__.py
# 相对导入：从同目录导入各个工具函数
from .weather_tool import get_weather
from .attraction_tool import get_attraction

# 工具映射字典
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}