import asyncio
import pytest
from backend.executors.base_executor import TaskConfig, TaskStatus
from backend.executors.python_executor import PythonExecutor
from backend.registry.executor_registry import ExecutorRegistry
from backend.runner.run_from_config import ConfigRunner

@pytest.fixture
async def executor():
    """创建并初始化Python执行器"""
    executor = PythonExecutor()
    await executor.initialize()
    yield executor
    await executor.cleanup()

@pytest.fixture
def registry():
    """获取执行器注册表实例"""
    registry = ExecutorRegistry()
    registry.register_executor("python", PythonExecutor)
    return registry

@pytest.fixture
async def runner():
    """创建配置运行器"""
    runner = ConfigRunner()
    yield runner
    await runner.cleanup()

async def test_basic_python_execution(executor):
    """测试基本的Python函数执行"""
    # 注册测试函数
    def test_func(x: int, y: int) -> int:
        return x + y
    
    executor.register_function("add", test_func)
    
    # 创建任务配置
    config = TaskConfig(
        task_id="test_task_1",
        task_type="add",
        parameters={"x": 1, "y": 2}
    )
    
    # 执行任务
    result = await executor.execute_task(config)
    
    # 验证结果
    assert result.status == TaskStatus.SUCCESS
    assert result.result == 3
    assert result.error is None

async def test_async_python_execution(executor):
    """测试异步Python函数执行"""
    # 注册异步测试函数
    async def async_test_func(delay: float) -> str:
        await asyncio.sleep(delay)
        return "done"
    
    executor.register_function("async_task", async_test_func)
    
    # 创建任务配置
    config = TaskConfig(
        task_id="test_task_2",
        task_type="async_task",
        parameters={"delay": 0.1}
    )
    
    # 执行任务
    result = await executor.execute_task(config)
    
    # 验证结果
    assert result.status == TaskStatus.SUCCESS
    assert result.result == "done"

async def test_error_handling(executor):
    """测试错误处理"""
    def failing_func():
        raise ValueError("Test error")
    
    executor.register_function("fail", failing_func)
    
    config = TaskConfig(
        task_id="test_task_3",
        task_type="fail",
        parameters={}
    )
    
    result = await executor.execute_task(config)
    
    assert result.status == TaskStatus.ERROR
    assert "Test error" in result.error

async def test_workflow_execution(runner):
    """测试工作流执行"""
    # 注册执行器
    runner.registry.register_executor("python", PythonExecutor)
    
    workflow_config = {
        "tasks": [
            {
                "id": "task1",
                "type": "print",
                "parameters": {"text": "Hello, World!"}
            },
            {
                "id": "task2",
                "type": "sum",
                "parameters": {"numbers": [1, 2, 3, 4, 5]}
            }
        ]
    }
    
    results = await runner.run_workflow(workflow_config)
    
    assert len(results) == 2
    assert all(result.status == TaskStatus.SUCCESS for result in results)

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 
