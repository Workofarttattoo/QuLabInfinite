import pytest
from backend.executors.simulation_executor import SimulationExecutor
from backend.executors.base_executor import TaskConfig, TaskStatus

@pytest.fixture
async def executor():
    """创建并初始化模拟设备执行器"""
    executor = SimulationExecutor()
    await executor.initialize()
    yield executor
    await executor.cleanup()

async def test_device_initialization(executor):
    """测试设备初始化"""
    device1 = executor.get_device_state("device_1")
    assert device1 is not None
    assert device1.status == "idle"
    assert device1.position == {"x": 0, "y": 0, "z": 0}
    
    device2 = executor.get_device_state("device_2")
    assert device2 is not None
    assert device2.status == "idle"
    assert device2.position == {"x": 100, "y": 100, "z": 0}

async def test_device_move_operation(executor):
    """测试设备移动操作"""
    config = TaskConfig(
        task_id="move_task_1",
        task_type="move",
        parameters={
            "device_id": "device_1",
            "operation": "move",
            "params": {
                "position": {"x": 50, "y": 50, "z": 10}
            }
        }
    )
    
    result = await executor.execute_task(config)
    
    assert result.status == TaskStatus.SUCCESS
    assert "new_position" in result.result
    assert result.result["new_position"]["x"] == 50
    assert len(result.logs) > 0

async def test_device_measure_operation(executor):
    """测试设备测量操作"""
    config = TaskConfig(
        task_id="measure_task_1",
        task_type="measure",
        parameters={
            "device_id": "device_1",
            "operation": "measure"
        }
    )
    
    result = await executor.execute_task(config)
    
    assert result.status == TaskStatus.SUCCESS
    assert "temperature" in result.result
    assert 20 <= result.result["temperature"] <= 30

async def test_device_reset_operation(executor):
    """测试设备重置操作"""
    # 先移动设备
    move_config = TaskConfig(
        task_id="move_task_2",
        task_type="move",
        parameters={
            "device_id": "device_1",
            "operation": "move",
            "params": {
                "position": {"x": 75, "y": 75, "z": 20}
            }
        }
    )
    await executor.execute_task(move_config)
    
    # 然后重置
    reset_config = TaskConfig(
        task_id="reset_task_1",
        task_type="reset",
        parameters={
            "device_id": "device_1",
            "operation": "reset"
        }
    )
    
    result = await executor.execute_task(reset_config)
    
    assert result.status == TaskStatus.SUCCESS
    device = executor.get_device_state("device_1")
    assert device.position == {"x": 0, "y": 0, "z": 0}
    assert device.temperature == 25.0

async def test_error_handling(executor):
    """测试错误处理"""
    # 测试未知设备
    config1 = TaskConfig(
        task_id="error_task_1",
        task_type="move",
        parameters={
            "device_id": "unknown_device",
            "operation": "move"
        }
    )
    
    result1 = await executor.execute_task(config1)
    assert result1.status == TaskStatus.ERROR
    assert "Unknown device" in result1.error
    
    # 测试缺少必要参数
    config2 = TaskConfig(
        task_id="error_task_2",
        task_type="move",
        parameters={
            "device_id": "device_1"
        }
    )
    
    result2 = await executor.execute_task(config2)
    assert result2.status == TaskStatus.ERROR
    assert "operation is required" in result2.error

async def test_device_busy_state(executor):
    """测试设备忙状态"""
    device = executor.get_device_state("device_1")
    device.status = "busy"
    
    config = TaskConfig(
        task_id="busy_task_1",
        task_type="move",
        parameters={
            "device_id": "device_1",
            "operation": "move",
            "params": {
                "position": {"x": 0, "y": 0, "z": 0}
            }
        }
    )
    
    result = await executor.execute_task(config)
    assert result.status == TaskStatus.ERROR
    assert "busy" in result.error.lower()

async def test_task_logs(executor):
    """测试任务日志记录"""
    config = TaskConfig(
        task_id="log_test_task",
        task_type="measure",
        parameters={
            "device_id": "device_1",
            "operation": "measure"
        }
    )
    
    result = await executor.execute_task(config)
    
    assert result.logs is not None
    assert len(result.logs) >= 2  # 至少应该有开始和结束的日志
    assert any("Starting task" in log for log in result.logs)
    assert any("completed successfully" in log for log in result.logs) 
