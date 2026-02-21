import pytest
from datetime import datetime
from backend.executors.sdl_catalyst_executor import SDLCatalystExecutor, SDLWorkflowState
from backend.executors.base_executor import TaskConfig, TaskStatus

@pytest.fixture
async def executor():
    """创建并初始化 SDL Catalyst 执行器"""
    executor = SDLCatalystExecutor()
    await executor.initialize()
    yield executor
    await executor.cleanup()

async def test_workflow_execution(executor):
    """测试工作流执行"""
    config = TaskConfig(
        task_id="test_workflow_1",
        task_type="sdl_analysis",
        parameters={
            "workflow_config": {
                "input_data": "sample_data.csv",
                "analysis_type": "performance",
                "parameters": {
                    "threshold": 0.8,
                    "max_iterations": 100
                }
            }
        }
    )
    
    result = await executor.execute_task(config)
    
    assert result.status == TaskStatus.SUCCESS
    assert "workflow_id" in result.result
    assert "workflow_state" in result.result
    assert "analysis_results" in result.result
    assert result.result["analysis_results"]["metrics"]["accuracy"] > 0.9
    assert len(result.logs) > 0

async def test_workflow_progress_tracking(executor):
    """测试工作流进度追踪"""
    config = TaskConfig(
        task_id="test_workflow_2",
        task_type="sdl_analysis",
        parameters={
            "workflow_config": {
                "input_data": "test_data.csv",
                "analysis_type": "basic"
            }
        }
    )
    
    # 启动工作流
    task_result = await executor.execute_task(config)
    workflow_id = task_result.result["workflow_id"]
    
    # 获取工作流状态
    workflow_state = executor.get_workflow_state(workflow_id)
    assert workflow_state is not None
    assert workflow_state.status == "completed"
    assert workflow_state.progress == 100.0
    assert workflow_state.start_time is not None
    assert workflow_state.end_time is not None
    assert workflow_state.error is None

async def test_error_handling(executor):
    """测试错误处理"""
    # 测试缺少必要参数
    config = TaskConfig(
        task_id="error_workflow_1",
        task_type="sdl_analysis",
        parameters={}
    )
    
    result = await executor.execute_task(config)
    assert result.status == TaskStatus.ERROR
    assert "workflow_config is required" in result.error
    assert len(result.logs) > 0

async def test_task_status_tracking(executor):
    """测试任务状态追踪"""
    config = TaskConfig(
        task_id="status_test_1",
        task_type="sdl_analysis",
        parameters={
            "workflow_config": {
                "input_data": "status_test.csv",
                "analysis_type": "quick"
            }
        }
    )
    
    # 检查初始状态
    initial_status = await executor.get_status("status_test_1")
    assert initial_status == TaskStatus.PENDING
    
    # 执行任务
    await executor.execute_task(config)
    
    # 检查最终状态
    final_status = await executor.get_status("status_test_1")
    assert final_status == TaskStatus.SUCCESS

async def test_workflow_results_format(executor):
    """测试工作流结果格式"""
    config = TaskConfig(
        task_id="format_test_1",
        task_type="sdl_analysis",
        parameters={
            "workflow_config": {
                "input_data": "format_test.csv",
                "analysis_type": "detailed"
            }
        }
    )
    
    result = await executor.execute_task(config)
    
    # 验证结果格式
    assert "analysis_results" in result.result
    analysis_results = result.result["analysis_results"]
    assert "status" in analysis_results
    assert "metrics" in analysis_results
    
    # 验证资源使用信息
    assert "resource_usage" in result.result
    resource_usage = result.result["resource_usage"]
    assert "cpu" in resource_usage
    assert "memory" in resource_usage
    
    # 验证执行时间
    assert "execution_time" in result.result 
