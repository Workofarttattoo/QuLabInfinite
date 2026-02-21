use tokio;
use serde_json::json;
use device_executor::{
    DeviceExecutor,
    devices::cva::{CVADevice, CVAConfig},
    DeviceConfig,
    events::EventSubscriber,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建 CVA 设备
    let device_id = "test_cva_1".to_string();
    let mut device = CVADevice::new(device_id.clone(), CVAConfig::default());
    
    // 初始化设备
    device.initialize().await?;
    println!("设备初始化完成");

    // 执行 CVA 测量
    let params = json!({
        "start_voltage": -0.5,
        "end_voltage": 0.5,
        "scan_rate": 0.1,
        "cycles": 1
    });

    println!("开始 CVA 测量...");
    let result = device.execute("measure_cv", params.as_object().unwrap().clone()).await?;
    
    println!("CVA 测量结果:");
    println!("{}", serde_json::to_string_pretty(&result)?);

    // 获取并打印设备状态
    let status = device.get_status();
    println!("设备状态: {:?}", status);

    Ok(())
} 
