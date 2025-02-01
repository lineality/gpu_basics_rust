// use std::fs;
// use std::path::Path;

// #[derive(Debug)]
// struct PCIDevice {
//     vendor_id: String,
//     device_id: String,
//     class: String,
//     path: String,
// }

// fn enumerate_pci_devices() -> Vec<PCIDevice> {
//     let mut devices = Vec::new();
//     let pci_path = Path::new("/sys/bus/pci/devices");
    
//     if let Ok(entries) = fs::read_dir(pci_path) {
//         for entry in entries.flatten() {
//             let path = entry.path();
            
//             // Read device information
//             if let (Ok(vendor), Ok(device), Ok(class)) = (
//                 fs::read_to_string(path.join("vendor")),
//                 fs::read_to_string(path.join("device")),
//                 fs::read_to_string(path.join("class"))
//             ) {
//                 devices.push(PCIDevice {
//                     vendor_id: vendor.trim().to_string(),
//                     device_id: device.trim().to_string(),
//                     class: class.trim().to_string(),
//                     path: path.to_string_lossy().to_string(),
//                 });
//             }
//         }
//     }
//     devices
// }

// fn find_gpu_devices(devices: &[PCIDevice]) -> Vec<&PCIDevice> {
//     // GPU devices typically have a class code starting with 0x03
//     // Intel's vendor ID is 0x8086
//     devices.iter()
//         .filter(|dev| {
//             dev.class.starts_with("0x03") && // Display controller class
//             dev.vendor_id == "0x8086"        // Intel vendor ID
//         })
//         .collect()
// }

// fn main() {
//     let devices = enumerate_pci_devices();
//     println!("All PCI devices found:");
//     for device in &devices {
//         println!("{:#?}", device);
//     }
    
//     println!("\nGPU devices:");
//     let gpu_devices = find_gpu_devices(&devices);
//     for gpu in gpu_devices {
//         println!("{:#?}", gpu);
//     }
// }

/// Represents a PCI (Peripheral Component Interconnect) device with its identifying information
#[derive(Debug)]
struct PCIDeviceIdentification {
    pci_vendor_id: String,
    pci_device_id: String,
    pci_class_code: String,
    pci_device_path: String,
}

/// Attempts to find Intel GPU devices in the system by reading PCI device information
/// from the Linux sysfs interface at /sys/bus/pci/devices/
/// 
/// Returns a Result containing a vector of PCIDeviceIdentification for found Intel GPUs,
/// or an error if the system cannot be queried.
/// 
/// # Errors
/// - Returns io::Error if the PCI device directory cannot be read
/// - Returns io::Error if required device files cannot be read
fn find_intel_gpu_devices() -> Result<Vec<PCIDeviceIdentification>, std::io::Error> {
    let pci_devices_path = Path::new("/sys/bus/pci/devices");
    let mut intel_gpu_devices = Vec::new();

    let dir_entries = std::fs::read_dir(pci_devices_path)?;

    for dir_entry in dir_entries {
        let entry = dir_entry?;
        let device_path = entry.path();

        // Read the three identifying files for each device
        let vendor_id = std::fs::read_to_string(device_path.join("vendor"))?;
        let device_id = std::fs::read_to_string(device_path.join("device"))?;
        let class_code = std::fs::read_to_string(device_path.join("class"))?;

        // Intel vendor ID is 0x8086 and display controller class starts with 0x03
        if vendor_id.trim() == "0x8086" && class_code.trim().starts_with("0x03") {
            intel_gpu_devices.push(PCIDeviceIdentification {
                pci_vendor_id: vendor_id.trim().to_string(),
                pci_device_id: device_id.trim().to_string(),
                pci_class_code: class_code.trim().to_string(),
                pci_device_path: device_path.to_string_lossy().to_string(),
            });
        }
    }

    Ok(intel_gpu_devices)
}

/// Constants for PCI device identification
const PCI_VENDOR_ID_INTEL: &str = "0x8086";
const PCI_CLASS_DISPLAY_CONTROLLER: &str = "0x03";

/// Represents memory-mapped regions of a PCI device
#[derive(Debug)]
struct PCIMemoryRegion {
    /// Physical base address of the memory region
    base_address: u64,
    /// Size of the memory region in bytes
    size: u64,
    /// Type of memory region (e.g., memory-mapped or I/O-mapped)
    region_type: PCIMemoryRegionType,
}

/// Specifies the type of PCI memory region
#[derive(Debug)]
enum PCIMemoryRegionType {
    /// Memory-mapped I/O region
    MemoryMapped,
    /// I/O-mapped region
    IoMapped,
}

impl PCIDeviceIdentification {
    /// Attempts to read and parse the memory regions (Base Address Registers - BARs)
    /// for this PCI device.
    /// 
    /// # Returns
    /// - Ok(Vec<PCIMemoryRegion>) containing the parsed memory regions
    /// - Err if reading or parsing fails
    /// 
    /// # Notes
    /// - BAR registers are used to map device memory into system address space
    /// - Each BAR specifies a memory region's base address, size, and type
    fn read_memory_regions(&self) -> std::io::Result<Vec<PCIMemoryRegion>> {
        let path = Path::new(&self.pci_device_path);
        let mut regions = Vec::new();

        // Read the 'resource' file which contains BAR information
        let resource_content = std::fs::read_to_string(path.join("resource"))?;

        // Parse each line of the resource file
        for (index, line) in resource_content.lines().enumerate() {
            if let Some(region) = parse_resource_line(line, index) {
                regions.push(region);
            }
        }

        Ok(regions)
    }
}

/// Parses a single line from the PCI resource file into a PCIMemoryRegion
/// 
/// # Arguments
/// * `line` - A line from the PCI resource file containing address range
/// * `index` - The BAR index (0-5) being parsed
/// 
/// # Returns
/// * Some(PCIMemoryRegion) if parsing succeeds
/// * None if the line represents an unused or invalid region
fn parse_resource_line(line: &str, index: usize) -> Option<PCIMemoryRegion> {
    // Format is typically: "0x00000000fed00000 0x00000000fed03fff 0x0000000000140204"
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }

    // Parse base address and end address
    let base_addr = u64::from_str_radix(&parts[0].trim_start_matches("0x"), 16).ok()?;
    let end_addr = u64::from_str_radix(&parts[1].trim_start_matches("0x"), 16).ok()?;
    let flags = u64::from_str_radix(&parts[2].trim_start_matches("0x"), 16).ok()?;

    // If base_addr is 0, this BAR is unused
    if base_addr == 0 {
        return None;
    }

    Some(PCIMemoryRegion {
        base_address: base_addr,
        size: end_addr - base_addr + 1,
        region_type: if flags & 0x1 == 0 {
            PCIMemoryRegionType::MemoryMapped
        } else {
            PCIMemoryRegionType::IoMapped
        },
    })
}

use std::path::Path;

/// Constants for PCI Configuration Space access
const PCI_CONFIG_SPACE_SIZE: usize = 256;  // Standard PCI Configuration Space is 256 bytes
const PCI_COMMAND_REGISTER_OFFSET: usize = 0x04;  // Command register location in config space
const PCI_BAR0_OFFSET: usize = 0x10;  // First Base Address Register location

/// Represents the PCI Configuration Space header structure
/// This is the standardized layout for PCI devices
#[derive(Debug)]
struct PCIConfigurationSpace {
    /// Raw configuration space data
    raw_data: [u8; PCI_CONFIG_SPACE_SIZE],
}

impl PCIConfigurationSpace {
    /// Safely reads the PCI Configuration Space for a device
    /// 
    /// # Arguments
    /// * `device_path` - Path to the PCI device in sysfs
    /// 
    /// # Returns
    /// * Ok(PCIConfigurationSpace) if read successfully
    /// * Err if reading fails or data is invalid
    /// 
    /// # Safety
    /// This function performs direct hardware access through sysfs.
    /// It requires appropriate permissions to read the config file.
    fn read_from_device(device_path: &str) -> std::io::Result<Self> {
        let config_path = Path::new(device_path).join("config");
        let mut config_data = [0u8; PCI_CONFIG_SPACE_SIZE];

        // Read the configuration space file
        let bytes_read = std::fs::read(&config_path)?;
        
        if bytes_read.len() < PCI_CONFIG_SPACE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Configuration space too small: got {} bytes, expected {}",
                    bytes_read.len(),
                    PCI_CONFIG_SPACE_SIZE
                )
            ));
        }

        config_data.copy_from_slice(&bytes_read[..PCI_CONFIG_SPACE_SIZE]);
        Ok(PCIConfigurationSpace { raw_data: config_data })
    }

    /// Retrieves the command register value from configuration space
    /// The command register controls fundamental features of the PCI device
    fn get_command_register(&self) -> u16 {
        u16::from_le_bytes([
            self.raw_data[PCI_COMMAND_REGISTER_OFFSET],
            self.raw_data[PCI_COMMAND_REGISTER_OFFSET + 1]
        ])
    }

    /// Checks if memory access is enabled for this device
    fn is_memory_access_enabled(&self) -> bool {
        const MEMORY_SPACE_ENABLED: u16 = 0x0002;
        (self.get_command_register() & MEMORY_SPACE_ENABLED) != 0
    }

    /// Gets the BAR0 (Base Address Register 0) value
    /// BAR0 typically contains the main memory-mapped register space
    fn get_bar0(&self) -> u32 {
        u32::from_le_bytes([
            self.raw_data[PCI_BAR0_OFFSET],
            self.raw_data[PCI_BAR0_OFFSET + 1],
            self.raw_data[PCI_BAR0_OFFSET + 2],
            self.raw_data[PCI_BAR0_OFFSET + 3]
        ])
    }
}

/// Reads and validates PCI configuration space for an Intel GPU
/// 
/// # Arguments
/// * `device_identification` - The PCIDeviceIdentification structure for the GPU
/// 
/// # Returns
/// * Ok(String) with configuration information if successful
/// * Err if reading or validation fails
fn read_gpu_configuration(
    device_identification: &PCIDeviceIdentification
) -> std::io::Result<String> {
    let config_space = PCIConfigurationSpace::read_from_device(&device_identification.pci_device_path)?;
    
    let mut config_info = String::new();
    config_info.push_str(&format!("Device Configuration Information:\n"));
    config_info.push_str(&format!("Memory Access Enabled: {}\n", 
        config_space.is_memory_access_enabled()));
    config_info.push_str(&format!("BAR0 Address: 0x{:08x}\n", 
        config_space.get_bar0()));
    
    Ok(config_info)
}


fn main() {
    // match find_intel_gpu_devices() {
    //     Ok(gpu_devices) => {
    //         println!("Found {} Intel GPU device(s):", gpu_devices.len());
    //         for device in gpu_devices {
    //             println!("\nDevice Information:");
    //             println!("PCI Vendor ID: {}", device.pci_vendor_id);
    //             println!("PCI Device ID: {}", device.pci_device_id);
    //             println!("PCI Class Code: {}", device.pci_class_code);
    //             println!("Device Path: {}", device.pci_device_path);
    //         }
    //     }
    //     Err(error) => {
    //         eprintln!("Error finding GPU devices: {}", error);
    //     }
    // }
    
    
    match find_intel_gpu_devices() {
        Ok(gpu_devices) => {
            println!("Found {} Intel GPU device(s):", gpu_devices.len());
            for device in gpu_devices {
                println!("\nDevice Information:");
                println!("PCI Vendor ID: {}", device.pci_vendor_id);
                println!("PCI Device ID: {}", device.pci_device_id);
                println!("PCI Class Code: {}", device.pci_class_code);
                println!("Device Path: {}", device.pci_device_path);
                
                match read_gpu_configuration(&device) {
                    Ok(config_info) => println!("\n{}", config_info),
                    Err(e) => eprintln!("Error reading configuration: {}", e),
                }
            }
        }
        Err(error) => {
            eprintln!("Error finding GPU devices: {}", error);
        }
    }
}