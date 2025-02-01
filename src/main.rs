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

/*
What kind of .csv tabular data (as would often be put into a pandas df) could be processed with a gpu?

yes, strings are not ideal, but NLP often involves strings (to put it lightly)

for example, can a 'string' table be converted into a one-hot table and be processed that way? another emphasis here is on processing batches or rows of large files without loading the whole file. 

A GPU is most suitable for data that:
1. Can be processed in parallel
2. Has high compute-to-transfer ratio (the computation time justifies the cost of moving data to/from GPU)
3. Involves uniform operations across many rows/columns
*/


use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use std::collections::HashMap;

/// Represents a batch of text data that has been vectorized and prepared for GPU processing.
/// This structure serves as the intermediate format between raw text data and GPU computation.
#[derive(Debug)]
struct TextBatch {
    /// The vectorized data ready for GPU processing.
    /// Each inner Vec<f32> represents one row of transformed text data.
    /// All inner vectors must have the same length (vector_size).
    vectors: Vec<Vec<f32>>,

    /// Number of rows in this batch.
    /// This must match vectors.len()
    row_count: usize,

    /// Size of each vector (number of elements per row).
    /// This must match the length of each inner vector in vectors.
    /// For one-hot encoding, this equals the vocabulary size.
    /// For token encoding, this equals max_seq_length.
    vector_size: usize,
}

/// Configuration for text processing
/// Configuration settings for processing text data into GPU-compatible format
/// 
/// This struct defines how text data should be processed, including batch sizes,
/// CSV formatting details, and vocabulary constraints. It is used to control
/// how text data is converted into numerical vectors suitable for GPU processing.
/// 
/// # Example
/// ```
/// let config = TextProcessingConfig {
///     batch_size: 1000,
///     has_header: true,
///     columns_to_process: vec![0, 1],  // Process first two columns
///     max_vocab_size: 10000,
/// };
/// ```
/// 
/// # Notes
/// - Batch size should be chosen based on available GPU memory and processing requirements
/// - Max vocabulary size limits memory usage but may cause rare terms to be ignored
/// - Column indices are 0-based
#[derive(Debug)]
struct TextProcessingConfig {
    /// Number of rows to process in each batch
    /// 
    /// This controls the memory usage and processing granularity.
    /// Larger batches may be more efficient but require more memory.
    /// Should be tuned based on:
    /// - Available system memory
    /// - GPU memory capacity
    /// - Processing time requirements
    pub batch_size: usize,

    /// Indicates whether the input file has a header row
    /// 
    /// If true, the first row of the file will be skipped during processing
    /// as it is assumed to contain column names rather than data.
    pub has_header: bool,

    /// Indices of columns to process from the input file
    /// 
    /// Zero-based indices of columns that should be processed.
    /// Other columns will be ignored.
    /// 
    /// # Example
    /// - vec![0, 2] will process the first and third columns
    /// - Empty vec![] will result in no processing
    /// 
    /// # Panics
    /// Will panic if any index is out of bounds for the input file
    pub columns_to_process: Vec<usize>,

    /// Maximum number of unique terms to include in vocabulary
    /// 
    /// Limits the size of the vocabulary to control memory usage
    /// and vector dimensionality. Terms beyond this limit will
    /// be ignored (effectively treated as out-of-vocabulary).
    /// 
    /// # Notes
    /// - Affects memory usage and processing time
    /// - Should be set based on dataset characteristics
    /// - Too small: may lose important information
    /// - Too large: may include noise and increase memory usage
    pub max_vocab_size: usize,
}

/// Processes CSV files in batches for GPU computation
/// Processes large CSV files in batches, preparing text data for GPU computation.
/// This processor is designed to:
/// 1. Handle large files that don't fit in memory
/// 2. Convert text data into numerical formats suitable for GPU processing
/// 3. Maintain consistent vocabularies across batches
/// 4. Process multiple columns independently
///
/// # Example usage:
/// ```
/// let config = TextProcessingConfig {
///     batch_size: 1000,
///     has_header: true,
///     columns_to_process: vec![0, 1],
///     max_vocab_size: 10000,
/// };
///
/// let mut processor = CSVBatchProcessor::new(
///     PathBuf::from("data/large_file.csv"),
///     config,
/// )?;
///
/// processor.build_vocabularies()?;
/// while let Some(batch) = processor.process_next_batch()? {
///     // Process batch.vectors on GPU
/// }
/// ```
struct CSVBatchProcessor {
    /// Path to the CSV file being processed
    file_path: PathBuf,

    /// Processing configuration parameters
    config: TextProcessingConfig,

    /// Vocabulary mappings for each processed column.
    /// Index in vector corresponds to index in columns_to_process.
    /// Each HashMap maps terms to their numerical indices.
    column_vocabularies: Vec<HashMap<String, usize>>,

    /// Current byte position in the input file.
    /// Used to track progress and enable batch processing.
    current_position: u64,
}

impl CSVBatchProcessor {
    /// Creates a new CSV processor with specified configuration
    /// Creates a new CSV processor with specified configuration.
    /// 
    /// # Arguments
    /// * `file_path` - Path to the CSV file to process
    /// * `config` - Configuration parameters for processing
    /// 
    /// # Returns
    /// * `Ok(CSVBatchProcessor)` if file exists and configuration is valid
    /// * `Err` if file doesn't exist or other IO error occurs
    /// 
    /// # Examples
    /// ```
    /// let config = TextProcessingConfig {
    ///     batch_size: 1000,
    ///     has_header: true,
    ///     columns_to_process: vec![0, 1],
    ///     max_vocab_size: 10000,
    /// };
    /// 
    /// let processor = CSVBatchProcessor::new(
    ///     PathBuf::from("data/example.csv"),
    ///     config,
    /// )?;
    /// ```
    fn new(
        file_path: PathBuf,
        config: TextProcessingConfig,
    ) -> io::Result<Self> {
        if !file_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CSV file not found"
            ));
        }

        // Initialize vocabularies for each column
        let column_vocabularies = vec![HashMap::new(); config.columns_to_process.len()];

        Ok(CSVBatchProcessor {
            file_path,
            config,
            column_vocabularies,
            current_position: 0,
        })
    }
    
    
    /// Builds vocabularies by scanning through the CSV file and creating mappings
    /// of unique terms to numeric indices for each processed column.
    /// 
    /// # Process
    /// 1. Reads the CSV file line by line
    /// 2. For each specified column, maintains a vocabulary of unique terms
    /// 3. Maps each unique term to a numeric index (0 to vocabulary_size - 1)
    /// 4. Respects max_vocab_size limit from configuration
    /// 
    /// # Arguments
    /// * `&mut self` - Mutable reference to modify internal vocabulary mappings
    /// 
    /// # Returns
    /// * `io::Result<()>` - Ok(()) if successful, Err if file operations fail
    /// 
    /// # Errors
    /// - Returns io::Error if file cannot be opened
    /// - Returns io::Error if file reading fails
    /// - Returns io::Error if line parsing fails
    /// 
    /// # Notes
    /// - Skips header row if config.has_header is true
    /// - Stops adding new terms to a column's vocabulary once max_vocab_size is reached
    /// - Terms encountered after max_vocab_size is reached will be ignored
    /// - Column indices that exceed the number of fields in a row are skipped
    /// Builds vocabulary from the first pass through the file
    fn build_vocabularies(&mut self) -> io::Result<()> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let mut first_row = true;

        for line in reader.lines() {
            let line = line?;
            
            // Skip header if configured
            if first_row && self.config.has_header {
                first_row = false;
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();
            
            // Process each specified column
            for (vocab_idx, &col_idx) in self.config.columns_to_process.iter().enumerate() {
                if col_idx < fields.len() {
                    let vocabulary = &mut self.column_vocabularies[vocab_idx];
                    // Get the current size before inserting
                    let current_size = vocabulary.len();
                    if current_size < self.config.max_vocab_size {
                        if !vocabulary.contains_key(fields[col_idx]) {
                            vocabulary.insert(
                                fields[col_idx].to_string(),
                                current_size
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // /// Builds vocabulary from the first pass through the file
    // fn build_vocabularies(&mut self) -> io::Result<()> {
    //     let file = File::open(&self.file_path)?;
    //     let reader = BufReader::new(file);
    //     let mut first_row = true;

    //     for line in reader.lines() {
    //         let line = line?;
            
    //         // Skip header if configured
    //         if first_row && self.config.has_header {
    //             first_row = false;
    //             continue;
    //         }

    //         let fields: Vec<&str> = line.split(',').collect();
            
    //         // Process each specified column
    //         for (vocab_idx, &col_idx) in self.config.columns_to_process.iter().enumerate() {
    //             if col_idx < fields.len() {
    //                 let vocabulary = &mut self.column_vocabularies[vocab_idx];
    //                 if vocabulary.len() < self.config.max_vocab_size {
    //                     vocabulary.entry(fields[col_idx].to_string())
    //                         .or_insert(vocabulary.len());
    //                 }
    //             }
    //         }
    //     }

    //     Ok(())
    // }

    
    
    /// Processes the next batch of rows from the input file.
    /// 
    /// # Returns
    /// * `Ok(Some(TextBatch))` if a batch was successfully read and processed
    /// * `Ok(None)` if end of file was reached
    /// * `Err` if IO error occurs during reading
    /// 
    /// # Notes
    /// * Batch size may be smaller than configured batch_size at end of file
    /// * Vectorization uses vocabularies built during build_vocabularies()
    /// * Each row is converted to fixed-size numerical vectors
    fn process_next_batch(&mut self) -> io::Result<Option<TextBatch>> {
        let file = File::open(&self.file_path)?;
        let mut reader = BufReader::new(file);
        let mut vectors = Vec::new();
        let mut rows_processed = 0;
        
        // Seek to current position
        use std::io::Seek;
        reader.seek(io::SeekFrom::Start(self.current_position))?;

        for line in reader.lines() {
            if rows_processed >= self.config.batch_size {
                break;
            }

            let line = line?;
            let fields: Vec<&str> = line.split(',').collect();
            
            // Process each column in this row
            for (vocab_idx, &col_idx) in self.config.columns_to_process.iter().enumerate() {
                if col_idx < fields.len() {
                    let vocabulary = &self.column_vocabularies[vocab_idx];
                    let mut vector = vec![0.0; vocabulary.len()];
                    
                    if let Some(&index) = vocabulary.get(fields[col_idx]) {
                        vector[index] = 1.0;
                    }
                    
                    vectors.push(vector);
                }
            }

            rows_processed += 1;
        }

        if vectors.is_empty() {
            Ok(None)
        } else {
            let vector_size = self.column_vocabularies[0].len();
            Ok(Some(TextBatch {
                vectors,
                row_count: rows_processed,
                vector_size,
            }))
        }
    }
}


/// Example of time-series data suitable for GPU processing
struct TimeSeriesData {
    /// Timestamps for each data point
    timestamps: Vec<i64>,
    /// Multiple channels of sensor readings
    sensor_readings: Vec<Vec<f32>>,
    /// Number of data points
    length: usize,
}

impl TimeSeriesData {
    /// Calculate moving averages for all channels in parallel
    fn calculate_moving_average(&self, window_size: usize) -> Result<Vec<Vec<f32>>, &'static str> {
        if window_size > self.length {
            return Err("Window size larger than data length");
        }
        // This would be GPU-accelerated
        todo!("Implement GPU moving average calculation");
    }
}


/// Large matrix operations suitable for GPU
struct LargeMatrix {
    /// Matrix data in row-major order
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl LargeMatrix {
    /// Matrix multiplication is highly parallel and GPU-suitable
    fn multiply(&self, other: &LargeMatrix) -> Result<LargeMatrix, &'static str> {
        if self.cols != other.rows {
            return Err("Invalid matrix dimensions for multiplication");
        }
        // This would be GPU-accelerated
        todo!("Implement GPU matrix multiplication");
    }
}


/// Parallel data transformations suitable for GPU
struct ParallelTransform {
    /// Multiple columns of numerical data
    columns: Vec<Vec<f32>>,
    /// Number of rows
    row_count: usize,
}

impl ParallelTransform {
    /// Apply same transformation to all elements
    fn transform_all(&self, operation: MathType) -> Result<Vec<Vec<f32>>, &'static str> {
        // This would be GPU-accelerated
        todo!("Implement GPU parallel transformation");
    }
}


/// Types of tabular data operations well-suited for GPU processing
enum GPUTableOperation {
    /// Column-wise mathematical operations
    /// Example: multiply every value in a column by a constant
    ColumnMath {
        operation_type: MathType,
        column_data: Vec<f32>,
    },
    
    /// Aggregations across rows or columns
    /// Example: sum, mean, min, max of large datasets
    Aggregation {
        aggregation_type: AggregationType,
        data: Vec<f32>,
    },
    
    /// Matrix operations when table represents a matrix
    /// Example: matrix multiplication, transpose
    MatrixOperation {
        operation_type: MatrixOpType,
        matrix_data: Vec<f32>,
        dimensions: (usize, usize),  // rows, columns
    },
    
    /// Filtering or transformation of many rows in parallel
    /// Example: where column_value > threshold
    ParallelFilter {
        filter_condition: FilterType,
        column_data: Vec<f32>,
    },
}

/// Basic mathematical operations suitable for GPU
enum MathType {
    Add(f32),
    Multiply(f32),
    Power(f32),
    Log,
    Normalize,
}

/// Common aggregation operations
enum AggregationType {
    Sum,
    Mean,
    StandardDeviation,
    Minimum,
    Maximum,
}

/// Matrix operations
enum MatrixOpType {
    Multiply,
    Transpose,
    Inverse,
}

/// Filter conditions
enum FilterType {
    GreaterThan(f32),
    LessThan(f32),
    Between(f32, f32),
    IsOutlier,  // e.g., outside n standard deviations
}

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
const PCI_CONFIG_SPACE_SIZE: usize = 256;      // Full configuration space size
const PCI_CONFIG_SPACE_MINIMAL: usize = 64;    // Minimal accessible size
const PCI_COMMAND_REGISTER_OFFSET: usize = 0x04;  // Command register location
const PCI_BAR0_OFFSET: usize = 0x10;          // First Base Address Register

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
    /// Safely reads the PCI Configuration Space for a device
    /// Handles both full (256 byte) and restricted (64 byte) access
    fn read_from_device(device_path: &str) -> std::io::Result<Self> {
        let config_path = Path::new(device_path).join("config");
        let mut config_data = [0u8; PCI_CONFIG_SPACE_SIZE];

        // Read whatever configuration space is available
        let bytes_read = std::fs::read(&config_path)?;
        
        if bytes_read.len() < PCI_CONFIG_SPACE_MINIMAL {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Configuration space too small: got {} bytes, expected at least {}",
                    bytes_read.len(),
                    PCI_CONFIG_SPACE_MINIMAL
                )
            ));
        }

        // Copy what we got, leave the rest as zeros
        let copy_size = bytes_read.len().min(PCI_CONFIG_SPACE_SIZE);
        config_data[..copy_size].copy_from_slice(&bytes_read[..copy_size]);
        
        Ok(PCIConfigurationSpace { raw_data: config_data })
    }

    
    
    
    
    // fn read_from_device(device_path: &str) -> std::io::Result<Self> {
    //     let config_path = Path::new(device_path).join("config");
    //     let mut config_data = [0u8; PCI_CONFIG_SPACE_SIZE];

    //     // Read the configuration space file
    //     let bytes_read = std::fs::read(&config_path)?;
        
    //     if bytes_read.len() < PCI_CONFIG_SPACE_SIZE {
    //         return Err(std::io::Error::new(
    //             std::io::ErrorKind::UnexpectedEof,
    //             format!(
    //                 "Configuration space too small: got {} bytes, expected {}",
    //                 bytes_read.len(),
    //                 PCI_CONFIG_SPACE_SIZE
    //             )
    //         ));
    //     }

    //     config_data.copy_from_slice(&bytes_read[..PCI_CONFIG_SPACE_SIZE]);
    //     Ok(PCIConfigurationSpace { raw_data: config_data })
    // }

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
/// Reads and validates PCI configuration space for an Intel GPU
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
    config_info.push_str(&format!("Configuration Space Size: {} bytes\n",
        PCI_CONFIG_SPACE_MINIMAL));
    
    Ok(config_info)
}

// fn read_gpu_configuration(
//     device_identification: &PCIDeviceIdentification
// ) -> std::io::Result<String> {
//     let config_space = PCIConfigurationSpace::read_from_device(&device_identification.pci_device_path)?;
    
//     let mut config_info = String::new();
//     config_info.push_str(&format!("Device Configuration Information:\n"));
//     config_info.push_str(&format!("Memory Access Enabled: {}\n", 
//         config_space.is_memory_access_enabled()));
//     config_info.push_str(&format!("BAR0 Address: 0x{:08x}\n", 
//         config_space.get_bar0()));
    
//     Ok(config_info)
// }



/// Represents different types of string-to-numeric transformations
/// suitable for GPU processing
enum StringVectorization {
    /// One-hot encoding for categorical strings
    OneHot {
        /// Mapping of strings to their numeric indices
        vocabulary: HashMap<String, usize>,
        /// Total size of vocabulary (dimensionality of one-hot vector)
        vocab_size: usize,
    },
    
    /// Token-based encoding (e.g., for NLP tasks)
    TokenEncoding {
        /// Maximum sequence length
        max_seq_length: usize,
        /// Vocabulary mapping
        token_to_id: HashMap<String, u32>,
    },
    
    /// Character-level encoding
    CharacterEncoding {
        /// Valid character set
        char_to_id: HashMap<char, u32>,
        /// Maximum string length
        max_length: usize,
    },
}

/// Processes large text files in batches, converting to GPU-suitable format
struct BatchTextProcessor {
    /// File being processed
    file_path: PathBuf,
    /// Size of each batch in rows
    batch_size: usize,
    /// Current position in file
    current_position: u64,
    /// Vectorization strategy
    vectorization: StringVectorization,
}

impl BatchTextProcessor {
    /// Creates a new batch processor with specified vectorization
    fn new(
        file_path: PathBuf, 
        batch_size: usize,
        vectorization: StringVectorization
    ) -> Result<Self, std::io::Error> {
        // Validate file exists
        if !file_path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found"
            ));
        }
        
        Ok(BatchTextProcessor {
            file_path,
            batch_size,
            current_position: 0,
            vectorization,
        })
    }

    /// Reads and processes the next batch of text data
    fn process_next_batch(&mut self) -> Result<Option<Vec<Vec<f32>>>, std::io::Error> {
        let file = std::fs::File::open(&self.file_path)?;
        let mut reader = std::io::BufReader::new(file);
        let mut processed_batch = Vec::new();
        
        // Seek to current position
        use std::io::Seek;
        reader.seek(std::io::SeekFrom::Start(self.current_position))?;
        
        // Read batch_size lines or until EOF
        let lines = reader.lines().take(self.batch_size);
        let mut reached_eof = false;
        
        for line in lines {
            match line {
                Ok(text) => {
                    match &self.vectorization {
                        StringVectorization::OneHot { vocabulary, vocab_size } => {
                            let encoded = self.one_hot_encode(&text, vocabulary, *vocab_size);
                            processed_batch.push(encoded);
                        },
                        StringVectorization::TokenEncoding { max_seq_length, token_to_id } => {
                            let encoded = self.token_encode(&text, token_to_id, *max_seq_length);
                            processed_batch.push(encoded);
                        },
                        StringVectorization::CharacterEncoding { char_to_id, max_length } => {
                            let encoded = self.char_encode(&text, char_to_id, *max_length);
                            processed_batch.push(encoded);
                        },
                    }
                },
                Err(e) => return Err(e),
            }
        }
        
        // Update position for next batch
        if reached_eof {
            Ok(None)
        } else {
            Ok(Some(processed_batch))
        }
    }

    /// One-hot encodes a string using vocabulary
    fn one_hot_encode(
        &self,
        text: &str,
        vocabulary: &HashMap<String, usize>,
        vocab_size: usize
    ) -> Vec<f32> {
        let mut encoding = vec![0.0; vocab_size];
        if let Some(&index) = vocabulary.get(text) {
            encoding[index] = 1.0;
        }
        encoding
    }

    /// Encodes text as token sequence
    fn token_encode(
        &self,
        text: &str,
        token_to_id: &HashMap<String, u32>,
        max_length: usize
    ) -> Vec<f32> {
        // Split text into tokens (simplified)
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut encoding = Vec::new();
        
        for token in tokens.iter().take(max_length) {
            if let Some(&id) = token_to_id.get(*token) {
                encoding.push(id as f32);
            }
        }
        
        // Pad if necessary
        while encoding.len() < max_length {
            encoding.push(0.0);
        }
        
        encoding
    }

    /// Character-level encoding
    fn char_encode(
        &self,
        text: &str,
        char_to_id: &HashMap<char, u32>,
        max_length: usize
    ) -> Vec<f32> {
        let mut encoding = Vec::new();
        
        for c in text.chars().take(max_length) {
            if let Some(&id) = char_to_id.get(&c) {
                encoding.push(id as f32);
            }
        }
        
        // Pad if necessary
        while encoding.len() < max_length {
            encoding.push(0.0);
        }
        
        encoding
    }
}


/// For testing without GPU access
fn create_test_gpu_config() -> PCIConfigurationSpace {
    let mut test_data = [0u8; PCI_CONFIG_SPACE_SIZE];
    // Set some test values
    test_data[PCI_COMMAND_REGISTER_OFFSET] = 0x02;  // Memory access enabled
    test_data[PCI_BAR0_OFFSET..PCI_BAR0_OFFSET + 4]
        .copy_from_slice(&[0x00, 0x00, 0x00, 0x80]);  // Test BAR0 address
    PCIConfigurationSpace { raw_data: test_data }
}

/// Example usage:
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("May require: sudo cargo run");
    
    // for testing:
    println!("\nTest Configuration:");
    let test_config = create_test_gpu_config();
    println!("Memory Access Enabled: {}", test_config.is_memory_access_enabled());
    println!("BAR0 Address: 0x{:08x}", test_config.get_bar0());
        
    
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
    
    // Example vocabulary for one-hot encoding
    let mut vocabulary = HashMap::new();
    vocabulary.insert("hello".to_string(), 0);
    vocabulary.insert("world".to_string(), 1);
    
    let vectorization = StringVectorization::OneHot {
        vocabulary,
        vocab_size: 2,
    };
    
    let mut processor = BatchTextProcessor::new(
        PathBuf::from("data/large_text_file.csv"),
        1000, // process 1000 rows at a time
        vectorization
    )?;
    
    // Process batches until EOF
    while let Some(batch) = processor.process_next_batch()? {
        // Here batch contains vectors ready for GPU processing
        println!("Processed batch of {} rows", batch.len());
    }
    
    
   // Example configuration
    let config = TextProcessingConfig {
        batch_size: 1000,
        has_header: true,
        columns_to_process: vec![0, 1], // Process first two columns
        max_vocab_size: 10000,
    };

    let mut processor = CSVBatchProcessor::new(
        PathBuf::from("data/example.csv"),
        config,
    )?;

    // First pass: build vocabularies
    processor.build_vocabularies()?;

    println!("Vocabulary sizes:");
    for (i, vocab) in processor.column_vocabularies.iter().enumerate() {
        println!("Column {}: {} unique terms", i, vocab.len());
    }

    // Second pass: process batches
    while let Some(batch) = processor.process_next_batch()? {
        println!(
            "Processed batch: {} rows, vector size: {}",
            batch.row_count,
            batch.vector_size
        );
        // Here you would send batch.vectors to GPU for processing
    }
    
    
    Ok(())
}

