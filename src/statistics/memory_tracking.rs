//! Universal Memory Tracking Interface
//!
//! Provides comprehensive memory tracking capabilities across all Zipora components
//! with detailed breakdown and analysis functionality.

use crate::error::ZiporaError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Universal memory size interface for comprehensive memory tracking
pub trait MemorySize {
    /// Returns the total memory footprint in bytes
    fn mem_size(&self) -> usize;
    
    /// Returns detailed memory breakdown by component
    fn detailed_mem_size(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            total: self.mem_size(),
            components: HashMap::new(),
        }
    }
    
    /// Returns memory alignment requirements
    fn mem_alignment(&self) -> usize where Self: Sized {
        std::mem::align_of::<Self>()
    }
    
    /// Returns whether memory is contiguous
    fn is_contiguous(&self) -> bool {
        true // Default assumption
    }
}

/// Detailed memory usage breakdown
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub total: usize,
    pub components: HashMap<String, usize>,
}

impl MemoryBreakdown {
    pub fn new() -> Self {
        Self {
            total: 0,
            components: HashMap::new(),
        }
    }
    
    pub fn add_component(&mut self, name: &str, size: usize) {
        self.components.insert(name.to_string(), size);
        self.total += size;
    }
    
    pub fn merge(&mut self, other: &MemoryBreakdown) {
        for (name, size) in &other.components {
            *self.components.entry(name.clone()).or_insert(0) += size;
        }
        self.total += other.total;
    }
    
    pub fn component_percentage(&self, component: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        
        self.components.get(component).map_or(0.0, |&size| {
            (size as f64 / self.total as f64) * 100.0
        })
    }
    
    pub fn largest_component(&self) -> Option<(&String, &usize)> {
        self.components.iter().max_by_key(|(_, size)| *size)
    }
    
    pub fn smallest_component(&self) -> Option<(&String, &usize)> {
        self.components.iter().min_by_key(|(_, size)| *size)
    }
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Global memory tracker for all Zipora components
#[derive(Debug)]
pub struct GlobalMemoryTracker {
    tracked_objects: Arc<RwLock<HashMap<String, TrackedObject>>>,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

/// Information about a tracked memory object
#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub object_type: String,
    pub size: usize,
    pub breakdown: MemoryBreakdown,
    pub created_at: std::time::Instant,
    pub last_updated: std::time::Instant,
}

impl GlobalMemoryTracker {
    /// Create new global memory tracker
    pub fn new() -> Self {
        Self {
            tracked_objects: Arc::new(RwLock::new(HashMap::new())),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }
    
    /// Register a new memory object
    pub fn register_object<T: MemorySize>(
        &self, 
        id: String, 
        object: &T, 
        object_type: String
    ) -> Result<(), ZiporaError> {
        let breakdown = object.detailed_mem_size();
        let size = breakdown.total;
        
        let tracked = TrackedObject {
            object_type,
            size,
            breakdown,
            created_at: std::time::Instant::now(),
            last_updated: std::time::Instant::now(),
        };
        
        {
            let mut objects = self.tracked_objects.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on tracked objects")
            })?;
            objects.insert(id, tracked);
        }
        
        // Update global statistics
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        let current_peak = self.peak_allocated.load(Ordering::Relaxed);
        if new_total > current_peak {
            self.peak_allocated.store(new_total, Ordering::Relaxed);
        }
        
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Update an existing tracked object
    pub fn update_object<T: MemorySize>(
        &self, 
        id: &str, 
        object: &T
    ) -> Result<(), ZiporaError> {
        let new_breakdown = object.detailed_mem_size();
        let new_size = new_breakdown.total;
        
        {
            let mut objects = self.tracked_objects.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on tracked objects")
            })?;
            
            if let Some(tracked) = objects.get_mut(id) {
                let old_size = tracked.size;
                tracked.size = new_size;
                tracked.breakdown = new_breakdown;
                tracked.last_updated = std::time::Instant::now();
                
                // Update global total
                if new_size > old_size {
                    let increase = new_size - old_size;
                    let new_total = self.total_allocated.fetch_add(increase, Ordering::Relaxed) + increase;
                    let current_peak = self.peak_allocated.load(Ordering::Relaxed);
                    if new_total > current_peak {
                        self.peak_allocated.store(new_total, Ordering::Relaxed);
                    }
                } else if old_size > new_size {
                    let decrease = old_size - new_size;
                    self.total_allocated.fetch_sub(decrease, Ordering::Relaxed);
                }
            } else {
                return Err(ZiporaError::invalid_data(format!("Object '{}' not found", id)));
            }
        }
        
        Ok(())
    }
    
    /// Unregister a memory object
    pub fn unregister_object(&self, id: &str) -> Result<(), ZiporaError> {
        let size = {
            let mut objects = self.tracked_objects.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on tracked objects")
            })?;
            
            objects.remove(id)
                .map(|tracked| tracked.size)
                .ok_or_else(|| ZiporaError::invalid_data(format!("Object '{}' not found", id)))?
        };
        
        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get memory breakdown for a specific object
    pub fn get_object_breakdown(&self, id: &str) -> Result<MemoryBreakdown, ZiporaError> {
        let objects = self.tracked_objects.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on tracked objects")
        })?;
        
        objects.get(id)
            .map(|tracked| tracked.breakdown.clone())
            .ok_or_else(|| ZiporaError::invalid_data(format!("Object '{}' not found", id)))
    }
    
    /// Get total memory usage
    pub fn total_memory(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }
    
    /// Get peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.peak_allocated.load(Ordering::Relaxed)
    }
    
    /// Get number of tracked objects
    pub fn object_count(&self) -> Result<usize, ZiporaError> {
        let objects = self.tracked_objects.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on tracked objects")
        })?;
        Ok(objects.len())
    }
    
    /// Get comprehensive memory breakdown by type
    pub fn breakdown_by_type(&self) -> Result<HashMap<String, MemoryBreakdown>, ZiporaError> {
        let objects = self.tracked_objects.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on tracked objects")
        })?;
        
        let mut type_breakdowns: HashMap<String, MemoryBreakdown> = HashMap::new();
        
        for tracked in objects.values() {
            let breakdown = type_breakdowns
                .entry(tracked.object_type.clone())
                .or_insert_with(MemoryBreakdown::new);
            breakdown.merge(&tracked.breakdown);
        }
        
        Ok(type_breakdowns)
    }
    
    /// Generate comprehensive memory report
    pub fn generate_report(&self) -> Result<String, ZiporaError> {
        let objects = self.tracked_objects.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on tracked objects")
        })?;
        
        let mut report = String::from("=== Global Memory Tracker Report ===\n\n");
        
        // Summary statistics
        report.push_str("Summary:\n");
        report.push_str(&format!("  Total Allocated: {} bytes\n", self.total_memory()));
        report.push_str(&format!("  Peak Allocated: {} bytes\n", self.peak_memory()));
        report.push_str(&format!("  Tracked Objects: {}\n", objects.len()));
        report.push_str(&format!("  Total Allocations: {}\n\n", self.allocation_count.load(Ordering::Relaxed)));
        
        // Breakdown by type
        drop(objects); // Release lock before calling breakdown_by_type
        let type_breakdowns = self.breakdown_by_type()?;
        
        if !type_breakdowns.is_empty() {
            report.push_str("Memory by Type:\n");
            let mut type_sizes: Vec<_> = type_breakdowns.iter().collect();
            type_sizes.sort_by(|a, b| b.1.total.cmp(&a.1.total));
            
            for (object_type, breakdown) in type_sizes {
                let percentage = (breakdown.total as f64 / self.total_memory() as f64) * 100.0;
                report.push_str(&format!(
                    "  {}: {} bytes ({:.1}%)\n",
                    object_type, breakdown.total, percentage
                ));
            }
            report.push('\n');
        }
        
        // Individual objects (top 10 by size)
        let objects = self.tracked_objects.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on tracked objects")
        })?;
        
        if !objects.is_empty() {
            report.push_str("Largest Objects:\n");
            let mut object_sizes: Vec<_> = objects.iter().collect();
            object_sizes.sort_by(|a, b| b.1.size.cmp(&a.1.size));
            
            for (id, tracked) in object_sizes.iter().take(10) {
                let percentage = (tracked.size as f64 / self.total_memory() as f64) * 100.0;
                report.push_str(&format!(
                    "  {} ({}): {} bytes ({:.1}%)\n",
                    id, tracked.object_type, tracked.size, percentage
                ));
            }
        }
        
        Ok(report)
    }
    
    /// Clear all tracked objects
    pub fn clear(&self) -> Result<(), ZiporaError> {
        let mut objects = self.tracked_objects.write().map_err(|_| {
            ZiporaError::system_error("Failed to acquire write lock on tracked objects")
        })?;
        
        objects.clear();
        self.total_allocated.store(0, Ordering::Relaxed);
        // Don't reset peak_allocated - it's a historical maximum
        
        Ok(())
    }
}

impl Default for GlobalMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local memory tracker for high-performance scenarios
thread_local! {
    static LOCAL_TRACKER: std::cell::RefCell<LocalMemoryTracker> = std::cell::RefCell::new(LocalMemoryTracker::new());
}

/// Thread-local memory tracking
#[derive(Debug)]
pub struct LocalMemoryTracker {
    local_total: usize,
    local_objects: HashMap<String, usize>,
}

impl LocalMemoryTracker {
    fn new() -> Self {
        Self {
            local_total: 0,
            local_objects: HashMap::new(),
        }
    }
    
    /// Track local memory allocation
    pub fn track_allocation(id: String, size: usize) {
        LOCAL_TRACKER.with(|tracker| {
            let mut tracker = tracker.borrow_mut();
            if let Some(old_size) = tracker.local_objects.insert(id, size) {
                tracker.local_total = tracker.local_total - old_size + size;
            } else {
                tracker.local_total += size;
            }
        });
    }
    
    /// Track local memory deallocation
    pub fn track_deallocation(id: &str) {
        LOCAL_TRACKER.with(|tracker| {
            let mut tracker = tracker.borrow_mut();
            if let Some(size) = tracker.local_objects.remove(id) {
                tracker.local_total -= size;
            }
        });
    }
    
    /// Get current local memory usage
    pub fn local_total() -> usize {
        LOCAL_TRACKER.with(|tracker| {
            tracker.borrow().local_total
        })
    }
    
    /// Get local object count
    pub fn local_object_count() -> usize {
        LOCAL_TRACKER.with(|tracker| {
            tracker.borrow().local_objects.len()
        })
    }
}

/// Memory tracking utilities
pub mod utils {
    use super::*;
    
    /// Calculate memory overhead percentage
    pub fn calculate_overhead(data_size: usize, total_size: usize) -> f64 {
        if data_size == 0 {
            return 0.0;
        }
        
        let overhead = total_size.saturating_sub(data_size);
        (overhead as f64 / data_size as f64) * 100.0
    }
    
    /// Format memory size in human-readable format
    pub fn format_size(size: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: f64 = 1024.0;
        
        if size == 0 {
            return "0 B".to_string();
        }
        
        let mut size_f = size as f64;
        let mut unit_index = 0;
        
        while size_f >= THRESHOLD && unit_index < UNITS.len() - 1 {
            size_f /= THRESHOLD;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", size, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size_f, UNITS[unit_index])
        }
    }
    
    /// Analyze memory fragmentation
    pub fn analyze_fragmentation(breakdown: &MemoryBreakdown) -> FragmentationAnalysis {
        let component_count = breakdown.components.len();
        
        if component_count <= 1 {
            return FragmentationAnalysis {
                fragmentation_score: 0.0,
                component_count,
                largest_component_ratio: if breakdown.total > 0 { 1.0 } else { 0.0 },
                uniformity_score: 1.0,
            };
        }
        
        let total = breakdown.total as f64;
        let largest_size = breakdown.components.values().max().copied().unwrap_or(0) as f64;
        let expected_size = total / component_count as f64;
        
        // Calculate variance from expected uniform distribution
        let variance: f64 = breakdown.components.values()
            .map(|&size| {
                let diff = size as f64 - expected_size;
                diff * diff
            })
            .sum::<f64>() / component_count as f64;
        
        let std_dev = variance.sqrt();
        let coefficient_of_variation = if expected_size > 0.0 { std_dev / expected_size } else { 0.0 };
        
        FragmentationAnalysis {
            fragmentation_score: coefficient_of_variation.min(2.0) / 2.0, // Normalize to [0,1]
            component_count,
            largest_component_ratio: if total > 0.0 { largest_size / total } else { 0.0 },
            uniformity_score: (1.0 / (1.0 + coefficient_of_variation)).min(1.0),
        }
    }
}

/// Memory fragmentation analysis results
#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    /// Fragmentation score (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_score: f64,
    /// Number of memory components
    pub component_count: usize,
    /// Ratio of largest component to total memory
    pub largest_component_ratio: f64,
    /// Uniformity score (1.0 = perfectly uniform, 0.0 = highly uneven)
    pub uniformity_score: f64,
}

// Implement MemorySize for common Rust types
impl MemorySize for String {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.capacity()
    }
}

impl<T> MemorySize for Vec<T> {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<Self>() + (self.capacity() * std::mem::size_of::<T>())
    }
}

impl<K, V> MemorySize for HashMap<K, V> {
    fn mem_size(&self) -> usize {
        // Approximation: actual HashMap memory usage is complex
        std::mem::size_of::<Self>() + 
        (self.capacity() * (std::mem::size_of::<K>() + std::mem::size_of::<V>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("nodes", 1000);
        breakdown.add_component("cache", 500);
        
        assert_eq!(breakdown.total, 1500);
        assert_eq!(breakdown.components.len(), 2);
        let percentage = breakdown.component_percentage("nodes");
        assert!((percentage - 66.66666666666667).abs() < 0.0000001); // Use approximate equality for floating point
        
        let (largest_name, largest_size) = breakdown.largest_component().unwrap();
        assert_eq!(largest_name, "nodes");
        assert_eq!(*largest_size, 1000);
    }

    #[test]
    fn test_memory_size_vec() {
        let vec = vec![1u8, 2, 3, 4, 5];
        let size = vec.mem_size();
        
        // Should include struct overhead plus capacity
        assert!(size >= std::mem::size_of::<Vec<u8>>() + 5);
    }

    #[test]
    fn test_global_memory_tracker() {
        let tracker = GlobalMemoryTracker::new();
        let test_vec = vec![1u8; 1000];
        
        tracker.register_object(
            "test_vec".to_string(),
            &test_vec,
            "Vec<u8>".to_string(),
        ).unwrap();
        
        assert!(tracker.total_memory() > 0);
        assert_eq!(tracker.object_count().unwrap(), 1);
        
        let breakdown = tracker.get_object_breakdown("test_vec").unwrap();
        assert!(breakdown.total > 0);
        
        tracker.unregister_object("test_vec").unwrap();
        assert_eq!(tracker.object_count().unwrap(), 0);
    }

    #[test]
    fn test_local_memory_tracker() {
        LocalMemoryTracker::track_allocation("test1".to_string(), 100);
        LocalMemoryTracker::track_allocation("test2".to_string(), 200);
        
        assert_eq!(LocalMemoryTracker::local_total(), 300);
        assert_eq!(LocalMemoryTracker::local_object_count(), 2);
        
        LocalMemoryTracker::track_deallocation("test1");
        assert_eq!(LocalMemoryTracker::local_total(), 200);
        assert_eq!(LocalMemoryTracker::local_object_count(), 1);
    }

    #[test]
    fn test_memory_utils() {
        assert_eq!(utils::calculate_overhead(100, 120), 20.0);
        assert_eq!(utils::format_size(1024), "1.00 KB");
        assert_eq!(utils::format_size(1048576), "1.00 MB");
        assert_eq!(utils::format_size(500), "500 B");
    }

    #[test]
    fn test_fragmentation_analysis() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("comp1", 1000); // Large component
        breakdown.add_component("comp2", 100);  // Small component
        breakdown.add_component("comp3", 50);   // Very small component
        
        let analysis = utils::analyze_fragmentation(&breakdown);
        
        assert!(analysis.fragmentation_score > 0.0);
        assert_eq!(analysis.component_count, 3);
        assert!(analysis.largest_component_ratio > 0.8); // comp1 dominates
        assert!(analysis.uniformity_score < 1.0); // Not uniform
    }

    #[test]
    fn test_tracker_update() {
        let tracker = GlobalMemoryTracker::new();
        let mut test_vec = vec![1u8; 100];
        
        tracker.register_object(
            "test_vec".to_string(),
            &test_vec,
            "Vec<u8>".to_string(),
        ).unwrap();
        
        let initial_total = tracker.total_memory();
        
        // Grow the vector
        test_vec.extend_from_slice(&[0u8; 900]);
        tracker.update_object("test_vec", &test_vec).unwrap();
        
        let final_total = tracker.total_memory();
        assert!(final_total > initial_total);
    }

    #[test]
    fn test_breakdown_by_type() {
        let tracker = GlobalMemoryTracker::new();
        
        let vec1 = vec![1u8; 100];
        let vec2 = vec![2u8; 200];
        let string1 = String::from("hello world");
        
        tracker.register_object("vec1".to_string(), &vec1, "Vec<u8>".to_string()).unwrap();
        tracker.register_object("vec2".to_string(), &vec2, "Vec<u8>".to_string()).unwrap();
        tracker.register_object("string1".to_string(), &string1, "String".to_string()).unwrap();
        
        let breakdown = tracker.breakdown_by_type().unwrap();
        
        assert!(breakdown.contains_key("Vec<u8>"));
        assert!(breakdown.contains_key("String"));
        assert!(breakdown["Vec<u8>"].total > breakdown["String"].total);
    }

    #[test]
    fn test_peak_memory_tracking() {
        let tracker = GlobalMemoryTracker::new();
        let test_vec = vec![1u8; 1000];
        
        tracker.register_object("test".to_string(), &test_vec, "Vec<u8>".to_string()).unwrap();
        let peak1 = tracker.peak_memory();
        
        tracker.unregister_object("test").unwrap();
        let peak2 = tracker.peak_memory();
        
        // Peak should remain the same even after deallocation
        assert_eq!(peak1, peak2);
        assert!(peak1 > 0);
    }
}