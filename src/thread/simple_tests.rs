//! Simple compilation tests for the new thread module

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_platform_sync_compilation() {
        // Just ensure it compiles
        let _ = std::marker::PhantomData::<DefaultPlatformSync>;
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_futex_mutex_basic() {
        let mutex = FutexMutex::new();
        let _guard = mutex.lock().unwrap();
        // Guard should automatically unlock on drop
    }

    #[test]
    fn test_atomic_stack_basic() {
        let stack = AtomicStack::new();
        stack.push(42);
        assert_eq!(stack.pop(), Some(42));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_atomic_ext_basic() {
        let atomic = std::sync::atomic::AtomicU32::new(10);
        
        // Test atomic maximize
        let result = atomic.atomic_maximize(5, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(result, 10);
        
        let result = atomic.atomic_maximize(15, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(result, 15);
    }

    #[test]
    fn test_memory_ordering() {
        // Just test that the functions compile and run
        memory_ordering::full_barrier();
        memory_ordering::load_barrier();
        memory_ordering::store_barrier();
    }

    #[test]
    fn test_spin_loop_hint() {
        // Test that it compiles and runs
        spin_loop_hint();
    }
}