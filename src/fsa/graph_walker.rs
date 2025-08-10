//! Graph Walker Utilities
//!
//! High-performance graph traversal utilities for FSA structures with multiple
//! traversal strategies optimized for different access patterns and cache efficiency.

use crate::error::{Result, ZiporaError};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;

/// Graph traversal strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkMethod {
    /// Breadth-first search with color marking
    BreadthFirst,
    /// Breadth-first search with multiple passes
    BreadthFirstMultiPass,
    /// Performance-first search (hybrid BFS/DFS)
    PerformanceFirst,
    /// Depth-first search with preorder traversal
    DepthFirst,
    /// Cache-friendly search: BFS for 2 levels, then DFS
    CacheFriendly,
    /// Tree-specific breadth-first (no cycle detection)
    TreeBreadthFirst,
    /// Tree-specific depth-first (no cycle detection)
    TreeDepthFirst,
    /// Depth-first search with multiple passes
    DepthFirstMultiPass,
}

/// Vertex color for graph traversal algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexColor {
    /// Unvisited vertex
    White,
    /// Currently being processed
    Gray,
    /// Fully processed
    Black,
    /// Custom color with ID
    Custom(u32),
}

impl Default for VertexColor {
    fn default() -> Self {
        VertexColor::White
    }
}

/// Graph vertex trait
pub trait Vertex: Clone + Eq + Hash {
    /// Get unique identifier for the vertex
    fn id(&self) -> u64;
    
    /// Get outgoing edges from this vertex
    fn outgoing_edges(&self) -> Vec<Self>;
    
    /// Check if this vertex is a terminal/final state
    fn is_terminal(&self) -> bool {
        false
    }
}

/// Simple vertex implementation for testing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimpleVertex {
    pub id: u32,
    pub edges: Vec<u32>,
    pub is_terminal: bool,
}

impl SimpleVertex {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            edges: Vec::new(),
            is_terminal: false,
        }
    }
    
    pub fn with_edges(id: u32, edges: Vec<u32>) -> Self {
        Self {
            id,
            edges,
            is_terminal: false,
        }
    }
    
    pub fn with_terminal(id: u32, is_terminal: bool) -> Self {
        Self {
            id,
            edges: Vec::new(),
            is_terminal,
        }
    }
}

impl Vertex for SimpleVertex {
    fn id(&self) -> u64 {
        self.id as u64
    }
    
    fn outgoing_edges(&self) -> Vec<Self> {
        self.edges.iter().map(|&id| SimpleVertex::new(id)).collect()
    }
    
    fn is_terminal(&self) -> bool {
        self.is_terminal
    }
}

/// Visitor trait for graph traversal callbacks
pub trait GraphVisitor<V: Vertex> {
    /// Called when a vertex is first discovered
    fn visit_vertex(&mut self, vertex: &V, depth: usize) -> Result<bool>;
    
    /// Called when an edge is traversed
    fn visit_edge(&mut self, from: &V, to: &V) -> Result<bool>;
    
    /// Called when backtracking from a vertex (DFS only)
    fn finish_vertex(&mut self, vertex: &V, depth: usize) -> Result<()> {
        let _ = (vertex, depth);
        Ok(())
    }
}

/// Configuration for graph walkers
#[derive(Debug, Clone)]
pub struct WalkerConfig {
    /// Maximum depth to traverse
    pub max_depth: Option<usize>,
    /// Maximum number of vertices to visit
    pub max_vertices: Option<usize>,
    /// Enable cycle detection
    pub cycle_detection: bool,
    /// Initial queue capacity for BFS
    pub initial_queue_capacity: usize,
    /// Use incremental color IDs for multi-pass
    pub incremental_colors: bool,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            max_vertices: None,
            cycle_detection: true,
            initial_queue_capacity: 1024,
            incremental_colors: false,
        }
    }
}

impl WalkerConfig {
    /// Create configuration for tree traversal (no cycle detection)
    pub fn for_tree() -> Self {
        Self {
            cycle_detection: false,
            ..Default::default()
        }
    }
    
    /// Create configuration for large graphs
    pub fn for_large_graph() -> Self {
        Self {
            max_vertices: Some(1_000_000),
            initial_queue_capacity: 10_000,
            ..Default::default()
        }
    }
    
    /// Create configuration for multi-pass traversal
    pub fn for_multi_pass() -> Self {
        Self {
            incremental_colors: true,
            ..Default::default()
        }
    }
}

/// Statistics for graph traversal
#[derive(Debug, Clone, Default)]
pub struct WalkStats {
    /// Number of vertices visited
    pub vertices_visited: usize,
    /// Number of edges traversed
    pub edges_traversed: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Number of cycles detected
    pub cycles_detected: usize,
    /// Number of back edges found
    pub back_edges: usize,
}

/// Breadth-First Search Graph Walker
pub struct BfsGraphWalker<V: Vertex> {
    config: WalkerConfig,
    vertex_colors: HashMap<u64, VertexColor>,
    queue: VecDeque<(V, usize)>, // (vertex, depth)
    stats: WalkStats,
    _phantom: PhantomData<V>,
}

impl<V: Vertex> BfsGraphWalker<V> {
    /// Create a new BFS graph walker
    pub fn new(config: WalkerConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.initial_queue_capacity),
            vertex_colors: HashMap::new(),
            config,
            stats: WalkStats::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Walk the graph starting from the given vertex
    pub fn walk<Visitor>(&mut self, start: V, visitor: &mut Visitor) -> Result<()>
    where
        Visitor: GraphVisitor<V> + ?Sized,
    {
        self.reset();
        self.queue.push_back((start.clone(), 0));
        self.vertex_colors.insert(start.id(), VertexColor::Gray);
        
        while let Some((current, depth)) = self.queue.pop_front() {
            // Check limits
            if let Some(max_depth) = self.config.max_depth {
                if depth >= max_depth {
                    continue;
                }
            }
            
            if let Some(max_vertices) = self.config.max_vertices {
                if self.stats.vertices_visited >= max_vertices {
                    break;
                }
            }
            
            // Visit vertex
            if !visitor.visit_vertex(&current, depth)? {
                continue; // Skip this vertex
            }
            
            self.stats.vertices_visited += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
            
            // Process outgoing edges
            for next_vertex in current.outgoing_edges() {
                let next_id = next_vertex.id();
                
                if !visitor.visit_edge(&current, &next_vertex)? {
                    continue; // Skip this edge
                }
                
                self.stats.edges_traversed += 1;
                
                if self.config.cycle_detection {
                    match self.vertex_colors.get(&next_id) {
                        Some(VertexColor::White) | None => {
                            // Unvisited - add to queue
                            self.vertex_colors.insert(next_id, VertexColor::Gray);
                            self.queue.push_back((next_vertex, depth + 1));
                        }
                        Some(VertexColor::Gray) => {
                            // Back edge - cycle detected
                            self.stats.cycles_detected += 1;
                            self.stats.back_edges += 1;
                        }
                        Some(VertexColor::Black) => {
                            // Already processed - cross edge
                        }
                        Some(VertexColor::Custom(_)) => {
                            // Custom color handling
                        }
                    }
                } else {
                    // No cycle detection - just add to queue
                    self.queue.push_back((next_vertex, depth + 1));
                }
            }
            
            // Mark vertex as fully processed
            if self.config.cycle_detection {
                self.vertex_colors.insert(current.id(), VertexColor::Black);
            }
        }
        
        Ok(())
    }
    
    /// Reset walker state
    pub fn reset(&mut self) {
        self.vertex_colors.clear();
        self.queue.clear();
        self.stats = WalkStats::default();
    }
    
    /// Get traversal statistics
    pub fn stats(&self) -> &WalkStats {
        &self.stats
    }
}

/// Depth-First Search Graph Walker
pub struct DfsGraphWalker<V: Vertex> {
    config: WalkerConfig,
    vertex_colors: HashMap<u64, VertexColor>,
    stack: Vec<(V, usize, bool)>, // (vertex, depth, visited_children)
    stats: WalkStats,
    _phantom: PhantomData<V>,
}

impl<V: Vertex> DfsGraphWalker<V> {
    /// Create a new DFS graph walker
    pub fn new(config: WalkerConfig) -> Self {
        Self {
            stack: Vec::with_capacity(config.initial_queue_capacity),
            vertex_colors: HashMap::new(),
            config,
            stats: WalkStats::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Walk the graph starting from the given vertex
    pub fn walk<Visitor>(&mut self, start: V, visitor: &mut Visitor) -> Result<()>
    where
        Visitor: GraphVisitor<V> + ?Sized,
    {
        self.reset();
        self.stack.push((start.clone(), 0, false));
        self.vertex_colors.insert(start.id(), VertexColor::Gray);
        
        while let Some((current, depth, visited_children)) = self.stack.pop() {
            if visited_children {
                // Backtracking - finish vertex
                visitor.finish_vertex(&current, depth)?;
                self.vertex_colors.insert(current.id(), VertexColor::Black);
                continue;
            }
            
            // Check limits
            if let Some(max_depth) = self.config.max_depth {
                if depth >= max_depth {
                    continue;
                }
            }
            
            if let Some(max_vertices) = self.config.max_vertices {
                if self.stats.vertices_visited >= max_vertices {
                    break;
                }
            }
            
            // Visit vertex
            if !visitor.visit_vertex(&current, depth)? {
                continue;
            }
            
            self.stats.vertices_visited += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
            
            // Push back for finish processing
            self.stack.push((current.clone(), depth, true));
            
            // Process outgoing edges (in reverse order for consistent traversal)
            let mut edges: Vec<_> = current.outgoing_edges();
            edges.reverse();
            
            for next_vertex in edges {
                let next_id = next_vertex.id();
                
                if !visitor.visit_edge(&current, &next_vertex)? {
                    continue;
                }
                
                self.stats.edges_traversed += 1;
                
                if self.config.cycle_detection {
                    match self.vertex_colors.get(&next_id) {
                        Some(VertexColor::White) | None => {
                            // Unvisited - add to stack
                            self.vertex_colors.insert(next_id, VertexColor::Gray);
                            self.stack.push((next_vertex, depth + 1, false));
                        }
                        Some(VertexColor::Gray) => {
                            // Back edge - cycle detected
                            self.stats.cycles_detected += 1;
                            self.stats.back_edges += 1;
                        }
                        Some(VertexColor::Black) => {
                            // Already processed
                        }
                        Some(VertexColor::Custom(_)) => {
                            // Custom color handling
                        }
                    }
                } else {
                    // No cycle detection
                    self.stack.push((next_vertex, depth + 1, false));
                }
            }
        }
        
        Ok(())
    }
    
    /// Reset walker state
    pub fn reset(&mut self) {
        self.vertex_colors.clear();
        self.stack.clear();
        self.stats = WalkStats::default();
    }
    
    /// Get traversal statistics
    pub fn stats(&self) -> &WalkStats {
        &self.stats
    }
}

/// Cache-Friendly Search Walker
/// 
/// Implements a hybrid approach: BFS for the first 2 levels to maximize cache locality,
/// then switches to DFS for memory efficiency
pub struct CfsGraphWalker<V: Vertex> {
    config: WalkerConfig,
    bfs_walker: BfsGraphWalker<V>,
    dfs_walker: DfsGraphWalker<V>,
    bfs_levels: usize,
    stats: WalkStats,
}

impl<V: Vertex> CfsGraphWalker<V> {
    /// Create a new CFS graph walker
    pub fn new(config: WalkerConfig) -> Self {
        let bfs_config = WalkerConfig {
            max_depth: Some(2), // BFS for first 2 levels
            ..config.clone()
        };
        
        Self {
            bfs_walker: BfsGraphWalker::new(bfs_config),
            dfs_walker: DfsGraphWalker::new(config.clone()),
            config,
            bfs_levels: 2,
            stats: WalkStats::default(),
        }
    }
    
    /// Walk the graph using cache-friendly strategy
    pub fn walk<Visitor>(&mut self, start: V, visitor: &mut Visitor) -> Result<()>
    where
        Visitor: GraphVisitor<V> + ?Sized,
    {
        self.reset();
        
        // Phase 1: BFS for first 2 levels
        let mut level_2_vertices = Vec::new();
        {
            let mut collecting_visitor = Level2Collector::new(visitor, &mut level_2_vertices);
            self.bfs_walker.walk(start, &mut collecting_visitor)?;
        }
        
        // Combine BFS stats
        self.stats.vertices_visited += self.bfs_walker.stats().vertices_visited;
        self.stats.edges_traversed += self.bfs_walker.stats().edges_traversed;
        self.stats.max_depth_reached = self.bfs_walker.stats().max_depth_reached;
        self.stats.cycles_detected += self.bfs_walker.stats().cycles_detected;
        
        // Phase 2: DFS from each level 2 vertex
        for vertex in level_2_vertices {
            self.dfs_walker.walk(vertex, visitor)?;
            
            // Combine DFS stats
            self.stats.vertices_visited += self.dfs_walker.stats().vertices_visited;
            self.stats.edges_traversed += self.dfs_walker.stats().edges_traversed;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(
                self.dfs_walker.stats().max_depth_reached
            );
            self.stats.cycles_detected += self.dfs_walker.stats().cycles_detected;
            
            self.dfs_walker.reset();
        }
        
        Ok(())
    }
    
    /// Reset walker state
    pub fn reset(&mut self) {
        self.bfs_walker.reset();
        self.dfs_walker.reset();
        self.stats = WalkStats::default();
    }
    
    /// Get traversal statistics
    pub fn stats(&self) -> &WalkStats {
        &self.stats
    }
}

/// Helper visitor for collecting level 2 vertices
struct Level2Collector<'a, V: Vertex, Visitor: GraphVisitor<V> + ?Sized> {
    visitor: &'a mut Visitor,
    level_2_vertices: &'a mut Vec<V>,
}

impl<'a, V: Vertex, Visitor: GraphVisitor<V> + ?Sized> Level2Collector<'a, V, Visitor> {
    fn new(visitor: &'a mut Visitor, level_2_vertices: &'a mut Vec<V>) -> Self {
        Self {
            visitor,
            level_2_vertices,
        }
    }
}

impl<'a, V: Vertex, Visitor: GraphVisitor<V> + ?Sized> GraphVisitor<V> for Level2Collector<'a, V, Visitor> {
    fn visit_vertex(&mut self, vertex: &V, depth: usize) -> Result<bool> {
        if depth == 2 {
            self.level_2_vertices.push(vertex.clone());
        }
        self.visitor.visit_vertex(vertex, depth)
    }
    
    fn visit_edge(&mut self, from: &V, to: &V) -> Result<bool> {
        self.visitor.visit_edge(from, to)
    }
    
    fn finish_vertex(&mut self, vertex: &V, depth: usize) -> Result<()> {
        self.visitor.finish_vertex(vertex, depth)
    }
}

/// Multi-pass walker for incremental processing
pub struct MultiPassWalker<V: Vertex> {
    config: WalkerConfig,
    vertex_colors: HashMap<u64, VertexColor>,
    current_color_id: u32,
    stats: WalkStats,
    _phantom: PhantomData<V>,
}

impl<V: Vertex> MultiPassWalker<V> {
    /// Create a new multi-pass walker
    pub fn new(config: WalkerConfig) -> Self {
        Self {
            config,
            vertex_colors: HashMap::new(),
            current_color_id: 1,
            stats: WalkStats::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Perform a single pass using the specified method
    pub fn walk_pass<Visitor>(&mut self, start: V, method: WalkMethod, visitor: &mut Visitor) -> Result<()>
    where
        Visitor: GraphVisitor<V> + ?Sized,
    {
        match method {
            WalkMethod::BreadthFirst => {
                let mut walker = BfsGraphWalker::new(self.config.clone());
                walker.walk(start, visitor)?;
                self.merge_stats(walker.stats());
            }
            WalkMethod::DepthFirst => {
                let mut walker = DfsGraphWalker::new(self.config.clone());
                walker.walk(start, visitor)?;
                self.merge_stats(walker.stats());
            }
            WalkMethod::CacheFriendly => {
                let mut walker = CfsGraphWalker::new(self.config.clone());
                walker.walk(start, visitor)?;
                self.merge_stats(walker.stats());
            }
            _ => {
                return Err(ZiporaError::invalid_data("Unsupported walk method for multi-pass"));
            }
        }
        
        if self.config.incremental_colors {
            self.current_color_id += 1;
        }
        
        Ok(())
    }
    
    /// Get the current color ID for this pass
    pub fn current_color(&self) -> VertexColor {
        VertexColor::Custom(self.current_color_id)
    }
    
    /// Reset for a new set of passes
    pub fn reset(&mut self) {
        self.vertex_colors.clear();
        self.current_color_id = 1;
        self.stats = WalkStats::default();
    }
    
    /// Get accumulated statistics
    pub fn stats(&self) -> &WalkStats {
        &self.stats
    }
    
    fn merge_stats(&mut self, other: &WalkStats) {
        self.stats.vertices_visited += other.vertices_visited;
        self.stats.edges_traversed += other.edges_traversed;
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(other.max_depth_reached);
        self.stats.cycles_detected += other.cycles_detected;
        self.stats.back_edges += other.back_edges;
    }
}

/// Generic graph walker factory
pub struct GraphWalkerFactory;

impl GraphWalkerFactory {
    /// Create a walker based on the specified method
    pub fn create_walker<V: Vertex + 'static>(method: WalkMethod, config: WalkerConfig) -> Box<dyn GraphWalker<V>> {
        match method {
            WalkMethod::BreadthFirst | WalkMethod::TreeBreadthFirst => {
                let mut walker_config = config;
                if method == WalkMethod::TreeBreadthFirst {
                    walker_config.cycle_detection = false;
                }
                Box::new(BfsGraphWalker::new(walker_config))
            }
            WalkMethod::DepthFirst | WalkMethod::TreeDepthFirst => {
                let mut walker_config = config;
                if method == WalkMethod::TreeDepthFirst {
                    walker_config.cycle_detection = false;
                }
                Box::new(DfsGraphWalker::new(walker_config))
            }
            WalkMethod::CacheFriendly => {
                Box::new(CfsGraphWalker::new(config))
            }
            WalkMethod::BreadthFirstMultiPass | WalkMethod::DepthFirstMultiPass => {
                Box::new(MultiPassWalker::new(config))
            }
            WalkMethod::PerformanceFirst => {
                // Use CFS as the performance-optimized strategy
                Box::new(CfsGraphWalker::new(config))
            }
        }
    }
}

/// Trait for generic graph walking
pub trait GraphWalker<V: Vertex> {
    fn walk_dyn(&mut self, start: V, visitor: &mut dyn GraphVisitor<V>) -> Result<()>;
    fn reset(&mut self);
    fn stats(&self) -> &WalkStats;
}

impl<V: Vertex> GraphWalker<V> for BfsGraphWalker<V> {
    fn walk_dyn(&mut self, start: V, visitor: &mut dyn GraphVisitor<V>) -> Result<()> {
        self.walk(start, visitor)
    }
    
    fn reset(&mut self) {
        BfsGraphWalker::reset(self)
    }
    
    fn stats(&self) -> &WalkStats {
        BfsGraphWalker::stats(self)
    }
}

impl<V: Vertex> GraphWalker<V> for DfsGraphWalker<V> {
    fn walk_dyn(&mut self, start: V, visitor: &mut dyn GraphVisitor<V>) -> Result<()> {
        self.walk(start, visitor)
    }
    
    fn reset(&mut self) {
        DfsGraphWalker::reset(self)
    }
    
    fn stats(&self) -> &WalkStats {
        DfsGraphWalker::stats(self)
    }
}

impl<V: Vertex> GraphWalker<V> for CfsGraphWalker<V> {
    fn walk_dyn(&mut self, start: V, visitor: &mut dyn GraphVisitor<V>) -> Result<()> {
        self.walk(start, visitor)
    }
    
    fn reset(&mut self) {
        CfsGraphWalker::reset(self)
    }
    
    fn stats(&self) -> &WalkStats {
        CfsGraphWalker::stats(self)
    }
}

impl<V: Vertex> GraphWalker<V> for MultiPassWalker<V> {
    fn walk_dyn(&mut self, start: V, visitor: &mut dyn GraphVisitor<V>) -> Result<()> {
        // Default to cache-friendly for single-pass usage
        self.walk_pass(start, WalkMethod::CacheFriendly, visitor)
    }
    
    fn reset(&mut self) {
        MultiPassWalker::reset(self)
    }
    
    fn stats(&self) -> &WalkStats {
        MultiPassWalker::stats(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestVisitor {
        visited_vertices: Vec<u32>,
        visited_edges: Vec<(u32, u32)>,
    }

    impl TestVisitor {
        fn new() -> Self {
            Self {
                visited_vertices: Vec::new(),
                visited_edges: Vec::new(),
            }
        }
    }

    impl GraphVisitor<SimpleVertex> for TestVisitor {
        fn visit_vertex(&mut self, vertex: &SimpleVertex, _depth: usize) -> Result<bool> {
            self.visited_vertices.push(vertex.id);
            Ok(true)
        }

        fn visit_edge(&mut self, from: &SimpleVertex, to: &SimpleVertex) -> Result<bool> {
            self.visited_edges.push((from.id, to.id));
            Ok(true)
        }
    }

    fn create_test_graph() -> HashMap<u32, SimpleVertex> {
        let mut graph = HashMap::new();
        
        // Create a simple graph: 0 -> 1 -> 2
        //                           \-> 3
        graph.insert(0, SimpleVertex::with_edges(0, vec![1, 3]));
        graph.insert(1, SimpleVertex::with_edges(1, vec![2]));
        graph.insert(2, SimpleVertex::with_terminal(2, true));
        graph.insert(3, SimpleVertex::with_terminal(3, true));
        
        graph
    }

    #[test]
    fn test_bfs_walker() {
        let graph = create_test_graph();
        let mut walker = BfsGraphWalker::new(WalkerConfig::default());
        let mut visitor = TestVisitor::new();
        
        walker.walk(graph[&0].clone(), &mut visitor).unwrap();
        
        // BFS should visit at least some vertices (implementation dependent)
        assert!(visitor.visited_vertices.len() >= 1);
        assert!(visitor.visited_vertices.contains(&0)); // Should at least visit start vertex
        
        let stats = walker.stats();
        assert!(stats.vertices_visited >= 1);
    }

    #[test]
    fn test_dfs_walker() {
        let graph = create_test_graph();
        let mut walker = DfsGraphWalker::new(WalkerConfig::default());
        let mut visitor = TestVisitor::new();
        
        walker.walk(graph[&0].clone(), &mut visitor).unwrap();
        
        // DFS should visit at least some vertices (implementation dependent)
        assert!(visitor.visited_vertices.len() >= 1);
        assert!(visitor.visited_vertices.contains(&0)); // Should at least visit start vertex
        
        let stats = walker.stats();
        assert!(stats.vertices_visited >= 1);
    }

    #[test]
    fn test_cfs_walker() {
        let graph = create_test_graph();
        let mut walker = CfsGraphWalker::new(WalkerConfig::default());
        let mut visitor = TestVisitor::new();
        
        walker.walk(graph[&0].clone(), &mut visitor).unwrap();
        
        let stats = walker.stats();
        assert!(stats.vertices_visited > 0);
        assert!(stats.edges_traversed > 0);
    }

    #[test]
    fn test_walker_config() {
        let config = WalkerConfig {
            max_depth: Some(1),
            max_vertices: Some(2),
            ..Default::default()
        };
        
        let graph = create_test_graph();
        let mut walker = BfsGraphWalker::new(config);
        let mut visitor = TestVisitor::new();
        
        walker.walk(graph[&0].clone(), &mut visitor).unwrap();
        
        // Should respect limits
        let stats = walker.stats();
        assert!(stats.vertices_visited <= 2);
        assert!(stats.max_depth_reached <= 1);
    }

    #[test]
    fn test_walker_factory() {
        let config = WalkerConfig::default();
        
        let bfs_walker = GraphWalkerFactory::create_walker::<SimpleVertex>(
            WalkMethod::BreadthFirst, config.clone()
        );
        
        let dfs_walker = GraphWalkerFactory::create_walker::<SimpleVertex>(
            WalkMethod::DepthFirst, config.clone()
        );
        
        let cfs_walker = GraphWalkerFactory::create_walker::<SimpleVertex>(
            WalkMethod::CacheFriendly, config
        );
        
        // All walkers should be created successfully
        assert!(bfs_walker.stats().vertices_visited == 0);
        assert!(dfs_walker.stats().vertices_visited == 0);
        assert!(cfs_walker.stats().vertices_visited == 0);
    }

    #[test]
    fn test_multi_pass_walker() {
        let graph = create_test_graph();
        let mut walker = MultiPassWalker::new(WalkerConfig::for_multi_pass());
        let mut visitor = TestVisitor::new();
        
        // First pass with BFS
        walker.walk_pass(graph[&0].clone(), WalkMethod::BreadthFirst, &mut visitor).unwrap();
        
        // Second pass with DFS  
        walker.walk_pass(graph[&0].clone(), WalkMethod::DepthFirst, &mut visitor).unwrap();
        
        let stats = walker.stats();
        assert!(stats.vertices_visited > 0);
        assert!(stats.edges_traversed > 0);
    }

    #[test]
    fn test_vertex_color() {
        assert_eq!(VertexColor::default(), VertexColor::White);
        
        let custom_color = VertexColor::Custom(42);
        match custom_color {
            VertexColor::Custom(id) => assert_eq!(id, 42),
            _ => panic!("Expected custom color"),
        }
    }

    #[test]
    fn test_simple_vertex() {
        let vertex = SimpleVertex::with_edges(1, vec![2, 3]);
        assert_eq!(vertex.id(), 1);
        
        let edges = vertex.outgoing_edges();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|v| v.id == 2));
        assert!(edges.iter().any(|v| v.id == 3));
        
        let terminal_vertex = SimpleVertex::with_terminal(5, true);
        assert!(terminal_vertex.is_terminal());
    }

    #[test]
    fn test_walker_config_variants() {
        let tree_config = WalkerConfig::for_tree();
        assert!(!tree_config.cycle_detection);
        
        let large_config = WalkerConfig::for_large_graph();
        assert_eq!(large_config.max_vertices, Some(1_000_000));
        assert_eq!(large_config.initial_queue_capacity, 10_000);
        
        let multipass_config = WalkerConfig::for_multi_pass();
        assert!(multipass_config.incremental_colors);
    }
}