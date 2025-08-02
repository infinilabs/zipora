#!/usr/bin/env python3
"""
Comprehensive Benchmark Analysis Tool for Rust vs C++ Performance Comparison

This script analyzes the output from the C++ comparison benchmarks and generates
detailed reports, visualizations, and performance insights.

Usage:
    python3 analyze_results.py [benchmark_results.json]
"""

import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization features disabled.")

@dataclass
class BenchmarkResult:
    """Represents a single benchmark result."""
    name: str
    language: str  # 'Rust' or 'C++'
    operation: str  # 'vector', 'string', 'hashmap', etc.
    throughput: float  # Operations per second or similar metric
    time_ns: float  # Average time in nanoseconds
    memory_usage: float = 0.0  # Memory usage in bytes
    cache_efficiency: float = 0.0  # Cache hit ratio or similar
    
class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates comprehensive reports."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = {}
        
    def load_criterion_results(self, file_path: str) -> None:
        """Load results from Criterion benchmark JSON output."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self._parse_criterion_data(data)
            print(f"Loaded {len(self.results)} benchmark results from {file_path}")
            
        except FileNotFoundError:
            print(f"Error: Benchmark results file '{file_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in '{file_path}': {e}")
            sys.exit(1)
    
    def _parse_criterion_data(self, data: Dict[str, Any]) -> None:
        """Parse Criterion JSON format and extract benchmark results."""
        # This is a simplified parser - actual Criterion format may vary
        if isinstance(data, dict) and 'benchmarks' in data:
            for bench in data['benchmarks']:
                result = self._parse_single_benchmark(bench)
                if result:
                    self.results.append(result)
        elif isinstance(data, list):
            for bench in data:
                result = self._parse_single_benchmark(bench)
                if result:
                    self.results.append(result)
    
    def _parse_single_benchmark(self, bench: Dict[str, Any]) -> BenchmarkResult:
        """Parse a single benchmark entry."""
        try:
            name = bench.get('name', 'Unknown')
            
            # Determine language and operation from name
            if 'Rust' in name or 'FastVec' in name or 'FastStr' in name or 'GoldHashMap' in name:
                language = 'Rust'
            elif 'C++' in name or 'valvec' in name or 'fstring' in name:
                language = 'C++'
            else:
                language = 'Unknown'
            
            # Extract operation type
            operation = 'unknown'
            if 'vector' in name.lower() or 'vec' in name.lower():
                operation = 'vector'
            elif 'string' in name.lower() or 'str' in name.lower():
                operation = 'string'
            elif 'hash' in name.lower() or 'map' in name.lower():
                operation = 'hashmap'
            elif 'succinct' in name.lower() or 'bitvector' in name.lower() or 'rank' in name.lower():
                operation = 'succinct'
            elif 'memory' in name.lower() or 'allocation' in name.lower():
                operation = 'memory'
            elif 'cache' in name.lower():
                operation = 'cache'
            
            # Extract timing information
            mean_time = bench.get('mean', {}).get('estimate', 0)
            if isinstance(mean_time, (int, float)):
                time_ns = float(mean_time)
            else:
                time_ns = 0.0
            
            # Calculate throughput (operations per second)
            if time_ns > 0:
                throughput = 1e9 / time_ns  # Convert from nanoseconds to operations per second
            else:
                throughput = 0.0
            
            return BenchmarkResult(
                name=name,
                language=language,
                operation=operation,
                throughput=throughput,
                time_ns=time_ns
            )
            
        except Exception as e:
            print(f"Warning: Could not parse benchmark '{bench}': {e}")
            return None
    
    def group_by_operation(self) -> Dict[str, List[BenchmarkResult]]:
        """Group benchmark results by operation type."""
        groups = {}
        for result in self.results:
            if result.operation not in groups:
                groups[result.operation] = []
            groups[result.operation].append(result)
        return groups
    
    def compare_languages(self) -> Dict[str, Dict[str, float]]:
        """Compare Rust vs C++ performance across different operations."""
        comparisons = {}
        groups = self.group_by_operation()
        
        for operation, results in groups.items():
            rust_results = [r for r in results if r.language == 'Rust']
            cpp_results = [r for r in results if r.language == 'C++']
            
            if rust_results and cpp_results:
                rust_avg = statistics.mean([r.throughput for r in rust_results])
                cpp_avg = statistics.mean([r.throughput for r in cpp_results])
                
                if cpp_avg > 0:
                    rust_vs_cpp_ratio = rust_avg / cpp_avg
                else:
                    rust_vs_cpp_ratio = float('inf') if rust_avg > 0 else 1.0
                
                comparisons[operation] = {
                    'rust_throughput': rust_avg,
                    'cpp_throughput': cpp_avg,
                    'rust_vs_cpp_ratio': rust_vs_cpp_ratio,
                    'rust_faster': rust_vs_cpp_ratio > 1.0
                }
        
        return comparisons
    
    def generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE RUST VS C++ PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total benchmarks analyzed: {len(self.results)}")
        report.append("")
        
        # Summary by operation
        comparisons = self.compare_languages()
        if comparisons:
            report.append("PERFORMANCE COMPARISON SUMMARY")
            report.append("-" * 40)
            report.append(f"{'Operation':<15} {'Rust Wins':<12} {'Ratio':<12} {'Performance Gap'}")
            report.append("-" * 60)
            
            overall_ratios = []
            for operation, comp in comparisons.items():
                ratio = comp['rust_vs_cpp_ratio']
                wins = "✓ YES" if comp['rust_faster'] else "✗ NO"
                
                if ratio == float('inf'):
                    ratio_str = "∞"
                    gap_str = "Rust only"
                elif ratio > 1:
                    ratio_str = f"{ratio:.2f}x"
                    gap_str = f"{((ratio - 1) * 100):.1f}% faster"
                else:
                    ratio_str = f"{1/ratio:.2f}x"
                    gap_str = f"{((1/ratio - 1) * 100):.1f}% slower"
                
                report.append(f"{operation:<15} {wins:<12} {ratio_str:<12} {gap_str}")
                
                if ratio != float('inf'):
                    overall_ratios.append(ratio)
            
            if overall_ratios:
                geometric_mean = statistics.geometric_mean(overall_ratios)
                report.append("-" * 60)
                if geometric_mean > 1:
                    report.append(f"OVERALL: Rust is {geometric_mean:.2f}x faster on average")
                else:
                    report.append(f"OVERALL: C++ is {1/geometric_mean:.2f}x faster on average")
        
        report.append("")
        
        # Detailed breakdown by operation
        groups = self.group_by_operation()
        for operation, results in groups.items():
            report.append(f"DETAILED ANALYSIS: {operation.upper()} OPERATIONS")
            report.append("-" * 50)
            
            rust_results = [r for r in results if r.language == 'Rust']
            cpp_results = [r for r in results if r.language == 'C++']
            
            if rust_results:
                report.append(f"Rust {operation} operations:")
                for result in rust_results:
                    report.append(f"  • {result.name}: {result.throughput:.0f} ops/sec")
            
            if cpp_results:
                report.append(f"C++ {operation} operations:")
                for result in cpp_results:
                    report.append(f"  • {result.name}: {result.throughput:.0f} ops/sec")
            
            report.append("")
        
        # Performance insights and recommendations
        report.append("PERFORMANCE INSIGHTS AND RECOMMENDATIONS")
        report.append("-" * 50)
        
        if comparisons:
            rust_wins = sum(1 for comp in comparisons.values() if comp['rust_faster'])
            total_operations = len(comparisons)
            
            report.append(f"• Rust wins in {rust_wins}/{total_operations} operation categories")
            
            if rust_wins > total_operations / 2:
                report.append("• Overall: Rust implementation shows superior performance")
                report.append("• Recommendation: Consider Rust for performance-critical applications")
            else:
                report.append("• Overall: C++ implementation shows competitive performance")
                report.append("• Recommendation: Both implementations are viable options")
            
            # Identify strongest and weakest areas
            best_rust = max(comparisons.items(), key=lambda x: x[1]['rust_vs_cpp_ratio'])
            worst_rust = min(comparisons.items(), key=lambda x: x[1]['rust_vs_cpp_ratio'])
            
            report.append(f"• Rust's strongest area: {best_rust[0]} operations")
            report.append(f"• Rust's improvement opportunity: {worst_rust[0]} operations")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None) -> None:
        """Save the text report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.txt"
        
        report = self.generate_text_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {filename}")
    
    def create_visualization(self, save_path: str = None) -> None:
        """Create performance comparison visualizations."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping visualization.")
            return
        
        comparisons = self.compare_languages()
        if not comparisons:
            print("No comparison data available for visualization.")
            return
        
        # Create subplots for different metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rust vs C++ Performance Comparison', fontsize=16, fontweight='bold')
        
        operations = list(comparisons.keys())
        rust_throughputs = [comparisons[op]['rust_throughput'] for op in operations]
        cpp_throughputs = [comparisons[op]['cpp_throughput'] for op in operations]
        ratios = [comparisons[op]['rust_vs_cpp_ratio'] for op in operations]
        
        # 1. Throughput comparison (bar chart)
        x = np.arange(len(operations))
        width = 0.35
        
        ax1.bar(x - width/2, rust_throughputs, width, label='Rust', color='#d67441', alpha=0.8)
        ax1.bar(x + width/2, cpp_throughputs, width, label='C++', color='#0077be', alpha=0.8)
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('Throughput (ops/sec)')
        ax1.set_title('Throughput Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(operations, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance ratio (ratio chart)
        colors = ['#d67441' if ratio > 1 else '#0077be' for ratio in ratios]
        bars = ax2.bar(operations, ratios, color=colors, alpha=0.8)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Performance Ratio (Rust/C++)')
        ax2.set_title('Performance Ratio (>1 = Rust Faster)')
        ax2.set_xticklabels(operations, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add ratio labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory efficiency (if available)
        groups = self.group_by_operation()
        memory_rust = []
        memory_cpp = []
        memory_ops = []
        
        for op, results in groups.items():
            rust_mem = [r.memory_usage for r in results if r.language == 'Rust' and r.memory_usage > 0]
            cpp_mem = [r.memory_usage for r in results if r.language == 'C++' and r.memory_usage > 0]
            
            if rust_mem and cpp_mem:
                memory_ops.append(op)
                memory_rust.append(statistics.mean(rust_mem))
                memory_cpp.append(statistics.mean(cpp_mem))
        
        if memory_ops:
            x_mem = np.arange(len(memory_ops))
            ax3.bar(x_mem - width/2, memory_rust, width, label='Rust', color='#d67441', alpha=0.8)
            ax3.bar(x_mem + width/2, memory_cpp, width, label='C++', color='#0077be', alpha=0.8)
            ax3.set_xlabel('Operations')
            ax3.set_ylabel('Memory Usage (bytes)')
            ax3.set_title('Memory Usage Comparison')
            ax3.set_xticks(x_mem)
            ax3.set_xticklabels(memory_ops, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No memory usage data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Memory Usage Comparison')
        
        # 4. Overall performance summary (pie chart)
        rust_wins = sum(1 for comp in comparisons.values() if comp['rust_faster'])
        cpp_wins = len(comparisons) - rust_wins
        
        if rust_wins > 0 or cpp_wins > 0:
            labels = []
            sizes = []
            colors = []
            
            if rust_wins > 0:
                labels.append(f'Rust Wins ({rust_wins})')
                sizes.append(rust_wins)
                colors.append('#d67441')
            
            if cpp_wins > 0:
                labels.append(f'C++ Wins ({cpp_wins})')
                sizes.append(cpp_wins)
                colors.append('#0077be')
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Performance Victory Distribution')
        else:
            ax4.text(0.5, 0.5, 'No comparison data available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Performance Victory Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"benchmark_comparison_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze Rust vs C++ benchmark results')
    parser.add_argument('file', nargs='?', default='benchmark_results.json',
                       help='Benchmark results file (JSON format)')
    parser.add_argument('--report', '-r', metavar='FILE',
                       help='Save text report to specified file')
    parser.add_argument('--visualize', '-v', metavar='FILE',
                       help='Create and save visualization to specified file')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display visualizations (only save)')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer()
    
    # Check if results file exists
    if not Path(args.file).exists():
        print(f"Benchmark results file '{args.file}' not found.")
        print("Please run the benchmarks first:")
        print("  cargo bench --bench cpp_comparison")
        print("  cargo bench --bench cpp_comparison -- --output-format=json > benchmark_results.json")
        sys.exit(1)
    
    analyzer.load_criterion_results(args.file)
    
    # Generate and display text report
    print(analyzer.generate_text_report())
    
    # Save report if requested
    if args.report:
        analyzer.save_report(args.report)
    
    # Create visualization if requested or by default
    if args.visualize or not args.no_display:
        if HAS_MATPLOTLIB:
            analyzer.create_visualization(args.visualize)
        else:
            print("Install matplotlib for visualization features: pip install matplotlib numpy")

if __name__ == '__main__':
    main()