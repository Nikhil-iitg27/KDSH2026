"""
Real-time GPU VRAM usage monitor with live plotting.
Monitors VRAM usage every 3 seconds and displays in a sliding window plot.
"""

import subprocess
import re
import time
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime


class GPUMonitor:
    """Monitor GPU VRAM usage and plot in real-time."""
    
    def __init__(self, window_size=10, update_interval=3, output_file="gpu_monitor.png"):
        """
        Initialize GPU monitor.
        
        Args:
            window_size: Number of data points to show (sliding window)
            update_interval: Seconds between updates
            output_file: Path to save the plot image
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.vram_data = deque(maxlen=window_size)
        self.timesteps = deque(maxlen=window_size)
        self.current_step = 0
        self.output_file = output_file
        
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=6)
        
        self.ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('VRAM Usage (GB)', fontsize=12, fontweight='bold')
        self.ax.set_title('Real-time GPU VRAM Usage Monitor', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def get_vram_usage(self):
        """Get current VRAM usage from nvidia-smi."""
        try:
            # Run nvidia-smi to get memory usage
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Parse output (in MB)
                vram_mb = float(result.stdout.strip().split('\n')[0])
                vram_gb = vram_mb / 1024.0
                return vram_gb
            else:
                print(f"Error running nvidia-smi: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("nvidia-smi command timed out")
            return None
        except FileNotFoundError:
            print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
            return None
        except Exception as e:
            print(f"Error getting VRAM usage: {e}")
            return None
    
    def update_plot(self):
        """Update the plot with current data."""
        if len(self.vram_data) > 0:
            self.line.set_data(list(self.timesteps), list(self.vram_data))
            
            # Update axis limits
            if len(self.timesteps) > 0:
                x_min = min(self.timesteps) - 0.5
                x_max = max(self.timesteps) + 0.5
                self.ax.set_xlim(x_min, x_max)
            
            if len(self.vram_data) > 0:
                y_min = 0
                y_max = max(self.vram_data) * 1.2  # 20% headroom
                if y_max < 1:
                    y_max = 1
                self.ax.set_ylim(y_min, y_max)
            
            # Update title with current stats
            if len(self.vram_data) > 0:
                current_vram = self.vram_data[-1]
                avg_vram = sum(self.vram_data) / len(self.vram_data)
                max_vram = max(self.vram_data)
                
                self.ax.set_title(
                    f'Real-time GPU VRAM Usage Monitor\n'
                    f'Current: {current_vram:.2f} GB | Avg: {avg_vram:.2f} GB | Max: {max_vram:.2f} GB',
                    fontsize=12,
                    fontweight='bold'
                )
            
            # Save plot to file
            self.fig.savefig(self.output_file, dpi=100, bbox_inches='tight')
            print(f"  → Plot saved to {self.output_file}")
    
    def run(self):
        """Main monitoring loop."""
        print("="*80)
        print("GPU VRAM USAGE MONITOR")
        print("="*80)
        print(f"Window size: {self.window_size} data points")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Output file: {os.path.abspath(self.output_file)}")
        print("Press Ctrl+C to stop")
        print("="*80)
        
        try:
            while True:
                # Get VRAM usage
                vram_gb = self.get_vram_usage()
                
                if vram_gb is not None:
                    # Add to data
                    self.vram_data.append(vram_gb)
                    self.timesteps.append(self.current_step)
                    
                    # Print to console
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Step {self.current_step}: {vram_gb:.2f} GB VRAM")
                    
                    # Update plot
                    self.update_plot()
                    
                    self.current_step += 1
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to get VRAM data")
                
                # Wait before next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n" + "="*80)
            print("Monitoring stopped by user")
            print("="*80)
            
            if len(self.vram_data) > 0:
                print(f"\nStatistics:")
                print(f"  Total samples: {len(self.vram_data)}")
                print(f"  Average VRAM: {sum(self.vram_data)/len(self.vram_data):.2f} GB")
                print(f"  Min VRAM: {min(self.vram_data):.2f} GB")
                print(f"  Max VRAM: {max(self.vram_data):.2f} GB")
                print(f"\nFinal plot saved to: {os.path.abspath(self.output_file)}")
            
            plt.close(self.fig)


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "gpu_monitor.png")
    monitor = GPUMonitor(window_size=10, update_interval=3, output_file=output_path)
    monitor.run()
