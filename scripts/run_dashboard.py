#!/usr/bin/env python3
"""
Dashboard Application Launcher for Steel Defect Monitoring System

This script serves as the production-ready entry point for starting the visualization
dashboard with comprehensive configuration options, multi-environment support,
and integration with the inference engine.

Author: Steel Defect Monitoring Team
"""

import argparse
import logging
import signal
import socket
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import queue
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.dashboard import DefectMonitoringDashboard
from src.inference.inference_engine import DefectPredictionEngine
from src.inference.stream_simulator import RealTimeStreamSimulator
import pandas as pd


class DashboardLauncher:
    """
    Production-ready launcher for the Steel Defect Monitoring Dashboard.
    
    Handles configuration, environment setup, data streaming, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the dashboard launcher."""
        self.dashboard = None
        self.inference_engine = None
        self.stream_simulator = None
        self.data_thread = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.config = {}
        self.logger = self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging configuration."""
        logs_path = Path('logs')
        if not logs_path.exists():
            logs_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(logs_path / 'dashboard.log', mode='a')
            ]
        )
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'dashboard': {
                'refresh_interval': 5000,
                'theme': 'plotly_white',
                'risk_colors': {
                    'safe': '#2E8B57',
                    'warning': '#FFD700',
                    'high_risk': '#FF6B35',
                    'alert': '#DC143C'
                }
            },
            'inference': {
                'model_types': ['baseline', 'lstm'],
                'ensemble': {
                    'baseline_weight': 0.4,
                    'lstm_weight': 0.6
                },
                'real_time_simulation': {
                    'playback_speed_multiplier': 10,
                    'update_interval_seconds': 30,
                    'buffer_size_seconds': 300,
                    'data_interval_seconds': 1.0
                },
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            }
        }
    
    def check_port_available(self, host: str, port: int) -> bool:
        """
        Check if the specified port is available.
        
        Args:
            host (str): Host address
            port (int): Port number
            
        Returns:
            bool: True if port is available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception as e:
            self.logger.error(f"Error checking port availability: {e}")
            return False
    
    def find_available_port(self, host: str, start_port: int, max_attempts: int = 10) -> Optional[int]:
        """
        Find the next available port starting from start_port.
        
        Args:
            host (str): Host address
            start_port (int): Starting port number
            max_attempts (int): Maximum number of ports to try
            
        Returns:
            Optional[int]: Available port number or None if none found
        """
        for i in range(max_attempts):
            port = start_port + i
            if self.check_port_available(host, port):
                return port
        return None
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock data for testing purposes."""
        import numpy as np
        
        # Generate realistic casting data
        n_points = 100
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='30s')
        
        # Simulate temperature with some realistic variation
        base_temp = 1450
        temp_variation = np.random.normal(0, 50, n_points)
        temperatures = base_temp + temp_variation
        
        # Add some other sensor readings
        pressure = 150 + np.random.normal(0, 10, n_points)
        flow_rate = 2.5 + np.random.normal(0, 0.3, n_points)
        
        mock_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'defect_probability': np.random.random(n_points)
        })
        
        self.logger.info(f"Generated mock data with {len(mock_data)} points")
        return mock_data
    
    def _background_data_fetcher(self):
        """Background thread for fetching data from inference engine."""
        self.logger.info("Starting background data fetching thread")
        
        while self.running:
            try:
                if self.stream_simulator and hasattr(self.stream_simulator, 'get_next_data'):
                    # Get data from stream simulator
                    data = self.stream_simulator.get_next_data()
                    if data is not None:
                        # Add to queue for dashboard consumption
                        if not self.data_queue.full():
                            self.data_queue.put(data)
                        else:
                            # Remove oldest data if queue is full
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put(data)
                            except queue.Empty:
                                pass
                
                # Sleep based on configuration
                sleep_interval = self.config.get('inference', {}).get(
                    'real_time_simulation', {}
                ).get('data_interval_seconds', 1.0)
                time.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background data fetcher: {e}")
                time.sleep(5)  # Wait before retrying
    
    def setup_inference_engine(self, mock_data: bool = False):
        """
        Setup the inference engine and data streaming.
        
        Args:
            mock_data (bool): Whether to use mock data instead of real inference
        """
        try:
            if mock_data:
                self.logger.info("Setting up inference engine with mock data")
                # Create mock data for simulation
                cast_data = self._generate_mock_data()
                
                # Initialize inference engine with default config
                self.inference_engine = DefectPredictionEngine()
                
                # Create stream simulator with mock data
                self.stream_simulator = RealTimeStreamSimulator(
                    cast_data=cast_data,
                    config=self.config,
                    inference_engine=self.inference_engine
                )
            else:
                self.logger.info("Setting up inference engine with real data")
                # Load inference configuration
                inference_config_path = Path("configs/inference_config.yaml")
                if inference_config_path.exists():
                    self.inference_engine = DefectPredictionEngine(str(inference_config_path))
                else:
                    self.inference_engine = DefectPredictionEngine()
                
                # For real data, we would initialize with actual data source
                # For now, using mock data as fallback
                cast_data = self._generate_mock_data()
                self.stream_simulator = RealTimeStreamSimulator(
                    cast_data=cast_data,
                    config=self.config,
                    inference_engine=self.inference_engine
                )
            
            self.logger.info("Inference engine and stream simulator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up inference engine: {e}")
            raise
    
    def start_background_tasks(self):
        """Start background data fetching tasks."""
        if self.data_thread is None or not self.data_thread.is_alive():
            self.running = True
            self.data_thread = threading.Thread(target=self._background_data_fetcher, daemon=True)
            self.data_thread.start()
            self.logger.info("Background data fetching started")
    
    def run_development_mode(self, host: str, port: int, debug: bool = True):
        """
        Run dashboard in development mode with hot-reloading.
        
        Args:
            host (str): Host address
            port (int): Port number
            debug (bool): Enable debug mode
        """
        self.logger.info(f"Starting dashboard in DEVELOPMENT mode on {host}:{port}")
        self.logger.info("Features enabled: hot-reloading, debug mode, detailed error pages")
        
        try:
            self.dashboard.app.run(
                host=host,
                port=port,
                debug=debug,
                dev_tools_hot_reload=debug,
                dev_tools_hot_reload_interval=1000,
                dev_tools_hot_reload_max_retry=5
            )
        except Exception as e:
            self.logger.error(f"Error running development server: {e}")
            raise
    
    def run_production_mode(self, host: str, port: int):
        """
        Run dashboard in production mode using a production WSGI server.
        
        Args:
            host (str): Host address
            port (int): Port number
        """
        self.logger.info(f"Starting dashboard in PRODUCTION mode on {host}:{port}")
        self.logger.info("Features: optimized for stability and performance")
        
        try:
            # Try to use gunicorn if available, otherwise fall back to Flask's built-in server
            try:
                import gunicorn.app.base
                
                class StandaloneApplication(gunicorn.app.base.BaseApplication):
                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.application = app
                        super().__init__()
                    
                    def load_config(self):
                        config = {key: value for key, value in self.options.items()
                                if key in self.cfg.settings and value is not None}
                        for key, value in config.items():
                            self.cfg.set(key.lower(), value)
                    
                    def load(self):
                        return self.application
                
                # Gunicorn configuration for production
                options = {
                    'bind': f'{host}:{port}',
                    'workers': 4,
                    'worker_class': 'sync',
                    'timeout': 120,
                    'keepalive': 5,
                    'max_requests': 1000,
                    'max_requests_jitter': 100,
                    'preload_app': True,
                    'access_logfile': '-',
                    'error_logfile': '-',
                }
                
                StandaloneApplication(self.dashboard.app.server, options).run()
                
            except ImportError:
                self.logger.warning("Gunicorn not available. Using Flask development server in production mode.")
                # Fallback to Flask with production settings
                self.dashboard.app.run(
                    host=host,
                    port=port,
                    debug=False,
                    dev_tools_hot_reload=False,
                    threaded=True
                )
                
        except Exception as e:
            self.logger.error(f"Error running production server: {e}")
            raise
    
    def setup_dashboard(self):
        """Setup the main dashboard application."""
        try:
            self.dashboard = DefectMonitoringDashboard(self.config.get('dashboard', {}))
            self.logger.info("Dashboard initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up dashboard: {e}")
            raise
    
    def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Starting graceful shutdown...")
        
        # Stop background tasks
        self.running = False
        
        if self.data_thread and self.data_thread.is_alive():
            self.logger.info("Stopping background data thread...")
            self.data_thread.join(timeout=5)
            if self.data_thread.is_alive():
                self.logger.warning("Background thread did not stop gracefully")
        
        # Stop stream simulator
        if self.stream_simulator and hasattr(self.stream_simulator, 'stop'):
            try:
                self.stream_simulator.stop()
                self.logger.info("Stream simulator stopped")
            except Exception as e:
                self.logger.error(f"Error stopping stream simulator: {e}")
        
        # Clean up inference engine
        if self.inference_engine:
            try:
                # Any cleanup needed for inference engine
                self.logger.info("Inference engine cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up inference engine: {e}")
        
        self.logger.info("Graceful shutdown completed")
        sys.exit(0)
    
    def run(self, args):
        """
        Main run method that orchestrates the dashboard startup.
        
        Args:
            args: Parsed command line arguments
        """
        try:
            # Load configuration
            if args.config:
                self.config = self.load_config(args.config)
            else:
                self.config = self._get_default_config()
            
            # Check and handle port availability
            if not self.check_port_available(args.host, args.port):
                self.logger.warning(f"Port {args.port} is already in use")
                if args.production:
                    self.logger.error("Cannot start in production mode with occupied port")
                    sys.exit(1)
                else:
                    # Try to find alternative port in development mode
                    available_port = self.find_available_port(args.host, args.port)
                    if available_port:
                        self.logger.info(f"Using alternative port {available_port}")
                        args.port = available_port
                    else:
                        self.logger.error("No available ports found")
                        sys.exit(1)
            
            # Setup inference engine
            self.setup_inference_engine(mock_data=args.mock_data)
            
            # Setup dashboard
            self.setup_dashboard()
            
            # Start background tasks
            self.start_background_tasks()
            
            # Log startup information
            self.logger.info("=" * 60)
            self.logger.info("🚀 Steel Defect Monitoring Dashboard")
            self.logger.info("=" * 60)
            self.logger.info(f"Mode: {'PRODUCTION' if args.production else 'DEVELOPMENT'}")
            self.logger.info(f"Host: {args.host}")
            self.logger.info(f"Port: {args.port}")
            self.logger.info(f"Debug: {args.debug}")
            self.logger.info(f"Mock Data: {args.mock_data}")
            self.logger.info(f"Dashboard URL: http://{args.host}:{args.port}")
            self.logger.info("=" * 60)
            
            # Run appropriate server mode
            if args.production:
                self.run_production_mode(args.host, args.port)
            else:
                self.run_development_mode(args.host, args.port, args.debug)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.shutdown()
            sys.exit(1)


def main():
    """Main entry point for the dashboard launcher."""
    parser = argparse.ArgumentParser(
        description="Steel Defect Monitoring Dashboard Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development mode with debug
  python scripts/run_dashboard.py --debug

  # Production mode with custom port
  python scripts/run_dashboard.py --production --port 8080

  # Custom configuration with mock data
  python scripts/run_dashboard.py --config configs/custom.yaml --mock-data

  # Bind to all interfaces for external access
  python scripts/run_dashboard.py --host 0.0.0.0 --port 8050
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom YAML configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the dashboard on (default: 8050)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Run in production mode with optimized settings'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with hot-reloading (development only)'
    )
    
    parser.add_argument(
        '--mock-data',
        action='store_true',
        help='Use simulated data for frontend development and testing'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.production and args.debug:
        print("Error: --production and --debug flags are mutually exclusive")
        sys.exit(1)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create and run launcher
    launcher = DashboardLauncher()
    launcher.run(args)


if __name__ == "__main__":
    main()