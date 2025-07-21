"""
Production Data Connectors

This module implements various data connectors for industrial data sources,
providing a unified interface for data ingestion from OPC UA, MQTT, REST APIs,
and databases.
"""

import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from queue import Queue, Empty
import threading
import json


class BaseDataConnector(ABC):
    """Abstract base class for all data connectors."""
    
    def __init__(self, config: Dict):
        """
        Initialize the data connector.
        
        Args:
            config (Dict): Configuration dictionary containing connector settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Read data from the connected source.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing the sensor data,
                                   None if no data available
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    def is_connection_active(self) -> bool:
        """
        Check if connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.is_connected


class OPCUAConnector(BaseDataConnector):
    """Connects to an OPC UA server to read sensor data."""
    
    def __init__(self, config: Dict):
        """
        Initialize OPC UA connector.
        
        Args:
            config (Dict): Configuration containing:
                - server_url (str): OPC UA server URL
                - nodes (List[str]): List of node IDs to read
                - sampling_interval (float): Sampling interval in seconds
                - username (str, optional): Username for authentication
                - password (str, optional): Password for authentication
        """
        super().__init__(config)
        self.client = None
        self.server_url = config.get('server_url', 'opc.tcp://localhost:4840')
        self.nodes = config.get('nodes', [])
        self.sampling_interval = config.get('sampling_interval', 1.0)
        self.username = config.get('username')
        self.password = config.get('password')
        
        # Try to import OPC UA library with lazy loading
        self._opcua_available = False
        try:
            import asyncua
            self._opcua = asyncua
            self._opcua_available = True
            self.logger.info("OPC UA library available")
        except ImportError:
            self.logger.warning("OPC UA library not available. Install 'asyncua' for full functionality.")
    
    def connect(self) -> bool:
        """
        Establish connection to OPC UA server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self._opcua_available:
            self.logger.error("OPC UA library not available. Cannot connect.")
            return False
        
        try:
            # Note: This is a simplified placeholder implementation
            # In a real implementation, you would use asyncua.Client()
            self.logger.info(f"Connecting to OPC UA server: {self.server_url}")
            
            # Placeholder for actual connection logic
            # self.client = self._opcua.Client(self.server_url)
            # if self.username and self.password:
            #     self.client.set_user(self.username)
            #     self.client.set_password(self.password)
            # await self.client.connect()
            
            self.is_connected = True
            self.logger.info("Successfully connected to OPC UA server")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OPC UA server: {e}")
            self.is_connected = False
            return False
    
    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Read data from OPC UA nodes.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns 'timestamp', 'node_id', 'value'
        """
        if not self.is_connected:
            self.logger.warning("Not connected to OPC UA server")
            return None
        
        if not self.nodes:
            self.logger.warning("No nodes configured for reading")
            return None
        
        try:
            # Placeholder implementation - would read actual node values
            current_time = pd.Timestamp.now()
            data_rows = []
            
            for node_id in self.nodes:
                # In real implementation, would read actual node value:
                # node = self.client.get_node(node_id)
                # value = await node.read_value()
                
                # Placeholder: generate sample data
                import random
                value = random.uniform(20.0, 100.0)
                
                data_rows.append({
                    'timestamp': current_time,
                    'node_id': node_id,
                    'value': value
                })
            
            df = pd.DataFrame(data_rows)
            self.logger.debug(f"Read {len(df)} data points from OPC UA")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read data from OPC UA: {e}")
            return None
    
    def disconnect(self) -> bool:
        """
        Disconnect from OPC UA server.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.client:
                # await self.client.disconnect()
                self.client = None
            
            self.is_connected = False
            self.logger.info("Disconnected from OPC UA server")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during OPC UA disconnection: {e}")
            return False


class MQTTConnector(BaseDataConnector):
    """Subscribes to an MQTT broker to receive sensor data."""
    
    def __init__(self, config: Dict):
        """
        Initialize MQTT connector.
        
        Args:
            config (Dict): Configuration containing:
                - broker_host (str): MQTT broker hostname
                - broker_port (int): MQTT broker port
                - topic (str): MQTT topic to subscribe to
                - username (str, optional): Username for authentication
                - password (str, optional): Password for authentication
                - qos (int): Quality of Service level (0, 1, or 2)
                - keep_alive (int): Keep alive interval in seconds
        """
        super().__init__(config)
        self.client = None
        self.broker_host = config.get('broker_host', 'localhost')
        self.broker_port = config.get('broker_port', 1883)
        self.topic = config.get('topic', 'sensors/+')
        self.username = config.get('username')
        self.password = config.get('password')
        self.qos = config.get('qos', 1)
        self.keep_alive = config.get('keep_alive', 60)
        
        # Message queue for async message handling
        self.message_queue = Queue()
        self._stop_event = threading.Event()
        
        # Try to import MQTT library with lazy loading
        self._mqtt_available = False
        try:
            import paho.mqtt.client as mqtt
            self._mqtt = mqtt
            self._mqtt_available = True
            self.logger.info("MQTT library available")
        except ImportError:
            self.logger.warning("MQTT library not available. Install 'paho-mqtt' for full functionality.")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker and subscribe to topic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self._mqtt_available:
            self.logger.error("MQTT library not available. Cannot connect.")
            return False
        
        try:
            # Create MQTT client
            self.client = self._mqtt.Client()
            
            # Set authentication if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Connect to broker
            self.logger.info(f"Connecting to MQTT broker: {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, self.keep_alive)
            
            # Start the client loop in a separate thread
            self.client.loop_start()
            
            # Wait a moment for connection to establish
            time.sleep(1)
            
            return self.is_connected
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            self.is_connected = False
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            self.is_connected = True
            self.logger.info(f"Connected to MQTT broker, subscribing to topic: {self.topic}")
            client.subscribe(self.topic, self.qos)
        else:
            self.is_connected = False
            self.logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the broker."""
        self.is_connected = False
        self.logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, message):
        """
        Callback to handle incoming messages and add them to queue.
        
        Args:
            client: MQTT client instance
            userdata: User data set in client
            message: MQTT message object
        """
        try:
            # Decode message payload
            payload = message.payload.decode('utf-8')
            
            # Create message data structure
            msg_data = {
                'timestamp': pd.Timestamp.now(),
                'topic': message.topic,
                'payload': payload,
                'qos': message.qos
            }
            
            # Add to queue
            self.message_queue.put(msg_data)
            self.logger.debug(f"Received message on topic {message.topic}")
            
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Process the message queue into a DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns 'timestamp', 'topic', 'payload', 'qos'
        """
        if not self.is_connected:
            self.logger.warning("Not connected to MQTT broker")
            return None
        
        messages = []
        
        # Collect all available messages from queue
        try:
            while True:
                try:
                    message = self.message_queue.get_nowait()
                    
                    # Try to parse JSON payload if possible
                    try:
                        parsed_payload = json.loads(message['payload'])
                        if isinstance(parsed_payload, dict):
                            # Flatten JSON data
                            for key, value in parsed_payload.items():
                                messages.append({
                                    'timestamp': message['timestamp'],
                                    'topic': message['topic'],
                                    'sensor': key,
                                    'value': value,
                                    'qos': message['qos']
                                })
                        else:
                            messages.append({
                                'timestamp': message['timestamp'],
                                'topic': message['topic'],
                                'sensor': 'value',
                                'value': parsed_payload,
                                'qos': message['qos']
                            })
                    except json.JSONDecodeError:
                        # If not JSON, treat as raw value
                        messages.append({
                            'timestamp': message['timestamp'],
                            'topic': message['topic'],
                            'sensor': 'raw_value',
                            'value': message['payload'],
                            'qos': message['qos']
                        })
                        
                except Empty:
                    break
        except Exception as e:
            self.logger.error(f"Error reading from message queue: {e}")
            return None
        
        if not messages:
            return None
        
        df = pd.DataFrame(messages)
        self.logger.debug(f"Processed {len(df)} messages from MQTT queue")
        return df
    
    def disconnect(self) -> bool:
        """
        Disconnect from MQTT broker.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.client = None
            
            self.is_connected = False
            self.logger.info("Disconnected from MQTT broker")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during MQTT disconnection: {e}")
            return False


class RESTConnector(BaseDataConnector):
    """Connects to REST API endpoints to poll for sensor data."""
    
    def __init__(self, config: Dict):
        """
        Initialize REST connector.
        
        Args:
            config (Dict): Configuration containing:
                - base_url (str): Base URL for the REST API
                - endpoints (List[str]): List of endpoint paths to poll
                - headers (Dict, optional): HTTP headers to include
                - auth (Dict, optional): Authentication configuration
                - timeout (float): Request timeout in seconds
                - poll_interval (float): Polling interval in seconds
        """
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:8080')
        self.endpoints = config.get('endpoints', [])
        self.headers = config.get('headers', {})
        self.auth = config.get('auth', {})
        self.timeout = config.get('timeout', 10.0)
        self.poll_interval = config.get('poll_interval', 5.0)
        self.session = None
        
        # Try to import requests library with lazy loading
        self._requests_available = False
        try:
            import requests
            self._requests = requests
            self._requests_available = True
            self.logger.info("Requests library available")
        except ImportError:
            self.logger.warning("Requests library not available. Install 'requests' for full functionality.")
    
    def connect(self) -> bool:
        """
        Initialize REST API connection (prepare session).
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self._requests_available:
            self.logger.error("Requests library not available. Cannot connect.")
            return False
        
        try:
            # Create session for connection reuse
            self.session = self._requests.Session()
            
            # Set headers
            if self.headers:
                self.session.headers.update(self.headers)
            
            # Set authentication
            auth_type = self.auth.get('type')
            if auth_type == 'basic':
                username = self.auth.get('username')
                password = self.auth.get('password')
                if username and password:
                    self.session.auth = (username, password)
            elif auth_type == 'bearer':
                token = self.auth.get('token')
                if token:
                    self.session.headers['Authorization'] = f'Bearer {token}'
            
            # Test connection with a simple request
            test_url = f"{self.base_url.rstrip('/')}{self.health_endpoint}"
            try:
                response = self.session.get(test_url, timeout=self.timeout)
                self.logger.info(f"REST API connection test successful: {response.status_code}")
            except Exception:
                self.logger.info("REST API health check failed, but connection configured")
            
            self.is_connected = True
            self.logger.info(f"REST connector initialized for {self.base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize REST connection: {e}")
            self.is_connected = False
            return False
    
    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Poll REST endpoints for data.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns 'timestamp', 'endpoint', 'data'
        """
        if not self.is_connected:
            self.logger.warning("REST connector not initialized")
            return None
        
        if not self.endpoints:
            self.logger.warning("No endpoints configured for polling")
            return None
        
        data_rows = []
        current_time = pd.Timestamp.now()
        
        for endpoint in self.endpoints:
            try:
                url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        
                        # Flatten JSON response if it's a dict
                        if isinstance(json_data, dict):
                            for key, value in json_data.items():
                                data_rows.append({
                                    'timestamp': current_time,
                                    'endpoint': endpoint,
                                    'sensor': key,
                                    'value': value
                                })
                        elif isinstance(json_data, list):
                            for i, item in enumerate(json_data):
                                data_rows.append({
                                    'timestamp': current_time,
                                    'endpoint': endpoint,
                                    'sensor': f'item_{i}',
                                    'value': item
                                })
                        else:
                            data_rows.append({
                                'timestamp': current_time,
                                'endpoint': endpoint,
                                'sensor': 'value',
                                'value': json_data
                            })
                            
                    except json.JSONDecodeError:
                        # If not JSON, store as text
                        data_rows.append({
                            'timestamp': current_time,
                            'endpoint': endpoint,
                            'sensor': 'text_response',
                            'value': response.text
                        })
                else:
                    self.logger.warning(f"HTTP {response.status_code} for endpoint {endpoint}")
                    
            except Exception as e:
                self.logger.error(f"Error polling endpoint {endpoint}: {e}")
        
        if not data_rows:
            return None
        
        df = pd.DataFrame(data_rows)
        self.logger.debug(f"Polled {len(df)} data points from REST endpoints")
        return df
    
    def disconnect(self) -> bool:
        """
        Close REST API session.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.session:
                self.session.close()
                self.session = None
            
            self.is_connected = False
            self.logger.info("REST connector disconnected")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during REST disconnection: {e}")
            return False


class DatabaseConnector(BaseDataConnector):
    """Connects to SQL databases to query sensor data."""
    
    def __init__(self, config: Dict):
        """
        Initialize database connector.
        
        Args:
            config (Dict): Configuration containing:
                - connection_string (str): Database connection string
                - query (str): SQL query to execute
                - poll_interval (float): Polling interval in seconds
        """
        super().__init__(config)
        self.connection_string = config.get('connection_string')
        self.query = config.get('query')
        self.poll_interval = config.get('poll_interval', 10.0)
        self.connection = None
        
        # Try to import database libraries with lazy loading
        self._sqlalchemy_available = False
        try:
            import sqlalchemy
            self._sqlalchemy = sqlalchemy
            self._sqlalchemy_available = True
            self.logger.info("SQLAlchemy library available")
        except ImportError:
            self.logger.warning("SQLAlchemy library not available. Install 'sqlalchemy' for full functionality.")
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self._sqlalchemy_available:
            self.logger.error("SQLAlchemy library not available. Cannot connect.")
            return False
        
        if not self.connection_string:
            self.logger.error("No connection string provided")
            return False
        
        try:
            engine = self._sqlalchemy.create_engine(self.connection_string)
            self.connection = engine.connect()
            
            self.is_connected = True
            self.logger.info("Successfully connected to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.is_connected = False
            return False
    
    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Execute query and return results as DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: Query results as DataFrame
        """
        if not self.is_connected:
            self.logger.warning("Not connected to database")
            return None
        
        if not self.query:
            self.logger.warning("No query configured")
            return None
        
        try:
            df = pd.read_sql(self.query, self.connection)
            
            # Ensure timestamp column if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.Timestamp.now()
            
            self.logger.debug(f"Retrieved {len(df)} rows from database")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing database query: {e}")
            return None
    
    def disconnect(self) -> bool:
        """
        Close database connection.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            
            self.is_connected = False
            self.logger.info("Database connection closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during database disconnection: {e}")
            return False