import unittest
import sys
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from queue import Queue

# Import connectors from src directory
from ..src.connectors.data_connectors import (
    BaseDataConnector,
    OPCUAConnector,
    MQTTConnector,
    RESTConnector,
    DatabaseConnector
)


class TestBaseDataConnector(unittest.TestCase):
    """Test cases for BaseDataConnector"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a concrete implementation for testing
        class ConcreteConnector(BaseDataConnector):
            def connect(self):
                self.is_connected = True
                return True
            
            def read_data(self):
                return pd.DataFrame({'test': [1, 2, 3]})
            
            def disconnect(self):
                self.is_connected = False
                return True
        
        self.config = {'test_param': 'test_value'}
        self.connector = ConcreteConnector(self.config)
    
    def test_initialization(self):
        """Test connector initialization"""
        self.assertEqual(self.connector.config, self.config)
        self.assertFalse(self.connector.is_connected)
    
    def test_connection_status(self):
        """Test connection status tracking"""
        self.assertFalse(self.connector.is_connection_active())
        
        self.connector.connect()
        self.assertTrue(self.connector.is_connection_active())
        
        self.connector.disconnect()
        self.assertFalse(self.connector.is_connection_active())


class TestOPCUAConnector(unittest.TestCase):
    """Test cases for OPCUAConnector"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'server_url': 'opc.tcp://test-server:4840',
            'nodes': ['ns=2;i=1001', 'ns=2;i=1002'],
            'sampling_interval': 1.0,
            'username': 'test_user',
            'password': 'test_pass'
        }
        self.connector = OPCUAConnector(self.config)
    
    def test_initialization(self):
        """Test OPC UA connector initialization"""
        self.assertEqual(self.connector.server_url, 'opc.tcp://test-server:4840')
        self.assertEqual(self.connector.nodes, ['ns=2;i=1001', 'ns=2;i=1002'])
        self.assertEqual(self.connector.sampling_interval, 1.0)
        self.assertEqual(self.connector.username, 'test_user')
        self.assertEqual(self.connector.password, 'test_pass')
        self.assertIsNone(self.connector.client)
    
    def test_library_availability_check(self):
        """Test OPC UA library availability check"""
        # The connector should handle missing library gracefully
        self.assertIsInstance(self.connector._opcua_available, bool)
    
    def test_connect_without_library(self):
        """Test connection attempt without OPC UA library"""
        # Mock library as unavailable
        self.connector._opcua_available = False
        
        result = self.connector.connect()
        self.assertFalse(result)
        self.assertFalse(self.connector.is_connected)
    
    def test_connect_success(self):
        """Test successful OPC UA connection"""
        # Mock library as available
        self.connector._opcua_available = True
        
        result = self.connector.connect()
        self.assertTrue(result)
        self.assertTrue(self.connector.is_connected)
    
    def test_read_data_not_connected(self):
        """Test reading data when not connected"""
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_no_nodes(self):
        """Test reading data with no nodes configured"""
        self.connector.is_connected = True
        self.connector.nodes = []
        
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    @patch('random.uniform')
    def test_read_data_success(self, mock_random):
        """Test successful data reading"""
        mock_random.return_value = 50.0
        self.connector.is_connected = True
        
        result = self.connector.read_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two nodes configured
        self.assertIn('timestamp', result.columns)
        self.assertIn('node_id', result.columns)
        self.assertIn('value', result.columns)
        self.assertEqual(result['value'].iloc[0], 50.0)
    
    def test_disconnect(self):
        """Test disconnection"""
        self.connector.is_connected = True
        
        result = self.connector.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.connector.is_connected)


class TestMQTTConnector(unittest.TestCase):
    """Test cases for MQTTConnector"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'broker_host': 'test-broker',
            'broker_port': 1883,
            'topic': 'sensors/temperature',
            'username': 'test_user',
            'password': 'test_pass',
            'qos': 1,
            'keep_alive': 60
        }
        self.connector = MQTTConnector(self.config)
    
    def test_initialization(self):
        """Test MQTT connector initialization"""
        self.assertEqual(self.connector.broker_host, 'test-broker')
        self.assertEqual(self.connector.broker_port, 1883)
        self.assertEqual(self.connector.topic, 'sensors/temperature')
        self.assertEqual(self.connector.username, 'test_user')
        self.assertEqual(self.connector.password, 'test_pass')
        self.assertEqual(self.connector.qos, 1)
        self.assertEqual(self.connector.keep_alive, 60)
        self.assertIsInstance(self.connector.message_queue, Queue)
    
    def test_library_availability_check(self):
        """Test MQTT library availability check"""
        self.assertIsInstance(self.connector._mqtt_available, bool)
    
    def test_connect_without_library(self):
        """Test connection attempt without MQTT library"""
        self.connector._mqtt_available = False
        
        result = self.connector.connect()
        self.assertFalse(result)
        self.assertFalse(self.connector.is_connected)
    
    @patch('time.sleep')
    def test_connect_success(self, mock_sleep):
        """Test successful MQTT connection"""
        # Mock MQTT client and set library as available
        self.connector._mqtt_available = True
        
        mock_client = Mock()
        self.connector._mqtt = Mock()
        self.connector._mqtt.Client.return_value = mock_client
        
        # Mock successful connection
        def mock_connect(*args):
            self.connector._on_connect(mock_client, None, None, 0)
        
        mock_client.connect.side_effect = mock_connect
        
        result = self.connector.connect()
        self.assertTrue(result)
        self.assertTrue(self.connector.is_connected)
        mock_client.loop_start.assert_called_once()
    
    def test_on_connect_callback(self):
        """Test MQTT on_connect callback"""
        mock_client = Mock()
        
        # Test successful connection (rc=0)
        self.connector._on_connect(mock_client, None, None, 0)
        self.assertTrue(self.connector.is_connected)
        mock_client.subscribe.assert_called_once_with('sensors/temperature', 1)
        
        # Test failed connection (rc!=0)
        self.connector._on_connect(mock_client, None, None, 1)
        self.assertFalse(self.connector.is_connected)
    
    def test_on_disconnect_callback(self):
        """Test MQTT on_disconnect callback"""
        self.connector.is_connected = True
        self.connector._on_disconnect(None, None, None)
        self.assertFalse(self.connector.is_connected)
    
    def test_on_message_callback(self):
        """Test MQTT on_message callback"""
        # Create mock message
        mock_message = Mock()
        mock_message.payload.decode.return_value = '{"temperature": 25.5, "humidity": 60.0}'
        mock_message.topic = 'sensors/temperature'
        mock_message.qos = 1
        
        # Test message processing
        self.connector._on_message(None, None, mock_message)
        
        # Check that message was added to queue
        self.assertFalse(self.connector.message_queue.empty())
        
        # Get message from queue and verify
        msg = self.connector.message_queue.get()
        self.assertEqual(msg['topic'], 'sensors/temperature')
        self.assertEqual(msg['qos'], 1)
        self.assertIn('timestamp', msg)
    
    def test_on_message_callback_invalid_json(self):
        """Test MQTT on_message callback with invalid JSON"""
        mock_message = Mock()
        mock_message.payload.decode.return_value = 'not json data'
        mock_message.topic = 'sensors/temperature'
        mock_message.qos = 1
        
        self.connector._on_message(None, None, mock_message)
        
        # Should still add message to queue
        self.assertFalse(self.connector.message_queue.empty())
    
    def test_read_data_not_connected(self):
        """Test reading data when not connected"""
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_empty_queue(self):
        """Test reading data with empty message queue"""
        self.connector.is_connected = True
        
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_with_messages(self):
        """Test reading data with messages in queue"""
        self.connector.is_connected = True
        
        # Add test messages to queue
        test_messages = [
            {
                'timestamp': pd.Timestamp.now(),
                'topic': 'sensors/temp1',
                'payload': '{"temperature": 25.5}',
                'qos': 1
            },
            {
                'timestamp': pd.Timestamp.now(),
                'topic': 'sensors/temp2',
                'payload': '{"temperature": 30.0}',
                'qos': 1
            }
        ]
        
        for msg in test_messages:
            self.connector.message_queue.put(msg)
        
        result = self.connector.read_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('timestamp', result.columns)
        self.assertIn('topic', result.columns)
        self.assertIn('sensor', result.columns)
        self.assertIn('value', result.columns)
    
    def test_disconnect(self):
        """Test disconnection"""
        mock_client = Mock()
        self.connector.client = mock_client
        self.connector.is_connected = True
        
        result = self.connector.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.connector.is_connected)
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()


class TestRESTConnector(unittest.TestCase):
    """Test cases for RESTConnector"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'base_url': 'http://test-api.com',
            'endpoints': ['sensors/temperature', 'sensors/pressure'],
            'headers': {'Authorization': 'Bearer token'},
            'auth': {
                'type': 'bearer',
                'token': 'test_token'
            },
            'timeout': 10.0,
            'poll_interval': 5.0
        }
        self.connector = RESTConnector(self.config)
    
    def test_initialization(self):
        """Test REST connector initialization"""
        self.assertEqual(self.connector.base_url, 'http://test-api.com')
        self.assertEqual(self.connector.endpoints, ['sensors/temperature', 'sensors/pressure'])
        self.assertEqual(self.connector.headers, {'Authorization': 'Bearer token'})
        self.assertEqual(self.connector.timeout, 10.0)
        self.assertEqual(self.connector.poll_interval, 5.0)
    
    def test_library_availability_check(self):
        """Test requests library availability check"""
        self.assertIsInstance(self.connector._requests_available, bool)
    
    def test_connect_without_library(self):
        """Test connection attempt without requests library"""
        self.connector._requests_available = False
        
        result = self.connector.connect()
        self.assertFalse(result)
        self.assertFalse(self.connector.is_connected)
    
    def test_connect_success(self):
        """Test successful REST connection"""
        # Mock library as available
        self.connector._requests_available = True
        
        # Mock requests
        mock_session = Mock()
        mock_session.headers = {}  # Make headers behave like a dict
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        
        self.connector._requests = Mock()
        self.connector._requests.Session.return_value = mock_session
        
        result = self.connector.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.connector.is_connected)
        self.assertIsNotNone(self.connector.session)
    
    def test_read_data_not_connected(self):
        """Test reading data when not connected"""
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_no_endpoints(self):
        """Test reading data with no endpoints configured"""
        self.connector.is_connected = True
        self.connector.endpoints = []
        
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_success(self):
        """Test successful data reading"""
        self.connector.is_connected = True
        
        # Mock session and response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'temperature': 25.5, 'humidity': 60.0}
        mock_session.get.return_value = mock_response
        
        self.connector.session = mock_session
        
        result = self.connector.read_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('timestamp', result.columns)
        self.assertIn('endpoint', result.columns)
        self.assertIn('sensor', result.columns)
        self.assertIn('value', result.columns)
    
    def test_read_data_http_error(self):
        """Test reading data with HTTP error"""
        self.connector.is_connected = True
        
        # Mock session with error response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        
        self.connector.session = mock_session
        
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_disconnect(self):
        """Test disconnection"""
        mock_session = Mock()
        self.connector.session = mock_session
        self.connector.is_connected = True
        
        result = self.connector.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.connector.is_connected)
        mock_session.close.assert_called_once()


class TestDatabaseConnector(unittest.TestCase):
    """Test cases for DatabaseConnector"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'connection_string': 'sqlite:///:memory:',
            'query': 'SELECT * FROM sensors WHERE timestamp > NOW() - INTERVAL 1 HOUR',
            'poll_interval': 10.0
        }
        self.connector = DatabaseConnector(self.config)
    
    def test_initialization(self):
        """Test database connector initialization"""
        self.assertEqual(self.connector.connection_string, 'sqlite:///:memory:')
        self.assertEqual(self.connector.query, 'SELECT * FROM sensors WHERE timestamp > NOW() - INTERVAL 1 HOUR')
        self.assertEqual(self.connector.poll_interval, 10.0)
        self.assertIsNone(self.connector.connection)
    
    def test_library_availability_check(self):
        """Test SQLAlchemy library availability check"""
        self.assertIsInstance(self.connector._sqlalchemy_available, bool)
    
    def test_connect_without_library(self):
        """Test connection attempt without SQLAlchemy library"""
        self.connector._sqlalchemy_available = False
        
        result = self.connector.connect()
        self.assertFalse(result)
        self.assertFalse(self.connector.is_connected)
    
    def test_connect_no_connection_string(self):
        """Test connection attempt without connection string"""
        self.connector._sqlalchemy_available = True
        self.connector.connection_string = None
        
        result = self.connector.connect()
        self.assertFalse(result)
        self.assertFalse(self.connector.is_connected)
    
    @patch('pandas.read_sql')
    def test_connect_success(self, mock_read_sql):
        """Test successful database connection"""
        # Mock library as available
        self.connector._sqlalchemy_available = True
        
        # Mock SQLAlchemy
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value = mock_connection
        
        self.connector._sqlalchemy = Mock()
        self.connector._sqlalchemy.create_engine.return_value = mock_engine
        
        result = self.connector.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.connector.is_connected)
        self.assertEqual(self.connector.connection, mock_connection)
    
    def test_read_data_not_connected(self):
        """Test reading data when not connected"""
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    def test_read_data_no_query(self):
        """Test reading data with no query configured"""
        self.connector.is_connected = True
        self.connector.query = None
        
        result = self.connector.read_data()
        self.assertIsNone(result)
    
    @patch('pandas.read_sql')
    def test_read_data_success(self, mock_read_sql):
        """Test successful data reading"""
        self.connector.is_connected = True
        self.connector.connection = Mock()
        
        # Mock pandas read_sql
        mock_df = pd.DataFrame({
            'sensor_id': [1, 2, 3],
            'value': [25.5, 30.0, 28.2],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='h')
        })
        mock_read_sql.return_value = mock_df
        
        result = self.connector.read_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('timestamp', result.columns)
        mock_read_sql.assert_called_once_with(self.connector.query, self.connector.connection)
    
    @patch('pandas.read_sql')
    def test_read_data_no_timestamp_column(self, mock_read_sql):
        """Test reading data that doesn't have timestamp column"""
        self.connector.is_connected = True
        self.connector.connection = Mock()
        
        # Mock pandas read_sql without timestamp
        mock_df = pd.DataFrame({
            'sensor_id': [1, 2, 3],
            'value': [25.5, 30.0, 28.2]
        })
        mock_read_sql.return_value = mock_df
        
        result = self.connector.read_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('timestamp', result.columns)  # Should be added automatically
    
    def test_disconnect(self):
        """Test disconnection"""
        mock_connection = Mock()
        self.connector.connection = mock_connection
        self.connector.is_connected = True
        
        result = self.connector.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.connector.is_connected)
        mock_connection.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()