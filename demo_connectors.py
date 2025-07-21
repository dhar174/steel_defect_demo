#!/usr/bin/env python3
"""
Demo script showing how to use the Production Data Connectors

This script demonstrates the basic usage of each connector type
with example configurations.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from connectors.data_connectors import (
    OPCUAConnector,
    MQTTConnector, 
    RESTConnector,
    DatabaseConnector
)

def demo_opc_ua_connector():
    """Demonstrate OPC UA connector usage"""
    print("\n=== OPC UA Connector Demo ===")
    
    config = {
        'server_url': 'opc.tcp://industrial-server:4840',
        'nodes': [
            'ns=2;i=1001',  # Temperature sensor
            'ns=2;i=1002',  # Pressure sensor
            'ns=2;i=1003',  # Flow rate sensor
        ],
        'sampling_interval': 1.0,
        'username': 'operator',
        'password': 'secure_password'
    }
    
    connector = OPCUAConnector(config)
    print(f"OPC UA library available: {connector._opcua_available}")
    
    # Simulate connection and data reading
    if connector.connect():
        print("Connected to OPC UA server")
        data = connector.read_data()
        if data is not None:
            print(f"Read {len(data)} data points")
            print(data.head())
        connector.disconnect()
    else:
        print("Failed to connect to OPC UA server (library not available)")

def demo_mqtt_connector():
    """Demonstrate MQTT connector usage"""
    print("\n=== MQTT Connector Demo ===")
    
    config = {
        'broker_host': 'mqtt.factory.com',
        'broker_port': 1883,
        'topic': 'sensors/+',
        'username': 'sensor_user',
        'password': 'mqtt_password',
        'qos': 1,
        'keep_alive': 60
    }
    
    connector = MQTTConnector(config)
    print(f"MQTT library available: {connector._mqtt_available}")
    
    if connector.connect():
        print("Connected to MQTT broker")
        # In real usage, messages would accumulate over time
        data = connector.read_data()
        if data is not None:
            print(f"Read {len(data)} messages")
            print(data.head())
        connector.disconnect()
    else:
        print("Failed to connect to MQTT broker (library not available)")

def demo_rest_connector():
    """Demonstrate REST API connector usage"""
    print("\n=== REST Connector Demo ===")
    
    config = {
        'base_url': 'https://api.factory-sensors.com',
        'endpoints': [
            'sensors/temperature',
            'sensors/pressure',
            'sensors/vibration'
        ],
        'headers': {
            'Authorization': 'Bearer your-api-token',
            'Content-Type': 'application/json'
        },
        'timeout': 10.0,
        'poll_interval': 5.0
    }
    
    connector = RESTConnector(config)
    print(f"Requests library available: {connector._requests_available}")
    
    if connector.connect():
        print("REST API connector initialized")
        # Note: This would fail in demo since the URL doesn't exist
        # In real usage, it would poll the configured endpoints
        connector.disconnect()
    else:
        print("Failed to initialize REST connector")

def demo_database_connector():
    """Demonstrate database connector usage"""
    print("\n=== Database Connector Demo ===")
    
    config = {
        'connection_string': 'postgresql://user:password@db-server:5432/sensors',
        'query': '''
            SELECT timestamp, sensor_id, value 
            FROM sensor_readings 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC
        ''',
        'poll_interval': 30.0
    }
    
    connector = DatabaseConnector(config)
    print(f"SQLAlchemy library available: {connector._sqlalchemy_available}")
    
    if connector.connect():
        print("Connected to database")
        data = connector.read_data()
        if data is not None:
            print(f"Read {len(data)} rows")
            print(data.head())
        connector.disconnect()
    else:
        print("Failed to connect to database (library not available)")

def main():
    """Run all connector demos"""
    print("Production Data Connectors Demo")
    print("=" * 40)
    
    demo_opc_ua_connector()
    demo_mqtt_connector()
    demo_rest_connector()
    demo_database_connector()
    
    print("\n" + "=" * 40)
    print("Demo completed!")
    print("\nTo use these connectors in production:")
    print("1. Install required libraries: pip install asyncua paho-mqtt sqlalchemy")
    print("2. Configure connection details for your specific environment")
    print("3. Handle errors and reconnection logic for production robustness")

if __name__ == '__main__':
    main()