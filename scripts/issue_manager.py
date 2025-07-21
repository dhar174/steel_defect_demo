#!/usr/bin/env python3
"""
Issue Management Tool for Steel Defect Demo

This script parses the main Phase 6 issue and generates comprehensive sub-issues
for each numbered section (6.1, 6.2, etc.). Each sub-issue includes detailed
requirements, acceptance criteria, and references to the parent issue.

Usage:
    python scripts/issue_manager.py --generate-sub-issues
    python scripts/issue_manager.py --export-json
    python scripts/issue_manager.py --export-markdown
"""

import json
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse


class IssueManager:
    """Manages the creation and organization of GitHub sub-issues."""
    
    def __init__(self):
        self.parent_issue_number = 10
        self.parent_issue_url = "https://github.com/dhar174/steel_defect_demo/issues/10"
        self.sub_issues = []
        
    def parse_phase_6_issue(self) -> Dict[str, Any]:
        """Parse the Phase 6 issue content and extract sub-sections."""
        
        # Issue content from GitHub API response
        issue_content = """# Phase 6: Production Deployment and Infrastructure

## Overview
Implement comprehensive production deployment infrastructure and operational readiness for the steel casting defect prediction system. This phase establishes containerization, orchestration, monitoring, security, and maintenance procedures necessary for industrial deployment and long-term operation.

## Objectives
- Develop containerized deployment architecture using Docker and Docker Compose
- Implement production-grade API services with FastAPI for external integrations
- Establish comprehensive monitoring, logging, and alerting infrastructure
- Create automated deployment pipelines and CI/CD workflows
- Implement security measures and access control systems
- Develop operational procedures for maintenance and troubleshooting
- Establish scalability and high availability configurations

## Implementation Tasks

### 6.1 Containerization and Orchestration
- **Files**: `Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`
- **Dependencies**: docker, docker-compose
- **Key Features**:
  - Multi-stage Docker builds for optimized production images
  - Separate containers for inference engine, dashboard, and API services
  - Environment-specific configuration management
  - Volume management for persistent data and model artifacts
  - Network configuration for secure inter-service communication
  - Health checks and restart policies for container reliability

### 6.2 Production API Services
- **File**: `src/api/main.py`
- **Dependencies**: fastapi, uvicorn, pydantic, sqlalchemy
- **Functionality**:
  - RESTful API endpoints for prediction services
  - WebSocket connections for real-time data streaming
  - Authentication and authorization middleware
  - Request validation and response serialization
  - Rate limiting and request throttling
  - API documentation with OpenAPI/Swagger integration
  - Comprehensive error handling and logging

### 6.3 Database Integration and Data Management
- **Files**: `src/database/`, `migrations/`
- **Dependencies**: sqlalchemy, alembic, postgresql
- **Capabilities**:
  - PostgreSQL database setup for production data storage
  - Database schema design for predictions, alerts, and system metrics
  - Migration scripts for schema updates and data transformations
  - Connection pooling and performance optimization
  - Backup and recovery procedures
  - Data retention policies and archival strategies

### 6.4 Monitoring and Observability Stack
- **Files**: `monitoring/`, `configs/monitoring.yaml`
- **Dependencies**: prometheus, grafana, elasticsearch, kibana
- **Features**:
  - Prometheus metrics collection for system and application monitoring
  - Grafana dashboards for infrastructure and application metrics
  - Elasticsearch and Kibana for centralized log management
  - Custom metrics for ML model performance and prediction accuracy
  - Alert rules for system failures and performance degradation
  - Distributed tracing for request flow analysis

### 6.5 Security and Access Control
- **Files**: `src/security/`, `configs/security.yaml`
- **Dependencies**: oauth2, jwt, cryptography, sqlalchemy
- **Implementation**:
  - JWT-based authentication system with refresh tokens
  - Role-based access control (RBAC) for different user types
  - API key management for external system integrations
  - Data encryption at rest and in transit
  - Security headers and CORS configuration
  - Audit logging for security events and user actions

### 6.6 CI/CD Pipeline and Automation
- **Files**: `.github/workflows/`, `scripts/deploy.sh`
- **Dependencies**: github-actions, pytest, docker
- **Capabilities**:
  - Automated testing pipeline with unit, integration, and performance tests
  - Code quality checks with linting, type checking, and security scanning
  - Automated Docker image building and registry publishing
  - Environment-specific deployment automation
  - Blue-green deployment strategies for zero-downtime updates
  - Rollback procedures and deployment validation

### 6.7 Configuration Management and Secrets
- **Files**: `configs/production/`, `scripts/config_management.py`
- **Dependencies**: pyyaml, python-dotenv, vault
- **Features**:
  - Environment-specific configuration management
  - Secrets management with HashiCorp Vault integration
  - Configuration validation and schema enforcement
  - Runtime configuration updates without service restart
  - Configuration versioning and change tracking
  - Secure credential storage and rotation procedures

### 6.8 Operational Tools and Maintenance
- **Files**: `scripts/operations/`, `docs/operations/`
- **Dependencies**: psutil, click, requests
- **Tools**:
  - Health check scripts for all system components
  - Backup and restore utilities for data and models
  - Performance tuning and optimization scripts
  - Log analysis and troubleshooting tools
  - Capacity planning and resource monitoring utilities
  - Incident response procedures and runbooks

### 6.9 Industrial Integration Connectors
- **Files**: `src/connectors/industrial/`
- **Protocols**:
  - OPC UA client for SCADA system integration
  - MQTT subscriber for IoT sensor networks
  - Modbus TCP/IP for legacy equipment connectivity
  - REST API adapters for modern industrial systems
  - Database connectors for historian systems

### 6.10 Edge Computing Deployment
- **Files**: `edge/`, `configs/edge.yaml`
- **Features**:
  - Lightweight containerized deployment for edge devices
  - Offline operation capabilities with data synchronization
  - Edge-specific model optimization and quantization
  - Local data processing and filtering
  - Secure edge-to-cloud communication channels

### 6.11 Multi-Site Deployment Management
- **Files**: `deployment/multi-site/`
- **Capabilities**:
  - Centralized management for multiple steel plants
  - Site-specific configuration and model deployment
  - Cross-site data aggregation and analysis
  - Federated learning for model improvement
  - Global monitoring and alerting coordination"""

        return {"content": issue_content}
    
    def extract_subsections(self, content: str) -> List[Dict[str, Any]]:
        """Extract numbered subsections (6.1, 6.2, etc.) from the issue content."""
        
        # Pattern to match subsections like "### 6.1 Title"
        pattern = r'### (6\.\d+)\s+([^\n]+)\n(.*?)(?=### \d|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        subsections = []
        for section_num, title, body in matches:
            subsections.append({
                'section_number': section_num,
                'title': title.strip(),
                'body': body.strip()
            })
        
        return subsections
    
    def create_sub_issue(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive sub-issue from a section."""
        
        section_num = section['section_number']
        title = section['title']
        body = section['body']
        
        # Parse the body to extract structured information
        files_match = re.search(r'- \*\*Files?\*\*:\s*([^\n]+)', body)
        dependencies_match = re.search(r'- \*\*Dependencies\*\*:\s*([^\n]+)', body)
        
        files = files_match.group(1) if files_match else "TBD"
        dependencies = dependencies_match.group(1) if dependencies_match else "TBD"
        
        # Generate comprehensive acceptance criteria
        acceptance_criteria = self.generate_acceptance_criteria(section_num, title, body)
        
        # Create the sub-issue
        sub_issue = {
            'title': f"{section_num} {title}",
            'body': self.generate_issue_body(section_num, title, body, files, dependencies, acceptance_criteria),
            'labels': ['Phase 6', 'Production', 'Infrastructure', section_num.replace('.', '-')],
            'assignees': [],
            'milestone': 'Phase 6: Production Deployment',
            'parent_issue': {
                'number': self.parent_issue_number,
                'url': self.parent_issue_url
            },
            'metadata': {
                'section_number': section_num,
                'files': files,
                'dependencies': dependencies,
                'estimated_effort': self.estimate_effort(title, body),
                'priority': self.determine_priority(section_num),
                'prerequisites': self.identify_prerequisites(section_num)
            }
        }
        
        return sub_issue
    
    def generate_acceptance_criteria(self, section_num: str, title: str, body: str) -> List[str]:
        """Generate comprehensive acceptance criteria for each sub-issue."""
        
        criteria_map = {
            '6.1': [
                "Multi-stage Dockerfile created with optimized production images",
                "docker-compose.yml for development environment implemented",
                "docker-compose.prod.yml for production deployment created",
                "Separate containers configured for inference engine, dashboard, and API services",
                "Environment-specific configuration management implemented",
                "Volume management for persistent data and model artifacts configured",
                "Network configuration for secure inter-service communication established",
                "Health checks and restart policies implemented for all containers",
                "Container builds successfully and runs without errors",
                "Production deployment tested and validated"
            ],
            '6.2': [
                "FastAPI application structure created in src/api/main.py",
                "RESTful API endpoints for prediction services implemented",
                "WebSocket connections for real-time data streaming functional",
                "Authentication and authorization middleware integrated",
                "Request validation and response serialization working",
                "Rate limiting and request throttling implemented",
                "OpenAPI/Swagger documentation automatically generated",
                "Comprehensive error handling and logging implemented",
                "API performance meets <500ms response time requirement",
                "API documentation accessible and complete"
            ],
            '6.3': [
                "PostgreSQL database setup for production data storage completed",
                "Database schema design for predictions, alerts, and system metrics implemented",
                "Migration scripts created in migrations/ directory",
                "Alembic integration for schema updates configured",
                "Connection pooling and performance optimization implemented",
                "Backup and recovery procedures documented and tested",
                "Data retention policies and archival strategies implemented",
                "Database performance meets throughput requirements",
                "Database security and access controls configured",
                "Migration rollback procedures tested and validated"
            ],
            '6.4': [
                "Prometheus metrics collection configured for system monitoring",
                "Grafana dashboards created for infrastructure and application metrics",
                "Elasticsearch and Kibana deployed for centralized log management",
                "Custom metrics for ML model performance implemented",
                "Alert rules for system failures and performance degradation configured",
                "Distributed tracing for request flow analysis implemented",
                "Monitoring stack integrated with existing infrastructure",
                "Alert notifications properly configured and tested",
                "Dashboard accessibility and usability validated",
                "Monitoring data retention policies implemented"
            ],
            '6.5': [
                "JWT-based authentication system with refresh tokens implemented",
                "Role-based access control (RBAC) for different user types configured",
                "API key management for external system integrations created",
                "Data encryption at rest and in transit implemented",
                "Security headers and CORS configuration applied",
                "Audit logging for security events and user actions implemented",
                "Security vulnerability scanning integrated",
                "Penetration testing completed and vulnerabilities addressed",
                "Security policies and procedures documented",
                "Compliance with relevant security standards validated"
            ],
            '6.6': [
                "GitHub Actions workflows created for automated testing",
                "Code quality checks with linting, type checking, and security scanning implemented",
                "Automated Docker image building and registry publishing configured",
                "Environment-specific deployment automation implemented",
                "Blue-green deployment strategies for zero-downtime updates configured",
                "Rollback procedures and deployment validation implemented",
                "CI/CD pipeline tested with sample deployments",
                "Deployment automation scripts created and tested",
                "Pipeline performance meets deployment time requirements",
                "CI/CD documentation and runbooks created"
            ],
            '6.7': [
                "Environment-specific configuration management implemented",
                "HashiCorp Vault integration for secrets management configured",
                "Configuration validation and schema enforcement implemented",
                "Runtime configuration updates without service restart capability added",
                "Configuration versioning and change tracking implemented",
                "Secure credential storage and rotation procedures implemented",
                "Configuration management tools tested and validated",
                "Configuration backup and recovery procedures documented",
                "Configuration security and access controls implemented",
                "Configuration management documentation completed"
            ],
            '6.8': [
                "Health check scripts for all system components created",
                "Backup and restore utilities for data and models implemented",
                "Performance tuning and optimization scripts developed",
                "Log analysis and troubleshooting tools created",
                "Capacity planning and resource monitoring utilities implemented",
                "Incident response procedures and runbooks documented",
                "Operational tools tested and validated in production-like environment",
                "Maintenance schedules and procedures documented",
                "Operational monitoring and alerting configured",
                "Staff training materials for operational procedures created"
            ],
            '6.9': [
                "OPC UA client for SCADA system integration implemented",
                "MQTT subscriber for IoT sensor networks created",
                "Modbus TCP/IP connectivity for legacy equipment established",
                "REST API adapters for modern industrial systems implemented",
                "Database connectors for historian systems created",
                "Industrial protocol security and authentication implemented",
                "Integration testing with simulated industrial systems completed",
                "Error handling and reconnection logic for industrial connections implemented",
                "Industrial integration documentation and configuration guides created",
                "Performance testing of industrial data ingestion completed"
            ],
            '6.10': [
                "Lightweight containerized deployment for edge devices created",
                "Offline operation capabilities with data synchronization implemented",
                "Edge-specific model optimization and quantization completed",
                "Local data processing and filtering capabilities implemented",
                "Secure edge-to-cloud communication channels established",
                "Edge deployment testing on representative hardware completed",
                "Edge device management and monitoring tools implemented",
                "Edge-specific configuration management created",
                "Edge deployment documentation and procedures documented",
                "Edge computing performance requirements validated"
            ],
            '6.11': [
                "Centralized management for multiple steel plants implemented",
                "Site-specific configuration and model deployment capabilities created",
                "Cross-site data aggregation and analysis functionality implemented",
                "Federated learning for model improvement framework established",
                "Global monitoring and alerting coordination implemented",
                "Multi-site deployment testing completed",
                "Site isolation and security measures implemented",
                "Multi-site configuration management tools created",
                "Cross-site communication and data sharing protocols established",
                "Multi-site operational procedures and documentation completed"
            ]
        }
        
        return criteria_map.get(section_num, [
            "Requirements analysis completed",
            "Implementation plan created",
            "Code implementation completed",
            "Testing completed successfully",
            "Documentation updated",
            "Integration testing passed"
        ])
    
    def estimate_effort(self, title: str, body: str) -> str:
        """Estimate effort required for each sub-issue."""
        
        complexity_indicators = [
            'integration', 'security', 'database', 'monitoring', 'ci/cd',
            'multi-site', 'edge', 'industrial', 'production'
        ]
        
        effort_score = 0
        title_lower = title.lower()
        body_lower = body.lower()
        
        for indicator in complexity_indicators:
            if indicator in title_lower or indicator in body_lower:
                effort_score += 1
        
        if effort_score >= 4:
            return "Large (3-4 weeks)"
        elif effort_score >= 2:
            return "Medium (1-2 weeks)"
        else:
            return "Small (3-5 days)"
    
    def determine_priority(self, section_num: str) -> str:
        """Determine priority based on dependencies and criticality."""
        
        high_priority = ['6.1', '6.2', '6.3']  # Core infrastructure
        medium_priority = ['6.4', '6.5', '6.6', '6.7', '6.8']  # Supporting systems
        low_priority = ['6.9', '6.10', '6.11']  # Advanced features
        
        if section_num in high_priority:
            return "High"
        elif section_num in medium_priority:
            return "Medium"
        else:
            return "Low"
    
    def identify_prerequisites(self, section_num: str) -> List[str]:
        """Identify prerequisites for each sub-issue."""
        
        prerequisites_map = {
            '6.1': [],  # No dependencies - foundation
            '6.2': ['6.1'],  # Needs containerization
            '6.3': ['6.1'],  # Needs containerization
            '6.4': ['6.1', '6.2', '6.3'],  # Needs basic infrastructure
            '6.5': ['6.2'],  # Needs API services
            '6.6': ['6.1', '6.2'],  # Needs containers and APIs
            '6.7': ['6.1'],  # Needs containerization
            '6.8': ['6.1', '6.2', '6.3', '6.4'],  # Needs most infrastructure
            '6.9': ['6.1', '6.2', '6.3'],  # Needs core infrastructure
            '6.10': ['6.1', '6.2'],  # Needs containers and APIs
            '6.11': ['6.1', '6.2', '6.3', '6.4']  # Needs full infrastructure
        }
        
        return prerequisites_map.get(section_num, [])
    
    def generate_issue_body(self, section_num: str, title: str, body: str, 
                          files: str, dependencies: str, acceptance_criteria: List[str]) -> str:
        """Generate the complete issue body with all necessary information."""
        
        return f"""# {section_num} {title}

**Parent Issue:** #{self.parent_issue_number} - Phase 6: Production Deployment and Infrastructure
**URL:** {self.parent_issue_url}

## Overview
{self.extract_overview_from_body(body)}

## Technical Specifications

### Files to be Created/Modified
```
{files}
```

### Dependencies
```
{dependencies}
```

### Detailed Requirements
{body}

## Acceptance Criteria

{chr(10).join([f"- [ ] {criterion}" for criterion in acceptance_criteria])}

## Testing Requirements

- [ ] Unit tests implemented for all new functionality
- [ ] Integration tests cover component interactions
- [ ] Performance tests validate requirements are met
- [ ] Security tests verify access controls and data protection
- [ ] Documentation tests ensure completeness and accuracy

## Definition of Done

- [ ] All acceptance criteria completed
- [ ] Code review completed and approved
- [ ] All tests passing (unit, integration, performance)
- [ ] Security review completed
- [ ] Documentation updated and reviewed
- [ ] Production deployment tested
- [ ] Monitoring and alerting configured
- [ ] Rollback procedures tested
- [ ] Knowledge transfer completed

## Resources and References

- [Phase 6 Parent Issue](https://github.com/dhar174/steel_defect_demo/issues/10)
- [Technical Implementation Spec](/TECHNICAL_IMPLEMENTATION_SPEC.md)
- [Production Deployment Guide](/docs/deployment_guide.md)
- [Steel Defect Demo Repository](https://github.com/dhar174/steel_defect_demo)

## Notes

This issue is part of Phase 6: Production Deployment and Infrastructure. Please ensure coordination with other Phase 6 sub-issues and maintain consistency with the overall system architecture.
"""

    def extract_overview_from_body(self, body: str) -> str:
        """Extract a concise overview from the detailed body text."""
        lines = body.split('\n')
        overview_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith('-'):
                overview_lines.append(line.strip())
                if len(overview_lines) >= 3:
                    break
        
        return ' '.join(overview_lines) if overview_lines else "Detailed implementation requirements for this production deployment component."
    
    def generate_all_sub_issues(self) -> List[Dict[str, Any]]:
        """Generate all sub-issues from the Phase 6 issue."""
        
        issue_data = self.parse_phase_6_issue()
        subsections = self.extract_subsections(issue_data['content'])
        
        self.sub_issues = []
        for section in subsections:
            sub_issue = self.create_sub_issue(section)
            self.sub_issues.append(sub_issue)
        
        return self.sub_issues
    
    def export_to_json(self, filename: str = None) -> str:
        """Export sub-issues to JSON format."""
        
        if not self.sub_issues:
            self.generate_all_sub_issues()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sub_issues_phase6_{timestamp}.json"
        
        export_data = {
            'parent_issue': {
                'number': self.parent_issue_number,
                'url': self.parent_issue_url,
                'title': 'Phase 6: Production Deployment and Infrastructure'
            },
            'sub_issues': self.sub_issues,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_sub_issues': len(self.sub_issues),
                'generator': 'steel_defect_demo.scripts.issue_manager'
            }
        }
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath.absolute())
    
    def export_to_markdown(self, filename: str = None) -> str:
        """Export sub-issues to Markdown format for easy viewing."""
        
        if not self.sub_issues:
            self.generate_all_sub_issues()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sub_issues_phase6_{timestamp}.md"
        
        markdown_content = f"""# Phase 6 Sub-Issues

**Parent Issue:** #{self.parent_issue_number} - Phase 6: Production Deployment and Infrastructure  
**URL:** {self.parent_issue_url}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This document contains {len(self.sub_issues)} comprehensive sub-issues extracted from Phase 6 of the Steel Defect Demo project. Each sub-issue includes detailed requirements, acceptance criteria, and implementation guidelines.

---

"""
        
        for i, issue in enumerate(self.sub_issues, 1):
            markdown_content += f"""## {i}. {issue['title']}

**Priority:** {issue['metadata']['priority']}  
**Estimated Effort:** {issue['metadata']['estimated_effort']}  
**Prerequisites:** {', '.join(issue['metadata']['prerequisites']) if issue['metadata']['prerequisites'] else 'None'}

### Labels
{', '.join([f'`{label}`' for label in issue['labels']])}

### Issue Body
{issue['body']}

---

"""
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(filepath.absolute())
    
    def generate_github_issue_commands(self) -> str:
        """Generate GitHub CLI commands to create the sub-issues."""
        
        if not self.sub_issues:
            self.generate_all_sub_issues()
        
        commands = []
        commands.append("#!/bin/bash")
        commands.append("")
        commands.append("# GitHub CLI commands to create Phase 6 sub-issues")
        commands.append("# Run this script after installing gh CLI and authenticating")
        commands.append("")
        
        for issue in self.sub_issues:
            # Escape quotes in the body
            body_escaped = issue['body'].replace('"', '\\"').replace('`', '\\`')
            
            cmd = f"""gh issue create \\
    --title "{issue['title']}" \\
    --body "{body_escaped}" \\
    --label "{','.join(issue['labels'])}" \\
    --milestone "{issue['milestone']}" \\
    --repo dhar174/steel_defect_demo"""
            
            commands.append(cmd)
            commands.append("")
        
        return '\n'.join(commands)


def main():
    """Main function to handle command line arguments."""
    
    parser = argparse.ArgumentParser(description='Manage GitHub sub-issues for Phase 6')
    parser.add_argument('--generate-sub-issues', action='store_true',
                       help='Generate all sub-issues and display summary')
    parser.add_argument('--export-json', action='store_true',
                       help='Export sub-issues to JSON format')
    parser.add_argument('--export-markdown', action='store_true',
                       help='Export sub-issues to Markdown format')
    parser.add_argument('--generate-gh-commands', action='store_true',
                       help='Generate GitHub CLI commands to create issues')
    parser.add_argument('--output-file', type=str,
                       help='Specify output filename')
    
    args = parser.parse_args()
    
    issue_manager = IssueManager()
    
    if args.generate_sub_issues or not any([args.export_json, args.export_markdown, args.generate_gh_commands]):
        # Default action
        sub_issues = issue_manager.generate_all_sub_issues()
        print(f"Generated {len(sub_issues)} sub-issues for Phase 6:")
        print()
        
        for issue in sub_issues:
            print(f"- {issue['title']}")
            print(f"  Priority: {issue['metadata']['priority']}")
            print(f"  Effort: {issue['metadata']['estimated_effort']}")
            print(f"  Prerequisites: {', '.join(issue['metadata']['prerequisites']) if issue['metadata']['prerequisites'] else 'None'}")
            print()
    
    if args.export_json:
        filepath = issue_manager.export_to_json(args.output_file)
        print(f"Sub-issues exported to JSON: {filepath}")
    
    if args.export_markdown:
        filepath = issue_manager.export_to_markdown(args.output_file)
        print(f"Sub-issues exported to Markdown: {filepath}")
    
    if args.generate_gh_commands:
        commands = issue_manager.generate_github_issue_commands()
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(commands)
            print(f"GitHub CLI commands written to: {args.output_file}")
        else:
            print("GitHub CLI Commands:")
            print("=" * 50)
            print(commands)


if __name__ == "__main__":
    main()