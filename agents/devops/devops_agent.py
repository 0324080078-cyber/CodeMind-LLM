"""
DevOps Agent
Generates infrastructure code, CI/CD pipelines,
Docker configs, Kubernetes manifests, and deployment scripts
"""

from agents.base_agent import BaseAgent


DEVOPS_SYSTEM = """You are a senior DevOps/SRE engineer with expertise in:
- Docker and Docker Compose
- Kubernetes (K8s) and Helm charts
- CI/CD: GitHub Actions, GitLab CI, Jenkins
- Infrastructure as Code: Terraform, Ansible, Pulumi
- Cloud: AWS, GCP, Azure
- Monitoring: Prometheus, Grafana, ELK Stack
- Nginx, Traefik, Caddy reverse proxies
- Linux system administration and bash scripting

You always provide complete, production-ready configurations with:
- Security best practices (non-root users, secrets management)
- Health checks and resource limits
- Logging and monitoring integration
- Rollback strategies
- Complete documentation"""


class DevOpsAgent(BaseAgent):

    def handle(self, prompt: str) -> str:
        full_prompt = f"""{DEVOPS_SYSTEM}

Request: {prompt}

Generate complete, production-ready DevOps configuration:"""

        result = self._generate(full_prompt, max_tokens=3072, temperature=0.2)
        return f"# 🚀 DevOps Configuration — CodeMind\n\n{result}"
