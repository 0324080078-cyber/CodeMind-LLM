"""
Security Research Agent
Helps with legitimate security research, CTF challenges,
penetration testing, and security tool development.
Strictly for authorized research and education.
"""

from agents.base_agent import BaseAgent


SECURITY_SYSTEM = """You are an expert cybersecurity researcher and educator.
You help with:
- CTF (Capture The Flag) challenges and solutions
- Penetration testing methodology and tools
- Network security scanning and auditing
- Cryptography implementation and analysis
- Secure coding practices and vulnerability analysis
- Security tool development in Python/Go/Rust
- Web application security testing (OWASP)
- Forensics and incident response

You always:
- Write tools for authorized, legal use only
- Include proper usage warnings and scope limitations
- Follow responsible disclosure principles
- Explain the security concepts behind each tool

You never assist with attacks on systems without authorization."""


class SecurityAgent(BaseAgent):

    CATEGORIES = {
        "ctf": ["ctf","flag","capture the flag","challenge","pwn","rev","forensics"],
        "network": ["nmap","scan","network","port","enumeration","reconnaissance"],
        "web": ["sql injection","xss","csrf","owasp","burp","web vuln","api security"],
        "crypto": ["cryptography","cipher","hash","aes","rsa","decrypt","encode"],
        "forensics": ["forensics","pcap","wireshark","memory","disk image","artifacts"],
        "pentest": ["pentest","penetration test","red team","exploit","payload","metasploit"],
    }

    def handle(self, prompt: str) -> str:
        category = self._detect_category(prompt)
        return self._generate_security_tool(prompt, category)

    def _detect_category(self, prompt: str) -> str:
        p = prompt.lower()
        for cat, keywords in self.CATEGORIES.items():
            if any(k in p for k in keywords):
                return cat
        return "general"

    def _generate_security_tool(self, prompt: str, category: str) -> str:
        full_prompt = f"""{SECURITY_SYSTEM}

Category: {category}
Request: {prompt}

Generate a complete, well-documented security tool or solution.
Include: usage examples, legal disclaimer, and educational explanation."""

        result = self._generate(full_prompt, max_tokens=3072, temperature=0.2)

        disclaimer = """
# ⚠️  SECURITY RESEARCH TOOL — CodeMind
# For authorized use only. Always obtain proper permission
# before testing systems. Unauthorized access is illegal.
# ================================================================

"""
        return disclaimer + result
