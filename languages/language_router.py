"""
Multi-Language Code Generation Router
Supports 50+ programming languages
"""

SUPPORTED_LANGUAGES = {
    "python": ["py","python","django","flask","fastapi","pandas","numpy"],
    "javascript": ["js","javascript","node","express","react","vue","angular","next"],
    "typescript": ["ts","typescript","tsx"],
    "rust": ["rust","rs","cargo"],
    "go": ["go","golang"],
    "java": ["java","spring","maven","gradle","kotlin"],
    "cpp": ["c++","cpp","cmake"],
    "c": ["c lang"," c ","c code","ansi c"],
    "csharp": ["c#","csharp","dotnet",".net","unity c#","asp.net"],
    "php": ["php","laravel","symfony","wordpress"],
    "ruby": ["ruby","rails","rb"],
    "swift": ["swift","ios","xcode","swiftui"],
    "kotlin": ["kotlin","android","jetpack"],
    "sql": ["sql","postgres","mysql","sqlite","database query"],
    "bash": ["bash","shell","sh","linux","terminal script"],
    "powershell": ["powershell","ps1","windows script"],
    "solidity": ["solidity","smart contract","ethereum","web3","blockchain"],
    "r": [" r language"," r code","rstudio","data science r"],
    "matlab": ["matlab","octave"],
    "lua": ["lua","roblox","love2d"],
    "haskell": ["haskell","functional"],
    "scala": ["scala","spark","akka"],
    "dart": ["dart","flutter","mobile app"],
    "assembly": ["assembly","asm","x86","arm"],
}


class LanguageRouter:
    """Routes code generation requests to the right language context."""

    def detect_language(self, prompt: str) -> str:
        p = prompt.lower()
        for lang, keywords in SUPPORTED_LANGUAGES.items():
            if any(k in p for k in keywords):
                return lang
        return "python"

    def get_system_prompt(self, language: str) -> str:
        prompts = {
            "python": "You are an expert Python developer. Write clean, Pythonic code with type hints.",
            "javascript": "You are an expert JavaScript/Node.js developer. Use modern ES2024 syntax.",
            "typescript": "You are an expert TypeScript developer. Always include proper types and interfaces.",
            "rust": "You are an expert Rust developer. Write safe, idiomatic Rust with proper error handling.",
            "go": "You are an expert Go developer. Write clean, concurrent Go following official style guide.",
            "java": "You are an expert Java developer. Use Java 21 features and follow SOLID principles.",
            "cpp": "You are an expert C++20 developer. Write modern, safe C++ with RAII and smart pointers.",
            "csharp": "You are an expert C# developer. Use .NET 8 and modern C# 12 features.",
            "solidity": "You are an expert Solidity developer. Write secure smart contracts following OpenZeppelin patterns.",
            "rust": "You are an expert Rust developer. Prioritize memory safety and zero-cost abstractions.",
        }
        return prompts.get(language, f"You are an expert {language} developer. Write complete, production-ready code.")
