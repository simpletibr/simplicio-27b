from dataclasses import dataclass


@dataclass
class AppConfig:
    port: int = 8080
