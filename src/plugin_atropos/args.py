from dataclasses import dataclass


@dataclass
class AtroposArgs:
    atropos_server_host: str = "http://localhost"
    atropos_server_port: int = 8000
    atropos_starting_step: int = 0

