from dataclasses import dataclass

from pydantic import model_validator


@dataclass
class AtroposArgs:
    atropos_server_host: str = "http://localhost"
    atropos_server_port: int = 8000
    atropos_starting_step: int = 0

    @model_validator(mode="after")
    def non_zero_beta_w_lora(self):
        if self.trl.beta != 0.0 and self.adapter:
            raise ValueError("Cannot use (q)LoRA with non-zero beta")
        return self
