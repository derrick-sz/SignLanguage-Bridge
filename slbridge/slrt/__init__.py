"""Optional EMNLP 2024 online CSLR backend. Importing it never loads Torch."""

from .assets import SLRTConfig, SLRTResourceError

__all__ = ['SLRTConfig', 'SLRTResourceError']
