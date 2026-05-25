"""ComfyUI registration shim.

Each node implementation lives in its own module. ComfyUI still imports this
file as the package entry point, so keep it small and only expose mappings here.
"""

try:
    from .pid_auto_settings import NODE_CLASS_MAPPINGS as AUTO_CLASS_MAPPINGS
    from .pid_auto_settings import NODE_DISPLAY_NAME_MAPPINGS as AUTO_DISPLAY_NAME_MAPPINGS
    from .pid_decode import NODE_CLASS_MAPPINGS as DECODE_CLASS_MAPPINGS
    from .pid_decode import NODE_DISPLAY_NAME_MAPPINGS as DECODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # Allows `python -c "import nodes"` from this folder.
    from pid_auto_settings import NODE_CLASS_MAPPINGS as AUTO_CLASS_MAPPINGS
    from pid_auto_settings import NODE_DISPLAY_NAME_MAPPINGS as AUTO_DISPLAY_NAME_MAPPINGS
    from pid_decode import NODE_CLASS_MAPPINGS as DECODE_CLASS_MAPPINGS
    from pid_decode import NODE_DISPLAY_NAME_MAPPINGS as DECODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **DECODE_CLASS_MAPPINGS,
    **AUTO_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DECODE_DISPLAY_NAME_MAPPINGS,
    **AUTO_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
