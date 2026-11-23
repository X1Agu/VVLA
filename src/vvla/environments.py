"""Locate isolated workers without importing model packages into the application."""
import os
from pathlib import Path
import sys


def worker_directory(model):
    return Path(os.environ.get('VVLA_ENV_ROOT', Path.home() / '.local/share/vvla/envs')).expanduser() / model


def worker_python(model, explicit=None):
    if explicit is not None:
        return str(explicit)
    configured = os.environ.get('VVLA_' + model.upper() + '_PYTHON')
    if configured:
        return configured
    default = worker_directory(model) / 'bin/python'
    return str(default) if default.is_file() else sys.executable
