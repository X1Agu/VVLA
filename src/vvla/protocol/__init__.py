"""Versioned local worker protocol; no pickle or executable payloads."""
from .wire import VERSION, decode, dumps, encode, loads

