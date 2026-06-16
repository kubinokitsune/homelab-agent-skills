"""Shared skills library for the homelab agent system.

Every agent (Forge, Mason, Apex, ...) imports the building blocks it needs
from here, so cross-cutting concerns — config, notifications, vault access,
vector memory — are written once and behave identically everywhere.
"""
