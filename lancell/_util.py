"""Shared internal utilities."""


def sql_escape(s: str) -> str:
    """Escape single quotes for LanceDB SQL string literals."""
    return s.replace("'", "''")
