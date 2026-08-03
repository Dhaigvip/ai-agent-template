"""
Entry point: python -m ai_agent_template

  python -m ai_agent_template     run the server (HOST/PORT from env, see .env.example)
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from ai_agent_template.config import load_config

    config = load_config()
    uvicorn.run("ai_agent_template.main:app", host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
