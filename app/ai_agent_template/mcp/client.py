"""MCP client wrapper — connects to the companion mcp-server-template server.

Uses StreamableHTTP transport with a single static Bearer token, matching
mcp-server-template's own DEV_TOKEN/OidcAuthProvider model — no cookie
forwarding, no org/ms/branch query params (that was the reference's
multi-tenancy plumbing, dropped here same as everywhere else in this
template).

LangGraph note: tools are loaded once via `mcp.ClientSession.list_tools()`
and converted to LangChain Tools — see agent/utils/tool_helpers.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared._httpx_utils import McpHttpClientFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry policy for MCP tool calls and list_tools().
#
# Why tenacity rather than model.with_retry():
#   - tool calls go through httpx -> MCP transport, not through LangChain runnables
#
# Policy:
#   - 4 attempts total (1 original + 3 retries)
#   - exponential jitter: initial ~1s, jitter +/-0.5x, cap 30s
#   - retries only on transient transport errors:
#       httpx.TimeoutException    - read/connect timeout
#       httpx.NetworkError        - connection reset, DNS failure
#       httpx.RemoteProtocolError - server closed connection mid-response
#   - propagates immediately on 4xx/5xx application errors
# ---------------------------------------------------------------------------
_MCP_RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)

_mcp_retry = retry(
    retry=retry_if_exception_type(_MCP_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@dataclass
class MCPClientConfig:
    """Configuration for connecting to the companion MCP server."""

    server_url: str = "http://localhost:3001/mcp"
    auth_token: str = ""  # sent as "Authorization: Bearer <token>" when set
    agent_session_id: str = ""  # optional session-correlation header
    verify_ssl: bool = True  # set False for local dev with self-signed certs

    def build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.agent_session_id:
            headers["X-Agent-Session"] = self.agent_session_id
        return headers

    def make_http_client_factory(self) -> McpHttpClientFactory:
        """Return an httpx factory that honours verify_ssl and sets safe timeouts.

        streamablehttp_client delegates all HTTP I/O to the httpx.AsyncClient
        produced by this factory. The default factory always sets verify=True;
        for local development against a self-signed cert we need verify=False.

        Timeout defaults:
          connect=10s - DNS + TCP handshake; short to surface dead hosts fast.
          read=60s    - MCP tool calls can be slow (LLM round-trips, large lists).
          write=10s   - requests are small; 10s is ample.
          pool=5s     - time to acquire a connection from the pool.
        """
        verify = self.verify_ssl
        _default_timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)

        def _factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers or {},
                timeout=timeout if timeout is not None else _default_timeout,
                auth=auth,
                follow_redirects=True,
                verify=verify,
            )

        return _factory


class MCPClient:
    """Async MCP client — fetches tools from the companion MCP server.

    Used at graph-build time:
      1. open the client (async with)
      2. list tools via list_tools() and convert to LangChain Tools
      3. close the client (the LangChain Tools keep their own httpx connection
         to the MCP server for tool calls)

    See ANYIO / ASYNC-GENERATOR CLEANUP NOTE in connect() for the cancel-scope
    discipline that keeps async cleanup from breaking under errors.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._streams_context = None
        self._session_context = None
        # Turn-scoped cache for read-only (get_*) tool results.
        # Cleared at the start of every turn and on any mutation call.
        self._read_cache: dict[str, Any] = {}

    @staticmethod
    def _stable_str(v: Any) -> Any:
        """Recursively sort dict keys and list elements for a stable JSON key."""
        if isinstance(v, list):
            return sorted(
                (MCPClient._stable_str(i) for i in v),
                key=lambda x: json.dumps(x, sort_keys=True),
            )
        if isinstance(v, dict):
            return {k: MCPClient._stable_str(val) for k, val in sorted(v.items())}
        return v

    def _cache_key(self, name: str, arguments: dict) -> str:
        return f"{name}:{json.dumps(MCPClient._stable_str(arguments), sort_keys=True)}"

    def clear_turn_cache(self) -> None:
        """Clear the turn-scoped read cache. Called at the start of every turn."""
        self._read_cache.clear()

    async def connect(self) -> None:
        headers = self._config.build_headers()

        logger.info(
            "MCP connect: %s (has_token=%s)",
            self._config.server_url,
            bool(self._config.auth_token),
        )

        self._streams_context = streamablehttp_client(
            url=self._config.server_url,
            headers=headers,
            httpx_client_factory=self._config.make_http_client_factory(),
        )
        try:
            read_stream, write_stream, _ = await self._streams_context.__aenter__()

            self._session_context = ClientSession(read_stream, write_stream)
            self._session = await self._session_context.__aenter__()

            await self._session.initialize()
        except BaseException:
            # Explicit cleanup in THIS task so anyio cancel scopes are exited
            # here, not in GC's implicit athrow task.
            try:
                await self.disconnect()
            except Exception:
                logger.debug("MCP cleanup error after connect failure", exc_info=True)
            raise

        logger.info("MCP session initialized")

    async def disconnect(self) -> None:
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            self._session = None
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
            self._streams_context = None
        logger.info("MCP disconnected")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    def _ensure_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        return self._session

    @property
    def session(self) -> ClientSession:
        """The live MCP session — for langchain_mcp_adapters.tools.convert_mcp_tool_to_langchain_tool."""
        return self._ensure_session()

    @_mcp_retry
    async def list_tools(self):
        """List tools from the MCP server.

        Retried on transient transport errors — called once at graph-build time
        but a momentary network blip should not abort the whole startup sequence.
        """
        session = self._ensure_session()
        return await session.list_tools()

    async def call_tool(self, name: str, arguments: dict | None = None):
        """Call a tool by name.

        Read tools (get_*): results are cached for the turn. Cache is keyed by
        tool name + stable-serialised arguments so argument ordering never causes
        spurious misses. Any mutation call clears the entire read cache.

        Retried on transient transport errors (timeout, network reset).
        Application-level errors (4xx/5xx from the server) are NOT retried.
        """
        args = arguments or {}
        is_read = name.startswith("get_")

        if is_read:
            key = self._cache_key(name, args)
            if key in self._read_cache:
                logger.debug("mcp.cache_hit: %s", name)
                return self._read_cache[key]
            result = await self._call_tool_inner(name, args)
            self._read_cache[key] = result
            return result

        # Mutation — server data changed, cached reads are now stale.
        self._read_cache.clear()
        return await self._call_tool_inner(name, args)

    @_mcp_retry
    async def _call_tool_inner(self, name: str, arguments: dict):
        session = self._ensure_session()
        logger.debug("mcp.call_tool: %s args=%s", name, arguments)
        return await session.call_tool(name, arguments=arguments)
