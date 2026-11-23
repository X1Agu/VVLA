"""Async client. HTTP and WebSocket use the same typed frame and request IDs."""
import asyncio
import math
import uuid

from .errors import VVLAError
from .protocol.binary import decode_frame, encode_frame


class RemoteError(VVLAError):
    def __init__(self, reply):
        entry = reply['error']
        self.code, self.status = entry['code'], entry['status']
        self.request_id = reply.get('request_id')
        super().__init__(entry['message'])


class VvlaClient:
    def __init__(self, url, *, transport='http'):
        if transport not in ('http', 'ws'):
            raise ValueError('transport must be http or ws')
        self.url = url.rstrip('/')
        self.transport = transport
        self.session = self.socket = self.reader = None
        self.pending = {}
        self.retiring = set()

    async def __aenter__(self):
        from aiohttp import ClientSession, ClientTimeout
        if self.session is not None:
            raise RuntimeError('Client is already open')
        self.session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=15))
        try:
            if self.transport == 'ws':
                self.socket = await self.session.ws_connect(self.url + '/ws', compress=0)
                self.reader = asyncio.create_task(self._receive())
        except BaseException:
            await self.close()
            raise
        return self

    async def _receive(self):
        from aiohttp import WSMsgType
        failure = ConnectionError('VVLA WebSocket closed')
        try:
            async for frame in self.socket:
                if frame.type != WSMsgType.BINARY:
                    break
                reply = decode_frame(frame.data)
                if reply.get('operation') == 'cancelled':
                    self.retiring.discard(reply['request_id'])
                    continue
                future = self.pending.get(reply['request_id'])
                if future is not None and not future.done():
                    future.set_result(reply)
        except Exception as error:
            failure = error
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(failure)

    async def close(self):
        if self.socket is not None:
            await self.socket.close()
        if self.reader is not None:
            await self.reader
        if self.session is not None:
            await self.session.close()
        self.session = self.socket = self.reader = None
        self.retiring.clear()

    async def __aexit__(self, *_):
        await self.close()

    async def infer(self, observation, *, seed, request_id=None, timeout_s=30., target_slo_ms=None,
                    valid_for_ms=None):
        if self.session is None:
            raise RuntimeError('Use async with VvlaClient(...)')
        if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be positive and finite')
        identity = uuid.uuid4().hex if request_id is None else request_id
        if type(identity) is not str or not 1 <= len(identity) <= 256:
            raise ValueError('Expected a nonempty request_id (at most 256 characters)')
        message = dict(request_id=identity, seed=seed, observation=observation,
                       timeout_s=timeout_s, target_slo_ms=target_slo_ms)
        if valid_for_ms is not None:
            message['valid_for_ms'] = valid_for_ms
        payload = encode_frame(message)
        async with asyncio.timeout(timeout_s + 2):
            if self.transport == 'http':
                async with self.session.post(self.url + '/infer', data=payload,
                           headers={'Content-Type': 'application/vnd.vvla.frame'}) as response:
                    if response.content_type != 'application/vnd.vvla.frame':
                        response.raise_for_status()
                        raise ValueError('Unexpected response media type')
                    reply = decode_frame(await response.read())
            else:
                if self.reader.done() or self.socket.closed:
                    raise ConnectionError('VVLA WebSocket closed')
                if identity in self.pending or identity in self.retiring:
                    raise ValueError('Duplicate live request ID')
                future = asyncio.get_running_loop().create_future()
                self.pending[identity] = future
                try:
                    await self.socket.send_bytes(payload)
                    reply = await future
                except asyncio.CancelledError:
                    if not self.socket.closed:
                        self.retiring.add(identity)
                        await self.socket.send_bytes(encode_frame(dict(operation='cancel', request_id=identity)))
                    raise
                finally:
                    self.pending.pop(identity, None)
        if reply.get('request_id') != identity:
            raise ValueError('VVLA returned a different request ID')
        if 'error' in reply:
            raise RemoteError(reply)
        return reply

    async def _get(self, path):
        if self.session is None:
            raise RuntimeError('Use async with VvlaClient(...)')
        async with self.session.get(self.url + path, timeout=30) as response:
            response.raise_for_status()
            return await response.json()

    async def health(self):
        return await self._get('/health')

    async def metadata(self):
        return await self._get('/metadata')

    async def metrics(self):
        return await self._get('/metrics')
