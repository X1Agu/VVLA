"""Binary observation transport for one shared VVLA model owner."""
import asyncio
import logging
import inspect
import math
import time

from aiohttp import WSMsgType, web

from ..errors import AdmissionRejected, InputError, WorkerFatalError
from ..protocol.binary import decode_frame, encode_frame
from ..protocol.wire import MAX_MESSAGE_BYTES
from .server import DynamicBatchServer

CONTENT_TYPE = 'application/vnd.vvla.frame'
DEFAULT_LIMIT = 16 * 1024 * 1024
LOG = logging.getLogger(__name__)


def error_reply(error, identity=None):
    if isinstance(error, AdmissionRejected):
        status, code = 429, 'admission_rejected'
    elif isinstance(error, OverflowError):
        status, code = 429, 'queue_full'
    elif isinstance(error, TimeoutError):
        status, code = 504, 'timeout'
    elif isinstance(error, (InputError, ValueError, TypeError, KeyError, EOFError, RecursionError)):
        status, code = 400, 'invalid_request'
    elif isinstance(error, (WorkerFatalError, RuntimeError)):
        status, code = 503, 'worker_unavailable'
    else:
        status, code = 500, 'internal_error'
        LOG.exception('Inference failed')
    message = str(error) or code
    return status, dict(request_id=identity, error=dict(code=code, status=status, message=message))


def request_fields(message):
    if not isinstance(message, dict) or set(message) - {'request_id', 'seed', 'observation', 'timeout_s', 'target_slo_ms', 'valid_for_ms'}:
        raise ValueError('Expected a binary inference request')
    identity = message.get('request_id')
    if type(identity) is not str or not 1 <= len(identity) <= 256:
        raise ValueError('request_id must contain 1 to 256 characters')
    if not isinstance(message.get('observation'), dict) or type(message.get('seed')) is not int:
        raise ValueError('An observation object and an integer seed are required')
    if 'valid_for_ms' in message:
        budget = message['valid_for_ms']
        if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
            raise ValueError('valid_for_ms must be finite and positive')
    return message


def create_app(service, *, max_request_bytes=DEFAULT_LIMIT, connection_limit=128):
    if not 0 < max_request_bytes <= MAX_MESSAGE_BYTES or connection_limit < 1:
        raise ValueError('Invalid network limits')
    app = web.Application(client_max_size=max_request_bytes)
    restart = asyncio.Lock()
    sockets = set()

    async def lifecycle(_app):
        if isinstance(service, DynamicBatchServer):
            await service.start(network=True)
        else:
            await service.start()
        try:
            yield
        finally:
            await service.close()

    async def shutdown(_app):
        await asyncio.gather(*(ws.close(code=1001) for ws in tuple(sockets)))

    async def execute(message):
        message = request_fields(message)
        if 'valid_for_ms' in message:
            message['valid_until_at'] = time.perf_counter() + message.pop('valid_for_ms') / 1000
        if service.health().get('restart_required'):
            async with restart:
                if service.health().get('restart_required'):
                    await service.restart()
        answer = await service.infer(**message, _owned=True)
        return dict(request_id=message['request_id'], **answer)

    async def infer(request):
        if request.content_type != CONTENT_TYPE:
            return web.json_response({'error': 'Expected ' + CONTENT_TYPE}, status=415)
        identity = None
        try:
            message = decode_frame(await request.read())
            if isinstance(message, dict):
                identity = message.get('request_id')
            reply = await execute(message)
            status = 200
        except web.HTTPException:
            raise
        except Exception as error:
            status, reply = error_reply(error, identity if isinstance(identity, str) else None)
        return web.Response(body=encode_frame(reply), status=status, content_type=CONTENT_TYPE)

    async def websocket(request):
        ws = web.WebSocketResponse(max_msg_size=max_request_bytes, compress=False)
        await ws.prepare(request)
        sockets.add(ws)
        pending = {}

        async def deliver(message):
            identity = message['request_id']
            try:
                try:
                    reply = await execute(message)
                except Exception as error:
                    _, reply = error_reply(error, identity)
                if not ws.closed:
                    await ws.send_bytes(encode_frame(reply))
            except (ConnectionError, OSError):
                pass
            finally:
                pending.pop(identity, None)

        try:
            async for frame in ws:
                identity = None
                try:
                    if frame.type == WSMsgType.ERROR:
                        break
                    if frame.type != WSMsgType.BINARY:
                        raise ValueError('Expected a binary frame')
                    message = decode_frame(frame.data)
                    if isinstance(message, dict) and isinstance(message.get('request_id'), str):
                        identity = message['request_id']
                    if isinstance(message, dict) and message.get('operation') == 'cancel':
                        if set(message) != {'operation', 'request_id'} or identity is None:
                            raise ValueError('Invalid cancellation')
                        task = pending.get(identity)
                        if task is not None:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        await ws.send_bytes(encode_frame(dict(operation='cancelled', request_id=identity)))
                        continue
                    request_fields(message)
                    if identity in pending:
                        raise ValueError('Duplicate live request ID on this connection')
                    if len(pending) >= connection_limit:
                        raise OverflowError('Connection in-flight capacity exhausted')
                    task = asyncio.create_task(deliver(message))
                    pending[identity] = task
                    # A task cancelled before its first coroutine step still retires its ID.
                    task.add_done_callback(lambda done, key=identity: pending.pop(key, None)
                                           if pending.get(key) is done else None)
                except Exception as error:
                    _, reply = error_reply(error, identity)
                    await ws.send_bytes(encode_frame(reply))
        finally:
            sockets.discard(ws)
            tasks = tuple(pending.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return ws

    async def health(_request):
        data = service.health()
        return web.json_response(data, status=200 if data['status'] == 'running' else 503)

    async def metadata(_request):
        try:
            return web.json_response(await service.metadata())
        except Exception as error:
            status, reply = error_reply(error)
            return web.json_response(reply, status=status)

    async def metrics(_request):
        value = service.metrics()
        return web.json_response(await value if inspect.isawaitable(value) else value)

    app.cleanup_ctx.append(lifecycle)
    app.on_shutdown.append(shutdown)
    app.add_routes([web.post('/infer', infer), web.get('/ws', websocket), web.get('/health', health),
                    web.get('/metadata', metadata), web.get('/metrics', metrics)])
    return app
