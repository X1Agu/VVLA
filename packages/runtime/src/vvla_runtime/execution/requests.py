"""Explicit episode generations prevent stale slot responses from being consumed."""
from dataclasses import dataclass
import threading
import uuid

@dataclass(frozen=True)
class RequestTicket:
    request_id: str
    generation: int
    sequence: int
    scope: str

class RequestLedger:
    def __init__(self):
        self._scope = uuid.uuid4().hex
        self._generation = 0
        self._active = {}
        self._lock = threading.Lock()

    def start(self, request_id):
        if not isinstance(request_id,str) or not request_id:
            raise ValueError('request_id must be a nonempty string')
        with self._lock:
            if request_id in self._active:
                raise ValueError('duplicate active request_id')
            self._generation += 1
            ticket = RequestTicket(request_id,self._generation,0,self._scope)
            self._active[request_id] = ticket
            return ticket

    def validate_batch(self, tickets):
        with self._lock:
            if len({t.request_id for t in tickets}) != len(tickets):
                raise ValueError('duplicate request in batch')
            if any(self._active.get(t.request_id) != t for t in tickets):
                raise ValueError('stale or cancelled request')

    def advance(self, ticket):
        with self._lock:
            if self._active.get(ticket.request_id) != ticket:
                raise ValueError('stale or cancelled request')
            updated = RequestTicket(ticket.request_id,ticket.generation,ticket.sequence+1,self._scope)
            self._active[ticket.request_id] = updated
            return updated

    def finish(self, ticket):
        with self._lock:
            if self._active.get(ticket.request_id) != ticket:
                raise ValueError('stale or cancelled request')
            del self._active[ticket.request_id]
