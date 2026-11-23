"""Optional deadline policy with a frozen, externally measured service profile.

Estimates cover the entire owner's infer_requests call, including worker IPC.
They are predictions, not deadline guarantees. No online tuning or model-specific
kernel knowledge is hidden in the scheduler.
"""
import math
from types import MappingProxyType


def _service_table(service_ms):
    if not isinstance(service_ms, dict) or not service_ms:
        raise ValueError('A measured batch-to-service-ms profile is required')
    raw = {}
    for key, value in service_ms.items():
        if type(key) is int:
            batch = key
        elif type(key) is str and key.isdecimal() and str(int(key)) == key:
            batch = int(key)
        else:
            raise ValueError('Profile batch keys must be positive integers')
        if batch < 1 or batch in raw or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Profile costs must be unique, finite and positive')
        raw[batch] = float(value)
    return raw


class EDFPolicy:
    """Stable earliest-deadline order; no predictive rejection."""
    name = 'edf'
    predictive = False

    def validate(self, supported):
        pass

    @staticmethod
    def deadline(request):
        # A closed-loop caller can accept a missed buffer SLO while actions
        # remain useful. An open-loop caller retains the strict SLO contract.
        useful = request.valid_until_at
        target = useful if useful is not None else request.slo_deadline_at
        return min(request.deadline_at, target) if target is not None else request.deadline_at

    @staticmethod
    def priority_deadline(request):
        """Buffer depletion determines urgency, even during useful recovery."""
        target = request.slo_deadline_at
        return target if target is not None else EDFPolicy.deadline(request)

    def order(self, requests):
        return sorted(requests, key=self.priority_deadline)  # stable admission-order ties

    def select(self, requests, now, supported):
        return max(b for b in supported if b <= len(requests))

    def wait_delay(self, queue, now, supported, wait_s):
        queue = self.order(queue)
        first = queue[0]
        if sum(r.compatibility == first.compatibility for r in queue) >= supported[-1]:
            return 0.
        return max(0., min(first.admitted_at + wait_s,
                          min(min(self.priority_deadline(r), self.deadline(r)) for r in queue)) - now)

    def metadata(self):
        return dict(name=self.name, selection='stable EDF compatibility group; largest real batch',
                    priority='slo_deadline_at; fallback to useful/hard deadline',
                    feasibility='min(valid_until_at, hard deadline); SLO if validity absent',
                    predictive_admission=False)


class SlackPolicy(EDFPolicy):
    name = 'slack'
    predictive = True

    def __init__(self, service_ms, *, safety_factor=1.1, margin_ms=1.0, ordering='fifo',
                 cohort_admission=False, short_budget=None):
        if ordering not in ('fifo', 'edf'):
            raise ValueError('ordering must be fifo or edf')
        if type(cohort_admission) is not bool:
            raise ValueError('cohort_admission must be a boolean')
        self.cohort_admission = cohort_admission
        self.ordering = ordering
        self.name = 'slack-edf' if ordering == 'edf' else 'slack-fifo'
        if type(safety_factor) not in (int, float) or not math.isfinite(safety_factor) or safety_factor < 1:
            raise ValueError('safety_factor must be finite and at least one')
        if type(margin_ms) not in (int, float) or not math.isfinite(margin_ms) or margin_ms < 0:
            raise ValueError('margin_ms must be finite and nonnegative')
        raw = _service_table(service_ms)
        # A monotone envelope makes B1 an actual optimistic lower bound even
        # when measured kernel choices have non-monotone execution times.
        costs = {}
        largest = 0.
        for batch in sorted(raw):
            largest = max(largest, raw[batch])
            costs[batch] = (largest * safety_factor + margin_ms) / 1000
        self.service_ms = MappingProxyType(raw)
        self.costs = MappingProxyType(costs)
        self.safety_factor = safety_factor
        self.margin_ms = margin_ms

        self.short_budget_s = None
        self.short_costs = MappingProxyType({})
        if short_budget is not None:
            if not isinstance(short_budget, dict) or set(short_budget) != {'budget_ms', 'effective_ms'}:
                raise ValueError('short_budget needs budget_ms and effective_ms')
            threshold = short_budget['budget_ms']
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
                raise ValueError('Short admission budget must be positive and finite')
            short = _service_table(short_budget['effective_ms'])
            if not short.keys() <= costs.keys():
                raise ValueError('Short-budget costs must refer to declared batch widths')
            combined = [short[b] / 1000 if b in short else costs[b] for b in sorted(costs)]
            if any(x > y for x, y in zip(combined, combined[1:])):
                raise ValueError('Short-budget costs must be monotone across enabled widths')
            self.short_budget_s = threshold / 1000
            self.short_costs = MappingProxyType({b: value/1000 for b,value in short.items()})

    def validate(self, supported):
        if set(supported) - self.costs.keys():
            raise ValueError('Service profile must cover every enabled real batch size')

    def cost(self, batch):
        return self.costs[batch]

    def service_cost(self, batch, members):
        """Select a frozen cost from the SLO budget visible at admission."""
        if batch in self.short_costs and any(r.slo_deadline_at is not None and
                r.slo_deadline_at <= r.admitted_at + self.short_budget_s + 1e-9 for r in members):
            return self.short_costs[batch]
        return self.cost(batch)

    def order(self, requests):
        return super().order(requests) if self.ordering == 'edf' else list(requests)

    def select(self, requests, now, supported):
        """Largest supported ordered prefix that can finish for every member."""
        requests = self.order(requests)
        for batch in reversed(supported):
            if batch <= len(requests):
                members = requests[:batch]
                finish = now + self.service_cost(batch, members)
                if all(finish <= self.deadline(r) for r in members):
                    return batch
        return 0

    def wait_delay(self, queue, now, supported, wait_s):
        queue = self.order(queue)
        first = queue[0]
        compatible = [r for r in queue if r.compatibility == first.compatibility]
        if len(compatible) >= supported[-1]:
            return 0.
        batch = self.select(compatible, now, supported)
        if not batch:
            return 0.
        latest = min(self.deadline(r) for r in compatible[:batch]) - self.service_cost(batch, compatible[:batch])
        other_latest = min(min(self.priority_deadline(r), self.deadline(r)) - self.service_cost(1, (r,)) for r in queue)
        return max(0., min(first.admitted_at + wait_s, latest, other_latest) - now)

    def collection_cutoff(self, queue, incoming, now, supported, wait_s):
        """Last instant to reconsider a rejected prefix in an idle collection window."""
        if not self.cohort_admission or not queue or not wait_s:
            return None
        ordered = self.order([*queue, incoming])
        if ordered[0].compatibility != incoming.compatibility:
            return None
        peers = [r for r in ordered if r.compatibility == incoming.compatibility]
        next_width = next((b for b in supported if b > len(peers)), None)
        if len(peers) < 2 or next_width is None:
            return None
        # A larger real batch could help only while it still fits every member.
        # Its missing lanes never enter the prediction or the executor.
        cutoff = min(min(r.admitted_at for r in peers) + wait_s,
                     min(self.deadline(r) for r in peers) - self.service_cost(next_width, peers),
                     now + self.wait_delay(ordered, now, supported, wait_s))
        return cutoff if now < cutoff else None

    def predict_finish(self, queue, incoming, now, supported, wait_s, *, following=()):
        """Simulate this same queue policy without assuming future arrivals.

        Existing requests that cannot meet their deadline are removed just as
        they will be at dispatch. Compatibility groups never share a batch.
        """
        pending = [*queue, incoming, *following]
        while pending:
            feasible = [r for r in pending if now + self.service_cost(1, (r,)) <= self.deadline(r)]
            if not any(r.request_id == incoming.request_id for r in feasible):
                return None
            pending = self.order(feasible)
            now += self.wait_delay(pending, now, supported, wait_s)
            head = pending[0].compatibility
            compatible = [r for r in pending if r.compatibility == head]
            batch = self.select(compatible, now, supported)
            if not batch:
                # Floating point at a latest-start boundary must not spin.
                if pending[0].request_id == incoming.request_id:
                    return None
                pending.pop(0)
                continue
            selected = {r.request_id for r in compatible[:batch]}
            now += self.service_cost(batch, compatible[:batch])
            if incoming.request_id in selected:
                return now
            pending = [r for r in pending if r.request_id not in selected]
        return None

    def metadata(self):
        result = dict(name=self.name, service_ms=dict(self.service_ms), safety_factor=self.safety_factor,
                    cohort_admission=self.cohort_admission,
                    priority='slo_deadline_at' if self.ordering == 'edf' else 'admission order',
                    feasibility='min(valid_until_at, hard deadline); SLO if validity absent',
                    margin_ms=self.margin_ms, effective_ms={b: 1000*v for b, v in self.costs.items()},
                    selection=f'{self.ordering.upper()} compatibility group; largest feasible real batch',
                    overrun_reserve='one B1 service estimate beyond now; bounded queue; dispatch recheck',
                    estimation='frozen monotone service profile; no future arrivals assumed')
        if self.short_budget_s is not None:
            result['short_budget'] = dict(budget_ms=self.short_budget_s*1000,
                effective_ms={b: v*1000 for b,v in self.short_costs.items()},
                scope='Any actual group member whose admission-time soft SLO budget is within the threshold')
        return result


def scheduling_policy(config=None):
    """JSON-friendly policy selection; omission preserves the original FIFO."""
    config = dict(config or {})
    name = config.pop('name', 'fifo')
    if name == 'fifo' and not config:
        return None
    if name == 'edf' and not config:
        return EDFPolicy()
    if name in ('slack', 'slack-fifo', 'slack-edf'):
        return SlackPolicy(**config, ordering='edf' if name == 'slack-edf' else 'fifo')
    raise ValueError('Use fifo/edf without options, or slack-fifo/slack-edf with a service profile')
