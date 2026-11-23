"""Private opt-in B32 inputs; checkpoint output transforms remain unchanged."""
from .batch_coalesced_prepare import build_coalesced_inputs
from .batch_processor import Pi05BatchProcessor
from .processor import Pi05LiteProcessor


class CoalescedBatchProcessor(Pi05BatchProcessor):
    def __init__(self, processor, batch_size, *, output_processor=None):
        if type(processor) is not Pi05LiteProcessor or type(batch_size) is not int or batch_size != 32:
            raise ValueError('Coalesced preparation requires the private original processor and exactly B32')
        super().__init__(processor, batch_size, output_processor=output_processor)

    def build_model_inputs(self, observations, *, return_host_layout=False):
        return build_coalesced_inputs(self._single, observations, return_host_layout=return_host_layout)
