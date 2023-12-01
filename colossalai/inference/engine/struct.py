import enum
from typing import List

class RequsetStatus(enum.Enum):
    """The status of Sentences"""
    
    WAITING = enum.auto()
    RUNNING = enum.auto()
    ABORTED = enum.auto()
    OVERLENGTH = enum.auto()
    COMPLETED = enum.auto()
    LENGTH_CAPPED = enum.auto()
    
    @staticmethod
    def is_finished(status: "SentenceStatus") -> bool:
        return  status in [
                    OVERLENGTH,
                    COMPLETED,
                    LENGTH_CAPPED,
                ]
    
    @staticmethod
    def is_running(status: "SentenceStatus") -> bool:
            return status == RUNNING
    
    @staticmethod
    def is_waiting(status: "SentenceStatus") -> bool:
            return status == WAITING

class InferReq:
    def __init__(self,
                 request_id: int,
                 token_id: int,
                 prompt: str,
                 blokc_size,
                 sample_params,
                 infer_state,
                 ):
        
        self.request_id = request_id
        self.input_token_id = token_id
        self.prompt = prompt
        self.blokc_size = blokc_size
        self.output_token_id = []
        self.output = ""
        self.status = SentenceStatus.WAITING
        self.sample_params=sample_params
        
        self._logical_blocks: List[LogicalTokenBlock] = []
    
    def get_sentence_len(self) -> None:
        return len(self.input_token_id) + len(self.output_token_id)
        
    def get_input_len(self) -> None:
        return len(self.input_token_id)
    
    def get_output_len(self) -> None:
        return len(self.output_token_id)
    
    def _add_new_block(self) -> None:
        self._logical_blocks.append(LogicalTokenBlock())
    
    def add_token_id(self, token_id: List[int]) -> None:
        currnet_index = 0
        remain_length = len(token_id)
        while (remain_length > 0):
            slot_num = last_block.empty_slots()
            if slot_num == 0:
                self._add_new_block()
                slot_num = last_block.empty_slots()
            last_block = self._logical_blocks[-1]
            last_block.add_tokens(token_id[currnet_index : currnet_index + slot_num])
            remain_length -= slot_num
            
    def check_finish(self):
        return SentenceStatus.check_finish(self.status)
    
    def __repr__(self) -> str:
        return (f"Request ID(request_id={self.request_id}, "
                f"prompt={self.prompt}, "
                f"status={self.status.name}, "
                f"sample_params={self.sample_params}, "
                f"logical block number={len(self._logical_blocks)})")
    