from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, num_gpu_blocks: int, num_cpu_blocks: int, 
                 block_size: int, eos: int, config: dict):
        # block manager
        self.block_manager = BlockManager(num_gpu_blocks, num_cpu_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.eos = eos

        self.enable_offload = config.get('enable_offload', False)


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool, dict, dict]:
        scheduled_sequences = []
        swap_in_map = {}
        swap_out_map = {}
        current_scheduled_tokens = 0

        # try schedule for swapped seqs firstly
        while self.enable_offload and self.swapped and len(self.running) < self.max_num_sequences:
            seq = self.swapped[0]
            if self.block_manager.can_swapin(seq, len(self.running)):
                seq = self.swapped.popleft()
                mapping = self.block_manager.swapin(seq)
                swap_in_map.update(mapping)
                print("swap in {} ".format(seq.seq_id))
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                break
                
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq, len(self.running)) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
                
        if scheduled_sequences:
            return scheduled_sequences, True, swap_in_map, swap_out_map
        
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                if self.running:
                    if self.enable_offload:
                        mapping = self.preempt(self.running.pop())
                        swap_out_map.update(mapping)
                    else:
                        self.preempt(self.running.pop())

                    self.running.appendleft(seq) # fixed：add back to running queue, otherwise it will become ghost occupying block
                else:
                    if  self.enable_offload:
                        mapping = self.preempt(seq)
                        swap_out_map.update(mapping)
                    else:
                        self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False, swap_in_map, swap_out_map


    def preempt(self, seq: Sequence) -> None:
        if self.enable_offload:
            print(f"swap out seq {seq.seq_id}")
            mapping = self.block_manager.swapout(seq)
            seq.status = SequenceStatus.SWAPPED
            self.swapped.appendleft(seq)      
            return mapping
        else:
            seq.status = SequenceStatus.WAITING
            self.block_manager.deallocate(seq)
            self.waiting.appendleft(seq)


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)