import xxhash
import numpy as np
from collections import deque
import torch
from myvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int):
        self.block_size: int = block_size
        
        # ==========================================
        # GPU Blocks
        # ==========================================
        self.gpu_blocks: list[Block] = [Block(i) for i in range(num_gpu_blocks)]
        self.hash_to_gpu_block_id: dict[int, int] = {} 
        self.free_gpu_block_ids: deque[int] = deque(range(num_gpu_blocks))
        self.used_gpu_block_ids: set[int] = set()

        # ==========================================
        # CPU Blocks 
        # ==========================================
        self.cpu_blocks: list[Block] = [Block(i) for i in range(num_cpu_blocks)]
        self.free_cpu_block_ids: deque[int] = deque(range(num_cpu_blocks))
        self.used_cpu_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))

        if not isinstance(token_ids, torch.Tensor):
             # if tokens_ids is a list of tensor
            tids = torch.as_tensor(token_ids, device='cpu')
        else:
            tids = token_ids.cpu()
        h.update(np.array(tids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_gpu_block(self, block_id: int) -> Block:
        block = self.gpu_blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_gpu_block_ids.remove(block_id)
        self.used_gpu_block_ids.add(block_id)
        return block

    def _deallocate_gpu_block(self, block_id: int) -> None:
        assert self.gpu_blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.gpu_blocks[block_id]
        block.token_ids = []
        if block.hash != -1 and self.hash_to_gpu_block_id.get(block.hash) == block_id:
            del self.hash_to_gpu_block_id[block.hash]
        self.used_gpu_block_ids.remove(block_id)
        self.free_gpu_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, num_running: int) -> bool:
        # watermark: num_running + 1
        return len(self.free_gpu_block_ids) >= (seq.num_blocks + num_running + 1)

    def allocate(self, seq: Sequence) -> None:
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_gpu_block_id.get(h, -1)
            
            if block_id == -1 or self.gpu_blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                seq.num_cached_tokens += self.block_size
                if block_id not in self.used_gpu_block_ids:
                    block = self._allocate_gpu_block(self.free_gpu_block_ids[0])
                    block.ref_count += 1 # fixed: every new allocated block should have ref_count = 1
                    block.update(h=h, token_ids=token_ids) # fixed: if block_id not in self.used_block_ids, treated as cache miss then
                    if h != -1:
                        self.hash_to_gpu_block_id[h] = block.block_id
                else:
                    block = self.gpu_blocks[self.hash_to_gpu_block_id[h]]
                    block.ref_count += 1
            else:
                block = self._allocate_gpu_block(self.free_gpu_block_ids[0])
                block.ref_count += 1 # fixed, every new allocated block should have ref_count = 1
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_gpu_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            block = self.gpu_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_gpu_block(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_swapin(self, seq: Sequence, num_running: int) -> bool:
        if not hasattr(seq, 'cpu_block_table') or not seq.cpu_block_table:
            return False
        # watermark: swap in happens only when there are 2 more free blocks after swaping in
        return len(self.free_gpu_block_ids) >= (len(seq.cpu_block_table) + num_running + 1)
        
    def swapout(self, seq: Sequence) -> dict:
        mapping = {}
        if not hasattr(seq, 'cpu_block_table'):
            seq.cpu_block_table = []
        
        for gpu_block_id in seq.block_table:
            assert len(self.free_cpu_block_ids) > 0, "Out of CPU memory for swapping!"
            cpu_block_id = self.free_cpu_block_ids.popleft()
            self.used_cpu_block_ids.add(cpu_block_id)
            
            gpu_block = self.gpu_blocks[gpu_block_id]
            cpu_block = self.cpu_blocks[cpu_block_id]
            
            cpu_block.update(h=gpu_block.hash, token_ids=gpu_block.token_ids)
            cpu_block.ref_count += 1
            
            mapping[gpu_block_id] = cpu_block_id
            seq.cpu_block_table.append(cpu_block_id)
            
            gpu_block.ref_count -= 1
            if gpu_block.ref_count == 0:
                self._deallocate_gpu_block(gpu_block_id)

        seq.block_table.clear()
        return mapping
                
    def swapin(self, seq: Sequence) -> dict:
        mapping = {}

        for cpu_block_id in seq.cpu_block_table:  
            cpu_block = self.cpu_blocks[cpu_block_id]
            h = cpu_block.hash

            if h != -1 and h in self.hash_to_gpu_block_id:
                # cache hit, reuse the existing GPU block without generating a mapping for swapping
                gpu_block_id = self.hash_to_gpu_block_id[h]
                gpu_block = self.gpu_blocks[gpu_block_id]
                gpu_block.ref_count += 1
                seq.block_table.append(gpu_block_id)
            else:
                # cache miss, need to actually move the data back to GPU and generate a mapping for swapping
                assert len(self.free_gpu_block_ids) > 0, "No free GPU blocks during swap in!"
                gpu_block_id = self.free_gpu_block_ids.popleft()
                self.used_gpu_block_ids.add(gpu_block_id)
                
                gpu_block = self.gpu_blocks[gpu_block_id]
                gpu_block.reset()
                gpu_block.ref_count += 1
                
                # zero-copy restore: reattach the pointer back to the GPU block
                gpu_block.update(h=h, token_ids=cpu_block.token_ids)
                if h != -1:
                    self.hash_to_gpu_block_id[h] = gpu_block_id

                mapping[cpu_block_id] = gpu_block_id
                seq.block_table.append(gpu_block_id)

            cpu_block.ref_count -= 1
            if cpu_block.ref_count == 0:
                cpu_block.reset()
                self.used_cpu_block_ids.remove(cpu_block_id)
                self.free_cpu_block_ids.append(cpu_block_id)

        seq.cpu_block_table.clear()
        return mapping

    def can_append(self, seq: Sequence) -> bool:
        # determine can append by actual capacity 
        current_capacity = len(seq.block_table) * self.block_size
        blocks_needed = 1 if seq.num_tokens >= current_capacity else 0
        return len(self.free_gpu_block_ids) >= blocks_needed

    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        ###
        # edge case: for sequence just swapped in, its last block could be partial full with h = -1. 
        # but the block manager attemp to allocate a new block which will cause assertation error below.
        ###
        
        # compute hash for the last block if it's full to ensure idempotent append, which is critical for correct swapping behavior
        if seq.num_tokens > 0 and seq.num_tokens % self.block_size == 0:
            last_block_for_seq_id = block_tables[-1]
            block = self.gpu_blocks[last_block_for_seq_id]
            if block.hash == -1:
                h = self.compute_hash(token_ids=seq.block(seq.num_blocks - 1), 
                                      prefix_hash_value=-1 if len(block_tables) == 1 else self.gpu_blocks[block_tables[-2]].hash)
                block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
                self.hash_to_gpu_block_id[h] = block.block_id

        # check if really need to allocate a new block for the new token
        current_capacity = len(block_tables) * self.block_size
        if seq.num_tokens >= current_capacity:
            last_block_for_seq_id = block_tables[-1]
            assert self.gpu_blocks[last_block_for_seq_id].hash != -1, "Previous block must be hashed before allocating a new one"
            
            block = self._allocate_gpu_block(self.free_gpu_block_ids[0])
            block.ref_count += 1 # fixed, every new allocated block should have ref_count = 1
            block_tables.append(block.block_id)
