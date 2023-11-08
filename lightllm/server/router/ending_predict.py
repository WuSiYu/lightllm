from collections import deque
from typing import Deque, Tuple
import numpy as np

class EndingPredictState:
    "Global state (like accurate) for ending predict"
    HISTORY_GT_LEN = 1000
    HISTORY_GT_LEN_MIN = 0

    def __init__(self, config_history_length: int = 50, config_p33_inital_threshold: float = -3.25) -> None:
        self.config_history_length = config_history_length
        self.config_p33_inital_threshold = config_p33_inital_threshold      # 50% of ending request can be selected out

        self.history_eosprob_gt = np.empty(self.HISTORY_GT_LEN)
        self._history_eosprob_gt_counter = 0

    def update_gt(self, ended_item: 'EndingPredictItem'):
        print(f"EndingPredictState gt update #{self._history_eosprob_gt_counter} {ended_item.recent_max()}")
        self.history_eosprob_gt[self._history_eosprob_gt_counter % self.HISTORY_GT_LEN] = ended_item.recent_max()
        self._history_eosprob_gt_counter += 1
        print(f"EndingPredictState gt updated, p33: {self.get_p33_threshold()}")

    def get_p33_threshold(self):
        if self._history_eosprob_gt_counter > self.HISTORY_GT_LEN_MIN:
            return np.percentile(self.history_eosprob_gt[:self._history_eosprob_gt_counter], 33)
        else:
            return self.config_p33_inital_threshold

class EndingPredictItem:
    def __init__(self, global_state: EndingPredictState) -> None:
        self._counter = 0
        self.global_state = global_state
        self.history_eos: Deque[Tuple[float, int]] = deque()      # Monotonic Queue, max eos token logprob in last N forwards

    def step(self, eos_prob: float, ended: bool) -> None:
        if not ended:
            self._counter += 1
            while self.history_eos and self._counter - self.history_eos[0][1] >= self.global_state.config_history_length:
                self.history_eos.popleft()
            while self.history_eos and self.history_eos[-1][0] <= eos_prob:
                self.history_eos.pop()
            self.history_eos.append((eos_prob, self._counter))
            # print(f"EndingPredictItem step: {eos_prob:.2f}, current max: {self.history_eos[0][0]:.2f}")
        else:
            self.global_state.update_gt(self)

    def recent_max(self) -> float:
        if not self.history_eos:
            return float('-inf')
        return self.history_eos[0][0]

    def is_ending_p33(self):
        """if true, at least 33% prob is a realy ending req"""
        return self.recent_max() >= self.global_state.get_p33_threshold()
