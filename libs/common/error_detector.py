import numpy as np
from typing import List
from copy import deepcopy

from utils.general import xyxy2xywh
import configs.infer as config_infer


class ErrorDetector:
    def __init__(self, ref_buttons: List[List]):
        """
        Args:
            ref_buttons: it should look like
                [[0, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 1]]
                corresponds to number of button each row
                0: normal button
                1: interesting button
        """
        self.ref_buttons = ref_buttons
        self.num_button = self._get_num_button()
        print("num_button", self.num_button)

    def _get_num_button(self):
        counter = 0
        for row in self.ref_buttons:
            for _ in row:
                counter += 1
        return counter

    def _stats(self, pred):
        """
        Check
        Args:
            pred:

        Returns:
            False: if num_pred != num_button
            True: else get stats for avg_size and convert xyxy -> xywh
        """
        np_pred = np.array(pred)
        cls = np_pred[:, 0].reshape(-1, 1)
        xyxy = np_pred[:, 1:]
        # convert to xywh
        xywh = xyxy2xywh(xyxy)
        # concat
        self.cls_xywh = np.concatenate([cls, xywh], axis=-1)
        self.avg_size = np.mean(xywh[:, 2:4].flatten())
        print("avg_size", self.avg_size)

    def _sort(self, pred):
        if not self._check_num_button(pred):
            return {"false_det": True,
                    "message": "num_pred != num_ref"}

        sorted_row = self._sort_by_y()

        if not self._check_num_each_row(sorted_row):
            return {"false_det": True,
                    "message": "num_pred each row is not aligned to num_ref"}

        sorted_button = self._sort_by_x(sorted_row)

        return sorted_button

    def _sort_by_y(self):
        """
        Sort list of bbox by y
        Returns:

        """
        _sorted_y = deepcopy(self.cls_xywh)
        inc_idx = _sorted_y[:, 2].argsort()
        _sorted_y = _sorted_y[inc_idx]

        # if there is only one bbox
        if len(_sorted_y) == 1:
            return [[_sorted_y[0][0]]]

        container = []
        row = []
        for i, bbox in enumerate(_sorted_y):
            if i == 0:
                # get state of button only
                row.append(bbox)
                continue
            # start checking from the second box
            else:
                # it would be next row
                # if the distance is greater than avg_size of bboxes
                if abs(bbox[2] - _sorted_y[i - 1][2]) > self.avg_size:
                    if len(row) > 0:
                        container.append(row)
                    row = [bbox]  # reset new row with current bbox
                # still in the same row
                else:
                    row.append(bbox)
        # add the last row
        if len(row) > 0:
            container.append(row)
        return container

    def _sort_by_x(self, sorted_y):
        _sorted = deepcopy(sorted_y)
        for i, row in enumerate(_sorted):
            np_row = np.array(row)
            inc_idx = np_row[:, 1].argsort()
            _sorted[i] = np_row[inc_idx].tolist()
        return _sorted

    def _check_num_button(self, pred):
        return len(pred) == self.num_button

    def _check_num_each_row(self, sorted_row):
        assert len(sorted_row) == len(self.ref_buttons), "num_row_pred != num_row_ref"
        for pred_row, ref_row in zip(sorted_row, self.ref_buttons):
            if len(pred_row) != len(ref_row):
                return False
        return True

    def _check_error(self, bboxes):
        container = []
        for i, row in enumerate(self.ref_buttons):
            for j, col in enumerate(row):
                # skip if it's not button of interest
                if not col:
                    continue
                # check button's status
                if bboxes[i][j][0] in config_infer.Recognition.ON_CLASSES:
                    container.append((i, j, 1))
                else:
                    container.append((i, j, 0))
        return container

    def __call__(self, pred: List[List]):
        """

        Args:
            pred:

        Returns:
            {
                "false_det": bool,
                "message": Union[None, error_pos, str]
            }
        """
        # do statistics
        self._stats(pred)

        bboxes = self._sort(pred)
        # error in detection, return message
        if isinstance(bboxes, dict):
            return bboxes

        # check error button if suitable detection
        print(*bboxes, sep="\n")
        err = self._check_error(bboxes)
        return {"false_det": False,
                "message": err}
