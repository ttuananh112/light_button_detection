import numpy as np
from typing import Union, List, Dict
from copy import deepcopy

from utils.general import xyxy2xywh

import libs.configs.infer as config_infer
import libs.configs.message as message


class ErrorDetector:
    def __init__(self, ref_buttons: List[List] = None):
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
        self.num_button = 0

    def set_ref_buttons(
            self,
            ref_buttons: List[List]
    ) -> None:
        """
        Set ref_buttons
        Args:
            ref_buttons: List[List]

        Returns:
            None
        """
        self.ref_buttons = ref_buttons
        self.num_button = self._get_num_button()
        print("num_button", self.num_button)

    def _get_num_button(self) -> int:
        counter = 0
        for row in self.ref_buttons:
            for _ in row:
                counter += 1
        return counter

    def _stats(self, pred):
        """
        Do statistic
            - Process pred to class_xywh
            - Estimate the average size of box's length
        Args:
            pred: List[List]

        Returns:
            if Dict: Error, num_pred == 0

        """
        np_pred = np.array(pred)
        if len(np_pred) == 0:
            return {message.IS_FALSE_DET: True,
                    message.MESSAGE: "num_pred == 0",
                    message.IS_ERROR: False}

        cls = np_pred[:, 0].reshape(-1, 1)
        xyxy = np_pred[:, 1:]
        # convert to xywh
        xywh = xyxy2xywh(xyxy)
        # concat
        self.cls_xywh = np.concatenate([cls, xywh], axis=-1)
        self.avg_size = np.mean(xywh[:, 2:4].flatten())
        print("avg_size", self.avg_size)
        return None

    def _sort(self, pred) -> Union[List, Dict]:
        """
        Sort prediction data position by row and col
        Args:
            pred: List[List]

        Returns:

        """
        if not self._check_num_button(pred):
            return {message.IS_FALSE_DET: True,
                    message.MESSAGE: "num_pred != num_ref",
                    message.IS_ERROR: False}

        sorted_row = self._sort_by_y()

        if not self._check_num_each_row(sorted_row):
            return {message.IS_FALSE_DET: True,
                    message.MESSAGE: "num_pred each row is not aligned to num_ref",
                    message.IS_ERROR: False}

        sorted_button = self._sort_by_x(sorted_row)

        return sorted_button

    def _sort_by_y(self) -> List[List]:
        """
        Sort list of bbox by y
        Returns:
            List[List]
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

    def _sort_by_x(self, sorted_y: List):
        """
        Sort columns
        Args:
            sorted_y: (List[List]) sorted in rows

        Returns:
            List[List]: Sorted in both row and col
        """
        _sorted = deepcopy(sorted_y)
        for i, row in enumerate(_sorted):
            np_row = np.array(row)
            inc_idx = np_row[:, 1].argsort()
            _sorted[i] = np_row[inc_idx].tolist()
        return _sorted

    def _check_num_button(self, pred):
        """
        Check whether num_pred is equal num_button
        Args:
            pred: List[List] pred from model

        Returns:
            bool
        """
        return len(pred) == self.num_button

    def _check_num_each_row(self, sorted_row):
        """

        Args:
            sorted_row:

        Returns:

        """
        # assert len(sorted_row) == len(self.ref_buttons), "num_row_pred != num_row_ref"
        if len(sorted_row) != len(self.ref_buttons):
            return False
        for pred_row, ref_row in zip(sorted_row, self.ref_buttons):
            if len(pred_row) != len(ref_row):
                return False
        return True

    def _check_error(self, bboxes):
        """
        Check whether there is any
        Args:
            bboxes:

        Returns:

        """
        container = []
        for i, row in enumerate(self.ref_buttons):
            for j, col in enumerate(row):
                # skip if it's not button of interest
                if not col:
                    continue
                # check button's status
                if bboxes[i][j][0] in config_infer.Recognition.ON_CLASSES:
                    container.append([i, j, 1])
                else:
                    container.append([i, j, 0])
        return {message.IS_FALSE_DET: False,
                message.MESSAGE: container,
                message.IS_ERROR: True}

    def __call__(self, pred: List[List]):
        """

        Args:
            pred: (List[List]): list detection from pred

        Returns:
            {
                "false_det": bool,
                "message": Union[None, error_pos, str]
                "is_error": bool
            }
            error_pos should be List[List]
            each list is (row_index, col_index, is_on)
        """
        # do statistics
        stats = self._stats(pred)
        if isinstance(stats, dict):
            return stats

        bboxes = self._sort(pred)
        # error in detection, return message
        if isinstance(bboxes, dict):
            return bboxes

        # check error button if suitable detection
        print(*bboxes, sep="\n")
        return self._check_error(bboxes)
