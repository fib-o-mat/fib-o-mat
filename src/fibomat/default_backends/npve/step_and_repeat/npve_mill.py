from typing import Optional

from fibomat.mill import MillBase
from fibomat.units import TimeQuantity, QuantityType


class NPVEMill(MillBase):
    def __init__(
        self,
        *,
        dwell_time: TimeQuantity,
        repeats: Optional[int] = None,
        dose: Optional[QuantityType] = None,
        scan_direction: Optional[float] = None,
        **kwargs,
    ):
        # angle = 0   : right to left
        # angle = pi/2: bottom to top

        if (repeats is not None and dose is not None) or (
            repeats is None and dose is None
        ):
            raise ValueError(
                "One of repeats or dose must be given and not both or none of them."
            )

        self._scan_direction = scan_direction

        super().__init__(dwell_time=dwell_time, repeats=repeats, dose=dose, **kwargs)
