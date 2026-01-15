# transferred from old gitlab repo
# https://gitlab.com/viggge/fib-o-mat/-/merge_requests/1
# code originally provided by Markus Lid

from typing import Dict, Any
import math
import numpy as np
from fibomat.units import scale_factor
from fibomat import units
from fibomat import utils, U_, Q_, Vector
from fibomat.default_backends import SpotListBackend



def stream_file_impl(n_rep=1, margin=0.9, dac16bit=True, hfw_rounding=True, time_low_res=True):
    """ Custom export to FEI Stream file type files. Allows to input variables that are convenient for Stream files. Exported file will have a filename 
    including horizontal field width (hfw) added to the file name.

    Keyword arguments:
        n_rep:          (int) The number of times to repeat whole pattern. More iterations does not increase file size.
        margin:         (float) How much of the imaging area is used for patterning. Range should be from 0.1 - 1.0.
        dac16bit:       (bool) The number of bits used in the digital to analog conversion by the instrument. 'True' will give 16 bit version,
        and 'False' will give 12 bit version. Instruments will use a 12 or 16 bit DAC. Check user manual for your instrument to know which one is appropriate.
        hfw_rounding:   (bool) Option for rounding the horizontal field width (hfw) to a value that ticks in place by instrument. Boolean value.
        time_low_res:   (bool) Choose if the time unit in Stream file is given in 'high' or 'low' resolution. 'high' corresponds to Stream file time unit of 25 ns,
        while 'low' corresponds to 100ns.

    Example use:
        exported = sample.export(
            default_backends.SpotListBackend,
            base_dwell_time=Q_('0.1 µs'),
            length_unit=U_('µm'),
            save_impl=stream_file_impl(n_rep=3)
            )
        exported.save('filename') # Writes a Stream file named 'filename(hfw=12um).str'

    """
    def streamfile_type(dac16bit=dac16bit, time_low_res=time_low_res):
        if dac16bit:
            x_res = 65536  # 2^16 = 65536
            #  actually the user manual is lying, y resolution is the same you just can't see it on the screen anymore# resolution is smaller i y direction. Ref. user manual.
            y_res = x_res  #56576
            if time_low_res == 'low':
                time_unit = 100  # [ns]
                header = 's16\n'
            else:
                time_unit = 25  # [ns]
                header = 's16,25ns\n'
        else:
            x_res = 4095  # 2^12 = 4095
            # resolution is smaller i y direction. Ref. user manual.
            y_res = 3816
            header = 's\n'
            time_unit = 100  # [ns]
            if not time_low_res:
                print(
                    "High resolution is not allowed with 12 bit DAC. Using 'low' instead")

        return x_res, y_res, time_unit, header

    def round_hfw(hfw, hfw_rounding=hfw_rounding):
        if hfw_rounding:
            # Round the HFW to a convenient number.
            oom = math.floor(math.log(hfw, 10))  # Order Of Magnitude
            # rounding up to closest 25 in 1000.
            hfw = np.round(np.ceil(hfw*4*10**(-oom)) / (4*10**(-oom)))
        return hfw

    def _custom_save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any], n_rep=n_rep, margin=margin):
        # fov is in units of length_unit
        x_res, y_res, time_unit, header = streamfile_type()
        fov = parameters["fov"]
        center = fov.center
        width, height = fov.width, fov.height
        # TODO doesn't work yet for boundinboxes with area 0, fix this
        if isinstance(width, Q_):  # fov was a dimensioned bounding box
            assert isinstance(height, Q_)
            height=units.scale_to(U_("µm"), height)
            width=units.scale_to(U_("µm"), width)
            center=Vector(units.scale_to(U_("µm"), center[0].magnitude), center[1].magnitude)
        xy_aspect_ratio = x_res/y_res
        # Setting the horizontal field width (HFW) based on which is FOV aspect ratio
        if height != 0 and width/height > xy_aspect_ratio: # TODO check this whole part, for now just avoid width/0 (happens for single point)
            hfw = width / margin
        else:
            hfw = height * xy_aspect_ratio / margin
        if hfw > 0: #TODO For now just avoid log(0) this way
            hfw = round_hfw(hfw)
        if hfw == 0:
            print("calculated horizontal field width to be 0. hfw was set to 1")
            hfw = 1 # TODO check this
        shift = center - (hfw/2, hfw/xy_aspect_ratio/2)
        dwell_points[:, 0:2] = (dwell_points[:, 0:2]-shift)*x_res/hfw
        # Setting the dwell time to correct units
        # time was expressed in base_dwell_time (e.g. 0.1 ns) by spotlist_backend.py . We want to change this to ns. scale_factor ignores magnitudes of quantities
        # so we have to correct this by scaling with the magnitude.
        dwell_points[:, 2] *= (scale_factor(U_('ns'),
                                           parameters["base_dwell_time"])*parameters["base_dwell_time"].magnitude)/time_unit
        dwell_points = dwell_points.round()

        stripped = filename.split('.', 1)[0]
        filename = stripped + \
            f'(HFW={hfw:0.0f}{parameters["length_unit"]:~P})' + '.str'
        with open(filename, 'w') as fp:
            # first, write header data.
            fp.writelines([
                header,
                f'{n_rep:d}\n',  # Number of times to repeat pattern
                f'{parameters["number_of_points"]:d}\n',
            ])

            # second, write dwell point data
            # (x, y, t_d) where x and y are the position of a spot and t_d the dwell time or dwell time multiplicand.
            # FEI Stream file takes the dwell time column first, therefore it is reordered. All values in stream
            # file are given as integers.
            np.savetxt(fp, dwell_points[:, [2, 0, 1]], "%d %d %d")

    return _custom_save_impl

class FEIStreamFile(SpotListBackend):
    name = "FEI stream file"

    def __init__(
        self,
        n_rep=1,
        margin=0.9,
        dac16bit=True,
        hfw_rounding=True,
        time_low_res=True,
        description=None
    ):
        """ Custom export to FEI Stream file type files. Allows to input variables that are convenient for Stream files. Exported file will have a filename 
        including horizontal field width (hfw) added to the file name.

        Keyword arguments:
            n_rep:          (int) The number of times to repeat whole pattern. More iterations does not increase file size.
            margin:         (float) How much of the imaging area is used for patterning. Range should be from 0.1 - 1.0.
            dac16bit:       (bool) The number of bits used in the digital to analog conversion by the instrument. 'True' will give 16 bit version,
            and 'False' will give 12 bit version. Instruments will use a 12 or 16 bit DAC. Check user manual for your instrument to know which one is appropriate.
            hfw_rounding:   (bool) Option for rounding the horizontal field width (hfw) to a value that ticks in place by instrument. Boolean value.
            time_low_res:   (bool) Choose if the time unit in Stream file is given in 'high' or 'low' resolution. 'high' corresponds to Stream file time unit of 25 ns,
            while 'low' corresponds to 100ns.
        """


        save_impl = stream_file_impl(n_rep, margin, dac16bit, hfw_rounding, time_low_res)

        super().__init__(
            save_impl=save_impl,
            base_dwell_time=Q_("0.1 µs"),
            length_unit=U_("µm"),
            time_unit=U_("µs"),
        )