"""Tests of `fibomat.raster_styles.scansequence`."""
import numpy as np
import pytest

from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence
from fibomat.raster_styles.one_d import Curve
from fibomat.raster_styles.scansequence import _apply_scan_sequence, _make_scan_index_sequence
from fibomat.shapes import Line
from fibomat.shapes._line_non_continuous import LineNonContinuous
from fibomat.units import unit


def sequence(n_lines, repeats, scan_sequence):
    indices, directions = _make_scan_index_sequence(n_lines, repeats, scan_sequence)
    return indices.tolist(), directions.tolist()


class TestScanSequenceEnum:
    def test_parse(self):
        assert ScanSequence.parse('serpentine') is ScanSequence.SERPENTINE
        assert ScanSequence.parse(ScanSequence.BACKSTITCH) is ScanSequence.BACKSTITCH

    def test_parse_unknown(self):
        with pytest.raises(ValueError, match='serpentine'):
            ScanSequence.parse('foo')

    def test_values(self):
        assert {member.value for member in ScanSequence} == {
            'consecutive', 'backstitch', 'back_and_forth', 'serpentine', 'double_serpentine',
            'double_serpentine_same_path', 'crossection'
        }


class TestIndexSequence:
    def test_consecutive(self):
        assert sequence(3, 2, ScanSequence.CONSECUTIVE) == ([0, 1, 2, 0, 1, 2], [True] * 6)

    def test_backstitch_even(self):
        assert sequence(4, 2, ScanSequence.BACKSTITCH) == ([1, 0, 3, 2, 1, 0, 3, 2], [True] * 8)

    def test_backstitch_odd(self):
        assert sequence(3, 1, ScanSequence.BACKSTITCH) == ([1, 0, 2], [True] * 3)

    def test_crossection(self):
        assert sequence(3, 5, ScanSequence.CROSSECTION) == ([0, 1, 2], [True] * 3)

    def test_serpentine(self):
        assert sequence(3, 2, ScanSequence.SERPENTINE) == (
            [0, 1, 2, 0, 1, 2], [True, False, True, True, False, True]
        )

    def test_double_serpentine(self):
        assert sequence(3, 3, ScanSequence.DOUBLE_SERPENTINE) == (
            [0, 1, 2, 2, 1, 0, 0, 1, 2], [True, False, True, False, True, False, True, False, True]
        )
        assert sequence(2, 2, ScanSequence.DOUBLE_SERPENTINE) == ([0, 1, 1, 0], [True, False, True, False])

    @pytest.mark.parametrize('n_lines', [1, 2, 3, 4, 7])
    def test_double_serpentine_has_no_jumps(self, n_lines):
        # line i goes from (0, i) to (1, i); consecutive exposures must start where the previous one ended
        indices, directions = _make_scan_index_sequence(n_lines, 5, ScanSequence.DOUBLE_SERPENTINE)
        starts = np.where(directions, 0., 1.)
        ends = 1. - starts
        assert np.array_equal(indices[1:], np.clip(indices[1:], 0, n_lines - 1))
        assert np.all(np.diff(indices) ** 2 <= 1)  # neighbouring lines or the same line again
        assert np.all(starts[1:] == ends[:-1])

    def test_double_serpentine_same_path(self):
        assert sequence(3, 2, ScanSequence.DOUBLE_SERPENTINE_SAME_PATH) == ([0, 1, 2, 2, 1, 0], [True] * 6)

    def test_back_and_forth_is_not_a_line_sequence(self):
        with pytest.raises(ValueError):
            _make_scan_index_sequence(3, 1, ScanSequence.BACK_AND_FORTH)

    @pytest.mark.parametrize('scan_sequence', [s for s in ScanSequence if s is not ScanSequence.BACK_AND_FORTH])
    @pytest.mark.parametrize('n_lines', [1, 2, 5])
    def test_lengths_match(self, scan_sequence, n_lines):
        indices, directions = _make_scan_index_sequence(n_lines, 3, scan_sequence)
        assert len(indices) == len(directions)
        assert set(indices.tolist()) == set(range(n_lines))


def rows(n):
    return [LineNonContinuous([Line((0, i), (1, i))]) for i in range(n)]


def apply(scan_sequence, n_rows=3, repeats=1, **kwargs):
    return _apply_scan_sequence(
        rows(n_rows), unit('µm'), scan_sequence, Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE),
        Mill(2. * unit('ms'), repeats), unit('µm'), unit('ms'), **kwargs
    )


class TestApplyScanSequence:
    def test_consecutive(self):
        pattern = apply(ScanSequence.CONSECUTIVE)
        assert pattern.positions.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
        assert np.all(pattern.dwell_times == 2.)

    def test_serpentine_reverses_every_second_line(self):
        pattern = apply(ScanSequence.SERPENTINE)
        assert pattern.positions.tolist() == [[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [1, 2]]

    def test_repeats_are_applied_once(self):
        pattern = apply(ScanSequence.CONSECUTIVE, repeats=3)
        assert len(pattern.dwell_points) == 3 * 6

    def test_crossection_repeats_each_line(self):
        pattern = apply(ScanSequence.CROSSECTION, repeats=2)
        assert pattern.positions.tolist() == [
            [0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1], [0, 2], [1, 2], [0, 2], [1, 2]
        ]

    def test_accepts_strings(self):
        assert apply('serpentine').positions.tolist() == apply(ScanSequence.SERPENTINE).positions.tolist()

    def test_no_rows(self):
        pattern = apply(ScanSequence.CONSECUTIVE, n_rows=0)
        assert pattern.dwell_points.shape == (0, 3)

    def test_units(self):
        pattern = _apply_scan_sequence(
            rows(1), unit('µm'), ScanSequence.CONSECUTIVE, Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE),
            Mill(2. * unit('ms'), 1), unit('nm'), unit('µs')
        )
        assert pattern.positions.round(6).tolist() == [[0, 0], [1000, 0]]
        assert pattern.dwell_times.tolist() == pytest.approx([2000., 2000.])
