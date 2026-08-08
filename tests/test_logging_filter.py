from __future__ import annotations

import logging

from beamng_lidar_bev.__main__ import _QuietBeamNGpy


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, "msg", None, None)


def test_drops_beamngpy_debug_even_when_it_forces_its_own_level() -> None:
    """
    beamngpy's Lidar.__init__ calls setLevel(DEBUG) on the "beamngpy.Lidar"
    logger. An explicit level on a child beats anything set on the "beamngpy"
    parent, so raising the parent's level does nothing and the records still
    reach the handlers. Filtering at the handler is the durable fix.
    """
    logging.getLogger("beamngpy").setLevel(logging.WARNING)
    child = logging.getLogger("beamngpy.Lidar")
    child.setLevel(logging.DEBUG)
    assert child.isEnabledFor(logging.DEBUG), "parent level does not gate the child"

    quiet = _QuietBeamNGpy()
    assert not quiet.filter(_record("beamngpy.Lidar", logging.DEBUG))
    assert not quiet.filter(_record("beamngpy.BeamNGpy", logging.INFO))


def test_keeps_beamngpy_warnings_and_all_app_records() -> None:
    quiet = _QuietBeamNGpy()
    assert quiet.filter(_record("beamngpy.Lidar", logging.WARNING))
    assert quiet.filter(_record("beamngpy.Lidar", logging.ERROR))
    assert quiet.filter(_record("beamng_lidar_bev.worker", logging.INFO))
    assert quiet.filter(_record("beamng_lidar_bev.main_window", logging.DEBUG))
