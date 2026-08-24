import pytest

from tools.camera_staging_probe import estimate_staging


def test_consistent_trials_produce_a_fixed_staging_suggestion() -> None:
    estimate = estimate_staging(
        stepped_frames=(2, 2, 2, 2),
        realtime_ms=(151.0, 158.0, 162.0, 155.0),
        camera_frame_ms=60.0,
    )

    assert estimate.accepted
    assert estimate.staging_frames == pytest.approx(2.0)
    assert estimate.staging_seconds == pytest.approx(0.12)
    assert estimate.reason == "four stable trials agree within one camera frame"


def test_contradictory_clocks_do_not_produce_a_global_constant() -> None:
    estimate = estimate_staging(
        stepped_frames=(2, 2, 2, 2),
        realtime_ms=(18.0, 22.0, 20.0, 21.0),
        camera_frame_ms=60.0,
    )

    assert not estimate.accepted
    assert estimate.staging_frames is None
    assert estimate.staging_seconds is None
    assert "disagree" in estimate.reason


def test_exactly_four_successful_trials_are_required() -> None:
    missing_stepped = estimate_staging(
        stepped_frames=(1, 1, None, 1),
        realtime_ms=(61.0, 63.0, 62.0, 60.0),
        camera_frame_ms=60.0,
    )
    missing_realtime = estimate_staging(
        stepped_frames=(1, 1, 1, 1),
        realtime_ms=(61.0, None, 62.0, 60.0),
        camera_frame_ms=60.0,
    )
    extra = estimate_staging(
        stepped_frames=(1, 1, 1, 1, 1),
        realtime_ms=(61.0, 63.0, 62.0, 60.0, 61.0),
        camera_frame_ms=60.0,
    )

    assert not missing_stepped.accepted
    assert "four successful" in missing_stepped.reason
    assert not missing_realtime.accepted
    assert "four successful" in missing_realtime.reason
    assert not extra.accepted
    assert "four successful" in extra.reason


def test_each_unstable_clock_is_rejected_independently() -> None:
    unstable_stepped = estimate_staging(
        stepped_frames=(1, 3, 2, 1),
        realtime_ms=(61.0, 62.0, 63.0, 60.0),
        camera_frame_ms=60.0,
    )
    unstable_realtime = estimate_staging(
        stepped_frames=(1, 1, 1, 1),
        realtime_ms=(61.0, 180.0, 122.0, 60.0),
        camera_frame_ms=60.0,
    )

    assert not unstable_stepped.accepted
    assert "stepped trials are not stable" == unstable_stepped.reason
    assert not unstable_realtime.accepted
    assert "real-time trials are not stable" == unstable_realtime.reason
