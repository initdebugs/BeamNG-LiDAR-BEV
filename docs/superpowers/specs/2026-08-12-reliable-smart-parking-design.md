# Reliable Smart Parking Design

## Goal

Make automatic parking plan against the live scene, commit to a stable bay,
track a continuous drivable trajectory, recover only for explicit reasons,
and finish only after the vehicle is stopped, aligned, secured with the
parking brake, and reported complete.

## Scope

This design implements review items 1–7: parking-specific perception wiring,
a real parking job, verified completion, Hybrid A* correctness, swept-body
collision checking, a consistent live corridor check, and trajectory tracking.
Automatic bay ranking, unmarked-bay perception, and dynamic actor prediction
remain follow-on work.

## Architecture

The existing detector remains responsible for candidate bays. On engagement,
the worker copies the selected `ParkingBay` into an immutable world-space
`ParkingJob`; rescans may refresh the overlay but cannot move the active goal.

A parking-specific world-cell memory receives observed road/free returns and
planner-band obstacle returns whenever parking is active. It projects a local
`Occupancy` snapshot for planning. Parking activates obstacle extraction
independently of self-driving and AEB, and an occupied bay is refused.

Hybrid A* searches `(right cell, forward cell, heading bin, gear)`. It checks
the complete swept footprint of every primitive, skips stale frontier entries,
and returns poses, directions, total cost, and search diagnostics. The partial
Reeds-Shepp implementation is retained only as a collision-checked analytic
shortcut; the search heuristic is an admissible lower bound and does not
depend on incomplete word coverage. Direction cusps retain a shared endpoint
in both adjacent legs.

Every maneuver leg carries its committed bay-relative path. The controller
projects the current vehicle pose onto that path, maintains monotonic progress,
and combines planned-curvature feed-forward with cross-track and heading
feedback. It replans only when the path becomes blocked, tracking error exceeds
the configured envelope, or progress stalls. Replanning is bounded and never
changes the latched bay.

## Collision model

Planning and execution share an oriented vehicle-footprint sampler. Samples
include the configured parking body margin and are spaced no farther apart
than half an occupancy cell. Primitive arcs are interpolated before checking,
so a thin obstacle cannot fall between Hybrid A* endpoints.

The live path check starts at the controller's current progress, returns
distance ahead rather than distance from the original path origin, and ignores
already-passed obstacles. A blockage latches braking until the vehicle stops;
only then may the controller wait for clearance or request a bounded replan.

## Job and terminal states

The externally meaningful job states are `PLANNING`, `EXECUTING`, `WAITING`,
`SECURING`, `SUCCEEDED`, `FAILED`, and `CANCELLED`. Existing display phases may
map onto these states, but `ARRIVED` means `SUCCEEDED`, never merely “the path
endpoint was crossed.”

Success requires the full footprint to lie inside the latched bay, lateral and
longitudinal pose error within tolerance, heading within tolerance, speed below
the stop threshold, and those conditions to remain true for a dwell. The final
command applies the parking brake. The worker then emits completion and ends
automatic parking without sending the normal teardown command that releases
the brake.

An unreadable transmission is not treated as confirmation. Direction changes
use signed vehicle speed and reported gear; when the report is unavailable,
motion in the requested direction must confirm engagement before propulsion is
allowed. Unexpected motion produces a stopped failure rather than a guess.

## Error handling

- Missing current perception holds or fails closed; it is never interpreted as
  a clear lot.
- An occupied selected bay is refused before any control command.
- A static blockage triggers a bounded replan after stopping.
- A transient blockage remains in `WAITING` and resumes only after a clear
  dwell.
- Repeated replanning, lost goal evidence beyond its memory allowance,
  transmission disagreement, or lack of progress ends in a stopped `FAILED`
  job with a precise reason.

## Testing

Pure tests directly cover Hybrid A* state identity, stale-node handling, swept
collision checks, cusp continuity, total-cost selection, and Reeds-Shepp
endpoint/gear semantics. Closed-loop tests add moving bay estimates, steering
lag, cross-track deviation, blockage/replan behavior, signed gear motion, and
verified stopped completion. Worker tests prove parking independently activates
perception, passes occupancy into every plan, latches world geometry, rejects
occupied bays, emits completion, and applies rather than immediately releases
the parking brake.

## Non-goals

No machine learning, automatic bay selection, semantic prediction, new GUI,
or generalized road-driving planner is introduced in this change.
