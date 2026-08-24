from __future__ import annotations

from pathlib import Path

APP_NAME = "BeamNG LiDAR BEV"
APP_VERSION = "1.2.0"

# The 0.39 install. beamngpy 1.36 speaks bridge protocol v1.27, which is what
# BeamNG.tech 0.39.x answers with -- against the 0.38.5 install the handshake
# refuses outright (its bridge is v1.26), so the pin in requirements.txt and
# this path move TOGETHER. Point this back at v0.38.5.0 only alongside
# beamngpy 1.35.x.
BEAMNG_EXE = Path(r"C:\Users\initd\Documents\BeamNG.tech.v0.39.4.0\BeamNG.tech.exe")
BEAMNG_HOME = BEAMNG_EXE.parent
BEAMNG_HOST = "127.0.0.1"
BEAMNG_PORT = 64256

# Horizontal cull radius, measured from the vehicle reference node. Sized for
# the FRONT sensor, which reaches much further than the other three -- see
# LIDAR_FRONT_MAX_DISTANCE_M. The side and rear sensors simply stop at their own
# shorter range; nothing is truncated.
LIDAR_RANGE_M = 190.0
# Slant range requested from the side and rear sensors. These feed the BEV
# picture and the planner, neither of which reasons past 35 m, so there is
# nothing to buy by making them reach further.
LIDAR_MAX_DISTANCE_M = 120.0

# --- The long-range forward sensor --------------------------------------------
#
# AEB has to act from far enough out to stop from motorway speed, and that is a
# question about ONE direction. The front unit is therefore configured
# differently from the other three: further, narrower, and denser.
#
# The narrowing is the part that matters, and it is not a cost saving -- it is
# what makes long range work at all. The four sensors share a ray budget set by
# LIDAR_DENSITY, spread over LIDAR_HORIZONTAL_FOV_DEG x LIDAR_VERTICAL_FOV_DEG.
# Measured, that is ~12.4k returns per sensor over 256 channels, so roughly 48
# azimuth samples across 170 deg: 3.5 deg apart, which is 9.3 m of gap between
# ray columns at 150 m. A car is 2 m wide. The vertical spacing over the same
# cloud is 0.118 deg (0.31 m at 150 m) -- thirty times finer -- so azimuth is
# the entire bottleneck and the only thing worth changing.
#
# 50 deg of sweep at a quarter of the sparsity divisor puts roughly 190 azimuth
# samples across the wedge: ~0.26 deg, or 0.7 m at 150 m, so a car is painted by
# several columns and each column lands several channels on it.
#
# NEEDS A LIVE CHECK. Whether `density` holds the ray COUNT constant as the FOV
# narrows or scales it with solid angle is not documented and was not measured;
# the attach-time log now prints per-sensor return counts and the furthest
# return so the real numbers are one line away. Coverage is unaffected either
# way: the left and right units sweep 170 deg each and already reach to within
# 5 deg of dead ahead, so this wedge fills the gap between them rather than
# being the only thing looking forward.
LIDAR_FRONT_MAX_DISTANCE_M = 200.0
LIDAR_FRONT_HORIZONTAL_FOV_DEG = 50.0
LIDAR_FRONT_DENSITY = 12.5

# --- The roof ground-fill sensor ----------------------------------------------
#
# A fifth unit, and like the front one it is a different instrument rather than
# a fourth copy. It exists for ONE job: filling the road surface, which the four
# low units physically cannot do past about 7 m.
#
# Adjacent channels are LIDAR_VERTICAL_FOV_DEG / (LIDAR_VERTICAL_RESOLUTION - 1)
# apart, and a channel at depression theta meets the ground at r = h / tan(theta),
# so consecutive ground RINGS land dr = (r^2 / h) * dtheta apart. At the 0.20 m
# mount that is 0.50 m at 7 m, 4.11 m at 20 m and 25.7 m at 50 m against a 0.5 m
# WORLD_CELL_SIZE_M -- the road arrives as concentric arcs with empty bands
# between them, and NO amount of extra rays closes them, because the extra rays
# go to azimuth. Only two things do: ego motion sweeping the rings through the
# world (see WORLD_CELL_MEMORY_M), and mount height.
#
# The fix is a narrow aperture aimed DOWN, pinned to a ground annulus in metres
# rather than to an angle, and derived per vehicle in derive_vehicle_geometry:
#
#   near_deg = atan(h / LIDAR_ROOF_NEAR_M),  far_deg = atan(h / LIDAR_ROOF_FAR_M)
#   vertical aperture = near_deg - far_deg,  aim = (near_deg + far_deg) / 2
#
# Derived, not constant, because the annulus is what has to be right and a fixed
# angle does not deliver it: at a 1.45 m saloon roof a 13 deg / 7.5 deg aperture
# starts at 6.0 m, but on a 2.0 m van the same angles start at 8.5 m -- past the
# ~7 m the low units cover, so the two sets leave a blind RING around the car.
# Pinning the metres instead makes the aperture track the roof.
#
# Ring spacing then falls out as, at 256 channels over that span:
#   0.06 m at 10 m, 0.24 m at 20 m, 0.54 m at 30 m, 1.5 m at 50 m, 3.9 m at 80 m
# i.e. sub-cell out to ~29 m in a SINGLE frame, against 7 m today.
#
# Note what that arithmetic says, because it is not what you would guess: with
# the annulus pinned, dtheta scales with h and dr = (r^2 / h) * dtheta cancels it
# -- ring spacing is a property of the ANNULUS and the channel count, and is
# near enough independent of how high the unit sits. Height is not buying the
# sampling. Height buys two other things, and they are the real case for it:
#
#   Occlusion. Shadow behind an occluder of height a at distance d is
#   d*a/(h-a), which blows up as h approaches a -- and at 0.20 m everything is
#   as tall as the sensor. A 0.15 m kerb 10 m out shadows 30 m of road behind it
#   at 0.20 m against 1.2 m at 1.6 m; the road's own 0.10 m crown at 20 m
#   shadows 20 m against 1.3 m. That camber shadow is believed to be a large
#   part of what reads as "gaps in the road", and no ray count touches it.
#
#   Pitch robustness. The same annulus at 0.20 m would need a 1.8 deg aperture,
#   and a road car pitches 1-3 deg under ordinary braking -- the whole ring set
#   would swing off the ground every time you slowed down. At roof height the
#   aperture is ~13 deg and the same pitch is a minor perturbation.
#
# Two things this deliberately does NOT do. It does not see building faces --
# every ray points below the horizon, and the four low units already sweep to
# +15 deg, so reaching higher is a LIDAR_VERTICAL_FOV_DEG question, not a
# mount-height one (a 1.6 m mount buys only +1.3 m of facade at 20 m). And it
# does not out-look a car: d*a/(h-a) is still unbounded when a >= h.
#
# COSTS RAYS, and that is the honest trade. At density 25 it is ~2 standard
# units of ray budget on top of the existing ~7 (front counts 4), so roughly
# +28% total. It ran at 12.5 (+57%) first; halving it halves only the azimuth
# sampling (~0.9 m -> ~1.8 m of stripe spacing at 30 m) while the radial ring
# spacing -- the thing this unit exists for -- is set by the channel count and
# is untouched. The wider stripes are what WORLD_ROAD_BRIDGE_CELLS went 3 -> 4
# to close. The attach-time `Sensor reach:` line prints this unit's own return
# count and furthest return, which is the live check.
#
# Slant range, and it must clear LIDAR_ROOF_FAR_M with margin: the far edge of
# the annulus is ~100.01 m of slant from a 1.6 m roof, plus the mount sits off
# the reference node, so 100.0 exactly would cull the outermost rings.
LIDAR_ROOF_MAX_DISTANCE_M = 110.0
LIDAR_ROOF_HORIZONTAL_FOV_DEG = 170.0
LIDAR_ROOF_DENSITY = 25.0
# The far-field lever, same as the global LIDAR_VERTICAL_RESOLUTION and just as
# nearly-free: channels decide how the ray budget spreads across the aperture,
# not how many rays there are. Ring spacing is dr = (r^2/h)*dtheta; over the
# 6-55 m annulus at 512 channels that is 0.09 m at 20 m and 0.86 m at 55 --
# inside the mesh bridge over the whole span this unit serves. NEEDS A LIVE
# CHECK like everything optical: 512 over a ~13 deg aperture is finer than
# anything measured on this engine so far; the `Sensor reach:` line settles
# whether it delivers.
LIDAR_ROOF_VERTICAL_RESOLUTION = 512
# The ground annulus the aperture is fitted to, in metres. NEAR must stay under
# the ~7 m the 0.20 m mounts resolve in a single frame or the two sets leave a
# blind ring. FAR was 100 and is now 55: an equal-angle aperture spends its
# channels quadratically close-in -- over 6-100 m, 74% of the channels landed
# inside 20 m (0.03 m rings the quarter-metre grid cannot even use) and ~33
# covered the whole 50-100 m stretch, which is why the far road stayed thin
# however many channels the span got. The far field now belongs to the ROAD
# unit below; this one owns the near bowl and the TERRAIN out to
# WORLD_SURFACE_RADIUS_M, which is what 55 matches. Over 6-55 the same 512
# channels put rings 0.11 m apart at 20 m and 0.86 m at 55 -- everything this
# unit still serves is inside the mesh bridge.
LIDAR_ROOF_NEAR_M = 6.0
LIDAR_ROOF_FAR_M = 55.0

# --- The forward road-scan sensor ---------------------------------------------
#
# A sixth unit, and the answer to "the road ahead is undetailed past ~50 m". It
# is NOT more of the roof unit: the roof unit's job is the ground bowl all
# around the car, and an equal-angle aperture over a wide annulus starves the
# far rings by construction (see LIDAR_ROOF_FAR_M above). This one squeezes its
# whole channel budget into the 20-100 m annulus -- a ~3.7 deg aperture, so 512
# channels put ground rings 0.20 m apart at 50 m and 0.78 m at 100 m,
# sub-bridge in a SINGLE frame the whole way out -- and its sweep is a forward
# wedge rather than a 170 deg fan, which concentrates the azimuth budget where
# the road actually is: ~0.9 deg columns are 0.8 m stripes at 50 m and 1.55 m
# at 100 m, against the 1.5 m the road mesh can bridge.
#
# NEAR overlaps the roof unit's annulus (20 < 55) so there is no seam, and the
# wedge is 80 deg because a road plus its verges at 100 m subtends far less --
# widening it would spend azimuth on terrain the roof unit already owns.
# Budget: ~+1.9 standard units at density 12.5 (~+15%), the same narrowness
# trade the long-range FRONT unit makes for AEB. Every figure here needs the
# live `Sensor reach:` check like the rest of the optics.
# THE VISUAL-PAINT EXPERIMENT, RETIRED: measured 2026-08-10 and the channel is
# not the scene. With is_annotated=False the colour channel carries BeamNG's
# own point-cloud visualisation colouring -- a rainbow ramp over RANGE
# (corr(range, R) -0.90, corr(range, B) +0.88 on a marked-road scan; the
# rendered probe image is concentric bands with no trace of the lane lines the
# road visibly had). No albedo, no intensity, so paint cannot be read from
# this sensor by brightness, on this engine version, full stop. Annotation
# remains the only channel that sees paint.
#
# The flag and the probe stay so the conclusion is re-testable on a future
# BeamNG: flip this on, attach on a marked road, and read the one-shot
# `Colour check:` line plus logs/road_colour_probe.npz. While on, the road
# unit's returns carry no semantic labels and reach the road store through
# the ground-fallback band.
LIDAR_ROAD_VISUAL_COLOUR = False
LIDAR_ROAD_MAX_DISTANCE_M = 110.0
LIDAR_ROAD_HORIZONTAL_FOV_DEG = 80.0
LIDAR_ROAD_DENSITY = 12.5
LIDAR_ROAD_VERTICAL_RESOLUTION = 512
LIDAR_ROAD_NEAR_M = 20.0
LIDAR_ROAD_FAR_M = 100.0

# Above the bounding-box top, so the mount tracks the vehicle rather than
# assuming a saloon. Sensor pos is referenced to the vehicle ground plane (see
# SENSOR_HEIGHT_ABOVE_GROUND_M), so this is added to the bbox HEIGHT, never to
# the bbox bottom.
ROOF_SENSOR_CLEARANCE_M = 0.12
# 179 is accepted by the engine but sits on the depth pre-pass's rectilinear
# tan() cliff: measured 7389 returns / only 3053 unique at 179, against
# 7826 / 5613 at 170. Four 170-degree wedges 90 degrees apart still union to a
# full 360-degree ring with overlap to spare.
LIDAR_HORIZONTAL_FOV_DEG = 170.0
# Total (not half) vertical aperture, centred on the horizontal mount direction.
# Do not narrow this: at a 0.20 m mount the bottom channel is what sees the kerb
# directly beside the car. See SENSOR_HEIGHT_ABOVE_GROUND_M.
LIDAR_VERTICAL_FOV_DEG = 30.0
# The far-field lever, and nearly free: measured live, vertical_resolution does
# NOT change the total point count -- density sets the ray budget, and this
# decides how those rays are spread vertically. Ground reach is set by the
# channel nearest horizontal, reach = height / tan(vFov / (vRes - 1) / 2):
#   32 channels over 30 deg -> nearest 0.484 deg -> ground stops at 24 m
#  256 channels over 30 deg -> nearest 0.059 deg -> ground reaches well past 100 m
# while the bottom channel still lands at 0.75 m for kerb detection. Measured at
# a 0.20 m mount, 128 -> 256 took returns beyond 50 m from 535 to 747 at an
# unchanged ~49.8k total.
LIDAR_VERTICAL_RESOLUTION = 256
# Sparsity divisor, NOT a density: 1 = dense, 100 = sparse. Point count scales
# as 1/density (measured 49,708 at 50 and 99,736 at 25, four sensors combined).
# This is the one dial to turn for more returns everywhere; the cost lands in
# two places, one per side. Sim-side, ray tracing on the GPU. App-side, every
# O(cloud) pass -- semantics, the AEB shape tests, the scene stores -- which is
# why it was held at 50 until the state-poll prefetch and the async store
# refresh opened CPU headroom. 35 buys ~+43% on the three standard wedges.
# NEEDS THE LIVE CHECK: read VISIBLE POINTS, POLL TIME and SCENE BUILD after
# attach, and drop back to 50 if the tick stops fitting.
LIDAR_DENSITY = 35.0
LIDAR_UPDATE_HZ = 30.0
LIDAR_UPDATE_TIME_S = 1.0 / LIDAR_UPDATE_HZ

# Measured from the vehicle's ground plane, and passed to the Lidar constructor
# VERBATIM -- the simulator already references sensor pos to that plane, so
# adding the bbox bottom on top buries the sensor underground. See
# geometry.derive_vehicle_geometry.
#
# 0.20 m is deliberate and load-bearing: a sensor this low grazes the road
# surface, so a 0.10-0.15 m kerb both breaks the height profile and casts a long
# occlusion shadow. A roof mount looks down onto kerbs, never sees their face,
# and casts no shadow. Reach is recovered through LIDAR_VERTICAL_RESOLUTION
# instead of by raising the sensor.
SENSOR_HEIGHT_ABOVE_GROUND_M = 0.20
SENSOR_BODY_CLEARANCE_M = 0.08

# --- Vision mode: the eight-camera rig ------------------------------------------
#
# Rung 0 of the vision-only ladder (docs/VISION_MODE_SPEC.md): eight streaming
# COLOUR cameras in a Tesla HW4-style layout, rendered as a live grid. Depth and
# annotation channels stay off at this rung on purpose -- each extra channel has
# a measured simulator cost (annotation is a second full geometry pass: 42 Hz ->
# 33 Hz sim rate on the reference machine; depth roughly doubles the bytes
# copied per read) and nothing consumes them yet. The unprojection rung turns
# them on per-mount when it lands.
#
# Resolution is nearly free (measured: 8 cams 320x240 -> 1280x960 costs
# 21 -> 16 Hz -- the cost is per-camera draw submission, not per-pixel), so this
# is NOT the dial to reach for if the sim rate drops; drop a camera instead.
#
# Back to 640x480 on 2026-08-23, and the history matters because the obvious
# reading of it is wrong. It was raised to 1280x960 in response to "pixelated",
# then the real defect turned out to be that `vision_view` was reading the
# camera buffer's fourth byte as OPACITY (see its `_IMAGE_FORMAT`): every frame
# was being composited against the dark tile, which reads as both softness and
# heavy speckle. With that fixed the extra pixels may be buying nothing, so the
# cheaper setting is being tried again on its merits.
#
# Measured either way on this machine, so the trade is known rather than
# guessed: sim-side 18.6 Hz per camera at 640x480 against 16.2 at 1280x960
# (live, the app read 17.5 Hz and 12.1 Hz respectively under load), and the
# worker's own per-tick copy of all eight buffers 1.31 ms against 4.41 ms of
# the 40 ms tick -- 9.8 MB a tick against 39.3, since every camera is copied
# every tick whether or not it delivered a new frame.
#
# 960x720 since 2026-08-23: 640 was tried once the alpha bug was fixed and the
# detail was judged too low, so this is the middle setting -- 2.25x the pixels
# of 640x480 and 0.56x those of 1280x960, with the copy cost scaling the same
# way (roughly 2.9 ms a tick for all eight). This is a display-quality dial with
# a known cost, not a correctness one. If the copy ever binds at a
# higher setting, digest the live buffer BEFORE copying and copy only the
# cameras that changed -- they update at ~16-18 Hz against a 25 Hz tick.
CAMERA_RESOLUTION = (960, 720)
# MUST be positive, and this is a trap, not a tuning choice: with
# is_streaming=True and requested_update_time=0.0 every shared-memory buffer
# stays zero-filled forever while the read loop happily spins -- a working rig
# producing black frames, measured live on BeamNG 0.39.4 / beamngpy 1.36.
# 0.05 asks for 20 Hz; measured delivery for 8 colour cameras at 640x480 is
# ~18 Hz, so the request is not the bottleneck.
CAMERA_UPDATE_TIME_S = 0.05
CAMERA_NEAR_FAR_PLANES = (0.05, 300.0)

# --- Vision mode rung 1: LiDAR-first hybrid A-pillar camera pair -------------
HYBRID_CAMERA_RESOLUTION = (1280, 960)
HYBRID_CAMERA_UPDATE_TIME_S = 0.10
HYBRID_CAMERA_HFOV_DEG = 105.0
HYBRID_CAMERA_YAW_DEG = 37.0
HYBRID_CAMERA_PITCH_DEG = 7.0
HYBRID_CAMERA_HEIGHT_FRACTION = 0.88
HYBRID_CAMERA_FRONT_FRACTION = 0.25
HYBRID_CAMERA_BODY_CLEARANCE_M = 0.12

# Per-mount horizontal FOVs. The Camera constructor takes a VERTICAL field of
# view (field_of_view_y); geometry.camera_vertical_fov_deg derives it from
# these and the aspect ratio, because the horizontal aperture is what the rig
# is designed around. Wide rectilinear apertures sit on the same tan() cliff
# the LiDAR's 179-degree sweep did, so nothing here goes past 110.
CAMERA_FRONT_MAIN_HFOV_DEG = 50.0
CAMERA_FRONT_WIDE_HFOV_DEG = 100.0
CAMERA_FRONT_BUMPER_HFOV_DEG = 110.0
# 90/90 rather than Tesla's 90/60, and the repeater aim moved with it, because
# the eight apertures have to TILE THE CIRCLE and at 80/60 they did not: the
# union left a 24.5-degree hole per side at bearings 95-120 -- over the
# driver's shoulder, which is exactly the blind spot the repeaters exist for.
# Reported live as the side FOVs feeling too narrow, and it is measurable
# rather than a matter of taste. Widening the pillar alone leaves 20 degrees;
# both to 90 leaves 5. With the repeaters re-aimed to 45 degrees off rearward
# the union closes with a 10-degree overlap either side, and 135 degrees is a
# better blind-spot bearing anyway. Rearward is unaffected: the 110-degree rear
# camera spans 125-235. `test_the_rig_leaves_no_gap_all_the_way_round` pins it.
CAMERA_PILLAR_HFOV_DEG = 90.0
CAMERA_REPEATER_HFOV_DEG = 90.0
# 130 since 2026-08-23, up from 110, and PITCHED DOWN (CAMERA_REAR_PITCH_DEG):
# the rear camera is the reversing camera, and a reversing camera's job is the
# ground immediately behind the bumper -- the kerb, the bollard, the neighbour's
# wing -- which a level 110-degree view cut off at the frame bottom a couple of
# metres out. Widening past the 110 the other mounts stop at costs centre
# density (3.9 px/deg at 960 wide against 5.9), which is affordable here because
# nothing long-range is asked of this camera: forward AEB range comes from the
# windshield pair, and the rear brake arms at a crawl. The 16:12 vertical
# aperture at 130 wide is ~116 degrees, so pitched 15 down the frame still
# reaches 43 degrees above the horizon (a following car's windscreen at 5 m)
# and the bottom edge meets the ground ~0.3 m behind the lens.
CAMERA_REAR_HFOV_DEG = 130.0
CAMERA_REAR_PITCH_DEG = 15.0
# B-pillar cameras look forward-outboard, repeaters (front fenders) look
# rear-outboard -- the HW4 pattern. Yaw is measured from straight ahead
# (pillars) and from straight behind (repeaters).
CAMERA_PILLAR_YAW_DEG = 55.0
# 45, not Tesla's ~30: see the FOV note above -- the aim is what closes the
# blind-spot gap cheaply, and it costs nothing rearward.
CAMERA_REPEATER_YAW_DEG = 45.0
# The bumper camera gets its OWN standoff, well past the ordinary
# SENSOR_BODY_CLEARANCE_M: reported live (2026-08-23) that 0.08 m beyond the
# bounding-box face still landed INSIDE the bumper shell -- the OOBB extreme is
# set by the widest point of the car, not by the bumper face at camera height,
# so the curved shell can enclose a point "outside" the box. The camera is
# invisible in-world, so a generous standoff costs nothing. If the rear camera
# is ever reported dark, it wants the same treatment.
CAMERA_FRONT_BUMPER_STANDOFF_M = 0.30

# --- Vision mode rung 0.5: engine-depth unprojection -----------------------------
#
# Phase 2 of docs/VISION_ROADMAP.md. Every camera renders the DEPTH and
# ANNOTATION channels beside colour, and the worker rebuilds the perception
# waist (`points_world + colours + state`) from them through `unprojection.py`,
# so the planner, both AEBs, the BEV and WORLD all run on the camera rig.
#
# Phase 1's verdict made this the PERMANENT ground-band source rather than
# scaffolding: computed stereo resolved a kerb at 15 m and nowhere beyond it
# (measured 2026-08-23, tools/kerb_experiment.py), so engine depth keeps kerbs
# and the road surface for good; stereo, when it lands, takes obstacles only.
#
# Measured costs the rung is budgeted against (spec section 2): annotation is a
# second full geometry pass per camera (sim 42 -> 33 Hz for eight), depth
# doubles the bytes the simulator writes. The worker does NOT copy the depth or
# annotation buffers -- it gathers only the strided sample below straight from
# the live shared memory, one vectorised read per channel per camera.
#
# Per-camera (column, row) sample strides. Rows are the RANGE axis for ground
# seen from a camera: a row at depression theta meets the ground at h/tan(theta)
# and consecutive sampled rows land (r^2/h) * dtheta apart, exactly the LiDAR
# ring-spacing arithmetic with the row stride standing in for the channel
# pitch. So the windshield main camera -- the only long-range instrument in the
# rig -- keeps a finer row stride than anything else: at 960x720 / 50 deg its
# focal length is ~1029 px, so a row stride of 2 is 0.11 deg and the ground is
# sampled every 0.6 m at 20 m from the 1.3 m windshield height. The wide and
# side cameras are context and near field, where a coarser lattice is ample.
# Budget at 960x720: ~280k sampled pixels across the rig, of which roughly half
# are sky and far plane and are culled before anything else runs, landing near
# the six-unit LiDAR rig's 100-150k. Raise a stride before touching the
# resolution if the tick binds; the `Unprojection check:` line reports the
# per-camera counts that decide it.
CAMERA_SAMPLE_STRIDES = {
    "front_main": (4, 2),
    "front_wide": (6, 3),
    "front_bumper": (6, 4),
    "pillar_left": (6, 4),
    "pillar_right": (6, 4),
    "repeater_left": (8, 4),
    "repeater_right": (8, 4),
    "rear": (6, 4),
}
CAMERA_DEFAULT_SAMPLE_STRIDE = (6, 4)
# The far-road row band, front_main only: every image row where LEVEL ground
# from 20 to 100 m lands is sampled at full density, overriding the coarse
# row stride there. Rows are the range axis and ring spacing goes as
# (r^2/h) x row pitch, so stride-2 rows put rings 2.3 m apart at 40 m against
# WORLD's 1.5 m road bridge -- the street oracle capture (2026-08-24,
# tools/oracle_data/street.npz) measured the camera ground band ACCURATE to
# -1..-2 cm against the LiDAR floor on every ring out to 60 m but STARVED
# past 20 m (~175 returns per 4 m ring at 20-24 m against the road-scan
# unit's ~1300, ~30 by 50 m): density, not accuracy, is the binder. The whole
# 20-100 m band is ~54 rows just under the horizon (planar geometry,
# image y = h/r), so full density there costs ~7k samples on a 283k lattice
# and moves the single-frame road edge from ~30 m to ~45 m, where stride-1
# rings outrun the bridge. Beyond that, accumulation while driving fills the
# road exactly as it did for the pre-road-scan LiDAR rig.
CAMERA_FAR_ROAD_BAND_M = (20.0, 100.0)
CAMERA_FAR_ROAD_ROW_STRIDE = 1
# The band is fitted to LEVEL ground, and the first milestone-5 drive showed
# what that costs: on a grade the far road climbs OUT of the strip (a 6%
# grade shifts it ~64 rows) and under pitch it swings (1 degree is ~18 rows,
# and a road car pitches 1-3 degrees braking), so on hills the far road lost
# its dense sampling exactly when it was wanted -- the drawn road popped
# between ~10 m and ~40 m, and AEB's coarse-base ceiling lost its floor
# context at range, which is what let tree canopy read as a wall (returns
# 8-14 m up with `ground rise` readings of 5-11 m: the estimator was
# following the canopy). The margin widens the dense strip by this many
# degrees of grade-plus-pitch each way (~36 rows, ~17k samples); grades and
# pitches beyond it fall back to the coarse stride exactly as before.
CAMERA_FAR_ROAD_PITCH_MARGIN_DEG = 2.0
# Depth decodes as raw float32 x far plane, in linear metres of PLANAR Z
# (measured: 10 m read 9.65, 25 m -> 24.17, 50 m -> 49.51). Sky and anything
# past the far plane come back AT the far plane, so a sample within this
# fraction of it is not a surface and is dropped before unprojection -- along
# with anything past LIDAR_RANGE_M, which the downstream cull would discard
# anyway but which this avoids transforming at all.
CAMERA_DEPTH_FAR_FRACTION = 0.98
# A sample nearer than this is bodywork or the lens's own housing (the bumper
# camera sees bonnet; the repeaters see the fender) and never a return.
CAMERA_DEPTH_MIN_M = 0.30
# The fixed part of each camera frame's age: the simulator stages frames with
# no timestamp, and the worker can only measure the part AFTER the buffer
# changed (the digest age). MEASURED ~= 0 on 2026-08-24, two independent ways
# -- tools/camera_staging_probe.py (a swung camera's buffer follows within
# 5-8 ms, which frames staged 1-2 behind could never do) and
# tools/ghosting_probe.py's fence-run regression (+32 +/- 17 ms of total
# speed-scaled age error, of which the probe's own detection latency predicts
# ~17-20) -- so zero is a measured value, not a default. Points are placed
# from the pose the car had `age` ago; at the 40 km/h cap an unmodelled 60 ms
# would be 0.66 m, in the late direction for AEB -- and in a turn, every
# unmodelled millisecond rotates the cloud about the car before it is stamped
# into the world stores.
CAMERA_FRAME_STAGING_S = 0.0
# Whether self-driving, both AEBs and parking may engage on the unprojected
# camera cloud. ON since 2026-08-24 -- roadmap milestone 5's code change,
# earned by the phase-2 measurements: the camera ground band agrees with the
# LiDAR floor to -1..-2 cm on every ring out to 60 m (street oracle),
# registration is measured (staging ~= 0; detection jitter zero-mean and
# half-tick bounded after the seen-time centring), and everything from
# `points_world + colours` on is the LiDAR path's own code. TRUST is still
# gated on the live phantom checklist -- hills, brake dive, bushes, kerbs,
# reverse, plus the two measured sampling differences (low canopy entering
# the planner band from the repeaters; a car's rear glass reading one 0.4 m
# cell nearer than its bumper) -- because the camera lattice is a new
# distribution (no azimuth stripes, density falling as 1/r^2) and every
# LiDAR-era phantom hid exactly there. This constant stays as the SHUT-OFF:
# one flip closes all four slots (self-driving, both AEBs, parking) through
# the one gate, worker-side (_vision_refuses_driving, the load-bearing half)
# and GUI-side (controls_offered) at once.
VISION_DRIVING_ENABLED = True

DISPLAY_RADIUS_M = 105.0
# poll_sensors("state") is a blocking round-trip measured at 32.7 ms (p95 35.3),
# so a 33 ms tick left the worker thread with no headroom at all.
DISPLAY_INTERVAL_MS = 40
MAX_ROAD_RENDER_POINTS = 120_000
MAX_OBSTACLE_RENDER_POINTS = 100_000

# Bridge auto-detection. BeamNG.tech cold-starts in tens of seconds, so a 2 s
# discovery cadence is imperceptible; once up, the probe is only a death
# detector. The timeout stays below the interval so probes can never overlap.
BRIDGE_PROBE_INTERVAL_MS = 2000
BRIDGE_PROBE_STREAM_INTERVAL_MS = 5000
BRIDGE_PROBE_TIMEOUT_S = 0.4
BRIDGE_LOSS_CONFIRMATIONS = 2

ROAD_RGBA = (143, 149, 154, 205)
OBSTACLE_RGBA = (255, 68, 78, 255)

# Tesla-style WORLD view. Scene construction runs off the BeamNG worker and is
# strictly bounded so no visual backlog can steal time from the 40 ms control
# cadence.
#
# The grid sizes here used to be set by what the QML BRIDGE could carry rather
# than by what the sensors resolve. The old ProceduralMesh path took a QML value
# list, so every vertex meant a QVector3D built in a Python loop on the GUI
# thread, every frame. `world_view.SceneGeometry` now hands Qt the numpy buffer
# verbatim through QQuick3DGeometry.setVertexData, which is O(1) Python
# regardless of vertex count, so the cell size is free to follow the DATA.
#
# 0.25 m is what the data supports: with the roof unit the ground ring spacing
# is 0.24 m at 20 m (LIDAR_ROOF_NEAR_M/FAR_M above), so a quarter-metre cell is
# still sub-ring out to about 20 m and accumulation covers the rest.
WORLD_CELL_SIZE_M = 0.25
# How much DRIVING the road surface is remembered for, in metres travelled --
# not in seconds. The detail on screen is ego motion sweeping the ground rings
# through the world, and a wall clock was only ever a proxy for how much of that
# sweep sat in the window. Stop the car and the sweep stops while the clock does
# not, so the window drained and the view collapsed to what one stationary frame
# resolves: concentric arcs with empty bands between them.
#
# This is safe here in a way it is not for a real vehicle: ego pose comes from
# the simulator and is GROUND TRUTH, so there is no odometry drift to decay
# against, and anything that MOVES is class-separated and keeps a wall clock (see
# WORLD_VEHICLE_TTL_S). Static geometry seen a minute ago is exactly as valid as
# geometry from this frame.
#
# 25 m reproduces today's feel at cruising speed -- the old 1.2 s was 13 m at the
# 40 km/h self-driving cap, 17 m at 50 km/h and 33 m at 100 -- and holds
# everything at a standstill, which is the entire point.
WORLD_CELL_MEMORY_M = 25.0
# Output bound on the road store, applied after accumulation. The store used to
# need neither this nor a radius cull because the 1.2 s TTL bounded it by itself;
# with distance-stamped expiry it has to be bounded explicitly, exactly as the
# voxel store is by WORLD_MAX_COLUMNS. Generous: the radius cull at
# WORLD_ROAD_RADIUS_M is what does the real work, and a quarter-metre grid over
# the whole 70 m disc would only reach this if every cell in it were road.
WORLD_MAX_ROAD_CELLS = 120_000
# Vehicle returns get their own, much shorter window, and it is the one that
# stays on the WALL CLOCK. Everything else in the store is static scenery that
# only improves with more looks, but a car MOVES: accumulating it over
# WORLD_COLUMN_MEMORY_M smears one car into a streak of itself, and a car
# crossing in front of a STOPPED ego has to fade even though the odometer is not
# advancing. 0.15 s is a handful of ticks -- enough to ride out a missed scan,
# short enough that a car at 30 m/s smears under half its own length.
WORLD_VEHICLE_TTL_S = 0.15
# How many empty road cells may be bridged between two observed ones when
# MESHING (never in the store). The finer grid resolves the near field properly
# but outruns the sampling at range: ground returns thin as r^2 radially and as
# r in azimuth, so past roughly 20 m they no longer reach every cell and the
# surface arrives as a checkerboard of disconnected quads. 6 cells is 1.5 m,
# sized to the ROAD unit's azimuth stripes -- ~0.9 deg columns are 1.55 m
# apart at 100 m, and azimuth is what bridging has to close because driving
# sweeps rings radially through the world but never sweeps stripes sideways.
# Radial spacing is far inside it everywhere (0.78 m at 100 m from the road
# unit). Still under anything the car could drive through: the narrowest gap
# a 1.8 m-wide body threads is wider than 1.5 m of missing cells.
WORLD_ROAD_BRIDGE_CELLS = 6
# How often the WORLD stores are refreshed, against a view that re-aims every
# snapshot. The build has two halves with very different costs and very
# different needs: folding the cloud into the stores and meshing them
# (~50-60 ms on an accumulated street drive, and WORLD-ANCHORED -- it depends
# on the stores, not on where the car is this instant) against re-projecting
# the cached world meshes into the ego frame, re-tinting and re-aiming the
# camera (a few ms, and the part that must track the car every snapshot or the
# whole scene visibly lags and swims). Refreshing the heavy half on this clock
# and the cheap half on every snapshot is what holds the view at the full
# display rate while the store work runs as fast as one core carries it.
# The cost is that NEWLY OBSERVED geometry appears at this cadence rather than
# the display's -- an eighth of a second of ground freshness, traded for the
# view tracking the car at 25 Hz instead of jerking along at the build rate.
WORLD_STORE_REFRESH_INTERVAL_S = 0.12
# --- How far the view reaches ------------------------------------------------
#
# TWO radii, because the road and the things standing on it are not observable
# to the same distance and pretending otherwise is what forced the earlier
# choice between "reaches far" and "looks solid".
#
# STRUCTURE is observable a long way out. The front unit reaches 200 m, a wall
# is a big vertical target, and azimuth spacing (which goes as r) still puts
# several returns on a building at 150 m. Drawing it is what makes the view feel
# like it can see, and it is the half the driver actually reads at speed.
WORLD_RADIUS_M = 150.0
# The ROAD SURFACE is not, and the bound is the sampling, not taste. Ahead,
# the ROAD unit's 20-100 m annulus puts rings 0.78 m apart at 100 m and its
# 80 deg wedge keeps azimuth stripes inside the 1.5 m bridge, so the road in
# front resolves to the full radius in a SINGLE frame. To the sides and
# behind, the roof unit carries the ground to ~55 m and ACCUMULATION -- rings
# sweeping down the road as the car drives, over WORLD_CELL_MEMORY_M of
# travel -- fills the rest. Past 100 nothing is drawn at all, which is honest:
# nothing was measured there.
WORLD_ROAD_RADIUS_M = 100.0
# ...and the UNPAVED surface stops sooner still. The road reaches further
# because it is driven ALONG -- accumulation over WORLD_CELL_MEMORY_M sweeps
# the rings down its length -- which never happens for the terrain out to one
# side, so the terrain gets roughly what a single frame resolves.
#
# Since the 512-channel roof unit the sampling carries ~58 m single-frame
# (ring spacing 2.99e-4 * r^2 against the 1.0 m bridge), so the COST is the
# binding constraint, not the sampling -- and 55 spends what the async store
# refresh freed: the refresh runs on its own thread now, so a bigger disc
# costs ground freshness (refresh cadence), never view frames or control
# latency. The road is a ribbon; the ground is a disc, and area goes as r^2
# (55/40 is 1.9x the cells), so this is still the constant to move first if
# SCENE BUILD starts logging heavily.
WORLD_SURFACE_RADIUS_M = 55.0
# The ground is ONE height per cell and must be CONNECTED to where the car
# stands (2026-08-23). Promotion hands the ground mesh the lowest short run
# of every 0.125 m column, and four columns land in one 0.25 m cell -- so
# where one column saw the ground under a parked car and its neighbour saw
# only the roof, the cell held BOTH as stacked layers. Measured on a car-park
# capture: 155k ground cells over 65k distinct (x, y), the stacks a median
# 3.3 m apart (car tops, wall tops, hedge tops, building roofs), every one of
# them drawn as a floating patch of floor and every refresh pushed onto the
# slow keyed corner path. The camera rig, which sees the top of everything,
# tripled what the LiDAR produced; the defect itself predates it.
#
# Two rules now. Per (x, y) the candidate NEAREST THE EGO'S OWN GROUND PLANE
# wins (not the lowest: a bridge deck the car is driving on beats the road
# seen beneath it). Then a raster connected-component pass keeps only cells
# reachable from seeds around the car by steps of at most
# WORLD_GROUND_STEP_M between 4-neighbours -- a kerb (0.15 m) and a hillside
# pass, a car roof (1.4 m up in one cell) and a building roof never do. The
# seeds are the cells within WORLD_GROUND_SEED_RADIUS_M of the car whose
# height is within WORLD_GROUND_SEED_HEIGHT_M of the ego plane; if there are
# none (the car is over a hole in the store) nothing is filtered, which is
# exactly the old behaviour. The known cost, in the same direction as the
# slab ceiling's: terrain behind a cliff steeper than 0.5 m per 0.25 m is not
# drawn until the car can see a gentler way onto it.
WORLD_GROUND_STEP_M = 0.5
WORLD_GROUND_SEED_RADIUS_M = 4.0
WORLD_GROUND_SEED_HEIGHT_M = 1.0
# Fragments the bridge could not join -- ground rings at range further apart
# than WORLD_ROAD_BRIDGE_CELLS -- are readmitted in bounded hops: an occupied
# cell within WORLD_GROUND_REACH_CELLS of a kept cell joins it when its height
# fits the step plus WORLD_GROUND_REACH_GRADE over the gap. Measured on the
# car-park capture, the plain component pass dropped 9% of the camera's
# level road at 20-30 m; two 3 m hops recover most of it while a roof 2 m up
# beside the road (0.5 + 0.3 x 0.5 m allowed against 2 m) still fails.
WORLD_GROUND_REACH_CELLS = 12
WORLD_GROUND_REACH_HOPS = 2
WORLD_GROUND_REACH_GRADE = 0.30
# The outer band of the road surface is dissolved into the air rather than cut
# off, because the road stops at its own radius while everything else runs on to
# WORLD_RADIUS_M. A hard rim reads as a cliff -- a drawn edge where there is
# only the end of what the sensors measured.
WORLD_EDGE_FADE_M = 14.0
# --- The ground height field the overlays drape on ----------------------------
#
# Every overlay -- the AEB corridors, the planned path, the navigation route --
# used to be drawn at ONE height, the ego's own ground plane extended flat, so
# on any gradient, crest, dip or camber the drawn road rose straight through it
# and the guidance vanished under the surface it is guidance ABOUT. The field is
# a world-anchored raster of the surface actually being drawn, built in the
# store refresh beside the ground mesh and carried in `_MeshCache`, so `compose`
# can lift each overlay vertex onto it without touching a store.
#
# COARSER than WORLD_CELL_SIZE_M (0.25) on purpose, and by a factor of four.
# This is a height LOOKUP, not geometry: a road's height varies smoothly at the
# metre scale (2% camera is 5 cm across a lane) and the overlay only has to
# clear the surface, so the extra resolution would buy nothing and cost 16x the
# raster. At 1 m the whole WORLD_ROAD_RADIUS_M disc is ~200x200 cells.
WORLD_GROUND_FIELD_CELL_M = 1.0
# How far the field is trusted beyond the cells that were actually observed.
# Ground returns thin with range and the surface has real holes (occlusion, the
# shadow behind a kerb), and a ribbon crossing one must not develop a notch --
# so observation is dilated by this many cells before it decays, and the decay
# itself is smoothed. Past it the overlay blends back to the ego's ground plane,
# which is exactly today's behaviour: the fallback is a fade, never an edge.
WORLD_GROUND_FIELD_FILL_CELLS = 3
# A guard on the raster, not a tuning knob. The stores are already culled to
# WORLD_ROAD_RADIUS_M, so a field wider than this means a store bug rather than
# a big map, and allocating for it would be the wrong answer either way.
WORLD_GROUND_FIELD_MAX_SPAN_CELLS = 512
WORLD_POSE_JUMP_RESET_M = 25.0
# The most of its own thread's time the store refresh may consume when a
# build overruns the cadence. The refresh runs on a one-thread pool in the
# same PROCESS as the worker tick and the compose thread, and its Python-level
# passes hold the GIL; back-to-back 150 ms builds (a big cloud, full stores)
# ran that thread flat out and taxed every other thread's tick. Overrunning
# builds now stretch the interval to last_build / duty, trading ground
# freshness -- which was already lost to the overrun -- for everyone else's
# latency, which was not.
WORLD_STORE_REFRESH_DUTY = 0.6
WORLD_ACTOR_REGISTRY_INTERVAL_S = 1.0
WORLD_ACTOR_STATE_INTERVAL_S = 0.1
# After the simulator REJECTS an actor-state request, how long before the next
# one. BeamNG.tech refuses `vehicles.get_states` in free-roam -- the normal
# workflow -- and until 2026-08-23 the worker asked again every 100 ms and
# logged a traceback each time: measured 39 ms per rejected round trip, plus
# 120 ms for the 1 Hz registry refresh, so the worker THREAD spent roughly
# half of every second blocked on the socket for nothing, which is most of
# what a 10 Hz vision tick looked like. Every round trip on this thread
# delays the tick that follows it, so a known-refused request is not free.
WORLD_ACTOR_RETRY_S = 15.0
WORLD_ACTOR_COAST_S = 0.35
WORLD_ACTOR_FADE_S = 0.8
# The QML actor delegate builds its model UP from its node, so the node is the
# actor's GROUND CONTACT. A simulator actor reports its reference node instead,
# which stands about this far above the road -- the same quantity
# `ground_z_vehicle` measures for the ego, which no other actor reports. This
# used to be a bare `y: model.y - 0.45` in the delegate; it lives here now
# because the LiDAR-fitted boxes report a true base height and must NOT be
# dropped, so the two paths can no longer share one correction in the QML.
WORLD_ACTOR_GROUND_DROP_M = 0.45
WORLD_MAX_UNCERTAIN_POINTS = 2_000

# --- Fitting a box to vehicle returns ------------------------------------------
#
# A traffic car is the one object in this scene that CANNOT be drawn by
# accumulating a surface, and the reason is arithmetic rather than tuning.
# Everything else is forgotten by the metre, so a stopped ego keeps every look it
# ever got and a wall fills in; traffic is forgotten by the second
# (`WORLD_VEHICLE_TTL_S`) because a car MOVES and would otherwise draw as a
# streak of itself. At a standstill that leaves ONE snapshot of evidence, and one
# snapshot of a car at 15 m is four or five azimuth stripes over a metre apart --
# which `_column_runs` then meshes into exactly what it was given: confetti.
#
# So the returns are not meshed at all. **Accumulation is how you build a surface
# whose shape you do not know; a car is an object whose shape you do.** Five
# stripes is nowhere near enough to mesh a surface and ample to fit a footprint,
# so the fit produces a `WorldActor` and the existing delegate draws a car. This
# runs in `compose`, per snapshot, so it also has none of the store's refresh lag.
#
# It is ADDITIVE. The returns still enter the voxel store and still draw as
# solids underneath, so a fit that is wrong or missing degrades to exactly
# today's picture rather than to an invisible car -- the same direction of error
# as every other rule here.

# Clustering happens in the SENSOR's own lattice (azimuth x range), not in world
# XY, and that is what makes one constant work at every distance. Azimuth stripes
# spread as `r`, so a fixed world gap either shatters a car at range or welds two
# parked cars together up close; in polar coordinates the spacing is constant by
# construction. Measured stripe spacing is 1.24 m at 20 m, i.e. 0.062 rad.
VEHICLE_FIT_STRIPE_RAD = 0.062
VEHICLE_FIT_RANGE_CELL_M = 0.5
# Link across this many missing stripes / range cells. Two is enough to bridge a
# stripe that fell in a window gap without reaching across the ~1 m a row of
# parked cars leaves between them.
VEHICLE_FIT_LINK_AZIMUTH_CELLS = 2
# Range links reach FURTHER than azimuth ones, and the asymmetry is measured.
# A flank seen obliquely is crossed by very few stripes, and each one lands
# much further down its length than the last: on a car 3 m off the axis at
# 12 m, its end face and its flank came back 5 range cells apart, so a
# symmetric link split every single car into two fragments.
VEHICLE_FIT_LINK_RANGE_CELLS = 6
# Below this there is no footprint to fit; the returns stay solids and nothing
# is claimed about them.
VEHICLE_FIT_MIN_POINTS = 12
# Where confidence saturates. Confidence rides the delegate's opacity, so it is
# the honesty channel: a car resolved by two stripes is drawn faint.
VEHICLE_FIT_FULL_POINTS = 120
# The footprint is fitted by MINIMUM-AREA RECTANGLE over a swept angle, not by
# PCA. PCA answers the wrong question on the shape a LiDAR actually returns: a
# car seen from behind is an L (rear face plus one flank), and the principal
# axis of an L lies diagonally across it. Minimum area locks onto the faces --
# and on a single flat face it degenerates gracefully to that face's own angle,
# which is precisely the case handled below.
VEHICLE_FIT_ANGLE_STEP_DEG = 2.0
# How close to an edge counts as ON it. The frame is chosen by summing
# `1 / distance-to-the-nearest-edge`, so this floor is what bounds the reward
# and therefore how sharply a frame the returns lie on beats one they lie
# inside. Roughly the noise on a face, and well under any real vehicle
# dimension.
VEHICLE_FIT_EDGE_FLOOR_M = 0.05
# A cluster longer than a vehicle can be is two vehicles, so it is split at its
# largest gap. Bounded below by the stripe spacing so a split never fires on the
# sampling itself.
VEHICLE_FIT_SPLIT_GAP_M = 0.8
# Above this a cluster is asked whether it is two vehicles. Longer than any
# car and shorter than a bus, so a bus is never interrogated and a queue of
# cars always is. The gap alone can never decide it -- at range one car's own
# end face and flank are separated by exactly the hole two parked cars leave
# between them, so the split is taken only when BOTH halves fit whole
# vehicles.
VEHICLE_FIT_SPLIT_LENGTH_M = 6.0
VEHICLE_FIT_MAX_CLUSTERS = 24
# One face and no measurable depth: is this the SIDE of a car or its END? The
# side of any vehicle is longer than this and the end of one is not.
VEHICLE_FIT_SIDE_LENGTH_M = 3.0
# What the unobserved dimension is assumed to be. The box is then pushed AWAY
# from the ego by the amount it was extended, because the face that was seen is
# the near one -- the inference goes behind the evidence, never in front of it.
VEHICLE_FIT_DEFAULT_WIDTH_M = 1.9
VEHICLE_FIT_DEFAULT_LENGTH_M = 4.5
# A box with an inferred dimension is drawn fainter than one measured on two
# faces. It is the same claim the ground-truth actors' confidence makes.
#
# Not lower, and this was rendered before it was chosen: confidence drives the
# delegate's opacity directly, and a parked car alongside the ego almost
# always has an inferred dimension -- its flank is foreshortened to nothing --
# so this value is the NORMAL case, not the marginal one. At 0.72 the near car
# showed the far one through itself and read as a rendering fault rather than
# as uncertainty. A fully measured box still draws at 1.0.
VEHICLE_FIT_ONE_FACE_CONFIDENCE = 0.85
# Carrying an id across frames is what lets the delegate's damping work at all:
# `ActorListModel.set_actors` only avoids a model reset when the id tuple is
# unchanged, and a reset rebuilds the delegate and discards its animation state.
VEHICLE_FIT_TRACK_MATCH_M = 2.5
# A fit this close to an actor the ground-truth path already draws is that same
# car; drawing both would put two models in one place.
VEHICLE_FIT_ACTOR_MATCH_M = 3.0
# Plausible vehicle envelope. A cluster outside it is not refused a drawing so
# much as left to the solids: these bounds are what stop a hedge, a bin or two
# merged cars being asserted to be one vehicle.
# Length-to-width of anything that drives. A small hatchback is 2.2, a van
# 2.4, an artic 3.5 -- so this bounds an over-read width without ever
# touching a shape that is really on the road.
VEHICLE_MIN_ASPECT = 2.2
VEHICLE_MIN_WIDTH_M = 1.2
VEHICLE_MAX_WIDTH_M = 3.2
# The floor on a BELIEVED length. Nothing shorter than VEHICLE_FIT_SIDE_LENGTH_M
# is ever measured as a length (see `vehicle_fit`), so this only ever rejects,
# and it is what stops a stubby corner observation being asserted as a car.
VEHICLE_MIN_LENGTH_M = 3.0
VEHICLE_MAX_LENGTH_M = 14.0
VEHICLE_MIN_HEIGHT_M = 0.6
VEHICLE_MAX_HEIGHT_M = 4.5

# --- Boundary columns (buildings, walls, kerbs) --------------------------------
#
# Boundary returns used to be drawn as one 0.16 x 0.32 m billboard per point,
# rebuilt from the current snapshot only and decimated to 4,000 marks. Three
# things were wrong with that and all three showed as "gappy buildings":
# 70-80% of the wall evidence was thrown away before rendering, the surviving
# points were re-chosen every frame from a re-ordered array so they shimmered,
# and a wall drawn as confetti has gaps between the confetti by construction.
#
# These are now accumulated into world-anchored VOXELS and extruded into slabs,
# which is honest -- "something occupies this box" is exactly what the sensor
# said -- and cheap, because a 20 m facade merges into a handful of boxes
# instead of thousands of cards.
#
# The store is a 3D voxel grid rather than one (base, top) span per XY column,
# and that is not a refinement -- a span cannot represent a TREE. Grass and
# terrain are boundary returns like everything that is not road, so a column
# under a canopy holds returns at ground level AND at 3-7 m. Collapsed to
# min/max that reads as "solid from the grass to the treetop", which is exactly
# how trees rendered: a canopy smeared down to the ground. Voxels keep the void
# in between, so `_boundary_mesh` extrudes RUNS of occupied height and the gap
# survives to the screen. The same bug drew a bridge deck, an overhead sign and
# a tunnel roof as solid columns to the ground.
#
# The gap here is AZIMUTH, not range, and it is the one axis the roof unit does
# not touch. A wall is sampled at r*dtheta vertically and r*dazimuth
# horizontally: at 20 m that is 0.04 m and 1.24 m, thirty to one, so a wall
# arrives as dense vertical stripes over a metre apart. Ego motion sweeps those
# stripes across the wall and fills it in, which is what the TTL is for; the
# single-frame gaps between stripes are closed by WORLD_COLUMN_BRIDGE_CELLS.
#
# 0.125 m, half WORLD_CELL_SIZE_M, and the asymmetry is deliberate: the SLABS
# are what read as blocky and the ground does not, because `_ground_mesh` shares
# lattice corners and so is one continuous surface with no steps to see. A
# slab's drawn thickness tracks this constant exactly -- measured 0.50 / 0.25 /
# 0.15 / 0.12 m of drawn wall at those cell sizes -- so a kerb was a full
# quarter-metre thick and a wall likewise.
#
# It is also the half the SAMPLING supports and the cheaper half of the two.
# Measured on an accumulated street drive (99.5k-point cloud, 70 m of travel),
# against the 120 ms WORLD_STORE_REFRESH_INTERVAL_S cadence:
#
#     ground 0.25, columns 0.25     44 ms      (what this replaces)
#     ground 0.25, columns 0.125    87 ms      <- this
#     ground 0.125, columns 0.25   129 ms
#     ground 0.125, columns 0.125  182 ms      over the cadence
#
# The ground is 2.5x the cost because it is a DISC and area goes as r^2, and it
# is also where refining buys least: the fraction of carriageway a return
# actually hits falls from 46% to 31% over 30-50 m and 27% to 11% over 50-80 m
# when the ground cell halves, so most of what the finer lattice would draw out
# there is `bridge_gaps` inference rather than measurement (5.7% -> 13.9% of the
# surface invented). A wall does not have that problem: it is sampled thirty
# times finer vertically than in azimuth, so the detail is genuinely there and
# was being quantised away.
WORLD_COLUMN_SIZE_M = 0.125
# Vertical quantisation of the voxel store, and it is deliberately COARSER than
# the horizontal size now rather than equal to it. Vertical ring spacing on a
# wall is 0.04 m at 20 m, thirty times denser than azimuth, so the returns would
# carry a finer bin -- but halving this too was measured at 87 ms against 86 ms
# and 164k voxels against 142k, i.e. it buys nothing visible and costs a fifth
# of the store. Coarser than 0.25 starts merging a kerb into the pavement
# behind it, so this is the floor rather than a free parameter.
#
# WORLD_COLUMN_VERTICAL_BRIDGE_BINS is counted in THESE bins, so it is unmoved
# by the horizontal change -- see its own comment.
WORLD_COLUMN_HEIGHT_M = 0.25
# Further than WORLD_CELL_MEMORY_M, and measured the same way -- in metres
# driven, not in seconds. A wall is static scenery that only improves with more
# looks, and the azimuth stripe sweep needs tens of metres of travel to cover the
# gaps at range, which is a distance rather than a duration.
#
# 90 m sits inside what the old 4 s window bought at speed (44 m at the 40 km/h
# cap, 56 m at 50 km/h, 111 m at 100) and well inside WORLD_RADIUS_M, so the
# radius cull in _expire_boundary_columns still bounds the store on every side.
WORLD_COLUMN_MEMORY_M = 90.0
# Output bound, applied after accumulation. Slabs are merged runs, so this is
# generous: a facade costs a handful of boxes, not one per voxel.
#
# It had to move with WORLD_COLUMN_SIZE_M, and this is the failure that makes
# the finer grid a REGRESSION if it is forgotten. The store fills with roughly
# 4x the voxels for the same geometry, and at 90k the cull binds and drops the
# excess OLDEST-FIRST -- which is precisely the accumulated stripe sweep that
# fills a striped facade in, so walls behind the car would start dissolving.
# Measured on the street drive, the store wants 142k at 0.125 m against 58k at
# 0.25; at the old cap it would have discarded 37% of the wall evidence it had
# already paid to collect. 200k leaves headroom for a denser scene and costs
# about 9 MB.
WORLD_MAX_COLUMNS = 200_000
# Input bound, applied before binning. Far higher than the old 4,000 render cap
# because binning is one vectorised pass over the cloud, not per-point geometry.
WORLD_MAX_BOUNDARY_POINTS = 60_000
# Slabs only merge sideways when they sit at the same altitude AND have the
# same height, so a 10 m facade and the 0.15 m kerb in front of it stay
# separate rather than averaging into one waist-high wall.
WORLD_SLAB_HEIGHT_BUCKET_M = 0.5
# How many empty columns may be bridged between two observed ones. At 0.125 m
# columns, 12 spans 1.5 m: azimuth stripes on a wall run past a metre apart at
# range (and wider since the standard wedges went to density 35), and a bridge
# that stops short of the stripe spacing leaves an oriented wall as a row of
# posts -- measured, a 30-degree wall sampled every 1.5 m went from 19
# fragments to one box when the bridge could reach the next stripe. It is an
# inference, but a sound one: a return either side at the same height means
# the ray passed through, so the surface between them was there to be hit.
# Still under anything a car could drive through, and the 4 m doorway case is
# pinned open by test_azimuth_stripe_gaps_are_bridged_but_a_real_opening_is_not.
#
# It is counted in CELLS and the stripe spacing it has to clear is a PHYSICAL
# distance, so it must move inversely with WORLD_COLUMN_SIZE_M or the same gap
# stops being closed: it went 6 -> 12 with the columns at 0.125 m, and 1.5 m is
# unchanged either side of that. Leaving it at 6 is the way to make a finer grid
# look WORSE than a coarse one -- the far field breaks into disconnected
# fragments, which reads as harder blockiness than the cell size ever did.
WORLD_COLUMN_BRIDGE_CELLS = 12
# How many empty height bins may be bridged INSIDE a column before the run is
# treated as two separate structures.
#
# 2 bins is 0.5 m. It has to clear the VERTICAL sampling gap on a wall, and that
# gap grows with range like everything else here: channels are 0.118 deg apart,
# so a wall is sampled every r*dtheta -- 0.10 m at 50 m, but 0.21 m at 100 m and
# 0.31 m at 150 m. At 1 bin the far field fell apart into floating fragments
# exactly where the extra reach was supposed to buy something, because every
# return past ~110 m became its own sub-minimum run and was dropped.
#
# It still has to stay far under the clear air beneath a canopy or a bridge
# deck, which is metres rather than centimetres, so trees still split from the
# grass under them -- test_a_tree_canopy_does_not_reach_down_to_the_ground pins
# both sides of that.
WORLD_COLUMN_VERTICAL_BRIDGE_BINS = 2
# Below this a slab is not worth a box -- it is road noise, not structure.
WORLD_MIN_SLAB_HEIGHT_M = 0.10

# --- Which way a slab faces ---------------------------------------------------
#
# The voxel lattice is world-aligned, so every slab used to be a world-aligned
# box: a wall running diagonally came out as a staircase of cubes and a car
# parked at an angle as a heap of them. The orientation is now MEASURED from the
# footprint and the box is rotated to match.
#
# It cannot be measured inside one column. A 0.125 m cell holds at most 0.125 m
# of evidence, which is far smaller than the azimuth stripe spacing the returns
# arrive with (about a metre and more at range), so the direction has to come
# from a NEIGHBOURHOOD: metre tiles, summed over a sliding 7x7 of them -- a 7 m
# window. It was a 3 m window first, and that is exactly what the staircase
# complaint was: a wall at range arrives as stripe samples over a metre apart,
# so 3 m often held two or three cells, the guards (correctly) refused to fit,
# and the wall fell back to world-aligned confetti -- measured, a 30-degree
# wall sampled every 0.8 m drew as 38 boxes wandering 0.28 m off its line, and
# at 1.5 m samples even 5 m left 10-20 fragments where 7 m leaves 2-5. The
# cost on a curve is the chord sagitta, 49/(8R): 0.10 m at a 60 m radius,
# under the wander it removes; tighter curves fail the anisotropy guard and
# stay world-aligned as before. The window SLIDES because a fixed tile is a
# world-aligned box whose corner a surface can clip, leaving too few cells to
# fit a direction to.
WORLD_ORIENT_CELL_M = 1.0
# Quantised, because merging only ever happens between cells that agree on a
# frame -- the angle is a key field exactly like the altitude and height buckets
# beside it. Over [0, 90) rather than [0, 180): the grid is square, so rotating
# a rectangle by 90 degrees just swaps its sides.
#
# Twelve is 7.5 degrees, so the worst case is a surface 3.75 degrees off the
# frame it is drawn in. Measured on a 20 m wall at the worst angle for each:
#
#     6 buckets   drawn up to 7.50 deg out, 0.12-0.17 m of wander off the line
#    12 buckets   drawn up to 3.75 deg out, 0.02-0.04 m -- flat, to the eye
#
# It is also CHEAPER, which is not the obvious direction: a frame that fits
# merges into fewer, longer boxes, and on a street scene 12 buckets gave 22
# slabs against 33 and 31.3 ms against 33.2 for 6. Quality and cost agree
# here, so the reason not to go further is the fixed cost of a group that
# holds almost nothing, not the geometry. 24 (1.9 deg worst case) went in with
# the 5 m orientation window: the near-axis wander of a road-edge barrier a
# few degrees off the world grid halves with it.
WORLD_ORIENT_BUCKETS = 24
# ...and it is only believed when the footprint actually supports it. Below
# these it falls back to bucket 0, which IS the world-aligned frame, so the
# fallback needs no separate code path.
#
# Anisotropy is (l1 - l2) / (l1 + l2) over the footprint covariance: 1.0 is a
# perfect line, 0.0 a shapeless blob. A bush is a blob and has no direction to
# find; so is the inside corner of an L-shaped building, where the two walls
# average to a 45-degree answer that fits neither -- both fall back, which is
# the honest result rather than an invented rotation.
#
# MIN_CELLS is 4, down from 6 with the window at 5 m: a sparsely-sampled wall
# is exactly the case the orientation exists for, and four collinear stripe
# samples are a direction while four clumped ones fail the anisotropy guard
# beside this one -- the two guards ask different questions.
#
# IT DOES NOT SCALE WITH WORLD_COLUMN_SIZE_M, and the obvious reasoning that it
# should is wrong. The window is 7 m of WORLD (a sliding 7x7 of metre tiles), so
# halving the cell size looks like it must quadruple the cells inside it and
# quietly relax this guard by 4x. Measured, it does not, because a cell count is
# bounded by the RETURNS and not by the lattice: a wall sampled at 1.5 m azimuth
# stripes holds 6 cells in the densest window at 0.25 m and 6 at 0.125 m -- each
# stripe lands in one cell either way. Raising this to 16 to "compensate" would
# therefore have refused every sparse wall and brought back the staircases the
# whole orientation pass exists to remove.
#
# Only a densely-sampled BLOB scales (a bush went 88 cells -> 298), and the blob
# was never this guard's to reject: measured at both sizes it comes back
# oriented 0 of 88 and 0 of 298, because ANISOTROPY is a scale-free ratio and is
# what does that work. The two guards asking different questions is exactly what
# makes the finer grid safe here.
WORLD_ORIENT_MIN_CELLS = 4
WORLD_ORIENT_MIN_ANISOTROPY = 0.6

# --- What counts as something you could hit -----------------------------------
#
# A run of occupied height whose underside clears this is drawn through, not
# driven into: tree canopies, bridge decks, gantries, overhead signs, tunnel
# roofs. Those are culled outright, which is the whole point of the view -- it
# shows what is in the way, not what is overhead.
#
# The reference is the LOWEST boundary return in that same column, so the test
# follows terrain rather than assuming the map is flat. Where the ground under a
# structure was never seen -- occluded by the canopy itself, or road rather than
# boundary under a bridge -- there is no local floor to use and the ego's own
# ground plane stands in. The known cost of that fallback is a structure sitting
# more than this far ABOVE the ego plane on a steep embankment, which is culled
# as though it were overhead; it is drawn again as soon as its own footing comes
# into view. Erring that way is deliberate: the request was for what is in the
# way, and something 3 m above the road plane is not.
WORLD_COLLISION_CEILING_M = 2.6

# --- WORLD palette and depth cueing -------------------------------------------
#
# The palette for every UNLIT surface lives here as sRGB text rather than in the
# QML, because those surfaces are now vertex-coloured: WorldScene.qml ships a
# white base colour and the per-vertex colour carries the palette outright. The
# lit materials (ego, actors, glass) keep their colours in the QML, where the
# shading path can use them; `test_world_palette.py` reads both sources and
# pins that the air colour agrees across the two.
#
# Measured on a real GPU rather than assumed (Qt 6.7.1, D3D11): a NoLighting
# DefaultMaterial multiplies its base colour by the vertex colour in LINEAR
# space, so a white base plus srgb_to_linear(target) reproduces `target`
# EXACTLY -- checked against all five palette entries below. A vertex colour
# above 1.0 clamps to white, so every value written stays in range.
#
# The same probe settled the question CLAUDE.md flagged as undocumented, and it
# went the pessimistic way: SceneEnvironment Fog is a NO-OP on NoLighting
# materials. A screaming-red fog from 20 m to 60 m over geometry spanning
# 5-75 m did not move one red channel. Every large surface here is NoLighting,
# so that fog block was decorative -- distance is cued by WORLD_DEPTH_* below,
# baked per vertex in `world_scene`, which no shader can ignore.
WORLD_AIR_RGB = "#d7dadc"
WORLD_ROAD_RGB = "#6a7176"
# Boundary is a RANGE, not a colour, and that is what stops a wall and the
# building behind it reading as one silhouette. Unlit flat boxes carry no edge
# information whatsoever: two abutting cuboids of one colour are one blob by
# construction. Faces are shaded along this ramp by orientation (see
# WORLD_SLAB_FACE_LIGHT), which gives every slab its own form, and the depth
# tint then separates the two by range on top of that.
WORLD_BOUNDARY_RGB = "#171c20"       # shadowed faces; the ladder's 12.2:1 vs air
WORLD_BOUNDARY_LIT_RGB = "#454f58"   # 2.0:1 above the shadow side
WORLD_UNCERTAIN_RGB = "#545c62"
# Surface materials. Every one of these sits on the ROAD's rung of the ladder
# and separates by HUE, because there is no room for it to do anything else:
# air-to-black is 14.96:1 in total, which supports two 3:1 steps and no more, so
# a material that separated itself by lightness would have to leave the rung and
# collide with either the air or the obstacle band.
#
# The usable band is therefore narrow and it is arithmetic rather than taste --
# 0.1335 to 0.1992 relative luminance for 3:1 both ways -- and every colour here
# was solved in CIELAB at a chosen hue and lightness and then converted back,
# rather than picked by eye. Measured: worst contrast 3.04:1 against the
# obstacle band and 3.08:1 against air, worst pairwise distance dE 7.1 (paved vs
# unknown, which are deliberately the closest pair -- see below).
# `test_world_palette.py` recomputes all of it.
#
# PAVED is WORLD_ROAD_RGB itself, so a road-classified surface is exactly the
# colour the road has always been and nothing about the existing view moves.
# The paved material IS the road colour, aliased rather than copied so the two
# can never drift: every existing contrast fact about the road is a fact about
# a paved surface, and a near-miss beside it would quietly invalidate them.
WORLD_SURFACE_PAVED_RGB = WORLD_ROAD_RGB
WORLD_SURFACE_UNKNOWN_RGB = "#6d6569"
# Deliberately the least distinct, for the same reason WORLD_UNCERTAIN_RGB is
# the weakest mark in the scene: this is ground the sensors resolved but nothing
# identified, so reading as "some sort of surface" is honest. It is the only
# pair below dE 10 and that is a choice, not a rounding error.
WORLD_SURFACE_SIDEWALK_RGB = "#83786a"
WORLD_SURFACE_VEGETATION_RGB = "#5d724f"
WORLD_SURFACE_BARE_RGB = "#85664c"
# Mud, sand, rock and gravel share one warm earth colour. Splitting rock off as
# its own grey was tried and abandoned: paved, sidewalk and rock are all greys,
# and three of them inside a band this narrow are not tellable apart -- rock
# came out dE 6.7 from paved, below the pair this palette already treats as its
# closest. Hard unpaved ground is one material here.
WORLD_SURFACE_WATER_RGB = "#447195"
# Road paint -- lane lines, arrows, crossings -- and the SECOND deliberate
# break in the luminance ladder, for the same reason WORLD_PATH_RGB is the
# first: paint is a graphic ON the road, not a surface silhouetted against the
# air, so its contrast partner is the tarmac it lies on rather than the sky it
# never touches. A near-white (3.4:1 against the road) is what makes it read
# as paint; the in-band paint-yellow tried first read as a road stain. The
# cost is honesty at the rim -- far paint dissolves into the air a little
# early under the depth tint -- which is the correct direction for the least
# structural thing in the scene. Exempted from the air-side ladder assertion
# in test_world_palette with this same argument.
WORLD_SURFACE_MARKING_RGB = "#c6c8c1"
WORLD_PATH_RGB = "#4ea8f2"
WORLD_PATH_ALERT_RGB = "#c0271e"
# The ROUTE ribbon: where navigation says to go, as opposed to the path the
# planner chose this tick. Subordinate to the path by CHROMA and ALPHA, never
# by luminance -- chroma is the overlay channel and the luminance rungs are
# full. A desaturated steel blue in the path's hue family (guidance reads as
# one family), dashed and translucent so the plan always reads over it; the
# exact value is held apart from both path and road in CIELAB by
# test_world_palette, never by eye.
WORLD_ROUTE_RGB = "#587a96"
WORLD_ROUTE_ALPHA = 0.55
# Dash geometry: 2 m of ribbon then 1.5 m of gap -- long enough to read as
# one line in motion, gapped enough never to be mistaken for the solid plan
# ribbon. The half-width is a thread against the path's half-vehicle.
WORLD_ROUTE_DASH_M = 2.0
WORLD_ROUTE_GAP_M = 1.5
WORLD_ROUTE_HALF_WIDTH_M = 0.35
# Parking bays. A bay is an INFERENCE drawn from two painted lines rather
# than something the sensors saw standing there, so like the path and the
# route it separates by CHROMA and never takes a luminance rung -- the two
# steps in the ladder are spoken for.
#
# **Every one of these is OPAQUE, and that is forced by the renderer.**
# Translucent vertex colour does not blend in this scene: measured on the real
# GPU over the road, one flat quad of `#c6c8c1` renders (198,200,193) at
# vertex alpha 1.0 and 0.999 -- correct -- then (220,222,215) at 0.9, which is
# BRIGHTER than opaque, and saturates to pure white at 0.5 and below.
# Premultiplying the colour changes nothing. So the whole visual hierarchy
# here comes from HUE and from GEOMETRY (outline against fill), never from a
# wash, and a bay is drawn as an outline so the road and its own dividers stay
# visible through the middle of it.
#
# Note this puts a question over the AEB overlay, which carries a 0.04 wash
# and a 0.80 rail in one buffer: CLAUDE.md's supporting measurement used a
# BLACK quad, and black is the one colour that cannot tell a correct blend
# from this failure -- both give `background * (1 - a)`.
#
# The free bay borrows the PAINT colour outright, and that is the point: the
# bay is exactly the space its dividers define, so drawing it in the colour of
# its own evidence ties the claim to what supports it.
WORLD_PARKING_FREE_RGB = WORLD_SURFACE_MARKING_RGB
# Occupied bays recede toward the road rather than dimming toward the
# obstacle band: "there is a bay here and it is taken" must not read as
# "there is something solid here", which the boundary colour would say.
WORLD_PARKING_OCCUPIED_RGB = "#7e8489"
# The selected bay, and the one warm hue with no other job in this scene:
# path and route are blue, AEB violet then red, everything else neutral. It
# is also the only bay drawn FILLED -- exactly one ever is, so an opaque fill
# is affordable there and is what makes the choice unmistakable.
WORLD_PARKING_SELECTED_RGB = "#ff9d4d"
# Outline thickness. Wide enough to survive at range (a bay 30 m out is a few
# pixels tall), narrow enough not to close up the bay it outlines.
WORLD_PARKING_BORDER_M = 0.14
# Lift above the draped surface. Over the road mesh but under the paint
# quads' 2 cm, so a bay outline never hides the dividers it came from.
WORLD_PARKING_LIFT_M = 0.012
# The entry chevron: which way the car would drive in, the one property of a
# bay a rectangle cannot show.
WORLD_PARKING_CHEVRON_M = 0.9
# Traffic drawn from LiDAR alone, in the same blue as the corroborated actor
# models, because hue is the only thing separating a car from a wall -- both
# live in the dark obstacle band. See WORLD_VEHICLE_TTL_S: these are the returns
# themselves, drawn whether or not the simulator ever confirms an actor.
WORLD_VEHICLE_RGB = "#22496e"
WORLD_VEHICLE_LIT_RGB = "#4c7ba8"

# --- Emergency-braking overlay ------------------------------------------------
#
# Violet while armed because it is the one hue nothing else in the view uses --
# a neutral grey was tried in the BEV overlay first and vanished into the road.
# Red only once the pedal is actually down, so the colour change IS the event.
WORLD_AEB_ARMED_RGB = "#7d5fc4"
WORLD_AEB_BRAKING_RGB = "#e0342a"
# Every element's transparency is per-VERTEX, so the numbers below are absolute
# rather than relative to a material opacity. Measured on a real GPU (Qt 6.7.1,
# D3D11): vertex alpha multiplies the material's own opacity exactly -- a black
# quad at vertex alpha 1.0/0.5/0.0 over white under a 0.8 material rendered
# 51/153/255, matching alpha*opacity to the pixel. The corridor material is
# therefore left fully opaque and the buffer carries the whole ramp.
#
# The WASH is the swept corridor on the road, drawn ONLY when there is something
# to brake for and only as far as that thing. Faint at the bumper and
# strengthening ahead, so the eye is pulled toward what matters rather than to
# the paint around the car. A clear corridor gets rails and no wash at all --
# filling the scanned length reads as "all 48 m of this is dangerous" and, over
# a depth-tinted far road, as a solid white band down the whole scene.
WORLD_AEB_WASH_ALPHA_NEAR = 0.06
WORLD_AEB_WASH_ALPHA_FAR = 0.42
# Once the pedal is actually down the wash goes much further, and it has to.
# Blending is linear, and the braking red is bright in linear (R 0.71), so at a
# watching alpha it LIGHTENS the grey road toward pink -- which reads as a glow
# rather than as danger, and in a palette where dark means important it reads as
# less urgent, not more. Pushing the alpha up is what recovers the saturation:
# at 0.42 the corridor lands on (163, 92, 85), at 0.71 on (194, 74, 65).
WORLD_AEB_BRAKING_BOOST = 1.7
# The RAILS are the corridor edges, and they run the full SCANNED length rather
# than stopping at the threat -- that is the part that says "I am watching this
# far", which a filled wash alone cannot express. They fade with distance so a
# 100 m scan at motorway speed does not end in two hard lines at the horizon.
WORLD_AEB_RAIL_WIDTH_M = 0.16
WORLD_AEB_RAIL_ALPHA = 0.80
WORLD_AEB_RAIL_FADE_M = 45.0
# The BRAKE-NOW bar is the last point to brake -- the trigger itself, and the
# one number the whole system turns on. Drawn across the corridor so the gap
# between it and the threat IS the margin remaining.
WORLD_AEB_BAR_THICKNESS_M = 0.40
WORLD_AEB_BAR_ALPHA = 0.55
# The threat MARKER: a panel standing across the corridor, framed on three sides
# so it reads as a targeting reticle rather than a floating card, with a pool of
# light where it meets the road.
WORLD_AEB_MARKER_HEIGHT_M = 1.25
WORLD_AEB_MARKER_ALPHA_BASE = 0.72
WORLD_AEB_MARKER_ALPHA_TOP = 0.06
WORLD_AEB_FRAME_WIDTH_M = 0.16
WORLD_AEB_FRAME_ALPHA = 0.95
WORLD_AEB_POOL_LENGTH_M = 1.4
WORLD_AEB_POOL_ALPHA = 0.55
# Urgency scales the wash while ARMED, from nothing to the full value above, so
# the corridor visibly builds as a threat closes rather than snapping on at the
# moment of firing. Hue still carries the STATE -- violet watching, red acting --
# because a colour change is what reads as an event; this only carries how close
# the system is to acting.
WORLD_AEB_URGENCY_FLOOR = 0.35
# Lifted clear of the road, and above the planned path (0.03), so neither
# z-fights with the surface it is drawn over. The rails and bar sit a little
# higher again so they are never eaten by the wash they lie on.
WORLD_AEB_GROUND_OFFSET_M = 0.07
WORLD_AEB_DETAIL_OFFSET_M = 0.02
# A fake key light for slab faces. HORIZONTAL direction only, in RENDER
# coordinates (+x right, +z toward the camera); render space rather than world
# space matters, because a world-space light would rotate with the car and every
# wall in the scene would change brightness as you went round a roundabout.
#
# Faked rather than real because the boundary material is deliberately unlit --
# a real light would also have to light the road, and the palette's whole
# structure depends on the road staying a flat known value.
#
# Horizontal-only, with the two flat faces pinned separately below, because one
# 3D light cannot do this job. A light from above gives the top face a high
# value and leaves all four sides bunched in the middle; a lateral light
# separates the sides but flattens the top into them. The faces a trailing
# camera actually sees are the top, the near face and one side, and all three
# have to differ.
WORLD_SLAB_LIGHT_DIR = (0.99, 0.14)
WORLD_SLAB_TOP_SHADE = 1.0
WORLD_SLAB_BOTTOM_SHADE = 0.0
# Vertical faces are held below the top face rather than spanning the full ramp,
# so "lit from above" still reads: no wall side is ever as bright as a roof.
WORLD_SLAB_SIDE_SHADE_RANGE = (0.05, 0.75)

# Aerial perspective. Distant geometry mixes toward the air colour, which is
# what "further away" looks like in the real world and what makes two surfaces
# at different ranges read as two surfaces rather than one.
#
# Nothing inside WORLD_DEPTH_NEAR_M is tinted at all, so the near field keeps
# the full contrast ladder the palette was built around; the curve then holds
# the mix low well past that before letting go toward the rim. A straight linear
# ramp from the ego washed out the 12 m band, which is exactly where the
# planner's obstacles live.
#
# EXPONENTIAL extinction, not a ramp between a near and a far distance, and the
# shape had to change when the view went from a 45 m rim to a 150 m one. A ramp
# normalised to its far end has to spend its gradient somewhere: normalised to
# 150 m, a wall at 12 m and a building at 20 m came back to 1.17:1 -- the exact
# complaint the tint was added to fix -- and no exponent fixed it, because
# lowering the exponent to rescue the mid-field flattens it again from the other
# side. An exponential has its strongest RELATIVE gradient immediately after the
# near cutoff and then asymptotes, so the 10-40 m band keeps the separation that
# does the work while the rim still fades out completely. It is also what haze
# physically does (Beer-Lambert), which is why it looks right.
#
# The scale is the distance over which roughly 63% of the fade happens. Shorter
# separates near objects harder but sinks boundary-against-road at range; 34 m
# holds 1.85:1 between 12 m and 20 m while keeping boundary 1.74:1 over road at
# 20 m.
WORLD_DEPTH_NEAR_M = 9.0
WORLD_DEPTH_SCALE_M = 34.0
# How far toward the air colour the very furthest geometry goes. Short of 1.0
# on purpose: at 1.0 the rim vanishes completely and the world reads as ending
# there, rather than as continuing out of sensor range.
WORLD_DEPTH_HAZE = 0.62

# --- Chase camera -------------------------------------------------------------
#
# Pulls back and climbs with speed, so the view always covers roughly the
# distance the car is about to travel rather than a fixed patch of road. The
# slopes are deliberately steep: at walking pace you want to see the kerb you
# are parking against, at motorway speed you want the next several seconds.
# `_camera` clamps both, and adds a little extra height in a corner so the
# inside of the bend does not hide behind the car.
WORLD_CAM_HEIGHT_BASE_M = 8.5
WORLD_CAM_HEIGHT_PER_MPS = 0.62
WORLD_CAM_HEIGHT_MAX_M = 34.0
WORLD_CAM_DISTANCE_BASE_M = 13.0
WORLD_CAM_DISTANCE_PER_MPS = 1.55
WORLD_CAM_DISTANCE_MAX_M = 66.0
WORLD_CAM_CORNER_LIFT_M = 22.0
# Reversing below this is still reversing; above it the driver has selected
# drive again and the camera should come back round. Hysteresis is not needed
# because the sign of the forward speed is what is being read, and a car
# crossing zero is momentarily stationary either way.
WORLD_CAM_REVERSE_SPEED_MPS = 0.35
# The chase pitch. Deliberately shallower than "look at the ego" (which at the
# parked height and distance would be about -33 deg): the car sits low in frame
# and the road ahead fills it, which is what the view is for at speed.
WORLD_CAM_PITCH_DEG = -21.0

# --- ...and how it MOVES ------------------------------------------------------
#
# Every quantity above is damped toward its target rather than jumping to it.
# Nothing was, which is why the reverse swing teleported: continuity came only
# from speed being continuous, so anything keyed on a threshold changed
# instantly. Exponential, with the step computed as 1 - exp(-dt/tau) so the
# result does not depend on the frame rate the scene thread happens to achieve.
WORLD_CAM_TAU_S = 0.35
# The swing round to reverse gets its own, quicker constant: it is a deliberate
# 180-degree move rather than a drift, and at the shared constant it would take
# well over a second to arrive. ~0.16 s puts it there in about half of one.
WORLD_CAM_YAW_TAU_S = 0.16

# There is no standstill framing, and the absence is deliberate. A near-vertical
# top-down tilt at a stop was built, with a hysteresis and a dwell to keep it
# from firing at every give-way line, and then removed: the speed terms above
# already close the view in as the car slows, so the second framing bought very
# little, and every threshold that could switch between the two sits inside the
# band ordinary driving spends real time in -- junctions, queues, traffic,
# parking. The view changing shape while the situation had not is worse than the
# view being slightly too far back while stopped.
#
# Never steeper than this. At exactly -90 the euler yaw becomes degenerate --
# pitch and yaw rotate about the same axis and the view spins on its own. With
# the standstill tilt gone nothing approaches it, so it stands as a guard on
# whatever pitch term comes next rather than as a working limit.
WORLD_CAM_PITCH_LIMIT_DEG = -80.0
# Yaw a few degrees into the bend, so the inside of the corner is not hidden
# behind the ego. Bounded hard: the view is a chase camera, not a cinematic one,
# and the palette and depth cues were designed around a stable frame.
WORLD_CAM_CORNER_YAW_DEG = 12.0
WORLD_CAM_CORNER_YAW_PER_CURVATURE = 260.0
# ...and on a full brake, lift and pull back so the threat marker and the ego
# are in frame together. Only ever while the pedal is DOWN: the colour change
# already carries the armed and watching states, and a camera that moved for
# those would be a nuisance rather than an alarm.
WORLD_CAM_ALERT_LIFT_M = 6.0
WORLD_CAM_ALERT_PULLBACK_M = 8.0

ROAD_CLASSES = frozenset(
    {
        "ASPHALT",
        "COBBLESTONE",
        "DASHED_LINE",
        "DRIVING_INSTRUCTIONS",
        "RESTRICTED_STREET",
        "SOLID_LINE",
        "STREET",
        "ZEBRA_CROSSING",
    }
)

# Some community maps do not provide semantic annotations. Low returns with an
# unknown/background label are treated as ground, while known non-road classes
# remain obstacles.
GROUND_FALLBACK_CLASSES = frozenset({"BACKGROUND"})
GROUND_FALLBACK_ABOVE_M = 0.12
GROUND_FALLBACK_BELOW_M = 0.50

# --- Surface materials --------------------------------------------------------
#
# What a surface is MADE OF, which is a different question from ROAD_CLASSES.
# That set answers "may the car drive here" and feeds the road store; this
# answers "what colour is the ground" and is display-only. A class can be in
# both -- everything in ROAD_CLASSES is the paved material -- and being in
# neither is fine and common, because most of the palette is not ground at all.
#
# The classes are BeamNG's own, verified against `tech/annotations.json` in
# 0.38.5. Names that do not exist there (GRAVEL, DIRT, TERRAIN) are listed
# anyway: a missing name simply never matches, and community maps and future
# versions cost nothing to accommodate.
#
# Only what the SHAPE test has already accepted as ground is ever coloured by
# these, so a class appearing here does not make it a surface. NATURE is the
# clearest case: it covers both grass and tree canopy, and the canopy is a tall
# run that never reaches the surface mesh at all.
SIDEWALK_CLASSES = frozenset({"SIDEWALK"})
VEGETATION_CLASSES = frozenset({"GRASS", "NATURE"})
BARE_GROUND_CLASSES = frozenset(
    {"DIRT", "GRAVEL", "MUD", "ROCK", "SAND", "TERRAIN"}
)
WATER_CLASSES = frozenset({"WATER"})
# Paint on the road. Every one of these is ALSO in ROAD_CLASSES -- a lane
# line is drivable tarmac -- and the material table is ordered so this set
# wins the colour: the marking rides on the road store exactly as before and
# only its paint shows. (Confirmed live 2026-08-10: decals DO annotate through
# the LiDAR.)
#
# DRIVING_INSTRUCTIONS and SPEED_BUMP are deliberately NOT here. Markings are
# decal QUADS and the annotation labels the whole quad, transparent texels
# included; a line's quad is barely wider than its paint, but junction
# furniture -- chevron hatching, arrows, give-way triangles -- ships as big
# rectangular decals, and their whole footprint came back as marking: entire
# roundabout approaches rendered as one sheet of paint. Until the sensor can
# tell paint from the quad it rides on, the wide-area classes stay tarmac.
# The `Marking check:` line logs per-class counts, which is the evidence to
# revisit this with.
MARKING_CLASSES = frozenset(
    {
        "DASHED_LINE",
        "SOLID_LINE",
        "ZEBRA_CROSSING",
    }
)

# --- Self-driving ------------------------------------------------------------
#
# The planner is deliberately GEOMETRIC: it never looks at the semantic classes
# the display uses. Drivable means "no return in the obstacle height band", so
# flat grass and car parks read as drivable and the car will explore them. On a
# kerbed road the 0.20 m sensor mount sees the kerb face, which is what keeps it
# on the tarmac. If that ever needs tightening, add a road-coverage *bonus* to
# the cost function -- the semantic mask is already computed for the display.

# Hard cap on commanded speed. Everything else only ever slows the car down.
# 40 km/h keeps the full braking envelope (v^2 / (2 * COMFORT_DECEL) +
# STOP_MARGIN ~= 28.7 m) inside the 35 m planning horizon; going faster needs
# the horizon and slope allowance re-validated live first.
MAX_SPEED_MPS = 40.0 / 3.6

# The planner reasons about the near field only. The far field is unreliable for
# obstacle work: the ground plane is taken from the ego's own bbox, so a hill
# 80 m away reads as a wall. 35 m is 5 s of travel at the speed cap, and it
# comfortably contains the full stopping envelope (see test_config).
PLANNER_HORIZON_M = 35.0
# Odd, so a perfectly straight arc is always among the candidates.
PLANNER_ARC_SAMPLES = 41
# Sets the widest candidate curvature (K_MAX = 1 / this). Roughly a small car's
# full-lock radius; the steering command is scaled against it.
MIN_TURN_RADIUS_M = 6.0
# The arc scan is an (obstacles x arcs) matrix, so obstacles are decimated first.
# 4,000 x 41 is under a millisecond and loses nothing at this horizon.
PLANNER_MAX_OBSTACLE_POINTS = 4000

# Obstacle height band, measured above the vehicle's ground plane
# (VehicleGeometry.ground_z_vehicle), matching how classify_road_points uses it.
# The floor matches GROUND_FALLBACK_ABOVE_M so a 0.12-0.15 m kerb is an obstacle.
# The ceiling drops bridges, gantries, signs and tree canopy, which would
# otherwise read as a wall across the road.
OBSTACLE_MIN_HEIGHT_M = 0.12
OBSTACLE_MAX_HEIGHT_M = 2.20
# The ground plane is only sampled under the ego, so terrain drifts away from it
# with distance. Past SLOPE_ALLOWANCE_START_M the floor is relaxed by
# SLOPE_ALLOWANCE_PER_M per further metre, which stops a gentle rise ahead from
# reading as a wall. The near field gets NO allowance on purpose -- that is
# where a 0.12 m kerb has to stay an obstacle, and it is the only zone the car
# can still stop inside. Needs a live check on hilly maps: too tight and the car
# brakes at every crest, too loose and it climbs kerbs at range.
SLOPE_ALLOWANCE_START_M = 10.0
SLOPE_ALLOWANCE_PER_M = 0.015
# ...but the cone alone made the car blind to the road's edges. The floor it
# builds reaches 0.27 m at 20 m and 0.50 m at 35 m, and a kerb is 0.10-0.15 m,
# so NO kerb was an obstacle beyond about 12 m -- 1.1 s of road-edge
# information at the speed cap. The car could not see a bend coming, ran wide
# into the outside of it, and blocked. Measured closed-loop: a 60 m-radius bend
# on a 6 m road ended in STUCK every time.
#
# So the local ground is estimated from the returns themselves, per range ring,
# and the cone becomes a BOUND on that estimate rather than the estimate
# itself. The clamp is two-sided and that is what makes this safe: never below
# the ego's own plane, so a ditch beside the road cannot drag the floor down
# and turn the road surface into a wall, and never above the cone, so terrain
# behaves exactly as it did before. On the flat ground the car actually drives
# on, the estimate collapses to the ego plane and kerbs stay visible to the
# full horizon.
GROUND_BIN_M = 2.5
# A low percentile, not the minimum: one stray return under the road surface
# would otherwise set the ground for the whole ring.
GROUND_PERCENTILE = 20.0
GROUND_MIN_SAMPLES = 24
# The estimate is a percentile, so it does not need every point.
GROUND_SAMPLE_STRIDE = 4

# Isolated-return rejection, applied to the obstacle set before the arc scan.
# The scan takes the NEAREST blocking point per arc, so a single spurious
# return -- one ray clipping a pebble, a leaf, a fence wire, or a ground point
# lifted over the floor by a suspension transient -- collapses that arc's free
# distance on its own. Measured: one stray point at 10 m took the free distance
# of a clear 5.2 m road from 33.2 m to 10.0 m, and the speed law brakes below
# about 25 m, so a single speck is a full brake application.
#
# A return is kept only if its 3x3 cell neighbourhood (so a 1.2 m box) holds at
# least OBSTACLE_MIN_SUPPORT returns in total, itself included. Real structures
# are surfaces: four sensors at 256 channels put dozens of returns on a kerb
# face or a pole even at the far end of the horizon, so a support of 2 rejects
# only genuinely isolated points and never a thing the car could hit.
OBSTACLE_CELL_M = 0.4
OBSTACLE_MIN_SUPPORT = 2
# The neighbourhood a cell-referenced band measures its CEILING from, which is
# a different question from the floor and cannot share the floor's cell.
#
# The floor asks "how high is this return above the ground under it" and wants
# the tightest local reference there is. The ceiling asks "is this thing
# overhead", and answering that from the same 0.4 m cell assumes the cell holds
# a ground return -- which at range it does not. The ground units' azimuth
# stripes are 0.8-1.2 m apart at 19 m, so a cell can hold a dense overhead
# surface and no floor at all; its base is then the soffit, and a bridge,
# gantry, tunnel mouth or tree canopy reads as a 0.6 m solid wall. Measured, a
# soffit at 2.4-3.0 m with its face at 19 m took the planner's free distance
# from 35.0 m to 19.0 m.
#
# 2.0 m is comfortably wider than the widest stripe spacing inside the
# planner's horizon, so the neighbourhood catches the ground BESIDE the
# structure. Larger would start reaching across a whole road and flattening
# real terrain into the reference; smaller stops being grade-proof for the
# ceiling and re-opens the "a wall on a steep hill is discarded as overhead"
# failure the fine cell base was introduced to fix.
OBSTACLE_COARSE_CELL_M = 2.0

# Added to the ego half-width when testing whether an arc is blocked.
CLEARANCE_MARGIN_M = 0.35
# Gap held from the right-hand edge of the measured corridor, on top of the ego
# half-width.
KEEP_RIGHT_MARGIN_M = 0.45
# Forward depth of the slice used to measure the corridor edges.
CORRIDOR_BAND_M = 3.0
# Beyond this there is no meaningful "right edge" to keep to -- an open car park
# rather than a road -- and the keep-right term is dropped instead of guessed.
MAX_CORRIDOR_HALF_WIDTH_M = 12.0
# Where along each arc the keep-right and nav-heading terms are evaluated.
# The length matters: the curvature needed to reach a lateral target falls as
# 1/L^2, so a short lookahead turns a polite keep-right nudge into a swerve
# that the collision test then rejects. The worker scales the lookahead with
# speed (LOOKAHEAD_TIME_S seconds of travel, clamped into the window below) to
# preserve that ~3 s character at every speed; PLANNER_LOOKAHEAD_M is the
# static default used when no speed is available.
PLANNER_LOOKAHEAD_M = 20.0
LOOKAHEAD_TIME_S = 2.8
LOOKAHEAD_MIN_M = 16.0
LOOKAHEAD_MAX_M = 30.0

# The "keep going, then turn" deferral options: each planner candidate holds
# the current curvature for one of these distances before bending to its
# target curvature. 0 first, so today's immediate-turn fan is always among the
# candidates (and is what the widget draws as the fan). The longest deferral
# stays inside REQUIRED_FREE_DISTANCE_M, so the speed law can always brake to
# a deferred corner's entry speed in the distance available. Three deferred
# families is a measured compute budget, not a guess: per-tick re-planning
# continuously refines the coarse 6 m grid as the corner approaches.
TRANSITION_DISTANCES_M = (0.0, 6.0, 12.0, 18.0)
# Clearance is scored on how much room an arc leaves BEYOND the collision
# envelope, and saturates quickly: it exists to break near-ties and to punish
# scraping, not to fight the keep-right term (which by definition asks the car
# to sit closer to one edge).
DESIRED_CLEARANCE_M = 0.5
# Lateral error at the lookahead that saturates the keep-right term.
KEEP_RIGHT_SCALE_M = 2.0

# Arc cost weights, each scoring a term normalised into roughly [0, 1] so the
# weights are directly comparable. Nav heading outranks keep-right, so a
# commanded turn beats lane discipline at a junction.
COST_FREE_DISTANCE = 1.0
COST_CLEARANCE = 0.8
COST_KEEP_RIGHT = 0.45
COST_NAV_HEADING = 3.0
COST_SMOOTHNESS = 1.5
# Deferral tie-break, scaled by transition/TRANSITION_DISTANCES_M[-1].
# Deliberately small: a deferred turn must win because the immediate turn
# collides (a genuine geometric fact, worth whole tenths of free-distance
# cost), never because it edged a near-tie on clearance -- under per-tick
# re-planning a near-tie deferral re-defers forever and the turn never comes.
COST_TRANSITION = 0.05

# Route reference-path terms. When `route_model` can build a path these two
# REPLACE the nav-heading and keep-right terms (two lateral targets fighting --
# kerb band versus route centreline -- is the failure that gate avoids); with
# no route the legacy terms run byte-identically. The tangent term inherits
# COST_NAV_HEADING's rank and normalisation, so its tuned "a commanded turn
# outranks lane discipline" character carries over. Cross-track is double
# keep-right's weight -- the route centreline is a better lane reference than
# a straight-band kerb slice -- but the term SATURATES at its weight, so free
# distance (whole tenths for any real pinch) plus clearance still dominate: a
# blocked arc can never be bought back by route conformance.
COST_ROUTE_HEADING = 3.0
COST_ROUTE_XTRACK = 0.9
# Lateral error that saturates the cross-track term at each matched sample;
# the same units and character as KEEP_RIGHT_SCALE_M, which it replaces.
ROUTE_XTRACK_SCALE_M = 2.0
# Conformance is the MEAN cross-track cost over this many samples along each
# candidate's composite path, each matched to its nearest route sample. One
# endpoint cannot measure a path through a bend: a shallow arc that cuts a
# 90-degree corner lands its endpoint ON the ribbon at the apex, and priced
# there alone it read as perfect -- measured 7.4 m inside the bend in the
# closed loop. Eight samples over a 30 m lookahead is one per 3.75 m.
ROUTE_MATCH_SAMPLES = 8

# Longitudinal limits. Target speed is the minimum of the cap, the cornering
# limit sqrt(a_lat / |k|), the corner-entry limit sqrt(a_lat / |k_next| +
# 2 * a_decel * transition) for a deferred turn, and the stopping limit
# sqrt(2 * a_decel * headroom).
# Two different quantities, and collapsing them into one is what stopped the
# car getting round a tight corner at all.
#
# MAX_LATERAL_ACCEL is the GRIP ceiling: the hardest the steering is ever
# allowed to ask for (controller._curvature_ceiling). CORNERING_ACCEL is the
# COMFORT figure the speed law plans corner speeds with. When both were 3.5 the
# planned speed needed exactly the ceiling curvature to make the corner, so
# there was no authority left over for tracking error -- any lag and the car
# simply could not follow the arc, ran wide and clipped the inside. Measured
# closed-loop: 35 m and 25 m corners both ended in STUCK with peak lateral
# accel pinned at the ceiling.
#
# The gap between them is the steering authority left for tracking error, and
# it has to be generous. 3.5 is a comfort number, not a grip number -- dry
# tarmac is around 8 -- and using it as the ceiling meant that any steady-state
# speed error (the speed loop is proportional, so there is always a few km/h of
# it) left the steering saturated and unable to correct. Measured: through a
# 25 m corner the car sat 3 km/h over target with the steering pinned on the
# ceiling for 20 m, tracked progressively wider, and blocked. At 6.0 the
# ceiling only binds when the car is well above its planned corner speed --
# which is exactly when it needs the authority.
MAX_LATERAL_ACCEL_MPS2 = 6.0
CORNERING_ACCEL_MPS2 = 2.8
COMFORT_DECEL_MPS2 = 2.5
# The speed target is low-passed before it reaches the pedals. Free distance is
# a nearest-point measurement over a cloud that changes every 40 ms, so its
# one-tick dips are noise, not corners; unfiltered they reach the brake
# directly. 0.45 s flattens a single-tick dip to a few percent while costing a
# sustained slowdown about a metre of travel at the speed cap.
TARGET_SPEED_TAU_S = 0.45
# ...except when the target really has collapsed, where any delay is a crash.
# A target this far below the current speed bypasses the filter outright.
TARGET_SPEED_BYPASS_MPS = 2.5
# Free distance is measured from the vehicle reference node, so this has to
# clear the front overhang as well as leave a standoff.
STOP_MARGIN_M = 4.0

# The pedals work in acceleration units: desired accel = SPEED_KV * speed
# error, clipped into [-HARD_DECEL, COMFORT_ACCEL], then mapped to pedal
# fractions through the two GAIN constants (nominal full-throttle and
# full-brake accelerations for a road car; the trim integrator absorbs the
# per-vehicle error). Between 0 and -COAST_DECEL the car coasts on engine
# drag -- the lift-off band a human uses instead of riding the brake -- which
# is also what keeps throttle and brake from chattering across the boundary.
SPEED_KV = 0.9
COMFORT_ACCEL_MPS2 = 2.0
HARD_DECEL_MPS2 = 4.5
# Engine drag in gear, and so the width of the no-brake band: the brake is only
# touched once the desired deceleration exceeds what lifting off would give.
# At 0.7 the band was 0.78 m/s (2.8 km/h) of overspeed, narrow enough that
# ordinary target jitter reached the pedal; 1.2 m/s^2 is a realistic figure for
# a road car in D and widens the band to 1.33 m/s (4.8 km/h).
COAST_DECEL_MPS2 = 1.2
# Once braking, hold the pedal until the demand falls to this fraction of the
# coast threshold. Without the hysteresis the pedal chatters on and off across
# the boundary, which is felt as a shudder rather than as braking.
BRAKE_RELEASE_FRACTION = 0.5
# Drag CREDITED against the brake demand, which is a different question from
# the band above: the band decides whether to touch the pedal at all, this
# decides how hard once you do. Subtracting the full coast figure assumed the
# engine was already delivering it, so every brake application came out short
# by whatever drag was really missing -- measured, the car entering a 35 m
# corner sat 10 km/h over target on 0.16 of brake, the steering saturated
# against its grip ceiling trying to make up the difference, and it ran wide
# until it blocked. Deliberately conservative: an automatic in D at part
# throttle gives little, and under-crediting only means braking slightly
# early, which the converging speed error then trims away.
ENGINE_DRAG_MPS2 = 0.3
THROTTLE_GAIN_MPS2 = 3.5
BRAKE_GAIN_MPS2 = 6.0
# Slow integrator on the throttle path only: it learns the steady-state
# throttle this vehicle needs (mass, drag, grade) rather than fighting the
# situation, so it survives mode changes and is clamped well under a lurch.
TRIM_RATE_PER_S = 0.02
TRIM_MAX = 0.35
# Pedal slew limits are the jerk limiting that makes inputs look human: brake
# releases faster than it applies, throttle lifts faster than it squeezes.
THROTTLE_SLEW_UP_PER_S = 1.2
THROTTLE_SLEW_DOWN_PER_S = 4.0
BRAKE_SLEW_UP_PER_S = 2.0
BRAKE_SLEW_DOWN_PER_S = 3.5
# Once stopped the brake relaxes to a hold instead of standing on the pedal;
# above the taper speed a BLOCKED entry is an emergency and bypasses the slews.
BRAKE_HOLD_FRACTION = 0.35
HOLD_TAPER_SPEED_MPS = 2.0
# Free distance the planner wants in hand at the speed cap: the full braking
# envelope plus the standoff. Arcs at or above this are not penalised at all --
# 20 m of clear road is not "worse" than 35 m, and treating it as worse is what
# makes a planner refuse to ever turn.
REQUIRED_FREE_DISTANCE_M = MAX_SPEED_MPS**2 / (2.0 * COMFORT_DECEL_MPS2) + STOP_MARGIN_M
# Steering is slewed in curvature, scheduled by speed: the rate is
# LAT_JERK_MAX / v^2 (lateral jerk = v^2 * dk/dt at constant speed), capped at
# K_RATE_CEIL for parking-speed manoeuvring. The ceiling matches the feel of
# the old 2.5/s steering-value slew (2.5 * K_MAX ~= 0.42), so low-speed
# behaviour is unchanged while 40 km/h gets the gentleness a lurch-free lane
# change needs.
# 2.5 was too slow to be followable: at the 40 km/h cap it allows 0.020 /s, so
# winding on the 0.04 curvature of an ordinary 25 m-radius bend took 2.0 s and
# 22 m of travel -- most of the 35 m horizon -- and the car ran wide into the
# outside of every corner. 4.0 m/s^3 is still inside normal-driving comfort
# (a relaxed lane change peaks near 3.5) and halves that to 1.2 s.
#
# The speed law deliberately does NOT consult this rate to credit wind-on
# distance ("hold speed, I will slow before I am really turning") -- that is
# the entry-allowance trap: the credit keeps the target high, the high target
# commands no braking, and the wind-on completes with the car still at speed.
# Anticipation comes from the path (deferred families, the route preview),
# never from the controller.
LAT_JERK_MAX_MPS3 = 4.0
K_RATE_CEIL_PER_S = 0.42
# Closed-loop steering trim: measured curvature (filtered yaw rate / speed) is
# compared against the command and a multiplicative gain adapts slowly to
# cancel per-vehicle steering ratio and understeer. Only adapted while
# actually cornering above walking pace with a consistent sign, and clamped so
# a bad measurement can never fold the steering authority or double it.
STEER_GAIN_MIN = 0.6
STEER_GAIN_MAX = 1.8
STEER_GAIN_ADAPT_RATE = 0.15
STEER_GAIN_MIN_SPEED_MPS = 3.0
STEER_GAIN_MIN_CURVATURE = 0.02
YAW_FILTER_ALPHA = 0.3
# Maps the planner's positive-curvature-is-LEFT onto BeamNG's steering input,
# where positive is RIGHT. Verified in the game source, not inferred:
# lua/vehicle/input.lua's kbdSteer sends `kbdSteerRight - kbdSteerLeft`, so the
# steer-right binding produces +1 and steer-left -1.
#
# BeamNGpy's Vehicle.control docstring claims the opposite ("negative = right").
# It is wrong, and trusting it made the car steer the mirror image of the path
# drawn in the BEV. Hence -1.0, and hence the regression tests in
# test_controller.py that pin left-arc-means-negative-steering.
STEERING_SIGN = -1.0

# Recovery. Blocked holds first in case the obstruction moves (traffic), and
# only then reverses.
BLOCKED_HOLD_S = 1.5
# Forward travel in DRIVING that counts as "the recovery worked" and resets the
# attempt counter.
#
# Without it MAX_RECOVERY_ATTEMPTS is unreachable and the car reverses for
# ever. `_enter` reset the counter on every entry to DRIVING, and every reverse
# recovery buys at least one tick of DRIVING on the way back -- backing 6 m
# recovers free distance to about 8.5 m, comfortably past
# STOP_MARGIN_M + RESUME_HYSTERESIS_M -- so the counter was cleared before it
# could ever reach 3. The recovery is open-loop (a fixed REVERSE_DISTANCE_M
# back, then the same approach from the same offset at the same heading), so
# what that produced was an exact limit cycle with zero net progress: reverse,
# creep forward, block, reverse, for ever, which is the "constantly reversing"
# complaint in its purest form.
#
# 15 m is further than one recovery can hand back (REVERSE_DISTANCE_M is 6),
# so it cannot be satisfied by the recovery itself -- only by actually getting
# somewhere.
RECOVERY_PROGRESS_M = 15.0
REVERSE_DISTANCE_M = 6.0
REVERSE_SPEED_MPS = 2.0
# Reversing needs its own room behind, or the recovery is just a slower crash.
REVERSE_MIN_CLEARANCE_M = 3.0
# Extra free distance required to pull away again, on top of STOP_MARGIN_M.
# Without the gap the car twitches between BLOCKED and DRIVING on the boundary.
RESUME_HYSTERESIS_M = 1.0
# Below this the car counts as stationary, for both the hold and stall checks.
STALL_SPEED_MPS = 0.3
# Throttle commanded but the car is not moving: kerbed, wedged, or nose-in.
STUCK_TIMEOUT_S = 3.0
MAX_RECOVERY_ATTEMPTS = 3

# --- Autonomous emergency braking ---------------------------------------------
#
# A last-resort brake, deliberately independent of the planner AND of
# self-driving: it has to work while a human is driving, so it predicts the path
# from the yaw the car is MEASURED to be turning at rather than from any
# command, and it touches nothing but the pedals. Steering, gear and the parking
# brake stay with whoever is driving.
#
# Obstacles are treated as STATIC. A single-frame cloud carries no velocity and
# nothing here tracks returns between frames, so there is no honest relative
# speed to be had. That is the conservative error for the cases that matter (a
# wall, a kerb face, a stopped car) and it errs toward braking early behind a
# moving leader -- which is what the standoff and the trigger threshold below
# are sized to keep out of ordinary following.

# Armed only above this. Below it every parking manoeuvre reads as a stream of
# near misses, and the yaw-derived path prediction is meaningless anyway. The
# gate blocks ARMING only: once engaged, AEB keeps braking all the way to rest.
AEB_MIN_SPEED_MPS = 2.0
# Where the car has to come to rest, measured BEYOND the front bumper. The
# geometry's own front overhang is added on top, because free distance is
# measured from the vehicle reference node -- this is the same correction
# STOP_MARGIN_M folds into a single number for the planner.
#
# Half a metre, not the planner's four: this is an emergency stop, and every
# centimetre here is a centimetre the brake comes on earlier. The planner's
# margin buys a comfortable ride; this one only has to avoid touching.
AEB_STANDOFF_M = 0.6
# Half-width added to the ego body when testing the corridor. Deliberately much
# slimmer than the planner's CLEARANCE_MARGIN_M: that margin buys comfortable
# paths, this one answers "will I actually hit it", and every extra centimetre
# here is another kerb that can trigger a full brake application.
AEB_CLEARANCE_MARGIN_M = 0.15

# --- ...and the three filters that keep it off flat, empty road ---------------
#
# AEB fires a full-authority brake, so the cost of a false positive here is far
# higher than anywhere else in the app and the evidence bar is set accordingly.
# All five of these exist to answer the same question -- is this a real thing
# the car is about to hit, or is it the road, or a bush?

# 1. HEIGHT. AEB's obstacle floor is its own, and it is much higher than the
# planner's OBSTACLE_MIN_HEIGHT_M (0.12). The two are answering different
# questions: the planner steers around a 0.12 m kerb, AEB brakes for things
# that would be a crash, and nobody wants an emergency stop for a kerb they
# were driving over deliberately.
#
# It is also what keeps the flat road itself out of the obstacle set. The ego
# ground plane is body-fixed while heights are gravity-referenced, so a brake
# dive lifts the whole road surface in the height band -- and planner.ground_rise
# cannot absorb it inside SLOPE_ALLOWANCE_START_M, where the cone that clamps it
# is zero. That is precisely the near field AEB works in. Nose-down 5 cm of
# suspension travel put the road at 0.05 against the planner's 0.12 floor; add
# road camber, a rough surface or a compression and it reaches it. The failure
# is a latch -- brake, dive, see more road, brake harder -- exactly the one
# gravity-referenced heights were introduced to kill for the planner. 0.30 m
# clears every bit of it while staying far below any car, wall, post or person.
#
# The height test alone is NOT enough, and cannot be made enough by tuning: on a
# grade the road itself climbs through any fixed floor. See the two shape tests
# below, which is where that is actually answered.
AEB_OBSTACLE_MIN_HEIGHT_M = 0.30
# 1b. VERTICAL EXTENT -- how tall is the thing standing in this cell, rather
# than how high above an estimated ground plane this return is.
#
# This is what makes AEB grade-proof, and the height floor above cannot be. The
# local ground estimate is clamped into a 1.5% cone (SLOPE_ALLOWANCE_PER_M) to
# protect the planner's kerb detection, so the system refuses to believe any
# steeper grade -- and none at all inside SLOPE_ALLOWANCE_START_M. Measured on a
# 5% climb at 25 m: the estimator saw 1.20 m of rise, the clamp allowed 0.225 m,
# and the whole hillside entered the obstacle band as a dense, persistent
# surface that sails through every other filter. At 40-70 km/h the horizon is
# 30-60 m, so a mild hill far ahead was a full stop.
#
# Vertical extent is immune to all of it. A 0.4 m cell on a 20% slope holds
# 0.08 m of spread; a wall holds metres (measured: 5% grade 0.10 m, wall 2.95 m).
# Being differential it is also immune to brake dive and to suspension heave,
# which is the near-field case the cone could never reach -- including the whole
# of the REVERSE system, which lives inside 10 m where the cone is exactly zero.
#
# Measured over every return in the cell, INCLUDING the ones the height floor
# rejects: a 0.35 m rock puts only 0.05 m above the floor but is 0.35 m tall, and
# measuring the spread on the survivors alone would delete every short solid
# object in the world.
AEB_MIN_VERTICAL_EXTENT_M = 0.25
# 1c. POROSITY. A bush is see-through -- rays pass between the leaves and return
# from the ground behind it -- and a wall is not. That is the physical difference
# between the two, and it is measurable without ever asking what class a return
# belongs to, which matters because community maps may not annotate at all.
#
# The window it is measured in is not a guess: an object of height `a` at range
# `r` seen from a sensor at height `h` hides the ground behind it for
# r*a/(h - a). Ground returns INSIDE that shadow mean the rays got through.
#
#     0.6 m bush at 20 m   solid twin would hide 12.2 m; ground seen at 21-32 m
#     0.6 m post at 20 m   ground genuinely hidden; no evidence; stays an obstacle
#     >= h  (wall, car)    shadow is infinite, the window is EMPTY, never vetoed
#
# That last line is the safety property and it is derived rather than imposed:
# anything as tall as the roof unit cannot be dismissed by this test at all, so
# no parallax between the five mount positions can talk AEB out of braking for a
# wall. Only short things are testable, and for them "no evidence" means solid.
#
# The gap keeps the object's own returns out of its own window; the hit count
# means one stray return cannot clear a real obstacle.
AEB_POROSITY_GAP_M = 1.0
AEB_POROSITY_MIN_HITS = 4
# The RESOLUTION of the evidence grid, and emphatically not the width of the
# window -- that is each candidate's own angular width, OBSTACLE_CELL_M / r, and
# is derived rather than configured for the same reason the shadow length is.
#
# It was 2.0 and read as the window itself, which is a fixed ANGLE against
# objects of fixed WIDTH, so it outgrew them with range: a 1.8 m car spans a
# 2 deg bin only inside ~52 m, and covers one outright only inside ~26 m. Past
# that the bin caught the road BESIDE the car, which no part of the car ever
# stood in front of, and the "evidence" vetoed it. Measured on a ring-sampled
# scene, AEB went blind to a stopped car through the whole 30-60 m band -- the
# band it must fire in at 60-100 km/h.
#
# Only bins lying ENTIRELY inside a candidate's wedge are consulted, so this
# wants to be fine enough that a candidate still covers one at the ranges the
# test is for. 0.5 deg is two of the front unit's 0.26 deg columns, and a
# 0.4 m cell covers a whole bin out to ~46 m; past that the test simply finds no
# evidence and reports solid, which is the correct direction for a filter whose
# whole job is dismissing roadside scrub in the near field.
AEB_POROSITY_AZIMUTH_DEG = 0.5
# 2. SUPPORT. The corridor scan is a nearest-return measurement, so one stray
# point ends it -- the same failure that made planner.despeckle necessary, but
# with a full brake application on the end of it instead of a cost term. The
# blocking distance is therefore the Nth nearest return in the corridor, not the
# nearest: a wall, a car or a person is a surface and puts dozens of returns
# inside a corridor this narrow at the distances AEB fires from, while a speck
# is one. Counted on the decimated cloud the planner already built, so this is
# a threshold on what survives PLANNER_MAX_OBSTACLE_POINTS, not on the raw scan.
AEB_MIN_HITS = 4
# 3. PERSISTENCE. A phantom lasts a frame; a wall does not. Expressed in seconds
# rather than ticks for the same reason _POLL_FAILURE_GRACE_S is -- a three-tick
# count is a 120 ms budget at this display rate and something else entirely at
# any other. Costs 1.3 m of travel at the speed cap, which is nothing against
# the braking envelope, and it is applied uniformly: a genuine obstacle that
# appears inside 0.12 s was never avoidable anyway.
AEB_CONFIRM_S = 0.12

# --- Whose car all of this was measured on ------------------------------------
#
# Documentation only: nothing reads this to make a decision. It exists because
# the braking tables below, AEB_OBSTACLE_MIN_HEIGHT_M and the whole plant are
# properties of ONE vehicle, and until now the repo did not record which one --
# so "it phantom-brakes on the pickup but not on this" had no baseline to be
# measured against. `worker` logs it beside the model actually attached, and a
# mismatch between the two lines is the first thing to check on any report of
# braking too early or too late.
PLANT_REFERENCE_VEHICLE = "vivace"

# The trigger is the LAST POINT TO BRAKE, not a deceleration threshold, and
# that distinction is the whole character of the feature.
#
# Scoring "how hard would I have to brake from here" against a threshold and
# then serving a pedal proportional to it makes AEB brake early and gently:
# measured, at 50 km/h it fired 22.9 m out on 0.52 of pedal and stopped with
# 7.2 m to spare. That is a driver-assist creeping into the driver's job, not
# an emergency brake, and it is what gets AEB switched off. So instead: work
# out the distance a full-authority stop actually needs from here, and do
# nothing at all until the car reaches it.
#
#     needed    = v * AEB_LATENCY_S + v^2 / (2 * AEB_BRAKING_DECEL_MPS2)
#     fire when   available <= AEB_TRIGGER_MARGIN * needed
#
# MEASURED on the vehicle in use, not assumed. Full-pedal braking distance from
# a standing-start run down, and the deceleration each one implies:
#
#      11 km/h   0.40 m   11.67 m/s^2      100 km/h   37.36 m   10.33 m/s^2
#      20 km/h   1.47 m   10.50            120 km/h   55.04 m   10.09
#      40 km/h   6.01 m   10.27            160 km/h   98.25 m   10.05
#      60 km/h  13.23 m   10.50            200 km/h  160.00 m    9.65
#      80 km/h  24.22 m   10.19
#
# A least-squares fit of d = A*v + B*v^2 puts A at -0.14 s, i.e. zero: these are
# pure braking distances with no reaction time folded in, so AEB_LATENCY_S stays
# a separate term rather than being double-counted here.
#
# The figure below has to be at or BELOW the worst deceleration the car actually
# achieves, or the model under-predicts the real stop and AEB fires too late.
# 10.05 is the worst over the range AEB can act in; 10.0 keeps a slack of
# 0.2-11 m at every measured speed once AEB_TRIGGER_MARGIN is applied, including
# 200 km/h where the car itself falls off to 9.65. Guessing 8.0 originally cost
# 1.9 m of unnecessary approach at 50 km/h.
#
# This is a property of the CAR. Re-measure for a heavy or low-grip vehicle --
# a van or a wet surface will not hold 1.02 g, and there the brake would fire
# too late to stop.
AEB_BRAKING_DECEL_MPS2 = 10.0
# Everything between "the geometry says brake now" and the car actually
# slowing: the display tick, the blocking Vehicle.control round-trip, and pad
# bite plus weight transfer. It matters most at LOW speed, where it is a large
# share of a short stopping distance -- a pure v^2 trigger with no latency term
# looks perfectly safe on paper and then fires too late to stop at 30 km/h.
#
# AEB_CONFIRM_S is deliberately NOT part of it, which was a real error worth
# 1.4 m of early braking at 50 km/h. `_seen_for` counts how long a threat has
# been in the corridor, not how long it has been urgent, and the corridor
# reaches AEB_HORIZON_MARGIN times further out than the trigger -- so for
# anything the car is approaching, the window has already elapsed by the time
# the trigger fires and adds no delay at all.
AEB_LATENCY_S = 0.15
# Held back against the nominal figures above, because a wet road, a heavy car
# or worn pads all brake worse than 9.0. Small on purpose: this is the entire
# margin between "last resort" and "nannying", and it is THE dial to turn if
# AEB feels early or late -- everything else in the trigger is physics.
AEB_TRIGGER_MARGIN = 1.10
# ...and released once there is comfortably more room than a stop needs -- but
# NEVER on the ratio alone, because braking satisfies it by itself. `needed`
# falls as v^2 while the distance to the obstacle falls only as v, so partway
# through a stop the ratio recovers even though the car is closer than when it
# fired. Traced at 50 km/h, the brake let go twice mid-event (at 11.6 m and
# again at 6.4 m) and re-fired each time: an emergency stop delivered as three
# pulses. The release is therefore latched against the gap the event STARTED
# with, which slowing down cannot manufacture.
AEB_RELEASE_MARGIN = 2.0
# There is no post-stop hold, and the absence is deliberate. Reaching a
# standstill IS the objective, so once the car is stopped the event is over and
# the pedal goes straight back to the driver -- the same rule every teardown
# path in `worker` follows, which hands back a coasting car rather than a braked
# one. A timed hold was tried and removed: it left a window in which neither the
# system nor the driver was clearly in charge of the pedal, and the car is not
# held against a gradient either way. AEB_MIN_ENGAGED_S is what stops a
# single-tick blip; it is not a hold.
# What "stopped" means for handing the pedal back. Deliberately far tighter
# than STALL_SPEED_MPS (0.3), which answers a different question -- "is this car
# moving" for the stall and hold checks. Releasing at 0.3 would hand back a car
# still rolling at over 1 km/h toward the obstacle it just braked for, and under
# a full pedal that is a single tick before it is genuinely at rest, so the
# looser threshold buys nothing and costs the last metre.
AEB_STOPPED_SPEED_MPS = 0.05
# No single-tick blips: once fired, the brake stays on at least this long.
AEB_MIN_ENGAGED_S = 0.3
# Pedal: FULL, with no slew limits -- an emergency outranks smoothness, the
# same reason controller._hold_brake bypasses them. There is nothing left to
# grade, because the trigger above only fires at the point where a full stop is
# the last thing that still works; grading the pedal instead of the timing is
# what produced the 0.52 application. Once stopped it relaxes to a hold rather
# than standing on the pedal indefinitely.
AEB_HOLD_BRAKE = 0.35
# How far ahead the corridor is scanned. Long enough that the trigger distance
# above is always inside it, with lead to spare for the confirmation window --
# otherwise AEB brakes late for the most basic reason possible: it could not
# see far enough. Measured against the OLD PLANNER_HORIZON_M ceiling of 35 m,
# 100 km/h needs 48 m of stopping distance, so the brake could not come on
# until 13 m after the last point it would have worked; it fired at the horizon
# on a full pedal and hit the wall 17 m short.
#
# This is why AEB has its own horizon rather than sharing the planner's. The
# planner's 35 m is about where a chosen PATH stays trustworthy; AEB only has
# to answer "is there something solid dead ahead", which survives further out.
# The trade is real and needs the live check: past ~50 m the return density is
# thin (measured 747 returns beyond 50 m of ~49.8k total), so detection there
# depends on the obstacle being large, and every extra metre is another metre
# over which a mis-aimed corridor can find scenery. Lower LIDAR_DENSITY to 25
# for double the returns everywhere if long-range detection proves marginal.
AEB_MIN_HORIZON_M = 6.0
AEB_MAX_HORIZON_M = 150.0
# Scanned this much further than the trigger distance, so an obstacle is SEEN
# before it becomes urgent. AEB_CONFIRM_S counts how long a threat has been in
# the corridor rather than how long it has been critical, and this margin is
# the lead time that lets it elapse in advance -- otherwise every real brake
# would be delayed by AEB_CONFIRM_S at exactly the moment it was needed.
#
# A clear corridor is not scored against this or anything else: `aeb` reports
# no threat at all rather than an obstacle sitting at the horizon. Conflating
# the two put the required deceleration on an EMPTY road at
# v^2 / (2 * (horizon - standoff)), which past the speed where the horizon
# clamped crossed the trigger by itself: on a flat, empty gridmap the car
# braked from 64 km/h down to 45, released, accelerated, and did it again.
AEB_HORIZON_MARGIN = 2.2
# Heavier than the controller's YAW_FILTER_ALPHA: this one aims a corridor
# rather than trimming a gain, and a corridor that jitters is a brake that fires
# at roadside scenery.
AEB_YAW_FILTER_ALPHA = 0.25

# --- Reverse AEB --------------------------------------------------------------
#
# The same machinery pointed backwards -- `aeb` runs it on a 180-degree-rotated
# cloud, so there is one state machine, one corridor scan and one set of phantom
# filters. What differs is the PLANT, and it differs enough that sharing the
# forward numbers would fire far too late.
#
# MEASURED on the same vehicle, reversing:
#
#      10 km/h   0.56 m   6.89 m/s^2       30 km/h   4.46 m   7.79 m/s^2
#      20 km/h   2.06 m   7.49             50 km/h  13.21 m   7.30
#
# 0.70-0.79 g against 1.02-1.07 forward: braking while reversing throws the load
# onto the rear axle, which carries the smaller pair of brakes, so the car stops
# roughly 30% worse backwards. As with the forward table the least-squares
# reaction term is ~0 (-0.04 s), so AEB_LATENCY_S stays separate. 6.8 sits just
# under the worst measured figure.
AEB_REVERSE_BRAKING_DECEL_MPS2 = 6.8
# Reversing is a parking manoeuvre, and stopping the car before it backs into a
# wall is the whole point -- so unlike the forward system this has to be awake
# at parking speed, not merely at walking pace. 0.8 m/s (2.9 km/h) was still too
# high: shuffling into a space spends most of its time under that, which is
# exactly when the system reported STANDBY and did nothing.
#
# Going this low is safe because the TRIGGER distance collapses with speed --
# at 0.5 m/s a full stop needs 0.09 m, so nothing fires until the obstacle is
# within about 0.45 m of the bumper however wrongly the corridor is aimed.
# Kept clear of STALL_SPEED_MPS (0.3) so an armed system is never immediately
# also "stopped", which would serve AEB_HOLD_BRAKE instead of a full pedal.
AEB_REVERSE_MIN_SPEED_MPS = 0.5
# ...and it stops closer, because reverse parking is deliberately close work and
# the forward 0.6 m standoff would refuse to let the car near anything.
AEB_REVERSE_STANDOFF_M = 0.35

# Navigation hint (LEGACY FALLBACK). The bearing-to-one-node hint survives for
# routes `route_model.build_route_path` cannot turn into a reference path
# (fewer than two usable nodes ahead); everywhere else the route now IS a path
# to follow -- see the ROUTE_* block below.
NAV_LOOKAHEAD_M = 18.0
# A blocking Lua round-trip, so it runs far slower than the display loop. The
# route only changes when the player sets a new destination.
NAV_POLL_INTERVAL_MS = 1000

# Route following. The bigmap route becomes an ego-frame REFERENCE PATH:
# cross-track and tangent cost terms in the arc fan (guidance, never
# authority -- a blocked arc outranks any conformance, exactly as the old
# bearing hint did), plus a curvature preview that lets the speed law brake
# for corners and the destination before the LiDAR's obstacle work would
# force it.
#
# How far ahead the preview looks. The cap's whole braking envelope is
# 24.7 m (v^2/2a + margin), so 120 m is ~5x that and ~10 s of driving --
# far enough that the low-passed speed target falls gently rather than
# reacting, and nothing beyond it can require action yet.
ROUTE_PREVIEW_M = 120.0
# Resample spacing. Navgraph nodes can be tens of metres apart on straights,
# so curvature MUST be measured on a resampled polyline; 3 m resolves
# v = sqrt(a/k) within ~5% for the tightest fan curvature while keeping the
# whole preview at <= 41 samples, so every pass over it is trivial.
ROUTE_SAMPLE_STEP_M = 3.0
# Curvature smoothing window. Sized against the worst encoding the navgraph
# produces: a 90-degree junction as a SINGLE VERTEX between two long chords.
# Smearing pi/2 over 9 m reads k ~= 0.17 -> ~4 m/s creep through the turn,
# which is conservative (over-reading curvature under-reads speed); a real
# R = 25 m bend spans 39 m of arc and is untouched by a 9 m window.
ROUTE_CURVATURE_SMOOTH_M = 9.0
# Half road width where the navgraph node carried none: the 6 m two-lane road
# the closed-loop tests drive.
ROUTE_DEFAULT_HALF_WIDTH_M = 3.0
# The navgraph radius is not always a lane statement: plazas and parking
# aprons carry radii of 10 m and more, and a lane target derived from those
# would aim the car across open ground. Clamp to ordinary road geometry --
# half a narrow lane up to a wide two-lane carriageway.
ROUTE_MIN_HALF_WIDTH_M = 1.5
ROUTE_MAX_HALF_WIDTH_M = 5.0
# Speed through a turning junction: v = sqrt(CORNERING_ACCEL / k) at
# R ~= 17.5 m, a typical urban corner mouth.
ROUTE_JUNCTION_SPEED_MPS = 7.0
# A junction only slows the car when the route actually TURNS there: heading
# change past 15 degrees across the junction window. linkCount alone would
# brake at every crossroads driven straight through (navgraph junction nodes
# often carry zero curvature on the through road); curvature alone misses
# single-vertex turns the smoothing dilutes. The conjunction is the term.
ROUTE_JUNCTION_TURN_RAD = 0.26
ROUTE_JUNCTION_WINDOW_M = 10.0
# The virtual stop line sits this far short of the destination marker --
# front overhang plus the same "stop short, not on top of" character as
# STOP_MARGIN_M.
ROUTE_ARRIVAL_MARGIN_M = 5.0
# The speed limit handed to the controller is the backward pass sampled THIS
# many seconds of travel down the path, not at the ego. The speed loop is a
# proportional law behind a low-pass: tracking a ramping target it carries a
# standing error of decel * (1/SPEED_KV + TARGET_SPEED_TAU_S) ~= 3.9 m/s --
# measured in the closed-loop harness as arriving at a R = 15 m corner at
# 7.9 m/s against a 6.5 m/s entry speed, and overshooting the destination
# stop line by 8 m. Sampling the pass one tracker time-constant ahead lowers
# the target by exactly the error the tracker will add back, and on a flat
# stretch of the pass it changes nothing at all. 1/0.9 + 0.45 ~= 1.56.
# The value handed over is the MINIMUM of the pass over [now, lead], never
# the point sample at the lead: the pass rises again right after every
# corner dip, and a point sample past the dip inverted the constraint with
# speed (see route_model.route_speed_limit).
ROUTE_PREVIEW_LEAD_S = 1.56
# One transient Lua failure used to wipe the cached route for a full poll
# interval (the fetch overwrote the cache unconditionally). Transport failures
# now keep the last good route for three polls; ~33 m of driving at the cap on
# a possibly stale reference, safe because LiDAR keeps obstacle authority on
# every arc. An explicit "no target" reply still clears immediately -- the
# player cancelling is data, not noise.
ROUTE_STALE_GRACE_S = 3.0
# Below this route speed limit, at rest, the controller holds rather than
# drives: "arrived" must be far tighter than STALL_SPEED_MPS (the same
# reasoning as AEB_STOPPED_SPEED_MPS), and the hold has to be an explicit
# branch because at a zero target the throttle path would serve the trim
# integrator -- wound up to 0.35 on the drive there -- and the car would
# creep-limit-cycle at the marker.
ROUTE_ARRIVED_SPEED_LIMIT_MPS = 0.05
# The terminal deceleration once the route says "stop here" and the car is
# below walking pace. The coast band cannot finish an arrival: it exists to
# stop brake chatter at cruise, so it hands any demand under COAST_DECEL to
# engine drag -- and measured in the closed loop the car idle-crept 6.7 m
# past the marker over eight seconds. Against a permanently-zero target there
# is no chatter to avoid, so the last metre gets a gentle dedicated brake.
ROUTE_ARRIVAL_DECEL_MPS2 = 1.5
# The gentle pedal is open-loop, and on a downgrade steeper than the ~1.5
# m/s^2 it delivers, gravity wins: the car hovered at ~2 m/s and descended
# past the marker indefinitely, never slow enough for the stopped hold to
# engage. So while the car is NOT slowing in the arrival branch, the pedal
# ratchets up at this rate per second -- an integrator on non-progress, the
# feedback a proportional pedal cannot provide against a constant grade. On
# flat ground the speed falls every tick and the ratchet never engages, so
# the gentle character is untouched.
ROUTE_ARRIVAL_BOOST_PER_S = 0.5
# Arriving CLEARS the in-game route (groundMarkers drops its target near the
# marker, and a nearly-consumed polyline is too short to build a path from),
# which would hand the speed law back its full cap right at the destination.
# Inside this remaining distance the worker latches the arrival: the hold
# survives the route disappearing, and only a route with more than this left
# -- a new destination -- releases it.
ROUTE_ARRIVAL_LATCH_M = 10.0

# Planning memory: a worker-thread store of obstacle cells the PLANNER (and
# only the planner -- AEB never reads it, because a full-authority brake on a
# remembered ghost is the unacceptable failure) merges with each tick's
# cloud. Kerbs persist through occlusion, the rear corridor keeps what the
# sensors saw a moment ago, and single-frame sampling noise stops dithering
# the free distance.
#
# The cell is the planner's own obstacle grid pitch (OBSTACLE_CELL_M = 0.4).
MEMORY_CELL_M = 0.4
# Forgotten by the METRE, like the WORLD stores: a wall-clock TTL drains the
# map at a red light while the sweep that would re-observe it is not moving.
# 20 m covers an occlusion pass behind a parked car and the 6 m reverse
# recovery with margin -- and it is deliberately short, because this window
# IS the lifetime bound on any remembered ghost. Note what "parked" means
# here: the odometer stalls, so a ghost in front of a stopped car expires
# only through the reverse recovery moving the car. That escape is designed,
# not accidental.
MEMORY_DISTANCE_M = 20.0
# The planner's 35 m horizon plus the turn sweep and the reverse reach.
MEMORY_RADIUS_M = 50.0
# A 50 m disc at 0.4 m holds ~49k cells; obstacles occupy a small fraction,
# and the cap bounds the per-tick sort-merge well under a millisecond.
MEMORY_MAX_CELLS = 20_000
# Cumulative returns before a remembered cell is believed. A real kerb face
# collects dozens per tick; 3 rejects the residue a lone speck leaves after
# despeckle.
MEMORY_MIN_SUPPORT = 3
# Anything the semantics call a vehicle lives on the WALL CLOCK instead --
# WORLD's two-clocks argument: a mover stamped by the odometer is a streak of
# itself. The same figure as WORLD_VEHICLE_TTL_S.
MEMORY_VEHICLE_TTL_S = 0.15
# The same teleport guard (and figure) as WORLD_POSE_JUMP_RESET_M: a respawn
# must never leave the old map's walls standing in the new one.
MEMORY_POSE_JUMP_RESET_M = 25.0
# Remembered points handed to the arc scan per tick, decimated evenly. The
# scan is O(points x arcs), and 2k remembered points cost ~1-2 ms on the
# measured 6.7 ms plan.
MEMORY_MAX_QUERY_POINTS = 2000
# Road-mask returns are strided lightly before entering the memory: besides
# the road store (which needs only presence for the coverage bonus), the same
# returns are the GROUND-CONTRADICTION evidence that evicts remembered cells
# the sensors can now see tarmac through -- and that test needs per-cell
# counts to clear MEMORY_MIN_SUPPORT, so 8:1 starved it at range.
MEMORY_ROAD_STRIDE = 2
MEMORY_MAX_ROAD_CELLS = 20_000

# The semantic road-coverage BONUS -- the upgrade path the planner's
# geometric-not-semantic design note always named. The planner still decides
# drivability by geometry alone (flat grass remains drivable, which is what
# keeps unannotated maps working); coverage of road-classified returns enters
# as a negative cost, so on an annotated kerbless road the tarmac wins and on
# a map with no annotations the term simply never appears.
#
# Sized so it can never outbid safety: the free-distance term alone reaches
# 0.35 at free = 18.6 m, so any genuinely pinched arc outranks full coverage
# -- while 0.35 dwarfs the smoothness cost of a gentle correction
# ((k/K_MAX)^2 * 1.5 ~= 0.02), so the bonus alone holds the road. Uniform
# coverage shifts every candidate equally and argmin's first-occurrence tie
# still lands on the straight immediate arc.
COST_ROAD_BONUS = 0.35
# Coarse on purpose: the bonus asks "is this ON the road", not "where is the
# kerb" -- the kerb is the height band's job. 0.8 m also quarters the scatter
# cost against the display grid.
ROAD_BONUS_CELL_M = 0.8
# Grid extent, BEV metres: lateral +-16 covers a full-lock arc's 2R = 12 m
# excursion; 32 m forward covers min(free, lookahead) windows. Out-of-grid
# samples count as off-road, which is the honest reading for a candidate
# that leaves the mapped area.
ROAD_BONUS_HALF_WIDTH_M = 16.0
ROAD_BONUS_REACH_M = 32.0
# Below this many occupied cells (~25 m^2 of road) the grid is an unannotated
# map or a near-empty tick, and the term drops out exactly as nav/keep-right
# do -- absent, not guessed.
ROAD_BONUS_MIN_CELLS = 40
# Coverage samples per candidate path, over min(free, lookahead): one per
# ~4 m of a 30 m lookahead. 4 x 41 x 8 lookups is trivial.
ROAD_BONUS_SAMPLES = 8

# The steered reverse runs the same arc fan, but reversing is a different
# REGIME and two of the forward weights are provably wrong for it. Free
# distance scored against the 40 km/h braking envelope (28.7 m) made 5 m and
# 25 m of reverse room nearly indistinguishable -- so it is scored against
# what the manoeuvre needs, REVERSE_DISTANCE_M plus working margin. And
# smoothness at the forward weight beat every clearing arc: "steer least"
# while backing at 2 m/s is a tie-break, not passenger comfort, and at 1.5
# the recovery chose a blocked straight over an open diagonal every time.
REVERSE_REQUIRED_FREE_M = 10.0
REVERSE_COST_SMOOTHNESS = 0.3

# Vehicle.control() is a blocking ack. Raise to 80 to actuate every other tick
# if POLL TIME ever exceeds DISPLAY_INTERVAL_MS with self-driving engaged.
CONTROL_INTERVAL_MS = 40

# --- Parking bay detection ----------------------------------------------------
#
# Bays are found from PAINT, not from gaps between parked cars. A gap-based
# finder is what production parking assists use and it works on unannotated
# maps, but it has one fatal property here: an EMPTY lot is one enormous gap
# and offers nothing to find. An empty bay is defined by its lines, so paint is
# the only signal that survives the empty case -- which is the case this was
# asked for. (Confirmed live: bays on this map ship as annotated decals and
# read through the LiDAR, the same way `Marking check:` confirmed lane paint.)
#
# Nothing here feeds the planner or either AEB band. Detection is a display and
# selection concern only, so a wrong bay costs a bad suggestion the user can
# see and decline -- not a phantom brake or a phantom wall.

# Marking cells accumulate in their own world-anchored store, separate from
# PlanningMemory: that store is documented planner-only and is gated on
# self-driving, whereas scanning for a bay is something you do while driving
# the car yourself.
#
# Finer than MEMORY_CELL_M because the quantity being measured is a LINE'S
# OFFSET, not whether a cell is occupied. A bay line is ~0.12 m of paint, so
# 0.2 m keeps it to one or two cells across and leaves the stripe narrow
# against PARKING_STRIPE_MAX_WIDTH_M.
PARKING_MARKING_CELL_M = 0.2
# Forgotten by the METRE, the WORLD/PlanningMemory two-clocks rule -- and much
# longer than MEMORY_DISTANCE_M's 20 m on purpose. Paint does not move, the
# ego pose is ground truth so old cells are exactly as valid as new ones, and
# a lot is crossed at a crawl: at 15 km/h a 20 m window empties while you are
# still deciding. The failure this risks is a stale bay the user can see is
# stale, which is not the class of failure the planner's window guards.
PARKING_MARKING_MEMORY_M = 80.0
# Paint returns thin with range like every other ground return; past this the
# stripes arrive too sparsely to fit a line to. Must stay comfortably ABOVE
# PARKING_SCAN_RADIUS_M or the store, not the scan, is what bounds how far
# bays are found -- and it would do it silently, since the scan would simply
# receive fewer cells.
PARKING_MARKING_RADIUS_M = 70.0
# A 45 m disc at 0.2 m is ~159k cells if it were solid paint; markings occupy
# a small fraction of that and the cap bounds the per-tick sort-merge.
PARKING_MARKING_MAX_CELLS = 60_000

# Bays are only offered inside this radius, and it has to be comparable to
# how far paint is DRAWN or the view contradicts itself: WORLD renders the
# road (and its markings) to WORLD_ROAD_RADIUS_M = 100 m, so at the original
# 35 m a lot showed a full row of painted bays with outlines on only the near
# few. Measured on a straight row, detection stopped dead at 32.9 m.
#
# 60 m is what the SENSORS support all round rather than a round number: the
# roof unit owns the near bowl out to LIDAR_ROOF_FAR_M (55 m) in every
# direction, and past that only the forward road-scan wedge reaches, so bays
# behind and beside the car would stop being found anyway. The sweep's cost
# is driven by cell count and the angle count, not by the radius -- measured
# 9.1 ms at 35 m against 9.3 at 60 on the same 2,786-cell lot.
PARKING_SCAN_RADIUS_M = 60.0
# How far along a divider a gap may run before it is two separate dividers.
# This is what makes TWO FACING ROWS across an aisle work, and without it
# they are not merely degraded but completely undetectable: facing rows put
# their dividers at the SAME perpendicular offsets, so each pair merged into
# one stripe spanning both rows and every bay then failed
# PARKING_BAY_MAX_DEPTH_M. Measured, 5 bays + 5 bays became 0.
#
# Bounded on both sides: above the observation gaps along one divider (ground
# returns thin with range, so metre-scale holes are normal), and below an
# aisle, which is 6 m and up. A divider split by an occluding car into
# fragments under PARKING_BAY_MIN_DEPTH_M correctly stops bounding a bay --
# that is the honest reading of not having seen enough of it.
PARKING_STRIPE_GAP_M = 3.0
# Below this the store holds a few stray marking returns rather than a lot,
# and every geometric conclusion drawn from them would be noise.
PARKING_MIN_MARKING_CELLS = 40
# Re-detection cadence. The stripes are world-anchored and re-projected into
# the BEV frame every tick, so the drawn bays stay glued to the ground between
# scans; only the SET of bays needs the sweep, and it does not need it at
# 25 Hz.
PARKING_SCAN_INTERVAL_S = 0.5

# The stripe-direction sweep. All bay dividers in a row are parallel, so the
# right angle is the one whose PERPENDICULAR projection concentrates the cells
# into narrow peaks. Coarse pass over [0, 180) then a fine refine around the
# winner: at 1 deg a 5 m stripe smears 0.09 m at its ends, comparable to the
# offset bin, and the refine costs 21 more evaluations.
PARKING_ANGLE_COARSE_DEG = 1.0
PARKING_ANGLE_FINE_DEG = 0.1
# This MUST NOT be finer than PARKING_MARKING_CELL_M, and the reason is the
# whole sweep. Store cells sit on a 0.2 m lattice, so at a 0.1 m bin a line
# seen ALONG its length lands in alternating occupied and empty bins -- it
# reads as a row of one-bin "stripes" and scores exactly as well as the same
# line seen end-on. Measured, that tied the correct angle with the one 90
# degrees from it and the detector found nothing at all. At or above the cell
# pitch the same line fills consecutive bins, becomes one wide run, and is
# rejected, which is what the width cap is for. 0.25 leaves margin for the
# sub-cell jitter of a stored MEAN position.
#
# It costs no precision: a stripe's offset is the MEAN of its cells' own
# positions, never its bin centre. The bin decides grouping only.
PARKING_OFFSET_BIN_M = 0.25
# A run of occupied offset bins wider than this is not a stripe seen end-on --
# it is a stripe seen ALONG its length, i.e. the wrong angle, or a broad decal.
#
# This width cap is also what makes the sweep's score honest, and a plain
# "sum of squared bin counts" concentration score is NOT: a single long line
# perpendicular to the bays (the head line many lots have) piles into one bin
# and, squared, outscores eight genuine stripes. Scoring instead by HOW MANY
# CELLS lie in narrow runs makes the eight stripes win on their own mass.
# 1.0 rather than 0.7 because it is also the tolerance on the SWEEP ANGLE: a
# divider whose own direction differs from the swept angle by `t` spreads its
# length over `L * sin(t)`, so at 5 m long a 0.7 m cap silently drops any
# divider more than 8 degrees off the sweep. A row following a curved wall is
# exactly that -- measured on a 150 m-radius row of ten bays, one divider was
# lost and with it a bay. Still less than half the narrowest bay width, so it
# cannot merge two adjacent dividers into one run.
PARKING_STRIPE_MAX_WIDTH_M = 1.0
# Cells in one offset bin before that bin counts as occupied at all, and this
# is what makes a head line survivable rather than merely outvoted. Seen from
# the dividers' own angle a line across their heads does not pile into one bin
# -- it smears along the whole row and BRIDGES every divider's bin, merging
# eight clean stripes into one run too wide to be a stripe. Measured, that
# took the scene from seven bays to none.
#
# The discriminator is that a divider seen end-on stacks its entire length
# into one bin (tens of cells) while anything crossing the projection leaves
# only a sample or two per bin. 3 is strictly weaker than the
# PARKING_MIN_STRIPE_CELLS filter that follows it, so it removes only the
# smear.
PARKING_MIN_BIN_CELLS = 3
# Cells backing one stripe before it is believed. At 0.2 m a 5 m line is ~25
# cells even sampled once; 8 rejects fragments without needing the whole line.
PARKING_MIN_STRIPE_CELLS = 8

# Bay dimensions. Width is the gap between adjacent dividers; a UK/EU bay is
# 2.4-2.5 m and a US one 2.7-2.9, so the band covers both with slack for the
# cell pitch. The lower bound is what rejects the two halves of a DOUBLE
# divider line, which some lots paint 0.3-0.5 m apart.
PARKING_BAY_WIDTH_MIN_M = 2.1
PARKING_BAY_WIDTH_MAX_M = 3.4
# Depth is the SHORTER DIVIDER'S LENGTH, not the overlap between the two, and
# that distinction is what makes ANGLED (herringbone) lots work at all.
#
# In an angled lot the dividers start along a common aisle edge and run off at
# an angle, so adjacent dividers are STAGGERED along their own direction by
# `width / tan(angle)`. Measuring depth as their overlap subtracts that
# stagger from a number that should not depend on it: measured on a proper
# 2.5 m x 5 m angled lot, a 60-degree bay -- the commonest angled layout --
# has 3.56 m of overlap against this 3.6 m floor and was rejected by 4 cm,
# while 45 degrees gave 2.50 m and 30 degrees 0.67 m. Perpendicular bays were
# fine, so a lot whose row curves has some bays detected and some not, which
# is exactly how it was reported.
#
# The maximum still rejects an AISLE: its two long edge lines are parallel and
# plausibly spaced, and it is their LENGTH that gives them away -- a test that
# works on the length directly rather than on the overlap.
PARKING_BAY_MIN_DEPTH_M = 3.6
PARKING_BAY_MAX_DEPTH_M = 7.5
# The overlap is still needed, but as an ADJACENCY test rather than as the
# depth: two dividers that barely overlap are not bounding one bay. This is
# what keeps two FACING rows across an aisle apart, where the overlap is
# strongly negative (measured -5 m), while leaving an angled bay's honest
# stagger alone. Kept well under PARKING_BAY_MIN_DEPTH_M so it constrains
# only the pairing, never the depth.
#
# 0.5 rather than 1.0 because the stagger grows as the bay angle falls: a
# 30-degree lot leaves only 0.67 m of overlap between neighbours and was the
# one layout still missed at 1.0. Facing rows are separated by metres of
# NEGATIVE overlap, so the margin costs nothing there -- verified at 0.5 that
# facing rows still return 10 bays rather than merging into 20.
PARKING_STRIPE_MIN_OVERLAP_M = 0.5

# Occupancy. The rectangle is shrunk by this before anything inside it counts,
# because the dividers themselves, the kerb at the head and a neighbour's
# wing mirror all sit on or just inside the boundary.
PARKING_OCCUPANCY_MARGIN_M = 0.3
# Returns above the ego's ground plane that count as something standing in the
# bay. Above the paint and the road crown, below a kerb face's top: a car
# fills this several hundred times over, so the threshold is not doing subtle
# work -- it is keeping the surface itself out.
PARKING_OCCUPANCY_MIN_HEIGHT_M = 0.30
PARKING_OCCUPANCY_MIN_CELLS = 4
# Nearest-first cap on what is drawn and clickable. Raised with the scan
# radius: a real lot inside 60 m holds far more than 20 bays, and a cap that
# binds looks exactly like the detection failures this feature has already
# had -- painted bays on screen with no outline on them. 48 outlines is about
# 1,150 vertices, which is nothing beside the ground mesh.
PARKING_MAX_SLOTS = 48
# How many row ORIENTATIONS to look for. The sweep returns one angle, so a
# single pass keeps whichever row carries the most paint and silently drops
# every row lying at a different angle -- in a real lot, the row you happen to
# be facing keeps its bays and the rest of the lot has none, which reads as a
# sensor that only looks forwards. It is not: nothing in the detector filters
# by bearing, and a row directly behind the car is found perfectly when it is
# the only one there. Each pass consumes the cells its stripes claimed and
# re-sweeps the remainder.
#
# 3 covers the shapes that actually occur -- two facing rows either side of an
# aisle, plus a perpendicular row along an end wall. Each pass is another full
# sweep over the residual, which shrinks fast, so the cost is well under the
# first pass's.
PARKING_MAX_ROWS = 3
# A selection is held as a WORLD pose, never as an index into the last scan:
# the set is rebuilt every PARKING_SCAN_INTERVAL_S and indices are not stable
# across a rebuild. After each scan the held pose is re-matched to the nearest
# bay centre inside this radius, and a selection that matches nothing survives
# unmatched rather than silently jumping to a different bay.
PARKING_SELECT_MATCH_M = 1.5

# How far two dividers' directions may differ and still bound one bay. This
# exists because pairing now runs across every sweep at once: a slightly
# CURVED row (one following a wall, which is what a real lot does) has its
# dividers claimed by two or three different sweep angles, and pairing within
# a pass meant neighbours found on different passes could never meet.
# Measured live on such a lot: 18 dividers -> 12 bays with 6 unpaired, and on
# a single sweep 10 dividers -> 5 bays with 5 unpaired.
#
# Wide enough to span the sweep-angle steps a curve is split across, narrow
# enough that two genuinely different rows never pair -- those differ by tens
# of degrees.
PARKING_STRIPE_ANGLE_TOL_DEG = 15.0

# --- Driving into the bay -----------------------------------------------------
#
# The manoeuvre, as opposed to finding the bay. Its own controller and its own
# constants, because the road laws are wrong here in ways that are not a matter
# of tuning -- see parking_drive's module docstring.

# Manoeuvring speed, and it is chosen to sit BELOW AEB_MIN_SPEED_MPS (2.0).
# That is not a coincidence to be tidied away: at parking speed the forward
# emergency brake stays in STANDBY, so it cannot fire at the kerbs, walls and
# neighbouring cars a park deliberately drives close to. Parking therefore
# does its own corridor check (`blocking_distance`) rather than leaning on a
# system that is, correctly, not watching.
PARKING_DRIVE_SPEED_MPS = 1.4
# Enough to keep rolling against a gentle grade without lurching; the
# distance-to-go law tapers through it to the stop.
PARKING_DRIVE_CREEP_MPS = 0.5
# Comfortable deceleration for the approach. Gentle on purpose: the whole
# manoeuvre is under 15 m and there is nothing to be gained by hurrying the
# last of it.
PARKING_DRIVE_DECEL_MPS2 = 1.0
# Pure-pursuit lookahead ceiling. It shortens as the car slows, because a
# fixed lookahead cuts the corner into the bay -- which is exactly where the
# tolerance is smallest.
PARKING_DRIVE_LOOKAHEAD_M = 3.0
# How far outside the mouth the car is brought onto the bay's own axis. This
# is what makes the entry straight rather than a swerve across the lines, and
# it is grown on retry when the sweep would be too tight.
PARKING_DRIVE_APPROACH_M = 5.0
# Abort if the car strays this far from the planned line.
PARKING_DRIVE_MAX_CROSS_TRACK_M = 1.5

# A committed path is replanned only after a genuine tracking failure, not on
# every tick. Progress means at least a few centimetres closer to the end.
PARKING_PROGRESS_TIMEOUT_S = 3.0
# How close the nose stops to the head of the bay.
PARKING_HEAD_CLEARANCE_M = 0.5
# Half-width margin added to the body for the manoeuvre's own corridor check.
# Tighter than the planner's 0.35: parking is close work by definition, and a
# generous margin here refuses bays the car fits in comfortably.
PARKING_BODY_CLEARANCE_M = 0.18
# Remaining path length at which the manoeuvre is finished.
PARKING_ARRIVE_TOLERANCE_M = 0.15
# Held once parked. A finished park must STAY put -- releasing at the stop
# line lets the car roll on out of the bay -- so this is a hold, unlike every
# teardown path in the worker, which deliberately hands back a coasting car.
PARKING_STOP_BRAKE = 0.55
# Samples along the swept approach. The curvature check and the corridor check
# both run on these, so it is a resolution, not a drawing detail.
PARKING_PATH_SAMPLES = 48
# How far the car may be past a path's ideal turn-in point and still have that
# path built for it. This is TRACKING LAG, not slop: the car follows the path
# with a lookahead, so it reaches the turn-in marginally beyond it and the
# exact construction goes infeasible by centimetres -- measured, a turning
# manoeuvre reported unreachable at run_in = -0.016 m, a few centimetres from
# arriving, and stopped on the line. Inside this the arc simply starts now.
PARKING_PATH_SLACK_M = 0.75
# Distance inside which the creep floor is released so the profile can reach
# zero. Above it a floor keeps the car rolling against a gentle grade instead
# of stalling short of the bay; below it that same floor is what would carry
# the car straight through the stop point.
PARKING_DRIVE_CREEP_HOLD_M = 1.0
# Heading still to be lost, above which the manoeuvre holds a crawl. Distance
# alone let the car cross the stop plane at cruising speed with the wheel
# still wound on -- centred in the bay but 7.2 degrees off square. Slowing
# while there is turn left gives the tracker time to straighten.
PARKING_TURN_SLOW_DEG = 8.0
# Where the car stops before backing into a bay, in the bay's own frame:
# how far out to the side, and how far past the mouth. These are the poses a
# driver actually uses -- alongside and a little past the space, squared to
# the aisle -- and they are searched rather than fixed because which one
# works depends on how much aisle there is and where the car starts.
PARKING_SETUP_ACROSS_M = (2.6, 3.4, 4.2, 5.0, 5.8)
PARKING_SETUP_OUT_M = (0.5, 1.5, 2.5, 3.5, 4.5)

# At or below this the car counts as stopped for a gear change. Tighter than
# STALL_SPEED_MPS because shifting a moving box is exactly what this exists to
# prevent, and looser than AEB_STOPPED_SPEED_MPS because a shift does not need
# the precision a brake release does.
PARKING_SHIFT_SPEED_MPS = 0.08
# Distance to a leg's pose inside which "no path reaches it" means the car is
# essentially ON it rather than unable to get there. The same endgame the
# single forward move hit, now once per leg: the tracker rolls a little past
# the pose and no forward construction reaches back to it.
PARKING_LEG_CLOSE_M = 1.5
# How square to its pose a SETUP leg must leave the car before the next leg
# begins. Position alone declared the leg made with the car still turning, the
# reverse then would not solve from where it actually was, and the manoeuvre
# cycled between re-planning and arriving at the same setup.
PARKING_LEG_SQUARE_DEG = 12.0
# Re-plans allowed before handing back. A plan that keeps producing a sequence
# the car cannot drive would otherwise cycle for ever.
PARKING_MAX_REPLANS = 4
# How far the car may travel before a bay the scan has stopped finding is
# forgotten. Detection is a chain of filters over an accumulating cloud, so a
# bay near several thresholds at once drops out and returns a moment later --
# reported as bays flashing, and worse, as the SELECTION vanishing with them.
#
# Measured in METRES like the WORLD stores and PlanningMemory, for the same
# reason: paint does not move and the ego pose is ground truth, so what makes
# a bay stale is the car leaving, not the clock passing. A freshly found bay
# always replaces its remembered twin, so this only ever fills gaps.
PARKING_BAY_MEMORY_M = 25.0
# Grid a bay's centre is rounded to when deciding whether two scans found the
# SAME bay. Coarse on purpose: a bay is re-measured every scan and its centre
# wanders a few centimetres as cells come and go, so an exact key would make
# every scan a different bay and remember nothing.
PARKING_BAY_MATCH_M = 1.0

# --- Hybrid A*: the pose-space planner ----------------------------------------
#
# Searches the car's state (position AND heading) by expanding short arcs it
# could really steer, so every node is drivable by construction. Reeds-Shepp
# supplies both the heuristic and an analytic shortcut to the goal.

# Grid the search collapses states onto. Too fine and it never revisits a
# state, so the frontier explodes; too coarse and genuinely different poses
# get merged and the path jinks.
HYBRID_CELL_M = 0.5
HYBRID_HEADING_BINS = 36
# One motion primitive. About a cell's worth of travel: shorter multiplies the
# expansions for no extra reach, longer skips past thin gaps.
HYBRID_STEP_M = 0.7
# Reversing is allowed but not free -- a plan that shuffles when it could
# simply drive in is a worse plan even when it is shorter.
HYBRID_REVERSE_PENALTY = 1.6
# Changing direction costs more than reversing does, because each one is a
# full stop, a gear change and a wait for the box to confirm. This is what
# keeps the answer to a manoeuvre a driver would recognise rather than a
# five-point shuffle that happens to measure shortest.
HYBRID_GEAR_PENALTY = 3.0
HYBRID_STEER_PENALTY = 0.15
# Cost per body sample standing in never-observed ground. Traversable at a
# price rather than forbidden: forbidding it strands the car in a lot it has
# only partly seen, while pricing it makes a route over ground the sensors
# have actually returned beat a route through the unseen.
HYBRID_UNKNOWN_PENALTY = 0.25
HYBRID_GOAL_RADIUS_M = 0.4
HYBRID_GOAL_HEADING_DEG = 8.0
# How often the Reeds-Shepp shot at the goal is tried. It is the expensive
# part of an expansion and far from the goal it almost never clears, so it is
# tried periodically -- and always once the search is near.
HYBRID_ANALYTIC_INTERVAL = 8
# Hard bound on the search. A parking manoeuvre that needs more than this is
# one to hand back rather than to keep grinding at, and it runs off the
# control tick so the ceiling is about answering promptly, not about safety.
HYBRID_MAX_EXPANSIONS = 12000
# How far past a leg's pose still counts as having made it. This is for
# OVERSHOOT -- the car tracks with a lookahead and rolls a few centimetres
# beyond -- so it must stay small: at a metre it fired while the car was
# still a metre off to the side and latched the park there, 0.96 m off the
# centreline of a 3.18 m bay.
PARKING_OVERSHOOT_M = 0.4
# How long the manoeuvre waits, stopped and asking for a gear, before it
# proceeds without confirmation. Confirmation is the evidence a shift took --
# but waiting for it forever is a hang, and an UNREADABLE gearbox is exactly
# that: `electrics` unavailable means the reported gear is None, neither the
# forward nor the reverse test can pass, and the car sits braking. Long
# enough that a box which does report gets to, short enough that a box which
# never will does not strand the manoeuvre.
PARKING_SHIFT_DWELL_S = 1.2

# Blockage and terminal-state hysteresis. These make WAITING and success real
# states rather than one-frame labels that can alternate with motion.
PARKING_BLOCKED_CLEAR_DWELL_S = 0.5
PARKING_SUCCESS_DWELL_S = 0.5
PARKING_SUCCESS_SPEED_MPS = 0.05
# Bay geometry comes from rasterised paint, so terminal containment has a
# small measurement envelope while remaining substantially tighter than the
# detector's bay matching tolerance.
PARKING_SUCCESS_POSITION_M = 0.55
PARKING_SUCCESS_HEADING_DEG = 12.0
PARKING_SUCCESS_BOUNDARY_TOLERANCE_M = 0.30
