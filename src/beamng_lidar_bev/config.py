from __future__ import annotations

from pathlib import Path

APP_NAME = "BeamNG LiDAR BEV"
APP_VERSION = "1.2.0"

BEAMNG_EXE = Path(r"C:\Users\initd\Documents\BeamNG.tech.v0.38.5.0\BeamNG.tech.exe")
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
# COSTS RAYS, and that is the honest trade. At density 12.5 it is ~4 standard
# units of ray budget on top of the existing ~7 (front counts 4), so roughly
# +57% total -- still less than halving LIDAR_DENSITY globally (+100%) and aimed
# at the one axis that is actually short. If the simulator bogs down, raise this
# to 25.0 first: it halves the azimuth sampling (0.9 m -> 1.8 m at 30 m) while
# leaving the radial win untouched. The attach-time `Sensor reach:` line prints
# this unit's own return count and furthest return, which is the live check.
LIDAR_ROOF_MAX_DISTANCE_M = 100.0
LIDAR_ROOF_HORIZONTAL_FOV_DEG = 170.0
# Matches the front unit, and for the same reason: at 30 m the radial spacing is
# 0.53 m, so 3.5 deg of azimuth (1.83 m) would make azimuth the limiter all over
# again. 12.5 puts ~192 columns across the wedge, i.e. 0.46 m at 30 m -- a
# roughly isotropic lattice, which is what a surface mesh wants.
LIDAR_ROOF_DENSITY = 12.5
# The ground annulus the aperture is fitted to, in metres. NEAR must stay under
# the ~7 m the 0.20 m mounts resolve in a single frame or the two sets leave a
# blind ring; FAR matches WORLD_RADIUS_M, since filling road the renderer then
# discards buys nothing. Widening the span costs ring spacing linearly, because
# the same 256 channels have to cover it.
LIDAR_ROOF_NEAR_M = 6.0
LIDAR_ROOF_FAR_M = 80.0
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
# This is the one dial to turn: drop to 25.0 for roughly double the returns
# everywhere, at double the ray-tracing cost inside BeamNG.
LIDAR_DENSITY = 50.0
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
# surface arrives as a checkerboard of disconnected quads. 3 cells is 0.75 m,
# which covers the azimuth spacing out to about 45 m and is far short of
# anything the car could drive through.
WORLD_ROAD_BRIDGE_CELLS = 3
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
# The ROAD SURFACE is not, and this is a hard sampling limit rather than a
# tuning choice. A channel at depression theta meets the ground at r = h/tan
# theta, so consecutive ground RINGS land dr = (r^2/h)*dtheta apart: with the
# roof unit that is 0.24 m at 20 m but ~1.5 m at 50 m and ~6 m at 100 m. Past
# about 70 m there is no surface to reconstruct -- only isolated rings metres
# apart -- and meshing them produces stripes that read as a corrugated road
# rather than as distance. Beyond this the ground is simply not drawn, which is
# honest: nothing was measured there.
WORLD_ROAD_RADIUS_M = 70.0
# The outer band of the road surface is dissolved into the air rather than cut
# off, because the road stops at its own radius while everything else runs on to
# WORLD_RADIUS_M. A hard rim reads as a cliff -- a drawn edge where there is
# only the end of what the sensors measured.
WORLD_EDGE_FADE_M = 14.0
WORLD_POSE_JUMP_RESET_M = 25.0
WORLD_ACTOR_REGISTRY_INTERVAL_S = 1.0
WORLD_ACTOR_STATE_INTERVAL_S = 0.1
WORLD_ACTOR_COAST_S = 0.35
WORLD_ACTOR_FADE_S = 0.8
WORLD_MAX_UNCERTAIN_POINTS = 2_000

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
WORLD_COLUMN_SIZE_M = 0.25
# Vertical quantisation of the voxel store. Finer than the horizontal size buys
# nothing -- vertical ring spacing on a wall is 0.04 m at 20 m, thirty times
# denser than azimuth -- and coarser starts merging a kerb into the pavement
# behind it.
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
WORLD_MAX_COLUMNS = 90_000
# Input bound, applied before binning. Far higher than the old 4,000 render cap
# because binning is one vectorised pass over the cloud, not per-point geometry.
WORLD_MAX_BOUNDARY_POINTS = 60_000
# Slabs only merge sideways when they sit at the same altitude AND have the
# same height, so a 10 m facade and the 0.15 m kerb in front of it stay
# separate rather than averaging into one waist-high wall.
WORLD_SLAB_HEIGHT_BUCKET_M = 0.5
# How many empty columns may be bridged between two observed ones. At 0.25 m
# columns this spans the 1.24 m azimuth stripe gap measured at 20 m. It is an
# inference, but a sound one: a return either side at the same height means the
# ray passed through, so the surface between them was there to be hit.
WORLD_COLUMN_BRIDGE_CELLS = 4
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
WORLD_PATH_RGB = "#4ea8f2"
WORLD_PATH_ALERT_RGB = "#c0271e"
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
# This constant is now shared: the speed law asks the controller for the same
# rate to work out how far the car travels while the steering winds on, so
# corner-entry braking and steering response cannot drift apart.
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
# Once stopped, hold this long and then hand the car back. Without it an AEB
# that stopped you a metre from a wall would keep the brake on for as long as
# the wall was still there, which is a trapped car rather than a safe one.
AEB_STOPPED_HOLD_S = 2.0
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

# Navigation hint. The in-game bigmap route is used for ONE thing: which way to
# go at a junction. It is a heading term in the arc cost, never a path to
# follow, and its absence simply drops the term.
NAV_LOOKAHEAD_M = 18.0
# A blocking Lua round-trip, so it runs far slower than the display loop. The
# route only changes when the player sets a new destination.
NAV_POLL_INTERVAL_MS = 1000

# Vehicle.control() is a blocking ack. Raise to 80 to actuate every other tick
# if POLL TIME ever exceeds DISPLAY_INTERVAL_MS with self-driving engaged.
CONTROL_INTERVAL_MS = 40
