import math
import random
import gin
from mathutils import Vector, Euler


class _BaseFlightPolicy:
    def __init__(self, rig_spec, terrain_bvh=None):
        self.rig_spec = rig_spec
        self.terrain_bvh = terrain_bvh
        self._frame = 0
        self._pos = Vector((0, 0, 100))
        self._heading = 0.0
        self._orbit_angle = 0.0

    def _sample_altitude(self):
        lo, hi = self.rig_spec.altitude_range
        return random.uniform(lo, hi)

    def _sample_speed(self):
        lo, hi = self.rig_spec.speed_range
        return random.uniform(lo, hi)

    def __call__(self, scene, cam_rig, frame):
        self._frame = frame
        pos, rot = self.propose_pose(frame)
        cam_rig.location = pos
        cam_rig.rotation_euler = rot
        return pos, rot

    def propose_pose(self, frame):
        return Vector((0, 0, 100)), Euler((0, 0, 0))


class ISROrbitPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        center = Vector((0, 0, 0))
        look_angle = math.radians(random.uniform(*self.rig_spec.look_angle_range))
        alt = self._sample_altitude()
        radius = alt * math.tan(look_angle)
        orbit_rate = self._sample_speed() / max(radius, 1.0)
        self._orbit_angle += orbit_rate
        x = center.x + radius * math.cos(self._orbit_angle)
        y = center.y + radius * math.sin(self._orbit_angle)
        pos = Vector((x, y, alt))
        look_dir = (center - pos).normalized()
        rot = look_dir.to_track_quat('-Y', 'Z').to_euler()
        return pos, rot


class ISRRasterPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        strip_length = 1000
        strip_spacing = 50
        strips = 10
        total_frames_per_strip = 100
        strip_idx = (frame // total_frames_per_strip) % strips
        frame_in_strip = frame % total_frames_per_strip
        direction = 1 if strip_idx % 2 == 0 else -1
        x = (frame_in_strip / total_frames_per_strip - 0.5) * strip_length * direction
        y = strip_idx * strip_spacing - (strips * strip_spacing) / 2
        z = self._sample_altitude()
        return Vector((x, y, z)), Euler((0, 0, 0))


class ISRLoiterPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        center = Vector((0, 0, 0))
        radius = 50.0
        self._orbit_angle += 0.02
        drift_x = math.sin(frame * 0.001) * 20
        drift_y = math.cos(frame * 0.0013) * 20
        x = center.x + radius * math.cos(self._orbit_angle) + drift_x
        y = center.y + radius * math.sin(self._orbit_angle) + drift_y
        z = self._sample_altitude()
        look_dir = (center - Vector((x, y, z))).normalized()
        rot = look_dir.to_track_quat('-Y', 'Z').to_euler()
        return Vector((x, y, z)), rot


class ISRPlanePolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        leg_length = 5000
        legs = 4
        frames_per_leg = 200
        leg_idx = (frame // frames_per_leg) % legs
        frame_in_leg = frame % frames_per_leg
        alt = random.uniform(*self.rig_spec.altitude_range)
        if leg_idx == 0:
            x = -leg_length / 2 + (frame_in_leg / frames_per_leg) * leg_length
            y = leg_length / 2
        elif leg_idx == 1:
            x = leg_length / 2
            y = leg_length / 2 - (frame_in_leg / frames_per_leg) * leg_length
        elif leg_idx == 2:
            x = leg_length / 2 - (frame_in_leg / frames_per_leg) * leg_length
            y = -leg_length / 2
        else:
            x = -leg_length / 2
            y = -leg_length / 2 + (frame_in_leg / frames_per_leg) * leg_length
        heading = 0 if leg_idx in (0, 2) else math.pi
        return Vector((x, y, alt)), Euler((0, 0, heading))


class FPVRacingPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        speed = random.uniform(*self.rig_spec.speed_range)
        alt = random.uniform(*self.rig_spec.altitude_range)
        turn_rate = 0.05
        self._heading += turn_rate * (1 if frame % 200 < 100 else -1)
        x = self._pos.x + math.cos(self._heading) * speed * 0.1
        y = self._pos.y + math.sin(self._heading) * speed * 0.1
        self._pos = Vector((x, y, alt))
        pitch = -0.2
        bank = turn_rate * speed * 0.5
        return self._pos, Euler((pitch, bank, self._heading))


class FPVScoutPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        speed = random.uniform(*self.rig_spec.speed_range) * 0.5
        alt = random.uniform(*self.rig_spec.altitude_range)
        self._heading += 0.01
        x = self._pos.x + math.cos(self._heading) * speed * 0.1
        y = self._pos.y + math.sin(self._heading) * speed * 0.1
        self._pos = Vector((x, y, alt))
        return self._pos, Euler((0, 0, self._heading))


class UGVWheeledPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        speed = random.uniform(*self.rig_spec.speed_range)
        alt = random.uniform(*self.rig_spec.altitude_range)
        self._heading += random.uniform(-0.02, 0.02)
        x = self._pos.x + math.cos(self._heading) * speed * 0.1
        y = self._pos.y + math.sin(self._heading) * speed * 0.1
        suspension_bounce = math.sin(frame * 0.3) * 0.05
        z = alt + suspension_bounce
        self._pos = Vector((x, y, z))
        return self._pos, Euler((0, 0, self._heading))


class UGVTrackedPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        speed = random.uniform(*self.rig_spec.speed_range)
        alt = random.uniform(*self.rig_spec.altitude_range)
        self._heading += random.uniform(-0.03, 0.03)
        x = self._pos.x + math.cos(self._heading) * speed * 0.1
        y = self._pos.y + math.sin(self._heading) * speed * 0.1
        track_clatter = math.sin(frame * 2.0) * 0.03 + math.sin(frame * 5.0) * 0.01
        z = alt + track_clatter
        self._pos = Vector((x, y, z))
        return self._pos, Euler((0, 0, self._heading))


class SatellitePolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        alt = random.uniform(*self.rig_spec.altitude_range)
        speed = self._sample_speed()
        x = (frame - 50) * speed * 0.1
        return Vector((x, 0, alt)), Euler((0, 0, 0))


_POLICY_MAP = {
    "isr_orbit": ISROrbitPolicy,
    "isr_raster": ISRRasterPolicy,
    "isr_loiter": ISRLoiterPolicy,
    "isr_plane": ISRPlanePolicy,
    "fpv_racing": FPVRacingPolicy,
    "fpv_scout": FPVScoutPolicy,
    "ugv_wheeled": UGVWheeledPolicy,
    "ugv_tracked": UGVTrackedPolicy,
    "satellite": SatellitePolicy,
}


def get_flight_policy(platform_name, rig_spec, terrain_bvh=None):
    policy_cls = _POLICY_MAP.get(platform_name)
    if policy_cls is None:
        return None
    return policy_cls(rig_spec, terrain_bvh)
