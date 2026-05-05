import math
import random
import gin
from mathutils import Vector, Euler, Matrix

class _BaseFlightPolicy:
    def __init__(self, rig_spec, terrain_bvh=None):
        self.rig_spec = rig_spec
        self.terrain_bvh = terrain_bvh
        self._frame = 0
        self._pos = Vector((0, 0, 100))
        self._heading = 0.0
        self._altitude = random.uniform(*rig_spec.altitude_range)
        self._speed = random.uniform(*rig_spec.speed_range)
        self._orbit_angle = 0.0
        self._prev_positions = []
    
    def _sample_altitude(self):
        lo, hi = self.rig_spec.altitude_range
        return random.uniform(lo, hi)
    
    def _sample_speed(self):
        lo, hi = self.rig_spec.speed_range
        return random.uniform(lo, hi)
    
    def _get_terrain_z(self, x, y, default=0):
        if self.terrain_bvh is None:
            return default
        from mathutils import Vector
        origin = Vector((x, y, 10000))
        direction = Vector((0, 0, -1))
        loc, _, _, _ = self.terrain_bvh.ray_cast(origin, direction)
        return loc.z if loc else default
    
    def __call__(self, scene, cam_rig, frame):
        self._frame = frame
        pos, rot = self.propose_pose(frame)
        cam_rig.location = pos
        cam_rig.rotation_euler = rot
        self._prev_positions.append(pos.copy())
        return pos, rot
    
    def propose_pose(self, frame):
        return Vector((0, 0, 100)), Euler((0, 0, 0))

class ISROrbitPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        center = Vector((0, 0, 0))
        look_angle = math.radians(random.uniform(*self.rig_spec.look_angle_range))
        self._altitude = self._sample_altitude()
        radius = self._altitude * math.tan(look_angle)
        orbit_rate = self._sample_speed() / radius
        self._orbit_angle += orbit_rate
        x = center.x + radius * math.cos(self._orbit_angle)
        y = center.y + radius * math.sin(self._orbit_angle)
        z = self._altitude
        pos = Vector((x, y, z))
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
        rot = Euler((0, 0, 0))
        return Vector((x, y, z)), rot

class ISRLoiterPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        center = Vector((0, 0, 0))
        radius = 50.0
        loiter_rate = 0.02
        self._orbit_angle += loiter_rate
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
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        z = self._altitude
        if leg_idx == 0:
            x = -leg_length/2 + (frame_in_leg / frames_per_leg) * leg_length
            y = leg_length/2
        elif leg_idx == 1:
            x = leg_length/2
            y = leg_length/2 - (frame_in_leg / frames_per_leg) * leg_length
        elif leg_idx == 2:
            x = leg_length/2 - (frame_in_leg / frames_per_leg) * leg_length
            y = -leg_length/2
        else:
            x = -leg_length/2
            y = -leg_length/2 + (frame_in_leg / frames_per_leg) * leg_length
        heading = 0 if leg_idx in (0, 2) else math.pi
        rot = Euler((0, 0, heading))
        return Vector((x, y, z)), rot

class FPVRacingPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        self._speed = random.uniform(*self.rig_spec.speed_range)
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        turn_rate = 0.05
        self._heading += turn_rate * (1 if frame % 200 < 100 else -1)
        velocity = self._speed
        x = self._pos.x + math.cos(self._heading) * velocity * 0.1
        y = self._pos.y + math.sin(self._heading) * velocity * 0.1
        z = self._altitude
        self._pos = Vector((x, y, z))
        pitch = -0.2
        bank = turn_rate * velocity * 0.5
        rot = Euler((pitch, bank, self._heading))
        return self._pos, rot

class FPVScoutPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        self._speed = random.uniform(*self.rig_spec.speed_range) * 0.5
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        self._heading += 0.01
        x = self._pos.x + math.cos(self._heading) * self._speed * 0.1
        y = self._pos.y + math.sin(self._heading) * self._speed * 0.1
        z = self._altitude
        self._pos = Vector((x, y, z))
        rot = Euler((0, 0, self._heading))
        return self._pos, rot

class UGVWheeledPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        self._speed = random.uniform(*self.rig_spec.speed_range)
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        self._heading += random.uniform(-0.02, 0.02)
        x = self._pos.x + math.cos(self._heading) * self._speed * 0.1
        y = self._pos.y + math.sin(self._heading) * self._speed * 0.1
        terrain_z = self._get_terrain_z(x, y)
        suspension_bounce = math.sin(frame * 0.3) * 0.05
        z = terrain_z + self._altitude + suspension_bounce
        self._pos = Vector((x, y, z))
        rot = Euler((0, 0, self._heading))
        return self._pos, rot

class UGVTrackedPolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        self._speed = random.uniform(*self.rig_spec.speed_range)
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        self._heading += random.uniform(-0.03, 0.03)
        x = self._pos.x + math.cos(self._heading) * self._speed * 0.1
        y = self._pos.y + math.sin(self._heading) * self._speed * 0.1
        terrain_z = self._get_terrain_z(x, y)
        track_clatter = math.sin(frame * 2.0) * 0.03 + math.sin(frame * 5.0) * 0.01
        z = terrain_z + self._altitude + track_clatter
        self._pos = Vector((x, y, z))
        rot = Euler((0, 0, self._heading))
        return self._pos, rot

class SatellitePolicy(_BaseFlightPolicy):
    def propose_pose(self, frame):
        self._altitude = random.uniform(*self.rig_spec.altitude_range)
        speed = self._sample_speed()
        x = (frame - 50) * speed * 0.1
        y = 0
        z = self._altitude
        return Vector((x, y, z)), Euler((0, 0, 0))

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
