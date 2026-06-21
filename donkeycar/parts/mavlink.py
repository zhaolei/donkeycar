#!/usr/bin/env python3
"""
MAVLink support for Donkeycar.

This module lets a Donkeycar talk to a MAVLink flight controller
(PX4 by default, with hooks for ArduPilot). It provides:

* ``MavlinkConnection`` - a shared link manager that runs a background
  reader thread, tracks the latest RC channel values / arm state and
  exposes helpers to arm/disarm and to send ``RC_CHANNELS_OVERRIDE``.
* ``MavlinkController`` - a Donkeycar *controller* part. It reads RC
  channels from the flight controller and produces
  ``user/steering``, ``user/throttle``, ``user/mode`` and ``recording``.
  A configurable RC channel selects autopilot (``local``) vs manual
  (``user``) mode and, optionally, an arm switch.
* ``MavlinkDriver`` - a Donkeycar *drivetrain* part. It converts the
  final ``steering`` / ``throttle`` into ``RC_CHANNELS_OVERRIDE`` PWM
  values. It supports a neutral-center throttle with reverse over the
  1000-2000us range and two-wheel differential drive.
* ``MavlinkArmer`` - a tiny part used to arm/disarm from a web button.

The link is created once per process (keyed by connection string) so the
controller, the driver and the web configuration page all share the same
serial/UDP connection and the same live-tunable parameter dictionary.
"""

import time
import logging
import threading

logger = logging.getLogger(__name__)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


#
# Default live-tunable parameters. The MAVLINK config dict from
# myconfig.py is merged on top of these so that older config files keep
# working and the web page always has a complete set of keys to edit.
#
DEFAULT_PARAMS = {
    # --- link ---
    "MAVLINK_CONNECTION": "/dev/ttyS0",
    "MAVLINK_BAUDRATE": 921600,
    "MAVLINK_AUTOPILOT": "px4",      # "px4" or "ardupilot"
    "MAVLINK_SOURCE_SYSTEM": 255,    # this companion's system id (GCS id)
    "MAVLINK_TARGET_SYSTEM": 0,      # 0 => auto-detect from heartbeat
    "MAVLINK_TARGET_COMPONENT": 0,   # 0 => auto-detect from heartbeat

    # --- reading RC (manual control + mode switch) ---
    "RC_STEERING_CHANNEL": 1,        # 1-based RC channel for steering
    "RC_THROTTLE_CHANNEL": 3,        # 1-based RC channel for throttle
    "RC_STEERING_MIN": 1000,
    "RC_STEERING_MID": 1500,
    "RC_STEERING_MAX": 2000,
    "RC_STEERING_INVERT": False,
    "RC_THROTTLE_MIN": 1000,
    "RC_THROTTLE_MID": 1500,
    "RC_THROTTLE_MAX": 2000,
    "RC_THROTTLE_INVERT": False,
    "RC_THROTTLE_HAS_REVERSE": True,  # throttle stick centered = stop, below = reverse
    "RC_DEADZONE": 20,                # us deadzone around the mid value

    # --- mode switch channel (auto vs manual) ---
    "RC_MODE_CHANNEL": 5,             # 0 to disable (mode then driven by web/default)
    "RC_MODE_3POSITION": False,       # False=2-pos (user/local), True=3-pos
    "RC_MODE_LOW_PWM": 1300,          # below => user
    "RC_MODE_HIGH_PWM": 1700,         # above => local (3-pos: middle => local_angle)

    # --- arming ---
    "ARM_CHANNEL": 0,                 # 0 to disable channel-based arming
    "ARM_PWM_THRESHOLD": 1700,        # above => arm, below => disarm
    "AUTO_ARM": False,                # arm automatically once link is up
    "ARM_FORCE": False,               # force (bypass pre-arm checks)

    # --- writing override (drivetrain) ---
    "OVERRIDE_STEERING_CHANNEL": 1,
    "OVERRIDE_THROTTLE_CHANNEL": 3,
    "OUT_STEERING_MIN": 1000,
    "OUT_STEERING_MID": 1500,
    "OUT_STEERING_MAX": 2000,
    "OUT_STEERING_INVERT": False,
    "OUT_THROTTLE_MIN": 1000,         # full reverse pwm
    "OUT_THROTTLE_MID": 1500,         # neutral / stop pwm
    "OUT_THROTTLE_MAX": 2000,         # full forward pwm
    "OUT_THROTTLE_INVERT": False,
    "OUT_THROTTLE_HAS_REVERSE": True,  # center-neutral with reverse below mid

    # --- two-wheel differential drive ---
    "DIFFERENTIAL_DRIVE": False,
    "OVERRIDE_LEFT_CHANNEL": 1,
    "OVERRIDE_RIGHT_CHANNEL": 3,

    # --- safety / behaviour ---
    # In manual ('user') mode, release the overridden channels so the
    # flight controller uses the physical RC directly. This is the safest
    # default; set False to always pass values through as override.
    "MANUAL_RELEASE": True,
    # When the AI/pilot output is missing, send neutral instead of holding.
    "FAILSAFE_NEUTRAL": True,
    "OVERRIDE_RATE_HZ": 0,            # 0 = send on every drive loop call
}


#
# Process-wide registry of connections, keyed by connection string, so
# the controller, the drivetrain and the web page share one link.
#
_CONNECTIONS = {}
_CONNECTIONS_LOCK = threading.Lock()


def get_existing_mavlink():
    """
    Return an already-created MavlinkConnection (the first one) or None.
    Used by the web configuration page, which should not create a link.
    """
    with _CONNECTIONS_LOCK:
        for conn in _CONNECTIONS.values():
            return conn
    return None


def merge_params(cfg):
    """
    Build the live parameter dict from DEFAULT_PARAMS overlaid with the
    user's cfg.MAVLINK dictionary.
    """
    params = dict(DEFAULT_PARAMS)
    user = getattr(cfg, "MAVLINK", None)
    if isinstance(user, dict):
        for k, v in user.items():
            params[k] = v
    return params


def get_mavlink(cfg=None, params=None):
    """
    Return the shared MavlinkConnection for this process, creating it on
    first use. Subsequent callers (driver, web page) get the same object.

    :param cfg: donkeycar config (used on first creation)
    :param params: optional pre-built params dict (overrides cfg)
    """
    if params is None:
        if cfg is None:
            raise ValueError("get_mavlink requires cfg or params on first call")
        params = merge_params(cfg)

    key = params["MAVLINK_CONNECTION"]
    with _CONNECTIONS_LOCK:
        conn = _CONNECTIONS.get(key)
        if conn is None:
            conn = MavlinkConnection(params)
            _CONNECTIONS[key] = conn
        return conn


class MavlinkConnection:
    """
    Owns the pymavlink link and a background reader thread. Thread-safe
    accessors expose the latest RC channels and arm state, and helpers
    send arm/disarm commands and RC channel overrides.
    """

    def __init__(self, params):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.mavlink = mavutil.mavlink
        self.params = params
        self.autopilot = str(params.get("MAVLINK_AUTOPILOT", "px4")).lower()

        connection_string = params["MAVLINK_CONNECTION"]
        baud = int(params.get("MAVLINK_BAUDRATE", 921600))
        source_system = int(params.get("MAVLINK_SOURCE_SYSTEM", 255))

        logger.info(f"MAVLink: connecting to {connection_string} "
                    f"(baud={baud}, autopilot={self.autopilot})")
        self.master = mavutil.mavlink_connection(
            connection_string, baud=baud, source_system=source_system)

        self.target_system = int(params.get("MAVLINK_TARGET_SYSTEM", 0))
        self.target_component = int(params.get("MAVLINK_TARGET_COMPONENT", 0))

        self._lock = threading.Lock()
        self._rc = {}            # 1-based channel -> pwm value
        self._rc_time = 0.0
        self._armed = False
        self._heartbeat_time = 0.0
        self._auto_armed = False
        self._running = True

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # ----------------------------------------------------------------- read
    def _read_loop(self):
        mavlink = self.mavlink
        while self._running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                logger.warning(f"MAVLink recv error: {e}")
                time.sleep(0.2)
                continue
            if msg is None:
                continue
            try:
                mtype = msg.get_type()
                if mtype == "HEARTBEAT":
                    self._on_heartbeat(msg)
                elif mtype in ("RC_CHANNELS", "RC_CHANNELS_RAW"):
                    self._on_rc(msg)
            except Exception as e:
                logger.debug(f"MAVLink message handling error: {e}")

    def _on_heartbeat(self, msg):
        mavlink = self.mavlink
        # Only track the autopilot's heartbeat, ignore other GCS/companion.
        src_comp = msg.get_srcComponent()
        if src_comp not in (mavlink.MAV_COMP_ID_AUTOPILOT1, 0, 1):
            return
        if self.target_system == 0:
            self.target_system = msg.get_srcSystem()
            self.target_component = msg.get_srcComponent()
            logger.info(f"MAVLink: detected target system={self.target_system} "
                        f"component={self.target_component}")
        armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        with self._lock:
            self._armed = armed
            self._heartbeat_time = time.time()
        if self.params.get("AUTO_ARM") and not self._auto_armed and not armed:
            self._auto_armed = True
            logger.info("MAVLink: AUTO_ARM enabled, arming vehicle")
            self.arm(True, force=bool(self.params.get("ARM_FORCE")))

    def _on_rc(self, msg):
        rc = {}
        for i in range(1, 19):
            field = f"chan{i}_raw"
            if hasattr(msg, field):
                val = getattr(msg, field)
                # 0 and 65535 mean "no signal" for that channel
                rc[i] = val if val not in (0, 65535) else 0
        with self._lock:
            self._rc = rc
            self._rc_time = time.time()

    # --------------------------------------------------------------- status
    def get_rc(self, channel):
        if not channel:
            return 0
        with self._lock:
            return self._rc.get(int(channel), 0)

    def get_all_rc(self):
        with self._lock:
            return dict(self._rc)

    def rc_age(self):
        with self._lock:
            return time.time() - self._rc_time if self._rc_time else 1e9

    def rc_is_fresh(self, max_age=1.0):
        return self.rc_age() < max_age

    def is_armed(self):
        with self._lock:
            return self._armed

    @property
    def connected(self):
        with self._lock:
            return self._heartbeat_time > 0 and \
                (time.time() - self._heartbeat_time) < 3.0

    def status(self):
        """Snapshot dict used by the web UI."""
        return {
            "connected": self.connected,
            "armed": self.is_armed(),
            "target_system": self.target_system,
            "target_component": self.target_component,
            "autopilot": self.autopilot,
            "rc_age": round(self.rc_age(), 2),
            "rc": self.get_all_rc(),
        }

    # ----------------------------------------------------------- arm/disarm
    def arm(self, arm=True, force=False):
        if self.target_system == 0:
            logger.warning("MAVLink: cannot arm, target not yet detected")
            return False
        param2 = 21196 if force else 0
        try:
            self.master.mav.command_long_send(
                self.target_system, self.target_component,
                self.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1 if arm else 0, param2, 0, 0, 0, 0, 0)
            logger.info(f"MAVLink: {'arm' if arm else 'disarm'} command sent "
                        f"(force={force})")
            return True
        except Exception as e:
            logger.error(f"MAVLink: arm/disarm failed: {e}")
            return False

    def disarm(self, force=False):
        return self.arm(False, force=force)

    # -------------------------------------------------------------- override
    def send_rc_override(self, channels):
        """
        Send an RC_CHANNELS_OVERRIDE.

        :param channels: dict {channel(1-based): pwm}. Channels not present
                         are sent as 0, which releases them back to the RC
                         radio on the flight controller.
        """
        target_sys = self.target_system or 1
        target_comp = self.target_component or 1
        vals = []
        for i in range(1, 19):
            vals.append(int(channels.get(i, 0)) if channels else 0)
        try:
            self.master.mav.rc_channels_override_send(
                target_sys, target_comp, *vals)
        except Exception as e:
            logger.warning(f"MAVLink: rc override send failed: {e}")

    def release_override(self):
        """Release all channels back to the physical RC radio."""
        self.send_rc_override({})

    def shutdown(self):
        self._running = False
        try:
            self.release_override()
        except Exception:
            pass
        try:
            self.master.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------
def pwm_to_unit(pwm, lo, mid, hi, deadzone=0, invert=False, has_reverse=True):
    """
    Convert an RC pwm value to a -1..1 (or 0..1) float.

    :param has_reverse: if True, ``mid`` maps to 0, values below map to
                        negative; if False the range is mapped lo->0, hi->1.
    """
    if pwm is None or pwm == 0:
        return 0.0
    if not has_reverse:
        value = (pwm - lo) / float(hi - lo) if hi != lo else 0.0
        value = clamp(value, 0.0, 1.0)
        return -value if invert else value

    if abs(pwm - mid) <= deadzone:
        return 0.0
    if pwm >= mid:
        value = (pwm - mid) / float(hi - mid) if hi != mid else 0.0
    else:
        value = (pwm - mid) / float(mid - lo) if mid != lo else 0.0
    value = clamp(value, -1.0, 1.0)
    return -value if invert else value


def unit_to_pwm(value, lo, mid, hi, invert=False, has_reverse=True):
    """
    Convert a -1..1 (or 0..1) float to an RC pwm value.
    """
    if value is None:
        value = 0.0
    if invert:
        value = -value
    if not has_reverse:
        value = clamp(value, 0.0, 1.0)
        return int(round(lo + value * (hi - lo)))
    value = clamp(value, -1.0, 1.0)
    if value >= 0:
        pwm = mid + value * (hi - mid)
    else:
        pwm = mid + value * (mid - lo)
    return int(round(clamp(pwm, min(lo, hi), max(lo, hi))))


# ---------------------------------------------------------------------------
# Donkeycar parts
# ---------------------------------------------------------------------------
class MavlinkController:
    """
    Donkeycar controller part: reads RC channels from the flight
    controller and outputs steering/throttle/mode/recording.

    Wire it like the other physical controllers::

        V.add(ctr,
              inputs=['user/mode', 'recording'],
              outputs=['user/steering', 'user/throttle', 'user/mode', 'recording'])
    """

    def __init__(self, cfg, debug=False):
        self.conn = get_mavlink(cfg)
        self.cfg = cfg
        self.debug = debug
        self.mode = getattr(cfg, "WEB_INIT_MODE", "user")
        self.recording = False
        self.recording_latch = None
        self.auto_record_on_throttle = getattr(cfg, "AUTO_RECORD_ON_THROTTLE", False)
        self.dead_zone = getattr(cfg, "JOYSTICK_DEADZONE", 0.01)
        self._prev_arm_request = None

    @property
    def p(self):
        return self.conn.params

    def _read_mode(self, default_mode):
        p = self.p
        ch = p.get("RC_MODE_CHANNEL", 0)
        if not ch or not self.conn.rc_is_fresh():
            # No mode channel or no RC signal: keep web/default value.
            return default_mode if default_mode is not None else self.mode
        pwm = self.conn.get_rc(ch)
        if pwm == 0:
            return default_mode if default_mode is not None else self.mode
        if p.get("RC_MODE_3POSITION"):
            if pwm <= p.get("RC_MODE_LOW_PWM", 1300):
                return "user"
            elif pwm >= p.get("RC_MODE_HIGH_PWM", 1700):
                return "local"
            else:
                return "local_angle"
        # 2-position
        threshold = (p.get("RC_MODE_LOW_PWM", 1300) +
                     p.get("RC_MODE_HIGH_PWM", 1700)) / 2.0
        return "local" if pwm >= threshold else "user"

    def _handle_arm_channel(self):
        p = self.p
        ch = p.get("ARM_CHANNEL", 0)
        if not ch or not self.conn.rc_is_fresh():
            return
        pwm = self.conn.get_rc(ch)
        if pwm == 0:
            return
        want_arm = pwm >= p.get("ARM_PWM_THRESHOLD", 1700)
        if want_arm != self._prev_arm_request:
            self._prev_arm_request = want_arm
            self.conn.arm(want_arm, force=bool(p.get("ARM_FORCE")))

    def run(self, mode=None, recording=None):
        p = self.p

        steering = pwm_to_unit(
            self.conn.get_rc(p.get("RC_STEERING_CHANNEL", 1)),
            p.get("RC_STEERING_MIN", 1000), p.get("RC_STEERING_MID", 1500),
            p.get("RC_STEERING_MAX", 2000), deadzone=p.get("RC_DEADZONE", 20),
            invert=p.get("RC_STEERING_INVERT", False), has_reverse=True)

        throttle = pwm_to_unit(
            self.conn.get_rc(p.get("RC_THROTTLE_CHANNEL", 3)),
            p.get("RC_THROTTLE_MIN", 1000), p.get("RC_THROTTLE_MID", 1500),
            p.get("RC_THROTTLE_MAX", 2000), deadzone=p.get("RC_DEADZONE", 20),
            invert=p.get("RC_THROTTLE_INVERT", False),
            has_reverse=p.get("RC_THROTTLE_HAS_REVERSE", True))

        self.mode = self._read_mode(mode)
        self._handle_arm_channel()

        # recording handling (matches other controllers)
        if recording is not None and recording != self.recording:
            self.recording = recording
        if self.recording_latch is not None:
            self.recording = self.recording_latch
            self.recording_latch = None
        if self.auto_record_on_throttle:
            self.recording = abs(throttle) > self.dead_zone and self.mode == "user"

        if self.debug:
            logger.info(f"MAVLink ctr steering={steering:.2f} "
                        f"throttle={throttle:.2f} mode={self.mode} "
                        f"armed={self.conn.is_armed()}")

        return steering, throttle, self.mode, self.recording

    def run_threaded(self, mode=None, recording=None):
        return self.run(mode, recording)

    def update(self):
        # The connection has its own reader thread, nothing to do here.
        pass

    def set_tub(self, tub):
        self.tub = tub

    def print_controls(self):
        print("MAVLink controller active. Mode/arm controlled via RC channels.")

    def shutdown(self):
        pass


class MavlinkDriver:
    """
    Donkeycar drivetrain part: converts final steering/throttle into
    RC_CHANNELS_OVERRIDE pwm values.

    Wire it as the drivetrain::

        V.add(MavlinkDriver(cfg), inputs=['steering', 'throttle', 'user/mode'])
    """

    def __init__(self, cfg, debug=False):
        self.conn = get_mavlink(cfg)
        self.debug = debug
        self._two_wheel = None
        self._last_send = 0.0

    @property
    def p(self):
        return self.conn.params

    def _differential(self, throttle, steering):
        # Replicates donkeycar.parts.actuator.TwoWheelSteeringThrottle
        throttle = clamp(throttle, -1.0, 1.0)
        steering = clamp(steering, -1.0, 1.0)
        left = throttle
        right = throttle
        if steering < 0:
            left *= (1.0 - (-steering))
        elif steering > 0:
            right *= (1.0 - steering)
        return left, right

    def run(self, steering, throttle, mode=None):
        p = self.p

        # Rate limit if configured.
        rate = p.get("OVERRIDE_RATE_HZ", 0)
        if rate and rate > 0:
            now = time.time()
            if (now - self._last_send) < (1.0 / rate):
                return
            self._last_send = now

        # Manual mode: optionally release channels to the physical RC.
        if mode == "user" and p.get("MANUAL_RELEASE", True):
            self.conn.release_override()
            return

        if steering is None:
            steering = 0.0
        if throttle is None:
            throttle = 0.0 if p.get("FAILSAFE_NEUTRAL", True) else 0.0

        channels = {}
        if p.get("DIFFERENTIAL_DRIVE", False):
            left, right = self._differential(throttle, steering)
            left_pwm = unit_to_pwm(
                left, p.get("OUT_THROTTLE_MIN", 1000),
                p.get("OUT_THROTTLE_MID", 1500), p.get("OUT_THROTTLE_MAX", 2000),
                invert=p.get("OUT_THROTTLE_INVERT", False),
                has_reverse=p.get("OUT_THROTTLE_HAS_REVERSE", True))
            right_pwm = unit_to_pwm(
                right, p.get("OUT_THROTTLE_MIN", 1000),
                p.get("OUT_THROTTLE_MID", 1500), p.get("OUT_THROTTLE_MAX", 2000),
                invert=p.get("OUT_THROTTLE_INVERT", False),
                has_reverse=p.get("OUT_THROTTLE_HAS_REVERSE", True))
            channels[int(p.get("OVERRIDE_LEFT_CHANNEL", 1))] = left_pwm
            channels[int(p.get("OVERRIDE_RIGHT_CHANNEL", 3))] = right_pwm
        else:
            steering_pwm = unit_to_pwm(
                steering, p.get("OUT_STEERING_MIN", 1000),
                p.get("OUT_STEERING_MID", 1500), p.get("OUT_STEERING_MAX", 2000),
                invert=p.get("OUT_STEERING_INVERT", False), has_reverse=True)
            throttle_pwm = unit_to_pwm(
                throttle, p.get("OUT_THROTTLE_MIN", 1000),
                p.get("OUT_THROTTLE_MID", 1500), p.get("OUT_THROTTLE_MAX", 2000),
                invert=p.get("OUT_THROTTLE_INVERT", False),
                has_reverse=p.get("OUT_THROTTLE_HAS_REVERSE", True))
            channels[int(p.get("OVERRIDE_STEERING_CHANNEL", 1))] = steering_pwm
            channels[int(p.get("OVERRIDE_THROTTLE_CHANNEL", 3))] = throttle_pwm

        if self.debug:
            logger.info(f"MAVLink driver override {channels} mode={mode}")

        self.conn.send_rc_override(channels)

    def shutdown(self):
        try:
            self.conn.release_override()
        except Exception:
            pass


class MavlinkArmer:
    """
    Tiny part to arm/disarm from a web button. Use it with a run_condition::

        armer = MavlinkArmer(cfg)
        V.add(armer, inputs=[], run_condition="web/arm")
        V.add(Lambda(lambda: armer.toggle()), ...)

    Or call ``arm()`` / ``disarm()`` / ``toggle()`` directly from a handler.
    """

    def __init__(self, cfg):
        self.conn = get_mavlink(cfg)
        self.force = bool(self.conn.params.get("ARM_FORCE"))

    def arm(self):
        return self.conn.arm(True, force=self.force)

    def disarm(self):
        return self.conn.disarm(force=self.force)

    def toggle(self):
        return self.conn.arm(not self.conn.is_armed(), force=self.force)

    def run(self):
        # Toggle when invoked (e.g. via run_condition tied to a web button).
        self.toggle()

    def shutdown(self):
        pass
