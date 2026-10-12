"""
AI-Enabled Micro-channel Battery Thermal Management System (EV)
Data Management Dashboard - Python mini project (Tkinter + SQLite)

Run:  python battery_thermal_dashboard.py

Sections in this file (and the lab experiment each one uses):
  1. Domain model   - abstract classes, polymorphism, exceptions      Exp 4, 5, 6
  2. Analytics      - lambda / map / filter / reduce                  Exp 6
  3. Database       - sqlite3, CRUD, GROUP BY / HAVING / JOIN         Exp 8
  4. Simulator      - threading, queue, exception handling            Exp 5
  5. Dashboard GUI  - Tkinter widgets, Canvas, grid layout            Exp 7

All sensor data is simulated.
"""
import csv
import queue
import random
import sqlite3
import threading
import time
import tkinter as tk
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import reduce
from tkinter import ttk, messagebox, filedialog


# ============================================================================
# DOMAIN MODEL  -  abstract classes, polymorphism, exceptions  (Exp 4, 5, 6)
# ============================================================================

# ---------------------------------------------------------------- constants
WARNING_TEMP = 40.0            # deg C  -> "Warning"
CRITICAL_TEMP = 50.0           # deg C  -> "Critical"
MAX_FLOW = 10.0                # L/min  (emergency flow)
COOLANT_INLET_TEMP = 25.0      # deg C

THERMAL_STATUS_CATEGORIES = {"Normal", "Warning", "Critical"}      # set

# valid sensor ranges: name -> (min, max)                            # dict
VALID_RANGES = {
    "battery_temp": (-20.0, 100.0),
    "coolant_temp": (-20.0, 80.0),
    "voltage": (2.5, 4.4),
    "current": (0.0, 200.0),
    "soc": (0.0, 100.0),
}


# --------------------------------------------------------- custom exceptions
class SensorFaultError(Exception):
    """Sensor returned missing, corrupted or out-of-range data."""


class OverheatingError(Exception):
    """Battery cell temperature crossed the critical limit."""


class CoolingSystemError(Exception):
    """Micro-channel cooling hardware (pump / channel) has failed."""


# --------------------------------------------------------------- helpers
def classify_thermal_state(temp):
    """Return (status, recommended cooling action) for a temperature."""
    if temp >= CRITICAL_TEMP:
        return "Critical", "Maximum Cooling"
    if temp >= WARNING_TEMP:
        return "Warning", "Increase Coolant Flow"
    return "Normal", "No Cooling"


def validate_reading(reading):
    """Raise SensorFaultError if any sensor value is missing or out of range."""
    for name, (low, high) in VALID_RANGES.items():
        value = reading.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SensorFaultError(f"{name} sensor returned no valid data ({value!r})")
        if not low <= value <= high:
            raise SensorFaultError(f"{name} = {value} is outside the valid range {low} to {high}")


# ------------------------------------------------------------- AI predictor
class AIThermalPredictor:
    """Tiny machine-learning model: least-squares linear regression.

    Fits a straight line through the last few temperature readings and
    extrapolates it `horizon` steps ahead, so cooling can react *before* the
    cell actually gets hot.
    """

    def __init__(self, window=5, horizon=3):
        self.window = window
        self.horizon = horizon

    def predict_next(self, history):
        points = list(history)[-self.window:]
        n = len(points)
        if n < 2:
            return points[-1] if points else 0.0
        mean_x = (n - 1) / 2
        mean_y = sum(points) / n
        denom = sum((x - mean_x) ** 2 for x in range(n))
        slope = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(points)) / denom
        return points[-1] + slope * self.horizon


# ------------------------------------- abstract class + polymorphism (Exp 4)
class CoolingStrategy(ABC):
    """Abstract base class: every cooling strategy must give a flow rate."""

    name = "Base strategy"

    @abstractmethod
    def calculate_flow(self, temperature, predicted=None):
        """Required coolant flow in L/min."""

    def describe(self):
        return f"Running cooling strategy: {self.name}"


class StandardCooling(CoolingStrategy):
    """Rule-based (stepwise) cooling - from Experiment 4."""

    name = "Standard Micro-channel"

    def calculate_flow(self, temperature, predicted=None):
        if temperature > 40.0:
            return 5.0
        if temperature > 30.0:
            return 2.5
        return 0.5


class AIEnhancedCooling(CoolingStrategy):
    """Proportional cooling driven by the AI-predicted temperature."""

    name = "AI-Enhanced Micro-channel"

    def calculate_flow(self, temperature, predicted=None):
        effective = max(temperature, predicted if predicted is not None else temperature)
        if effective <= 25.0:
            return 0.2
        return round(min((effective - 25.0) * 0.25, 8.0), 2)


STRATEGIES = {"AI-Enhanced": AIEnhancedCooling, "Standard": StandardCooling}


# ------------------------------------------------------------ hardware model
class BatteryCell:
    """Simulated battery cell with a simple heat-balance model."""

    def __init__(self, cell_id, base_temp=30.0, soc=85.0, fault_rate=0.03):
        self.cell_id = cell_id
        self.temperature = base_temp
        self.coolant_temp = COOLANT_INLET_TEMP
        self.soc = soc
        self.current = 30.0
        self.fault_rate = fault_rate
        self.force_fault = False          # set by the GUI "inject fault" button
        self.history = deque(maxlen=10)   # recent temperatures for the AI model
        self.last_flow = 0.5
        self.active_alerts = set()        # alert episodes currently open (no duplicates)

    def update_load(self, low, high):
        """Move the load current towards a random target in [low, high]."""
        target = random.uniform(low, high)
        self.current += 0.4 * (target - self.current)

    def step(self, flow_rate, dt=3.0):
        """Advance the physics by dt seconds with the given coolant flow."""
        heat = (self.current ** 2) * 7e-5
        removed = flow_rate * 0.012 * max(self.temperature - COOLANT_INLET_TEMP, 0.0)
        ambient = 0.02 * (self.temperature - 25.0)
        self.temperature += dt * (heat - removed - ambient)
        rise = max(self.temperature - COOLANT_INLET_TEMP, 0.0) * flow_rate / 12.0
        self.coolant_temp = COOLANT_INLET_TEMP + min(rise, 15.0)
        self.soc = max(5.0, self.soc - self.current * 0.0005 * dt)

    def read_sensor(self):
        """Read all sensors; raises SensorFaultError on bad data."""
        reading = {
            "battery_temp": round(self.temperature + random.gauss(0, 0.15), 2),
            "coolant_temp": round(self.coolant_temp + random.gauss(0, 0.1), 2),
            "voltage": round(3.0 + 1.2 * self.soc / 100 - self.current * 0.002
                             + random.gauss(0, 0.005), 3),
            "current": round(self.current, 1),
            "soc": round(self.soc, 1),
        }
        if self.force_fault or random.random() < self.fault_rate:
            self.force_fault = False
            broken = random.choice(list(reading))
            reading[broken] = random.choice([None, -999.0, 999.0])
        validate_reading(reading)
        return reading


class MicroChannelCoolingSystem:
    """Micro-channel liquid cooling plate (pump + channels)."""

    def __init__(self):
        self._pump_failed = threading.Event()

    @property
    def pump_failed(self):
        return self._pump_failed.is_set()

    @pump_failed.setter
    def pump_failed(self, value):
        if value:
            self._pump_failed.set()
        else:
            self._pump_failed.clear()

    def apply_cooling(self, cell, flow_rate):
        """Push coolant through the cell's channel. Returns the delivered flow."""
        if self.pump_failed:
            raise CoolingSystemError(f"Coolant pump failure - no flow to Cell-{cell.cell_id}")
        delivered = max(0.0, min(flow_rate, MAX_FLOW))
        cell.step(delivered)
        return delivered


# ============================================================================
# ANALYTICS  -  lambda / map / filter / reduce  (Exp 6)
# ============================================================================

def average(values):
    """Mean of a list using reduce()."""
    if not values:
        return 0.0
    return reduce(lambda a, b: a + b, values) / len(values)


def to_fahrenheit(temps):
    """map() + lambda: convert Celsius readings to Fahrenheit."""
    return list(map(lambda c: round(c * 9 / 5 + 32, 1), temps))


def high_temperatures(temps, limit=WARNING_TEMP):
    """filter() + lambda: readings that need thermal attention."""
    return list(filter(lambda t: t >= limit, temps))


def thermal_scores(temps, current_a):
    """map() + lambda: temperature adjusted for the load current."""
    score = lambda t: round(t + current_a * 0.05, 2)
    return list(map(score, temps))


def safe_range_percent(temps, limit=WARNING_TEMP):
    """Percentage of readings below the warning limit."""
    if not temps:
        return 0.0
    safe = list(filter(lambda t: t < limit, temps))
    return round(100 * len(safe) / len(temps), 1)


def summarize(temps, statuses=()):
    """Return a dict summary of a list of temperature readings."""
    if not temps:
        return None
    high = high_temperatures(temps)
    return {
        "count": len(temps),
        "average": round(average(temps), 2),
        "maximum": max(temps),
        "minimum": min(temps),
        "high_count": len(high),
        "high_percent": round(100 * len(high) / len(temps), 1),
        "safe_percent": safe_range_percent(temps),
        "fahrenheit_sample": to_fahrenheit(temps[:5]),
        "score_sample": thermal_scores(temps[:5], 60.0),
        "categories": set(statuses),
    }


# ============================================================================
# DATABASE  -  SQLite storage layer  (Exp 8)
# ============================================================================

DB_FILE = "battery_thermal.db"

READING_COLS = ("cell_id", "timestamp", "battery_temp", "coolant_temp", "voltage",
                "current", "soc", "predicted_temp", "flow_rate", "status", "action")

SCHEMA = """
CREATE TABLE IF NOT EXISTS battery_cells (
    cell_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id      TEXT NOT NULL,
    position     TEXT,
    capacity_ah  REAL
);
CREATE TABLE IF NOT EXISTS sensor_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id        INTEGER NOT NULL REFERENCES battery_cells(cell_id) ON DELETE CASCADE,
    timestamp      TEXT NOT NULL,
    battery_temp   REAL,
    coolant_temp   REAL,
    voltage        REAL,
    current        REAL,
    soc            REAL,
    predicted_temp REAL,
    flow_rate      REAL,
    status         TEXT,
    action         TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id    INTEGER REFERENCES battery_cells(cell_id) ON DELETE CASCADE,
    timestamp  TEXT NOT NULL,
    level      TEXT NOT NULL,
    message    TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_cell_time ON sensor_readings(cell_id, timestamp);
"""

# Ready-made queries shown in the SQL Explorer (all read-only)
NAMED_QUERIES = {
    "Latest 50 readings (ORDER BY, LIMIT)":
        "SELECT id, cell_id, timestamp, battery_temp, status FROM sensor_readings "
        "ORDER BY id DESC LIMIT 50",
    "Avg / max / min temperature per cell (GROUP BY)":
        "SELECT cell_id, COUNT(*) AS readings, ROUND(AVG(battery_temp), 2) AS avg_temp, "
        "MAX(battery_temp) AS max_temp, MIN(battery_temp) AS min_temp "
        "FROM sensor_readings GROUP BY cell_id",
    "Readings per thermal status (GROUP BY)":
        "SELECT status, COUNT(*) AS readings FROM sensor_readings "
        "GROUP BY status ORDER BY readings DESC",
    "Warning-range readings 40-50 C (BETWEEN)":
        "SELECT id, cell_id, timestamp, battery_temp, flow_rate FROM sensor_readings "
        "WHERE battery_temp BETWEEN 40 AND 49.99 ORDER BY battery_temp DESC",
    "Hottest 10 readings (ORDER BY DESC)":
        "SELECT id, cell_id, timestamp, battery_temp, status FROM sensor_readings "
        "ORDER BY battery_temp DESC LIMIT 10",
    "Cells with more than 1 Critical reading (HAVING)":
        "SELECT cell_id, COUNT(*) AS critical_readings FROM sensor_readings "
        "WHERE status = 'Critical' GROUP BY cell_id HAVING COUNT(*) > 1",
    "Average coolant flow per status (AVG)":
        "SELECT status, ROUND(AVG(flow_rate), 2) AS avg_flow_lpm FROM sensor_readings "
        "GROUP BY status",
    "Low state-of-charge readings < 40 % (WHERE)":
        "SELECT id, cell_id, timestamp, soc, voltage FROM sensor_readings "
        "WHERE soc < 40 ORDER BY soc ASC",
    "Distinct cooling actions (DISTINCT)":
        "SELECT DISTINCT action FROM sensor_readings",
    "Alerts per level (GROUP BY)":
        "SELECT level, COUNT(*) AS alerts FROM alerts GROUP BY level",
    "Cell info + avg temperature (JOIN)":
        "SELECT c.cell_id, c.pack_id, c.position, ROUND(AVG(r.battery_temp), 2) AS avg_temp "
        "FROM battery_cells c JOIN sensor_readings r ON r.cell_id = c.cell_id "
        "GROUP BY c.cell_id",
}


class ThermalDB:
    """Wrapper around one SQLite connection (used from the GUI thread only)."""

    def __init__(self, path=DB_FILE):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if not self.get_cells():
            for position in ("Front-Left", "Front-Right", "Rear-Left", "Rear-Right"):
                self.add_cell("PACK-A1", position, 50.0)

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------ cells
    def add_cell(self, pack_id, position, capacity_ah):
        cur = self.conn.execute(
            "INSERT INTO battery_cells (pack_id, position, capacity_ah) VALUES (?, ?, ?)",
            (pack_id, position, capacity_ah))
        self.conn.commit()
        return cur.lastrowid

    def update_cell(self, cell_id, pack_id, position, capacity_ah):
        self.conn.execute(
            "UPDATE battery_cells SET pack_id = ?, position = ?, capacity_ah = ? "
            "WHERE cell_id = ?", (pack_id, position, capacity_ah, cell_id))
        self.conn.commit()

    def delete_cell(self, cell_id):
        self.conn.execute("DELETE FROM battery_cells WHERE cell_id = ?", (cell_id,))
        self.conn.commit()

    def get_cells(self):
        cur = self.conn.execute(
            "SELECT c.cell_id, c.pack_id, c.position, c.capacity_ah, COUNT(r.id) "
            "FROM battery_cells c LEFT JOIN sensor_readings r ON r.cell_id = c.cell_id "
            "GROUP BY c.cell_id ORDER BY c.cell_id")
        return [tuple(r) for r in cur.fetchall()]

    # --------------------------------------------------------- readings
    def add_readings(self, readings):
        rows = [{c: r.get(c) for c in READING_COLS} for r in readings]
        placeholders = ", ".join(f":{c}" for c in READING_COLS)
        self.conn.executemany(
            f"INSERT INTO sensor_readings ({', '.join(READING_COLS)}) VALUES ({placeholders})",
            rows)
        self.conn.commit()

    def add_reading(self, reading):
        self.add_readings([reading])

    def update_reading(self, reading_id, reading):
        data = {c: reading.get(c) for c in READING_COLS}
        data["id"] = reading_id
        sets = ", ".join(f"{c} = :{c}" for c in READING_COLS if c != "timestamp")
        self.conn.execute(f"UPDATE sensor_readings SET {sets} WHERE id = :id", data)
        self.conn.commit()

    def delete_reading(self, reading_id):
        self.conn.execute("DELETE FROM sensor_readings WHERE id = ?", (reading_id,))
        self.conn.commit()

    def clear_readings(self):
        self.conn.execute("DELETE FROM sensor_readings")
        self.conn.commit()

    def get_readings(self, cell_id=None, status=None, limit=500):
        sql = ("SELECT id, cell_id, timestamp, battery_temp, coolant_temp, voltage, current, "
               "soc, predicted_temp, flow_rate, status, action FROM sensor_readings")
        where, params = [], []
        if cell_id is not None:
            where.append("cell_id = ?")
            params.append(cell_id)
        if status:
            where.append("status = ?")
            params.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [tuple(r) for r in self.conn.execute(sql, params).fetchall()]

    def count_readings(self):
        return self.conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]

    def all_temps(self, limit=5000):
        cur = self.conn.execute(
            "SELECT battery_temp FROM sensor_readings WHERE battery_temp IS NOT NULL "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return [r[0] for r in cur.fetchall()]

    # ----------------------------------------------------------- alerts
    def add_alert(self, alert):
        self.conn.execute(
            "INSERT INTO alerts (cell_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
            (alert.get("cell_id"), alert["timestamp"], alert["level"], alert.get("message", "")))
        self.conn.commit()

    def get_alerts(self, level=None, limit=300):
        sql = ("SELECT a.id, a.timestamp, a.cell_id, a.level, a.message FROM alerts a")
        params = []
        if level:
            sql += " WHERE a.level = ?"
            params.append(level)
        sql += " ORDER BY a.id DESC LIMIT ?"
        params.append(limit)
        return [tuple(r) for r in self.conn.execute(sql, params).fetchall()]

    def count_alerts(self):
        return self.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def clear_alerts(self):
        self.conn.execute("DELETE FROM alerts")
        self.conn.commit()

    # ------------------------------------------------------- statistics
    def avg_temp_by_cell(self):
        cur = self.conn.execute(
            "SELECT cell_id, AVG(battery_temp) FROM sensor_readings "
            "GROUP BY cell_id ORDER BY cell_id")
        return [(r[0], r[1]) for r in cur.fetchall()]

    def status_counts(self):
        cur = self.conn.execute(
            "SELECT status, COUNT(*) FROM sensor_readings GROUP BY status")
        return dict(cur.fetchall())

    # ------------------------------------------------------- SQL explorer
    def run_select(self, sql):
        """Run a read-only SELECT; returns (column names, rows)."""
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed in the SQL Explorer.")
        cur = self.conn.execute(sql)
        columns = [d[0] for d in cur.description]
        return columns, [tuple(r) for r in cur.fetchall()]

    # ------------------------------------------------------------ export
    @staticmethod
    def export_csv(path, columns, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)


# ============================================================================
# SIMULATOR  -  threading, queue, exception handling  (Exp 5)
# ============================================================================

# driving profile -> (min current, max current) in amperes
PROFILES = {
    "Idle / Parked": (5, 20),
    "City Driving": (20, 60),
    "Highway": (50, 90),
    "Fast Charging": (100, 160),
}


class SimulationContext:
    """Shared settings read by every monitor thread."""

    def __init__(self):
        self.profile = "City Driving"
        self.strategy = STRATEGIES["AI-Enhanced"]()
        self.cooling = MicroChannelCoolingSystem()
        self.predictor = AIThermalPredictor()

    def load_range(self):
        return PROFILES[self.profile]

    def set_strategy(self, name):
        self.strategy = STRATEGIES[name]()

    def set_profile(self, name):
        self.profile = name


def _alert(cell, ts, level, message):
    return {"type": "alert", "cell_id": cell.cell_id, "timestamp": ts,
            "level": level, "message": message}


def run_cycle(cell, ctx, last_status, ts=None):
    """One monitoring cycle for one cell.

    Returns (list of events, new status). Used by both the live threads and
    the offline sample-data generator so the behaviour is identical.
    """
    ts = (ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    events = []

    def note(kind, level, message):
        if kind not in cell.active_alerts:      # one alert per episode, not per cycle
            cell.active_alerts.add(kind)
            events.append(_alert(cell, ts, level, message))

    cell.update_load(*ctx.load_range())
    try:
        reading = cell.read_sensor()
    except SensorFaultError as e:
        events.append(_alert(cell, ts, "Fault",
                             f"SENSOR FAULT: {e}. Cycle skipped, keeping last flow."))
        cell.step(cell.last_flow)           # the real battery keeps running
        return events, last_status

    temp = reading["battery_temp"]
    cell.history.append(temp)
    predicted = ctx.predictor.predict_next(cell.history)
    status, action = classify_thermal_state(temp)
    flow = ctx.strategy.calculate_flow(temp, predicted)

    pump_ok = True
    try:
        if status == "Critical":
            raise OverheatingError(f"Cell-{cell.cell_id} at {temp} C (limit {CRITICAL_TEMP} C)")
        flow = ctx.cooling.apply_cooling(cell, flow)
    except OverheatingError as e:
        note("overheat", "Critical", f"OVERHEATING: {e}. Emergency maximum cooling triggered.")
        try:
            flow = ctx.cooling.apply_cooling(cell, MAX_FLOW)
        except CoolingSystemError as ce:
            pump_ok = False
            flow = 0.0
            cell.step(0.0)
            note("pump", "Critical", f"{ce}. Recommend immediate cell isolation!")
    except CoolingSystemError as e:
        pump_ok = False
        flow = 0.0
        cell.step(0.0)
        note("pump", "Fault", f"COOLING SYSTEM ERROR: {e}. Falling back to passive cooling.")

    if status == "Warning" and last_status != "Warning":
        note("warning", "Warning", f"Cell-{cell.cell_id} reached {temp} C - increasing coolant flow.")
    # alert episodes end when the condition clears
    if status != "Critical":
        cell.active_alerts.discard("overheat")
    if pump_ok:
        cell.active_alerts.discard("pump")
    if status == "Normal":
        cell.active_alerts.discard("warning")
    cell.last_flow = flow

    events.append({
        "type": "reading", "cell_id": cell.cell_id, "timestamp": ts,
        "battery_temp": temp, "coolant_temp": reading["coolant_temp"],
        "voltage": reading["voltage"], "current": reading["current"],
        "soc": reading["soc"], "predicted_temp": round(predicted, 2),
        "flow_rate": round(flow, 2), "status": status, "action": action,
    })
    return events, status


class CellMonitor(threading.Thread):
    """Monitoring thread for a single battery cell."""

    def __init__(self, cell, controller):
        super().__init__(name=f"Cell-{cell.cell_id}-Thread", daemon=True)
        self.cell = cell
        self.ctl = controller

    def run(self):
        last_status = "Normal"
        while not self.ctl.stop_event.is_set():
            try:
                events, last_status = run_cycle(self.cell, self.ctl.context, last_status)
                for ev in events:
                    self.ctl.out_queue.put(ev)
            except Exception as e:           # never let a thread die silently
                self.ctl.out_queue.put({
                    "type": "alert", "cell_id": self.cell.cell_id, "level": "Fault",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Unexpected error in monitor thread: {e}"})
            self.ctl.stop_event.wait(self.ctl.interval)


class SimulationController:
    """Starts / stops the monitor threads and exposes the shared settings."""

    def __init__(self, out_queue, interval=1.0):
        self.out_queue = out_queue
        self.interval = interval
        self.context = SimulationContext()
        self.stop_event = threading.Event()
        self.monitors = []

    def is_running(self):
        return any(m.is_alive() for m in self.monitors)

    def start(self, cell_ids):
        if self.is_running():
            return
        self.stop_event.clear()
        self.monitors = [
            CellMonitor(BatteryCell(cid, base_temp=30.0 + 2 * i), self)
            for i, cid in enumerate(cell_ids)
        ]
        for m in self.monitors:
            m.start()

    def stop(self):
        self.stop_event.set()
        for m in self.monitors:
            m.join(timeout=2.0)

    def inject_sensor_fault(self):
        for m in self.monitors:
            m.cell.force_fault = True


def simulate_offline(cell_ids, step_seconds=3):
    """Generate a realistic history without threads (for sample data)."""
    ctx = SimulationContext()
    cells = {cid: BatteryCell(cid, base_temp=30.0 + 2 * i) for i, cid in enumerate(cell_ids)}
    last = {cid: "Normal" for cid in cell_ids}
    phases = [("Idle / Parked", 8), ("City Driving", 14), ("Highway", 14),
              ("Fast Charging", 16), ("City Driving", 8)]
    total = sum(n for _, n in phases)
    start = datetime.now() - timedelta(seconds=total * step_seconds)
    events, tick = [], 0
    for profile, count in phases:
        ctx.set_profile(profile)
        for k in range(count):
            ctx.cooling.pump_failed = (profile == "Fast Charging" and 6 <= k <= 10)
            for cid, cell in cells.items():
                ts = start + timedelta(seconds=tick * step_seconds)
                evs, last[cid] = run_cycle(cell, ctx, last[cid], ts=ts)
                events.extend(evs)
            tick += 1
    ctx.cooling.pump_failed = False
    return events


# ============================================================================
# DASHBOARD GUI  -  Tkinter, Canvas, grid layout  (Exp 7)
# ============================================================================

# ------------------------------------------------------------------ theme
BG = "#f3f5f8"
NAVY = "#14213d"
STATUS_COLOR = {"Normal": "#2e9e5b", "Warning": "#e69f00", "Critical": "#d64545", "Fault": "#7a5cc7"}
STATUS_BG = {"Normal": "#e8f6ee", "Warning": "#fff4d6", "Critical": "#fde2e2", "Fault": "#efe8fb"}
LINE_COLORS = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#a6761d", "#666666"]
SEVERITY = {"Normal": 0, "Warning": 1, "Critical": 2}


# ------------------------------------------------------------ small helpers
def make_tree(parent, columns, height=10):
    """Treeview with scrollbars. columns = [(id, heading, width), ...]"""
    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings",
                        height=height, selectmode="browse")
    for cid, heading, width in columns:
        tree.heading(cid, text=heading)
        tree.column(cid, width=width, minwidth=40, anchor="center")
    vs = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vs.grid(row=0, column=1, sticky="ns")
    hs.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    for name, color in STATUS_BG.items():
        tree.tag_configure(name, background=color)
    return frame, tree


def fill_tree(tree, rows, tag_index=None):
    tree.delete(*tree.get_children())
    for row in rows:
        shown = tuple("" if v is None else v for v in row)
        tag = row[tag_index] if tag_index is not None and row[tag_index] in STATUS_BG else ""
        tree.insert("", "end", values=shown, tags=(tag,))


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------ canvas charts
class LineChart(tk.Canvas):
    """Multi-series line chart drawn on a Tkinter Canvas."""

    def __init__(self, master, **kw):
        super().__init__(master, bg="white", highlightthickness=1,
                         highlightbackground="#d0d7de", **kw)
        self.series = {}
        self.bind("<Configure>", lambda e: self.redraw())

    def set_data(self, series):
        self.series = series
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120 or h < 100:
            return
        left, right, top, bottom = 46, 16, 30, 24
        values = [v for s in self.series.values() for v in s]
        if not values:
            self.create_text(w / 2, h / 2, fill="#8a94a6", font=("Helvetica", 11),
                             text="No live data yet - press  Start Monitoring")
            return
        lo = min(20.0, min(values) - 2)
        hi = max(60.0, max(values) + 2)
        ymap = lambda v: top + (hi - v) / (hi - lo) * (h - top - bottom)
        tick = int(lo // 10 * 10) + 10
        while tick <= hi:
            y = ymap(tick)
            self.create_line(left, y, w - right, y, fill="#eceff3")
            self.create_text(left - 6, y, text=f"{tick}", anchor="e", fill="#6b7280",
                             font=("Helvetica", 8))
            tick += 10
        for limit, color, label in ((WARNING_TEMP, STATUS_COLOR["Warning"], "Warning"),
                                    (CRITICAL_TEMP, STATUS_COLOR["Critical"], "Critical")):
            y = ymap(limit)
            self.create_line(left, y, w - right, y, fill=color, dash=(5, 3))
            self.create_text(w - right - 2, y - 7, text=f"{label} {int(limit)}\u00b0C",
                             anchor="e", fill=color, font=("Helvetica", 8))
        longest = max(len(s) for s in self.series.values())
        x_step = (w - left - right) / max(longest - 1, 1)
        for idx, (name, data) in enumerate(self.series.items()):
            color = LINE_COLORS[idx % len(LINE_COLORS)]
            coords = []
            for i, v in enumerate(data):
                coords += [left + i * x_step, ymap(v)]
            if len(coords) >= 4:
                self.create_line(*coords, fill=color, width=2)
            lx = left + 8 + idx * 78
            self.create_rectangle(lx, 10, lx + 10, 20, fill=color, outline=color)
            self.create_text(lx + 14, 15, text=name, anchor="w", font=("Helvetica", 9))
        self.create_text(8, h / 2, text="\u00b0C", fill="#6b7280", angle=90)


class BarChart(tk.Canvas):
    """Simple vertical bar chart on a Canvas. items = [(label, value, color)]"""

    def __init__(self, master, title="", unit="", **kw):
        super().__init__(master, bg="white", highlightthickness=1,
                         highlightbackground="#d0d7de", **kw)
        self.items, self.title, self.unit = [], title, unit
        self.bind("<Configure>", lambda e: self.redraw())

    def set_data(self, items):
        self.items = items
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 100 or h < 100:
            return
        self.create_text(10, 12, text=self.title, anchor="w", font=("Helvetica", 10, "bold"),
                         fill=NAVY)
        if not self.items:
            self.create_text(w / 2, h / 2, text="No data", fill="#8a94a6")
            return
        left, right, top, bottom = 14, 14, 34, 28
        top_val = max(v for _, v, _ in self.items) * 1.18 or 1
        slot = (w - left - right) / len(self.items)
        bar_w = min(slot * 0.6, 60)
        base = h - bottom
        self.create_line(left, base, w - right, base, fill="#c4cad3")
        for i, (label, value, color) in enumerate(self.items):
            cx = left + slot * (i + 0.5)
            bar_h = (value / top_val) * (base - top)
            self.create_rectangle(cx - bar_w / 2, base - bar_h, cx + bar_w / 2, base,
                                  fill=color, outline=color)
            self.create_text(cx, base - bar_h - 8, text=f"{value:.1f}{self.unit}"
                             if isinstance(value, float) else f"{value}{self.unit}",
                             font=("Helvetica", 9, "bold"), fill=NAVY)
            self.create_text(cx, base + 12, text=str(label), font=("Helvetica", 9), fill="#374151")


# ------------------------------------------------------------------- tabs
class BaseTab(ttk.Frame):
    """Every tab shares this interface (polymorphism): refresh()."""

    def __init__(self, notebook, app):
        super().__init__(notebook, padding=10)
        self.app = app

    def refresh(self):
        raise NotImplementedError

    def on_cells_changed(self):
        pass


class OverviewTab(BaseTab):
    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=3)
        self.rowconfigure(3, weight=1)
        self.kpi_vars = {}
        self._controls()
        self._kpis()
        self._live_area()
        self._alert_list()

    # --- control panel
    def _controls(self):
        box = ttk.LabelFrame(self, text=" Simulation control ", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.start_btn = ttk.Button(box, text="\u25b6  Start Monitoring", command=self.app.start_monitoring)
        self.stop_btn = ttk.Button(box, text="\u25a0  Stop", command=self.app.stop_monitoring,
                                   state="disabled")
        self.start_btn.grid(row=0, column=0, padx=(0, 6))
        self.stop_btn.grid(row=0, column=1, padx=(0, 16))

        ttk.Label(box, text="Cooling strategy:").grid(row=0, column=2)
        self.strategy_var = tk.StringVar(value="AI-Enhanced")
        cb = ttk.Combobox(box, textvariable=self.strategy_var, state="readonly", width=12,
                          values=list(STRATEGIES))
        cb.grid(row=0, column=3, padx=(4, 14))
        cb.bind("<<ComboboxSelected>>", lambda e: self.app.sim.context.set_strategy(self.strategy_var.get()))

        ttk.Label(box, text="Driving profile:").grid(row=0, column=4)
        self.profile_var = tk.StringVar(value="City Driving")
        cb2 = ttk.Combobox(box, textvariable=self.profile_var, state="readonly", width=14,
                           values=list(PROFILES))
        cb2.grid(row=0, column=5, padx=(4, 14))
        cb2.bind("<<ComboboxSelected>>", lambda e: self.app.sim.context.set_profile(self.profile_var.get()))

        ttk.Button(box, text="\u26a1 Inject sensor fault", command=self.app.inject_fault
                   ).grid(row=0, column=6, padx=(0, 10))
        self.pump_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Pump failure", variable=self.pump_var,
                        command=lambda: setattr(self.app.sim.context.cooling, "pump_failed",
                                                self.pump_var.get())).grid(row=0, column=7)

    # --- KPI cards
    def _kpis(self):
        row = ttk.Frame(self)
        row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        titles = ["Pack status", "Average temp", "Hottest cell", "Average SOC",
                  "Total coolant flow", "Session alerts", "Readings stored"]
        for i, title in enumerate(titles):
            row.columnconfigure(i, weight=1, uniform="kpi")
            card = tk.Frame(row, bg="white", highlightthickness=1, highlightbackground="#d0d7de")
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tk.Label(card, text=title, bg="white", fg="#6b7280", font=("Helvetica", 9)
                     ).pack(anchor="w", padx=10, pady=(8, 0))
            var = tk.StringVar(value="-")
            lbl = tk.Label(card, textvariable=var, bg="white", fg=NAVY, font=("Helvetica", 16, "bold"))
            lbl.pack(anchor="w", padx=10, pady=(0, 8))
            self.kpi_vars[title] = (var, lbl)

    # --- pack map + live chart
    def _live_area(self):
        area = ttk.Frame(self)
        area.grid(row=2, column=0, sticky="nsew")
        area.columnconfigure(0, weight=2, uniform="a")
        area.columnconfigure(1, weight=3, uniform="a")
        area.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(area, text=" Battery pack map ", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tiles = tk.Canvas(left, bg="white", highlightthickness=0, height=200)
        self.tiles.grid(row=0, column=0, sticky="nsew")
        self.tiles.bind("<Configure>", lambda e: self.draw_tiles())
        right = ttk.LabelFrame(area, text=" Live battery temperature ", padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.chart = LineChart(right, height=200)
        self.chart.grid(row=0, column=0, sticky="nsew")

    def _alert_list(self):
        box = ttk.LabelFrame(self, text=" Recent alerts ", padding=6)
        box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        frame, self.alert_tree = make_tree(
            box, [("time", "Time", 140), ("cell", "Cell", 50), ("level", "Level", 80),
                  ("msg", "Message", 700)], height=4)
        frame.grid(row=0, column=0, sticky="nsew")

    def set_running(self, running):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    # --- drawing
    def draw_tiles(self):
        c = self.tiles
        c.delete("all")
        w = c.winfo_width()
        if w < 60:
            return
        cells = self.app.db.get_cells()
        if not cells:
            c.create_text(w / 2, 60, text="No battery cells - add some in the Cells tab", fill="#8a94a6")
            return
        tile_w, tile_h, gap = 150, 96, 10
        cols = max(1, int((w + gap) // (tile_w + gap)))
        for i, (cid, pack, pos, _cap, _n) in enumerate(cells):
            x = gap + (i % cols) * (tile_w + gap)
            y = gap + (i // cols) * (tile_h + gap)
            live = self.app.live.get(cid)
            faulty = time.time() < self.app.fault_until.get(cid, 0)
            if faulty:
                status, color = "SENSOR FAULT", STATUS_COLOR["Fault"]
            elif live:
                status, color = live["status"], STATUS_COLOR[live["status"]]
            else:
                status, color = "idle", "#9aa4b2"
            c.create_rectangle(x, y, x + tile_w, y + tile_h, fill=color, outline="")
            c.create_text(x + 10, y + 14, anchor="w", fill="white", font=("Helvetica", 10, "bold"),
                          text=f"Cell {cid}  \u2022  {pos}")
            if live:
                c.create_text(x + 10, y + 44, anchor="w", fill="white", font=("Helvetica", 20, "bold"),
                              text=f"{live['battery_temp']:.1f}\u00b0C")
                c.create_text(x + 10, y + 72, anchor="w", fill="white", font=("Helvetica", 9),
                              text=f"{status}  |  {live['flow_rate']:.1f} L/min")
            else:
                c.create_text(x + 10, y + 56, anchor="w", fill="white", font=("Helvetica", 11),
                              text="waiting for data")

    def refresh(self):
        app = self.app
        live = list(app.live.values())
        v = self.kpi_vars
        if live:
            temps = [r["battery_temp"] for r in live]
            worst = max((r["status"] for r in live), key=lambda s: SEVERITY[s])
            v["Pack status"][0].set(worst.upper())
            v["Pack status"][1].configure(fg=STATUS_COLOR[worst])
            v["Average temp"][0].set(f"{average(temps):.1f} \u00b0C")
            hot = max(live, key=lambda r: r["battery_temp"])
            v["Hottest cell"][0].set(f"Cell {hot['cell_id']}  {hot['battery_temp']:.1f}\u00b0")
            v["Average SOC"][0].set(f"{average([r['soc'] for r in live]):.1f} %")
            v["Total coolant flow"][0].set(f"{sum(r['flow_rate'] for r in live):.1f} L/min")
        else:
            for k in ("Pack status", "Average temp", "Hottest cell", "Average SOC", "Total coolant flow"):
                v[k][0].set("-")
            v["Pack status"][1].configure(fg=NAVY)
        v["Session alerts"][0].set(str(app.session_alerts))
        v["Readings stored"][0].set(f"{app.db.count_readings():,}")
        self.draw_tiles()
        self.chart.set_data({f"Cell {cid}": list(h) for cid, h in sorted(app.history.items()) if h})
        rows = [(a["timestamp"], a["cell_id"], a["level"], a["message"]) for a in app.recent_alerts]
        fill_tree(self.alert_tree, rows, tag_index=2)


class DataTab(BaseTab):
    COLS = [("id", "ID", 50), ("cell", "Cell", 50), ("time", "Timestamp", 145),
            ("temp", "Batt temp \u00b0C", 95), ("cool", "Coolant \u00b0C", 85),
            ("volt", "Voltage V", 80), ("cur", "Current A", 80), ("soc", "SOC %", 65),
            ("pred", "Predicted \u00b0C", 95), ("flow", "Flow L/min", 85),
            ("status", "Status", 80), ("action", "Cooling action", 150)]
    FORM = [("cell", "Cell"), ("battery_temp", "Battery temp (\u00b0C)"),
            ("coolant_temp", "Coolant temp (\u00b0C)"), ("voltage", "Voltage (V)"),
            ("current", "Current (A)"), ("soc", "SOC (%)")]

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._filters()
        frame, self.tree = make_tree(self, self.COLS, height=14)
        frame.grid(row=1, column=0, sticky="nsew", pady=8)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self._form()

    def _filters(self):
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        self.f_cell = tk.StringVar(value="All")
        self.f_status = tk.StringVar(value="All")
        self.f_limit = tk.StringVar(value="200")
        ttk.Label(bar, text="Cell:").pack(side="left")
        self.cell_cb = ttk.Combobox(bar, textvariable=self.f_cell, width=6, state="readonly")
        self.cell_cb.pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Status:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.f_status, width=9, state="readonly",
                     values=["All"] + sorted(THERMAL_STATUS_CATEGORIES)).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Show last:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.f_limit, width=6, state="readonly",
                     values=["50", "200", "500", "1000"]).pack(side="left", padx=(4, 12))
        ttk.Button(bar, text="Apply filter", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(bar, text="Load sample data", command=self.app.load_sample_data).pack(side="right", padx=4)
        ttk.Button(bar, text="Clear all readings", command=self.clear_all).pack(side="right", padx=4)

    def _form(self):
        box = ttk.LabelFrame(self, text=" Add / edit a sensor reading ", padding=8)
        box.grid(row=2, column=0, sticky="ew")
        self.vars = {}
        for i, (key, label) in enumerate(self.FORM):
            ttk.Label(box, text=label).grid(row=0, column=i, sticky="w", padx=(0 if i == 0 else 8, 0))
            var = tk.StringVar()
            self.vars[key] = var
            if key == "cell":
                self.form_cell = ttk.Combobox(box, textvariable=var, width=8, state="readonly")
                self.form_cell.grid(row=1, column=i, sticky="w", padx=(0 if i == 0 else 8, 0))
            else:
                ttk.Entry(box, textvariable=var, width=16).grid(
                    row=1, column=i, sticky="w", padx=(8, 0))
        btns = ttk.Frame(box)
        btns.grid(row=2, column=0, columnspan=len(self.FORM), sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Add reading", command=self.add).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Update selected", command=self.update).pack(side="left", padx=6)
        ttk.Button(btns, text="Delete selected", command=self.delete).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear form", command=self.clear_form).pack(side="left", padx=6)
        ttk.Label(btns, text="  (Status, cooling action and flow are calculated automatically.)",
                  foreground="#6b7280").pack(side="left")

    # --- data
    def on_cells_changed(self):
        ids = [str(c[0]) for c in self.app.db.get_cells()]
        self.cell_cb.configure(values=["All"] + ids)
        self.form_cell.configure(values=ids)
        if self.f_cell.get() not in ["All"] + ids:
            self.f_cell.set("All")

    def refresh(self):
        cell = None if self.f_cell.get() == "All" else int(self.f_cell.get())
        status = None if self.f_status.get() == "All" else self.f_status.get()
        rows = self.app.db.get_readings(cell, status, int(self.f_limit.get()))
        fill_tree(self.tree, rows, tag_index=10)

    def on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        v = self.tree.item(sel[0])["values"]
        for key, idx in (("cell", 1), ("battery_temp", 3), ("coolant_temp", 4),
                         ("voltage", 5), ("current", 6), ("soc", 7)):
            self.vars[key].set(v[idx])

    def clear_form(self):
        for var in self.vars.values():
            var.set("")
        self.tree.selection_remove(*self.tree.selection())

    def read_form(self):
        try:
            reading = {"cell_id": int(self.vars["cell"].get())}
            for key in ("battery_temp", "coolant_temp", "voltage", "current", "soc"):
                reading[key] = float(self.vars[key].get())
        except ValueError:
            raise SensorFaultError("Select a cell and enter numeric values in every field.")
        validate_reading(reading)                       # raises SensorFaultError
        status, action = classify_thermal_state(reading["battery_temp"])
        reading.update(status=status, action=action, predicted_temp=None,
                       flow_rate=AIEnhancedCooling().calculate_flow(reading["battery_temp"]))
        return reading

    def add(self):
        try:
            reading = self.read_form()
            reading["timestamp"] = now_str()
            self.app.db.add_reading(reading)
        except SensorFaultError as e:
            messagebox.showerror("Invalid sensor data", str(e))
            return
        except sqlite3.Error as e:
            messagebox.showerror("Database error", str(e))
            return
        self.clear_form()
        self.app.refresh_all()

    def update(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Update", "Select a row in the table first.")
            return
        rid = self.tree.item(sel[0])["values"][0]
        try:
            self.app.db.update_reading(rid, self.read_form())
        except SensorFaultError as e:
            messagebox.showerror("Invalid sensor data", str(e))
            return
        except sqlite3.Error as e:
            messagebox.showerror("Database error", str(e))
            return
        self.app.refresh_all()

    def delete(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Delete", "Select a row in the table first.")
            return
        rid = self.tree.item(sel[0])["values"][0]
        if messagebox.askyesno("Delete reading", f"Delete reading #{rid}?"):
            self.app.db.delete_reading(rid)
            self.clear_form()
            self.app.refresh_all()

    def clear_all(self):
        if messagebox.askyesno("Clear readings", "Delete ALL stored sensor readings?"):
            self.app.db.clear_readings()
            self.app.live.clear()
            self.app.history.clear()
            self.app.refresh_all()

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="sensor_readings.csv",
                                            filetypes=[("CSV file", "*.csv")])
        if not path:
            return
        try:
            rows = self.app.db.get_readings(limit=1_000_000)
            ThermalDB.export_csv(path, [c[1] for c in self.COLS], rows)
            messagebox.showinfo("Export", f"Saved {len(rows)} rows to\n{path}")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))


class CellsTab(BaseTab):
    COLS = [("id", "Cell ID", 70), ("pack", "Pack ID", 130), ("pos", "Position", 150),
            ("cap", "Capacity (Ah)", 110), ("n", "Readings stored", 120)]

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame, self.tree = make_tree(self, self.COLS, height=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        box = ttk.LabelFrame(self, text=" Add / edit a battery cell ", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.vars = {k: tk.StringVar() for k in ("pack", "pos", "cap")}
        for i, (key, label) in enumerate((("pack", "Pack ID"), ("pos", "Position"),
                                          ("cap", "Capacity (Ah)"))):
            ttk.Label(box, text=label).grid(row=0, column=i, sticky="w", padx=(0 if i == 0 else 8, 0))
            ttk.Entry(box, textvariable=self.vars[key], width=20).grid(
                row=1, column=i, sticky="w", padx=(0 if i == 0 else 8, 0))
        btns = ttk.Frame(box)
        btns.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Add cell", command=self.add).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Update selected", command=self.update).pack(side="left", padx=6)
        ttk.Button(btns, text="Delete selected", command=self.delete).pack(side="left", padx=6)

    def refresh(self):
        fill_tree(self.tree, self.app.db.get_cells())

    def on_select(self, _e=None):
        sel = self.tree.selection()
        if sel:
            v = self.tree.item(sel[0])["values"]
            self.vars["pack"].set(v[1])
            self.vars["pos"].set(v[2])
            self.vars["cap"].set(v[3])

    def _read(self):
        pack, pos = self.vars["pack"].get().strip(), self.vars["pos"].get().strip()
        try:
            cap = float(self.vars["cap"].get())
        except ValueError:
            raise ValueError("Capacity must be a number.")
        if not pack or cap <= 0:
            raise ValueError("Pack ID is required and capacity must be positive.")
        return pack, pos, cap

    def _guard(self):
        if self.app.sim.is_running():
            messagebox.showinfo("Monitoring active", "Stop monitoring before changing cells.")
            return False
        return True

    def add(self):
        if not self._guard():
            return
        try:
            self.app.db.add_cell(*self._read())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.app.cells_changed()

    def update(self):
        sel = self.tree.selection()
        if not sel or not self._guard():
            if not sel:
                messagebox.showinfo("Update", "Select a cell first.")
            return
        try:
            self.app.db.update_cell(self.tree.item(sel[0])["values"][0], *self._read())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.app.cells_changed()

    def delete(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Delete", "Select a cell first.")
            return
        if not self._guard():
            return
        cid = self.tree.item(sel[0])["values"][0]
        if messagebox.askyesno("Delete cell", f"Delete Cell {cid} and ALL its readings and alerts?"):
            self.app.db.delete_cell(cid)
            self.app.live.pop(cid, None)
            self.app.history.pop(cid, None)
            self.app.cells_changed()


class AnalyticsTab(BaseTab):
    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.columnconfigure((0, 1, 2), weight=1, uniform="col")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)
        box = ttk.LabelFrame(self, text=" Functional analysis (lambda / map / filter / reduce) ", padding=6)
        box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.summary = tk.Text(box, height=10, width=40, wrap="word", font=("Courier", 9),
                               bg="white", relief="flat")
        self.summary.grid(row=0, column=0, sticky="nsew")
        self.avg_chart = BarChart(self, title="Average temperature per cell", unit="\u00b0C", height=190)
        self.avg_chart.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        self.status_chart = BarChart(self, title="Readings per thermal status", height=190)
        self.status_chart.grid(row=0, column=2, sticky="nsew")
        self._explorer()

    def _explorer(self):
        box = ttk.LabelFrame(self, text=" SQL Explorer (SELECT queries on the SQLite database) ", padding=8)
        box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(2, weight=1)
        top = ttk.Frame(box)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Ready-made query:").grid(row=0, column=0, sticky="w")
        self.query_var = tk.StringVar(value=list(NAMED_QUERIES)[1])
        ttk.Combobox(top, textvariable=self.query_var, state="readonly",
                     values=list(NAMED_QUERIES)).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Run", command=self.run_named).grid(row=0, column=2)
        ttk.Label(top, text="Custom SQL:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.sql_var = tk.StringVar(value="SELECT * FROM battery_cells")
        ttk.Entry(top, textvariable=self.sql_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="Run", command=self.run_custom).grid(row=1, column=2, pady=(6, 0))
        self.sql_label = ttk.Label(box, text="", foreground="#6b7280", wraplength=1000)
        self.sql_label.grid(row=1, column=0, sticky="w", pady=4)
        frame, self.result = make_tree(box, [("c", "Result", 100)], height=8)
        frame.grid(row=2, column=0, sticky="nsew")

    def show_result(self, sql, columns, rows):
        self.sql_label.configure(text=sql)
        self.result.delete(*self.result.get_children())
        self.result.configure(columns=columns)
        for c in columns:
            self.result.heading(c, text=c)
            self.result.column(c, width=max(90, 900 // len(columns)), anchor="center")
        for row in rows:
            self.result.insert("", "end", values=tuple("" if v is None else v for v in row))

    def _run(self, sql):
        try:
            self.show_result(sql, *self.app.db.run_select(sql))
        except (sqlite3.Error, ValueError) as e:
            messagebox.showerror("Query error", str(e))

    def run_named(self):
        self._run(NAMED_QUERIES[self.query_var.get()])

    def run_custom(self):
        self._run(self.sql_var.get())

    def refresh(self):
        db = self.app.db
        counts = db.status_counts()
        temps = db.all_temps()
        info = summarize(temps, counts.keys())
        text = self.summary
        text.configure(state="normal")
        text.delete("1.0", "end")
        if info:
            lines = [
                f"Readings analysed : {info['count']}",
                f"Average (reduce)  : {info['average']} C",
                f"Max / Min         : {info['maximum']} / {info['minimum']} C",
                f"High >= {WARNING_TEMP:.0f}C (filter): {info['high_count']} ({info['high_percent']}%)",
                f"In safe range     : {info['safe_percent']}%",
                f"Status set        : {sorted(info['categories'])}",
                f"Newest 5 in F (map): {info['fahrenheit_sample']}",
                f"Scores @60A (map) : {info['score_sample']}",
            ]
            text.insert("1.0", "\n".join(lines))
        else:
            text.insert("1.0", "No readings yet.")
        text.configure(state="disabled")
        self.avg_chart.set_data([(f"Cell {cid}", float(avg), LINE_COLORS[i % len(LINE_COLORS)])
                                 for i, (cid, avg) in enumerate(db.avg_temp_by_cell())])
        self.status_chart.set_data([(s, counts[s], STATUS_COLOR[s])
                                    for s in ("Normal", "Warning", "Critical") if s in counts])
        if not self.result.get_children():
            self.run_named()


class AlertsTab(BaseTab):
    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        self.level = tk.StringVar(value="All")
        ttk.Label(bar, text="Level:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.level, width=10, state="readonly",
                     values=["All", "Warning", "Critical", "Fault"]).pack(side="left", padx=6)
        ttk.Button(bar, text="Apply filter", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="Clear alert log", command=self.clear).pack(side="right")
        frame, self.tree = make_tree(self, [("id", "ID", 50), ("time", "Timestamp", 150),
                                            ("cell", "Cell", 50), ("level", "Level", 80),
                                            ("msg", "Message", 800)], height=16)
        frame.grid(row=1, column=0, sticky="nsew", pady=8)

    def refresh(self):
        level = None if self.level.get() == "All" else self.level.get()
        fill_tree(self.tree, self.app.db.get_alerts(level), tag_index=3)

    def clear(self):
        if messagebox.askyesno("Clear alerts", "Delete the whole alert log?"):
            self.app.db.clear_alerts()
            self.app.recent_alerts.clear()
            self.app.session_alerts = 0
            self.app.refresh_all()


# -------------------------------------------------------------------- app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-Enabled Micro-channel Battery Thermal Management System - EV Dashboard")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.db = ThermalDB()
        self.events = queue.Queue()
        self.sim = SimulationController(self.events)
        self.live = {}
        self.history = defaultdict(lambda: deque(maxlen=60))
        self.recent_alerts = deque(maxlen=40)
        self.session_alerts = 0
        self.fault_until = {}
        self.last_refresh = 0.0

        self._style()
        self._header()
        self._tabs()
        if self.db.count_readings() == 0:
            self.load_sample_data(silent=True)
        self.cells_changed()
        self.after(300, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG)
        s.configure("TLabelframe", background=BG)
        s.configure("TLabelframe.Label", background=BG, foreground=NAVY, font=("Helvetica", 10, "bold"))
        s.configure("TCheckbutton", background=BG)
        s.configure("Treeview", rowheight=24, font=("Helvetica", 9))
        s.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))
        s.configure("TNotebook.Tab", padding=(16, 7), font=("Helvetica", 10, "bold"))
        s.configure("TButton", padding=(10, 4))

    def _header(self):
        bar = tk.Frame(self, bg=NAVY)
        bar.pack(fill="x")
        tk.Label(bar, text="\u26a1  AI-Enabled Micro-channel Battery Thermal Management System",
                 bg=NAVY, fg="white", font=("Helvetica", 15, "bold")).pack(side="left", padx=16, pady=10)
        self.state_label = tk.Label(bar, text="\u25cf IDLE", bg=NAVY, fg="#9aa4b2",
                                    font=("Helvetica", 10, "bold"))
        self.state_label.pack(side="right", padx=16)
        tk.Label(bar, text="EV Data Management Dashboard  |  Tkinter + SQLite", bg=NAVY,
                 fg="#c7d0de", font=("Helvetica", 10)).pack(side="right", padx=10)

    def _tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.overview = OverviewTab(self.nb, self)
        self.data_tab = DataTab(self.nb, self)
        self.cells_tab = CellsTab(self.nb, self)
        self.analytics = AnalyticsTab(self.nb, self)
        self.alerts_tab = AlertsTab(self.nb, self)
        self.tabs = [self.overview, self.data_tab, self.cells_tab, self.analytics, self.alerts_tab]
        for tab, name in zip(self.tabs, ["  Live Overview  ", "  Sensor Data  ", "  Battery Cells  ",
                                         "  Analytics & SQL  ", "  Alerts  "]):
            self.nb.add(tab, text=name)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.current_tab().refresh())

    def current_tab(self):
        return self.nametowidget(self.nb.select())

    # --- refresh helpers
    def refresh_all(self):
        for tab in self.tabs:
            tab.refresh()

    def cells_changed(self):
        for tab in self.tabs:
            tab.on_cells_changed()
        self.refresh_all()

    # --- simulation control
    def start_monitoring(self):
        ids = [c[0] for c in self.db.get_cells()]
        if not ids:
            messagebox.showwarning("No cells", "Add at least one battery cell first.")
            return
        ctx = self.sim.context
        ctx.set_strategy(self.overview.strategy_var.get())
        ctx.set_profile(self.overview.profile_var.get())
        ctx.cooling.pump_failed = self.overview.pump_var.get()
        self.live.clear()
        self.history.clear()
        self.sim.start(ids)
        self.overview.set_running(True)
        self.state_label.configure(text="\u25cf MONITORING", fg="#4ade80")

    def stop_monitoring(self):
        self.sim.stop()
        self.overview.set_running(False)
        self.state_label.configure(text="\u25cf IDLE", fg="#9aa4b2")

    def inject_fault(self):
        if not self.sim.is_running():
            messagebox.showinfo("Not running", "Start monitoring first, then inject a fault.")
            return
        self.sim.inject_sensor_fault()

    def load_sample_data(self, silent=False):
        ids = [c[0] for c in self.db.get_cells()]
        if not ids:
            messagebox.showwarning("No cells", "Add at least one battery cell first.")
            return
        events = simulate_offline(ids)
        self.db.add_readings([e for e in events if e["type"] == "reading"])
        for e in events:
            if e["type"] == "alert":
                self.db.add_alert(e)
        if not silent:
            messagebox.showinfo("Sample data", f"Loaded {len(events)} simulated events.")
        self.refresh_all()

    # --- queue polling (threads -> GUI -> database)
    def poll(self):
        readings, alerts = [], []
        try:
            while True:
                ev = self.events.get_nowait()
                (readings if ev["type"] == "reading" else alerts).append(ev)
        except queue.Empty:
            pass
        if readings or alerts:
            try:
                if readings:
                    self.db.add_readings(readings)
                for a in alerts:
                    self.db.add_alert(a)
            except sqlite3.Error as e:
                self.stop_monitoring()
                messagebox.showerror("Database error", str(e))
            for r in readings:
                self.live[r["cell_id"]] = r
                self.history[r["cell_id"]].append(r["battery_temp"])
            for a in alerts:
                self.recent_alerts.appendleft(a)
                self.session_alerts += 1
                if a["level"] == "Fault" and "SENSOR" in a["message"]:
                    self.fault_until[a["cell_id"]] = time.time() + 2.0
            self.overview.refresh()
            if time.time() - self.last_refresh > 2.0:
                self.last_refresh = time.time()
                tab = self.current_tab()
                if tab is not self.overview:
                    tab.refresh()
        self.after(300, self.poll)

    def on_close(self):
        self.sim.stop()
        self.db.close()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
