import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
from enum import Enum
from collections import deque
import argparse


# ---------------------------------------------------------------------------
# Distribution helpers — every function returns a *single* positive variate
# with the requested mean.
# ---------------------------------------------------------------------------

SUPPORTED_DISTRIBUTIONS = ("exponential", "pareto", "uniform", "lognormal")


def _exponential_variate(mean: float, **_kw) -> float:
    """Exponential distribution with the given mean."""
    return np.random.exponential(mean)


def _pareto_variate(mean: float, *, shape: float = 3.0, **_kw) -> float:
    """
    Pareto (Lomax / Type-II) variate with the given mean.

    numpy.random.pareto(a) produces X >= 0 with E[X] = 1/(a-1) for a > 1.
    We rescale so the result has the requested mean:
        Y = mean * (shape - 1) * X   =>  E[Y] = mean
    """
    if shape <= 1.0:
        raise ValueError("Pareto shape parameter must be > 1 for finite mean")
    return mean * (shape - 1.0) * np.random.pareto(shape)


def _uniform_variate(mean: float, **_kw) -> float:
    """Uniform(0, 2*mean) — symmetric around *mean*."""
    return np.random.uniform(0.0, 2.0 * mean)


def _lognormal_variate(mean: float, *, shape: float = 1.0, **_kw) -> float:
    """
    Log-normal variate with the given mean.

    *shape* is σ (the std-dev of the underlying normal).
    We set μ_norm = ln(mean) - σ²/2  so that E[e^N] = mean.
    """
    sigma = shape
    mu_norm = np.log(mean) - 0.5 * sigma ** 2
    return np.random.lognormal(mu_norm, sigma)


_DIST_FUNCS = {
    "exponential": _exponential_variate,
    "pareto": _pareto_variate,
    "uniform": _uniform_variate,
    "lognormal": _lognormal_variate,
}


def make_variate_fn(dist: str = "exponential", shape: float = 3.0) -> Callable:
    """
    Return a callable  ``fn(mean) -> float``  that draws one random variate
    from the chosen distribution, scaled to have the requested mean.
    """
    if dist not in _DIST_FUNCS:
        raise ValueError(
            f"Unknown distribution '{dist}'. Choose from {SUPPORTED_DISTRIBUTIONS}"
        )
    base = _DIST_FUNCS[dist]
    return lambda mean: base(mean, shape=shape)


class EventType(Enum):
    PACKET_ARRIVAL = "packet_arrival"
    COMPLETE_SWITCH_PROCESS = "complete_switch_process"
    CONTROLLER_ARRIVAL = "controller_arrival"
    COMPLETE_CONTROLLER_PROCESS = "complete_controller_process"
    SWITCH_ARRIVAL_FROM_CONTROLLER = "switch_arrival_from_controller"
    FLOW_RULE_TIMEOUT = "flow_rule_timeout"


@dataclass
class Event:
    time: float
    event_type: EventType
    packet_id: Optional[int] = None
    flow_key: Optional[str] = None

    def __lt__(self, other):
        return self.time < other.time


@dataclass
class Packet:
    id: int
    arrival_time: float
    src: str = "h1"
    dst: str = "h2"

    def flow_key(self) -> str:
        return f"{self.src}->{self.dst}"


@dataclass
class FlowRule:
    key: str
    install_time: float
    last_used: float
    timeout: float = 10.0

    def is_expired(self, current_time: float) -> bool:
        return current_time - self.last_used > self.timeout


class SDNSimulation:
    def __init__(
        self,
        lambda_rate: float,
        mu_switch: float,
        mu_controller: float,
        t_cs: float,
        flow_timeout: float = 10.0,
        max_time: float = 100.0,
        variate_fn: Optional[Callable] = None,
    ):
        self.lambda_rate = lambda_rate
        self.mu_switch = mu_switch
        self.mu_controller = mu_controller
        self.t_cs = t_cs
        self.flow_timeout = flow_timeout
        self.max_time = max_time
        # variate_fn(mean) -> positive random sample with that mean
        self.variate_fn: Callable = variate_fn or _exponential_variate

        # Clock and event queue
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.packet_counter = 0

        # Network state
        self.flow_table: Dict[str, FlowRule] = {}
        self.switch_queue: deque = deque()
        self.controller_queue: deque = deque()
        self.switch_busy = False
        self.controller_busy = False

        self.installation_in_progress: Set[str] = set()
        self.waiting_for_install: Dict[str, List[int]] = {}

        # Bookkeeping
        self.packet_store: Dict[int, Packet] = {}

        # Statistics
        self.packet_delays: List[float] = []
        self.miss_count = 0
        self.total_arrivals = 0

        # Pre-generate arrivals
        self.generate_all_arrivals()

    # ---------------------
    # Utility & helpers
    # ---------------------
    def schedule_event(self, event: Event):
        heapq.heappush(self.event_queue, event)

    def is_flow_valid_no_update(self, flow_key: str) -> bool:
        if flow_key not in self.flow_table:
            return False
        rule = self.flow_table[flow_key]
        if rule.is_expired(self.current_time):
            del self.flow_table[flow_key]
            return False
        return True

    def generate_all_arrivals(self):
        current_time = 0.0
        pid = 0
        while current_time < self.max_time:
            ia = self.variate_fn(1.0 / self.lambda_rate)
            current_time += ia
            if current_time < self.max_time:
                pid += 1
                ev = Event(current_time, EventType.PACKET_ARRIVAL, pid)
                self.schedule_event(ev)
        self.packet_counter = pid

    def create_packet(self, packet_id: int, arrival_time: float) -> Packet:
        pkt = Packet(packet_id, arrival_time)
        self.packet_store[packet_id] = pkt
        return pkt

    def install_flow_rule(self, packet: Packet):
        fk = packet.flow_key()
        rule = FlowRule(fk, self.current_time, self.current_time, self.flow_timeout)
        self.flow_table[fk] = rule
        timeout_event = Event(
            self.current_time + self.flow_timeout,
            EventType.FLOW_RULE_TIMEOUT,
            flow_key=fk,
        )
        self.schedule_event(timeout_event)

    # ---------------------
    # Switch service control
    # ---------------------
    def start_switch_service(self, packet_id: int):
        try:
            if self.switch_queue and self.switch_queue[0] == packet_id:
                self.switch_queue.popleft()
            else:
                try:
                    self.switch_queue.remove(packet_id)
                except ValueError:
                    pass
        except Exception:
            pass

        pkt = self.packet_store.get(packet_id)
        if pkt is None:
            return

        self.switch_busy = True
        service_time = self.variate_fn(1.0 / self.mu_switch)
        ev = Event(
            self.current_time + service_time,
            EventType.COMPLETE_SWITCH_PROCESS,
            packet_id,
        )
        self.schedule_event(ev)

    # ---------------------
    # Controller service control
    # ---------------------
    def start_controller_service(self, packet_id: int):
        pkt = self.packet_store.get(packet_id)
        if pkt is None:
            return
        self.controller_busy = True
        proc = self.variate_fn(1.0 / self.mu_controller)
        ev = Event(
            self.current_time + proc,
            EventType.COMPLETE_CONTROLLER_PROCESS,
            packet_id,
        )
        self.schedule_event(ev)

    # ---------------------
    # Event handlers
    # ---------------------
    def process_packet_arrival(self, packet_id: int):
        self.create_packet(packet_id, self.current_time)
        self.total_arrivals += 1

        if self.switch_busy:
            self.switch_queue.append(packet_id)
        else:
            self.start_switch_service(packet_id)

    def complete_switch_process(self, packet_id: int):
        pkt = self.packet_store.get(packet_id)
        if not pkt:
            self.switch_busy = False
            if self.switch_queue:
                next_pid = self.switch_queue.popleft()
                self.start_switch_service(next_pid)
            return

        fk = pkt.flow_key()
        is_hit = self.is_flow_valid_no_update(fk)

        if is_hit:
            self.flow_table[fk].last_used = self.current_time
            timeout_event = Event(
                self.current_time + self.flow_timeout,
                EventType.FLOW_RULE_TIMEOUT,
                flow_key=fk,
            )
            self.schedule_event(timeout_event)

            delay = self.current_time - pkt.arrival_time
            self.packet_delays.append(delay)
        else:
            self.miss_count += 1

            if fk in self.installation_in_progress:
                if fk not in self.waiting_for_install:
                    self.waiting_for_install[fk] = []
                self.waiting_for_install[fk].append(packet_id)
            else:
                self.installation_in_progress.add(fk)
                ev = Event(
                    self.current_time + self.t_cs,
                    EventType.CONTROLLER_ARRIVAL,
                    packet_id,
                )
                self.schedule_event(ev)

        self.switch_busy = False
        if self.switch_queue:
            next_pid = self.switch_queue.popleft()
            self.start_switch_service(next_pid)

    def process_controller_arrival(self, packet_id: int):
        if self.controller_busy:
            self.controller_queue.append(packet_id)
        else:
            self.start_controller_service(packet_id)

    def complete_controller_process(self, packet_id: int):
        pkt = self.packet_store.get(packet_id)
        if pkt:
            switch_arrival_time = self.current_time + self.t_cs
            ev = Event(
                switch_arrival_time,
                EventType.SWITCH_ARRIVAL_FROM_CONTROLLER,
                packet_id,
            )
            self.schedule_event(ev)

        self.controller_busy = False
        if self.controller_queue:
            next_pid = self.controller_queue.popleft()
            self.start_controller_service(next_pid)

    def process_switch_arrival_from_controller(self, packet_id: int):
        pkt = self.packet_store.get(packet_id)
        if not pkt:
            return

        fk = pkt.flow_key()

        self.install_flow_rule(pkt)
        self.installation_in_progress.discard(fk)

        delay = self.current_time - pkt.arrival_time
        self.packet_delays.append(delay)

        if fk in self.waiting_for_install:
            waiting_pids = self.waiting_for_install.pop(fk)
            for wpid in waiting_pids:
                wpkt = self.packet_store.get(wpid)
                if wpkt:
                    delay_w = self.current_time - wpkt.arrival_time
                    self.packet_delays.append(delay_w)

    def process_flow_timeout(self, flow_key: str):
        if flow_key in self.flow_table:
            rule = self.flow_table[flow_key]
            if rule.is_expired(self.current_time):
                del self.flow_table[flow_key]

    # ---------------------
    # Event loop
    # ---------------------
    def run_simulation(self):
        while self.event_queue and self.current_time < self.max_time:
            event = heapq.heappop(self.event_queue)
            if event.time > self.max_time:
                break
            self.current_time = event.time

            if event.event_type == EventType.PACKET_ARRIVAL:
                self.process_packet_arrival(event.packet_id)
            elif event.event_type == EventType.COMPLETE_SWITCH_PROCESS:
                self.complete_switch_process(event.packet_id)
            elif event.event_type == EventType.CONTROLLER_ARRIVAL:
                self.process_controller_arrival(event.packet_id)
            elif event.event_type == EventType.COMPLETE_CONTROLLER_PROCESS:
                self.complete_controller_process(event.packet_id)
            elif event.event_type == EventType.SWITCH_ARRIVAL_FROM_CONTROLLER:
                self.process_switch_arrival_from_controller(event.packet_id)
            elif event.event_type == EventType.FLOW_RULE_TIMEOUT:
                self.process_flow_timeout(event.flow_key)

    # ---------------------
    # Statistics
    # ---------------------
    def get_statistics(self) -> Dict:
        if not self.packet_delays:
            return {"error": "No packets completed"}

        observed_pmiss = (
            self.miss_count / self.total_arrivals if self.total_arrivals > 0 else 0.0
        )

        return {
            "total_delay": float(np.mean(self.packet_delays)),
            "miss_count": self.miss_count,
            "total_arrivals": self.total_arrivals,
        }


# ============================================================================ #
# ANALYTICAL MODEL — Closed-form E[D]                                          #
# ============================================================================ #
def analytical_mean_delay(lambda_rate, mu_switch, mu_controller, tau, theta):
    """
    Compute the analytical mean delay from the formula:

        E[D] = 1/(mu_s - lambda)
             + P_miss * E[T]
             + lambda_m * E[T^2] / 2

    where:
        P_miss  = e^{-lambda * theta}
        lambda_m = lambda * P_miss
        E[T]    = 2*tau + 1/(mu_c - lambda_m)
        E[T^2]  = 4*tau^2 + 4*tau/(mu_c - lambda_m) + 2/(mu_c - lambda_m)^2
    """
    if lambda_rate >= mu_switch:
        return np.nan
    P_miss = np.exp(-lambda_rate * theta)
    lambda_m = lambda_rate * P_miss
    if lambda_m >= mu_controller:
        return np.nan

    alpha = mu_controller - lambda_m

    W_s = 1.0 / (mu_switch - lambda_rate)

    ET = 2.0 * tau + 1.0 / alpha
    ET2 = 4.0 * tau ** 2 + 4.0 * tau / alpha + 2.0 / alpha ** 2

    if lambda_m * ET >= 1.0:
        return np.nan

    ED = W_s + P_miss * ET + lambda_m * ET2 / 2.0
    return ED


def run_single_configuration(
    lambda_rate: float,
    mu_switch: float,
    mu_controller: float,
    tau: float,
    timeout: float,
    sim_time: float,
    seed: int | None = None,
    dist: str = "exponential",
    dist_shape: float = 3.0,
):
    """
    Run the SDN simulation for a single parameter set.

    Parameters
    ----------
    dist : str
        Distribution used for inter-arrival and service times.
        Supported: "exponential", "pareto", "uniform", "lognormal".
    dist_shape : float
        Shape parameter consumed by distributions that need one
        (Pareto α, log-normal σ).  Ignored by exponential/uniform.

    Returns a dict with:
      - "stats": output of SDNSimulation.get_statistics()
                 (keys: total_delay, miss_count, total_arrivals)
      - "analytical_mean_delay": scalar E[D] from analytical_mean_delay()
    """
    if seed is not None:
        np.random.seed(seed)

    variate_fn = make_variate_fn(dist=dist, shape=dist_shape)

    sim = SDNSimulation(
        lambda_rate=lambda_rate,
        mu_switch=mu_switch,
        mu_controller=mu_controller,
        t_cs=tau,
        flow_timeout=timeout,
        max_time=sim_time,
        variate_fn=variate_fn,
    )

    sim.run_simulation()
    stats = sim.get_statistics()

    ana_delay = analytical_mean_delay(
        lambda_rate=lambda_rate,
        mu_switch=mu_switch,
        mu_controller=mu_controller,
        tau=tau,
        theta=timeout,
    )

    # Expose analytical delay alongside raw stats dict
    return {
        "stats": stats,  # contains fields shown in lines 393-412
        "analytical_mean_delay": ana_delay,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SDN idle-timeout simulation for a single parameter set."
    )
    parser.add_argument("--lambda-rate", type=float, default=1.0, help="Packet arrival rate λ")
    parser.add_argument("--mu-switch", type=float, default=3.0, help="Switch service rate μ_s")
    parser.add_argument("--mu-controller", type=float, default=2.0, help="Controller service rate μ_c")
    parser.add_argument("--tau", type=float, default=0.4, help="One-way propagation delay τ")
    parser.add_argument("--timeout", type=float, default=2.0, help="Idle timeout θ")
    parser.add_argument("--sim-time", type=float, default=10000.0, help="Simulation horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dist",
        type=str,
        default="exponential",
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Distribution for inter-arrival and service times "
             "(default: exponential)",
    )
    parser.add_argument(
        "--dist-shape",
        type=float,
        default=3.0,
        help="Shape parameter for the chosen distribution: "
             "Pareto α (must be >1, default 3.0), "
             "log-normal σ (default 3.0). "
             "Ignored by exponential and uniform.",
    )

    args = parser.parse_args()

    lambda_rate = args.lambda_rate
    mu_switch = args.mu_switch
    mu_controller = args.mu_controller
    t_cs = args.tau
    flow_timeout = args.timeout
    sim_time = args.sim_time

    result = run_single_configuration(
        lambda_rate=lambda_rate,
        mu_switch=mu_switch,
        mu_controller=mu_controller,
        tau=t_cs,
        timeout=flow_timeout,
        sim_time=sim_time,
        seed=args.seed,
        dist=args.dist,
        dist_shape=args.dist_shape,
    )

    stats = result["stats"]
    ana_delay = result["analytical_mean_delay"]

    print("=== SDN Simulation (single configuration) ===")
    print(f"seed={args.seed}")
    print(f"λ={lambda_rate}, μ_s={mu_switch}, μ_c={mu_controller}, τ={t_cs}, θ={flow_timeout}")
    dist_info = args.dist
    if args.dist in ("pareto", "lognormal"):
        dist_info += f" (shape={args.dist_shape})"
    print(f"distribution={dist_info}")
    print(f"simulation_time={sim_time}")
    print()
    print(f"Simulated total delay     : {stats.get('total_delay', np.nan):.8f}")
    print(f"Analytical mean delay     : {ana_delay:.8f}")
    print(f"Miss count                : {stats.get('miss_count', 0)}")
    print(f"Total arrivals            : {stats.get('total_arrivals', 0)}")


if __name__ == "__main__":
    main()