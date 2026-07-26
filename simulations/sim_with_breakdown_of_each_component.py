import argparse
import csv
import heapq
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

import numpy as np


SUPPORTED_DISTRIBUTIONS = ("exponential", "pareto", "uniform", "lognormal")


def _exponential_variate(mean: float, **_kw) -> float:
    return np.random.exponential(mean)


def _pareto_variate(mean: float, *, shape: float = 3.0, **_kw) -> float:
    if shape <= 1.0:
        raise ValueError("Pareto shape parameter must be > 1 for finite mean")
    return mean * (shape - 1.0) * np.random.pareto(shape)


def _uniform_variate(mean: float, **_kw) -> float:
    return np.random.uniform(0.0, 2.0 * mean)


def _lognormal_variate(mean: float, *, shape: float = 1.0, **_kw) -> float:
    sigma = shape
    mu_norm = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mu_norm, sigma)


_DIST_FUNCS = {
    "exponential": _exponential_variate,
    "pareto": _pareto_variate,
    "uniform": _uniform_variate,
    "lognormal": _lognormal_variate,
}


def make_variate_fn(dist: str = "exponential", shape: float = 3.0) -> Callable:
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
    switch_processing_timestamp: Optional[float] = None
    exit_timestamp: Optional[float] = None
    residual_delay: float = 0.0
    is_miss: bool = False
    switch_delay: Optional[float] = None
    miss_delay: float = 0.0

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
        deterministic_service: bool = False,
    ):
        self.lambda_rate = lambda_rate
        self.mu_switch = mu_switch
        self.mu_controller = mu_controller
        self.t_cs = t_cs
        self.flow_timeout = flow_timeout
        self.max_time = max_time
        self.variate_fn: Callable = variate_fn or _exponential_variate
        self.deterministic_service = deterministic_service

        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.packet_counter = 0

        self.flow_table: Dict[str, FlowRule] = {}
        self.switch_queue: deque = deque()
        self.controller_queue: deque = deque()
        self.switch_busy = False
        self.controller_busy = False
        self.installation_in_progress: Set[str] = set()
        self.waiting_for_install: Dict[str, List[int]] = {}

        self.packet_store: Dict[int, Packet] = {}
        self.packet_delays: List[float] = []
        self.switch_delays: List[float] = []
        self.miss_delays: List[float] = []
        self.residual_delays: List[float] = []
        self.miss_count = 0
        self.total_arrivals = 0

        self.generate_all_arrivals()

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
            current_time += self.variate_fn(1.0 / self.lambda_rate)
            if current_time < self.max_time:
                pid += 1
                self.schedule_event(
                    Event(current_time, EventType.PACKET_ARRIVAL, pid)
                )
        self.packet_counter = pid

    def create_packet(self, packet_id: int, arrival_time: float) -> Packet:
        packet = Packet(packet_id, arrival_time)
        self.packet_store[packet_id] = packet
        return packet

    def install_flow_rule(self, packet: Packet):
        flow_key = packet.flow_key()
        self.flow_table[flow_key] = FlowRule(
            flow_key, self.current_time, self.current_time, self.flow_timeout
        )
        self.schedule_event(
            Event(
                self.current_time + self.flow_timeout,
                EventType.FLOW_RULE_TIMEOUT,
                flow_key=flow_key,
            )
        )

    def start_switch_service(self, packet_id: int):
        if self.switch_queue and self.switch_queue[0] == packet_id:
            self.switch_queue.popleft()
        else:
            try:
                self.switch_queue.remove(packet_id)
            except ValueError:
                pass

        if packet_id not in self.packet_store:
            return

        self.switch_busy = True
        service_time = (
            1.0 / self.mu_switch
            if self.deterministic_service
            else self.variate_fn(1.0 / self.mu_switch)
        )
        self.schedule_event(
            Event(
                self.current_time + service_time,
                EventType.COMPLETE_SWITCH_PROCESS,
                packet_id,
            )
        )

    def start_controller_service(self, packet_id: int):
        if packet_id not in self.packet_store:
            return

        self.controller_busy = True
        service_time = (
            1.0 / self.mu_controller
            if self.deterministic_service
            else self.variate_fn(1.0 / self.mu_controller)
        )
        self.schedule_event(
            Event(
                self.current_time + service_time,
                EventType.COMPLETE_CONTROLLER_PROCESS,
                packet_id,
            )
        )

    def finish_packet(self, packet: Packet):
        """Record component delays after the packet reaches its exit point."""
        if packet.exit_timestamp is None or packet.switch_processing_timestamp is None:
            raise RuntimeError("Cannot finish packet without required timestamps")

        if packet.is_miss:
            packet.miss_delay = (
                packet.exit_timestamp - packet.switch_processing_timestamp
            )
            packet.switch_delay = (
                packet.switch_processing_timestamp
                - packet.arrival_time
                - packet.residual_delay
            )
        else:
            packet.switch_delay = (
                packet.exit_timestamp - packet.arrival_time - packet.residual_delay
            )

        self.packet_delays.append(packet.exit_timestamp - packet.arrival_time)
        self.switch_delays.append(packet.switch_delay)
        self.miss_delays.append(packet.miss_delay)
        self.residual_delays.append(packet.residual_delay)

    def add_residual_delay_to_switch_queue(self, exit_timestamp: float):
        """
        A completed miss contributes residual delay to packets still waiting
        in the switch queue at the instant its flow rule is installed.
        """
        for packet_id in self.switch_queue:
            packet = self.packet_store.get(packet_id)
            if packet is not None:
                packet.residual_delay += exit_timestamp - packet.arrival_time

    def process_packet_arrival(self, packet_id: int):
        self.create_packet(packet_id, self.current_time)
        self.total_arrivals += 1

        if self.switch_busy:
            self.switch_queue.append(packet_id)
        else:
            self.start_switch_service(packet_id)

    def complete_switch_process(self, packet_id: int):
        packet = self.packet_store.get(packet_id)
        if packet is None:
            self.switch_busy = False
            if self.switch_queue:
                self.start_switch_service(self.switch_queue.popleft())
            return

        packet.switch_processing_timestamp = self.current_time
        flow_key = packet.flow_key()

        if self.is_flow_valid_no_update(flow_key):
            self.flow_table[flow_key].last_used = self.current_time
            self.schedule_event(
                Event(
                    self.current_time + self.flow_timeout,
                    EventType.FLOW_RULE_TIMEOUT,
                    flow_key=flow_key,
                )
            )
            packet.exit_timestamp = self.current_time
            self.finish_packet(packet)
        else:
            packet.is_miss = True
            self.miss_count += 1
            if flow_key in self.installation_in_progress:
                self.waiting_for_install.setdefault(flow_key, []).append(packet_id)
            else:
                self.installation_in_progress.add(flow_key)
                self.schedule_event(
                    Event(
                        self.current_time + self.t_cs,
                        EventType.CONTROLLER_ARRIVAL,
                        packet_id,
                    )
                )

        self.switch_busy = False
        if self.switch_queue:
            self.start_switch_service(self.switch_queue.popleft())

    def process_controller_arrival(self, packet_id: int):
        if self.controller_busy:
            self.controller_queue.append(packet_id)
        else:
            self.start_controller_service(packet_id)

    def complete_controller_process(self, packet_id: int):
        if packet_id in self.packet_store:
            self.schedule_event(
                Event(
                    self.current_time + self.t_cs,
                    EventType.SWITCH_ARRIVAL_FROM_CONTROLLER,
                    packet_id,
                )
            )

        self.controller_busy = False
        if self.controller_queue:
            self.start_controller_service(self.controller_queue.popleft())

    def process_switch_arrival_from_controller(self, packet_id: int):
        packet = self.packet_store.get(packet_id)
        if packet is None:
            return

        flow_key = packet.flow_key()
        self.install_flow_rule(packet)
        self.installation_in_progress.discard(flow_key)

        # The installation completion timestamp is the miss packet's exit time
        # and the timestamp used for residual-delay updates.
        packet.exit_timestamp = self.current_time
        self.add_residual_delay_to_switch_queue(packet.exit_timestamp)
        self.finish_packet(packet)

        for waiting_packet_id in self.waiting_for_install.pop(flow_key, []):
            waiting_packet = self.packet_store.get(waiting_packet_id)
            if waiting_packet is not None:
                waiting_packet.exit_timestamp = self.current_time
                self.finish_packet(waiting_packet)

    def process_flow_timeout(self, flow_key: str):
        if flow_key in self.flow_table:
            rule = self.flow_table[flow_key]
            if rule.is_expired(self.current_time):
                del self.flow_table[flow_key]

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

    def get_statistics(self) -> Dict:
        if not self.packet_delays:
            return {"error": "No packets completed"}

        return {
            "total_delay": float(np.mean(self.packet_delays)),
            "switch_delay": float(np.mean(self.switch_delays)),
            "miss_delay": float(np.mean(self.miss_delays)),
            "residual_delay": float(np.mean(self.residual_delays)),
            "miss_count": self.miss_count,
            "total_arrivals": self.total_arrivals,
        }

    def write_packet_delays_csv(self, output_path: str):
        """Write one row for every packet completed before the time horizon."""
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "packet_id",
                    "is_miss",
                    "arrival_timestamp",
                    "switch_processing_timestamp",
                    "exit_timestamp",
                    "switch_delay",
                    "miss_delay",
                    "residual_delay",
                    "total_delay",
                    "component_delay_sum",
                    "total_delay_minus_component_sum",
                ],
            )
            writer.writeheader()
            for packet in sorted(self.packet_store.values(), key=lambda item: item.id):
                if packet.exit_timestamp is None or packet.switch_delay is None:
                    continue
                total_delay = packet.exit_timestamp - packet.arrival_time
                component_delay_sum = (
                    packet.switch_delay
                    + packet.miss_delay
                    + packet.residual_delay
                )
                writer.writerow(
                    {
                        "packet_id": packet.id,
                        "is_miss": packet.is_miss,
                        "arrival_timestamp": packet.arrival_time,
                        "switch_processing_timestamp": (
                            packet.switch_processing_timestamp
                        ),
                        "exit_timestamp": packet.exit_timestamp,
                        "switch_delay": packet.switch_delay,
                        "miss_delay": packet.miss_delay,
                        "residual_delay": packet.residual_delay,
                        "total_delay": total_delay,
                        "component_delay_sum": component_delay_sum,
                        "total_delay_minus_component_sum": (
                            total_delay - component_delay_sum
                        ),
                    }
                )


def analytical_mean_delay(lambda_rate, mu_switch, mu_controller, tau, theta):
    if lambda_rate >= mu_switch:
        return np.nan
    p_miss = np.exp(-lambda_rate * theta)
    lambda_m = lambda_rate * p_miss
    if lambda_m >= mu_controller:
        return np.nan

    alpha = mu_controller - lambda_m
    switch_delay = 1.0 / (mu_switch - lambda_rate)
    mean_miss_time = 2.0 * tau + 1.0 / alpha
    second_moment_miss_time = (
        4.0 * tau**2 + 4.0 * tau / alpha + 2.0 / alpha**2
    )
    if lambda_m * mean_miss_time >= 1.0:
        return np.nan
    return (
        switch_delay
        + p_miss * mean_miss_time
        + lambda_m * second_moment_miss_time / 2.0
    )


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
    output_csv: Optional[str] = None,
):
    if seed is not None:
        np.random.seed(seed)

    sim = SDNSimulation(
        lambda_rate=lambda_rate,
        mu_switch=mu_switch,
        mu_controller=mu_controller,
        t_cs=tau,
        flow_timeout=timeout,
        max_time=sim_time,
        variate_fn=make_variate_fn(dist=dist, shape=dist_shape),
        deterministic_service=(dist == "pareto"),
    )
    sim.run_simulation()
    if output_csv is not None:
        sim.write_packet_delays_csv(output_csv)

    return {
        "stats": sim.get_statistics(),
        "analytical_mean_delay": analytical_mean_delay(
            lambda_rate, mu_switch, mu_controller, tau, timeout
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SDN simulation with per-component delay breakdown."
    )
    parser.add_argument("--lambda-rate", type=float, default=1.0)
    parser.add_argument("--mu-switch", type=float, default=3.0)
    parser.add_argument("--mu-controller", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--sim-time", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dist",
        default="exponential",
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Distribution for inter-arrival and service times.",
    )
    parser.add_argument(
        "--dist-shape",
        type=float,
        default=3.0,
        help="Pareto α (>1) or log-normal σ.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for a CSV containing one row per completed packet.",
    )
    args = parser.parse_args()

    result = run_single_configuration(
        lambda_rate=args.lambda_rate,
        mu_switch=args.mu_switch,
        mu_controller=args.mu_controller,
        tau=args.tau,
        timeout=args.timeout,
        sim_time=args.sim_time,
        seed=args.seed,
        dist=args.dist,
        dist_shape=args.dist_shape,
        output_csv=args.output_csv,
    )
    stats = result["stats"]

    print("=== SDN Simulation (delay component breakdown) ===")
    print(f"seed={args.seed}")
    print(
        f"λ={args.lambda_rate}, μ_s={args.mu_switch}, "
        f"μ_c={args.mu_controller}, τ={args.tau}, θ={args.timeout}"
    )
    print(f"distribution={args.dist}")
    print(f"simulation_time={args.sim_time}")
    print()
    print(f"Mean total delay          : {stats.get('total_delay', np.nan):.8f}")
    print(f"Mean switch delay         : {stats.get('switch_delay', np.nan):.8f}")
    print(f"Mean miss delay           : {stats.get('miss_delay', np.nan):.8f}")
    print(f"Mean residual delay       : {stats.get('residual_delay', np.nan):.8f}")
    print(f"Analytical mean delay     : {result['analytical_mean_delay']:.8f}")
    print(f"Miss count                : {stats.get('miss_count', 0)}")
    print(f"Total arrivals            : {stats.get('total_arrivals', 0)}")
    if args.output_csv is not None:
        print(f"Per-packet CSV            : {args.output_csv}")


if __name__ == "__main__":
    main()
