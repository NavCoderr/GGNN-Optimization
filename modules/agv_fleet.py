import random
import networkx as nx
from modules.congestion_management import update_congestion
from modules.energy_optimization import compute_energy_cost
from modules.tms import get_dynamic_task_updates

class AGV:
    def __init__(self, agv_id, warehouse_graph):
        self.id = agv_id
        self.warehouse_graph = warehouse_graph
        self.position = self._get_valid_start_node()
        self.battery = random.uniform(60, 100)
        self.current_task = None
        self.completed_tasks = 0
        self.total_distance_traveled = 0
        self.reroute_count = 0
        self.energy_consumed = 0.0
        self.waiting_cycles = 0
        self.charging = False

    def _get_valid_start_node(self):
        valid = [n for n in self.warehouse_graph.nodes if nx.degree(self.warehouse_graph, n) > 1]
        return random.choice(valid) if valid else 0

    def assign_task(self, task):
        if not task or not isinstance(task, tuple) or len(task) != 2:
            return
        start, goal = task
        if nx.has_path(self.warehouse_graph, start, goal):
            self.current_task = {'start': start, 'goal': goal}

    def move(self, path_algorithm):
        if not self.current_task:
            return True
        start = self.current_task['start']
        goal  = self.current_task['goal']
        if self.position == goal:
            self.completed_tasks += 1
            self.current_task = None
            return True
        if not nx.has_path(self.warehouse_graph, self.position, goal):
            self.reroute()
            return False
        path = path_algorithm(self.warehouse_graph, self.position, goal)
        if path and len(path) > 1:
            nxt = path[1]
            cost = compute_energy_cost(self.warehouse_graph, self.position, nxt)
            cost = max(0.1, min(cost, 5.0))
            self.position = nxt
            self.total_distance_traveled += 1
            self.energy_consumed += cost
            self.battery = max(0.0, self.battery - cost)
            update_congestion(self.warehouse_graph, nxt)
            if self.battery < 20.0:
                self.recharge_battery()
            return True
        else:
            self.waiting_cycles += 1
            return False

    def reroute(self):
        for algo in [nx.astar_path, nx.dijkstra_path]:
            try:
                new_path = algo(self.warehouse_graph, self.position, self.current_task['goal'])
                if new_path and len(new_path) > 1:
                    self.position = new_path[1]
                    self.reroute_count += 1
                    return
            except nx.NetworkXNoPath:
                continue
        self.waiting_cycles += 1

    def recharge_battery(self):
        stations = [n for n in self.warehouse_graph.nodes if self.warehouse_graph.nodes[n].get('type') == 'charging_station']
        if not stations:
            return
        try:
            nearest = min(
                stations,
                key=lambda s: nx.shortest_path_length(self.warehouse_graph, self.position, s, weight='weight')
            )
            self.position = nearest
            self.charging = True
            self.battery = min(100.0, self.battery + 50.0)
        except nx.NetworkXNoPath:
            pass

class AGV_Fleet:
    def __init__(self, warehouse_graph, num_agvs=3):
        self.warehouse_graph = warehouse_graph
        self.agvs = [AGV(i, warehouse_graph) for i in range(num_agvs)]

    def assign_tasks(self, tasks):
        unique = list(set(tasks))
        random.shuffle(unique)
        for agv in self.agvs:
            if unique:
                agv.assign_task(unique.pop(0))

    def move_fleet(self, path_algorithm):
        for agv in self.agvs:
            agv.move(path_algorithm)

    def status_report(self):
        for agv in self.agvs:
            print(f"AGV {agv.id} - Pos: {agv.position}, Bat: {agv.battery:.1f}%, Done: {agv.completed_tasks}")

if __name__ == "__main__":
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    fleet = AGV_Fleet(G, num_agvs=3)
    tasks = [(random.randint(0,499), random.randint(0,499)) for _ in range(5)]
    fleet.assign_tasks(tasks)
    fleet.move_fleet(nx.dijkstra_path)
    fleet.status_report()
