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
        self.energy_consumed = 0
        self.waiting_cycles = 0
        self.charging = False

    def _get_valid_start_node(self):
        valid_nodes = [n for n in self.warehouse_graph.nodes if nx.degree(self.warehouse_graph, n) > 1]
        return random.choice(valid_nodes) if valid_nodes else 0  

    def assign_task(self, task):
        if not task or not isinstance(task, tuple) or len(task) != 2:
            return
        
        start, goal = task
        if nx.has_path(self.warehouse_graph, start, goal):
            self.current_task = {"start": start, "goal": goal}

    def move(self, path_algorithm):
        if not self.current_task:
            return

        start, goal = self.current_task["start"], self.current_task["goal"]

        if self.position == goal:
            self.completed_tasks += 1
            self.current_task = None
            return

        if not nx.has_path(self.warehouse_graph, self.position, goal):
            self.reroute()
            return

        new_path = path_algorithm(self.warehouse_graph, self.position, goal)
        if new_path and len(new_path) > 1:
            next_position = new_path[1]
            energy_cost = compute_energy_cost(self.warehouse_graph, start, next_position)

            if energy_cost < 0 or energy_cost > 100:
                energy_cost = max(0.1, min(energy_cost, 5))  

            self.position = next_position
            self.total_distance_traveled += 1
            self.energy_consumed += energy_cost
            self.battery = max(0, self.battery - energy_cost)  
            update_congestion(self.warehouse_graph, next_position)
        else:
            self.waiting_cycles += 1

        if self.battery < 20:
            self.recharge_battery()

    def reroute(self):
        alternative_algorithms = [nx.astar_path, nx.dijkstra_path]
        for algo in alternative_algorithms:
            try:
                new_path = algo(self.warehouse_graph, self.position, self.current_task["goal"])
                if new_path and len(new_path) > 1:
                    self.position = new_path[1]
                    self.reroute_count += 1
                    return
            except nx.NetworkXNoPath:
                continue
        
        self.waiting_cycles += 1

    def recharge_battery(self):
        charging_stations = [node for node in self.warehouse_graph.nodes if self.warehouse_graph.nodes[node].get("type") == "charging_station"]
        
        if not charging_stations:
            return
        
        try:
            nearest_station = min(charging_stations, key=lambda s: nx.shortest_path_length(self.warehouse_graph, self.position, s, weight="weight"))
            self.position = nearest_station
            self.charging = True
            self.battery = min(100, self.battery + 50)  
        except nx.NetworkXNoPath:
            return

class AGV_Fleet:
    def __init__(self, warehouse_graph, num_agvs=3):
        self.warehouse_graph = warehouse_graph
        self.agvs = [AGV(i, warehouse_graph) for i in range(num_agvs)]

    def assign_tasks(self, tasks):
        unique_tasks = list(set(tasks))  
        random.shuffle(unique_tasks)

        for agv in self.agvs:
            if unique_tasks:
                task = unique_tasks.pop(0)
                agv.assign_task(task)

    def move_fleet(self, path_algorithm):
        for agv in self.agvs:
            agv.move(path_algorithm)

    def status_report(self):
        for agv in self.agvs:
            print(f"AGV {agv.id} - Position: {agv.position}, Battery: {agv.battery:.2f}%, Tasks Completed: {agv.completed_tasks}")

if __name__ == "__main__":
    from modules.warehouse import create_warehouse_graph

    warehouse_graph = create_warehouse_graph(num_nodes=500, num_edges=2000)
    agv_fleet = AGV_Fleet(warehouse_graph, num_agvs=3)

    tasks = [(random.randint(0, 499), random.randint(0, 499)) for _ in range(5)]
    agv_fleet.assign_tasks(tasks)

    agv_fleet.move_fleet(nx.dijkstra_path)
    agv_fleet.status_report()
