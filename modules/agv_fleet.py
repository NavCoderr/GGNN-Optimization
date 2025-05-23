# -*- coding: utf-8 -*-
import random
import networkx as nx
from datetime import datetime
from modules.congestion_management import update_congestion
from modules.energy_optimization import compute_energy_cost

class AGV:
    def __init__(self, agv_id, warehouse_graph):
        self.id = agv_id
        self.warehouse_graph = warehouse_graph
        self.position = self._get_valid_start_node()
        self.battery = random.uniform(60, 100)
        self.current_task = None
        self.completed_tasks = 0
        self.total_distance_traveled = 0.0
        self.reroute_count = 0
        self.energy_consumed = 0.0
        self.waiting_cycles = 0
        self.charging = False
        self.move_history = []

    def _get_valid_start_node(self):
        valid = [n for n in self.warehouse_graph.nodes if self.warehouse_graph.degree(n) > 1]
        return random.choice(valid) if valid else 0

    def assign_task(self, task):
        if not task or not isinstance(task, tuple) or len(task) < 2:
            return
        start, goal = task[0], task[1]
        if nx.has_path(self.warehouse_graph, start, goal):
            task_id = task[2] if len(task) >= 3 else None
            self.current_task = {'start': start, 'goal': goal, 'task_id': task_id}

    def move(self, path_algorithm, current_cycle=None, current_workload=None, current_route=None):
        if not self.current_task:
            return True

        goal = self.current_task['goal']
        if self.position == goal:
            self.completed_tasks += 1
            self.current_task = None
            return True

        prev = self.position
        if not nx.has_path(self.warehouse_graph, prev, goal):
            self._record_move(prev, prev, 0, self.battery, self.battery,
                              obstacle=True, rerouted=False,
                              current_cycle=current_cycle,
                              current_workload=current_workload,
                              current_route=current_route)
            self.reroute()
            return False

        path = path_algorithm(self.warehouse_graph, prev, goal)
        if path and len(path) > 1:
            nxt = path[1]
            ebefore = self.battery

            # raw energy cost, only lower-bound
            cost = compute_energy_cost(self.warehouse_graph, prev, nxt)
            cost = max(0.1, cost)

            # update state
            self.position = nxt
            self.total_distance_traveled += 1
            self.energy_consumed += cost
            self.battery = max(0.0, self.battery - cost)
            update_congestion(self.warehouse_graph, prev, nxt)

            # metrics
            edge   = self.warehouse_graph.edges[prev, nxt]
            length = edge.get('length', 1.0)             # meters
            speed  = random.uniform(0.5, 1.5)            # m/s
            duration = length / speed                    # seconds
            power    = cost / duration                   # watts
            cong     = edge.get('congestion', 0.0)
            urg      = edge.get('urgency',    0)

            # record history
            self.move_history.append({
                'agv_id':        self.id,
                'cycle':         current_cycle,
                'timestamp':     datetime.now().isoformat(),
                'task_id':       self.current_task.get('task_id'),
                'from_node':     prev,
                'to_node':       nxt,
                'distance':      edge.get('weight', 1.0),
                'distance_m':    length,
                'duration_s':    duration,
                'speed_m_s':     speed,
                'energy_before': ebefore,
                'energy_used':   cost,
                'energy_after':  self.battery,
                'congestion':    cong,
                'urgency':       urg,
                'power_W':       power,
                'obstacle_detected': False,
                'rerouted':          False,
                'mes_workload':      current_workload,
                'tms_route_len':     len(current_route) if current_route else None
            })

            if self.battery < 20.0:
                self.recharge_battery()
            return True

        # waiting (no move)
        self.waiting_cycles += 1
        self._record_move(prev, prev, 0, self.battery, self.battery,
                          obstacle=False, rerouted=False,
                          current_cycle=current_cycle,
                          current_workload=current_workload,
                          current_route=current_route)
        return False

    def reroute(self):
        prev = self.position
        for algo in (nx.astar_path, nx.dijkstra_path):
            try:
                new_path = algo(self.warehouse_graph, prev, self.current_task['goal'])
                if new_path and len(new_path) > 1:
                    nxt = new_path[1]
                    self.position = nxt
                    self.reroute_count += 1
                    edge = self.warehouse_graph.edges[prev, nxt]
                    self.move_history.append({
                        'agv_id':        self.id,
                        'cycle':         None,
                        'timestamp':     datetime.now().isoformat(),
                        'task_id':       self.current_task.get('task_id'),
                        'from_node':     prev,
                        'to_node':       nxt,
                        'distance':      edge.get('weight',1.0),
                        'distance_m':    edge.get('length',1.0),
                        'duration_s':    0.0,
                        'speed_m_s':     0.0,
                        'energy_before': None,
                        'energy_used':   None,
                        'energy_after':  self.battery,
                        'congestion':    edge.get('congestion',0.0),
                        'urgency':       edge.get('urgency',0),
                        'power_W':       None,
                        'obstacle_detected': True,
                        'rerouted':          True,
                        'mes_workload':      None,
                        'tms_route_len':     None
                    })
                    return
            except nx.NetworkXNoPath:
                continue
        self.waiting_cycles += 1

    def recharge_battery(self):
        stations = [n for n, d in self.warehouse_graph.nodes(data=True)
                    if d.get('type') == 'charging_station']
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

    def _record_move(self, frm, to, dist, ebefore, eafter,
                     obstacle, rerouted, current_cycle=None, current_workload=None, current_route=None):
        self.move_history.append({
            'agv_id':        self.id,
            'cycle':         current_cycle,
            'timestamp':     datetime.now().isoformat(),
            'task_id':       self.current_task.get('task_id') if self.current_task else None,
            'from_node':     frm,
            'to_node':       to,
            'distance':      dist,
            'distance_m':    0.0,
            'duration_s':    0.0,
            'speed_m_s':     0.0,
            'energy_before': ebefore,
            'energy_used':   0,
            'energy_after':  eafter,
            'congestion':    None,
            'urgency':       None,
            'power_W':       0.0,
            'obstacle_detected': obstacle,
            'rerouted':          rerouted,
            'mes_workload':      current_workload,
            'tms_route_len':     len(current_route) if current_route else None
        })

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

    def move_fleet(self, path_algorithm, current_cycle=None, current_workload=None, current_route=None):
        for agv in self.agvs:
            agv.move(path_algorithm,
                     current_cycle=current_cycle,
                     current_workload=current_workload,
                     current_route=current_route)

    def status_report(self):
        for agv in self.agvs:
            print(f"AGV {agv.id} - Pos: {agv.position}, "
                  f"Bat: {agv.battery:.1f}%, Done: {agv.completed_tasks}")
