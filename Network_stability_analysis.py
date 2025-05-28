import random
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque

class UPATrial: # Класс для генерации новых соединений в UPA графе
    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for _ in range(num_nodes) if node != _]

    def run_trial(self, num_nodes): # Проводит num_nodes испытаний и возвращает множество соседей
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        self._node_numbers.extend(list(new_node_neighbors) + [self._num_nodes] * len(new_node_neighbors))
        self._num_nodes += 1
        return new_node_neighbors

def load_graph(file_path): # Загрузка тестовой модели сети из файла
    graph = dict()
    with open(file_path, 'r') as f:
        for line in f:
            nodes = list(map(int, line.strip().split()))
            graph[nodes[0]] = set(nodes[1:])
    return graph

def er_graph(n, p): # Генерация ER-графа
    graph = {node: set() for node in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def upa_graph(n, m): # Генерация UPA-графа
    # Начинаем с полного графа на m вершинах
    graph = {node: {i for i in range(m)}.difference([node]) for node in range(m)}
    upa_trial = UPATrial(m)

    # Добавляем остальные n-m вершин
    for new_node in range(m, n):
        graph[new_node] = upa_trial.run_trial(m)
        for neighbor in graph[new_node]: graph[neighbor].add(new_node)

    return graph

def bfs_size(graph, start): # Вычисляет размер компоненты связности через BFS
    visited, queue, size = set([start]), deque([start]), 1

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                size += 1
    return size

def largest_cc_size(graph): # Находит размер наибольшей компоненты связности
    if not graph:
        return 0

    visited = set()
    max_size = 0

    for node in graph:
        if node not in visited:
            size = bfs_size(graph, node)
            if size > max_size:
                max_size = size
    return max_size

def compute_resilience(graph, attack_order): # Вычисляет устойчивость графа при заданном порядке атаки
    resilience = [largest_cc_size(graph)]
    working_graph = {i: set(j) for i, j in graph.items()}

    for node in attack_order:
        if node in working_graph:
            # Удаляем все ребра, связанные с атакованным узлом
            for neighbor in working_graph[node]:
                working_graph[neighbor].discard(node)
            del working_graph[node]
        resilience.append(largest_cc_size(working_graph))

    return resilience

def random_order(graph): # Генерирует случайный порядок атаки узлов
    nodes = list(graph.keys())
    random.shuffle(nodes)
    return nodes

def is_resilient(resilience, threshold = 0.2, tolerance = 0.25): # Проверяет, является ли сеть устойчивой при удалении threshold% узлов
    n = len(resilience) - 1  # Общее количество узлов
    attack_size = int(n * threshold)

    for i in range(1, attack_size + 1):
        expected_size = n - i
        actual_size = resilience[i]

        if abs(actual_size - expected_size) > tolerance * expected_size:
            return False
    return True

def targeted_order(graph): # Наивный алгоритм целенаправленной атаки
    graph_copy = {node: set(neighbors) for node, neighbors in graph.items()}
    order = []

    while graph_copy:
        # Находим узел с максимальной степенью
        max_degree = -1
        max_node = None

        for node in graph_copy:
            degree = len(graph_copy[node])
            if degree > max_degree:
                max_degree = degree
                max_node = node

        # Удаляем узел из графа
        for neighbor in graph_copy[max_node]:
            graph_copy[neighbor].discard(max_node)
        del graph_copy[max_node]
        order.append(max_node)

    return order

def fast_targeted_order(graph): # Эффективный алгоритм целенаправленной атаки
    degree_sets = defaultdict(set)
    node_degrees = {}

    # Инициализация
    for node in graph:
        degree = len(graph[node])
        degree_sets[degree].add(node)
        node_degrees[node] = degree

    order = []
    max_degree = max(degree_sets.keys()) if degree_sets else 0

    for degree in range(max_degree, -1, -1):
        while degree_sets.get(degree, set()):
            # Берем произвольный узел из множества
            node = degree_sets[degree].pop()

            # Обновляем степени соседей
            for neighbor in graph[node]:
                if neighbor in node_degrees:
                    neighbor_degree = node_degrees[neighbor]
                    degree_sets[neighbor_degree].remove(neighbor)
                    new_degree = neighbor_degree - 1
                    degree_sets[new_degree].add(neighbor)
                    node_degrees[neighbor] = new_degree

            # Добавляем узел в порядок атаки
            order.append(node)
            del node_degrees[node]

    return order

# Анализ времени выполнения
def analyze_algorithms():
    sizes = range(10, 1000, 10)
    m = 5
    naive_times = []
    fast_times = []

    for n in sizes:
        graph = upa_graph(n, m)

        start = time.time()
        _ = targeted_order(graph)
        naive_times.append(time.time() - start)

        start = time.time()
        _ = fast_targeted_order(graph)
        fast_times.append(time.time() - start)
    return [sizes, native_times, fast_times]
