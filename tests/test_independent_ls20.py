"""An independently transcribed rule model checks the seven calibrated fixtures."""
import ast
import copy
import json
from pathlib import Path
import unittest

from tools import independent_ls20 as independent
from tools.audit_independent_ls20 import check_engine

FIXTURES = Path(__file__).parent / 'fixtures/ls20_reference'


def fixture(tier):
    return json.loads((FIXTURES / f'tier{tier}.json').read_text())


class IndependentLs20Tests(unittest.TestCase):
    def test_model_has_only_standard_library_imports(self):
        tree = ast.parse(Path(independent.__file__).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {'collections', 'math', 'time'})

    def test_all_seven_fixture_minima_and_real_engine_transitions(self):
        for tier in range(1, 8):
            with self.subTest(tier=tier):
                row = fixture(tier)
                result = independent.bfs(independent.Level(row), seconds=30, limit=1_000_000)
                self.assertTrue(result['complete'], result)
                self.assertEqual(result['minimum_actions'], row['optimal_actions'])
                engine = check_engine(row, offroute=8)
                self.assertEqual(engine['mismatches'], 0)
                self.assertTrue(engine['engine_win_three_lives'])
                self.assertEqual(engine['controlled_life_losses'], 1)

    def test_budget_dominance_matches_plain_breadth_first_search(self):
        # Includes hints, refills, static multi-attribute changes and launchers.
        for tier in (1, 2, 3, 4):
            with self.subTest(tier=tier):
                level = independent.Level(fixture(tier))
                plain = independent.bfs(level, budget_dominance=False, limit=100_000)
                pruned = independent.bfs(level, budget_dominance=True, limit=100_000)
                self.assertTrue(plain['complete'])
                self.assertEqual(plain['minimum_actions'], pruned['minimum_actions'])
                self.assertLessEqual(pruned['discovered_states'], plain['discovered_states'])

    def test_search_limit_never_claims_a_minimum(self):
        result = independent.bfs(independent.Level(fixture(1)), limit=1)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['minimum_actions'])
        self.assertEqual(result['reason'], 'state_limit')

    def test_domain_guard_refuses_unbacked_and_overlapping_launchers(self):
        row = fixture(3)
        broken = copy.deepcopy(row)
        cell = broken['launchers'][0]['cell']
        dx, dy = broken['launchers'][0]['delta']
        broken['walls'].remove([cell[0]-dx, cell[1]-dy])
        with self.assertRaises(ValueError):
            independent.Level(broken)
        broken = copy.deepcopy(row)
        broken['goals'][0]['cell'] = broken['launchers'][0]['cell']
        with self.assertRaises(ValueError):
            independent.Level(broken)

    def test_final_goal_can_win_on_negative_budget(self):
        free = {(1, 1), (2, 1), (3, 1)}
        row = dict(walls=[[x, y] for x in range(12) for y in range(12) if (x, y) not in free],
                   goals=[dict(cell=[3, 1], triple=[0, 0, 0])], refills=[],
                   step_counter=1, step_cost=1, training_context_index=1,
                   start=[1, 1], start_triple=[0, 0, 0], rails=[], cyclers=[], launchers=[])
        level = independent.Level(row)
        state, _ = independent.step(level, level.start_state(), 4)
        state, outcome = independent.step(level, state, 4)
        self.assertEqual((state[6], outcome), (-1, 'won'))
        self.assertEqual(independent.bfs(level)['minimum_actions'], 2)


if __name__ == '__main__':
    unittest.main()
