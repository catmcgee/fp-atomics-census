import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('factorial', Path(__file__).with_name('factorial.py'))
factorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factorial)

class FactorialPlanTests(unittest.TestCase):
    def test_each_setting_can_be_changed_while_others_are_fixed(self):
        cells = factorial.matrix()
        settings = {(c['deterministic'], c['combo_kernels'], c['benchmark_combo_kernel']) for c in cells}
        self.assertEqual(len(cells), 8)
        self.assertEqual(len(settings), 8)
        for setting in settings:
            for bit in range(3):
                neighbour = list(setting)
                neighbour[bit] = not neighbour[bit]
                self.assertIn(tuple(neighbour), settings)

    def test_each_reference_has_explicit_settings_and_distinct_name(self):
        names = set()
        for cell in factorial.matrix():
            command = factorial.record_args(cell)
            settings = json.loads(command[command.index('--inductor-config') + 1])
            self.assertEqual(settings, {k: cell[k] for k in ('combo_kernels', 'benchmark_combo_kernel')})
            names.add(command[command.index('--tag') + 1])
            self.assertIn('--mixed', command)
        self.assertEqual(len(names), 8)

    def test_continuation_requires_known_unique_cells(self):
        self.assertEqual([c['label'] for c in factorial.selected_cells('d1_c1_b0,d1_c1_b1')], ['d1_c1_b0', 'd1_c1_b1'])
        for selection in ('unknown', 'd0_c0_b0,d0_c0_b0', ''):
            with self.assertRaises(ValueError): factorial.selected_cells(selection)

if __name__ == '__main__': unittest.main()
