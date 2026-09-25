import argparse
import unittest

from graft_kda_gqa import check_layers, parse_layers


class GraftSelectionTests(unittest.TestCase):
    def test_parse_layers_sorts(self):
        self.assertEqual(parse_layers("16, 0,8"), [0, 8, 16])

    def test_parse_layers_rejects_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_layers("0,0")

    def test_kda_donor_must_exist_at_requested_layer(self):
        with self.assertRaisesRegex(ValueError, "no KDA module"):
            check_layers([4, 5], 32, {0, 4, 8}, "kda-into-gqa")

    def test_gqa_donor_can_target_any_layer(self):
        check_layers([5], 32, {0, 4, 8}, "gqa-into-kda")

    def test_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            check_layers([32], 32, {0, 4, 8}, "gqa-into-kda")


if __name__ == "__main__":
    unittest.main()
