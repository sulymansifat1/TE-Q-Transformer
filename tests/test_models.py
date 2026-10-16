"""Unit tests for TE-Q-Transformer architecture instantiation and forward pass."""

import unittest
import torch

from models.proposed import TEQTransformer, TEQTransformerConfig, rich_entangler_config


class TestModels(unittest.TestCase):
    """Verifies that TE-Q-Transformer instantiates and performs forward inference."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 512
        self.input_dim = 4
        # Channel 0: Voltage (scaled [0, 1])
        # Channel 1: Current (scaled [0, 1])
        # Channel 2: Temperature (Celsius, e.g. 24.0)
        # Channel 3: Normalized time ([0, 1])
        torch.manual_seed(42)
        self.dummy_x = torch.rand(self.batch_size, self.seq_len, self.input_dim)
        self.dummy_x[:, :, 2] = 24.0  # Physical temperature in Celsius

    def test_te_q_transformer_forward(self):
        """Verifies proposed TE-Q-Transformer forward pass [B, 512, 4] -> [B]."""
        model = TEQTransformer(rich_entangler_config())
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(out).all())

    def test_te_q_transformer_batch_invariance(self):
        """Verifies forward inference across single and multi-batch inputs."""
        model = TEQTransformer(rich_entangler_config())
        model.eval()
        with torch.no_grad():
            single_x = self.dummy_x[:1]
            out_single = model(single_x)
            self.assertEqual(out_single.shape, (1,))

            four_x = torch.cat([self.dummy_x, self.dummy_x], dim=0)
            out_four = model(four_x)
            self.assertEqual(out_four.shape, (4,))

    def test_te_q_transformer_configs(self):
        """Verifies default and rich entangler configuration initialization."""
        cfg_default = TEQTransformerConfig()
        model_def = TEQTransformer(cfg_default)
        self.assertIsInstance(model_def, torch.nn.Module)

        cfg_rich = rich_entangler_config()
        self.assertEqual(cfg_rich.quantum_dim, 4)
        self.assertEqual(cfg_rich.entangler_type, "rich")


if __name__ == "__main__":
    unittest.main()
