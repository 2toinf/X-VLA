"""Device selection and tiny-model CPU versus XPU parity checks.

Run with ``python -m pytest -q tests`` from the repository root.
"""

import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deploy  # noqa: E402
from models import device as device_mod  # noqa: E402
from models.configuration_xvla import XVLAConfig  # noqa: E402
from models.modeling_xvla import XVLA  # noqa: E402

HAS_XPU = device_mod.is_available("xpu")


def _fake_host(xpu=False, cuda=False):
    return mock.patch.multiple(
        device_mod,
        is_available=lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu"),
    )


class ResolveDeviceTest(unittest.TestCase):
    def test_cpu_only_host(self):
        with _fake_host():
            self.assertEqual(device_mod.resolve_device(), torch.device("cpu"))
            self.assertEqual(device_mod.resolve_device("auto"), torch.device("cpu"))

    def test_cuda_host(self):
        with _fake_host(cuda=True):
            self.assertEqual(device_mod.resolve_device(), torch.device("cuda", 0))

    def test_xpu_host_prefers_xpu(self):
        with _fake_host(xpu=True, cuda=True):
            self.assertEqual(device_mod.resolve_device(), torch.device("xpu", 0))

    def test_explicit_device_wins(self):
        with _fake_host(xpu=True):
            self.assertEqual(device_mod.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(device_mod.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(device_mod.resolve_device(torch.device("xpu", 1)), torch.device("xpu", 1))

    def test_bare_accelerator_gets_index_zero(self):
        self.assertEqual(device_mod.resolve_device("xpu"), torch.device("xpu", 0))
        self.assertEqual(device_mod.resolve_device("cuda"), torch.device("cuda", 0))


class DeployDispatchTest(unittest.TestCase):
    def test_device_flag_defaults_to_autodetect(self):
        args = deploy.build_parser().parse_args(["--model_path", "m"])
        self.assertIsNone(args.device)

    def _run_main(self, argv, host):
        fake_model = mock.MagicMock()
        fake_model.to.return_value = fake_model
        with _fake_host(**host), mock.patch.object(sys, "argv", ["deploy.py"] + argv), \
                mock.patch.object(deploy.XVLA, "from_pretrained", return_value=fake_model), \
                mock.patch.object(deploy.XVLAProcessor, "from_pretrained", return_value=None):
            deploy.main()
        return fake_model

    def test_loader_gets_resolved_device(self):
        import tempfile

        for argv, host, expected in [
            ([], {"xpu": True}, torch.device("xpu", 0)),
            ([], {}, torch.device("cpu")),
            (["--device", "cpu"], {"xpu": True}, torch.device("cpu")),
            (["--device", "xpu"], {}, torch.device("xpu", 0)),
        ]:
            with tempfile.TemporaryDirectory() as out:
                model = self._run_main(["--model_path", "m", "--output_dir", out] + argv, host)
            self.assertEqual(model.to.call_args_list[0].args[0], expected)


def _tiny_config():
    text = dict(
        vocab_size=128, d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=128,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
    )
    vision = dict(
        patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_padding=[3, 1, 1, 1],
        patch_prenorm=[False, True, True, True], dim_embed=[16, 32, 32, 32],
        num_heads=[2, 2, 4, 4], num_groups=[2, 2, 4, 4], depths=[1, 1, 1, 1],
        window_size=7, projection_dim=32, drop_path_rate=0.0,
        visual_temporal_embedding={"type": "COSINE", "max_temporal_embeddings": 100},
        image_pos_embed={"type": "learned_abs_2d", "max_pos_embeddings": 50},
        image_feature_source=["spatial_avg_pool", "temporal_avg_pool"],
    )
    florence = dict(
        model_type="florence2", projection_dim=32, vocab_size=128,
        text_config=text, vision_config=vision,
    )
    return XVLAConfig(
        florence_config=florence, hidden_size=32, depth=2, num_heads=4, num_domains=4,
        len_soft_prompts=4, max_len_seq=256, num_actions=6, action_mode="ee6d",
    )


@contextlib.contextmanager
def _cpu_noise():
    """Draw ``randn`` on the CPU and move it, so every device sees the same noise."""
    orig = torch.randn

    def randn(*size, device=None, dtype=None, **kw):
        return orig(*size, generator=torch.Generator().manual_seed(0), **kw).to(device=device, dtype=dtype)

    with mock.patch.object(torch, "randn", randn):
        yield


@unittest.skipUnless(HAS_XPU, "needs an Intel XPU")
class TinyModelParityTest(unittest.TestCase):
    def test_generate_actions_matches_cpu(self):
        torch.manual_seed(0)
        model = XVLA(_tiny_config()).eval()
        # Re-randomise every parameter so zero-initialised layers cannot hide a mismatch.
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        inputs = dict(
            input_ids=torch.randint(0, 128, (1, 8)),
            image_input=torch.randn(1, 2, 3, 224, 224),
            image_mask=torch.tensor([[True, True]]),
            domain_id=torch.tensor([1]),
            proprio=torch.randn(1, 20),
        )

        def run(device):
            m = model.to(device)
            with torch.no_grad(), _cpu_noise():
                return m.generate_actions(**{k: v.to(device) for k, v in inputs.items()}, steps=3).cpu()

        ref = run("cpu")
        out = run(device_mod.resolve_device("xpu"))
        self.assertGreater(ref.std().item(), 1e-3)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
