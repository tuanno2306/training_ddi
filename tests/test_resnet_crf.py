"""Run with: python -m unittest discover -s tests -v (CPU, no downloads)."""

import io
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from resnet_crf import BertResNetCRF, LinearChainCRF, ResNetCRF


LABELS = ["O", "B-drug", "I-drug", "B-brand", "I-brand"]
ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(24, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class CRFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_partition_loss_and_viterbi_match_exhaustive_search(self):
        for constrained in (False, True):
            with self.subTest(constrained=constrained):
                crf = LinearChainCRF(3, ["O", "B-drug", "I-drug"] if constrained else None)
                emissions = torch.randn(2, 3, 3, requires_grad=True)
                mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
                tags = torch.tensor([[1, 2, 0], [0, -100, -100]])
                expected_losses, expected_paths = [], []
                for b, length in enumerate((3, 1)):
                    scores, paths = [], []
                    for path in itertools.product(range(3), repeat=length):
                        if not crf.start_allowed[path[0]]:
                            continue
                        if any(not crf.transition_allowed[a, c] for a, c in zip(path, path[1:])):
                            continue
                        value = crf.start_transitions[path[0]] + crf.end_transitions[path[-1]]
                        value = value + sum(emissions[b, t, label] for t, label in enumerate(path))
                        value = value + sum(crf.transitions[a, c] for a, c in zip(path, path[1:]))
                        paths.append(list(path))
                        scores.append(value)
                    scores = torch.stack(scores)
                    gold_index = paths.index(tags[b, :length].tolist())
                    expected_losses.append(torch.logsumexp(scores, 0) - scores[gold_index])
                    expected_paths.append(paths[scores.argmax().item()])
                loss = crf(emissions, tags, mask)
                torch.testing.assert_close(loss, torch.stack(expected_losses).mean())
                self.assertEqual(crf.decode(emissions, mask), expected_paths)
                loss.backward()
                self.assertTrue(torch.isfinite(emissions.grad).all())
                for parameter in crf.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bio_constraints_reject_invalid_gold_and_decode_valid_paths(self):
        crf = LinearChainCRF(len(LABELS), LABELS)
        emissions = torch.zeros(1, 3, len(LABELS))
        emissions[:, :, 4] = 1000  # Strongly prefer I-brand even at the start.
        mask = torch.ones(1, 3, dtype=torch.bool)
        path = crf.decode(emissions, mask)[0]
        self.assertFalse(LABELS[path[0]].startswith("I-"))
        self.assertTrue(all(crf.transition_allowed[a, b] for a, b in zip(path, path[1:])))
        for tags in ([[2, 2, 0]], [[1, 4, 0]]):
            with self.assertRaisesRegex(ValueError, "BIO"):
                crf(emissions, torch.tensor(tags), mask)

    def test_crf_rejects_empty_or_noncontiguous_masks(self):
        crf = LinearChainCRF(3)
        for mask in ([[0, 0, 0]], [[1, 0, 1]], [[1, 2, 0]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                crf.decode(torch.randn(1, 3, 3), torch.tensor(mask))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def make_head(self):
        return ResNetCRF(8, len(LABELS), channels=12, num_blocks=3,
                         dilations=[1, 2, 4], dropout=0.0, labels=LABELS)

    def test_padding_cannot_change_valid_outputs_or_loss(self):
        head = self.make_head().eval()
        features = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        gold = torch.tensor([[0, 1, 2, 0]])
        loss, emissions = head(features, mask, gold)
        padded = torch.cat([features, torch.full((1, 5, 8), float("nan"))], dim=1)
        padded.requires_grad_()
        padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        padded_gold = torch.tensor([[0, 1, 2, 0, -100, -100, -100, -100, -100]])
        padded_loss, padded_emissions = head(padded, padded_mask, padded_gold)
        torch.testing.assert_close(emissions, padded_emissions[:, :4])
        torch.testing.assert_close(loss, padded_loss)
        self.assertEqual(head.decode(features, mask), head.decode(padded, padded_mask))
        padded_loss.backward()
        self.assertEqual(padded.grad[:, 4:].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(padded.grad).all())

    def test_tag_mask_packs_special_tokens_and_restores_alignment(self):
        head = self.make_head()
        emissions = torch.randn(2, 6, len(LABELS), requires_grad=True)
        attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        selected = torch.tensor([[0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.bool)
        labels = torch.tensor([[-100, 1, -100, 2, 0, -100], [-100, 3, 4, -100, -100, -100]])
        loss = head.loss_from_emissions(emissions, labels, attention, selected)
        individual = []
        for b in range(2):
            values = emissions[b, selected[b]].unsqueeze(0)
            tags = labels[b, selected[b]].unsqueeze(0)
            individual.append(head.crf(values, tags, torch.ones_like(tags, dtype=torch.bool)))
        torch.testing.assert_close(loss, torch.stack(individual).mean())
        paths = head.decode_emissions(emissions, attention, selected)
        self.assertEqual([len(path) for path in paths], [6, 4])
        for b, path in enumerate(paths):
            self.assertTrue(all((tag != -100) == bool(selected[b, t]) for t, tag in enumerate(path)))
        loss.backward()
        self.assertEqual(emissions.grad[~selected].count_nonzero().item(), 0)

    def test_encoder_residual_and_crf_all_receive_gradients(self):
        model = BertResNetCRF(None, len(LABELS), encoder=TinyEncoder(), channels=12,
                             num_blocks=2, labels=LABELS, dropout=0.0)
        ids = torch.tensor([[2, 4, 5, 6, 3], [2, 7, 3, 0, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        selected = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0]])
        gold = torch.tensor([[-100, 1, 2, 0, -100], [-100, 3, -100, -100, -100]])
        loss, emissions = model(ids, attention, gold, tag_mask=selected)
        self.assertEqual(emissions.shape, (2, 5, len(LABELS)))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in [model.bert.embedding.weight, model.head.blocks[0].conv1.weight,
                          model.head.blocks[1].conv2.weight, model.head.crf.transitions]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_tiny_batch_can_be_learned_and_checkpoint_restored(self):
        model = self.make_head()
        features = torch.randn(2, 4, 8)
        attention = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        gold = torch.tensor([[0, 1, 2, 0], [3, 4, 0, -100]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        initial = model(features, attention, gold)[0].item()
        for _ in range(60):
            optimizer.zero_grad()
            loss, _ = model(features, attention, gold)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.1)
        self.assertEqual(model.decode(features, attention), [[0, 1, 2, 0], [3, 4, 0]])
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.make_head()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.assertEqual(model.decode(features, attention), restored.decode(features, attention))

    def test_invalid_tag_selection_is_rejected(self):
        model = self.make_head()
        values = torch.randn(1, 3, len(LABELS))
        attention = torch.tensor([[1, 1, 0]])
        for selected in ([[0, 0, 0]], [[0, 1, 1]]):
            with self.assertRaises(ValueError):
                model.decode_emissions(values, attention, torch.tensor(selected))

    def test_cpu_mixed_precision_loss_has_finite_gradients(self):
        model = self.make_head()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = model(torch.randn(2, 3, 8), torch.tensor([[1, 1, 1], [1, 0, 0]]),
                            torch.tensor([[1, 2, 0], [3, -100, -100]]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


class NotebookTests(unittest.TestCase):
    def test_standalone_model_matches_module_and_all_cells_compile(self):
        notebook = json.loads((ROOT / "BERT_RESNET_CRF.ipynb").read_text())
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        self.assertEqual("".join(cells["model-definitions"]["source"]),
                         (ROOT / "resnet_crf.py").read_text())
        for name, cell in cells.items():
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook:{name}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
