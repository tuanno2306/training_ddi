"""Offline integration test of all new notebook cells with real Transformers."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from resnet_crf import BertResNetCRF


ROOT = Path(__file__).resolve().parents[1]


def write_fixture(path, doc_id):
    root = ET.Element("document", id=doc_id)
    text = "Aspirin and acetyl salicylic acid interact with Warfarin."
    sentence = ET.SubElement(root, "sentence", id=f"{doc_id}.s0", text=text)
    for i, term in enumerate(("Aspirin", "acetyl salicylic acid", "Warfarin")):
        start = text.index(term)
        ET.SubElement(sentence, "entity", id=f"{doc_id}.s0.e{i}",
                      text=term, type="drug", charOffset=f"{start}-{start + len(term) - 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8")


class NotebookIntegrationTests(unittest.TestCase):
    def test_train_checkpoint_evaluate_and_predict_offline(self):
        torch.set_num_threads(1)
        notebook = json.loads((ROOT / "BERT_RESNET_CRF.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder_dir = root / "tiny-bert"
            encoder_dir.mkdir()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "aspirin",
                     "and", "acetyl", "salicylic", "acid", "interact", "with",
                     "warfarin", ".", "may", "increase", "the", "anticoagulant",
                     "activities", "of"]
            vocab_path = encoder_dir / "vocab.txt"
            vocab_path.write_text("\n".join(vocab) + "\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
            tokenizer.save_pretrained(encoder_dir)
            BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=24, max_position_embeddings=64)).save_pretrained(encoder_dir)
            corpus = root / "DDICorpus"
            for index in range(4):
                write_fixture(corpus / "Train" / f"train{index}.xml", f"train{index}")
            for domain in ("DrugBank", "MedLine"):
                write_fixture(corpus / "Test" / "Test for DrugNER task" / domain / "test.xml", domain)
            output = root / "output"
            output.mkdir()
            namespace = {"__name__": "__main__"}
            logs = io.StringIO()
            with patch.dict(os.environ, {"DDI_ROOT": str(corpus), "HF_HUB_OFFLINE": "1"}), \
                    contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    name = cell["id"]
                    try:
                        exec(compile("".join(cell["source"]), f"notebook:{name}", "exec"), namespace)
                    except Exception as exc:
                        self.fail(f"Cell {name} failed: {exc}\n{logs.getvalue()}")
                    if name == "seed-device":
                        namespace["device"] = torch.device("cpu")
                    if name == "config":
                        namespace.update({
                            "MODEL_NAME": str(encoder_dir), "OUTPUT_DIR": output,
                            "BEST_MODEL_PATH": output / "best.pt", "LABEL_PATH": output / "labels.json",
                            "EPOCHS": 1, "MAX_LENGTH": 32, "TRAIN_BATCH_SIZE": 2,
                            "EVAL_BATCH_SIZE": 2, "NUM_WORKERS": 0,
                            "MODEL_CONFIG": {"channels": 12, "num_blocks": 2, "kernel_size": 3,
                                             "dilations": [1, 2], "dropout": 0.0},
                        })
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "biomedbert_resnet_crf_history.csv").is_file())
            self.assertEqual(len(namespace["history"]), 1)
            # Two sentences, each containing three annotated entities.
            self.assertEqual(namespace["test_metrics"]["gold_entities"], 6)
            self.assertEqual(len(namespace["subset_results"]), 2)
            all_model_ids = {id(p) for p in namespace["model"].parameters()}
            optimized_ids = [id(p) for group in namespace["optimizer"].param_groups for p in group["params"]]
            self.assertEqual(all_model_ids, set(optimized_ids))
            self.assertEqual(len(all_model_ids), len(optimized_ids))

            # Reload independently using the module API documented in README.
            checkpoint = torch.load(output / "best.pt", weights_only=True)
            names = [checkpoint["id2label"][i] for i in range(len(checkpoint["label2id"]))]
            restored = BertResNetCRF(checkpoint["model_name"], len(names), labels=names,
                                    **checkpoint["model_config"])
            restored.load_state_dict(checkpoint["model_state_dict"])
            restored.eval()
            predict = namespace["predict_entities"]
            text = "Aspirin interacts with Warfarin."
            actual = predict(text, restored, namespace["tokenizer"], checkpoint["id2label"], 32)
            expected = predict(text, namespace["model"], namespace["tokenizer"], checkpoint["id2label"], 32)
            self.assertEqual(actual, expected)
            self.assertEqual(predict("   ", restored, namespace["tokenizer"], checkpoint["id2label"]), [])
            for span in actual:
                self.assertEqual(span["text"], text[span["start"]:span["end"]])


if __name__ == "__main__":
    unittest.main()
