from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


class OpenDDEFeatureBuilder:
    """Build official no-MSA/no-template residue features for one sequence."""

    def __init__(self, root_dir: str | Path, *, seed: int = 101):
        self.root_dir = Path(root_dir).resolve()
        checkpoint = self.root_dir / "checkpoint" / "opendde.pt"
        common = self.root_dir / "common"
        if not checkpoint.is_file() or not common.is_dir():
            raise RuntimeError(f"OpenDDE runtime data is incomplete: {self.root_dir}")
        os.environ["OPENDDE_ROOT_DIR"] = str(self.root_dir)
        from opendde.config.inference import build_inference_config
        from opendde.data.inference.infer_dataloader import InferenceDataset

        self._temporary = tempfile.TemporaryDirectory(prefix="lassodiff-opendde-features-")
        input_path = Path(self._temporary.name) / "placeholder.json"
        input_path.write_text("[]\n", encoding="utf-8")
        configs = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
        configs.input_json_path = str(input_path)
        configs.dump_dir = self._temporary.name
        configs.use_msa = False
        configs.use_template = False
        configs.use_rna_msa = False
        configs.sample_diffusion.guidance["enable"] = False
        self.dataset = InferenceDataset(configs)
        self.seed = int(seed)

    @staticmethod
    def _canonical(sequence: str) -> str:
        sequence = "".join(str(sequence).upper().split())
        if not sequence or any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence):
            raise ValueError("OpenDDE feature sequence must contain canonical amino acids")
        return sequence

    def build(self, sequence: str, name: str = "lassodiff_sequence"):
        sequence = self._canonical(sequence)
        sample = {
            "name": str(name), "modelSeeds": [self.seed],
            "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        }
        data, _atom_array, _timing = self.dataset.process_one(sample)
        features = data["input_feature_dict"]
        features["_sequence"] = sequence
        features["_feature_builder_flags"] = json.dumps({"use_msa": False, "use_template": False})
        return features
