from types import SimpleNamespace
import json
import torch

from lassodiff.data import mini_grouped_pdb_dataset as mod
from lassodiff.sidechain_builder import atom14_names


def test_grouped_pdb_dataset_retains_multiple_rank_targets(tmp_path, monkeypatch):
    metadata = [{"LP_ID": "LP001", "Core_Sequence": "AAADRAAA", "Ring_Length": 4,
                 "Upper_Plug_1": 6, "Upper_Plug_2": 6}]
    metadata_path = tmp_path / "metadata.json"; metadata_path.write_text(json.dumps(metadata))
    root = tmp_path / "LP001"; root.mkdir()
    (root / "min1.pdb").write_text(""); (root / "min2.pdb").write_text("")
    def fake_process(path, candidate):
        names = atom14_names(candidate.sequence, candidate)
        heavy = tuple({name: torch.tensor([float(i), float(j), 0.0]) for j, name in enumerate(row) if name}
                      for i, row in enumerate(names))
        core = torch.zeros((8, 7, 3)); mask = candidate.core_atom_mask.clone()
        for i in range(8):
            core[i, 0] = torch.tensor([i, 0, 0]); core[i, 1] = torch.tensor([i, 1, 0]); core[i, 2] = torch.tensor([i, 2, 0]); core[i, 3] = torch.tensor([i, 3, 0])
            if mask[i, 4]: core[i, 4] = torch.tensor([i, 1.5, 0])
            if mask[i, 5]: core[i, 5] = torch.tensor([i, 2.5, 0])
            if mask[i, 6]: core[i, 6] = torch.tensor([i, 3.5, 0])
        return SimpleNamespace(core_coordinates=core, core_atom_mask=mask,
                               heavy_atom_coordinates=heavy, residue_names=(), source=str(path))
    monkeypatch.setattr(mod, "process_lasso_structure", fake_process)
    dataset = mod.GroupedMiniPDBDataset(metadata_path, tmp_path, record_ids=["LP001"])
    assert len(dataset) == 1
    assert int(dataset.examples[0]["conformer_mask"].sum()) == 2
    assert dataset.examples[0]["conformer_ranks"] == (1, 2)
