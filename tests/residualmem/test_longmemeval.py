from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from residualmem.benchmarks.longmemeval import (
    LONGMEMEVAL_ANCHOR_PROTOCOL,
    canonicalize_axtree,
    extract_anchor,
    iter_anchor_lines,
    normalize_action,
    read_lme_trajectory,
)
from residualmem.benchmarks.longmemeval_index import (
    LongMemEvalLatentIndex,
    RetrievedObservation,
)
from experiments.state_tokenizer.prepare_webchain_legacy import serialize_axtree
from experiments.state_tokenizer.filter_longmemeval_store import (
    filter_axtree,
    structured_anchor,
)
from experiments.state_tokenizer.train_query_retrieval_bridge import (
    multi_positive_query_loss,
)
from experiments.state_tokenizer.run_longmemeval_local import _memory_segments


def test_canonicalization_preserves_opaque_ids_and_state_attributes():
    raw = "RootWebArea 'Page'\r\n\t[a56] button 'Menu' expanded=False\r\n"
    result = canonicalize_axtree(raw)
    assert result == "RootWebArea 'Page'\n  [a56] button 'Menu' expanded=False"
    lines = list(iter_anchor_lines(result))
    assert "[a56] button 'Menu' expanded=False" in lines


def test_anchor_keeps_idless_text_and_complete_lines():
    tree = (
        "RootWebArea 'Shop'\n"
        "  StaticText 'Total 19.95'\n"
        "  [b-19] textbox 'Amount' value='19.95' disabled=False\n"
        "  generic ''\n"
    )
    packed = extract_anchor(tree, max_bytes=1000)
    assert packed["protocol"] == LONGMEMEVAL_ANCHOR_PROTOCOL
    assert "StaticText 'Total 19.95'" in packed["text"]
    assert "[b-19] textbox 'Amount' value='19.95' disabled=False" in packed["text"]
    assert "generic" not in packed["text"]


def test_anchor_budget_drops_whole_lines_without_cutting_values():
    tree = "\n".join(
        ["RootWebArea 'Page'"]
        + [f"  [{i}-x] textbox 'Field {i}' value='exact-value-{i}'" for i in range(20)]
    )
    packed = extract_anchor(tree, max_bytes=180)
    assert packed["bytes"] <= 180 or packed["lines"] == 1
    for line in packed["text"].splitlines():
        assert not line.endswith("exact-value-")


def test_action_normalization_is_stable():
    assert normalize_action(None) is None
    assert normalize_action("  click   b19  ") == "click b19"
    assert normalize_action({"type": "fill", "text": "A"}) == '{"text":"A","type":"fill"}'


def test_read_lme_trajectory_validates_and_resolves_screenshot(tmp_path: Path):
    image = tmp_path / "shot.png"
    Image.new("RGB", (8, 8), "white").save(image)
    source = tmp_path / "trajectories.jsonl"
    row = {
        "id": "traj-1",
        "domain": "web",
        "states": [{
            "state_index": 0,
            "step": 0,
            "url": "https://example.test",
            "action": None,
            "thought": None,
            "accessibility_tree": "RootWebArea 'Example'",
            "screenshot": "shot.png",
        }],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
                                                                              
                                                                          
    object_path = tmp_path / "trajectory.json"
    object_path.write_text(json.dumps(row), encoding="utf-8")
    loaded = read_lme_trajectory(object_path)
    assert loaded["id"] == "traj-1"
    assert loaded["observations"][0].screenshot == image.resolve()


def test_latent_index_retrieval_and_expansion_are_trajectory_local(tmp_path: Path):
    root = tmp_path / "cache" / "trajectories"
    root.mkdir(parents=True)
    records = [
        {"anchor_text": "state 0"},
        {"anchor_text": "state 1"},
    ]
    npz_path = root / "traj-a.npz"
    import numpy as np
    np.savez(
        npz_path,
        xbar=np.zeros((2, 2, 3), np.float32),
        valid=np.ones((2, 2), np.bool_),
        key=np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32),
        metadata=np.asarray(json.dumps({"trajectory_id": "traj-a", "records": records})),
    )
    index = LongMemEvalLatentIndex(tmp_path / "cache")
    hits = index.query(np.asarray([1.0, 0.0], np.float32), top_k=1)
    assert [(hit.trajectory_id, hit.record_index) for hit in hits] == [("traj-a", 0)]
    expanded = index.expanded_context(hits, radius=1)
    assert [row.record_index for row in expanded] == [0, 1]


def test_hybrid_index_fuses_anchor_bm25_and_keeps_one_trajectory(tmp_path: Path):
    import numpy as np

    root = tmp_path / "cache" / "trajectories"
    root.mkdir(parents=True)
    np.savez(
        root / "traj-a.npz",
        xbar=np.zeros((1, 2, 3), np.float32),
        valid=np.ones((1, 2), np.bool_),
        key=np.asarray([[1.0, 0.0]], np.float32),
        metadata=np.asarray(json.dumps({
            "trajectory_id": "traj-a",
            "records": [{"url": "http://localhost:9082/", "anchor_text": ""}],
        })),
    )
    np.savez(
        root / "traj-b.npz",
        xbar=np.zeros((1, 2, 3), np.float32),
        valid=np.ones((1, 2), np.bool_),
        key=np.asarray([[0.0, 1.0]], np.float32),
        metadata=np.asarray(json.dumps({
            "trajectory_id": "traj-b",
            "records": [{"url": "http://localhost:9080/", "anchor_text": "Source Code"}],
        })),
    )
    index = LongMemEvalLatentIndex(tmp_path / "cache")
    latent_only = index.query_hybrid(
        np.asarray([1.0, 0.0], np.float32), "Source Code",
        top_k=2, bm25_weight=0.0, trajectory_top_k=1,
    )
    lexical_only = index.query_hybrid(
        np.asarray([1.0, 0.0], np.float32), "Source Code",
        top_k=2, bm25_weight=1.0, trajectory_top_k=1,
    )
    goal_prior = index.query_hybrid(
        np.asarray([1.0, 0.0], np.float32), "Source Code",
        top_k=2, bm25_weight=1.0, trajectory_top_k=1,
        trajectory_prior={"traj-a": 1.0, "traj-b": 0.0},
        trajectory_prior_weight=1.0,
    )
    assert {row.trajectory_id for row in latent_only} == {"traj-a"}
    assert {row.trajectory_id for row in lexical_only} == {"traj-b"}
    assert {row.trajectory_id for row in goal_prior} == {"traj-a"}
    assert index.trajectory_ids_for_netloc("localhost:9082") == {"traj-a"}


def test_text_index_retrieves_globally_and_can_scope_trajectories(tmp_path: Path):
    import numpy as np

    from residualmem.benchmarks.longmemeval_text_index import (
        LongMemEvalTextIndex,
        PROTOCOL,
    )

    path = tmp_path / "text-index.npz"
    np.savez_compressed(
        path,
        embedding=np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32),
        trajectory_id=np.asarray(["traj-a", "traj-b"]),
        center_index=np.asarray([3, 7], np.int64),
        slice_start=np.asarray([2, 6], np.int64),
        slice_end=np.asarray([4, 8], np.int64),
        goal=np.asarray(["goal-a", "goal-b"]),
        context_text=np.asarray(["context-a", "context-b"]),
        metadata=np.asarray(json.dumps({"protocol": PROTOCOL})),
    )
    index = LongMemEvalTextIndex(path)
    hits = index.query(np.asarray([1.0, 0.0], np.float32), top_k=2)
    assert [(hit.trajectory_id, hit.center_index) for hit in hits] == [
        ("traj-a", 3), ("traj-b", 7),
    ]
    scoped = index.query(
        np.asarray([1.0, 0.0], np.float32),
        trajectory_ids=["traj-b"],
        top_k=2,
    )
    assert [(hit.trajectory_id, hit.center_index) for hit in scoped] == [("traj-b", 7)]


def _compact_state() -> dict:
    return {
        "state_index": 4,
        "url": "http://localhost:9082/catalogsearch/result/?q=printer",
        "action": "click('739')",
        "axtree": (
            "RootWebArea 'Search results'\n"
            "  [12] link 'Home'\n"
            "  [16] StaticText 'One Stop Market'\n"
            "  [21] combobox 'Search' value='Canon printer'\n"
            "  [24] button 'Search'\n"
            "  [85] StaticText '$184.99'\n"
            "  [86] button 'Add to Cart'\n"
            "  [182] checkbox 'Add to Cart' checked='false'\n"
            "  [201] StaticText 'There was an error processing the order.'\n"
            "  [203] StaticText 'Welcome back, Emma Lopez!'\n"
        ),
    }


def test_compact_view_keeps_controls_literals_and_state_context():
    from residualmem.benchmarks.longmemeval_compact import (
        compact_axtree_text,
        compact_state_text,
    )

    state = _compact_state()
    text = compact_state_text(state)
                                         
    assert "State 4" in text
    assert f"URL: {state['url']}" in text
    assert "Page: Search results" in text
    assert "Action: click('739')" in text
                                                                   
    for control in ("[12] link 'Home'", "[21] combobox 'Search'", "[24] button 'Search'"):
        assert control in text
                                                            
    assert "[85] StaticText '$184.99'" in text
    assert "'There was an error processing the order.'" in text
                                                       
    assert "Welcome back, Emma Lopez!" not in compact_axtree_text(state)
                                                                      
    assert "checked=" not in text


def test_compact_context_states_replace_only_the_axtree_payload():
    from residualmem.benchmarks.longmemeval_compact import (
        compact_context_states,
    )

    state = _compact_state()
    first, second = compact_context_states([state, dict(state)])
    assert "'$184.99'" in first["axtree"]
    assert second["axtree"] == (
        "(no new lines relative to the previous state in this slice)"
    )
    assert first["url"] == state["url"]
    assert second["url"] == state["url"]
                                              
    assert state["axtree"].startswith("RootWebArea")


def test_compact_keeps_labels_bound_to_values_and_drops_bare_counters():
    from residualmem.benchmarks.longmemeval_compact import compact_axtree_text

    state = {
        "state_index": 7,
        "url": "http://localhost:9083/admin/report",
        "action": "click('42')",
        "axtree": (
            "RootWebArea 'Low Stock Report'\n"
            "  [769] columnheader 'Source Code'\n"
            "  StaticText 'Summary'\n"
            "  StaticText '6444'\n"
            "  StaticText 'Qty'\n"
            "  StaticText '5'\n"
            "  [177] DisclosureTriangle 'Hide this forum'\n"
            "  StaticText 'Copyright © 2013-present Magento, Inc. All rights reserved.'\n"
            "  StaticText '12 items'\n"
        ),
    }
    text = compact_axtree_text(state)
                                                                  
    assert "[769] columnheader 'Source Code'" in text
    assert "'Summary'" in text
    assert "'Qty'" in text
                                                           
    assert "[177] DisclosureTriangle 'Hide this forum'" in text
                                                                       
    assert "6444" not in text
    assert "Copyright" not in text
    assert "12 items" not in text
                                                                              
    inside_cell = {
        "state_index": 8,
        "url": state["url"],
        "action": "click('43')",
        "axtree": (
            "RootWebArea 'Low Stock Report'\n"
            "  [804] table ''\n"
            "    [805] rowgroup ''\n"
            "      [806] row ''\n"
            "        [807] columnheader 'Qty'\n"
            "        [812] gridcell ''\n"
            "          StaticText '5'\n"
        ),
    }
    cell_text = compact_axtree_text(inside_cell)
    assert "StaticText '5'" in cell_text
    assert "[807] columnheader 'Qty'" in cell_text


def test_reader_system_prompts_keep_official_bytes_and_add_the_premise_rule():
    from experiments.state_tokenizer.longmemeval_reader import (
        LONGMEMEVAL_SYSTEM_PROMPT,
        SYSTEM_PROMPTS,
        LME_WEB_SYSTEM_PROMPT,
        LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES,
        longmemeval_system_prompt,
    )

                                                                             
                                                                           
    assert "in \x08oxed{}" in LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES
    assert "\\boxed{}" in LME_WEB_SYSTEM_PROMPT
    assert len(LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES) == len(LME_WEB_SYSTEM_PROMPT) - 1
                                                                              
    assert LONGMEMEVAL_SYSTEM_PROMPT.startswith(LME_WEB_SYSTEM_PROMPT)
    assert "If the premise is false" in longmemeval_system_prompt
    assert "instead of \\boxed{UNKNOWN}" in LONGMEMEVAL_SYSTEM_PROMPT
                                                                                
    assert len(LONGMEMEVAL_SYSTEM_PROMPT) - len(LME_WEB_SYSTEM_PROMPT) <= 110
    assert longmemeval_system_prompt == LONGMEMEVAL_SYSTEM_PROMPT
    assert set(SYSTEM_PROMPTS) == {"longmemeval", "official", "abscontract"}
    assert SYSTEM_PROMPTS["longmemeval"] == LONGMEMEVAL_SYSTEM_PROMPT


def test_abscontract_prompt_is_the_byte_exact_deployed_protocol():
    import hashlib

    from experiments.state_tokenizer.longmemeval_reader import SYSTEM_PROMPTS
    from experiments.state_tokenizer.prompts import ABS_CONTRACT_SYSTEM_PROMPT

                                                                    
                                                                           
                                                                             
                                           
    assert SYSTEM_PROMPTS["abscontract"] == ABS_CONTRACT_SYSTEM_PROMPT
    assert (
        hashlib.sha256(ABS_CONTRACT_SYSTEM_PROMPT.encode()).hexdigest()
        == "f0fc164adc10ef62edf48ba9b2f4f55ccc0ff1674a42a9834dd65e2132abf778"
    )
                                                                               
    assert "output exactly \\boxed{UNKNOWN}" not in ABS_CONTRACT_SYSTEM_PROMPT
    assert "never a bare \\boxed{UNKNOWN}" in ABS_CONTRACT_SYSTEM_PROMPT
    assert "check whether the memory context actually contains" in ABS_CONTRACT_SYSTEM_PROMPT


def test_compact_drops_timestamps_in_large_grids_and_duplicate_lines():
    from residualmem.benchmarks.longmemeval_compact import (
        compact_slice_text,
        compact_state_text,
    )

    def _grid(count: int) -> str:
        rows = "\n".join(
            f"      [80{6 + i}] row ''\n        [81{i}] gridcell 'Order'\n"
            f"          StaticText '1000{i}'\n"
            f"          StaticText 'Apr 1{i}, 2023 12:13:40 PM'"
            for i in range(count)
        )
        return f"    [805] rowgroup ''\n{rows}"

    state = {
        "state_index": 1,
        "url": "http://localhost:9083/admin/sales/order",
        "action": "click('9')",
        "axtree": f"RootWebArea 'Orders'\n  [803] grid ''\n{_grid(8)}",
    }
    large = compact_state_text(state)
    assert "Apr 10, 2023" not in large
    assert "10000" in large
    state["axtree"] = f"RootWebArea 'Orders'\n  [803] grid ''\n{_grid(3)}"
    assert "Apr 10, 2023" in compact_state_text(state)
                                                                    
    slice_text = compact_slice_text([state, dict(state)])
    assert slice_text.count("Apr 10, 2023") == 1
    assert slice_text.count("State 1") == 2
                                              
    assert state["axtree"].startswith("RootWebArea")


def test_longmemeval_anchor_is_an_explicit_context_ablation():
    import numpy as np

    row = RetrievedObservation(
        trajectory_id="traj-a",
        record_index=0,
        score=1.0,
        xbar=np.zeros((2, 3), np.float32),
        valid=np.ones(2, np.bool_),
        key=np.ones(4, np.float32),
        anchor_text="exact UI value",
        record={},
    )
    with_anchor = _memory_segments([row], include_anchor=True)
    latent_only = _memory_segments([row], include_anchor=False)
    assert len(with_anchor) == 2
    assert with_anchor[0].latent is not None
    assert with_anchor[1].text == "exact UI value"
    assert len(latent_only) == 1
    assert latent_only[0].latent is not None


def test_webchain_json_axtree_serialization_preserves_ids_and_false_attrs():
    value = serialize_axtree(json.dumps({
        "role": "generic", "name": "Root",
        "attributes": {"data-imean-axt-id": "17"},
        "children": [{
            "role": "button", "name": "Save",
            "attributes": {"disabled": False, "aria-label": "Save"},
        }],
    }))
    assert "[17] generic 'Root'" in value
    assert "button 'Save'" in value
    assert "disabled='False'" in value


def test_structure_filter_preserves_multiline_ui_text_and_deduplicates_wrappers():
    value = (
        "[1] generic 'Skip\\nmain content\\nEnds Monday | Online Only'\n"
        "  [2] generic 'Ends Monday | Online Only'\n"
        "    [3] generic 'Ends Monday | Online Only'\n"
        "  StaticText 'window.customerConfig = JSON.parse(\\\"telemetry\\\");'\n"
        "  [4] button 'Checkout' disabled='False'\n"
    )
    filtered, stats = filter_axtree(
        value, max_chars=1000, max_node_chars=512, max_href=256,
    )
    anchor = structured_anchor(value, max_bytes=1000)
    combined = filtered + "\n" + str(anchor["text"])
    assert "Ends Monday | Online Only" in combined
    assert "Checkout" in combined
    assert "customerConfig" not in combined
    assert filtered.count("Ends Monday | Online Only") <= 2
    assert str(anchor["text"]).count("Ends Monday | Online Only") <= 2
    assert stats["filtered_chars"] <= 1000
    assert anchor["bytes"] <= 1000


def test_multi_positive_query_loss_rewards_all_questions_for_the_same_state():
    import torch

    state = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    query = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    target = torch.tensor([0, 0, 1])
    aligned, parts = multi_positive_query_loss(state, query, target, temperature=0.1)
    swapped, _ = multi_positive_query_loss(
        state, query, torch.tensor([1, 1, 0]), temperature=0.1,
    )
    assert torch.isfinite(aligned)
    assert aligned < swapped
    assert set(parts) == {"query_to_state", "state_to_query"}
    aligned.backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


def test_multi_positive_query_loss_rejects_states_without_a_positive_query():
    import pytest
    import torch

    with pytest.raises(ValueError, match="every sampled state"):
        multi_positive_query_loss(
            torch.eye(3), torch.eye(2, 3), torch.tensor([0, 1]), temperature=0.1,
        )
