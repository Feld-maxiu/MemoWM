import numpy as np
import torch

from experiments.state_tokenizer.longmemeval_query_supervision import (
    contextualize_question, query_state_relations,
)
from experiments.state_tokenizer.train_query_retrieval_bridge import multi_positive_query_loss


def test_contextualization_excludes_answer_bearing_title():
    record = {"url": "https://store.example/products", "axtree": "RootWebArea 'Product catalogue'"}
    q = contextualize_question("What is the heading?", "Product catalogue", record)
    assert "Product catalogue" not in q and "store.example" in q
    q = contextualize_question("How many results?", "17", record)
    assert "Page: Product catalogue." in q
    q = contextualize_question("Which website?", "store.example", record)
    assert "Website:" not in q and "store.example" not in q


def test_duplicate_relations_mask_ambiguous_but_do_not_make_them_positive():
    pos, ignored = query_state_relations(["Which value?"] * 3, ["17", "17", "18"], [0, 1, 2], 3)
    assert pos.tolist() == [[True, True, False], [True, True, False], [False, False, True]]
    assert ignored.tolist() == [[False, False, True], [False, False, True], [True, True, False]]


def test_query_loss_does_not_repulse_equivalent_states():
    state = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    old, _ = multi_positive_query_loss(state, query, targets, .05)
    new, _ = multi_positive_query_loss(state, query, targets, .05, positive_mask=torch.ones(2, 2, dtype=torch.bool))
    assert new < old and abs(float(new)) < 1e-6
    new.backward()
    assert torch.isfinite(state.grad).all()


class _CountTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def test_sparse_anchor_omits_common_controls_and_keeps_bound_literal_values():
    from residualmem.benchmarks.longmemeval_anchor import extract_sparse_anchor
    tree = """RootWebArea 'Example'
  form 'Order details'
    [1] button 'Press'
    [2] button 'Save'
    [3] textbox 'Amount' value='129.37' disabled=False
    [4] button 'Export reconciliation statement'
  complementary 'Account'
    [5] StaticText 'Payment failed: account reference XH-92841 is invalid.'
"""
    a = extract_sparse_anchor(tree, {'label_document_frequency': {}, 'common_min_observations': 8}, _CountTokenizer(), max_tokens=500)
    assert '"Press"' not in a['text'] and '"Save"' not in a['text']
    assert '129.37' in a['text'] and 'Order details' in a['text'] and '"disabled":"False"' in a['text']
    assert 'Export reconciliation statement' in a['text'] and 'XH-92841' in a['text']


def test_incoming_action_is_resolved_in_previous_observation():
    from residualmem.benchmarks.longmemeval_anchor import annotated_incoming_action
    assert 'Place order' in annotated_incoming_action("click('12')", "[12] button 'Place order'")
    assert 'target in previous state' in annotated_incoming_action("click('12')", "[12] button 'Place order'")


def test_sparse_anchor_binds_price_to_item_and_does_not_keep_menu_semantics():
    from residualmem.benchmarks.longmemeval_anchor import extract_sparse_anchor
    tree = """RootWebArea 'Example'
  [1] menuitem 'Home & Kitchen'
  list ''
    listitem ''
      [2] link 'Example portable photo printer'
      StaticText '$184.99'
      [3] button 'Add to Cart'
"""
    a = extract_sparse_anchor(tree, {'label_document_frequency': {}, 'common_min_observations': 8}, _CountTokenizer())
    import json
    rows = [json.loads(line) for line in a['text'].splitlines()]
    price = next(row for row in rows if row['label'] == '$184.99')
    assert 'Example portable photo printer' in price['region']
    assert 'Home & Kitchen' not in a['text'] and 'Add to Cart' not in a['text']


def _row(trajectory, position, key, score):
    from residualmem.benchmarks.longmemeval_index import RetrievedObservation
    return RetrievedObservation(trajectory, position, score, np.zeros((2, 3), np.float32),
                                np.ones(2, bool), np.asarray(key, np.float32), f'exact-{trajectory}-{position}',
                                {'step_idx': position, 'url': 'https://example.test', 'incoming_action_text': 'click previous'})


def test_observation_blocks_bind_latent_anchor_and_group_steps():
    from experiments.state_tokenizer.run_longmemeval_local import _memory_segments
    rows = [_row('A', 7, [1, 0], 1), _row('B', 1, [0, 1], .9), _row('A', 3, [1, 0], .8)]
    segments = _memory_segments(rows, include_anchor=True, framed=True)
    texts = '\n'.join(s.text for s in segments if s.text)
    assert texts.index('state="3"') < texts.index('state="7"') < texts.index('trajectory="B"')
    assert texts.count('<observation ') == texts.count('</observation>') == 3
    for i in [0, 4, 8]:
        assert '<observation ' in segments[i].text
        assert segments[i + 1].latent is not None
        assert 'exact-' in segments[i + 2].text
        assert '</observation>' in segments[i + 3].text


def test_diversity_does_not_select_overlapping_slices():
    from residualmem.benchmarks.longmemeval_index import diverse_hits
    rows = [_row('A', 1, [1, 0], 1), _row('A', 2, [1, 0], .99),
            _row('B', 0, [0, 1], .9), _row('A', 8, [1, 0], .8)]
    out = diverse_hits(rows, top_k=3)
    assert [(r.trajectory_id, r.record_index) for r in out] == [('A', 1), ('B', 0), ('A', 8)]
