"""
Unit tests for pure utility functions.
No external dependencies (no PDF, no Supabase, no embedding model).

Run with:
    pytest ingest/test/test_unit.py -v
"""
import numpy as np
import pytest

from ingest.toc_chunk import slug, norm_title, flatten, windows, get_ancestor_titles
from ingest.embedding_import import validate_embeddings


# ─────────────────────────────────────────────
# slug()
# ─────────────────────────────────────────────

class TestSlug:
    def test_basic(self):
        assert slug("Hello World") == "hello-world"

    def test_special_chars_removed(self):
        assert slug("Chapter 1: Introduction!") == "chapter-1-introduction"

    def test_multiple_spaces_collapsed(self):
        assert slug("too  many   spaces") == "too-many-spaces"

    def test_truncated_at_80(self):
        long = "a" * 100
        assert len(slug(long)) == 80

    def test_empty_string_returns_untitled(self):
        assert slug("") == "untitled"

    def test_only_special_chars_returns_untitled(self):
        assert slug("!!!") == "untitled"

    def test_leading_trailing_whitespace_stripped(self):
        assert slug("  hello  ") == "hello"


# ─────────────────────────────────────────────
# norm_title()
# ─────────────────────────────────────────────

class TestNormTitle:
    def test_collapses_internal_whitespace(self):
        assert norm_title("Hello   World") == "Hello World"

    def test_strips_leading_trailing(self):
        assert norm_title("  hello  ") == "hello"

    def test_none_returns_empty_string(self):
        assert norm_title(None) == ""

    def test_empty_string_returns_empty(self):
        assert norm_title("") == ""

    def test_tabs_and_newlines_collapsed(self):
        assert norm_title("hello\t\nworld") == "hello world"

    def test_normal_string_unchanged(self):
        assert norm_title("Hello World") == "Hello World"


# ─────────────────────────────────────────────
# flatten()
# ─────────────────────────────────────────────

class TestFlatten:
    def test_single_node_no_children(self):
        nodes = [{"id": "a", "children": []}]
        result = flatten(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_nested_children_included(self):
        nodes = [{
            "id": "a",
            "children": [{
                "id": "b",
                "children": [{"id": "c", "children": []}]
            }]
        }]
        result = flatten(nodes)
        ids = [n["id"] for n in result]
        assert ids == ["a", "b", "c"]

    def test_multiple_top_level_nodes(self):
        nodes = [
            {"id": "a", "children": []},
            {"id": "b", "children": []},
        ]
        result = flatten(nodes)
        assert [n["id"] for n in result] == ["a", "b"]

    def test_empty_list(self):
        assert flatten([]) == []

    def test_depth_first_order(self):
        nodes = [{
            "id": "root",
            "children": [
                {"id": "child1", "children": [{"id": "grandchild", "children": []}]},
                {"id": "child2", "children": []},
            ]
        }]
        result = flatten(nodes)
        assert [n["id"] for n in result] == ["root", "child1", "grandchild", "child2"]


# ─────────────────────────────────────────────
# windows()
# ─────────────────────────────────────────────

class TestWindows:
    def test_short_text_returned_as_single_window(self):
        text = "hello world"
        result = windows(text, size=100, overlap=10)
        assert result == [text]

    def test_long_text_split_into_multiple_windows(self):
        text = "a" * 500
        result = windows(text, size=200, overlap=0)
        assert len(result) == 3  # 200, 200, 100

    def test_overlap_repeats_content(self):
        text = "a" * 300
        result = windows(text, size=200, overlap=50)
        # Second window should start at 200-50=150
        assert result[1] == text[150:350]

    def test_exact_size_text_is_single_window(self):
        text = "x" * 200
        result = windows(text, size=200, overlap=10)
        assert result == [text]

    def test_each_window_bounded_by_size(self):
        text = "b" * 1000
        result = windows(text, size=200, overlap=20)
        for w in result:
            assert len(w) <= 200


# ─────────────────────────────────────────────
# get_ancestor_titles()
# ─────────────────────────────────────────────

class TestGetAncestorTitles:
    def _make_node(self, id, title, level, page_start, children=None):
        return {"id": id, "title": title, "level": level, "page_start": page_start,
                "page_end": page_start + 5, "children": children or []}

    def test_no_ancestors_for_top_level(self):
        node = self._make_node("h1-1__intro", "Introduction", 1, 1)
        flat = [node]
        assert get_ancestor_titles(node, flat) == []

    def test_single_ancestor(self):
        parent = self._make_node("h1-1__ch1", "Chapter 1", 1, 1)
        child  = self._make_node("h1-1-1__sec", "Section 1.1", 2, 2)
        flat = [parent, child]
        result = get_ancestor_titles(child, flat)
        assert result == [(1, "Chapter 1")]

    def test_two_levels_of_ancestors(self):
        h1 = self._make_node("h1-1__ch1", "Chapter 1", 1, 1)
        h2 = self._make_node("h1-1-1__sec", "Section 1.1", 2, 2)
        h3 = self._make_node("h1-1-1-1__sub", "Subsection", 3, 3)
        flat = [h1, h2, h3]
        result = get_ancestor_titles(h3, flat)
        assert result == [(1, "Chapter 1"), (2, "Section 1.1")]

    def test_returns_list_of_tuples(self):
        parent = self._make_node("h1-1__ch1", "Chapter 1", 1, 1)
        child  = self._make_node("h1-1-1__sec", "Section", 2, 2)
        flat = [parent, child]
        result = get_ancestor_titles(child, flat)
        assert all(isinstance(level, int) and isinstance(title, str)
                   for level, title in result)


# ─────────────────────────────────────────────
# validate_embeddings()
# ─────────────────────────────────────────────

class TestValidateEmbeddings:
    def _normalized(self, n, dim=4):
        """Create n normalized random vectors of dimension dim."""
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_valid_embeddings_pass(self):
        emb = self._normalized(10)
        passed, issues = validate_embeddings(emb)
        assert passed
        assert issues == []

    def test_wrong_shape_1d_fails(self):
        emb = np.array([1.0, 2.0, 3.0])
        passed, issues = validate_embeddings(emb)
        assert not passed
        assert any("Wrong shape" in i for i in issues)

    def test_wrong_dimension_flagged(self):
        emb = self._normalized(5, dim=4)   # dim=4, expected=8
        passed, issues = validate_embeddings(emb, expected_dim=8)
        assert not passed
        assert any("Wrong embedding dimension" in i for i in issues)

    def test_correct_dimension_passes(self):
        emb = self._normalized(5, dim=384)
        passed, issues = validate_embeddings(emb, expected_dim=384)
        assert passed

    def test_nan_values_flagged(self):
        emb = self._normalized(5)
        emb[2, 0] = float("nan")
        passed, issues = validate_embeddings(emb)
        assert not passed
        assert any("NaN" in i for i in issues)

    def test_inf_values_flagged(self):
        emb = self._normalized(5)
        emb[1, 1] = float("inf")
        passed, issues = validate_embeddings(emb)
        assert not passed
        assert any("Inf" in i for i in issues)

    def test_zero_vector_flagged(self):
        emb = self._normalized(5)
        emb[0] = 0.0
        passed, issues = validate_embeddings(emb)
        assert not passed
        assert any("zero" in i for i in issues)

    def test_near_duplicate_flagged(self):
        emb = self._normalized(3)
        emb[1] = emb[0]  # exact duplicate
        passed, issues = validate_embeddings(emb)
        assert not passed
        assert any("near-duplicate" in i for i in issues)

    def test_single_embedding_skips_duplicate_check(self):
        emb = self._normalized(1)
        passed, issues = validate_embeddings(emb)
        assert passed
