# PR #22893 — Test Coverage & Review Analysis

> **Branch:** `fix-tflite-gemma-issues-test-coverage`  
> **Base PR:** `pctablet505:fix-tflite-gemma-issues` (keras-team/keras #22893)  
> **Status:** Test-only changes committed. Implementation fixes pending.  
> **Last updated:** 2026-05-19

---

## 1. What This Branch Contains

This branch adds **only test files and test fixes** on top of the original PR. No implementation logic has been changed.

### Committed changes

| File | Action | Purpose |
|------|--------|---------|
| `keras/src/backend/torch/core_test.py` | **New** | Unit tests for `SymInt`/`SymFloat` handling in `convert_to_tensor` and `slice` |
| `keras/src/layers/layer_test.py` | **Modified** | 6 new tests: `_is_concrete_shapes_dict`, scope guards, save-after-export, graceful failure handling, explicit-`None` dim preservation |
| `keras/src/export/litert_test.py` | **Modified** | Fixed incorrect TFLite comment; strengthened custom-build test to use varying seq lengths |
| `keras/src/export/litert_torch_test.py` | **Modified** | Added single-tensor subclass model signature test |

---

## 2. Test Results (verified on torch backend)

All **12 newly added tests pass**.

```bash
KERAS_BACKEND=torch pytest keras/src/backend/torch/core_test.py \
  keras/src/layers/layer_test.py::LayerTest::test_is_concrete_shapes_dict_edge_cases \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_skips_shapes_dict_in_stateless_scope \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_skips_shapes_dict_in_symbolic_scope \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_gracefully_handles_shapes_dict_failure \
  keras/src/layers/layer_test.py::LayerTest::test_functional_dict_input_preserves_explicit_none_dims \
  keras/src/layers/layer_test.py::LayerTest::test_model_save_after_torch_export
```

```
12 passed in 3.5s
```

Pre-commit (`api_gen`, `ruff`, `ruff-format`) also passes.

---

## 3. Review Comment Tracker

This table maps every `amitsrivastava78` comment to whether it is addressed by tests in this branch or still needs implementation work.

| # | Comment | File / Line | Priority | Status | Notes |
|---|---------|-------------|----------|--------|-------|
| 1 | **Zero coverage for torch SymInt/SymFloat** | `torch/core.py:646` | Medium | ✅ **Tested** | New `torch/core_test.py` covers scalar, list, tuple, explicit dtype, and slice fast-path |
| 2 | **`_is_concrete_shapes_dict` needs direct tests** | `layer.py:2034` | Medium | ✅ **Tested** | `test_is_concrete_shapes_dict_edge_cases` covers None, nested dict, mixed symbolic, np.integer, empty dict, list/tuple |
| 3 | **Dict-only gate untested with non-dict subclass inputs** | `litert_torch_test.py:124` | Medium | ✅ **Tested** | `test_subclass_model_single_tensor_input_signature_reflects_recent_call` added |
| 4 | **Untested try/except and scope guards** | `layer.py:1574` | Medium | ✅ **Tested** | `test_maybe_build_skips_shapes_dict_in_stateless_scope`, `test_maybe_build_skips_shapes_dict_in_symbolic_scope`, `test_maybe_build_gracefully_handles_shapes_dict_failure` |
| 5 | **Missing end-to-end save/load after tracing** | `layer.py:1545` | **High** | ✅ **Tested** | `test_model_save_after_torch_export` does round-trip `torch.export` → `save()` → `load_model()` |
| 6 | **Dict check breaks explicit dynamic shapes** | `export_utils.py:46` | **Critical** | ⚠️ **Regression test added** | `test_functional_dict_input_preserves_explicit_none_dims` currently passes because Functional models skip `_maybe_build`, but it guards against future refactors |
| 7 | **Sequential models unhandled** | `export_utils.py:42` | Low | ⏳ **Pending** | Out of scope for current PR unless expanded |
| 8 | **Unsafe recursion in `_update_spec`** | `export_utils.py:66` | Medium | ⏳ **Pending** | Needs structural-alignment guards in implementation |
| 9 | **`torch.jit.is_tracing()` wrong for `torch.export`** | `layer.py:1546` | Medium | ⏳ **Pending** | Dead code; should be removed or replaced with Dynamo-aware guard |
| 10 | **Hot-path regression in `_maybe_build`** | `layer.py:1547` | **High** | ⏳ **Pending** | Needs fast-path optimization (move imports, cache `is_default` check) |
| 11 | **Incorrect TFLite comment** | `litert_test.py:1305` | Trivial | ✅ **Fixed** | Comment corrected |
| 12 | **Custom-build test uses identical feature dim** | `litert_test.py:1334` | Medium | ✅ **Fixed** | Test now uses `Embedding` + varying seq length to actually exercise the guard |

---

## 4. Implementation Fixes Still Needed

The following require **code changes outside test files**:

### 4.1 Critical — Dynamic shape clobbering (Comment 6)
**File:** `keras/src/export/export_utils.py`  
**Problem:** `_update_spec` does `(None,) + tuple(shape)[1:]`, which preserves batch as `None` but overwrites explicitly dynamic inner dimensions (e.g. `Input(shape=(None,))` becomes `(None, 32)` after a call at length 32).  
**Fix:** When merging `actual_shapes` into `spec.shape`, preserve existing `None` entries from the original `spec.shape` instead of blindly copying all concrete dimensions.

### 4.2 High — Hot-path regression (Comment 10)
**File:** `keras/src/layers/layer.py` (`_maybe_build`)  
**Problem:** Every call to a built model now pays for imports, scope checks, `get_shapes_dict()`, and recursive `_is_concrete_shapes_dict()` traversal.  
**Fix suggestions:**
- Move imports to module top-level.
- Cache `utils.is_default(self.build)` on first check.
- Skip the entire block for regular `Layer` instances quickly.
- Only run full logic when shapes have actually changed or backend is torch.

### 4.3 Medium — Dead `torch.jit.is_tracing()` check (Comment 9)
**File:** `keras/src/layers/layer.py`  
**Problem:** `torch.jit.is_tracing()` is False during `torch.export` (Dynamo/FX). The `try/except` below already handles symbolic failures, making this check dead weight.  
**Fix:** Remove the `torch.jit.is_tracing()` block or replace with a `torch._dynamo` guard if one exists.

### 4.4 Medium — Unsafe recursion guards (Comment 8)
**File:** `keras/src/export/export_utils.py` (`_update_spec`)  
**Problem:** If `spec` is a list/tuple but `shape` is `None` or shorter, `shape[i]` crashes.  
**Fix:** Add structural-alignment validation before recursing.

### 4.5 Low — Sequential model support (Comment 7)
**File:** `keras/src/export/export_utils.py`  
**Problem:** Sequential models are not covered by the shape-update logic.  
**Fix:** Extend `_maybe_build` / `get_input_signature` to handle Sequential models if maintainers agree.

---

## 5. Quick Commands for Verification

### Run only the new tests
```bash
KERAS_BACKEND=torch pytest keras/src/backend/torch/core_test.py \
  keras/src/layers/layer_test.py::LayerTest::test_is_concrete_shapes_dict_edge_cases \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_skips_shapes_dict_in_stateless_scope \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_skips_shapes_dict_in_symbolic_scope \
  keras/src/layers/layer_test.py::LayerTest::test_maybe_build_gracefully_handles_shapes_dict_failure \
  keras/src/layers/layer_test.py::LayerTest::test_functional_dict_input_preserves_explicit_none_dims \
  keras/src/layers/layer_test.py::LayerTest::test_model_save_after_torch_export \
  -v
```

### Run pre-commit on changed files
```bash
pre-commit run --files \
  keras/src/export/litert_test.py \
  keras/src/export/litert_torch_test.py \
  keras/src/layers/layer_test.py \
  keras/src/backend/torch/core_test.py
```

### Full layer test suite (sanity check)
```bash
KERAS_BACKEND=torch pytest keras/src/layers/layer_test.py -v --tb=short
```

---

## 6. Environment Used for Verification

- **OS:** Linux
- **Python:** 3.12.13
- **Keras:** 3.15.0 (local PR branch)
- **Torch:** 2.10.0+cu128
- **keras_hub:** 0.26.0 (gemma3_270m load blocked by 403, skipped)
- **Backend:** torch

---

## 7. Original Review Analysis

For the full deep-dive on each comment (including which ones show AI-copy vs. original human analysis), see the earlier file:

> **`pr-22893-review-analysis.md`** (also in this branch)

---

*End of document — ready for handoff to another device.*
