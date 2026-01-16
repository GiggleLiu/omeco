# Julia vs Rust Test Gap Analysis

## Missing Julia Tests in Rust Implementation

---

## 1. Greedy Optimizer (greedy.jl)

### ❌ NOT Implemented in Rust:

#### 1.1 **`flatten` operation** (Julia lines 73-108, 121-122)
- **What it does**: Converts `NestedEinsum` back to flat `EinCode`, verifies `flatten(optcode) == code`
- **Julia code**:
  ```julia
  flatten(code::NestedEinsum) = ... # converts nested tree back to flat EinCode
  @test flatten(optcode) == code     # verifies round-trip
  ```
- **Missing in Rust**: No equivalent `to_eincode()` or `flatten()` method
- **Importance**: **MEDIUM** - Useful for verification and debugging
- **Recommendation**: Add `impl NestedEinsum { fn to_eincode(&self) -> EinCode }`

#### 1.2 **`flop()` computation tests** (Julia lines 48-50)
- **What it does**: Tests FLOP count calculation
- **Julia code**:
  ```julia
  @test cc.tc ≈ log2(flop(optcode, size_dict))
  @test flop(EinCode([['i']], []), Dict('i'=>4)) == 4
  ```
- **Missing in Rust**: `nested_flop` exists but no explicit unit tests
- **Importance**: **MEDIUM** - Validates time complexity calculation
- **Recommendation**: Add explicit tests for `nested_flop` function

#### 1.3 **`parse_tree` round-trip tests** (Julia lines 40-41)
- **What it does**: Tests `NestedEinsum → ContractionTree → NestedEinsum` round-trip
- **Julia code**:
  ```julia
  optcode1 = parse_eincode(incidence_list, tree, vertices=vertices)
  tree2 = parse_tree(optcode1, vertices)
  @test tree2 == tree
  ```
- **Missing in Rust**: No `parse_tree` function
- **Importance**: **LOW** - Julia-specific utility, different design in Rust
- **Recommendation**: Not needed (architectural difference)

#### 1.4 **Low-level function tests**:

**a) `contract_pair!` tests** (lines 24-30)
- Tests mutation of IncidenceList after contracting two vertices
- **Missing in Rust**: Internal implementation, not exposed
- **Importance**: **LOW** - internal implementation detail

**b) `evaluate_costs` tests** (lines 31-32)
- Tests cost graph generation
- **Missing in Rust**: Different cost evaluation architecture
- **Importance**: **LOW** - internal implementation detail

**c) `analyze_contraction` tests** (lines 183-192)
- Tests leg categorization (l1, l2, l12, l01, l02, l012)
- **Missing in Rust**: No equivalent function exposed
- **Importance**: **LOW** - internal implementation detail

**d) `compute_contraction_dims` tests** (lines 194-242)
- Comprehensive tests comparing optimized vs reference implementation
- Tests dimension computation for various vertex pairs
- **Missing in Rust**: `compute_contraction_output_with_hypergraph` not unit tested
- **Importance**: **MEDIUM** - validates correctness of dimension calculations
- **Recommendation**: Add unit tests for `compute_contraction_output_with_hypergraph`

**e) `greedy_loss` function tests** (lines 244-270)
- Tests loss function with different alpha values
- Compares optimized vs original implementation
- **Missing in Rust**: Loss computation is internal, not unit tested
- **Importance**: **MEDIUM** - validates cost function accuracy
- **Recommendation**: Add unit tests for loss computation logic

---

## 2. TreeSA Optimizer (treesa.jl)

### ❌ NOT Implemented in Rust:

#### 2.1 **Slicer data structure tests** (lines 8-20)
- **What it does**: Tests `Slicer` object for managing sliced indices
- **Julia code**:
  ```julia
  s = Slicer(log2_sizes, [])
  push!(s, 1)
  replace!(s, 1=>4)
  @test s.legs == Dict(2=>2.0, 3=>3.0, 4=>4.0)
  ```
- **Missing in Rust**: No equivalent low-level `Slicer` object tests
- **Importance**: **LOW** - internal data structure
- **Recommendation**: TreeSA logic is tested at higher level, not critical

#### 2.2 **`ExprTree` random generation tests** (lines 22-41)
- **What it does**: Tests random expression tree generation
- **Julia code**:
  ```julia
  tree = random_exprtree([[1,2,5], [2,3], [2,4]], [5], 5, TreeDecomp())
  @test tree isa ExprTree
  ```
- **Missing in Rust**: No `random_exprtree` function in Rust
- **Importance**: **LOW** - Julia uses ExprTree internally, Rust uses NestedEinsum directly
- **Recommendation**: Not needed (architectural difference)

#### 2.3 **Tree transformation rules tests** (lines 43-73)
- **What it does**: Tests tree rewriting rules for SA optimization
- **Julia code**:
  ```julia
  @test ruleset(TreeDecomp(), t1) == 3:4
  @test ruleset(PathDecomp(), t2) == 1:2
  t11 = update_tree!(copy(t1), 3, [2])
  ```
- **Missing in Rust**: No exposed `ruleset` or `update_tree!` functions
- **Importance**: **LOW** - internal SA implementation detail
- **Recommendation**: Rules are tested indirectly through SA optimization

#### 2.4 **`tcscrw` computation tests** (lines 58-62)
- **What it does**: Tests time/space/read-write complexity calculation
- **Julia code**:
  ```julia
  @test all(_tcsc(t1, log2_sizes) .≈ (2.0, 1.0, log2(10)))
  ```
- **Missing in Rust**: No direct unit tests for complexity calculation on trees
- **Importance**: **MEDIUM** - validates complexity metrics
- **Recommendation**: Add unit tests for `tree_complexity` calculation

#### 2.5 **`optimize_subtree!` tests** (lines 75-98)
- **What it does**: Tests local subtree optimization
- **Julia code**:
  ```julia
  optimize_subtree!(opt_tree, 100.0, log2_sizes, 5, 2.0, 1.0, TreeDecomp())
  @test sc1 < sc0 || (sc1 == sc0 && tc1 < tc0)
  ```
- **Missing in Rust**: No `optimize_subtree` function exposed
- **Importance**: **LOW** - internal SA implementation
- **Recommendation**: Not needed (internal detail)

#### 2.6 **`optimize_tree_sa!` tests** (lines 100-118)
- **What it does**: Tests low-level tree SA optimization
- **Julia code**:
  ```julia
  optimize_tree_sa!(opttree, log2_sizes; βs=0.1:0.1:10.0, niters=100, ...)
  @test sc1 < sc0
  ```
- **Missing in Rust**: No low-level `optimize_tree_sa!` exposed
- **Importance**: **LOW** - internal implementation
- **Recommendation**: Tested via `optimize_treesa` at high level

#### 2.7 **Actual tensor contraction validation** (lines 138-158)
- **What it does**: **CRITICAL** - Performs actual numerical tensor contractions to verify correctness
- **Julia code**:
  ```julia
  xs = [[2*randn(2, 2) for i=1:75]..., [randn(2) for i=1:50]...]
  resg = decorate(codeg)(xs...)  # Execute greedy contraction
  resk = decorate(codek)(xs...)  # Execute TreeSA contraction
  @test resg ≈ resk              # Verify numerical equality
  ```
- **Missing in Rust**: **YES - This is a GAP!**
- **Importance**: **HIGH** - End-to-end numerical validation
- **Current Rust status**: We have `NaiveContractor` in `test_utils.rs` for greedy tests, but NOT used for TreeSA
- **Recommendation**: **ADD THIS!** Use `NaiveContractor` to validate TreeSA results match greedy results numerically

#### 2.8 **`fast_log2sumexp2` tests** (lines 161-165)
- **What it does**: Tests fast log-sum-exp computation
- **Julia code**:
  ```julia
  @test fast_log2sumexp2(a, b) ≈ log2(sum(exp2.([a,b])))
  ```
- **Missing in Rust**: No `fast_log2sumexp2` function
- **Importance**: **LOW** - utility function
- **Recommendation**: Not critical unless performance optimization needed

#### 2.9 **`is_path_decomposition` validation** (lines 167-180)
- **What it does**: Tests that PathDecomp produces valid path decompositions
- **Julia code**:
  ```julia
  @test !is_path_decomposition(codek)  # Greedy doesn't guarantee path
  @test is_path_decomposition(codeg)   # PathDecomp guarantees path
  ```
- **Missing in Rust**: No `is_path_decomposition` validation function
- **Importance**: **MEDIUM** - validates PathDecomp correctness
- **Recommendation**: Add validation function for PathDecomp mode

---

## 3. Cross-Optimizer Validation (interfaces.jl)

### ❌ CRITICAL MISSING TEST:

#### 3.1 **Cross-optimizer numerical validation** (interfaces.jl lines 6-73)
- **What it does**: **MOST IMPORTANT TEST** - Runs multiple optimizers and verifies they all produce the same numerical result
- **Julia code**:
  ```julia
  for optimizer in [TreeSA(ntrials=1), GreedyMethod(), SABipartite(), HyperND()]
      res = optimize_code(code, uniformsize(code, 2), optimizer)
      push!(results, res(xs...)[])  # Execute actual contraction
  end
  for i=1:length(results)-1
      @test results[i] ≈ results[i+1]  # All results must match!
  end
  ```
- **Missing in Rust**: **YES - MAJOR GAP!**
- **Importance**: **CRITICAL** - Validates that all optimizers produce correct results
- **Current Rust status**: No cross-optimizer validation
- **Recommendation**: **MUST ADD** - Use `NaiveContractor` to validate:
  - Greedy result == TreeSA result (numerically)
  - Different TreeSA configurations produce same result
  - Sliced results == unsliced results (when summed over slices)

#### 3.2 **Peak memory computation tests** (interfaces.jl lines 86-99)
- **What it does**: Tests `peak_memory` calculation (actual memory, not log2)
- **Julia code**:
  ```julia
  @test peak_memory(code, uniformsize(code, 5)) == 75
  @test 10 * 2^sc1 > pm1 > 2^sc1  # Peak memory related to space complexity
  ```
- **Missing in Rust**: Have `peak_memory` function but no unit tests
- **Importance**: **MEDIUM** - validates memory calculation
- **Recommendation**: Add unit tests for `peak_memory` function

---

## 4. Summary: Most Critical Missing Tests

### 🔴 **CRITICAL (Must Add)**:

1. **Cross-optimizer numerical validation**
   - File: `interfaces.jl` lines 6-73
   - **Action**: Add tests that validate Greedy and TreeSA produce the same numerical results
   - **Implementation**: Use existing `NaiveContractor` from `test_utils.rs`

2. **TreeSA numerical validation with actual tensors**
   - File: `treesa.jl` lines 138-158
   - **Action**: Add tests that execute TreeSA with actual tensor data and verify against greedy
   - **Implementation**: Use `NaiveContractor` similar to greedy tests

### 🟡 **HIGH PRIORITY (Should Add)**:

3. **`flatten()` / `to_eincode()` round-trip**
   - File: `greedy.jl` lines 73-108
   - **Action**: Add `NestedEinsum::to_eincode()` method and tests
   - **Benefit**: Verification and debugging utility

4. **`is_path_decomposition()` validation**
   - File: `treesa.jl` lines 167-180
   - **Action**: Add function to validate PathDecomp produces valid paths
   - **Benefit**: Ensures PathDecomp correctness

5. **Unit tests for `compute_contraction_output_with_hypergraph`**
   - File: `greedy.jl` lines 194-242
   - **Action**: Add comprehensive unit tests for dimension computation
   - **Benefit**: Validates core correctness

### 🟢 **MEDIUM PRIORITY (Nice to Have)**:

6. **`nested_flop()` explicit tests**
   - File: `greedy.jl` lines 48-50
   - **Action**: Add unit tests for FLOP calculation
   - **Benefit**: Validates time complexity metric

7. **`peak_memory()` unit tests**
   - File: `interfaces.jl` lines 86-99
   - **Action**: Add tests for peak memory calculation
   - **Benefit**: Validates memory metric

8. **Loss function unit tests**
   - File: `greedy.jl` lines 244-270
   - **Action**: Add tests for greedy loss computation
   - **Benefit**: Validates cost function accuracy

9. **Tree complexity (`tcscrw`) unit tests**
   - File: `treesa.jl` lines 58-62
   - **Action**: Add tests for tree complexity calculation
   - **Benefit**: Validates SA complexity metrics

### ⚪ **LOW PRIORITY (Not Critical)**:

10. Low-level internal function tests (analyze_contraction, contract_pair!, etc.)
11. ExprTree-specific tests (Rust uses different architecture)
12. Julia-specific utilities (parse_tree, fast_log2sumexp2, etc.)

---

## 5. Recommended Action Plan

### Phase 1: Critical Numerical Validation (Do This Now!)
```rust
// In greedy.rs or integration tests
#[test]
fn test_cross_optimizer_numerical_validation() {
    // Test case: 50-node 3-regular graph
    let (ixs, out) = generate_random_regular_eincode(50, 3, 42);
    let sizes = uniform_size_dict(&EinCode::new(ixs.clone(), out.clone()), 2);

    // Create random tensors
    let mut contractor = NaiveContractor::default();
    for (idx, ix) in ixs.iter().enumerate() {
        let shape: Vec<usize> = ix.iter().map(|&label| sizes[&label]).collect();
        contractor.add_tensor(idx, shape);
    }

    // Optimize with Greedy
    let greedy_result = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();
    let greedy_value = execute_nested(&greedy_result, &mut contractor.clone());

    // Optimize with TreeSA
    let treesa_result = optimize_treesa(&code, &sizes, &TreeSA::fast()).unwrap();
    let treesa_value = execute_nested(&treesa_result, &mut contractor.clone());

    // Verify numerical equality
    assert_tensors_approx_equal(&greedy_value, &treesa_value);
}
```

### Phase 2: Utility Functions
- Add `NestedEinsum::to_eincode()` method
- Add `is_path_decomposition()` validation
- Add unit tests for `nested_flop()` and `peak_memory()`

### Phase 3: Internal Validation
- Add unit tests for `compute_contraction_output_with_hypergraph`
- Add unit tests for loss function computation
- Add unit tests for tree complexity calculation

---

## 6. Test Coverage Summary

| Test Category | Julia Tests | Rust Tests | Coverage % | Priority |
|--------------|-------------|------------|------------|----------|
| **Greedy basic** | 9 tests | 25 tests | ✅ 100%+ | - |
| **Greedy numerical validation** | 3 tests | 8 tests | ✅ 100%+ | - |
| **TreeSA basic** | 8 tests | 27 tests | ✅ 100%+ | - |
| **TreeSA numerical validation** | 3 tests | 0 tests | ❌ 0% | 🔴 CRITICAL |
| **Cross-optimizer validation** | 1 test | 0 tests | ❌ 0% | 🔴 CRITICAL |
| **Utility functions** | 5 tests | 0 tests | ❌ 0% | 🟡 HIGH |
| **Internal functions** | 15 tests | 2 tests | 🟡 13% | 🟢 MEDIUM |

**Overall Assessment**:
- ✅ Rust has EXCELLENT coverage for basic functionality and edge cases
- ❌ Missing CRITICAL numerical validation across optimizers
- 🟡 Missing some utility functions for debugging/verification
