# JPMC CCAR Analyst II - Interview Practice Questions
**Mixed CCAR/Banking + General Statistics (50/50)**
**Format: Story-based, progressive difficulty with follow-ups**

---

## Topic 1: Correlation of Residuals

### Question Story 1: Credit Card Default Model
**Q1.1:** You build a linear regression model to predict credit card default amounts for HDFC Bank customers. After fitting the model, you plot residuals against fitted values and notice a pattern. What does correlation in residuals indicate about your model?

**Answer:** Correlation (autocorrelation) in residuals violates the independence assumption of linear regression. It suggests that errors are not random and contain information that the model hasn't captured. This means the model is misspecified - perhaps missing important predictors, wrong functional form, or temporal dependencies.

**Q1.2:** You check the Durbin-Watson statistic and get a value of 0.85. What does this tell you?

**Answer:** Durbin-Watson ranges from 0 to 4, with 2 indicating no autocorrelation. A value of 0.85 (<<2) indicates strong positive autocorrelation in residuals. Consecutive residuals are positively correlated, meaning if one residual is positive, the next is likely positive too.

**Q1.3:** Why is this particularly problematic for CCAR stress testing models?

**Answer:** In CCAR, we forecast losses under stress scenarios over 9-13 quarters. If residuals are autocorrelated:
1. Standard errors are underestimated, making confidence intervals too narrow
2. We underestimate forecast uncertainty
3. Regulatory capital calculations based on these forecasts will be incorrect
4. The model may fail Federal Reserve model validation requirements

**Q1.4:** You suspect the autocorrelation is due to quarterly seasonality in defaults. How would you test this hypothesis formally?

**Answer:** Use the Ljung-Box Q-test at lag 4 (since quarterly data):
- H₀: No autocorrelation up to lag 4
- H₁: At least one autocorrelation ≠ 0
- Calculate Q = n(n+2) Σ(ρ²ₖ/(n-k)) for k=1 to 4
- Compare to χ² distribution with 4 degrees of freedom
- If p < 0.05, reject H₀ and conclude seasonal autocorrelation exists

**Q1.5:** What's the difference between Durbin-Watson test and Breusch-Godfrey test for autocorrelation?

**Answer:** 
- **Durbin-Watson**: Only tests for first-order (lag 1) autocorrelation, assumes regressors are strictly exogenous, not valid with lagged dependent variables
- **Breusch-Godfrey**: Tests for higher-order autocorrelation (any lag), valid even with lagged dependent variables and other ARMA errors, more general and flexible
For CCAR models with lagged variables, use Breusch-Godfrey.

**Q1.6:** If you find significant autocorrelation, what remedies would you apply?

**Answer:**
1. **Add lagged dependent variable**: Y_t = β₀ + β₁X_t + β₂Y_{t-1} + ε_t
2. **Include omitted variables**: Time trends, seasonality dummies, macroeconomic indicators
3. **Use Cochrane-Orcutt or Prais-Winsten transformation**: Estimate ρ from residuals, transform data
4. **Use HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors**: Newey-West estimator
5. **Switch to time series models**: ARIMA, VAR models

**Q1.7:** After adding a lagged dependent variable, your Durbin-Watson is now 2.1, but you still see some structure in residual ACF plot at lag 4. What's happening?

**Answer:** The first-order autocorrelation is fixed (DW ≈ 2), but there's still seasonal autocorrelation at lag 4. This is common in quarterly financial data. You should:
1. Add seasonal dummy variables (Q1, Q2, Q3)
2. Or include the lag-4 dependent variable: Y_t = β₀ + β₁X_t + β₂Y_{t-1} + β₃Y_{t-4} + ε_t
3. Re-run Breusch-Godfrey test at lag 4 to verify

**Q1.8:** The model now passes all autocorrelation tests. However, during Federal Reserve's SR 11-7 model validation, they ask: "Why didn't you use a time series model instead?" How do you respond?

**Answer:** Valid response should cover:
1. **Interpretability**: Linear regression with lagged terms maintains clear economic interpretation - we can explain how specific risk factors (unemployment, interest rates) impact defaults
2. **CCAR requirement**: Fed requires conditional forecasting based on specific macroeconomic scenarios, which is more natural in regression framework
3. **Trade-off**: Pure time series models (ARIMA) forecast well but are "black boxes" - regulators need economically meaningful coefficients for stress testing
4. **Hybrid approach**: Our model is actually a distributed lag model / ADL (autoregressive distributed lag), combining benefits of both approaches
5. **Validation**: Show diagnostic tests confirm no remaining autocorrelation, heteroskedasticity, or other violations

---

## Topic 2: Stationarity of Data

### Question Story 2: Loan Loss Provisioning Model

**Q2.1:** You're modeling net charge-off rates (NCO) for JPMorgan's consumer loan portfolio for CCAR. Your colleague says "always check stationarity first." What is stationarity and why does it matter?

**Answer:** **Stationarity** means statistical properties (mean, variance, autocorrelation) don't change over time. Specifically:
- **Weak/covariance stationarity**: Constant mean E(Y_t) = μ, constant variance Var(Y_t) = σ², autocorrelation depends only on lag not time Cov(Y_t, Y_{t-k}) = γ_k
- **Strong stationarity**: Entire distribution unchanged over time
**Why it matters**: Most time series models (ARIMA, regression with time series) assume stationarity. Non-stationary series lead to:
1. Spurious regressions (high R² but meaningless relationships)
2. Invalid hypothesis tests
3. Poor out-of-sample forecasts

**Q2.2:** You plot NCO rates from 2000-2025. You see mean around 1% pre-2008, spikes to 7% during 2008-09, returns to 1.5% afterward, and has increasing variance since 2020. Is this stationary?

**Answer:** **No, it's non-stationary** on multiple fronts:
1. **Mean non-stationarity**: Mean shifts across different periods (1% → 7% → 1.5%)
2. **Variance non-stationarity (heteroskedastic)**: Increasing variance post-2020
3. **Structural breaks**: 2008 financial crisis represents regime change
This is typical for financial time series - they exhibit trend, structural breaks, and time-varying volatility.

**Q2.3:** What is the Augmented Dickey-Fuller (ADF) test and how do you implement it?

**Answer:** **ADF test** checks for unit root (non-stationarity). 
**Test equation**: ΔY_t = α + βt + γY_{t-1} + Σδᵢ ΔY_{t-i} + ε_t
- **H₀**: γ = 0 (unit root exists, non-stationary)
- **H₁**: γ < 0 (no unit root, stationary)

**Implementation**:
1. Choose specification: none, constant only, or constant + trend
2. Select lag length using AIC/BIC
3. Calculate test statistic τ = γ̂ / SE(γ̂)
4. Compare to ADF critical values (NOT standard t-distribution)
5. If τ < critical value (or p < 0.05), reject H₀ → series is stationary

**Key**: Use "constant + trend" specification for trending series like GDP; "constant only" for series fluctuating around constant mean.

**Q2.4:** You run ADF test on NCO rates:
- Lag length (AIC): 4
- Test statistic: τ = -2.15
- Critical value (5%): -2.89
- p-value: 0.22

What do you conclude and what are the implications for your CCAR model?

**Answer:** 
**Conclusion**: τ = -2.15 > -2.89, p = 0.22 > 0.05 → Fail to reject H₀ → **NCO series is non-stationary** (has unit root).

**Implications**:
1. Cannot use NCO in levels for regression - will get spurious results
2. Need to difference the series: ΔNCO_t = NCO_t - NCO_{t-1}
3. Or use cointegration framework if modeling relationship with other non-stationary variables
4. For CCAR, this means modeling **change in NCO** rather than level, or establishing cointegrating relationships with macro variables

**Q2.5:** What's the difference between ADF and Phillips-Perron (PP) test?

**Answer:**
- **ADF**: Parametric test, controls for serial correlation by adding lagged difference terms ΔY_{t-i}, requires choosing lag length, assumes homoskedastic errors
- **PP**: Non-parametric test, uses Newey-West HAC correction for serial correlation and heteroskedasticity, no need to specify lags, more robust to heteroskedasticity and general error structures
**When to use which**: Use PP when you suspect heteroskedasticity in errors (common in financial data with volatility clustering). Use ADF when you want explicit control over autoregressive structure.

**Q2.6:** Your manager asks about the KPSS test. How is it different from ADF/PP, and why would you use it?

**Answer:** **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)** test reverses the null hypothesis:
- **H₀**: Series is stationary
- **H₁**: Series has unit root (non-stationary)

**Difference from ADF/PP**:
| Test | H₀ | Power |
|------|----|----|
| ADF/PP | Non-stationary | Good at detecting non-stationarity |
| KPSS | Stationary | Good at detecting stationarity |

**Why use both**: 
- If ADF says stationary AND KPSS says stationary → **Definitely stationary**
- If ADF says non-stationary AND KPSS says non-stationary → **Definitely non-stationary**
- If they disagree → Data may be near unit root boundary, need more investigation

**Best practice for CCAR**: Run both tests for robustness.

**Q2.7:** You difference NCO once: ΔNCO_t = NCO_t - NCO_{t-1}. New ADF test gives τ = -5.87 (p < 0.01). What does this mean, and how do you interpret the order of integration?

**Answer:**
**Conclusion**: After first differencing, series is now stationary → Original NCO series is **I(1)** (integrated of order 1).

**Order of integration I(d)**:
- **I(0)**: Stationary in levels, no differencing needed
- **I(1)**: Stationary after first difference, most economic/financial series
- **I(2)**: Needs two differences (rare, except some price indices)

**Interpretation for NCO**: 
- NCO level has unit root (random walk component)
- Changes in NCO (ΔNCO) are stationary
- Shocks to NCO have permanent effects on the level
- For modeling: Use ΔNCO as dependent variable, or find cointegrating relationships

**Q2.8:** During 2020 COVID crisis, RBI decreased repo rate from 5.15% to 4% despite supply shocks. You're modeling relationship between repo rate and loan defaults. Both series are I(1). Your regression in levels gives R² = 0.78. Should you trust it?

**Answer:** **No, likely spurious regression.** Here's the issue:

**Spurious regression problem**: Two I(1) series may show high R² even if unrelated, because both trend over time. Standard errors are invalid, t-stats are inflated.

**What to do**:
1. **Test for cointegration**: Use Engle-Granger or Johansen test
   - If cointegrated → Regression in levels is valid (captures long-run equilibrium)
   - If not cointegrated → Must use first differences or other specification
2. **Error Correction Model (ECM)**: If cointegrated, model short-run dynamics:
   Δdefault_t = α + β₁Δrepo_t + β₂(default_{t-1} - γrepo_{t-1}) + ε_t
   
**COVID repo rate example**: 
- Supply shock → inflation should rise → repo should rise
- But RBI decreased repo to support liquidity (policy override)
- This is a **structural break** - relationship changed
- Need dummy variable for COVID period or break tests (Chow test, Bai-Perron)

**Q2.9:** You run Engle-Granger cointegration test:
Step 1: Regress defaults_t = α + βrepo_t + u_t
Step 2: ADF test on residuals û_t gives τ = -3.75, critical value = -3.37
What do you conclude?

**Answer:**
**Conclusion**: τ = -3.75 < -3.37 → Reject H₀ of no cointegration → **defaults and repo rate are cointegrated**.

**Interpretation**:
1. Despite both being I(1), they share a common stochastic trend
2. There exists a long-run equilibrium relationship: defaults_t = α + βrepo_t
3. Deviations from equilibrium (û_t) are temporary and stationary
4. Regression in levels is valid and not spurious
5. Can build ECM to model both long-run equilibrium and short-run adjustments

**For CCAR**: Cointegration is crucial for long-horizon forecasts. If variables are cointegrated, they won't drift apart over 9-quarter stress horizon.

**Q2.10:** What is the intuition behind why differencing removes trend and achieves stationarity?

**Answer:**
**Intuition**: 
- Non-stationary I(1) series: Y_t = Y_{t-1} + ε_t (random walk) or Y_t = μ + Y_{t-1} + ε_t (random walk with drift)
- This accumulates all past shocks: Y_t = Y₀ + Σε_i → Mean changes over time, variance grows
- First difference: ΔY_t = Y_t - Y_{t-1} = μ + ε_t → Just the shock term (stationary)

**Trend removal**:
- Deterministic trend: Y_t = α + βt + stationary → Differencing also works
- Differencing removes both stochastic trends (unit roots) and deterministic trends

**Trade-off**: Differencing removes long-run information. If interested in levels, use cointegration instead.

---

## Topic 3: Sorting Algorithms - QuickSort

### Question Story 3: CCAR Data Processing

**Q3.1:** You have unsorted loan IDs that need to be sorted for efficient lookup during stress testing calculations. Why is QuickSort O(n log n) on average?

**Answer:**
**QuickSort algorithm**:
1. Choose pivot element
2. Partition array: elements < pivot go left, elements > pivot go right
3. Recursively sort left and right subarrays

**Complexity derivation**:
- **Partitioning step**: O(n) - single pass through array
- **Recursive depth**: log₂(n) for balanced partitions (each split divides array in half)
- **Total**: O(n) per level × O(log n) levels = **O(n log n)**

**Pseudocode**:
```
QuickSort(arr, low, high):
    if low < high:
        pivot_index = Partition(arr, low, high)
        QuickSort(arr, low, pivot_index - 1)
        QuickSort(arr, pivot_index + 1, high)

Partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j = low to high - 1:
        if arr[j] <= pivot:
            i = i + 1
            swap arr[i] and arr[j]
    swap arr[i + 1] and arr[high]
    return i + 1
```

**Q3.2:** What is QuickSort's worst-case complexity and when does it occur?

**Answer:**
**Worst case**: **O(n²)**

**When it occurs**:
1. Array already sorted (ascending or descending)
2. Pivot always picks smallest/largest element
3. Results in maximally unbalanced partitions: one side has n-1 elements, other has 0

**Why O(n²)**:
- First call: partition n elements
- Second call: partition n-1 elements
- ...
- Total: n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = **O(n²)**

**Real-world impact**: If CCAR data is pre-sorted by date and you use first/last element as pivot, QuickSort degrades badly.

**Q3.3:** How do you prevent worst-case behavior in QuickSort?

**Answer:**
**Prevention strategies**:
1. **Randomized pivot**: Randomly select pivot instead of first/last → Expected O(n log n) for any input
2. **Median-of-three**: Choose median of first, middle, and last elements as pivot → Better for partially sorted data
3. **Three-way partitioning**: Partition into <pivot, =pivot, >pivot → Efficient for many duplicates

**Best for CCAR**: Median-of-three, because financial data often has natural ordering (dates) and many duplicates (loan types, ratings).

**Modified pseudocode**:
```
MedianOfThree(arr, low, high):
    mid = (low + high) / 2
    if arr[mid] < arr[low]:
        swap arr[low] and arr[mid]
    if arr[high] < arr[low]:
        swap arr[low] and arr[high]
    if arr[mid] < arr[high]:
        swap arr[mid] and arr[high]
    return arr[high]  // median now at high position
```

**Q3.4:** What is the space complexity of QuickSort and why?

**Answer:**
**Space complexity**: 
- **Best/Average case**: **O(log n)** - recursion stack depth
- **Worst case**: **O(n)** - maximally unbalanced recursion tree

**Why recursion stack**:
- Each recursive call stores: function parameters, local variables, return address
- Depth of recursion = height of partition tree
- Balanced tree: height = log n
- Unbalanced tree: height = n

**Optimization**: Tail recursion elimination - always recurse on smaller partition first, iterate on larger:
```
QuickSort_Optimized(arr, low, high):
    while low < high:
        pivot_index = Partition(arr, low, high)
        if (pivot_index - low) < (high - pivot_index):
            QuickSort_Optimized(arr, low, pivot_index - 1)
            low = pivot_index + 1
        else:
            QuickSort_Optimized(arr, pivot_index + 1, high)
            high = pivot_index - 1
```
This guarantees **O(log n)** space even in worst case.

**Q3.5:** You need to sort 10 million loan records (each 1 KB) by default probability. Your system has limited memory. Should you use QuickSort? What are the alternatives?

**Answer:**
**QuickSort considerations**:
- **In-place sorting**: Uses O(log n) extra space ✓
- **Not stable**: Equal elements may change relative order ✗ (matters if secondary sort needed)
- **Cache efficiency**: Good locality of reference ✓

**Better alternative: External sorting (Multi-way MergeSort)**
For 10 million × 1 KB = 10 GB data exceeding RAM:
1. **Split**: Divide data into chunks that fit in memory
2. **Sort**: Sort each chunk using QuickSort or MergeSort
3. **Merge**: K-way merge sorted chunks (reading from disk)

**Why not QuickSort alone**: QuickSort isn't naturally suited to external sorting (disk I/O), while MergeSort's merge operation is I/O efficient.

**Practical CCAR approach**: Use database sorting (SQL ORDER BY) - databases implement hybrid algorithms optimized for large data.

**Q3.6:** Compare QuickSort vs MergeSort vs HeapSort for your CCAR application:

**Answer:**
| Algorithm | Avg Time | Worst Time | Space | Stable | Cache Perf | Use Case |
|-----------|----------|------------|-------|--------|-----------|----------|
| QuickSort | O(n log n) | O(n²) | O(log n) | No | Excellent | General purpose, in-memory |
| MergeSort | O(n log n) | O(n log n) | O(n) | Yes | Good | Need stability, external sort |
| HeapSort | O(n log n) | O(n log n) | O(1) | No | Poor | Memory constrained, guaranteed worst-case |

**For CCAR loan data**:
- **If memory available + speed critical**: QuickSort (with random pivot)
- **If need stable sort** (preserve loan ordering within same default prob): MergeSort
- **If need guaranteed worst-case + limited space**: HeapSort
- **Reality**: Use `std::sort()` (C++ STL) which implements **IntroSort** = QuickSort + HeapSort fallback + InsertionSort for small arrays

**Q3.7:** Your QuickSort is running slower than expected on CCAR portfolio data with many loans having identical credit scores. What's happening and how do you fix it?

**Answer:**
**Problem**: Many duplicates cause **O(n²) behavior** with standard two-way partitioning.

**Why**: If many elements equal pivot:
- Standard partition puts all equals on one side
- Creates unbalanced partitions even with good pivot selection
- Example: Array of all identical elements → O(n²)

**Solution: Dutch National Flag (Three-way partitioning)**
Partition into three sections: < pivot, = pivot, > pivot
```
ThreeWayPartition(arr, low, high):
    pivot = arr[high]
    i = low          // boundary of < region
    j = low          // current element
    k = high         // boundary of > region
    
    while j <= k:
        if arr[j] < pivot:
            swap arr[i] and arr[j]
            i++; j++
        else if arr[j] > pivot:
            swap arr[j] and arr[k]
            k--
        else:  // arr[j] == pivot
            j++
    
    return (i, k)  // boundaries of = region

QuickSort_ThreeWay(arr, low, high):
    if low < high:
        (lt, gt) = ThreeWayPartition(arr, low, high)
        QuickSort_ThreeWay(arr, low, lt - 1)
        QuickSort_ThreeWay(arr, gt + 1, high)
```

**Result**: Elements equal to pivot are already in correct position, not recursed upon. **O(n log k)** where k = number of distinct elements. For high duplicates, much faster.

**Q3.8:** During CCAR scenario processing, you need to sort loan data by multiple keys: (credit_score, loan_amount, loan_id). How would you implement this efficiently?

**Answer:**
**Approach 1: Composite key comparison**
Create comparison function that compares lexicographically:
```
Compare(loan1, loan2):
    if loan1.credit_score != loan2.credit_score:
        return loan1.credit_score < loan2.credit_score
    if loan1.loan_amount != loan2.loan_amount:
        return loan1.loan_amount < loan2.loan_amount
    return loan1.loan_id < loan2.loan_id
    
QuickSort(arr, low, high, Compare)
```
**Complexity**: Still O(n log n), but comparison cost is O(k) for k keys → O(kn log n)

**Approach 2: Radix approach (stable sorts)**
Sort by least significant key first, working backwards:
```
StableSort(arr, by=loan_id)
StableSort(arr, by=loan_amount)
StableSort(arr, by=credit_score)
```
Requires stable sort (MergeSort). Total: O(kn log n)

**Best practice**: Use Approach 1 with IntroSort. Modern compilers optimize comparison chains efficiently.

**Q3.9:** You're asked to find the median default probability (50th percentile) from unsorted CCAR data. Do you need O(n log n) sorting?

**Answer:**
**No!** Can use **QuickSelect** - finds k-th smallest element in **O(n) average time**.

**QuickSelect algorithm**:
```
QuickSelect(arr, low, high, k):
    if low == high:
        return arr[low]
    
    pivot_index = Partition(arr, low, high)
    
    if k == pivot_index:
        return arr[k]
    else if k < pivot_index:
        return QuickSelect(arr, low, pivot_index - 1, k)
    else:
        return QuickSelect(arr, pivot_index + 1, high, k)

FindMedian(arr):
    n = length(arr)
    return QuickSelect(arr, 0, n-1, n/2)
```

**Complexity**:
- **Average**: O(n) - each recursion eliminates half the array, n + n/2 + n/4 + ... = 2n
- **Worst**: O(n²) - same as QuickSort worst case

**For CCAR percentile calculations** (P50, P90, P99): QuickSelect is much faster than full sort when you only need specific percentiles.

**Q3.10:** The Federal Reserve asks you to explain why your CCAR processing system uses QuickSort instead of MergeSort, given that MergeSort has guaranteed O(n log n) worst case. How do you justify it?

**Answer:**
**Justification**:

1. **Practical performance**: QuickSort's average case **O(n log n)** has smaller constant factors than MergeSort. Real-world speedup: 2-3× faster.

2. **Memory efficiency**: QuickSort is in-place O(log n) space vs MergeSort O(n) space. For large CCAR datasets, memory matters.

3. **Cache performance**: QuickSort has better locality - partition works on contiguous memory. MergeSort requires auxiliary array, worse cache behavior. Cache misses are expensive in modern hardware.

4. **Worst-case mitigation**: Using randomized pivot or median-of-three makes O(n²) extremely unlikely (probability < 1/n!). With tail recursion elimination, stack overflow also prevented.

5. **Hybrid approach**: Production systems use **IntroSort** (C++ std::sort):
   - Start with QuickSort
   - If recursion depth exceeds 2 log n → switch to HeapSort (guarantees O(n log n))
   - For small subarrays (< 16 elements) → InsertionSort (O(n²) but low overhead)
   
6. **Benchmark results**: Show empirical tests on CCAR data demonstrating QuickSort (with optimizations) outperforms MergeSort by 40-60% in wall-clock time.

**Conclusion**: QuickSort with proper implementation is the pragmatic choice balancing speed, memory, and worst-case guarantees for production CCAR systems.

---

## Topic 4: Linear Regression Assumptions & Tests

### Question Story 4: Mortgage Default Rate Model

**Q4.1:** You're building a linear regression model for mortgage default rates using unemployment rate, interest rate, and HPI (Home Price Index) as predictors. What are the classical linear regression assumptions?

**Answer:**
**Classical OLS assumptions** (Gauss-Markov):

1. **Linearity**: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε (linear in parameters)
2. **Independence**: Observations are independent (no autocorrelation in errors)
3. **Homoskedasticity**: Constant error variance Var(ε) = σ² for all i
4. **No perfect multicollinearity**: No predictor is perfect linear combination of others
5. **Exogeneity**: E(ε|X) = 0, errors uncorrelated with predictors
6. **Normality** (for inference): ε ~ N(0, σ²)

**Gauss-Markov Theorem**: Under assumptions 1-5, OLS estimators are BLUE (Best Linear Unbiased Estimators).

**Q4.2:** Your model has R² = 0.85, all coefficients are significant (p < 0.01), but your manager says "check the assumptions first." Why?

**Answer:**
High R² and significant coefficients **do not guarantee valid model**. If assumptions are violated:

1. **Coefficient estimates may be biased** (e.g., omitted variable bias, measurement error)
2. **Standard errors are wrong** → Confidence intervals and p-values are invalid → False significance
3. **Predictions are unreliable** → Poor out-of-sample performance
4. **Violates CCAR model governance** → Federal Reserve requires diagnostic checks under SR 11-7

**Example**: Autocorrelated errors → Standard errors underestimated → t-stats inflated → Variables appear significant when they're not → Over-confident forecasts → Insufficient capital reserves.

**Best practice**: Always run diagnostic tests before interpreting results.

**Q4.3:** How do you test the linearity assumption?

**Answer:**
**Tests for linearity**:

**1. Visual inspection**:
- Plot Y vs each X → Should show roughly linear relationship
- Plot residuals vs fitted values → Should show random scatter (no pattern)
- Plot residuals vs each predictor → Should show random scatter

**2. Ramsey RESET test** (Regression Equation Specification Error Test):
- **H₀**: Model is correctly specified (linear)
- **H₁**: Nonlinear relationships exist
- **Method**:
  1. Run original model, get fitted values Ŷ
  2. Create powers: Ŷ², Ŷ³, Ŷ⁴
  3. Run auxiliary regression: Y = β₀ + βX + γ₁Ŷ² + γ₂Ŷ³ + γ₃Ŷ⁴ + ε
  4. Test H₀: γ₁ = γ₂ = γ₃ = 0 using F-test
  5. If F-stat significant (p < 0.05) → Reject H₀ → Nonlinearity detected

**3. Component-plus-residual plots** (partial regression plots):
- For each predictor Xⱼ, plot (residuals + βⱼXⱼ) vs Xⱼ
- Should show linear relationship
- Curve indicates nonlinearity for that predictor

**If nonlinearity found**: Add polynomial terms (X²), interactions (X₁X₂), log transformations, or use GAM (Generalized Additive Models).

**Q4.4:** You plot residuals vs fitted values and see a "cone shape" (funnel). What assumption is violated and how do you test it formally?

**Answer:**
**Violation**: **Heteroskedasticity** (non-constant variance). Variance increases with fitted values.

**Visual test**: Residual plot shows systematic pattern (cone, funnel, or other shape) ≠ random scatter.

**Formal tests**:

**1. Breusch-Pagan test**:
- **H₀**: Homoskedasticity (constant variance)
- **H₁**: Heteroskedasticity
- **Method**:
  1. Run original regression, get residuals ê
  2. Calculate squared residuals ê²
  3. Regress ê² on all predictors: ê² = α₀ + α₁X₁ + ... + αₖXₖ + u
  4. Test statistic: LM = nR² ~ χ²(k)
  5. If p < 0.05 → Reject H₀ → Heteroskedasticity present

**2. White test**:
- More general than Breusch-Pagan
- Regress ê² on X₁, X₂, X₁², X₂², X₁X₂ (all predictors, squares, and cross-products)
- LM = nR² ~ χ²(# of regressors in auxiliary model)
- Doesn't assume specific form of heteroskedasticity

**3. Goldfeld-Quandt test**:
- Split sample into low and high fitted value groups
- Compare variances using F-test

**For CCAR**: Use White test (most robust) or Breusch-Pagan with robust standard errors.

**Q4.5:** Your White test gives LM = 45.2, p < 0.001. What are the consequences and how do you fix heteroskedasticity?

**Answer:**
**Consequences of heteroskedasticity**:
1. **OLS estimators still unbiased** ✓
2. **No longer efficient** (not BLUE) - can find better estimators ✗
3. **Standard errors are wrong** → Confidence intervals and hypothesis tests invalid ✗
4. **Forecasts still unbiased** ✓ but prediction intervals are wrong ✗

**Remedies**:

**1. Robust standard errors** (White/Huber standard errors):
- Don't fix heteroskedasticity, just correct standard errors
- Use HC (Heteroskedasticity-Consistent) covariance matrix
- Easiest fix, widely accepted
```
# Robust SE formula
Var(β̂) = (X'X)⁻¹ X' Ω̂ X (X'X)⁻¹
where Ω̂ = diag(ê₁², ê₂², ..., êₙ²)
```

**2. Weighted Least Squares (WLS)**:
- If you know variance structure Var(εᵢ) = σ²wᵢ
- Weight observations: minimize Σ[(yᵢ - ŷᵢ)²/wᵢ]
- Restores efficiency
- Example: If Var(εᵢ) ∝ X₁ᵢ, use weights wᵢ = 1/X₁ᵢ

**3. Transformation**:
- Log transformation: log(Y) often stabilizes variance
- Box-Cox transformation

**4. Generalized Least Squares (GLS)**:
- Estimate variance structure, then apply WLS

**For CCAR models**: Use robust standard errors (option 1) - simplest, doesn't change coefficients, accepted by regulators. Document in model validation report.

**Q4.6:** How do you test for multicollinearity and what are its effects?

**Answer:**
**Multicollinearity**: High correlation between predictors (X variables).

**Detection**:

**1. Correlation matrix**: Look for |r| > 0.8 between predictors
- Only detects pairwise collinearity, misses more complex dependencies

**2. Variance Inflation Factor (VIF)**:
```
VIFⱼ = 1 / (1 - R²ⱼ)
where R²ⱼ = R² from regressing Xⱼ on all other predictors
```
- **Rule of thumb**: VIF > 10 indicates severe multicollinearity (some use VIF > 5)
- **Interpretation**: VIF = 10 means SE of βⱼ is √10 ≈ 3.16 times larger than if Xⱼ uncorrelated with others

**3. Condition number**: κ = √(λₘₐₓ/λₘᵢₙ) of X'X matrix
- κ > 30 suggests multicollinearity

**Effects**:
1. **Coefficients remain unbiased** ✓
2. **Standard errors inflated** → Wide confidence intervals, low t-stats ✗
3. **Coefficients unstable** → Small data changes cause large coefficient changes ✗
4. **Model as a whole may still predict well** (R² unaffected) ✓
5. **Individual coefficients uninterpretable** ✗

**Example**: In mortgage model, if you include both "interest rate" and "mortgage payment" (which depends on interest rate), they're highly correlated → Can't separate their individual effects.

**Q4.7:** Your VIF values are: Unemployment = 2.3, Interest Rate = 8.7, HPI = 9.2. What should you do?

**Answer:**
**Analysis**: Interest Rate and HPI have VIF > 5, indicating multicollinearity (makes sense - both relate to housing market conditions).

**Solutions**:

**1. Remove one predictor**: Drop either Interest Rate or HPI
- **Decision**: Keep the more theoretically important one for CCAR
- **Or**: Test nested models, use AIC/BIC to compare

**2. Combine into index**: Create composite variable
- Example: Housing_Market_Index = weighted average of (Interest Rate, HPI)
- PCA (Principal Component Analysis) can create orthogonal components

**3. Ridge regression** (L2 regularization):
- Adds penalty λΣβⱼ² to minimize variance inflation
- Biases coefficients slightly but reduces variance significantly
- Trade-off controlled by λ

**4. Collect more data**: More observations can reduce standard errors despite collinearity

**5. Do nothing if**:
- Your goal is prediction (not interpretation) → Multicollinearity doesn't hurt predictions
- Coefficients are still significant → Collinearity not severe enough to matter

**For CCAR**: 
- Option 1 (remove one) or Option 3 (ridge regression) most common
- Must document rationale in model documentation
- If using ridge, validate forecasting performance improves

**Q4.8:** Connect hypothesis testing to testing regression assumptions. How are they related?

**Answer:**
**Connection**: Most assumption tests use hypothesis testing framework:

**1. Durbin-Watson (autocorrelation)**:
- H₀: ρ = 0 (no autocorrelation)
- Test statistic: DW ≈ 2(1-ρ̂)
- Compare to DW critical values

**2. Breusch-Pagan (heteroskedasticity)**:
- H₀: Var(ε) = σ² (constant variance)
- Test statistic: LM = nR² ~ χ²(k)
- p-value from chi-square distribution

**3. Ramsey RESET (linearity)**:
- H₀: E(Y|X) is linear
- F-test on added powers of Ŷ

**4. Jarque-Bera (normality)**:
- H₀: ε ~ Normal
- Test statistic: JB = n[(S²/6) + (K-3)²/24] ~ χ²(2)
- S = skewness, K = kurtosis

**5. Chow test (structural break)**:
- H₀: β₁ = β₂ (coefficients stable across periods)
- F-test comparing full vs split models

**Pattern**: 
- Formulate null hypothesis (assumption holds)
- Calculate test statistic from data
- Derive distribution under H₀
- Calculate p-value
- Reject H₀ if p < α (assumption violated)

**CCAR implication**: Every diagnostic test is a hypothesis test. Need to understand type I/II errors, multiple testing issues, power of tests.

**Q4.9:** Your Jarque-Bera test gives JB = 125.3, p < 0.001, indicating non-normal residuals. The histogram shows heavy tails. How concerned should you be for CCAR modeling?

**Answer:**
**Analysis**: Residuals are non-normal with heavy tails (leptokurtic distribution).

**Concerns**:
1. **For coefficient estimates**: Not concerned - OLS is unbiased regardless of normality (by Gauss-Markov)
2. **For hypothesis tests (t, F)**: Somewhat concerned - these assume normality for exact validity
3. **For forecasting**: Very concerned - heavy tails mean more extreme outcomes than normal distribution predicts

**Severity depends on sample size**:
- **Large sample (n > 100)**: Central Limit Theorem kicks in → t and F tests approximately valid even without normality
- **Small sample**: Tests may be invalid

**When normality matters most**:
- Constructing confidence/prediction intervals
- Calculating Value-at-Risk (VaR) or tail probabilities
- CCAR stress scenarios (99th percentile losses)

**Solutions for heavy tails**:

**1. Robust regression**: Use methods less sensitive to outliers
- Huber M-estimator
- Least Absolute Deviation (LAD)

**2. Transform dependent variable**:
- Log transform: log(Y + c)
- Box-Cox transformation

**3. Quantile regression**: Model different quantiles (P90, P95, P99) directly instead of mean
- Perfect for CCAR where we care about tail losses

**4. Check for outliers**: Heavy tails may be due to few extreme values
- Investigate outliers: data errors or true extreme events?
- If true events (like 2008 crisis), keep them - they're precisely what CCAR models!

**For CCAR**: 
- Heavy tails are actually realistic for financial data (fat-tailed distributions)
- Use robust standard errors
- Report tail behavior explicitly
- Consider extreme value theory for tail modeling

**Q4.10:** During model validation, Federal Reserve says "Your model shows significant autocorrelation (DW = 1.15) and heteroskedasticity (White test p < 0.01). The coefficients are still significant. Why is this problematic for CCAR?"

**Answer:**
**Why it's problematic even though coefficients are significant**:

**1. Invalid inference**:
- Autocorrelation → SE underestimated → t-stats too large → **False confidence** in coefficients
- Heteroskedasticity → SE wrong → hypothesis tests unreliable
- Variables may not actually be significant; significance is artifact of violated assumptions

**2. Forecast uncertainty underestimated**:
```
Forecast variance = Var(β̂) + Var(ε)
```
- Both components are wrong if assumptions violated
- CCAR requires accurate loss distributions → Underestimated uncertainty → Insufficient capital buffers

**3. Multi-step ahead forecasts compounding error**:
- CCAR forecasts 9 quarters ahead
- Autocorrelated errors: ε_t correlated with ε_{t-1}, ε_{t-2}, ...
- Forecast errors accumulate and correlate across horizon
- Standard forecast intervals (assuming iid errors) are far too narrow

**4. Scenario analysis validity**:
- CCAR: "What if unemployment rises 4%?"
- If errors correlated with predictors (endogeneity), causal interpretation invalid
- Can't trust model's response to Fed's hypothetical scenarios

**5. Regulatory compliance**:
- SR 11-7 requires "rigorous analysis of model assumptions"
- Fed specifically checks: autocorrelation tests, heteroskedasticity tests, stability tests
- Failing diagnostics = model may be rejected → Bank must revise = costly delay

**Proper response**:
1. "We acknowledge the assumption violations"
2. "We've implemented HAC standard errors (Newey-West) to correct inference"
3. "We added lagged dependent variable to address autocorrelation"
4. "We used White robust SE for heteroskedasticity"
5. "Re-validation shows diagnostics now pass, and forecasting performance improved by X%"
6. "Updated forecast distributions properly reflect uncertainty for capital planning"

**Bottom line**: CCAR is about risk management. Violated assumptions → Underestimated risk → Insufficient capital → Regulatory failure. Must fix, not ignore.

---

## Topic 5: R² and Adjusted R²

### Question Story 5: Model Selection for Credit Risk

**Q5.1:** You build a baseline model predicting credit losses using 3 predictors: unemployment, GDP growth, interest rate. R² = 0.72. Your colleague adds 5 more variables and gets R² = 0.78. Is the second model better?

**Answer:**
**Not necessarily.** R² always increases (or stays same) when you add variables, even if they're irrelevant.

**R² definition**:
```
R² = 1 - (SSR / SST) = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]

Where:
SSR = Sum of Squared Residuals (unexplained variance)
SST = Total Sum of Squares (total variance)
```

**Why R² increases with more variables**:
- Adding variable can't increase SSR (OLS minimizes it)
- SSR stays same or decreases → R² stays same or increases
- True even for random noise variables!

**Problem**: R² doesn't penalize model complexity → Can't use for model comparison with different number of predictors.

**Need**: Adjusted R² or other metrics (AIC, BIC) that penalize complexity.

**Q5.2:** What is adjusted R² and how does it differ from R²?

**Answer:**
**Adjusted R² formula**:
```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]

or equivalently:

Adjusted R² = 1 - [SSR/(n-k-1)] / [SST/(n-1)]
            = 1 - (MSE / MST)

Where:
n = sample size
k = number of predictors (excluding intercept)
MSE = Mean Squared Error
MST = Mean Squared Total
```

**Key differences**:

| Metric | Formula | Range | When adds variable |
|--------|---------|-------|-------------------|
| R² | 1 - SSR/SST | [0, 1] | Always increases |
| Adj R² | 1 - [SSR/(n-k-1)]/[SST/(n-1)] | (-∞, 1] | Increases only if variable improves fit enough to justify complexity |

**Intuition**: Adjusted R² penalizes adding predictors. It increases only if new variable reduces SSR by more than expected by chance.

**Note**: Adjusted R² can be negative (if model fits worse than horizontal line through mean).

**Q5.3:** For the models with R² = 0.72 (k=3) and R² = 0.78 (k=8), if n = 80 observations, which has higher adjusted R²?

**Answer:**
**Model 1** (baseline):
```
Adj R²₁ = 1 - [(1 - 0.72)(80 - 1) / (80 - 3 - 1)]
        = 1 - [0.28 × 79 / 76]
        = 1 - [22.12 / 76]
        = 1 - 0.2911
        = 0.7089
```

**Model 2** (added variables):
```
Adj R²₂ = 1 - [(1 - 0.78)(80 - 1) / (80 - 8 - 1)]
        = 1 - [0.22 × 79 / 71]
        = 1 - [17.38 / 71]
        = 1 - 0.2448
        = 0.7552
```

**Conclusion**: Model 2 has higher adjusted R² (0.7552 > 0.7089), so the 5 additional variables do add sufficient explanatory power despite the complexity penalty.

**However**: Still need to check:
1. Are the new coefficients theoretically meaningful?
2. Out-of-sample performance
3. Multicollinearity (VIF)
4. Assumption diagnostics

Adjusted R² is one criterion, not the only one.

**Q5.4:** What's the minimum increase in R² needed when adding one variable to also increase adjusted R²?

**Answer:**
**Derivation**: Adjusted R² increases when:
```
Adj R²(k+1) > Adj R²(k)

1 - [(1-R²new)(n-1)/(n-k-2)] > 1 - [(1-R²old)(n-1)/(n-k-1)]

(1-R²new)/(n-k-2) < (1-R²old)/(n-k-1)

(1-R²new) < (1-R²old) × (n-k-2)/(n-k-1)

1-R²new < (1-R²old) × [1 - 1/(n-k-1)]

R²new > R²old + (1-R²old)/(n-k-1)
```

**Minimum increase**:
```
ΔR² > (1 - R²old) / (n - k - 1)
```

**Interpretation**: 
- If model already fits well (R²old high), need smaller increase
- If sample size large, need smaller increase
- If already have many predictors, need larger increase

**Example**: n=80, k=3, R²old=0.72
```
ΔR² > (1 - 0.72) / (80 - 3 - 1) = 0.28 / 76 = 0.00368
```
Need at least 0.37% increase in R² to justify adding one more variable.

**Q5.5:** Your CCAR model has Adj R² = 0.68 for in-sample data (2000-2020) but R² = 0.42 for out-of-sample validation (2021-2025). What's happening?

**Answer:**
**Problem**: **Overfitting** - model learned noise in training data rather than true relationships.

**Evidence**:
- Large gap between in-sample and out-of-sample R²
- Model performs much worse on new data

**Causes**:
1. **Too many predictors** relative to sample size (high k/n ratio)
2. **Data snooping**: Selected model after trying many specifications
3. **Structural break**: Relationships changed after 2020 (COVID)
4. **Outliers**: Model fitted to extreme events in training period that don't repeat

**Solutions**:

**1. Reduce model complexity**:
- Use stepwise selection, LASSO, or ridge regression
- Remove insignificant or multicollinear variables

**2. Cross-validation**: 
- K-fold CV on training data to detect overfitting early
- Select model with best CV performance, not training R²

**3. Regularization**:
- Ridge (L2): Penalizes large coefficients
- LASSO (L1): Penalizes and performs variable selection
- Elastic net: Combines both

**4. Test for structural break**:
- Chow test at 2020
- If break exists, include regime dummies or estimate separate models

**5. Increase training sample**: Collect more historical data if possible

**For CCAR**: 
- Out-of-sample validation is required by SR 11-7
- Fed specifically looks for overfitting
- Must demonstrate model stable over time and across scenarios

**Q5.6:** Compare three model selection criteria: Adjusted R², AIC, and BIC. When would each give different answers?

**Answer:**
**Formulas**:

**Adjusted R²**: 1 - [(1-R²)(n-1)/(n-k-1)]

**AIC (Akaike Information Criterion)**: 
```
AIC = 2k + n·ln(SSR/n) = 2k + n·ln(σ̂²)
```
Lower is better. Penalizes complexity by 2k.

**BIC (Bayesian Information Criterion)**:
```
BIC = k·ln(n) + n·ln(SSR/n) = k·ln(n) + n·ln(σ̂²)
```
Lower is better. Penalizes complexity by k·ln(n).

**Comparison**:

| Criterion | Penalty | Behavior | Best for |
|-----------|---------|----------|----------|
| Adj R² | (n-1)/(n-k-1) | Weak penalty | Goodness of fit, large n |
| AIC | 2k | Moderate | Prediction, out-of-sample performance |
| BIC | k·ln(n) | Strong (grows with n) | Parsimony, finding "true" model |

**When they differ**:
- **Large n**: BIC penalty k·ln(n) >> 2k → BIC selects simpler models than AIC
- **Small n**: BIC and AIC similar
- **Prediction focus**: Use AIC
- **True model recovery**: Use BIC (consistent estimator)

**Example**: n=100, comparing k=5 vs k=10
- AIC penalty difference: 2(10) - 2(5) = 10
- BIC penalty difference: 10·ln(100) - 5·ln(100) = 5×4.6 = 23

BIC penalizes the complex model much more heavily.

**For CCAR**: Use AIC for forecasting models (prediction focus) or BIC for risk factor identification (parsimony). Document choice in model documentation.

**Q5.7:** You're reporting to management. Your model has R² = 0.67. They ask "Is 67% good? Should we accept this model?" How do you respond?

**Answer:**
**Response**: "R² alone doesn't determine model quality. Here's the full picture:"

**1. Context matters**:
- **Time series financial data**: R² of 0.4-0.7 is typical and acceptable
  - Financial markets have high inherent randomness
  - Compare: Physics R² often > 0.95, but social sciences R² = 0.3-0.5 common
- **Cross-sectional data**: Higher R² expected
- **Benchmark**: How does R² compare to previous models, industry standards, academic literature?

**2. R² measures in-sample fit, not predictive power**:
- Must evaluate out-of-sample performance
- Can have high R² but poor forecasts (overfitting)
- Can have modest R² but good forecasts (captures key drivers)

**3. Economic significance vs statistical fit**:
- Are coefficients economically meaningful?
- Do they have correct signs (e.g., unemployment ↑ → defaults ↑)?
- Are magnitudes reasonable?

**4. Other evaluation criteria**:
- **Adjusted R²**: Accounts for model complexity
- **AIC/BIC**: Model selection
- **Out-of-sample RMSE, MAE**: Forecast accuracy
- **Directional accuracy**: Does model predict turning points?
- **Assumption diagnostics**: All tests pass?

**5. Regulatory standards**:
- CCAR doesn't specify minimum R²
- Fed evaluates: conceptual soundness, data quality, assumption testing, sensitivity analysis, out-of-sample validation

**6. 33% unexplained variance**:
- Some randomness is inherent and irreducible
- Perfect fit (R²=1) would be suspicious (overfitting)
- Our model captures 67% of systematic variation → Substantial

**Conclusion**: "R² = 0.67 is solid for financial time series. More importantly, the model passes all diagnostic tests, performs well out-of-sample (R²_test = 0.61), coefficients are economically interpretable, and meets regulatory standards. I recommend acceptance with ongoing monitoring."

**Q5.8:** Connect hypothesis testing to R². How can you test if R² is statistically significant?

**Answer:**
**Overall F-test for regression**:

**Hypotheses**:
- H₀: β₁ = β₂ = ... = βₖ = 0 (all slopes zero, R² = 0, model has no explanatory power)
- H₁: At least one βⱼ ≠ 0 (R² > 0, model explains something)

**Test statistic**:
```
F = [R²/k] / [(1-R²)/(n-k-1)]
  = [SSM/k] / [SSR/(n-k-1)]
  = MSM / MSE

Where:
SSM = Sum of Squares Model = Σ(ŷᵢ - ȳ)² = R²·SST
SSR = Sum of Squares Residual = Σ(yᵢ - ŷᵢ)² = (1-R²)·SST
MSM = Mean Square Model
MSE = Mean Square Error
```

**Distribution**: F ~ F(k, n-k-1) under H₀

**Decision**: If p-value < α (usually 0.05), reject H₀ → Model is statistically significant

**Example**: R² = 0.67, k = 5, n = 100
```
F = [0.67/5] / [(1-0.67)/(100-5-1)]
  = [0.134] / [0.33/94]
  = 0.134 / 0.00351
  = 38.2

Compare to F(5, 94) critical value ≈ 2.31 at α=0.05
38.2 >> 2.31 → p < 0.001 → Highly significant
```

**Connection to R²**: The F-test directly tests whether R² is significantly different from zero.

**Note**: This tests overall model significance, not individual coefficients (use t-tests for those).

**Q5.9:** You fit two models: Model A (unemployment, interest rate) with R² = 0.58, and Model B (adds HPI, consumer confidence, industrial production) with R² = 0.67. Test if the additional variables significantly improve the model.

**Answer:**
**Nested model F-test** (also called partial F-test):

**Setup**:
- Restricted model (A): k₁ = 2 predictors, R²₁ = 0.58
- Unrestricted model (B): k₂ = 5 predictors, R²₂ = 0.67
- Additional variables: q = k₂ - k₁ = 3
- Sample size: n = 100 (assumed)

**Hypotheses**:
- H₀: β_HPI = β_confidence = β_production = 0 (restricted model adequate)
- H₁: At least one of the additional βs ≠ 0 (unrestricted model better)

**Test statistic**:
```
F = [(R²₂ - R²₁) / q] / [(1 - R²₂) / (n - k₂ - 1)]
  = [(SSR₁ - SSR₂) / q] / [SSR₂ / (n - k₂ - 1)]
```

**Calculation**:
```
F = [(0.67 - 0.58) / 3] / [(1 - 0.67) / (100 - 5 - 1)]
  = [0.09 / 3] / [0.33 / 94]
  = 0.03 / 0.00351
  = 8.55
```

**Distribution**: F ~ F(3, 94) under H₀

**Critical value**: F(3, 94, 0.05) ≈ 2.70

**Decision**: 8.55 > 2.70, p < 0.001 → Reject H₀ → **The three additional variables significantly improve the model.**

**Interpretation**: The increase in R² from 0.58 to 0.67 (9 percentage points) is statistically significant, not due to chance. Model B is preferred.

**Connection**: This is a direct hypothesis test on whether the improvement in R² is meaningful.

**Q5.10:** During 2020 COVID crisis, repo rates dropped despite inflation concerns. You model repo rate using inflation with R² = 0.72 for 2000-2019, but R² = 0.15 for 2015-2020. How do you explain this change in R² to the Federal Reserve in context of CCAR model validation?

**Answer:**
**Explanation**:

**1. Structural break identification**:
"The dramatic drop in R² (0.72 → 0.15) indicates a structural break in the relationship between inflation and repo rate. This is confirmed by formal testing:"

**Chow test for structural break at 2020**:
```
F = [(SSR_pooled - SSR₁ - SSR₂) / (k+1)] / [(SSR₁ + SSR₂) / (n₁+n₂-2k-2)]

Where:
SSR_pooled = Residual sum of squares from single model (2000-2020)
SSR₁ = RSS from model 1 (2000-2019)
SSR₂ = RSS from model 2 (2020)
```
Result would show highly significant break (p < 0.001).

**2. Economic rationale**:
"The policy regime changed during COVID:
- **Normal times** (2000-2019): RBI follows Taylor rule → Inflation ↑ → Repo rate ↑ (R² = 0.72)
- **Crisis period** (2020): Liquidity support overrides inflation targeting → Inflation concerns secondary to growth support → Repo rate ↓ despite inflation (R² = 0.15)
- This is analogous to Fed's response: maintained near-zero rates despite later inflation surge"

**3. Implications for CCAR modeling**:
"This demonstrates model risk from structural instability:
- Historical relationships may not hold under stress
- Single-regime models inadequate for stress testing
- R² is period-dependent and scenario-dependent"

**4. Proposed model enhancement**:

**Option A - Regime-switching model**:
```
repo_t = β₀ + β₁inflation_t + β₂crisis_dummy_t + β₃(inflation_t × crisis_t) + ε_t

Where crisis_dummy captures policy regime changes
```

**Option B - Time-varying parameter model**:
Allow coefficients to evolve over time using state-space models.

**Option C - Incorporate policy indicators**:
Add central bank balance sheet size, forward guidance index, credit spreads as additional predictors.

**5. Validation approach**:
"We will:
1. Estimate model with structural break terms
2. Test stability across multiple crisis periods (2008, 2020)
3. Evaluate out-of-sample performance across regimes
4. Conduct scenario analysis: 'What if next crisis?'
5. Report conditional R² for each regime separately
6. Use judgment overlays when model signals regime change"

**6. Documentation**:
"Model documentation will include:
- Formal break tests and dates
- Economic interpretation of regime changes  
- Conditional performance metrics (R² by regime)
- Limitation statements about structural instability
- Governance process for judgment overlays during crises"

**Bottom line**: "R² varies across regimes. We acknowledge this limitation, have enhanced the model to account for it, and established governance for periods when historical relationships break down. This makes our model more robust for CCAR stress scenarios."

---

## Topic 6: Data Augmentation Techniques

### Question Story 6: Limited CCAR Stress Data

**Q6.1:** You're building a CCAR model but only have two historical severe recession observations (2001, 2008). Your supervisor suggests data augmentation. What is data augmentation and why is it useful?

**Answer:**
**Data augmentation**: Techniques to artificially expand training dataset by creating modified versions of existing data or generating synthetic data.

**Purpose**:
1. **Overcome limited data**: Especially for rare events (recessions, defaults, crises)
2. **Reduce overfitting**: More diverse training examples help model generalize
3. **Improve model robustness**: Model sees more scenarios
4. **Address class imbalance**: Create more examples of minority class

**Types relevant for CCAR**:

**1. Time series bootstrapping**:
- Resample residuals from historical model
- Add resampled residuals to fitted values
- Creates new synthetic scenarios preserving statistical properties

**2. Scenario generation**:
- Use economic models to generate plausible stress scenarios
- Combine observed data in new ways

**3. Interpolation/smoothing**:
- Generate intermediate scenarios between observed events
- SMOTE-like techniques for time series

**4. Monte Carlo simulation**:
- Simulate from estimated distributions
- Add controlled noise

**Caution**: Augmented data should be economically plausible and not used to artificially boost in-sample fit. Fed scrutinizes synthetic data usage.

**Q6.2:** Explain the bootstrap method for augmenting time series data. What are block bootstrap vs residual bootstrap?

**Answer:**
**Residual Bootstrap** (for regression):
```
1. Fit model: Y_t = β₀ + β₁X_t + ε_t
2. Calculate residuals: ê_t = Y_t - Ŷ_t
3. Resample residuals with replacement: ê*_t
4. Generate new Y*_t = Ŷ_t + ê*_t
5. Refit model on (X_t, Y*_t)
6. Repeat B times (e.g., B=1000)
7. Use distribution of β* for inference
```

**Assumption**: Residuals are iid (independent). Violated if autocorrelation exists.

**Block Bootstrap** (for time series with dependence):
```
1. Divide series into overlapping blocks of length L
2. Resample blocks (not individual observations) with replacement
3. Concatenate resampled blocks to form new series
4. Refit model
5. Repeat B times
```

**Why blocks**: Preserves autocorrelation structure within blocks.

**Example**: 
- Original: [Y₁, Y₂, Y₃, Y₄, Y₅, Y₆, Y₇, Y₈]
- Blocks (L=3): [Y₁Y₂Y₃], [Y₂Y₃Y₄], [Y₃Y₄Y₅], ..., [Y₆Y₇Y₈]
- Resample blocks: [Y₃Y₄Y₅], [Y₁Y₂Y₃], [Y₃Y₄Y₅], ...
- New series: Y₃, Y₄, Y₅, Y₁, Y₂, Y₃, Y₃, Y₄, ...

**Block length selection**: 
- Too small: Doesn't capture dependence
- Too large: Reduces diversity
- Rule of thumb: L ≈ n^(1/3) or use data-driven methods

**For CCAR**: Use block bootstrap for quarterly time series with autocorrelation.

**Q6.3:** What is SMOTE (Synthetic Minority Oversampling Technique) and how might you adapt it for financial time series?

**Answer:**
**SMOTE** (original - for classification):
Creates synthetic examples for minority class by interpolating between nearest neighbors.

**Algorithm**:
```
1. For each minority class example x:
2.    Find k nearest neighbors in feature space
3.    Randomly select one neighbor x_nn
4.    Generate synthetic example:
       x_new = x + λ(x_nn - x), where λ ~ Uniform(0,1)
5. Repeat until desired balance achieved
```

**Result**: Synthetic examples lie along line segments between existing minority examples.

**Adaptation for CCAR time series**:

**Problem**: In CCAR, "severe recession" is minority (rare event). Have few stress scenarios.

**Time Series SMOTE (TS-SMOTE)**:
```
1. Identify stress periods (e.g., 2001, 2008)
2. Extract feature vectors: [unemployment, GDP_growth, interest_rate, ...]
3. Find nearest stress scenario in feature space
4. Interpolate macroeconomic paths:
   unemployment_new(t) = unemployment₁(t) + λ[unemployment₂(t) - unemployment₁(t)]
   GDP_new(t) = GDP₁(t) + λ[GDP₂(t) - GDP₁(t)]
5. Use augmented scenarios for model training
```

**Variations**:
- **Time warping**: Stretch/compress time series to create variations
- **Magnitude scaling**: Scale severity (e.g., 0.8× → 1.2× of 2008 crisis)
- **Component mixing**: Mix GDP path from one crisis with unemployment path from another

**Caution**: 
- Ensure economic consistency (e.g., unemployment and GDP should be negatively correlated)
- Don't create economically impossible scenarios
- Fed may question synthetic scenarios - must document rationale

**Q6.4:** Your model training uses bootstrapped scenarios. How do you properly validate model performance given that test data might be similar to synthetic training data?

**Answer:**
**Key principle**: **Never validate on augmented data.** Use only real historical data for testing.

**Proper validation workflow**:

```
1. Split data FIRST:
   - Training: 2000-2018 (real data)
   - Testing: 2019-2023 (real data, held out)

2. Augment training data ONLY:
   - Bootstrap from 2000-2018
   - Generate synthetic scenarios
   - Training set expands from n to 10n

3. Train model on augmented training set

4. Validate on real test data (2019-2023)
   - Calculate metrics: RMSE, MAE, directional accuracy
   - Never use synthetic data for validation

5. Additional validation:
   - Walk-forward validation on real data
   - Scenario analysis on Fed's specified stress scenarios
   - Sensitivity analysis
```

**Why this matters**:
- Validating on synthetic data creates **circular validation** - model sees patterns it helped create
- Overestimates performance
- Fed requires out-of-sample validation on real data

**Cross-validation with augmentation**:
```
For each fold k:
    Training = folds ≠ k (real) + augmented versions of training
    Testing = fold k (real, no augmentation)
    Calculate performance
Average across folds
```

**Q6.5:** What is the difference between oversampling, undersampling, and SMOTE for imbalanced data? When would you use each for CCAR?

**Answer:**

| Technique | Method | Pros | Cons | CCAR Use Case |
|-----------|--------|------|------|---------------|
| **Oversampling** | Duplicate minority class | Simple, no information loss | Exact duplicates, overfitting risk | Quick baseline for rare default events |
| **Undersampling** | Remove majority class | Simple, faster training | Information loss, throws away data | When majority class is extremely large (e.g., non-defaulted loans) |
| **SMOTE** | Synthesize new minority | Creates diverse examples, no duplication | May create unrealistic examples, interpolation in feature space | Generating stress scenarios between observed recessions |

**CCAR Application Example**:

**Problem**: Predicting bank failures.
- Majority class: Survived banks (95%)
- Minority class: Failed banks (5%)

**Approach 1 - Undersampling**:
- Randomly select subset of survived banks to match failed banks
- **When**: Very large dataset, computational constraints
- **Risk**: Discard valuable information

**Approach 2 - Oversampling**:
- Replicate failed bank observations
- **When**: Quick baseline, limited time
- **Risk**: Model memorizes specific failures

**Approach 3 - SMOTE**:
- Create synthetic failed banks by interpolating
- **When**: Want diverse failure scenarios, sufficient features
- **Risk**: Synthetic banks may not be economically realistic

**Approach 4 - Class weights** (no augmentation):
- Assign higher loss penalty to minority class during training
- **When**: Want to avoid synthetic data altogether
- **Implementation**: In regression, weight observations; in classification, adjust loss function

**Best practice for CCAR**: 
- Use class weights first (no synthetic data issues)
- If more examples needed, use SMOTE with economic constraints
- Validate on real test data
- Document approach for Fed review

**Q6.6:** Connect hypothesis testing to bootstrapping. How do bootstrap confidence intervals work?

**Answer:**
**Bootstrap confidence interval** construction:

**Traditional approach** (parametric):
- Estimate β̂ from sample
- Calculate SE(β̂) assuming normality
- CI: β̂ ± t_α/2 · SE(β̂)
- **Requires**: Normality assumption, closed-form SE formula

**Bootstrap approach** (non-parametric):
```
1. Resample data with replacement B times (B=1000)
2. Calculate β̂* for each bootstrap sample
3. Get empirical distribution of β̂*
4. Construct CI from quantiles:
   - Percentile method: [β̂*_0.025, β̂*_0.975] for 95% CI
   - Or: bias-corrected accelerated (BCa) method
```

**Connection to hypothesis testing**:

**Test H₀: β = β₀**
- **Parametric**: t = (β̂ - β₀)/SE(β̂), compare to t-distribution
- **Bootstrap**: 
  - Check if β₀ falls within bootstrap CI
  - Or: Calculate p-value as proportion of bootstrap samples where |β̂* - β̂| > |β̂ - β₀|

**Advantages**:
1. No normality assumption needed
2. Works for complex estimators (median, ratio, quantiles)
3. Captures skewness in sampling distribution
4. Robust to violations

**Example - Testing significance of median**:
```
Observed median = $50,000
H₀: median = 0 (no effect)

Bootstrap procedure:
1. Resample data 1000 times
2. Calculate median for each sample
3. Get distribution of medians
4. 95% CI: [$42,000, $58,000]
5. 0 not in CI → Reject H₀ → Median significantly different from 0
```

**CCAR application**: Bootstrap credit loss distributions to get confidence bands around forecasts. If lower bound of 95% CI exceeds capital buffer, need more capital.

**Q6.7:** You use Monte Carlo simulation to generate 1000 synthetic stress scenarios by sampling from estimated distributions of macro variables. How do you ensure the simulated scenarios are economically coherent?

**Answer:**
**Challenge**: Independent sampling from marginal distributions destroys relationships between variables.

**Bad approach**:
```
unemployment ~ N(8%, 2%)     # Sample independently
GDP_growth ~ N(-2%, 3%)      # Sample independently
interest_rate ~ N(3%, 1%)    # Sample independently
```
**Problem**: Might generate unemployment=10% with GDP_growth=+5% (impossible - unemployment rises when GDP falls).

**Solutions**:

**1. Multivariate simulation preserving correlation**:
```
# Estimate covariance matrix Σ from historical data
# Sample from multivariate normal:
[unemployment, GDP_growth, interest_rate]' ~ MVN(μ, Σ)
```
Preserves linear correlations but not nonlinear relationships.

**2. Copula-based simulation**:
```
# Separate marginal distributions from dependence structure
# Model marginals: unemployment ~ t-distribution, GDP ~ skew-normal, ...
# Model dependence: Use copula (e.g., Gaussian, t, Clayton)
# Sample: 
   - Draw from copula to get correlated uniforms
   - Transform using inverse CDFs of marginals
```
Captures complex dependencies and non-normal marginals.

**3. Vector Autoregression (VAR)**:
```
# Jointly model dynamics:
X_t = A₀ + A₁X_{t-1} + ε_t, where ε ~ MVN(0, Σ)

# Simulate:
   - Sample ε from MVN
   - Iterate forward: X_{t+1} = A₀ + A₁X_t + ε_{t+1}
```
Preserves dynamics and comovement over time.

**4. Scenario trees with constraints**:
```
# Generate scenarios subject to economic constraints:
minimize distance_from_historical_distribution
subject to:
   - unemployment ↑ → GDP ↓ (negative correlation)
   - interest_rate bounded by [0, 15%]
   - No simultaneous boom in all indicators
```

**5. Expert judgment overlay**:
- Generate scenarios using statistical methods
- Review for economic plausibility
- Adjust or discard implausible scenarios
- Document rationale

**For CCAR**: Fed provides 3 scenarios (baseline, adverse, severely adverse). For internal stress testing, use VAR or copulas to generate additional scenarios. Document assumptions and validate against historical crisis episodes.

**Q6.8:** What is k-fold cross-validation and how does it differ from bootstrap validation? Which is better for CCAR time series?

**Answer:**

**K-fold Cross-Validation**:
```
1. Divide data into k equal folds (e.g., k=5)
2. For i = 1 to k:
     - Training: All folds except i
     - Validation: Fold i
     - Calculate error_i
3. Average error across folds: CV_error = (1/k)Σerror_i
```

**Bootstrap Validation**:
```
1. For b = 1 to B (e.g., B=1000):
     - Sample n observations with replacement (training)
     - Observations not sampled (≈37%) form validation set
     - Calculate error_b
2. Average error: Bootstrap_error = (1/B)Σerror_b
```

**Comparison**:

| Aspect | K-fold CV | Bootstrap |
|--------|-----------|-----------|
| Training size | (k-1)n/k | ≈0.632n (smaller) |
| Validation size | n/k | ≈0.368n |
| Bias | Low | Higher (smaller training set) |
| Variance | Higher | Lower (more replicates) |
| Computation | Moderate (k fits) | High (B fits) |
| Time series | Problematic (violates temporal order) | Problematic (breaks dependencies) |

**For CCAR time series**:

**Neither standard approach is ideal** because:
- K-fold: Training data may come from future (look-ahead bias)
- Bootstrap: Resampling breaks temporal order

**Better alternatives**:

**1. Time Series Cross-Validation (Rolling window)**:
```
Training: 2000-2010 → Test: 2011
Training: 2000-2011 → Test: 2012
Training: 2000-2012 → Test: 2013
...
Average test errors
```
Respects temporal order, mimics realistic forecasting.

**2. Blocked Cross-Validation**:
```
Block 1: 2000-2004 | Block 2: 2005-2009 | Block 3: 2010-2014 | Block 4: 2015-2019
Train on 1,2,3 → Test on 4
Train on 1,2,4 → Test on 3
...
```
Ensures temporal gaps between train and test.

**3. Block Bootstrap**:
```
Resample contiguous blocks of time series
Preserves within-block dependencies
```

**Recommendation for CCAR**: **Rolling window time series CV**. It's what Fed expects - model trained on history, tested on future periods.

**Q6.9:** During Federal Reserve review, they question: "You augmented training data using bootstrap. Doesn't this artificially improve model fit and violate independence assumptions?" How do you respond?

**Answer:**
**Structured response**:

**1. Clarify the purpose**:
"Augmentation was not used to improve in-sample fit or achieve higher R². It was used to enhance model training for better generalization, particularly for rare stress events underrepresented in our 20-year sample."

**2. Distinguish training from validation**:
"**Critically**: All model performance metrics reported—R², RMSE, MAE, forecast accuracy—are calculated on **real, held-out test data only**. No synthetic or augmented data was used in validation. This ensures true out-of-sample performance measurement."

**3. Address independence concern**:
"The bootstrap approach maintains the statistical properties of residuals:
- We use residual bootstrap for iid errors or block bootstrap for autocorrelated errors
- This preserves the dependency structure in the data
- Each bootstrap sample is an independent draw from the empirical distribution
- Model sees diverse realizations of the same stochastic process, not duplicate data"

**4. Justification for augmentation**:
"Financial crises are rare but critical for CCAR:
- Only 2-3 severe stress episodes in our data (2001, 2008, 2020)
- Risk of overfitting to specific crisis characteristics
- Augmentation allows model to learn generalizable stress patterns rather than memorizing specific events
- Analogous to how image recognition uses data augmentation for rare but important objects"

**5. Safeguards implemented**:
"To ensure methodological rigor:
- Documented augmentation procedure in model documentation
- Compared augmented vs non-augmented model performance—augmented version shows more stable coefficients and better out-of-sample forecasting
- Conducted sensitivity analysis with different bootstrap parameters
- Validated that synthetic scenarios fall within historical range (no extrapolation)
- External reviewers (independent model validation team) verified approach"

**6. Regulatory precedent**:
"Bootstrap and resampling methods are standard in econometrics and accepted by regulators:
- Used in Basel III market risk models (VaR backtesting)
- Recommended in academic literature for limited data scenarios
- IMF and BIS publications discuss bootstrap for stress testing
- We follow best practices from [cite specific papers/guidelines]"

**7. Alternative would be worse**:
"Without augmentation, we face two bad alternatives:
- Train only on limited data → Model doesn't generalize to stress not exactly like 2008
- Use longer history → Include data from pre-modern banking system (1980s) which isn't relevant to current environment"

**Conclusion**: "Augmentation is a tool to improve generalization, not fit. All validation uses real data. The approach is methodologically sound, well-documented, and improves model robustness for CCAR stress scenarios."

**Q6.10:** Design a complete data augmentation strategy for a CCAR credit loss model with limited stress scenario data, including how you'd address regulatory concerns.

**Answer:**
**Comprehensive CCAR Data Augmentation Framework**:

**Phase 1: Assess Need**
```
Baseline assessment:
- Historical data: 2000-Q1 to 2023-Q4 (96 quarters)
- Stress periods: 2001 (3 quarters), 2008-2009 (6 quarters), 2020 (2 quarters)
- Stress data: 11/96 = 11.5% of sample
- Class imbalance: Severe (need augmentation)
```

**Phase 2: Select Augmentation Methods**

**Method 1: Residual Bootstrap for Base Case**
```
For normal economic periods:
1. Fit baseline model: loss_t = f(unemployment_t, GDP_t, ...) + ε_t
2. Residual bootstrap with block length L=4 (annual cycles)
3. Generate 5× augmented normal scenarios
4. Use for training stability in normal times
```

**Method 2: Stress Scenario Synthesis**
```
For stress periods:
1. Extract 2001, 2008, 2020 macroeconomic paths
2. Apply time-series SMOTE:
   - Interpolate between crisis pairs in feature space
   - Generate paths like "0.7×2008 + 0.3×2020"
3. Ensure economic coherence using VAR:
   - Estimate VAR on combined crisis data
   - Simulate conditional on severity
4. Create 20 synthetic stress scenarios
5. Expert review for plausibility
```

**Method 3: Monte Carlo with Copulas**
```
For tail risk scenarios:
1. Fit marginal distributions to each macro variable (t-distributions for fat tails)
2. Estimate copula for dependency (t-copula to capture tail dependence)
3. Simulate 100 scenarios
4. Filter for severity: Keep scenarios where unemployment > 8% and GDP < -1%
5. Select 10 most plausible for augmentation
```

**Phase 3: Quality Control**

**Economic Coherence Checks**:
```
For each synthetic scenario:
1. Correlation check: unemployment vs GDP should be negative
2. Magnitude check: Variables within historical range ± 20%
3. Dynamics check: No impossible period-to-period changes
4. Expert review: SME reviews scenarios quarterly
5. Documentation: Log all accepted and rejected scenarios with rationale
```

**Phase 4: Training Methodology**

**Stratified Training**:
```
Training set composition:
- Real normal periods: 85 quarters
- Real stress periods: 11 quarters (100% inclusion, no substitution)
- Augmented normal: 85 × 2 = 170 quarters
- Augmented stress: 11 × 3 = 33 quarters
Total: 299 quarters for training

Weighting:
- Real data: weight = 1.0
- Augmented data: weight = 0.5
- Ensures real data dominates model training
```

**Phase 5: Validation Framework**

**Multi-layer Validation**:
```
1. Out-of-sample test: 2021-2023 (real data only)
   - Metrics: RMSE, MAE, directional accuracy
   
2. Crisis-only performance:
   - Hold out 2008-Q3 to 2009-Q2 (peak crisis)
   - Train on 2000-2007, 2010-2020 + augmented data
   - Test on 2008-2009
   - Compare to non-augmented model
   
3. Scenario-based validation:
   - Apply model to Fed's severely adverse scenario
   - Compare forecast distribution to Fed expectations
   
4. Stress test across augmentation parameters:
   - Test with 0×, 1×, 3×, 5× augmentation factors
   - Optimal factor shows best out-of-sample performance
```

**Phase 6: Documentation for Federal Reserve**

**Model Documentation Sections**:
```
1. Executive Summary
   - Why augmentation needed (data limitations)
   - Methods used with references
   - Impact on model performance
   
2. Methodology Detail
   - Mathematical description of each technique
   - Parameter selection (bootstrap block length, SMOTE k)
   - Economic coherence constraints
   
3. Validation Results
   - Comparison table: Augmented vs Non-augmented
   - Real data test performance
   - Sensitivity to augmentation parameters
   
4. Limitations and Risks
   - Synthetic data may not capture future crisis types
   - Model uncertainty higher for rare events
   - Mitigation: Judgment overlays in severely adverse scenarios
   
5. Governance
   - Quarterly review of augmentation assumptions
   - Annual revalidation
   - Trigger for model updates if new stress events occur
```

**Phase 7: Ongoing Monitoring**

**Quarterly Review Process**:
```
1. Compare model forecasts to actual outcomes
2. If new stress data becomes available (e.g., new crisis):
   - Incorporate real data immediately
   - Reduce reliance on augmented data
   - Revalidate model
3. Track distribution drift in real vs augmented data
4. Update augmentation strategy as needed
```

**Key Success Metrics**:
```
1. Out-of-sample RMSE: Augmented model < non-augmented
2. Crisis forecast accuracy: Lower prediction intervals include actual losses
3. Coefficient stability: Lower standard errors with augmentation
4. Regulatory acceptance: Model approved in annual CCAR submission
```

**Implementation Timeline**:
```
Q1: Design framework, obtain data, develop code
Q2: Generate augmented scenarios, perform QC
Q3: Train models, conduct validation
Q4: Document, present to governance, submit to Fed
Ongoing: Monitor, update quarterly
```

**Conclusion**: This comprehensive approach balances the need for more training data with methodological rigor, economic plausibility, and regulatory requirements. Key principle: Augmentation enhances training; validation uses only real data.

---

## Topic 7: Case Studies - Regression Modeling

### Question Story 7: Practical CCAR Modeling Scenarios

**Q7.1:** **Case A**: A retail bank wants to model credit card charge-off rates. Available data: monthly charge-off rates (2010-2023), unemployment rate, consumer confidence index, interest rates, and credit bureau scores. The bank executive says "just give me a model with highest R²." Walk through your modeling approach.

**Answer:**
**Modeling Framework**:

**Step 1: Data exploration**
```
- Check for missing values, outliers
- Plot time series of charge-off rate → Identify trends, seasonality
- Plot against each predictor → Initial relationships
- Check stationarity (ADF test) on all variables
```

**Step 2: Address the "highest R²" request**
"**Push back**: Highest R² is not the goal. Explain:
- Overfitting risk with too many variables
- Out-of-sample performance matters more
- Regulatory requirements (SR 11-7) emphasize conceptual soundness
- Need economically interpretable coefficients"

**Goal**: Build model that forecasts well, is interpretable, and meets regulatory standards.

**Step 3: Variable transformation/stationarity**
```
If non-stationary:
- Difference variables: Δcharge_off_t = charge_off_t - charge_off_{t-1}
- Or establish cointegrating relationships
```

**Step 4: Model specification**
```
Start with theory-driven specification:
charge_off_t = β₀ + β₁unemployment_t + β₂interest_rate_t + β₃confidence_t + ε_t

Economic hypotheses:
- β₁ > 0: Higher unemployment → more charge-offs
- β₂ > 0: Higher rates → harder to service debt
- β₃ < 0: Higher confidence → lower charge-offs
```

**Step 5: Estimate and test assumptions**
```
1. Estimate OLS
2. Check residual diagnostics:
   - Autocorrelation: Durbin-Watson, Breusch-Godfrey
   - Heteroskedasticity: Breusch-Pagan, White
   - Normality: Jarque-Bera, Q-Q plot
   - Linearity: Ramsey RESET
   - Multicollinearity: VIF
3. If violations found → remedy (add lags, robust SE, transform variables)
```

**Step 6: Model selection**
```
Try alternative specifications:
- Add lagged dependent variable
- Add interaction terms (e.g., unemployment × interest_rate)
- Add seasonal dummies
Compare using:
- AIC, BIC (penalize complexity)
- Out-of-sample RMSE (rolling window validation)
- Economic interpretation
```

**Step 7: Out-of-sample validation**
```
- Train on 2010-2021
- Test on 2022-2023
- Calculate: RMSE, MAE, directional accuracy
- If performance poor → iterate
```

**Step 8: Final model documentation**
```
- Chosen specification and rationale
- Diagnostic test results
- Coefficient interpretation
- Out-of-sample performance
- Limitations and uncertainties
- Forecasting procedure
```

**Response to executive**: "The final model has R² = 0.68, but more importantly:
- All diagnostics pass
- Out-of-sample RMSE is 0.15% (accurate forecasts)
- Coefficients are economically sensible and significant
- Ready for regulatory submission"

**Q7.2:** **Case B**: During model estimation, you find unemployment has correct sign but is not statistically significant (p = 0.18). Interest rate and confidence are highly significant. The bank's chief risk officer says "remove unemployment - it's not significant." How do you respond?

**Answer:**
**Response**: "I recommend keeping unemployment despite p = 0.18. Here's why:"

**1. Economic theory vs statistical significance**:
- "Unemployment is theoretically the most important driver of consumer credit losses - this is well-established in literature
- Statistical insignificance may be due to multicollinearity, limited variation in sample, or imprecise measurement
- **Type II error risk**: Removing unemployment could create omitted variable bias"

**2. Check for multicollinearity**:
```
Calculate VIF for unemployment:
- If VIF > 10: Multicollinearity is the problem, not lack of relationship
- High correlation with other predictors makes it appear insignificant
- The variable still adds information but jointly with others
```

**3. Out-of-sample forecasting**:
```
Test two models:
- Model A (with unemployment): Out-of-sample RMSE = 0.15%
- Model B (without unemployment): Out-of-sample RMSE = 0.19%

"Model A forecasts better, confirming unemployment is valuable despite p = 0.18"
```

**4. CCAR stress testing concern**:
"In Fed's severely adverse scenario, unemployment spikes from 4% to 10%. If we remove unemployment:
- Model cannot respond to this key stress driver
- Charge-off forecast will be unrealistic
- Fed will reject the model for lack of economic content"

**5. Joint significance**:
```
Perform F-test on unemployment and its interactions/lags jointly:
H₀: β_unemployment = β_unemployment_lag = 0
If F-test significant → Unemployment matters, just not in isolation
```

**6. Alternative: Keep but augment**:
```
Instead of removing, try:
- Add lagged unemployment (distributed lag effect)
- Add change in unemployment (Δunemployment)
- Add unemployment × recession_dummy interaction
These may be more significant while keeping economic content
```

**7. Regulatory perspective**:
"SR 11-7 states models must be 'consistent with economic theory and sound judgment.' Removing unemployment contradicts both. Fed specifically checks that key risk drivers are included."

**Conclusion**: 
"I recommend: 
1. Keep unemployment for conceptual soundness
2. Add lagged term to capture delayed effects
3. Document that economic theory justifies inclusion despite p = 0.18
4. Emphasize out-of-sample forecasting performance improvement
5. Ready to explain rationale to Fed if questioned"

**Trade-off**: "We prioritize forecasting accuracy and economic coherence over maximizing statistical significance of every single coefficient."

**Q7.3:** **Case C**: You model auto loan defaults using borrower credit score, loan-to-value ratio (LTV), and unemployment. Plot of residuals vs fitted values shows clear heteroskedasticity (cone shape - variance increases with fitted values). What do you do?

**Answer:**
**Diagnosis**: Heteroskedasticity confirmed visually. Higher default levels have higher variance.

**Step-by-step remedy**:

**Step 1: Formal test**
```
White test:
Regress ê² on X₁, X₂, X₃, X₁², X₂², X₃², X₁X₂, X₁X₃, X₂X₃
LM = nR² ~ χ²(8)
Result: LM = 42, p < 0.001 → Heteroskedasticity confirmed
```

**Step 2: Investigate source**
```
Plot ê² vs each predictor individually:
- vs credit_score: Flat → Not the source
- vs LTV: Increasing → Source likely here
- vs unemployment: Flat → Not the source

Interpretation: High-LTV loans have more variable default outcomes
Makes sense: Borrowers with less equity are more sensitive to economic shocks
```

**Step 3: Choose remedy**

**Option A: Robust standard errors (simplest)**
```
Advantages:
- No model respecification needed
- Coefficients unchanged (still unbiased)
- Fixes inference (hypothesis tests, CIs)
- Widely accepted

Implementation:
Use HC3 robust SE (best in small samples):
SE_robust = √{diagonal of (X'X)⁻¹ X' Ω̂ X (X'X)⁻¹}

Result:
- Coefficient on LTV: β = 0.08, SE_OLS = 0.02, SE_robust = 0.035
- t-stat: OLS = 4.0, robust = 2.29
- Still significant but less so → More honest inference
```

**Option B: Weighted Least Squares**
```
If variance structure is known/estimable:
1. Estimate variance as function of LTV:
   log(ê²) = α₀ + α₁LTV + u
   
2. Predicted variance: σ̂²ᵢ = exp(α̂₀ + α̂₁LTV_i)
   
3. Weight observations by 1/σ̂²ᵢ
   
4. Re-estimate: minimize Σ[(y_i - ŷ_i)²/σ̂²ᵢ]

Advantages:
- Restores efficiency (BLUE)
- Proper forecast intervals

Disadvantages:
- Two-stage estimation (more complexity)
- Misspecified weights can make things worse
```

**Option C: Transform dependent variable**
```
Try log transformation:
log(default_rate + c) = β₀ + β₁credit_score + β₂LTV + β₃unemployment + ε

Or logit transformation:
logit(default_rate) = log[default_rate/(1-default_rate)] = ...

Advantages:
- Often stabilizes variance
- Bounded predictions (for logit)

Disadvantages:
- Coefficients harder to interpret
- Need to back-transform forecasts
```

**Step 4: Choose for CCAR context**

**Recommendation**: **Robust standard errors (Option A)**

**Rationale**:
1. **Simplicity**: Model structure unchanged, easy to explain to Fed
2. **Transparency**: Coefficients retain original interpretation
3. **Conservatism**: Wider confidence intervals (appropriate for risk management)
4. **Standard practice**: Widely used and accepted in CCAR models
5. **Robustness**: Works regardless of exact variance structure

**Implementation in report**:
```
"We detect heteroskedasticity (White test, p < 0.001). Variance increases with LTV, 
reflecting greater uncertainty in high-LTV loan performance. We address this by:
1. Using Huber-White robust standard errors for inference
2. Reporting heteroskedasticity-consistent confidence intervals
3. Validating that coefficient estimates remain stable and economically sensible

All reported p-values and confidence intervals use robust standard errors."
```

**Step 5: Verify fix**
```
- Rerun significance tests with robust SE
- Confirm key variables still significant
- Document in model validation
- Proceed with forecasting
```

**Q7.4:** **Case H**: You're asked: "Why use linear regression at all? Why not machine learning like random forest or neural networks for better predictions?" This is for a CCAR submission to the Fed.

**Answer:**
**Structured response defending linear regression for CCAR**:

**1. Regulatory requirements**:
"Federal Reserve SR 11-7 emphasizes:
- **Transparency**: Model logic must be explainable to non-technical regulators
- **Economic interpretation**: Coefficients must have economic meaning
- **Causality**: Need to understand 'what if unemployment rises 4%?'

Linear regression provides clear coefficients (β₁ = 0.08 means 1pp ↑ unemployment → 0.08pp ↑ defaults).
Random forest gives feature importance, not causal elasticities."

**2. Scenario analysis**:
"CCAR requires conditional forecasting under Fed's prescribed scenarios:
- 'What happens if unemployment goes to 10%, GDP falls 3%, rates stay at 2%?'
- Linear regression: Straightforward to plug in scenario values
- Neural network: Black box, hard to ensure it responds economically to counterfactual scenarios never seen in training data"

**3. Sample size concerns**:
"We have ~50 quarterly observations for stressed periods. Machine learning needs much more data:
- Random forest: Thousands of observations to learn complex patterns
- Neural network: Even more data to avoid overfitting
- Linear regression: Performs well with moderate sample sizes, especially with good theory"

**4. Extrapolation**:
"Stress tests require extrapolation beyond historical range (e.g., unemployment to 12% when max observed is 10%):
- Linear models extrapolate based on estimated relationships
- ML models generally interpolate poorly outside training distribution
- Neural networks can produce nonsensical predictions outside training domain"

**5. Model validation and diagnostics**:
"Linear regression has mature toolkit:
- Diagnostic tests (DW, BP, VIF, RESET) with established interpretation
- Residual analysis to detect violations
- ML models lack comparable diagnostic framework
- How do you test 'assumptions' of a random forest?"

**6. Comparability and benchmarking**:
"CCAR models across banks are compared:
- Linear coefficients are comparable (Bank A's unemployment coefficient vs Bank B's)
- Can't compare two neural networks meaningfully
- Industry standards and academic literature based on linear models"

**7. When complexity adds value**:
"For prediction-only tasks (fraud detection, customer churn), ML shines. But CCAR is about:
- Understanding drivers (not just prediction)
- Explaining to regulators why model made specific forecast
- Justifying capital requirements based on transparent logic"

**8. Hybrid approach**:
"We can get best of both:
- Use linear regression for primary CCAR model (regulatory submission)
- Use ML as **challenger model** for comparison
- If ML significantly outperforms, investigate why:
  - Missing interactions in linear model?
  - Nonlinear relationships?
- Incorporate findings back into linear framework (add interaction terms, polynomials)"

**9. Practical example**:
"Suppose random forest forecast is 2% and linear model forecast is 2.5%:
- Regulator asks: 'Why did you forecast 2%?'
- Linear model: 'Because unemployment rose 3pp, and our estimated coefficient is 0.08, and interest rates...'
- Random forest: 'The algorithm weighted these variables based on impurity reduction across 1000 trees...' ❌"

**10. Fed guidance**:
"From SR 11-7: 'Models should be documented...in a manner that can be understood by parties not directly involved in model development.'
Linear regression meets this. Neural networks generally don't."

**Conclusion**:
"For CCAR, we choose linear regression because:
- Regulatory acceptance and transparency requirements
- Economic interpretability
- Sufficient forecasting accuracy for our sample size and problem structure
- Ability to perform scenario analysis
- Mature diagnostic tools

We can augment with ML as a validation check, but primary submission model should be linear regression or similar transparent approach (GLM, maybe quantile regression for tail modeling).

**Transparency, interpretability, and economic content trump marginal prediction gains** in regulatory context."

**Q7.5:** **Case D**: Your model passes all diagnostic tests during 2000-2019 training period. But in 2020 COVID period, forecast errors are massive (actual charge-offs 3× higher than predicted). What went wrong and how do you fix for future CCAR?

**Answer:**
**Diagnosis**: **Structural break** due to unprecedented economic shock and policy interventions.

**What went wrong**:

**1. Unprecedented event**:
```
COVID shock characteristics:
- Fastest unemployment spike in history (3.5% → 14.7% in 2 months)
- Simultaneous: Supply shock (lockdowns) + demand shock (fear, unemployment)
- Massive policy response: Stimulus checks, PPP, eviction moratoriums, student loan forbearance

Model trained on historical recessions (2001, 2008) didn't capture:
- Speed of deterioration
- Effectiveness of policy support (forestalled defaults)
- Behavioral changes (increased savings from stimulus)
```

**2. Nonlinearity not captured**:
```
Linear model assumes: ΔY = β·ΔX
But in extreme stress: Relationship may be nonlinear
- Small unemployment increase (4% → 5%): Small effect
- Large unemployment spike (4% → 14%): Disproportionately large effect (credit networks collapse, liquidity dries up)
```

**3. Missing policy variables**:
```
Model included: unemployment, GDP, interest rates
Model missed: Fiscal stimulus scale, forbearance programs, direct cash transfers
These were critical in COVID - prevented expected defaults despite high unemployment
```

**How to fix for future CCAR**:

**Fix 1: Add nonlinear terms**
```
Extend model:
charge_off_t = β₀ + β₁unemployment_t + β₂unemployment_t² + β₃X_t + ε_t

Or threshold effects:
charge_off_t = β₀ + β₁unemployment_t + β₂(unemployment_t > 8%)·(unemployment_t - 8) + β₃X_t + ε_t

This captures: Mild stress → linear response; severe stress → accelerated deterioration
```

**Fix 2: Add policy indicators**
```
Include:
- Fiscal_support_t: Size of stimulus/GDP
- Forbearance_dummy_t: 1 if forbearance programs active
- Fed_balance_sheet_t: Monetary policy stance

Model:
charge_off_t = β₁unemployment_t + β₂fiscal_support_t + β₃unemployment_t×fiscal_support_t + ...

Interaction term captures: High unemployment with high fiscal support → lower charge-offs
```

**Fix 3: Regime-switching framework**
```
Estimate separate models for normal vs crisis regimes:

Normal regime (prob p):
charge_off_t = β₀^N + β₁^N unemployment_t + ε_t

Crisis regime (prob 1-p):
charge_off_t = β₀^C + β₁^C unemployment_t + β₂^C fiscal_t + ε_t

Markov-switching model determines regime probabilities based on data
```

**Fix 4: Include COVID period in training**
```
Originally trained on 2000-2019 → Now include 2020-2023
Model learns from COVID:
- Extreme stresses
- Policy effectiveness
- Nonlinear relationships

Caution: Only 4 quarters of COVID → Still limited, but better than excluding
```

**Fix 5: Stress-specific coefficients**
```
Augment with dummy:
charge_off_t = β₀ + β₁unemployment_t + β₂(crisis_dummy)×unemployment_t + β₃X_t + ε_t

Crisis_dummy = 1 if severe stress (unemployment > 7%)
Allows different sensitivity in stress vs normal times
```

**Fix 6: Expert judgment overlay**
```
For future unprecedented events:
1. Model provides baseline forecast
2. Stress testing committee overlays judgment:
   - "If scenario includes massive stimulus, adjust forecast down by X%"
   - Document assumptions and rationale
3. Create judgment-adjusted forecast
4. Report both model-only and adjusted forecasts to Fed with explanation
```

**Fix 7: Scenario analysis & sensitivity testing**
```
Run model under extreme scenarios not in training data:
- "What if unemployment → 15%?"
- "What if 2008 + COVID simultaneously?"
- Identify model weaknesses proactively
- Document limitations in model documentation
```

**Validation approach going forward**:

**1. Backtest on COVID**:
```
- Hold out 2020-2021 as test set
- Evaluate each fix:
  - Model A (original): RMSE = 1.5%
  - Model B (+ nonlinear terms): RMSE = 0.9%
  - Model C (+ policy variables): RMSE = 0.7%
  - Model D (regime-switching): RMSE = 0.8%
- Select Model C (lowest error, interpretable)
```

**2. Stress test across scenarios**:
```
Test model on Fed's hypothetical severely adverse scenario:
- Verify response is economically sensible
- Not overly sensitive (predicts 100% charge-offs) or insensitive (flat line)
```

**3. Limitations statement**:
```
"Model is estimated on data through 2023, including COVID pandemic. However:
- Future crises may differ in nature (cyber attack, geopolitical event)
- Policy responses are inherently uncertain and model may not capture novel interventions
- For unprecedented scenarios, judgment overlays may be necessary
- Model will be updated as new stress events occur"
```

**Response to Fed**:
"COVID revealed model limitations. We've enhanced the model by:
1. Including COVID data (2020-2023) in training
2. Adding nonlinear terms to capture extreme stress dynamics
3. Including fiscal policy variables to reflect interventions
4. Implementing regime-switching to allow different relationships in stress vs normal times
5. Validating that enhanced model backtests well on COVID period (RMSE reduced from 1.5% to 0.7%)
6. Establishing governance for judgment overlays in unprecedented scenarios

The model is now more robust to tail events while remaining transparent and economically interpretable."

**Key lesson**: No model captures every future scenario. Build in flexibility, monitor performance, update as new data arrives, and maintain judgment overlay process for truly unprecedented events.

---

## Topic 8: Null Hypothesis and P-value

### Question Story 8: Understanding Statistical Inference

**Q8.1:** In plain terms, explain what a null hypothesis, alternative hypothesis, and p-value are. Why do we need them?

**Answer:**
**Null Hypothesis (H₀)**: Statement of "no effect" or "status quo." Assumes nothing interesting is happening.
- Examples: 
  - "This drug has no effect" (treatment coefficient = 0)
  - "These two groups have equal means"
  - "Model coefficients are all zero"

**Alternative Hypothesis (H₁)**: Statement we're trying to find evidence for. The research hypothesis.
- Examples:
  - "Drug reduces blood pressure" (coefficient < 0)
  - "Group A has higher mean than group B"
  - "At least one coefficient ≠ 0"

**P-value**: Probability of seeing data as extreme as ours (or more extreme), **assuming H₀ is true**.
- p = 0.03 means: "If null hypothesis were true, there's only 3% chance we'd see data this extreme"
- Small p-value → Data is unlikely under H₀ → Evidence against H₀ → Reject H₀

**Why we need them**:
- Can't prove hypotheses with certainty from samples (always some randomness)
- Framework to make decisions under uncertainty
- Controls error rates (Type I error = false positives)
- Standardizes statistical reasoning across fields

**Process**:
1. State H₀ and H₁
2. Choose significance level α (usually 0.05)
3. Collect data and calculate test statistic
4. Calculate p-value
5. Decision: If p < α → Reject H₀ (conclude H₁ is supported)
              If p ≥ α → Fail to reject H₀ (insufficient evidence for H₁)

**Q8.2:** You test whether unemployment rate affects default rates. Your regression gives:
- Coefficient on unemployment: β₁ = 0.085
- Standard error: SE = 0.032
- t-statistic: 2.66
- p-value: 0.012

Walk through the hypothesis test interpretation.

**Answer:**
**Setup**:
- **H₀**: β₁ = 0 (unemployment has no effect on defaults)
- **H₁**: β₁ ≠ 0 (unemployment affects defaults)
- **Significance level**: α = 0.05 (conventional)

**Test statistic**:
```
t = β̂₁ / SE(β̂₁) = 0.085 / 0.032 = 2.66
```

**Degrees of freedom**: n - k - 1 (assume n=100, k=3 predictors → df = 96)

**Critical value**: t₀.₀₂₅,₉₆ ≈ 1.98 (two-tailed test at α = 0.05)

**P-value**: 0.012
- Interpretation: "If unemployment truly had zero effect, there's only 1.2% probability we'd observe a coefficient as large as 0.085 (or larger in absolute value) purely by chance."

**Decision**:
- p = 0.012 < α = 0.05 → **Reject H₀**
- OR: |t| = 2.66 > critical value = 1.98 → **Reject H₀**

**Conclusion**:
"Unemployment rate has a statistically significant effect on default rates (p = 0.012). Specifically, a 1 percentage point increase in unemployment is associated with a 0.085 percentage point increase in default rates, and this relationship is unlikely to be due to chance."

**Confidence interval** (alternative presentation):
```
95% CI: β̂₁ ± t₀.₀₂₅ × SE = 0.085 ± 1.98 × 0.032 = [0.022, 0.148]
```
Since 0 is not in the interval → Same conclusion: Reject H₀

**Economic significance**:
"Beyond statistical significance, this is economically meaningful. In a severe recession where unemployment rises 5 percentage points (e.g., 4% → 9%), our model predicts default rates increase by 5 × 0.085 = 0.425 percentage points, which is substantial for a large loan portfolio."

**Q8.3:** What are Type I and Type II errors? What's the trade-off? How does this relate to α and power?

**Answer:**
**Decision matrix**:

|  | **H₀ True** | **H₀ False** |
|--|-------------|--------------|
| **Reject H₀** | Type I Error (α) | Correct (Power = 1-β) |
| **Fail to reject H₀** | Correct (1-α) | Type II Error (β) |

**Type I Error** (False Positive):
- Rejecting H₀ when it's actually true
- Concluding effect exists when it doesn't
- Example: "Drug works" when it doesn't → Patients get useless treatment
- Probability: α (significance level, typically 0.05)

**Type II Error** (False Negative):
- Failing to reject H₀ when it's actually false
- Missing a real effect
- Example: "Drug doesn't work" when it actually does → Patients miss beneficial treatment
- Probability: β (depends on effect size, sample size, α)

**Power**:
- Power = 1 - β = Probability of correctly rejecting false H₀
- Power = Probability of detecting a true effect
- Higher power = better test

**Trade-off**:
```
Lower α → Harder to reject H₀ → Less Type I errors BUT more Type II errors (lower power)
Higher α → Easier to reject H₀ → More Type I errors BUT fewer Type II errors (higher power)
```

**Example**:
- α = 0.01 (very strict): Less chance of false discoveries, but might miss real effects
- α = 0.10 (lenient): Catch more real effects, but more false positives

**What affects power**:
1. **Sample size (n)**: ↑ n → ↑ power (most important)
2. **Effect size**: Larger true effect → easier to detect → ↑ power
3. **Significance level (α)**: ↑ α → ↑ power (but also ↑ Type I error)
4. **Variance (σ²)**: ↓ σ² → ↑ power (cleaner data, less noise)

**CCAR context**:
- Type I error: Concluding variable is important when it's not → Overcomplicates model
- Type II error: Missing important risk driver → Model fails to capture key risks → Insufficient capital