# 📋 NumPy Quick Reference Cheat Sheet

## 🚀 Import
```python
import numpy as np
```

---

## 1️⃣ ARRAY CREATION

### Basic Creation
```python
np.array([1, 2, 3])              # From list
np.array([[1,2], [3,4]])         # 2D array
```

### Pre-filled Arrays
```python
np.zeros((3, 4))                 # 3x4 array of zeros
np.ones((2, 3))                  # 2x3 array of ones
np.full((2, 2), 7)               # 2x2 array filled with 7
np.eye(3)                        # 3x3 identity matrix
```

### Ranges
```python
np.arange(0, 10, 2)              # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)             # 5 evenly spaced numbers from 0 to 1
```

---

## 2️⃣ ARRAY ATTRIBUTES

```python
arr.shape                        # Dimensions (rows, cols)
arr.size                         # Total elements
arr.dtype                        # Data type
arr.ndim                         # Number of dimensions
arr.itemsize                     # Bytes per element
```

---

## 3️⃣ INDEXING & SLICING

### 1D Arrays
```python
arr[0]                           # First element
arr[-1]                          # Last element
arr[2:5]                         # Elements 2, 3, 4
arr[::2]                         # Every 2nd element
```

### 2D Arrays
```python
arr[0, 1]                        # Row 0, Column 1
arr[0]                           # Entire first row
arr[:, 1]                        # Entire second column
arr[1:, :2]                      # Rows 1+, Columns 0-1
```

### Boolean Indexing
```python
arr[arr > 5]                     # Elements greater than 5
arr[(arr > 2) & (arr < 8)]       # Multiple conditions
```

### Fancy Indexing
```python
arr[[0, 2, 4]]                   # Elements at indices 0, 2, 4
arr[[0, 1], [1, 2]]              # Elements at (0,1) and (1,2)
```

---

## 4️⃣ RESHAPING

```python
arr.reshape(3, 4)                # Change shape to 3x4
arr.reshape(-1, 2)               # Auto-calculate first dimension
arr.flatten()                    # To 1D (copy)
arr.ravel()                      # To 1D (view, faster)
arr.T                            # Transpose
arr[:, np.newaxis]               # Add dimension
np.squeeze(arr)                  # Remove single dimensions
```

---

## 5️⃣ STACKING & SPLITTING

### Stacking
```python
np.vstack((arr1, arr2))          # Stack vertically (rows)
np.hstack((arr1, arr2))          # Stack horizontally (columns)
np.stack((arr1, arr2), axis=0)   # Stack along new axis
np.concatenate((arr1, arr2), axis=0)  # General concatenation
```

### Splitting
```python
np.vsplit(arr, 2)                # Split vertically into 2
np.hsplit(arr, 3)                # Split horizontally into 3
np.split(arr, 2, axis=0)         # Split along axis 0
np.array_split(arr, 3)           # Split into 3 (handles uneven)
```

---

## 6️⃣ MATHEMATICAL OPERATIONS

### Element-wise
```python
arr + 5                          # Add 5 to all elements
arr * 2                          # Multiply all by 2
arr ** 2                         # Square all elements
np.sqrt(arr)                     # Square root
np.exp(arr)                      # Exponential
np.log(arr)                      # Natural logarithm
```

### Array Operations
```python
arr1 + arr2                      # Element-wise addition
arr1 * arr2                      # Element-wise multiplication
arr1 @ arr2                      # Matrix multiplication
np.dot(arr1, arr2)               # Dot product
```

---

## 7️⃣ STATISTICAL FUNCTIONS

```python
np.sum(arr)                      # Sum of all elements
np.mean(arr)                     # Mean (average)
np.median(arr)                   # Median
np.std(arr)                      # Standard deviation
np.var(arr)                      # Variance
np.min(arr)                      # Minimum value
np.max(arr)                      # Maximum value
np.argmin(arr)                   # Index of minimum
np.argmax(arr)                   # Index of maximum
```

### With Axis
```python
np.sum(arr, axis=0)              # Sum along rows (for each column)
np.mean(arr, axis=1)             # Mean along columns (for each row)
```

---

## 8️⃣ FILTERING & SORTING

### Filtering
```python
arr[arr > 5]                     # Values greater than 5
np.where(arr > 5, 1, 0)          # Replace: if >5 then 1, else 0
```

### Sorting
```python
np.sort(arr)                     # Sort array
np.argsort(arr)                  # Indices that would sort array
np.unique(arr)                   # Unique values
np.unique(arr, return_counts=True)  # Unique values with counts
```

---

## 9️⃣ RANDOM MODULE

```python
np.random.seed(42)               # Set random seed
np.random.rand(3, 4)             # Random floats [0, 1), shape (3,4)
np.random.randn(2, 3)            # Normal distribution, shape (2,3)
np.random.randint(1, 100, 10)    # Random integers from 1-99, 10 values
np.random.choice([1,2,3], 5)     # Random choice from array
np.random.shuffle(arr)           # Shuffle array in-place
np.random.normal(75, 10, 100)    # Normal dist: mean=75, std=10, n=100
```

---

## 🔟 LINEAR ALGEBRA

```python
np.dot(A, B)   or   A @ B        # Matrix multiplication
np.linalg.inv(A)                 # Matrix inverse
np.linalg.det(A)                 # Determinant
np.linalg.eig(A)                 # Eigenvalues and eigenvectors
np.linalg.solve(A, b)            # Solve Ax = b
np.linalg.norm(v)                # Vector norm (magnitude)
```

---

## 1️⃣1️⃣ BROADCASTING

```python
arr + 5                          # Scalar broadcast to all elements
arr + np.array([1, 2, 3])        # 1D broadcast to 2D
```

**Broadcasting Rules:**
1. Smaller array is "stretched" to match larger
2. Dimensions are compatible if equal or one is 1

---

## 1️⃣2️⃣ MISSING DATA

```python
np.nan                           # NaN constant
np.isnan(arr)                    # Check for NaN
np.nan_to_num(arr, nan=0)        # Replace NaN with 0
np.nanmean(arr)                  # Mean, ignoring NaN
np.nansum(arr)                   # Sum, ignoring NaN
np.nanmax(arr)                   # Max, ignoring NaN
```

---

## 1️⃣3️⃣ TYPE CONVERSION

```python
arr.astype(int)                  # Convert to integer
arr.astype(float)                # Convert to float
arr.astype(str)                  # Convert to string
arr.tolist()                     # Convert to Python list
```

---

## 1️⃣4️⃣ SET OPERATIONS

```python
np.intersect1d(arr1, arr2)       # Common elements
np.union1d(arr1, arr2)           # All unique elements
np.setdiff1d(arr1, arr2)         # In arr1 but not arr2
np.in1d(arr1, arr2)              # Check if elements in arr1 are in arr2
```

---

## 1️⃣5️⃣ COMMON PATTERNS

### Normalization (0-1 range)
```python
normalized = (arr - arr.min()) / (arr.max() - arr.min())
```

### Standardization (mean=0, std=1)
```python
standardized = (arr - arr.mean()) / arr.std()
```

### One-hot Encoding
```python
n_classes = 3
labels = np.array([0, 1, 2, 1, 0])
one_hot = np.eye(n_classes)[labels]
```

### Train-Test Split
```python
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

## 1️⃣6️⃣ PERFORMANCE TIPS

✅ **Use vectorization** instead of loops
```python
# Bad (slow)
for i in range(len(arr)):
    arr[i] = arr[i] * 2

# Good (fast)
arr = arr * 2
```

✅ **Pre-allocate arrays**
```python
# Bad
arr = np.array([])
for i in range(n):
    arr = np.append(arr, i)

# Good
arr = np.zeros(n)
for i in range(n):
    arr[i] = i

# Best
arr = np.arange(n)
```

✅ **Use views instead of copies**
```python
arr.ravel()      # View (faster)
arr.flatten()    # Copy (slower)
```

---

## 1️⃣7️⃣ INTEGRATION

### With Pandas
```python
import pandas as pd

# NumPy to Pandas
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Pandas to NumPy
arr = df.to_numpy()
```

---

## 🎯 Most Used Functions Summary

| Category | Functions |
|----------|-----------|
| **Creation** | `array`, `zeros`, `ones`, `arange`, `linspace` |
| **Stats** | `mean`, `std`, `min`, `max`, `sum` |
| **Reshape** | `reshape`, `flatten`, `T` |
| **Filter** | Boolean indexing, `where` |
| **Math** | `+`, `*`, `@`, `sqrt`, `exp` |
| **Random** | `rand`, `randn`, `randint`, `seed` |

---

## 💡 Remember

- **Indexing starts at 0**
- **Slicing is [start:stop:step]** (stop is exclusive)
- **axis=0** means along rows (down)
- **axis=1** means along columns (across)
- **Use vectorization** for performance
- **Check shapes** when debugging

---

**Keep this handy while coding with NumPy! 🚀**
