# Pipeline Reasoning

## 1. **Custom Target Transformer**

- **Column**: `city`
- **Purpose**: Applies smoothed target encoding to the city column.
- **Reasoning**: `city` is a high-cardinality categorical feature. Using smoothed target encoding reduces dimensionality and encodes predictive information. Smoothing helps balance category-specific means with the global mean to prevent overfitting, especially for rare cities.

## 2. **Custom Target Transformer for Binary & Categorical Features**

- **Columns**:  
  `has_status`, `has_website`, `gender`, `has_birth_date`, `occupation_type_university`,  
  `has_occupation`, `occupation_type_work`, `has_mobile`, `has_maiden_name`,  
  `all_posts_visible`, `audio_available`

- **Purpose**: Applies target encoding to multiple binary or categorical features.
- **Reasoning**: These columns often have "Unknown" or missing entries, likely **Missing Not at Random (MNAR)** due to private user settings on a social media platform. Instead of imputing these, we encode "Unknown" as a special category via target encoding, allowing models to learn from their presence.

## 3. **Custom Tukey Transformer**

- **Columns**: `avg_views`, `posting_frequency_days`, `avg_likes`
- **Purpose**: Detects and caps outliers using the Tukey method (outer fences).
- **Reasoning**: These numerical features can contain extreme values (e.g., viral users or bots). Rather than removing them, we transform them to reduce their influence without discarding the signal.

## 4. **Custom Robust Transformer**

- **Columns**: `avg_views`, `posting_frequency_days`, `avg_likes`
- **Purpose**: Applies robust scaling to make numerical features more stable and comparable.
- **Reasoning**: Since we preserve outliers using Tukey’s method, robust scaling (which uses medians and IQR) is more appropriate than standard scaling. It avoids being skewed by outliers and ensures numeric consistency for downstream models and imputers.

## 5. **Custom KNN Transformer**

- **Columns**: All columns with remaining missing values
- **Purpose**: Imputes missing values using K-Nearest Neighbors (K=7).
- **Reasoning**: After all encoding and scaling is complete, we apply KNN imputation to fill in any missing values using information from similar rows. `K=7` is chosen to provide a balance between local similarity and generalization, suitable for a moderately large dataset.

---

---

# Execution Order Reasoning

1. **Target Encoding (city)**  
   Done first since it relies on the original categorical values and target labels.

2. **Mapping via Target Encoding (binary/categorical flags)**  
   Also done early, preserving "Unknown" states as informative signals.

3. **Outlier Handling via Tukey Transform**  
   Applied before scaling to adjust extreme values.

4. **Robust Scaling of Numerical Features**  
   Ensures consistent numeric ranges across features for later KNN-based distance calculations.

5. **KNN Imputation (K=7)**  
   Performed last to fill in any remaining missing values after all other transformations.

# Performance Considerations

- **Target Encoding vs One-Hot**: Too many unique values to OHE all of the features, target encoding will help with generalization.
- **Unknown Preservation via Target Encoding**: Keeps MNAR "Unknown" values informative rather than trying to impute or drop them.
- **Tukey + Robust Scaling**: Smooths outliers without removing them, which may be critical (e.g., for identifying bots or influencers).
- **KNN Imputation (K=7)**: 7 seems like a reasonable number given the size of our dataset. This could be tuned in the future.
