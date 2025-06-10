### Pipeline Reasoning:

1. **Custom Target Transformer:**
- Columns: city
- Reason: This categorical column needs to be represented numericaly and we can do so by target encoding. The smoothing is to help balance between using the global mean, and the category mean.

2. **Custom Mapping Transformer:**
- Columns: has_status, has_website, gender, has_birth_date, occupation_type_university, has_occupation, occupation_type_work, has_mobile, has_maiden_name, all_posts_visible, audio_available.
- Reasoning: This data has a lot of "Unknown" values across each of these columns. Given that this is a social media site, I'm assuming that this is mostly "Missing Not at Random" from private profiles. As such, it wouldn't make very much sense to try and fill in the missing values. Instead, I'm opting to encode the "Unknown" values as a differient number in order to keep the information.

3. **Custom Robust Transformer:**
- Columns: avg_views, posting_frequency_days, avg_likes
- Reasoning: We need to scale our numerical columns down. I'm using the robust scalar as we didn't trim any outliers (they might indicate bots!) and it's good at handling them.

4. **Custom KNN Transformer:**
- Columns: All with missing values!
- Reasoning: We want to fill in any missing values and we're going to do so by looking at similar rows. K=5 should balance between over and under fitting.

#### Execution Order Reasoning:
- The target encoding only cares about the cities column and the labels, neither of which will change later in the pipeline. Might as well do it first.
- The maping transformations won't impact eachother so the order doesn't really matter.
- We must scale the numerical columns before we impute them to fill in missing values!
- We impute last as it has the greatest number of side effects and will modify our remaining nan values in any of the columns.

#### Performance Considerations:
- Target transformer is the clear option as One-Hot-Encoding the cities would lead to too many new columns.
- Robust transformer will preserve some of the outliers which may be key to prediction later.
- Map transformer allows us to use a new value as a placeholder for "Unknown" which might be useful for our models.
- KNN imputation helps to preserve the relationships between features.