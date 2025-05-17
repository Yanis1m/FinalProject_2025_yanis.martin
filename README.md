<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Short Video Recommendation System

## Project Overview

This project implements a recommendation system for short videos using the KuaiRec dataset. The goal is to predict which videos users are likely to enjoy and generate personalized recommendations. The system leverages deep learning techniques to understand user preferences and video characteristics, similar to recommendation engines used in platforms like TikTok, Instagram Reels, or YouTube Shorts.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering](#feature-engineering)
    - [Model Architecture](#model-architecture)
    - [Training Process](#training-process)
- [Results](#results)
    - [Model Performance](#model-performance)
    - [Evaluation Metrics](#evaluation-metrics)
- [Discussion](#discussion)
    - [Challenges](#challenges)
    - [Insights](#insights)
- [Conclusion](#conclusion)
- [Future Work](#future-work)


## Dataset Description

The KuaiRec dataset is a large-scale collection of user-video interactions from the Kuaishou platform, a popular short video application. The dataset includes:

- **Big matrix**: 11.56M interactions between 7,176 users and 10,728 videos
- **Small matrix**: 4.49M interactions for evaluation
- **User features**: Demographic and behavioral information
- **Video features**: Technical attributes, engagement metrics, and content categories
- **Social network**: User social connections

The dataset has a sparsity of 0.866, meaning only about 13.4% of all possible user-video pairs have interactions. This high sparsity is typical of recommendation systems and presents a significant challenge.

*Figure 1: Distribution of confidence scores derived from user-video interactions*

## Methodology

### Data Preprocessing

The preprocessing phase focused on cleaning the data and preparing it for feature engineering:

1. **Data cleaning**: Removed missing values and duplicates from the interaction matrices
2. **Timestamp validation**: Filtered out interactions with negative timestamps
3. **Feature transformation**: Converted categorical features to numerical representations
4. **Metadata processing**: Parsed JSON-like structures in social network and video category data
```python
# Clean and preprocess data
def preprocess_data(df):
    """Clean and preprocess interaction data"""
    initial_size = len(df)
    # Remove missing values and duplicates
    df = df.dropna().drop_duplicates()
    # Remove negative timestamps (invalid data)
    df = df[df['timestamp'] > 0]
    final_size = len(df)
    print(f"Removed {initial_size - final_size} rows ({(initial_size - final_size)/initial_size:.2%})")
    return df
```

The preprocessing removed 7.71% of rows from the big matrix and 3.89% from the small matrix, ensuring data quality for subsequent steps.

### Feature Engineering

Feature engineering was crucial for capturing the complex patterns in user-video interactions:

1. **User-item mappings**: Created efficient index mappings between user/video IDs and their internal indices
2. **Engagement metrics**: Calculated watch ratio, completion rate, and replay factor
3. **Confidence scores**: Developed a sophisticated confidence calculation that balances precision and recall
4. **Dimensionality reduction**: Applied PCA to reduce feature dimensions while preserving 95% of variance
5. **Sequence features**: Created temporal sequences of user interactions for potential sequence-aware models
```python
# Calculate advanced confidence scores
base_confidence = 2.0  # Higher base value
alpha = 60  # Much higher weight for watch_ratio
beta = 0.8   # Higher weight for completion_rate
gamma = 0.5  # Higher weight for replay_factor

# Create the recall-focused confidence
interaction_agg['confidence_recall'] = base_confidence + \
                          alpha * np.log1p(interaction_agg['watch_ratio_capped']) * \
                          (1 + beta * interaction_agg['completion_rate']) * \
                          (1 + gamma * np.log1p(interaction_agg['replay_factor']))
```

The feature engineering process resulted in:

- 23 user features after PCA (from 26 original features)
- 51 item features after PCA (from 69 original features)
- Preserved variance: 96.4% for user features and 95.4% for item features


### Model Architecture

After exploring various approaches, I selected a Two-Tower neural network architecture for the recommendation system. This architecture has proven effective in large-scale recommendation systems at companies like YouTube and Google.

The Two-Tower model consists of:

1. **User Tower**: Processes user features through multiple dense layers
2. **Item Tower**: Processes video features through a parallel network
3. **Combination Layer**: Merges user and item embeddings to predict user-video affinity
```python
def create_two_tower_model(user_dim, item_dim, embedding_dim=64):
    # User tower
    user_input = Input(shape=(user_dim,), name='user_input')
    user_tower = Sequential([
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(embedding_dim, activation='relu', name='user_embedding'),
        L2NormalizationLayer()
    ], name='user_tower')
    user_embedding = user_tower(user_input)
    
    # Item tower
    item_input = Input(shape=(item_dim,), name='item_input')
    item_tower = Sequential([
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(embedding_dim, activation='relu', name='item_embedding'),
        L2NormalizationLayer()
    ], name='item_tower')
    item_embedding = item_tower(item_input)
    
    # Combine towers
    merged = Concatenate()([user_embedding, item_embedding])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=[user_input, item_input], outputs=output, name='TwoTowerRecommender')
    
    return model, user_tower, item_tower
```

Key design choices:

- **Embedding dimension**: 64 (balances expressiveness and computational efficiency)
- **Dropout layers**: 0.3 dropout rate to prevent overfitting
- **L2 normalization**: Applied to embeddings for better similarity calculations
- **Activation functions**: ReLU for hidden layers, linear for output


### Training Process

The model was trained using the following approach:

1. **Data sampling**: Used 2 million interactions for training and 2 million for testing
2. **Loss function**: Mean Squared Error (MSE) for watch ratio prediction
3. **Optimizer**: Adam with learning rate of 0.001
4. **Batch size**: 512 samples per batch
5. **Early stopping**: Monitored validation loss with patience of 5 epochs
6. **Model checkpointing**: Saved the best model based on validation loss

*Figure 2: Training and validation loss curves for the Two-Tower model*

The model converged after 7 epochs, with early stopping preventing overfitting. The training process showed a steady decrease in both training and validation loss.

## Results

### Model Performance

The Two-Tower model achieved impressive performance on the test set:

- **Mean Absolute Error (MAE)**: 0.4078
- **Baseline MAE** (mean prediction): 0.4866
- **Improvement**: 16.19% over baseline

This indicates that the model can predict watch ratios significantly better than a simple average-based approach.

### Evaluation Metrics

For recommendation quality, I evaluated the model using standard ranking metrics:


| Metric | Value |
| :-- | :-- |
| NDCG@25 | 0.9624 |
| Precision@25 | 0.8783 |
| Recall@25 | 0.0335 |

The high NDCG (Normalized Discounted Cumulative Gain) score of 0.9624 indicates that the model is excellent at ranking relevant videos higher in the recommendation list. The precision of 0.8783 shows that 87.83% of recommended videos are relevant to users (using a relevance threshold of 0.8 for watch ratio).

The relatively low recall of 0.0335 suggests that while the recommendations are highly precise, they only capture about 3.35% of all relevant videos for each user. This is a common trade-off in recommendation systems, where precision is often prioritized over recall.

## Discussion

### Challenges

Several challenges were encountered during this project:

1. **Data sparsity**: With a sparsity of 0.866, most user-video pairs have no interaction data
2. **Cold start problem**: New users or videos have limited or no historical data
3. **Computational efficiency**: Processing millions of interactions requires efficient implementations
4. **Balancing metrics**: Optimizing for both precision and recall is inherently challenging

### Insights

Key insights gained from this project:

1. **Feature importance**: User engagement metrics (watch ratio, completion rate) are the strongest predictors of user preferences
2. **Embedding power**: The Two-Tower architecture effectively learns meaningful user and video embeddings
3. **Precision-recall trade-off**: The model achieves high precision at the cost of lower recall
4. **Dimensionality reduction**: PCA significantly reduces feature dimensions while preserving predictive power

## Conclusion

This project successfully implemented a Two-Tower neural network model for short video recommendations. The model demonstrates strong performance in predicting user preferences and ranking videos, achieving high precision and NDCG scores.

The approach balances collaborative and content-based filtering by incorporating both user-video interaction patterns and rich feature representations. The model's architecture allows for efficient training and inference, making it suitable for large-scale recommendation systems.

## Future Work

Several directions for future improvement include:

1. **Sequence modeling**: Incorporate temporal patterns using RNNs or Transformers
2. **Attention mechanisms**: Add attention layers to better weight feature importance
3. **Multi-objective optimization**: Balance multiple objectives (e.g., watch time, engagement)
4. **Cold start handling**: Develop specialized approaches for new users and videos
5. **A/B testing**: Implement online evaluation to measure real-world performance

---

*This project was completed as part of the Recommender Systems course at [University Name], [Semester Year].*

<div style="text-align: center">⁂</div>

[^1]: data_preprocessing.ipynb

[^2]: feature_engineering.ipynb

[^3]: model_training.ipynb

