# Custom Learning Rate Scheduler

This project implements a custom learning rate schedule for training deep learning models, particularly transformers. The schedule adjusts the learning rate dynamically to improve training performance.

## Features
- Custom learning rate scheduling based on model dimension and training steps.
- Designed for TensorFlow and Keras.
- Used in transformer-based models.

## How to Use
1. Import the `CustomSchedule` class.
2. Create an instance with your model's `d_model` value.
3. Use it as the learning rate in an Adam optimizer.

```python
learning_rate = CustomSchedule(d_model=512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Requirements
Python 3.x
TensorFlow 2.x