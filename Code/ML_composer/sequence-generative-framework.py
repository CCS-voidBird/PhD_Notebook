import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class SequenceEncoder(layers.Layer):
    """
    Encodes long sequences into 2D matrices using attention and spatial transformation.
    """
    def __init__(self, output_shape=(64, 64), embedding_dim=16):
        super().__init__()
        self.output_shapes = output_shape
        self.embedding_dim = embedding_dim
        self.output_dim = output_shape[0] * output_shape[1]
        
        # Sequence processing layers
        self.embedding = layers.Dense(embedding_dim)
        self.attention = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=embedding_dim
        )
        
        # Transformation layers
        self.transform_layers = [
            layers.Dense(64, activation='gelu'),
            layers.Dense(64, activation='gelu'),
            #layers.Dense(1024, activation='gelu'),
            layers.Dense(self.output_dim, activation='linear')
        ]
        
        self.norm_layers = [
            layers.LayerNormalization() for _ in range(len(self.transform_layers))
        ]

    def build(self, input_shape):
        # Position encoding
        self.pos_encoding = self.add_weight(
            "position_encoding",
            shape=(1, input_shape[1], self.embedding_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs, training=False):
        # Embed sequence
        x = self.embedding(inputs[..., tf.newaxis])
        x = x + self.pos_encoding
        
        # Apply attention
        x = self.attention(x, x)
        
        # Transform through dense layers
        for dense, norm in zip(self.transform_layers, self.norm_layers):
            x = dense(x)
            x = norm(x)
        
        # Extract and reshape final output
        return tf.reshape(x[:, -1, :], (-1, *self.output_shapes))

class ShortSequenceDecoder(layers.Layer):
    """
    Decodes short sequences into 2D matrices using pattern expansion.
    """
    def __init__(self, output_shape=(64, 64)):
        super().__init__()
        self.output_shapes = output_shape
        self.output_dim = output_shape[0] * output_shape[1]
        
        # Pattern generation layers
        self.generator = [
            layers.Dense(64, activation='gelu'),
            layers.Dense(64, activation='gelu'),
            #layers.Dense(512, activation='gelu'),
            #layers.Dense(1024, activation='gelu'),
            layers.Dense(self.output_dim, activation='linear')
        ]
        
        self.norm_layers = [
            layers.LayerNormalization() for _ in range(len(self.generator))
        ]

    def build(self, input_shape):
        pass


    def call(self, inputs, training=False):
        x = inputs
        
        # Transform through generator layers
        for dense, norm in zip(self.generator, self.norm_layers):
            x = dense(x)
            x = norm(x)
        
        return tf.reshape(x, (-1, *self.output_shapes))

class GenerativeFramework(Model):
    """
    Complete framework for sequence-to-matrix generation and training.
    """
    def __init__(self, matrix_shape=(64, 64)):
        super().__init__()
        self.encoder = SequenceEncoder(matrix_shape)
        self.decoder = ShortSequenceDecoder(matrix_shape)
        
        # Additional processing
        self.final_norm = layers.LayerNormalization()
        
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_tracker = keras.metrics.Mean(name="mse_loss")
        self.similarity_tracker = keras.metrics.Mean(name="similarity_loss")

    def compute_similarity_loss(self, matrix1, matrix2):
        """
        Computes similarity-based loss between matrices.
        """
        # Normalize matrices
        m1_norm = tf.nn.l2_normalize(matrix1, axis=[1, 2])
        m2_norm = tf.nn.l2_normalize(matrix2, axis=[1, 2])
        
        # Compute cosine similarity
        similarity = tf.reduce_sum(m1_norm * m2_norm, axis=[1, 2])
        return 1 - tf.reduce_mean(similarity)

    def train_step(self, data):
        long_seq, short_seq = data
        
        with tf.GradientTape() as tape:
            # Generate matrices
            encoded_matrix = self.encoder(long_seq)
            decoded_matrix = self.decoder(short_seq)
            
            # Compute losses
            mse_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(encoded_matrix, decoded_matrix)
            )
            similarity_loss = self.compute_similarity_loss(encoded_matrix, decoded_matrix)
            
            # Combine losses
            total_loss = mse_loss + 0.1 * similarity_loss
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.similarity_tracker.update_state(similarity_loss)
        
        return {
            "loss": self.loss_tracker.result(),
            "mse_loss": self.mse_tracker.result(),
            "similarity_loss": self.similarity_tracker.result()
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker, self.similarity_tracker]


def create_training_data(batch_size, long_seq_length, short_seq_length):
    """
    Creates training data for the generative framework.
    """
    long_sequences = tf.random.normal((batch_size, long_seq_length))
    short_sequences = tf.random.normal((batch_size, short_seq_length))
    return long_sequences, short_sequences

def train_model(epochs=30, batch_size=8, long_seq_length=1000, short_seq_length=8):
    """
    Sets up and trains the generative framework.
    """
    # Create model
    model = GenerativeFramework()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        run_eagerly=True
    )
    
    # Create training data
    data = create_training_data(
        batch_size, long_seq_length, short_seq_length
    )
    print(data[1].shape)
    
    # Train model
    history = model.fit(
        data[0],data[1],
        epochs=epochs,
        batch_size=batch_size,verbose=2
    )

    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print(history.history)