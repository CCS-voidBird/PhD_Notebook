import tensorflow as tf
from tensorflow.keras import layers, Model

class LearnablePatternEncoding(layers.Layer):
    """
    A custom TensorFlow layer that implements Learnable Pattern Encoding (LPE).
    LPE creates a 2D representation of sequences by learning base patterns and 
    position-specific weights to combine them.
    
    The layer learns two sets of parameters:
    1. Base patterns: A set of fundamental patterns that can be combined
    2. Position weights: Weights that determine how each position combines the base patterns
    
    Args:
        max_seq_length: Maximum length of input sequences
        pattern_dim: Dimension of the pattern vectors (width of output matrix)
        num_patterns: Number of base patterns to learn
        pattern_initializer: Initializer for base patterns (default: random normal)
        position_initializer: Initializer for position weights (default: random normal)
    """
    def __init__(
        self,
        max_seq_length,
        pattern_dim,
        num_patterns,
        pattern_initializer='random_normal',
        position_initializer='random_normal',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.pattern_dim = pattern_dim
        self.num_patterns = num_patterns
        self.pattern_initializer = pattern_initializer
        self.position_initializer = position_initializer

    def build(self, input_shape):
        # Initialize base patterns [num_patterns, pattern_dim]
        self.patterns = self.add_weight(
            name='patterns',
            shape=(self.num_patterns, self.pattern_dim),
            initializer=self.pattern_initializer,
            trainable=True
        )
        
        # Initialize position-specific weights [max_seq_length, num_patterns]
        self.position_weights = self.add_weight(
            name='position_weights',
            shape=(self.max_seq_length, self.num_patterns),
            initializer=self.position_initializer,
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: [batch_size, seq_length]
        
        # Generate position-specific patterns by combining base patterns
        # position_weights: [seq_length, num_patterns]
        # patterns: [num_patterns, pattern_dim]
        # Result: [seq_length, pattern_dim]
        position_patterns = tf.matmul(
            self.position_weights[:tf.shape(inputs)[1]], 
            self.patterns
        )
        
        # Combine input sequence with position patterns
        # inputs: [batch_size, seq_length] -> [batch_size, seq_length, 1]
        # position_patterns: [seq_length, pattern_dim] -> [1, seq_length, pattern_dim]
        encoded = tf.expand_dims(inputs, -1) * tf.expand_dims(position_patterns, 0)
        
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'pattern_dim': self.pattern_dim,
            'num_patterns': self.num_patterns,
            'pattern_initializer': self.pattern_initializer,
            'position_initializer': self.position_initializer
        })
        return config

# Example of a complete model using LPE
class SequencePredictionModel(Model):
    """
    Example model architecture that uses LPE for sequence prediction.
    
    Args:
        max_seq_length: Maximum sequence length
        pattern_dim: Dimension of the pattern vectors
        num_patterns: Number of base patterns
        num_outputs: Number of output values to predict
    """
    def __init__(
        self,
        max_seq_length,
        pattern_dim,
        num_patterns,
        num_outputs,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # LPE layer for sequence encoding
        self.lpe = LearnablePatternEncoding(
            max_seq_length=max_seq_length,
            pattern_dim=pattern_dim,
            num_patterns=num_patterns
        )
        
        # Processing layers
        self.lstm = layers.LSTM(64, return_sequences=True)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(num_outputs)

    def call(self, inputs):
        x = self.lpe(inputs)
        x = self.lstm(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.output_layer(x)

# Example usage
def create_example_model():
    # Model parameters
    max_seq_length = 100
    pattern_dim = 64
    num_patterns = 8
    num_outputs = 1
    
    # Create model
    model = SequencePredictionModel(
        max_seq_length=max_seq_length,
        pattern_dim=pattern_dim,
        num_patterns=num_patterns,
        num_outputs=num_outputs
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Example of training the model
def train_example_model():
    # Generate dummy data
    batch_size = 32
    seq_length = 50
    X = tf.random.normal((batch_size, seq_length))
    y = tf.random.normal((batch_size, 1))
    
    # Create and train model
    model = create_example_model()
    model.fit(X, y, epochs=5, batch_size=32)
    
    return model
