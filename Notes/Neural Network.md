## Instantiating a model from an input tensor and a list of output tensors

```python
layer_outputs = [layer.output for layer in classifier.layers[:12]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
```

source: https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md