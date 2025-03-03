import os
import logging
import tensorflow as tf
from src.utils.config import instanciate_module

class BaseEvaluator:
    def __init__(self, model_path: str, parameters: dict):
        self.model_path = model_path
        self.parameters = parameters
        
        self.model = tf.keras.models.load_model(model_path)
        
        self.metric = instanciate_module(parameters['metric']['module_name'],
                                         parameters['metric']['class_name'],
                                         parameters['metric']['parameters'])
        
        self.model.compile(metrics=[self.metric])
    
    def run(self, test_ds):
        logging.info("Evaluating model with test dataset.")
        results = self.model.evaluate(test_ds, return_dict=True)
        logging.info(f"Evaluation results: {results}")
        return results
