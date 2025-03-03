import wandb, os, logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.config import instanciate_module
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

class BaseTrainer:
    def __init__(self, model: tf.keras.Model, parameters: dict, log_dir: str):
        self.model = model
        self.parameters = parameters
        self.log_dir = log_dir
        
        # OPTIMIZER
        self.optimizer = Adam(
            learning_rate=parameters['lr'],
            decay=parameters['weight_decay']
        )
        
        # CRITERION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                          parameters['loss']['class_name'],
                                          parameters['loss']['parameters'])
        
        # METRIC
        self.metric = instanciate_module(parameters['metric']['module_name'],
                                         parameters['metric']['class_name'],
                                         parameters['metric']['parameters'])
        
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=[self.metric])

    def run(self, train_ds, val_ds):
        num_epochs = self.parameters['num_epochs']
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.parameters['early_stopping_patience'], restore_best_weights=True)]
        
        if self.parameters['track']:
            callbacks.append(WandbMetricsLogger())
            callbacks.append(WandbModelCheckpoint(
                filepath=os.path.join(self.log_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ))
        else:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.log_dir, 'best_model.keras'),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
            )
        
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )
        
        if self.parameters['track']:
            wandb.finish()
        
        logging.info("Training complete.")
