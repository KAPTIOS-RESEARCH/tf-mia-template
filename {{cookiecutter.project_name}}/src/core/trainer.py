import wandb, os, logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from src.utils.config import instanciate_module

class BaseTrainer:
    def __init__(self, model: tf.keras.Model, parameters: dict, log_dir: str):
        self.model = model
        self.parameters = parameters
        self.log_dir = log_dir
        self.best_loss = float('inf')
        
        # OPTIMIZER
        self.optimizer = Adam(
            learning_rate=parameters['lr'],
            decay=parameters['weight_decay']
        )
        
        # LR SCHEDULER
        self.lr_scheduler = None
        lr_scheduler_type = parameters.get('lr_scheduler', 'none')
        
        if lr_scheduler_type == 'cosine':
            def cosine_decay(epoch, lr):
                return 0.5 * (1 + tf.math.cos(tf.constant(epoch) * 3.1415926535 / 100)) * parameters['lr']
            self.lr_scheduler = LearningRateScheduler(cosine_decay)
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        elif lr_scheduler_type == 'exponential':
            def exponential_decay(epoch, lr):
                return lr * 0.97
            self.lr_scheduler = LearningRateScheduler(exponential_decay)
        
        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                          parameters['loss']['class_name'],
                                          parameters['loss']['parameters'])
        
        # METRIC
        self.metric = instanciate_module(parameters['metric']['module_name'],
                                         parameters['metric']['class_name'],
                                         parameters['metric']['parameters'])
        
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=[self.metric])
        
        # CHECKPOINTING
        self.checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.log_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        self.early_stop = EarlyStopping(monitor='val_loss', patience=parameters['early_stopping_patience'], restore_best_weights=True)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.criterion,
            metrics=[self.metric],
        )
        
    def fit(self, train_ds, val_ds):
        num_epochs = self.parameters['num_epochs']
        callbacks = [self.early_stop, self.checkpoint_callback]
        
        if self.lr_scheduler:
            callbacks.append(self.lr_scheduler)
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )
        
        if self.parameters['track']:
            for epoch in range(num_epochs):
                wandb.log({f"Train/{self.parameters['loss']['class_name']}": history.history['loss'][epoch], '_step_': epoch})
                wandb.log({f"Test/{self.parameters['loss']['class_name']}": history.history['val_loss'][epoch], '_step_': epoch})
                wandb.log({f"Test/{self.parameters['metric']['class_name']}": history.history.get(self.parameters['metric']['class_name'], [None])[epoch], '_step_': epoch})
            wandb.finish()
        
        logging.info("Training complete.")
