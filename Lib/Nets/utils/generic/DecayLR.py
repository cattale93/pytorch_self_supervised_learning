class DecayLR:
    """
    The learning scheduler requires a function that it is used to update the lr value
    This is the function used in the project models
    """
    def __init__(self, epochs, checkpoint_epoch, decay_epochs):
        self.epochs = epochs
        self.checkpoint_epoch = checkpoint_epoch
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.checkpoint_epoch - self.decay_epochs) / (self.epochs - self.decay_epochs)
