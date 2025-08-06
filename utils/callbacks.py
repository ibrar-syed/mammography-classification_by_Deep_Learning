# utils/callbacks.py,.....

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from livelossplot.inputs.keras import PlotLossesCallback
from livelossplot.inputs.keras import PlotLossesCallback

def get_callbacks():
    """Returns standard training callbacks for model fitting."""
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=6,
        min_delta=1e-4,
        min_lr=1e-5,
        verbose=1
    )

    live_plot = PlotLossesCallback()

    return [early_stop, reduce_lr, live_plot]
