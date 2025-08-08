# utils/callbacks.py,.....
# data/dataset_loader.py
# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
