

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
# utils/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_true, y_pred, average='weighted'), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='weighted'), 4),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
    }
