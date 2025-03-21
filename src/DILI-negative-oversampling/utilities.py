import json
import lzma
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Iterable, Iterator
from warnings import simplefilter

import matplotlib.pyplot as plt
import natsort
import networkx as nx
import numpy as np
import pandas as pd
import prodec
import torch
import xgboost
from PyComplexHeatmap import *
from imblearn.over_sampling import SMOTE
from matplotlib.transforms import TransformedBbox, Bbox
from mordred import Calculator, descriptors
from pandarallel import pandarallel
from papyrus_scripts import PapyrusDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import (r2_score, mean_squared_error, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, explained_variance_score, max_error, mean_absolute_error,
                             mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance,
                             multilabel_confusion_matrix)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import trange, tqdm
from tqdm.std import tqdm as std_tqdm, trange as std_trange
from xgboost import XGBClassifier

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

_ = np.seterr(all='ignore')


NOTEBOOK_CONTEXT = tqdm != std_tqdm


def serialize_standard_scaler(model):
    serialized_model = {
        'meta': 'standard-scaler',
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }
    if model.var_ is None:
        serialized_model['var_'] = model.var_
    else:
        serialized_model['var_'] = model.var_.tolist()
    if model.mean_ is None:
        serialized_model['mean_'] = model.mean_
    else:
        serialized_model['mean_'] = model.mean_.tolist()
    if isinstance(model.scale_, np.ndarray):
        serialized_model['scale_'] = model.scale_.tolist()
    else:
        serialized_model['scale_'] = model.scale_,
    if isinstance(model.n_samples_seen_, (int, float)):
        serialized_model['n_samples_seen_'] = model.n_samples_seen_
    else:
        serialized_model['n_samples_seen_'] = model.n_samples_seen_.tolist()
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    return serialized_model


def deserialize_standard_scaler(model_dict):
    model = StandardScaler(**model_dict['params'])
    model.n_features_in_ = model_dict['n_features_in_']
    if isinstance(model_dict['mean_'], list):
        model.mean_ = np.array(model_dict['mean_'])
    else:
        model.mean_ = model_dict['mean_']
    if isinstance(model_dict['var_'], list):
        model.var_ = np.array(model_dict['var_'])
    else:
        model.var_ = model_dict['var_']
    if isinstance(model_dict['scale_'], list):
        model.scale_ = np.array(model_dict['scale_'])
    else:
        model.scale_ = model_dict['scale_']
    if isinstance(model_dict['n_samples_seen_'], list):
        model.n_samples_seen_ = np.array(model_dict['n_samples_seen_'])
    else:
        model.n_samples_seen_ = model_dict['n_samples_seen_']
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    return model


def standard_scaler_to_json(model, out_file):
    with open(out_file, 'w') as oh:
        json.dump(serialize_standard_scaler(model), oh)


def standard_scaler_from_json(json_file):
    with open(json_file, 'r') as fh:
        model = deserialize_standard_scaler(json.load(fh))
    return model


def plot_heatmap(data: pd.DataFrame, variable: str = 'GFP_pos1i', folder: str = '', colormap: str = 'RdYlBu'):
    data = pd.concat([data.iloc[:, :5], data[['binaryDILI']], data.iloc[:, 5:-1]], axis=1)

    data = data.loc[:, data.columns.str.endswith(variable) | data.columns.isin(data.columns[:6])]
    data['binaryDILI'] = data['binaryDILI'].replace({0: 'Negative', 1: 'Positive'})
    data.index = data.Name.str.lower()


    data.columns = data.columns[:6].tolist() + pd.Series(data.columns[6:]).str.split('_').apply(lambda x: f"{x[0]}_{x[2]}_{x[1]}_{'_'.join(x[3:])}").tolist()
    data = data[data.columns[:6].tolist() + natsort.natsorted(data.columns[6:])]
    data['binaryDILI'] = pd.Categorical(data.binaryDILI, categories=["Negative", "Positive"], ordered=True)

    melt = pd.DataFrame(pd.Series(data.columns[6:]).str.split('_').apply(lambda x: x[:3]).values.tolist(), columns=['Cell line', 'Time point', 'Concentration'])
    melt = melt.replace({'tp24': '24', 'tp48': '48', 'tp72': '72'})
    melt = melt.replace({'cmax1': '1 CMAX', 'cmax5': '5 CMAX', 'cmax10': '10 CMAX', 'cmax25': '25 CMAX', 'cmax50': '50 CMAX', 'cmax100': '100 CMAX'})
    melt['Cell line'] = melt['Cell line'].str.upper()
    melt['Cell line'] =  pd.Categorical(melt['Cell line'], ordered=True, categories=['SRXN1', 'HMOX1', 'CHOP', 'BIP', 'P21', 'BTG2', 'HSPA1B', 'ICAM1'])

    melt.index = data.columns[6:]

    col_annot =  HeatmapAnnotation(**{'Cell line': anno_simple(melt['Cell line'], add_text=False, legend=True, height=5,
                                                              cmap='Accent'),
                                      'Time point': anno_simple(melt['Time point'], add_text=False,legend=True, height=5,
                                                                colors=dict(zip(['24', '48', '72'],
                                                                                ['#f5c7e2', '#c82783', '#8e0152']))),
                                      'Concentration': anno_simple(melt.Concentration, add_text=False, legend=True, height=5,
                                                                   colors=dict(zip(['1 CMAX', '5 CMAX', '10 CMAX', '25 CMAX', '50 CMAX', '100 CMAX'],
                                                                                   ['#eef9fb', '#ccdfed', '#abc5dc', '#94a1cd', '#8b7dba', '#8955a7']
                                                                                   ))),
                                      },
                                   axis=1,
                                   label_kws={'visible': False},
                                   legend=False,
                                   legend_gap=5,
                                   hgap=0.5,
                                   verbose=0)


    row_annot = HeatmapAnnotation(**{'DILI label': anno_simple(data.binaryDILI.astype(str),
                                                               legend=True,
                                                               add_text=True,
                                                               colors=dict(zip(['Negative', 'Positive'],
                                                                               ["#eef9fb", "#31a260"]))),
                                    },
                                  axis=0,
                                  label_kws={'visible': False},
                                  legend=False,
                                  verbose=0)

    fig = plt.figure(figsize=(32, 25))
    cm = ClusterMapPlotter(data=data.iloc[:, 6:],
                           top_annotation=col_annot,
                           left_annotation=row_annot,
                           col_cluster=False,
                           col_split=melt['Cell line'],
                           col_split_order=['HMOX1', 'SRXN1', 'CHOP', 'BIP', 'P21', 'BTG2', 'HSPA1B', 'ICAM1'],
                           row_cluster=True,
                           row_dendrogram_size=50,
                           row_split=data.binaryDILI,
                           row_split_gap=5,
                           label='GFP intensity',
                           row_dendrogram=True,
                           show_rownames=True,
                           show_colnames=False,
                           cmap=colormap,
                           legend_gap=5,
                           legend_width=5,
                           legend_hpad=7,
                           legend_vpad=5,
                           verbose=0
                          )

    bbox = fig.get_tightbbox()
    bbox_ext = TransformedBbox(Bbox(bbox._bbox._points + np.array([[-30, 0], [50, 30]])), bbox._transform)
    plt.savefig(os.path.join(folder, f'Heatmap_{variable}.png'), bbox_inches=bbox_ext)
    plt.savefig(os.path.join(folder, f'Heatmap_{variable}.svg'), bbox_inches=bbox_ext)
    with open(os.path.join(folder, f'Heatmap_{variable}_molecule_order.json'), 'w') as oh:
        json.dump(cm.row_order, oh)
    plt.close()


def plot_figure_heatmap(data: pd.DataFrame, melt: pd.DataFrame, folder: str = '', colormap: str = 'RdYlBu'):
    col_annot = HeatmapAnnotation(**{'Cell line': anno_simple(melt['Cell line'], add_text=False, legend=True, height=5,
                                                              cmap='Accent'),
                                     'Time point': anno_simple(melt['Time point'], add_text=False, legend=True,
                                                               height=5,
                                                               colors=dict(zip(['24', '48', '72'],
                                                                               ['#f5c7e2', '#c82783', '#8e0152']))),
                                     'Concentration': anno_simple(melt.Concentration, add_text=False, legend=True,
                                                                  height=5,
                                                                  colors=dict(
                                                                      zip(['1 CMAX', '5 CMAX', '10 CMAX', '25 CMAX',
                                                                           '50 CMAX', '100 CMAX'],
                                                                          ['#eef9fb', '#ccdfed', '#abc5dc', '#94a1cd',
                                                                           '#8b7dba', '#8955a7']
                                                                          ))),
                                     },
                                  axis=1,
                                  label_kws={'visible': False},
                                  legend=False,
                                  legend_gap=5,
                                  hgap=0.5,
                                  verbose=0)

    row_annot = HeatmapAnnotation(**{'DILI label': anno_simple(data.binaryDILI.astype(str),
                                                               legend=True,
                                                               add_text=True,
                                                               colors=dict(zip(['Negative', 'Positive'],
                                                                               ["#eef9fb", "#31a260"]))),
                                     },
                                  axis=0,
                                  label_kws={'visible': False},
                                  legend=False,
                                  verbose=0)

    fig = plt.figure(figsize=(32, 25))
    cm = ClusterMapPlotter(data=data.iloc[:, 6:],
                           top_annotation=col_annot,
                           left_annotation=row_annot,
                           col_cluster=False,
                           col_split=melt['Cell line'],
                           col_split_order=['HMOX1', 'SRXN1', 'CHOP', 'BIP', 'P21', 'BTG2', 'HSPA1B', 'ICAM1'],
                           row_cluster=True,
                           row_dendrogram_size=50,
                           row_split=data.binaryDILI,
                           row_split_gap=5,
                           label='GFP intensity',
                           row_dendrogram=True,
                           show_rownames=True,
                           show_colnames=False,
                           cmap=colormap,
                           legend_gap=5,
                           legend_width=5,
                           legend_hpad=7,
                           legend_vpad=5,
                           verbose=0
                           )

    bbox = fig.get_tightbbox()
    bbox_ext = TransformedBbox(Bbox(bbox._bbox._points + np.array([[-30, 0], [50, 30]])), bbox._transform)
    plt.savefig(os.path.join(folder, f'Heatmap_{colormap}.png'), bbox_inches=bbox_ext)
    plt.savefig(os.path.join(folder, f'Heatmap_{colormap}.svg'), bbox_inches=bbox_ext)
    with open(os.path.join(folder, f'Heatmap_molecule_order.json'), 'w') as oh:
        json.dump(cm.row_order, oh)
    plt.close()


def getMetrics(model, y_true, X_test, continuous=True):
    """Determine performance metrics for a model.

    :param model: The model to evaluate which must have a `predict` and (optionally) a `predict_proba` method.
    :param y_true: The set of ground truth values.
    :param X_test: Input features to obtain predictions from.
    :param continuous: Indicates whether the supplied model is a regressor (True) or a classifier (False).
    """
    y_pred = model.predict(X_test)
    if continuous:
        return {'number': y_true.size,
                'R2': r2_score(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred, squared=True),
                'RMSE': mean_squared_error(y_true, y_pred, squared=False),
                'MSLE': mean_squared_log_error(y_true, y_pred),
                'RMLSE': np.sqrt(mean_squared_log_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'Explained variance': explained_variance_score(y_true, y_pred),
                'Max error': max_error(y_true, y_pred),
                'Mean Poisson distrib': mean_poisson_deviance(y_true, y_pred),
                'Mean Gamma distrib': mean_gamma_deviance(y_true, y_pred),
                'Spearman r': stats.spearmanr(y_true, y_pred)[0],
                'Kendall tau': stats.kendalltau(y_true, y_pred)[0]
                }
    else:  # classifier
        values = {}
        if len(model.classes_) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                values['MCC'] = mcc
            except RuntimeWarning:
                values['MCC'] = 0
            values[':'.join(str(class_) for class_ in model.classes_)] = ':'.join(
                [str(int(sum(y_true == class_))) for class_ in model.classes_])
            values['Acc'] = (tp + tn) / (tp + tn + fp + fn)
            values['BAcc'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
            values['Sen'] = tp / (tp + fn) if tp + fn != 0 else 0
            values['Spe'] = tn / (tn + fp) if tn + fp != 0 else 0
            values['PPV'] = tp / (tp + fp) if tp + fp != 0 else 0
            values['NPV'] = tn / (tn + fn) if tn + fn != 0 else 0
            values['F1'] = 2 * values['Sen'] * values['PPV'] / (values['Sen'] + values['PPV'])
            if hasattr(model, 'predict_proba'):  # able to predict probability
                y_probas = model.predict_proba(X_test)
                for i in range(len(model.classes_)):
                    y_proba = y_probas[:, i].ravel()
                    values[f'AUC {model.classes_[i]}'] = roc_auc_score(y_true, y_proba)
        else:  # multiclass
            i = 0
            for contingency_matrix in multilabel_confusion_matrix(y_true, y_pred):
                tn, fp, fn, tp = contingency_matrix.ravel()
                class_ = model.classes_[i]
                try:
                    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn * fn))
                    values['MCC'] = mcc
                except RuntimeWarning:
                    values['MCC'] = 0
                values[f'{class_}|number'] = int(sum(y_true == model.classes_[i]))
                values[f'{class_}|Acc'] = (tp + tn) / (tp + tn + fp + fn)
                values[f'{class_}|BAcc'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
                values[f'{class_}|Sen'] = tp / (tp + fn) if tp + fn != 0 else 0
                values[f'{class_}|Spe'] = tn / (tn + fp) if tn + fp != 0 else 0
                values[f'{class_}|PPV'] = tp / (tp + fp) if tp + fp != 0 else 0
                values[f'{class_}|NPV'] = tn / (tn + fn) if tn + fn != 0 else 0
                values[f'{class_}|F1'] = 2 * values[f'{class_}|Sen'] * values[f'{class_}|PPV'] / (
                        values[f'{class_}|Sen'] + values[f'{class_}|PPV'])
                i += 1
            if hasattr(model, 'predict_proba'):  # able to predict probability
                y_probas = model.predict_proba(X_test)
                values[f'AUC 1 vs 1'] = roc_auc_score(y_true, y_probas, average='macro', multi_class='ovo')
                values[f'AUC 1 vs all'] = roc_auc_score(y_true, y_probas, average='macro', multi_class='ovr')
        return values


def list2dict(l):
    """Convert a list of dicts into a dict of lists"""
    values = defaultdict(list)
    for dict_ in l:
        for key in dict_.keys():
            values[key].append(dict_[key])
    return dict(values)


def sum_classes(classes):
    """Return the sum of colon-separated list values."""
    if ':' in str(classes[0]):
        return ':'.join(map(str, map(sum, zip(*[map(int, class_.split(':')) for class_ in classes]))))
    return sum(classes)


def cv_estimate(model, continuous, X_train, y_train, n_splits, stratify, shuffle, random_state):
    """Build a model with (stratified) cross-validation and give records of fold performances."""
    if n_splits is not None:  # do CV
        if stratify:
            folds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        val_scores = []
        for train, test in folds.split(X_train, y_train):
            model.fit(X_train.iloc[train, :], y_train[train])
            val_scores.append(getMetrics(model, y_train[test], X_train.iloc[test, :], continuous))
        # fit model on complete dataset
        model.fit(X_train, y_train)
        return val_scores
    else:
        if shuffle:
            X_train = X_train.sample(frac=1, axis=0, random_state=random_state).reset_index(drop=True)
            np.random.seed(random_state)
            np.random.shuffle(y_train)
        return getMetrics(model, y_train, X_train, continuous)


def oversample_multiple_times(times, dataset_X, dataset_Y,
                              model, splits=5,
                              continuous=False, stratify=False, shuffle=False,
                              random_state=0, pbar: bool = True,
                              leave_pbar: bool = True, force_console_pbar: bool = False):
    """Oversample data using SMOTE.

    :param times: number of SMOTE rounds
    :param dataset_X: feature matrix
    :param dataset_Y: label vector
    :param model: sklearn model
    :param split: number of cross validation splits
    :param pbar: if True display a progress bar
    :param leave_pbar: if False, the progress bar is removed when completed
    :param force_console_pbar: if True and used in a notebook, uses the console progress bar.
    """
    results = []
    for i in (std_trange if force_console_pbar else trange)(times, leave=leave_pbar,
                                                            ncols=(None if NOTEBOOK_CONTEXT else 90),
                                                            disable=(not pbar)):
        oversample = SMOTE(random_state=random_state)
        balanced_X, balanced_Y = oversample.fit_resample(dataset_X, dataset_Y)
        model.random_state = random_state
        result = pd.DataFrame.from_dict(
            cv_estimate(model, continuous=continuous,
                        X_train=balanced_X, y_train=balanced_Y,
                        n_splits=splits, stratify=stratify, shuffle=shuffle, random_state=random_state
                        ))
        result.insert(0, 'model', model)
        results.append(result)
        random_state += 1
    return pd.concat(results, axis=0)


class MLPRegressor(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self, in_features: int, dropout=0.1):
        """Instantiate a MLP regressor with `in_features` input nodes,
        8000, 4000, and 2000 hidden nodes and 1 output node.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, 8000)
        self.fc2 = nn.Linear(8000, 4000)
        self.fc3 = nn.Linear(4000, 2000)
        self.fc4 = nn.Linear(2000, 1)
        self.act = nn.functional.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dropout: bool = False):
        """Forward pass."""
        res = self.act(self.fc1(x))
        if dropout:
            res = self.dropout(res)
        res = self.act(self.fc2(res))
        if dropout:
            res = self.dropout(res)
        res = self.act(self.fc3(res))
        if dropout:
            res = self.dropout(res)
        res = self.fc4(res)
        return res

    @property
    def num_params(self):
        """Number of parameters."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


def prepare_biospectra_inputs(smiles, sequences: list[str] = None,
                              know_sequences: bool = True,
                              pairing: list[tuple[str, str]] = None,
                              data_folder: str = None,
                              verbose: bool = True
                              ) -> pd.DataFrame:
    """Prepare inputs to train or predict from a PCM model.

    :param smiles: list of SMILES of molecules
    :param sequences: list of protein sequences to combine with
    :param know_sequences: If True, the parameter `sequences` is ignored and a default set of proteins is used instead.
    Ignored when `pairing` is None.
    :param pairing: List of SMILES and protein sequences pairs.
    If None, the cross combination of SMILES and sequences is returned (used for predictions), otherwise limit
    the combination of molecular and protein descriptors to these given SMILES and sequence pairs (used for training).
    :param data_folder: Folder containing the file 'Papyrus_05-7_BioSpectra_protein-descriptors-ZscalesVanWesten.tsv.gz'.
    Ignored if `know_sequences` is False.
    :param verbose: Should progress be displayed for molecular descriptor calculation.
    :return: Molecular and protein descriptors to be used to train the PCM model or obtain predictions from it.
    First columns are 'SMILES' and either 'sequence', if `known_sequences=True`, or 'target_id' otherwise.
    """
    # Obtain molecular descriptors
    moldescs = Calculator(descriptors, ignore_3D=True).pandas([Chem.MolFromSmiles(x) for x in np.unique(smiles)],
                                                              quiet=(not verbose))
    # Replace NaNs by 0
    moldescs.fillna(0, inplace=True)
    # Remove aberrant values
    moldescs = moldescs.mask(moldescs.abs() > 1e5, 0)
    # Round
    moldescs = moldescs.round(5)
    moldescs.insert(0, 'SMILES', np.unique(smiles))
    # Obtain protein descriptors
    if know_sequences:
        protdescs = pd.read_csv(f'{data_folder}/Papyrus_05-7_BioSpectra_protein-descriptors-ZscalesVanWesten.tsv.gz',
                                sep='\t')
        merged = moldescs.merge(protdescs, how='cross')
        del moldescs, protdescs
        merged = merged[['SMILES', 'target_id'] + [col for col in merged.columns if col not in ['SMILES', 'target_id']]]
    else:
        assert sequences is not None, "Sequences must be provided when `know_sequences` is False."
        factory = prodec.ProteinDescriptors()
        desc = factory.get_descriptor('Zscale van Westen')
        transform = prodec.Transform('AVG', desc)
        seqs = [re.sub('(?![ACDEFGHIKLMNPQRSTVWY]).', '-', x) for x in np.unique(sequences)]
        protdescs = transform.pandas_get(seqs, domains=50, quiet=True)
        protdescs = pd.concat((pd.Series(np.unique(sequences), name='sequence'), protdescs.round(5)), axis=1)
        # Pair the descriptors with inputs
        if pairing is None:
            merged = moldescs.merge(protdescs, how='cross')
        else:
            merged = (pd.DataFrame.from_records(pairing, columns=['SMILES', 'sequence'])
                      .merge(moldescs, on='SMILES')
                      .merge(protdescs, on='sequence')
                      )
        del moldescs, protdescs
        merged = merged[['SMILES', 'sequence'] + [col for col in merged.columns if col not in ['SMILES', 'sequence']]]
    return merged

def prepare_biospectra_inputs_from_papyrus(papyrus_dataset: PapyrusDataset,
                                           know_sequences: bool = True,
                                           data_folder: str = None) -> Iterator[pd.DataFrame]:
    """Prepare inputs to train or predict from a PCM model.

    :param papyrus_dataset: Instance of a `PapyrusDataset` to obtain bioactivity inputs from.
    :param know_sequences: If True, the parameter `sequences` is ignored and a default set of proteins is used instead.
    :param data_folder: Folder containing the file 'Papyrus_05-7_BioSpectra_protein-descriptors-ZscalesVanWesten.tsv.gz'.
    Ignored if `know_sequences` is False.
    :return: Molecular and protein descriptors to be used to train the PCM model or obtain predictions from it.
    First columns are 'connectivity' and 'target_id'.
    """
    if papyrus_dataset.papyrus_params['chunksize'] is None:
        raise ValueError('PapyrusDataset must have a given chunksize for molecular descriptors to be obtained.')
    # Obtain molecular descriptors
    moldescs = pd.concat([x for x in papyrus_dataset.molecular_descriptors('mordred', progress=False)])
    connectivity, moldescs = moldescs['connectivity'], moldescs.drop(columns=['connectivity'])
    # Replace NaNs by 0
    moldescs.fillna(0, inplace=True)
    # Remove aberrant values
    moldescs = moldescs.mask(moldescs.abs() > 1e5, 0)
    # Round
    moldescs = moldescs.round(5)
    moldescs.insert(0, 'connectivity', connectivity) #papyrus_dataset.to_dataframe().SMILES
    # Obtain protein descriptors
    if know_sequences:
        protdescs = pd.read_csv(f'{data_folder}/Papyrus_05-7_BioSpectra_protein-descriptors-ZscalesVanWesten.tsv.gz',
                                sep='\t')
        # Ensure that the chunksize is used for the size of the output after merging
        chunksize = papyrus_dataset.papyrus_params['chunksize'] / int(len(protdescs))
        for i in range(0, len(moldescs), chunksize ):
            merged = moldescs.iloc[i: i+chunksize].merge(protdescs, how='cross')
            merged = merged[['connectivity', 'target_id'] + [col for col in merged.columns if col not in ['connectivity', 'target_id']]]
            yield merged
        del moldescs, protdescs
    else:
        sequences = papyrus_dataset.proteins().to_dataframe().drop_duplicates(subset='Sequence')
        factory = prodec.ProteinDescriptors()
        desc = factory.get_descriptor('Zscale van Westen')
        transform = prodec.Transform('AVG', desc)
        seqs = [re.sub('(?![ACDEFGHIKLMNPQRSTVWY]).', '-', x) for x in sequences.Sequence]
        protdescs = transform.pandas_get(seqs, domains=50, quiet=True)
        protdescs = pd.concat((sequences.target_id, protdescs.round(5)), axis=1)
        # Pair the descriptors with inputs
        chunksize = papyrus_dataset.papyrus_params['chunksize']
        for i in range(0, len(papyrus_dataset.papyrus_bioactivity_data), chunksize):
            merged = (papyrus_dataset.papyrus_bioactivity_data.iloc[i : i+chunksize][['connectivity', 'target_id']]
                      .merge(moldescs, on='connectivity')
                      .merge(protdescs, on='target_id')
                      )
            merged = merged[['connectivity', 'target_id'] + [col for col in merged.columns if col not in ['connectivity', 'target_id']]]
            yield merged
        del moldescs, protdescs


def make_biospectra_predictions(df: pd.DataFrame, model_prefix: str, batch_size: int = 4096,
                                verbose: bool = True) -> pd.DataFrame:
    """Run specified data through an `MLPRegressor` with specified model weights.

    :param df: Features to run the prediction on.
    :param model_prefix: Path with filename stem to load model weights, feature scaler and selector from.
    :param batch_size: Size of chunks of data to be fed to the model for inference.
    :param verbose: Display progress over the set of inputs (True).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load scaler
    model_scaler = f'{model_prefix}_scaler.json'
    scaler = standard_scaler_from_json(model_scaler)
    # Feature variance remover
    model_novar = f'{model_prefix}_novar.json'
    with open(model_novar) as fh:
        novar = json.load(fh)
    # Load DNN model weights
    model_weight_file = f'{model_prefix}.pkg'
    model_weights = torch.load(model_weight_file, map_location=device)
    # Prepare model
    mlp = MLPRegressor(model_weights['fc1.weight'].shape[1]).to(device)
    mlp.load_state_dict(model_weights)
    # mlp = torch.compile(mlp)
    # Iterate over batches
    res = []
    for i in trange(0, len(df), batch_size, ncols=(None if NOTEBOOK_CONTEXT else 90), disable=(not verbose)):
        batch = df.iloc[i: i + batch_size]
        # Scale data
        if (~pd.Series(scaler.feature_names_in_).isin(batch.columns)).any():
            raise ValueError(f'missing columns from dataframe: {scaler.feature_names_in_}')
        batch = batch.loc[:, scaler.feature_names_in_]
        batch = pd.DataFrame(scaler.transform(batch), columns=scaler.feature_names_in_)
        # Remove superfluous features
        batch = batch.loc[:, [col for col in batch.columns if col not in novar]]
        # Data to DataLoader
        batch = torch.from_numpy(batch.values).float()
        # Ensure the feautures are correct
        if model_weights['fc1.weight'].shape[1] != batch.shape[1]:
            raise ValueError(
                f'Mismatch in the number of features: expected {model_weights["fc1.weight"].shape[1]} while received {batch.shape[1]}')
        # Make predictions
        res.extend(mlp(batch.to(device)).numpy(force=True).tolist())
    # Return predictions
    return pd.DataFrame(res).squeeze().round(3).rename('pChEMBL_value_pred')


def truncnorm(n: int, mean: float, sd: float, lwr: float, upr: float, rounding: int, random_state: int):
    """Return samples drawn from the truncated normal distribution.

    :param n: number of samples to be drawn
    :param mean: mean of distribution
    :param sd: standard deviation of distribition
    :param lwr: minimum value of distribution
    :param upr: maximum value of distribution
    :param rouding: significant digits to keep
    :param random_state: seed of the random sampling
    """
    rng = np.random.default_rng(random_state)
    samp = np.round(rng.normal(mean, sd, n), rounding)
    samp[samp < lwr] = lwr
    samp[samp > upr] = upr
    return samp


def get_synthetic_biological_samples(n, df_description, num_stds: float, min: bool, max: bool, random_state: int):
    """Fit truncated normal distribution on features and generate synthetic samples.

    :param n: number of samples to generate
    :param df_description: DataFrame.describe result of the feature matrix
    :param num_stds: number of strandard deviations around the mean to sample
    :param min: should feature distributions be capped to the minimum of each feature
    :param max: should feature distributions be capped to the maximum of each feature
    :return: a dataframe of n sampled datapoints with feature values rounded to 5 decimals
    :param random_state: seed of the random sampling
    """
    columns = [pd.Series(truncnorm(n,
                                   df_description.loc['mean', :][i],
                                   df_description.loc['std', :][i] * num_stds,
                                   df_description.loc['min', :][i] if min else -np.inf,
                                   df_description.loc['max', :][i] if max else np.inf,
                                   5,
                                   random_state
                                   ), name=df_description.columns[i]
                         ) for i in range(df_description.shape[1])]
    return pd.concat(columns, axis=1)


@dataclass
class SamplingDistribution:
    num_stds: float
    cap_min: bool
    cap_max: bool


def get_bio_neg_random_samples(n: int, df_description: pd.DataFrame, num_stds: float, cap_min: bool, cap_max: bool,
                               label: str, real_bio_samples: pd.DataFrame, random_state: int):
    """Obtain random negative samples with oversampled biological features.

    :param n: Number of samples to be oversampled.
    :param df_description: Summary of feature distributions to sample values from (result of `df.describe()` where `df` contains features to be oversampled).
    :param num_stds: Number of standard deviations around one feature's mean to sample from.
    :param cap_min: Should randomly sampled values lower than this be capped to it.
    :param cap_max: Should randomly sampled values greater than this be capped to it.
    :param label: Name of the column containing the readout label.
    :param real_bio_samples: Set of experimentally tested samples unaltered and concatenated with the new samples.
    :param randaom_state: Seed for random initialization.
    """
    # Obtain random oversampled samples
    new_samples = get_synthetic_biological_samples(n, df_description, num_stds, cap_min, cap_max, random_state)
    # Set their label as negative (0)
    new_samples.insert(0, label, 0)
    # Concatenate
    dataset = pd.concat([real_bio_samples.reset_index(drop=True),
                         new_samples.reset_index(drop=True)],
                        axis=0).reset_index(drop=True)
    return dataset


def get_biochem_neg_random_samples(n: int, df_description: pd.DataFrame, num_stds: float, cap_min: bool, cap_max: bool,
                                   label: str, real_biochem_samples: pd.DataFrame, real_chem_samples: pd.DataFrame,
                                   random_state: int):
    """Obtain random negative samples with oversampled biological  and molecular features.

    :param n: Number of samples to be oversampled.
    :param df_description: Summary of feature distributions to sample values from (result of `df.describe()` where `df` contains features to be oversampled).
    :param num_stds: Number of standard deviations around one feature's mean to sample from.
    :param cap_min: Should randomly sampled values lower than this be capped to it.
    :param cap_max: Should randomly sampled values greater than this be capped to it.
    :param label: Name of the column containing the readout label.
    :param real_biochem_samples: Set of experimentally tested samples unaltered and concatenated with the new samples.
    :param real_chem_samples: Set of physicochemical features to be unaltered and assigned to the new samples.
    :param randaom_state: Seed for random initialization.
    """
    if n > 0:
        # Obtain random oversampled samples
        new_samples = get_synthetic_biological_samples(n, df_description, num_stds, cap_min, cap_max, random_state)
        new_samples = new_samples[[col for col in new_samples.columns if col in real_biochem_samples.columns.tolist()]]
        # Set their label as negative (0)
        new_samples.insert(0, label, 0)
        # Concatenate physicochemical descriptors of oversampled samples
        new_samples = pd.concat([new_samples,
                                 real_chem_samples.sample(n=n, replace=False, random_state=random_state).reset_index(
                                     drop=True).drop(columns=label)],
                                axis=1)
        # Concatenate
        dataset = pd.concat([real_biochem_samples.reset_index(drop=True),
                             new_samples.reset_index(drop=True)],
                            axis=0)
    else:
        dataset = real_biochem_samples.reset_index(drop=True)
    return dataset


def evaluate_custom_oversampling(oversampling_times: int, n_from: int, n_to: int, n_step: int,
                                 sampling_distribs: Iterable[SamplingDistribution], df_desc: pd.DataFrame,
                                 default_df: pd.DataFrame, label: str, sampling_times: int, random_state=0,
                                 n_jobs: int = -1, force_console_pbar: bool = False):
    """Evaluate custom oversampling.

    1) Fit truncated normal distribution on features.
    2) Generate synthetic samples.
    3) Fit and evaluate model with CV using oversamples

    :param oversampling_times: number of oversampling rounds for statistical significance
    :param n_from: minimum number of synthetic sample to generate
    :param n_to: maximum number of synthetic sample to generate
    :param n_step: step in synthetic samples to generate
    :param sampling_distribs: SamplingDistribution objects defining the oversampling strategy
    :param df_desc: DataFrame.describe result of the feature matrix
    :param default_df: full original matrix with label and features
    :param label: name of the label column
    :param sampling_times: number of rounds of model fitting for stats significance
    :param random_state: the random seed to use for model building (internally incremented to make models different)
    :param n_jobs: number of parallel jobs to use fo model training
    :param force_console_pbar: if True and used in a notebook, uses the console progress bar.
    """
    results = []
    # Setup progress bars
    distrib_pbar = (std_tqdm if force_console_pbar else tqdm)(total=len(sampling_distribs), leave=False,
                                                              desc='#distributions',
                                                              ncols=(None if NOTEBOOK_CONTEXT else 90))
    nround_pbar = (std_tqdm if force_console_pbar else tqdm)(total=oversampling_times, leave=False,
                                                             desc='#oversampling',
                                                             ncols=(None if NOTEBOOK_CONTEXT else 90))
    step_pbar = (std_tqdm if force_console_pbar else tqdm)(total=len(range(n_from, n_to, n_step)), leave=False,
                                                           desc='points', ncols=(None if NOTEBOOK_CONTEXT else 90))
    repeat_pbar = (std_tqdm if force_console_pbar else tqdm)(total=sampling_times, leave=False, desc='#model fittings',
                                                             ncols=(None if NOTEBOOK_CONTEXT else 90))
    for distrib in sampling_distribs:
        # Reset previous progress bar (useful when used in a notebook)
        if nround_pbar.n == nround_pbar.total:
            nround_pbar.n, nround_pbar.last_print_n = 0, 0
            nround_pbar.refresh()
        for nround in range(oversampling_times):
            # Reset previous progress bar (useful when used in a notebook)
            if step_pbar.n == step_pbar.total:
                step_pbar.n, step_pbar.last_print_n = 0, 0
                step_pbar.refresh()
            for i in range(n_from, n_to, n_step):
                # Reset previous progress bar (useful when used in a notebook)
                if repeat_pbar.n == repeat_pbar.total:
                    repeat_pbar.n, repeat_pbar.last_print_n = 0, 0
                    repeat_pbar.refresh()
                # Obtain oversampled dataset
                dataset = get_bio_neg_random_samples(i, df_desc, distrib.num_stds, distrib.cap_min, distrib.cap_max,
                                                     label, default_df, random_state)
                # Extract label and features
                dataset_Y = dataset[[label]].values.ravel()
                dataset_X = dataset.drop([label], axis=1)
                # Train models
                for repeat in range(sampling_times):
                    model = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3,
                                          colsample_bytree=0.7, colsample_bylevel=0.7, colsample_bynode=0.7,
                                          random_state=random_state, n_jobs=n_jobs)
                    # Obtain metrics
                    result = pd.DataFrame.from_dict(
                        cv_estimate(model, False, dataset_X, dataset_Y, 5, True, True, random_state))
                    result.insert(0, 'samples added', i)
                    result.insert(1, 'folds', list(range(1, 6)))
                    result['oversampling_repeat'] = nround
                    result['model_fitting'] = repeat
                    result['num_stds'] = distrib.num_stds
                    result['cap_to_min'] = distrib.cap_min
                    result['cap_to_max'] = distrib.cap_max
                    result['random_state'] = random_state
                    result['model'] = model
                    results.append(result)
                    random_state += 1
                    repeat_pbar.update()
                step_pbar.update()
            nround_pbar.update()
        distrib_pbar.update()
    # Close progress bars
    distrib_pbar.close()
    nround_pbar.close()
    step_pbar.close()
    repeat_pbar.close()
    return pd.concat(results)


def evaluate_custom_oversampling_biochem(oversampling_times: int, n_from: int, n_to: int, n_step: int,
                                         sampling_distribs: Iterable[SamplingDistribution], df_desc: pd.DataFrame,
                                         default_df: pd.DataFrame, chem_df: pd.DataFrame, chem_df2: pd.DataFrame,
                                         label: str, sampling_times: int, random_state: int = 0, n_jobs: int = -1,
                                         force_console_pbar: bool = False):
    """Evaluate custom oversampling.

    1) Fit truncated normal distribution on features.
    2) Generate synthetic samples.
    3) Fit and evaluate model with CV using oversamples

    :param oversampling_times: number of oversampling rounds for statistical significance
    :param n_from: minimum number of synthetic sample to generate
    :param n_to: maximum number of synthetic sample to generate
    :param n_step: step in synthetic samples to generate
    :param sampling_distribs: SamplingDistribution objects defining the oversampling strategy
    :param df_desc: DataFrame.describe result of the feature matrix
    :param default_df: full original matrix with label and biological features
    :param chem_df: dataset of physicochemical features corresponding to molecules of the `default_df`
    :param chem_df2: dataset of physicochemical features corresponding to molecules to be oversampled
    :param label: name of the label column
    :param sampling_times: number of rounds of model fitting for stats significance
    :param random_state: the random seed to use for model building (internally incremented to make models different)
    :param n_jobs: number of parallel jobs to use fo model training
    :param force_console_pbar: if True and used in a notebook, uses the console progress bar.
    """
    results = []
    # Concatenate physicochemical descriptors of original molecules
    # (as is an invariant)
    physchem_orig = pd.concat([default_df, chem_df],
                              axis=1)
    # Setup progress bars
    distrib_pbar = (std_tqdm if force_console_pbar else tqdm)(total=len(sampling_distribs), leave=False,
                                                              desc='#distributions',
                                                              ncols=(None if NOTEBOOK_CONTEXT else 90))
    nround_pbar = (std_tqdm if force_console_pbar else tqdm)(total=oversampling_times, leave=False,
                                                             desc='#oversampling',
                                                             ncols=(None if NOTEBOOK_CONTEXT else 90))
    step_pbar = (std_tqdm if force_console_pbar else tqdm)(total=len(range(n_from, n_to, n_step)), leave=False,
                                                           desc='points', ncols=(None if NOTEBOOK_CONTEXT else 90))
    repeat_pbar = (std_tqdm if force_console_pbar else tqdm)(total=sampling_times, leave=False, desc='#model fittings',
                                                             ncols=(None if NOTEBOOK_CONTEXT else 90))
    for distrib in sampling_distribs:
        # Reset previous progress bar (useful when used in a notebook)
        if nround_pbar.n == nround_pbar.total:
            nround_pbar.n, nround_pbar.last_print_n = 0, 0
            nround_pbar.refresh()
        for nround in range(oversampling_times):
            # Reset previous progress bar (useful when used in a notebook)
            if step_pbar.n == step_pbar.total:
                step_pbar.n, step_pbar.last_print_n = 0, 0
                step_pbar.refresh()
            for i in range(n_from, n_to, n_step):
                # Reset previous progress bar (useful when used in a notebook)
                if repeat_pbar.n == repeat_pbar.total:
                    repeat_pbar.n, repeat_pbar.last_print_n = 0, 0
                    repeat_pbar.refresh()
                # Obtain oversampled dataset
                dataset = get_biochem_neg_random_samples(i, df_desc, distrib.num_stds, distrib.cap_min, distrib.cap_max,
                                                         label, physchem_orig, chem_df2, random_state)
                # Extract label and features
                dataset_Y = dataset[[label]].values.ravel()
                dataset_X = dataset.drop([label], axis=1)
                # Train models
                for repeat in range(sampling_times):
                    model = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3,
                                          colsample_bytree=0.7, colsample_bylevel=0.7, colsample_bynode=0.7,
                                          random_state=random_state, n_jobs=n_jobs)
                    # Obtain metrics
                    result = pd.DataFrame.from_dict(
                        cv_estimate(model, False, dataset_X, dataset_Y, 5, True, True, random_state))
                    result.insert(0, 'samples added', i)
                    result.insert(1, 'folds', list(range(1, 6)))
                    result['oversampling_repeat'] = nround
                    result['model_fitting'] = repeat
                    result['num_stds'] = distrib.num_stds
                    result['cap_to_min'] = distrib.cap_min
                    result['cap_to_max'] = distrib.cap_max
                    result['random_state'] = random_state
                    result['model'] = model
                    results.append(result)
                    random_state += 1
                    repeat_pbar.update()
                step_pbar.update()
            nround_pbar.update()
        distrib_pbar.update()
    # Close progress bars
    distrib_pbar.close()
    nround_pbar.close()
    step_pbar.close()
    repeat_pbar.close()
    return pd.concat(results)


def serialize_xgboost_classifier(model):
    """Serialize a XGBoost classifier to LZMA-compressed bytes."""
    serialized_model = {
        'meta': 'xgboost-classifier',
        'params': model.get_params(),
        'n_classes_': model.n_classes_,
        'classes_': model.classes_.tolist(),
        'xgboost_version': xgboost.__version__,
    }

    filename = f'{str(uuid.uuid4())}.json'
    model.save_model(filename)
    with open(filename, 'r') as fh:
        serialized_model['advanced-params'] = fh.read()
    os.remove(filename)

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return lzma.compress(str(serialized_model).encode(), preset=9 | lzma.PRESET_EXTREME)


def deserialize_xgboost_classifier(compressed_model: bytes):
    """Deserialize LZMA-compressed bytes to a XGBoost classifier."""
    # Decompress with LZMA
    decompressed_model = lzma.decompress(compressed_model)
    # Replace Numpy NaNs if any
    decompressed_model = (decompressed_model.decode()
                          .replace('np.NaN', 'nan')
                          .replace('np.nan', 'nan')
                          .replace('nan', 'np.nan'))
    # Parse as a dictionary
    model_dict = eval(decompressed_model)
    # Ensure this is a classifier
    assert model_dict.get('meta') == 'xgboost-classifier', 'This is not a LZMA-compressed serilized XGBoost classifier'
    # Instantiate the model
    model = XGBClassifier(**model_dict['params'])
    # Load parameters
    filename = f'{str(uuid.uuid4())}.json'
    with open(filename, 'w') as fh:
        fh.write(model_dict['advanced-params'])
    model.load_model(filename)
    os.remove(filename)
    # Set optional parameters
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'n_classes_' in model_dict.keys():
        model.n_classes_ = model_dict['n_classes_']
    if 'classes_' in model_dict.keys():
        model.classes_ = np.array(model_dict['classes_'])
    return model


def get_xgboost_calssifier_importance_gain(model):
    """Obtain the feature importance gain of a XGBoostClassifier."""
    return model.get_booster().get_score(importance_type='gain')


def deserialize_xgboost_classifier_and_get_importance_gain(compressed_model: bytes):
    """Obtain the feature importance gain of a LZMA-compressed serialized XGBoostClassifier."""
    return get_xgboost_calssifier_importance_gain(deserialize_xgboost_classifier(compressed_model))


# Patterns required to determine scaffolds and CSKs
PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
REPL = Chem.MolFromSmarts("[*]")


def generate_scaffold(smiles: str, real_bm: bool = True, use_csk: bool = False, use_bajorath: bool = False) -> str:
    """Obtain the molecular scaffold of then given molecule.

    Adapted from @dehaenw's solution: https://github.com/rdkit/rdkit/discussions/6844

    :param smiles: SMILES of the molecule
    :param real_bm: Add electron placeholders per removed exocyclic bond, as in Bemis-Murcko's seminal paper,
    isntead of RDKit's behaviour that retains the first atom of the exo-bonded substituent.
    :param use_csk: Obtain cyclic skeletton, also known as frameworks.
    :param use_bajorath: Do not retain either exo-bonded substituents nor electron placeholders.
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)  # Important for canonicalization of CSK!
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if use_bajorath:
        scaff = AllChem.DeleteSubstructs(scaff, PATT)
    if real_bm:
        scaff = AllChem.ReplaceSubstructs(scaff, PATT, REPL, replaceAll=True)[0]
    if use_csk:
        scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
        if real_bm:
            scaff = MurckoScaffold.GetScaffoldForMol(scaff)
    return Chem.MolToSmiles(scaff)


class ScaffoldSplitter:
    """Class for doing data splits by chemical scaffold.

    Refer to Deepchem for the implementation, https://git.io/fXzF4
    """

    def __init__(self):
        """
        :param include_chirality: should the scaffolds include chirality
        """
        self.get_scaffold = partial(generate_scaffold)

    def _split(self, dataset: pd.DataFrame,
               smiles_list: list[str] = None,
               scaffold_list: list[str] = None,
               frac_train: float = 0.8,
               frac_valid: float = 0.1,
               frac_test: float = 0.1,
               random_state: int = 0,
               n_jobs: int = 1):
        """Split a dataset into training, validation a test sets based on Bemis-Murcko scaffolds.

        :param dataset: The dataset to be split.
        :param smiles_list: The list of SMILES to obtain scaffolds from. Ignored if `scaffold_list` is not None.
        :param scaffold_list: The list of (repeated) scaffolds corresponding to the `dataset`'s samples,
        to be assigned to the training, validation a test sets.
        :param frac_train: Fraction of scaffolds to be assigned to the training set.
        :param frac_valid: Fraction of scaffolds to be assigned to the validation set.
        :param frac_test: Fraction of scaffolds to be assigned to the test set.
        :param random_state: Seed for random initialization.
        :param n_jobs: Number of concurrent processes to parallelize the scaffold generation over.
        :return: The indices of molecules/samples that belong to the training, validation a test sets, repsectively.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a pandas dataframe")
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        if smiles_list is None and scaffold_list is None:
            raise ValueError('Either SMILES or murcko scaffold need to be supplied.')
        if smiles_list is not None and len(dataset) != len(smiles_list):
            raise ValueError(
                f"The lengths of dataset({len(dataset)}) and smiles_list ({len(smiles_list)}) are different")
        if scaffold_list is not None and len(dataset) != len(scaffold_list):
            raise ValueError(
                f"The lengths of dataset({len(dataset)}) and scaffold_list ({len(scaffold_list)}) are different")
        rng = np.random.RandomState(random_state)
        # Obtain scaffolds
        if scaffold_list is not None:  # precalculated
            scaffolds = (pd.concat((pd.Series(range(len(scaffold_list)), name='index'),
                                    pd.Series(scaffold_list).rename('scaffold')),
                                   axis=1)
                         .groupby('scaffold', sort=False)
                         ['index']
                         .apply(list)
                         .to_dict()
                         )
        else:  # Need to calculate
            smiles_list = list(smiles_list)
            if n_jobs == 1:
                scaffolds = defaultdict(list)
                for ind, smiles in enumerate(
                        tqdm(smiles_list, desc='Obtain scaffolds', ncols=(None if NOTEBOOK_CONTEXT else 90))):
                    scaffold = self.get_scaffold(smiles)
                    scaffolds[scaffold].append(ind)
            else:  # parallelize
                pandarallel.initialize(nb_workers=n_jobs, progress_bar=True)
                scaffolds = (pd.concat((pd.Series(range(len(smiles_list)), name='index'),
                                        pd.Series(smiles_list).parallel_apply(generate_scaffold).rename('scaffold')),
                                       axis=1)
                             .groupby('scaffold', sort=False)
                             ['index']
                             .apply(list)
                             .to_dict()
                             )
        # Get mapping from scaffold to molecule index
        scaffold_inds = list(scaffolds.values())
        scaffold_sets = [scaffold_inds[i] for i in rng.permutation(list(range(len(scaffold_inds))))]
        # Maximum number of samples to include in each set
        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))
        # Fill in sets until maximum is reached
        train_index = []
        valid_index = []
        test_index = []
        for scaffold_set in tqdm(scaffold_sets, desc='Subset assignment', ncols=(None if NOTEBOOK_CONTEXT else 90)):
            if len(valid_index) + len(scaffold_set) <= n_total_valid:
                valid_index.extend(scaffold_set)
            elif len(test_index) + len(scaffold_set) <= n_total_test:
                test_index.extend(scaffold_set)
            else:
                train_index.extend(scaffold_set)
        return np.array(train_index), np.array(valid_index), np.array(test_index)

    def train_valid_test_split(self, dataset: pd.DataFrame,
                               smiles_list: list[str] = None,
                               scaffold_list: list[str] = None,
                               frac_train: float = 0.8,
                               frac_valid: float = 0.1,
                               frac_test: float = 0.1,
                               random_state: int = 0,
                               verbose: bool = True,
                               logfile=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, valid and test set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        :param dataset: pandas dataframe
        :param smiles_list: SMILES list corresponding to the dataset
        :param scaffold_list: list of Murcko scaffolds if already precomputed
        :param frac_train: fraction of dataset put into training data
        :param frac_valid: fraction of dataset put into validation data
        :param frac_test: fraction of dataset put into test data
        :param random_state: random seed
        :param verbose: display splitting information
        :param logfile: path to a file in which to log the splitting. Ignored if verbose is None.
        :return: split dataset into training, validation and test
        """
        train_inds, valid_inds, test_inds = self._split(dataset=dataset,
                                                        smiles_list=smiles_list,
                                                        scaffold_list=scaffold_list,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=frac_test,
                                                        random_state=random_state)

        train = dataset.iloc[train_inds]
        valid = dataset.iloc[valid_inds]
        test = dataset.iloc[test_inds]

        if verbose:
            self.log_results(len(dataset), (frac_train, frac_valid, frac_test),
                             len(train), len(valid), len(test), logfile=logfile)

        return train, valid, test

    def train_valid_split(self, dataset: pd.DataFrame,
                          smiles_list: list[str] = None,
                          scaffold_list: list[str] = None,
                          frac_train: float = 0.9,
                          frac_valid: float = 0.1,
                          random_state: int = 0,
                          verbose: bool = True,
                          logfile=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and valid set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        :param dataset: pandas dataframe
        :param smiles_list: SMILES list corresponding to the dataset
        :param scaffold_list: list of Murcko scaffolds if already precomputed
        :param frac_train: fraction of dataset put into training data
        :param frac_valid: fraction of dataset put into validation data
        :param random_state: random seed
        :param verbose: display splitting information
        :param logfile: path to a file in which to log the splitting. Ignored if verbose is None.
        :return: split dataset into training and validation
        """
        train_inds, valid_inds, test_inds = self._split(dataset=dataset,
                                                        smiles_list=smiles_list,
                                                        scaffold_list=scaffold_list,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=0.,
                                                        random_state=random_state)
        assert len(test_inds) == 0

        train = dataset.iloc[train_inds]
        valid = dataset.iloc[valid_inds]

        if verbose:
            self.log_results(len(dataset), (frac_train, frac_valid),
                             len(train), len(valid), logfile=logfile)

        return train, valid

    def log_results(self, total_len: int,
                    ideal_ratios: tuple[float, float] | tuple[float, float, float],
                    len_train_inds: int, len_valid_inds: int, len_test_inds: int = None,
                    logfile=None):
        """Log splitting status in a given path."""
        ideal_train_len = int(round(ideal_ratios[0] * total_len))
        train_error = (len_train_inds - ideal_train_len) / ideal_train_len
        print(f'#rows in taining set: {len_train_inds}, ideally: {ideal_train_len}, error: {train_error:.1%}',
              file=logfile)
        ideal_valid_len = int(round(ideal_ratios[1] * total_len))
        valid_error = (len_valid_inds - ideal_valid_len) / ideal_valid_len
        print(f'#rows in validation set: {len_valid_inds}, ideally: {ideal_valid_len}, error: {valid_error:.1%}',
              file=logfile)
        if len_test_inds is not None:
            ideal_test_len = int(round(ideal_ratios[2] * total_len))
            test_error = (len_test_inds - ideal_test_len) / ideal_test_len
            print(f'#rows in validation set: {len_test_inds}, ideally: {ideal_test_len}, error: {test_error:.5%}',
                  file=logfile)


class FrameworkSplitter(ScaffoldSplitter):
    """Class to split data by chemical cyclic skeletton."""

    def __init__(self):
        super().__init__()
        self.get_scaffold = partial(generate_scaffold, use_csk=True)


class ConnectedComponentsSplitter(ScaffoldSplitter):
    """Class to split data by connected components."""

    def _split(self, components: pd.DataFrame,
               frac_train: float = 0.8,
               frac_valid: float = 0.1,
               frac_test: float = 0.1,
               tolerance: float = 0.01,
               random_state: int = 0):
        if not isinstance(components, pd.DataFrame) or 'group' not in components.columns.values:
            raise ValueError(
                "components must be a pandas dataframe resulting from a call to `ConnectedComponentsSplitter.get_connected_components`.")
        # Determine connected components
        component_sizes = components.group.value_counts().to_dict()
        # If the largest group is more than frac_train
        if component_sizes[0] >= frac_train * sum(component_sizes.values()):
            # Then only balance the validation and test set
            testval_assign = self.two_groups_split(df=components.query('group != 0'),
                                                   group_column='group',
                                                   target_ratio=(frac_valid / (frac_valid + frac_test)),
                                                   tolerance=tolerance,
                                                   random_state=random_state)
            return ([0], *testval_assign)
        else:
            # Revert to balancing the 3 groups
            return self.three_groups_split(df=components,
                                           group_column='group',
                                           set1_frac=frac_train,
                                           set2_frac=frac_valid,
                                           set3_frac=frac_test,
                                           random_state=random_state,
                                           max_iterations=int(1e5))

    def train_valid_test_split(self, dataset: pd.DataFrame,
                               col1: str, col2: str,
                               frac_train: float = 0.8,
                               frac_valid: float = 0.1,
                               frac_test: float = 0.1,
                               random_state: int = 0,
                               verbose: bool = True,
                               logfile=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, valid and test set.

        :param dataset: Pandas dataframe where each row represents an edge of the bipartite graph.
        :param col1: column containing first nodes of the edges
        :param col2: column containing second nodes of the edges
        :param frac_train: fraction of dataset put into training data
        :param frac_valid: fraction of dataset put into validation data
        :param frac_test: fraction of dataset put into test data
        :param random_state: random seed
        :param verbose: display splitting information
        :param logfile: path to a file in which to log the splitting. Ignored if verbose is None.
        :return: split dataset into training, validation and test
        """
        components = self.get_connected_components(df=dataset, col1=col1, col2=col2, verbose=verbose)
        train_inds, valid_inds, test_inds = self._split(components=components,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=frac_test,
                                                        random_state=random_state)
        train = components.query('group.isin(@train_inds)')
        valid = components.query('group.isin(@valid_inds)')
        test = components.query('group.isin(@test_inds)')
        if verbose:
            self.log_results(len(dataset), (frac_train, frac_valid, frac_test),
                             len(train), len(valid), len(test), logfile=logfile)
        return train, valid, test

    def train_valid_split(self, dataset: pd.DataFrame,
                          col1: str, col2: str,
                          frac_train: float = 0.9,
                          frac_valid: float = 0.1,
                          random_state: int = 0,
                          verbose: bool = True,
                          logfile=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and valid set.

        :param dataset: Pandas dataframe where each row represents an edge of the bipartite graph.
        :param col1: column containing first nodes of the edges
        :param col2: column containing second nodes of the edges
        :param frac_train: fraction of dataset put into training data
        :param frac_valid: fraction of dataset put into validation data
        :param random_state: random seed
        :param verbose: display splitting information
        :param logfile: path to a file in which to log the splitting. Ignored if verbose is None.
        :return: split dataset into training and validation
        """
        components = self.get_connected_components(df=dataset, col1=col1, col2=col2, verbose=verbose)
        train_inds, valid_inds, test_inds = self._split(components=components,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=0.,
                                                        random_state=random_state)
        assert len(test_inds) == 0

        train = components.query('group.isin(@train_ids)')
        valid = components.query('group.isin(@valid_inds)')

        if verbose:
            self.log_results(len(dataset), (frac_train, frac_valid),
                             len(train), len(valid), logfile=logfile)

        return train, valid

    @staticmethod
    def get_connected_components(df: pd.DataFrame, col1: str, col2: str, verbose: bool = True) -> pd.DataFrame:
        """Determine connected components in a bipartite graph.
        Useful to derive a data split to train models and test them on cold molecules AND proteins at the same time.

        :param df: Pandas dataframe where each row represents an edge of the bipartite graph.
        :param col1: column containing first nodes of the edges
        :param col2: column containing second nodes of the edges
        :param verbose: should a progress bar be shown.
        """
        df = df.copy()
        G = nx.Graph()
        # Add edges to the graph
        for idx, row in tqdm(df.iterrows(), total=len(df), ncols=(None if NOTEBOOK_CONTEXT else 90),
                             desc='Creating a bipartite graph', disable=(not verbose)):
            G.add_edge(row[col1], row[col2])
        # Find connected components
        groups = list(nx.connected_components(G))
        # Assign group identifiers
        group_mapping = {node: i for i, group in enumerate(groups) for node in group}
        df['group'] = df.apply(lambda row: group_mapping[row[col1]], axis=1)
        return df

    @staticmethod
    def two_groups_split(df: pd.DataFrame, group_column: str, target_ratio: float = 0.5, tolerance: float = 0.02,
                         random_state: int = 0):
        """Split a dataframe into two groups given their relative ratio.

        :param df: Pandas dataframe to split.
        :param group_column: Name of the column containing group assignments.
        :param target_ratio: The final ratio between the two output groups
        :param tolerance: Tolerance around the target ratio to accept a solution
        :param random_state: Seed for randomness to ensure reproducibility or obtain different assignments
        :return:
        """
        df = df.copy()
        np.random.seed(random_state)  # Set random seed for reproducibility
        # Step 1: Compute group sizes
        group_sizes = df[group_column].value_counts().reset_index()
        group_sizes.columns = ['group', 'size']
        total_size = group_sizes['size'].sum()
        target_size = int(total_size * target_ratio)
        # Step 2: Shuffle for randomness
        group_sizes = group_sizes.sample(frac=1, random_state=random_state).reset_index(drop=True)
        # Step 3: Greedy assignment ensuring the closest balance
        set1 = []
        sum1 = 0
        for _, row in group_sizes.iterrows():
            group, size = row['group'], row['size']
            # Assign group to A if it keeps balance close to target
            if sum1 + size <= target_size + (tolerance * target_ratio * total_size):
                set1.append(group)
                sum1 += size
        # Step 4: Get second set
        set2 = [row['group'] for _, row in group_sizes.iterrows() if row['group'] not in set1]
        return set1, set2

    @staticmethod
    def three_groups_split(df: pd.DataFrame, group_column: str, set1_frac: float, set2_frac: float, set3_frac: float,
                           random_state: int = 0, max_iterations: int = 10000):
        """
        Assign groups to 3 clusters aiming to meet target ratios as close as possible.
        Groups are not split. A simulated annealing approach is used to optimize
        the assignment with respect to the target sizes.

        :param df: Pandas dataframe to split.
        :param group_column: Name of the column containing group assignments.
        :param set1_frac: Target fraction for the first set of groups.
        :param set2_frac: Target fraction for the second set of groups.
        :param set3_frac: Target fraction for the third set of groups.
        :param random_state: Seed for reproducibility.
        :param max_iterations: Maximum number of iterations for the simulated annealing.
        :return: Tuple of three lists, each containing the group names assigned to that set.
        """
        np.random.seed(random_state)  # Set random seed for reproducibility
        # Step 1: Compute group sizes
        groups = df[group_column].value_counts().reset_index().to_dict()
        total_size = sum(groups.values())
        targets = [total_size * frac for frac in (set1_frac, set2_frac, set3_frac)]

        # List of groups
        group_keys = list(groups.keys())

        # Greedy initialization: assign each group (largest first) to the cluster with the smallest ratio
        # of current sum to target.
        assignment = {}
        cluster_sums = [0, 0, 0]
        sorted_groups = sorted(group_keys, key=lambda g: groups[g], reverse=True)
        for g in sorted_groups:
            ratios = [cluster_sums[i] / targets[i] for i in range(3)]
            best_cluster = np.argmin(ratios)
            assignment[g] = best_cluster
            cluster_sums[best_cluster] += groups[g]

        def compute_error(assignment_dict):
            sums = [0, 0, 0]
            for g, cl in assignment_dict.items():
                sums[cl] += groups[g]
            # Error is total absolute deviation from targets
            return sum(abs(s - t) for s, t in zip(sums, targets))

        current_error = compute_error(assignment)
        best_assignment = assignment.copy()
        best_error = current_error

        # Simulated annealing parameters
        T = 1.0  # initial temperature
        T_min = 1e-4  # minimum temperature
        alpha = 0.999  # cooling rate

        for _ in range(max_iterations):
            # Randomly pick a group and propose moving it to a different cluster.
            g = np.random.choice(group_keys)
            current_cluster = assignment[g]
            new_cluster = np.random.choice([i for i in range(3) if i != current_cluster])

            new_assignment = assignment.copy()
            new_assignment[g] = new_cluster
            new_error = compute_error(new_assignment)

            delta = new_error - current_error
            # Accept move if improvement, or with a probability exp(-delta/T) if worse.
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                assignment = new_assignment
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_assignment = assignment.copy()

            T *= alpha
            if T < T_min:
                break

        # Build clusters from best_assignment
        clusters = [[], [], []]
        for g, cl in best_assignment.items():
            clusters[cl].append(g)

        return tuple(clusters)


def train_bioactivity_spectra_model(data: pd.DataFrame,
                                    split_type: str,
                                    activity_type: str,
                                    model_type: str,
                                    out_folder: str = None) -> None:
    """Train a bioactivity spectrum model based on the supplied parameters.

    :param data: the prepared data to be used to train the bioactivity spectra model.
    :param split_type: Type of training, validation, test split to use.
    One of {'murcko_scaffold_split', 'murcko_cyclic_skeleton_split', 'random_split', 'non_overlappling_split'}.
    :param activity_type: Type of activity to be modelled (raw inhibition values or deviation from the average value).
    One of {'pchembl_value_Mean', 'pchembl_value_dev'}.
    :param model_type: Type of model to train. One of {'QSAR', 'QSAM', 'PCM'}.
    :param out_folder: Path to the folder to store the results in. If None, determines a name based on
    """
    # Handle parameters
    if split_type not in ['murcko_scaffold_split', 'murcko_cyclic_skeleton_split',
                          'random_split', 'non_overlappling_split']:
        raise ValueError('Type of split is not supported')
    if activity_type not in ['pchembl_value_Mean', 'pchembl_value_dev']:
        raise ValueError('Type of activity is not supported')
    if model_type not in ['QSAR', 'QSAM', 'PCM']:
        raise ValueError('Type of model is not supported')
    if not os.path.exists(out_folder):
        raise RuntimeError('The output folder does not exist.')
    # Handle splitting of the data
    if split_type == 'non_overlappling_split':
        # In the original connected groups are different from the ones currently calculated
        if 'connected_component_split' in data.columns: # recalculated groups
            train = data[data.connected_component_split == 'train'].drop(
                columns=['connectivity', 'target_id', 'SMILES', 'accession',
                         'murcko_scaffold', 'murcko_cyclic_skeleton',
                         'murcko_scaffold_split', 'murcko_cyclic_skeleton_split',
                         'random_split', 'group', 'connected_component_split',
                         'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
            val = data[data.connected_component_split == 'val'].drop(
                columns=['connectivity', 'target_id', 'SMILES', 'accession',
                         'murcko_scaffold', 'murcko_cyclic_skeleton', 'murcko_scaffold_split',
                         'murcko_cyclic_skeleton_split', 'random_split', 'group', 'connected_component_split',
                         'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
            test = data[data.connected_component_split == 'test'].drop(
                columns=['connectivity', 'target_id', 'SMILES', 'accession',
                         'murcko_scaffold', 'murcko_cyclic_skeleton', 'murcko_scaffold_split',
                         'murcko_cyclic_skeleton_split', 'random_split', 'group', 'connected_component_split',
                         'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
        else:  # original groups as deposited onZenodo
            train = data[data.group == 0].drop(columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                                                        'murcko_scaffold', 'murcko_cyclic_skeleton',
                                                        'murcko_scaffold_split',
                                                        'murcko_cyclic_skeleton_split', 'random_split',
                                                        'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
            val = data[(data.group != 0) & (data.group <= 250)].drop(
                columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                         'murcko_scaffold', 'murcko_cyclic_skeleton', 'murcko_scaffold_split',
                         'murcko_cyclic_skeleton_split', 'random_split',
                         'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
            test = data[(data.group != 0) & (data.group > 250)].drop(
                columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                         'murcko_scaffold', 'murcko_cyclic_skeleton', 'murcko_scaffold_split',
                         'murcko_cyclic_skeleton_split', 'random_split',
                         'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
    else:
        train = data[data[split_type] == 'train'].drop(
            columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                     'murcko_scaffold', 'murcko_cyclic_skeleton',
                     'murcko_scaffold_split', 'connected_component_split',
                     'murcko_cyclic_skeleton_split', 'random_split',
                     'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
        val = data[data[split_type] == 'val'].drop(columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                                                            'murcko_scaffold', 'murcko_cyclic_skeleton',
                                                            'murcko_scaffold_split', 'connected_component_split',
                                                            'murcko_cyclic_skeleton_split', 'random_split',
                                                            'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
        test = data[data[split_type] == 'test'].drop(
            columns=['connectivity', 'target_id', 'SMILES', 'accession', 'group',
                     'murcko_scaffold', 'murcko_cyclic_skeleton',
                     'murcko_scaffold_split', 'connected_component_split',
                     'murcko_cyclic_skeleton_split', 'random_split',
                     'pchembl_value_Mean' if activity_type != 'pchembl_value_Mean' else 'pchembl_value_dev'])
    # Drop features depending on model type
    if model_type == 'PCM':
        pass
    elif model_type == 'QSAR':
        train = train.drop(columns=train.columns[train.columns.str.startswith('AVG_domains50_Zscale-van-Westen')])
        val = val.drop(columns=val.columns[val.columns.str.startswith('AVG_domains50_Zscale-van-Westen')])
        test = test.drop(columns=test.columns[test.columns.str.startswith('AVG_domains50_Zscale-van-Westen')])
    elif model_type == 'QSAM':
        train = train.drop(columns=train.columns[
            ~train.columns.str.startswith('AVG_domains50_Zscale-van-Westen') & (train.columns != activity_type)])
        val = val.drop(columns=val.columns[
            ~val.columns.str.startswith('AVG_domains50_Zscale-van-Westen') & (val.columns != activity_type)])
        test = test.drop(columns=test.columns[
            ~test.columns.str.startswith('AVG_domains50_Zscale-van-Westen') & (test.columns != activity_type)])
    else:
        raise ValueError(f'model type not supported: {model_type}')
    # Output log file
    outfile_prefix = f'{out_folder}/BioSpectra-Reg-{model_type}_Mordred-ZscalesVanWesten-FFNN_model_{split_type}_{"mean_pChEMBL" if activity_type == "pchembl_value_Mean" else "dev_pChEMBL"}'
    logfile = open(f'{outfile_prefix}.log', 'w')

    start_time = time.monotonic()
    X_train, y_train = train.drop(columns=activity_type), train[activity_type]
    X_val, y_val = val.drop(columns=activity_type), val[activity_type]
    X_test, y_test = test.drop(columns=activity_type), test[activity_type]
    scaler = StandardScaler()
    X_train.loc[:, :] = scaler.fit_transform(X_train)
    X_val.loc[:, :] = scaler.transform(X_val)
    X_test.loc[:, :] = scaler.transform(X_test)
    standard_scaler_to_json(scaler, f'{outfile_prefix}_scaler.json')
    with open(f'{outfile_prefix}_novar.json', 'w') as oh:
        json.dump(X_train.loc[:, X_train.std() <= .1].columns.tolist(), oh)

    X_train = X_train.loc[:, X_train.std() > .1].astype('float32')
    X_val = X_val.loc[:, X_train.columns].astype('float32')
    X_test = X_test.loc[:, X_train.columns].astype('float64')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Set fixed random number seed
    torch.manual_seed(1234)

    # Define patience and minimum number of epochs
    patience = 200
    min_epochs = 500
    last_save = 0  # last epoch at which the model was written to disk

    # Initialize the MLP
    mlp = MLPRegressor(X_train.shape[1], dropout=0.025).to(device)
    print(mlp, file=logfile)
    print(f'Number of trainable parameters: {mlp.num_params:n}', file=logfile)
    # try:
    #     mlp = torch.compile(mlp)
    # except:
    #     pass
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    best_loss = float('-inf')

    # Create dataloaders
    train_inputs, train_targets = torch.from_numpy(X_train.values), torch.from_numpy(y_train.values)
    train_inputs, train_targets = train_inputs.float(), train_targets.float()
    train_targets = train_targets.reshape((train_targets.shape[0], 1))
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)  # 4096

    val_inputs, val_targets = torch.from_numpy(X_val.values), torch.from_numpy(y_val.values)
    val_inputs, val_targets = val_inputs.float(), val_targets.float()
    val_targets = val_targets.reshape((val_targets.shape[0], 1))
    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=True)  # 2048
    pbar1, pbar2 = None, None
    # Run the training loop
    for epoch in trange(0, 100_000, ncols=(None if NOTEBOOK_CONTEXT else 90)):
        if epoch == 0:
            pbar1 = tqdm(total=len(train_loader), leave=True, ncols=(None if NOTEBOOK_CONTEXT else 90))
        else:
            pbar1.n, pbar1.last_print_n, pbar1.pos = 0, 0, 0
            pbar1.start_t = time.time()
            pbar1.last_print_t = pbar1.start_t
            pbar1.refresh()
        all_x, all_y = [], []
        for x, y in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            x = mlp(x.to(device), dropout=True)
            # Compute loss
            loss = loss_function(x, y.to(device))
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # Detach tensors
            all_y.append(y.cpu().detach().numpy().reshape((y.shape[0],)))
            all_x.append(x.cpu().detach().numpy().reshape((x.shape[0],)))
            pbar1.update()
        # Gather all predictions (all_x) and all target labels (all_y)
        all_x, all_y = np.hstack(all_x), np.hstack(all_y)
        # Print training metrics
        print(pd.Series(all_y).isna().any(), pd.Series(all_x).isna().any())
        print(f'Epoch {epoch + 1} (training):\tMSE Loss: {mean_squared_error(all_y, all_x):.3f}\t'
              f'R²: {r2_score(all_y, all_x):.3f}\tPearson\'s r: {pearsonr(all_y, all_x).statistic:.3f}',
              file=logfile)
        print(f'Epoch {epoch + 1} (training):\tMSE Loss: {mean_squared_error(all_y, all_x):.3f}\t'
              f'R²: {r2_score(all_y, all_x):.3f}\tPearson\'s r: {pearsonr(all_y, all_x).statistic:.3f}')
        if epoch == 0:
            pbar2 = tqdm(total=len(val_loader), leave=True, ncols=(None if NOTEBOOK_CONTEXT else 90))
        else:
            pbar2.n, pbar2.last_print_n, pbar2.pos = 0, 0, 0
            pbar2.start_t = time.time()
            pbar2.last_print_t = pbar2.start_t
            pbar2.refresh()
        # Obtain and print validation metrics
        all_x, all_y = [], []
        for x, y in val_loader:
            x = mlp(x.to(device), dropout=False)
            # Detach tensors
            all_y.append(y.cpu().detach().numpy().reshape((y.shape[0],)))
            all_x.append(x.cpu().detach().numpy().reshape((x.shape[0],)))
            pbar2.update()
        # Gather all predictions (all_x) and all target labels (all_y)
        all_x, all_y = np.hstack(all_x), np.hstack(all_y)
        # Determine if converged
        conv_criteria = pearsonr(all_y, all_x).statistic
        # Print metrics
        print(f'\tEpoch {epoch + 1} (validation):\tMSE Loss: {mean_squared_error(all_y, all_x):.3f}\t'
              f'R²: {r2_score(all_y, all_x):.3f}\tPearson\'s r: {pearsonr(all_y, all_x).statistic:.3f}',
              file=logfile)
        print(f'\tEpoch {epoch + 1} (validation):\tMSE Loss: {mean_squared_error(all_y, all_x):.3f}\t'
              f'R²: {r2_score(all_y, all_x):.3f}\tPearson\'s r: {pearsonr(all_y, all_x).statistic:.3f}'
              f'{"    *" if conv_criteria > best_loss else ""}')
        # Save the model if validation loss improved
        if conv_criteria > best_loss:
            torch.save(mlp.state_dict(), f'{outfile_prefix}.pkg')
            best_loss = conv_criteria
            last_save = epoch
        # Early stopping
        if (epoch >= min_epochs) and (epoch - last_save > patience):
            print(
                f'No loss improvement in the last {patience} epochs: best criterion = {best_loss:.6f} from epoch {last_save}',
                file=logfile)
            print(
                f'No loss improvement in the last {patience} epochs: best criterion = {best_loss:.6f} from epoch {last_save}')
            break
    pbar2.close()
    pbar1.close()
    stop_time = time.monotonic()
    # Process is complete.
    print(f'Training process has finished after {timedelta(seconds=stop_time - start_time)}',
          file=logfile)
    print(f'Training process has finished after {timedelta(seconds=stop_time - start_time)}')
    # Run test predictions
    inputs, targets = torch.from_numpy(X_test.values), torch.from_numpy(y_test.values)
    inputs, targets = inputs.float(), targets.float()
    targets = targets.reshape((targets.shape[0], 1))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    pbar3 = tqdm(total=len(loader), leave=True, ncols=(None if NOTEBOOK_CONTEXT else 90))
    all_x, all_y = [], []
    for x, y in loader:
        x = mlp(x.to(device), dropout=False)
        # Detach tensors
        all_y.append(y.cpu().detach().numpy().reshape((y.shape[0],)))
        all_x.append(x.cpu().detach().numpy().reshape((x.shape[0],)))
        pbar3.update()
    pbar3.close()
    all_x, all_y = np.hstack(all_x), np.hstack(all_y)
    print(f'Holdout test:\tMSE Loss: {mean_squared_error(all_y, all_x):.3f}\t'
          f'R²: {r2_score(all_y, all_x):.3f}\tPearson\'s r: {pearsonr(all_y, all_x).statistic:.3f}',
          file=logfile)
    logfile.close()
    return
