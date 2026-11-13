"""
Checks on what a registered method returns.

Methods can come from plugin files written by the users, so the engine does not
trust shapes and types: whenever a pretreatment, a model or a validation scheme
returns something in the wrong format, the user gets a sentence naming the
method and saying what was expected, instead of a numpy error from deep inside
the calculation. The formats are described in plugins/LEIA-ME.md.
"""
import numpy as np


class ContractError(ValueError):
    """A method returned something in the wrong format (a ValueError, so the screens show it as a message)."""


def _name(item):
    return f"“{getattr(item, 'label', '') or getattr(item, 'id', '?')}”"


def check_state(item, state):
    """The state returned by `fit`: a dict whose values are numbers or numeric arrays."""
    if not isinstance(state, dict):
        raise ContractError(f"{_name(item)}: fit deve devolver um dicionário (dict); devolveu {type(state).__name__}.")
    for key, value in state.items():
        if not isinstance(key, str):
            raise ContractError(f"{_name(item)}: as chaves do estado devem ser texto; encontrada {key!r}.")
        try:
            kind = np.asarray(value).dtype.kind
        except Exception:
            kind = "O"
        if kind not in "biuf":
            raise ContractError(
                f"{_name(item)}: o item '{key}' do estado não é um número nem uma matriz de números "
                f"({type(value).__name__}); só números podem ser guardados no arquivo de modelo.")
    return state


def check_pretreatment_output(item, n_rows, result):
    """What `apply` returns: (X, wavelengths) with the same number of rows and matching lengths."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise ContractError(f"{_name(item)}: deve devolver dois itens, (espectros, eixo); devolveu {type(result).__name__}.")
    try:
        X = np.asarray(result[0], dtype=float)
        wavelengths = np.asarray(result[1], dtype=float).ravel()
    except (TypeError, ValueError):
        raise ContractError(f"{_name(item)}: os espectros e o eixo devolvidos devem ser matrizes de números.") from None
    if X.ndim != 2:
        raise ContractError(f"{_name(item)}: os espectros devolvidos devem ter 2 dimensões (amostras x pontos); têm {X.ndim}.")
    if X.shape[0] != n_rows:
        raise ContractError(
            f"{_name(item)}: recebeu {n_rows} espectros e devolveu {X.shape[0]}; um pré-tratamento não pode "
            "criar nem remover amostras.")
    if wavelengths.size != X.shape[1]:
        raise ContractError(
            f"{_name(item)}: os espectros devolvidos têm {X.shape[1]} pontos, mas o eixo devolvido tem {wavelengths.size}.")
    return X, wavelengths


def check_predictions(item, values, n_rows):
    """What a model's `predict` returns: one number per spectrum."""
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ContractError(f"{_name(item)}: predict deve devolver números.") from None
    if values.ndim == 2 and 1 in values.shape:
        values = values.ravel()
    if values.shape != (n_rows,):
        raise ContractError(
            f"{_name(item)}: predict recebeu {n_rows} espectros e devolveu uma matriz de formato {values.shape}; "
            f"esperado um valor por espectro, formato ({n_rows},).")
    return values


def check_cv_path(item, values, max_k, n_rows):
    """What `cv_path` returns: predictions for every complexity, shape (max_k, n_rows)."""
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ContractError(f"{_name(item)}: cv_path deve devolver números.") from None
    if values.shape != (max_k, n_rows):
        raise ContractError(
            f"{_name(item)}: cv_path devolveu uma matriz de formato {values.shape}; esperado ({max_k}, {n_rows}), "
            "uma linha para cada complexidade de 1 até o máximo e uma coluna por espectro.")
    return values


def check_selection(item, value, n_points):
    """What a variable selection returns: a bool per point of the spectrum, or the positions of the chosen points."""
    try:
        value = np.asarray(value)
    except Exception:
        raise ContractError(f"{_name(item)}: select deve devolver um vetor.") from None
    if value.dtype.kind == "b":
        if value.shape != (n_points,):
            raise ContractError(
                f"{_name(item)}: select devolveu um vetor de verdadeiro/falso de formato {value.shape}; "
                f"esperado um item por ponto do espectro, formato ({n_points},).")
        mask = value.copy()
    elif value.dtype.kind in "iu" and value.ndim == 1:
        if value.size and (value.min() < 0 or value.max() >= n_points):
            raise ContractError(f"{_name(item)}: as posições devolvidas por select devem estar entre 0 e {n_points - 1}.")
        mask = np.zeros(n_points, dtype=bool)
        mask[value] = True
    else:
        raise ContractError(
            f"{_name(item)}: select deve devolver um vetor de verdadeiro/falso (um item por ponto do espectro) "
            "ou as posições inteiras dos pontos escolhidos.")
    if not mask.any():
        raise ContractError(f"{_name(item)}: nenhum ponto do espectro foi escolhido.")
    return mask


def check_split(item, train, val, n):
    """One (train, validation) pair of a validation scheme: positions between 0 and n - 1, without overlap."""
    try:
        train = np.asarray(train).astype(int, casting="same_kind").ravel()
        val = np.asarray(val).astype(int, casting="same_kind").ravel()
    except (TypeError, ValueError):
        raise ContractError(f"{_name(item)}: cada divisão deve ser um par (posições de treino, posições de validação).") from None
    if train.size == 0 or val.size == 0:
        raise ContractError(f"{_name(item)}: uma das divisões ficou sem amostras de treino ou sem amostras de validação.")
    if min(train.min(), val.min()) < 0 or max(train.max(), val.max()) >= n:
        raise ContractError(f"{_name(item)}: as posições devem estar entre 0 e {n - 1}.")
    if np.intersect1d(train, val).size:
        raise ContractError(f"{_name(item)}: a mesma amostra aparece no treino e na validação da mesma divisão.")
    return train, val
