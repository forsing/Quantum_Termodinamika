#!/usr/bin/env python3

# Quantum_Termodinamika


"""
QUANTUM TERMODINAMIKA V1 — Qiskit Gibbs amplitudes — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Preserved dimensionless Hamiltonian:
    H = -sum_i h_i n_i - sum_i,d J_i,d n_i n_(i-d)
    n_i = (I-Z_i)/2
    total occupation = 7

The coefficients represent beta*H, with beta fixed to one.

Gibbs distribution:
    P(y) = exp(-E(y)) / Z

Coherent encoding prepared by the circuit:
    |psi> = sum_y sqrt(P(y)) |y>

This pure state is not the mixed thermal density matrix.
Its computational-basis measurements reproduce the Gibbs law.

Thermodynamic U, F and S refer to the diagonal Gibbs state.

The Hamiltonian is diagonal and commuting.
No noncommuting quantum model or quantum advantage is claimed.

Exact transfer recursions provide:
    partition function;
    inclusion and pair marginals;
    quantum conditional rotation angles;
    minimum-energy combination;
    U, F and S.

Qiskit:
    SparsePauliOp Hamiltonian;
    full preparation circuit with occupancy and count qubits;
    small full-state self-checks;
    compact local circuit contractions for the final NEXT amplitude.

For 39 labels and seven selections:
    39 logical occupation qubits;
    3 count qubits;
    no dense 42-qubit statevector allocation.

Local conditional circuits require at most six qubits
for interaction range two.

No random sampling, injected noise, row shuffling, classifier,
rounding, duplicate repair or cosmetic filters.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import SparsePauliOp, Statevector


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)

NUMBER_COUNT = 39
DRAW_SIZE = 7
HISTORY_LAGS = 2

CANDIDATES = tuple(
    (interaction_range, ridge)
    for interaction_range in (1, 2)
    for ridge in (0.03, 0.3, 3.0)
)

OPTIMIZER_STEPS = 500
GRADIENT_LIMIT = 5e-5

CHECKPOINT_VERSION = "QISKIT_DIAGONAL_GIBBS_V1_20261029"

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "Quantum_Termodinamika_v1_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def edge_mask(m, interaction_range):
    return (
        np.arange(m)[:, None]
        > np.arange(interaction_range)[None, :]
    ).astype(float)


def state_bits(interaction_range):
    return np.array(
        [
            [
                (state >> distance) & 1
                for distance in range(interaction_range)
            ]
            for state in range(1 << interaction_range)
        ],
        dtype=float,
    )


def pair_features(occupancy, interaction_range):
    n, m = occupancy.shape
    result = np.zeros((n, m, interaction_range))

    for index in range(interaction_range):
        distance = index + 1

        result[:, distance:, index] = (
            occupancy[:, distance:]
            * occupancy[:, :-distance]
        )

    return result


def history_context(occupancy, targets, k):
    targets = np.asarray(targets, dtype=np.int64)

    require(
        targets.ndim == 1
        and np.all(targets >= 1)
        and np.all(targets <= len(occupancy)),
        "Neispravni indeksi istorijskih konteksta.",
    )

    m = occupancy.shape[1]
    result = np.zeros((len(targets), HISTORY_LAGS, m))

    for lag in range(1, HISTORY_LAGS + 1):
        previous = targets - lag
        valid = previous >= 0

        result[valid, lag - 1, :] = (
            occupancy[previous[valid]] - k / m
        )

    return result


def make_data(occupancy, k, interaction_range):
    return {
        "x": history_context(
            occupancy,
            np.arange(1, len(occupancy)),
            k,
        ),
        "hp": pair_features(occupancy[:-1], interaction_range),
        "y": occupancy[1:].astype(float),
        "yp": pair_features(occupancy[1:], interaction_range),
    }


def subset(data, start, stop):
    return {
        key: value[start:stop]
        for key, value in data.items()
    }


def parameter_count(m, interaction_range):
    return (HISTORY_LAGS + 1) * m + 2 * m * interaction_range


def unpack(parameters, m, interaction_range):
    first = (HISTORY_LAGS + 1) * m
    pair_size = m * interaction_range

    field_parameters = parameters[:first].reshape(
        HISTORY_LAGS + 1, m
    )

    static_pairs = parameters[
        first:first + pair_size
    ].reshape(m, interaction_range)

    history_pairs = parameters[
        first + pair_size:
    ].reshape(m, interaction_range)

    return field_parameters, static_pairs, history_pairs


def potentials(parameters, context, historical_pairs, interaction_range):
    m = context.shape[2]

    field_parameters, static_pairs, history_pairs = unpack(
        parameters, m, interaction_range
    )

    fields = (
        field_parameters[0][None, :]
        + np.einsum(
            "nlm,lm->nm",
            context,
            field_parameters[1:],
            optimize=True,
        )
    )

    couplings = (
        static_pairs[None, :, :]
        + historical_pairs * history_pairs[None, :, :]
    )

    couplings *= edge_mask(m, interaction_range)[None, :, :]

    require(
        np.all(np.isfinite(fields))
        and np.all(np.isfinite(couplings)),
        "Hamiltonijan sadrži nevažeće parametre.",
    )

    return fields, couplings


def infer(fields, couplings, k, marginals=False):
    fields = np.asarray(fields, dtype=float)
    couplings = np.asarray(couplings, dtype=float)

    require(
        fields.ndim == 2
        and couplings.ndim == 3
        and fields.shape == couplings.shape[:2]
        and np.all(np.isfinite(fields))
        and np.all(np.isfinite(couplings)),
        "Neispravan Hamiltonijan.",
    )

    n, m = fields.shape
    interaction_range = couplings.shape[2]

    require(
        n > 0 and 1 <= k <= m and interaction_range in (1, 2),
        "Neispravne dimenzije kanonskog modela.",
    )

    states = 1 << interaction_range
    mask = states - 1
    bits = state_bits(interaction_range)
    valid = edge_mask(m, interaction_range)

    forward = np.full((n, k + 1, states), -np.inf)
    forward[:, 0, 0] = 0.0

    cache = [forward] if marginals else None

    for number in range(m):
        local = (
            fields[:, number, None]
            + (couplings[:, number, :] * valid[number]) @ bits.T
        )

        following = np.full_like(forward, -np.inf)

        for state in range(states):
            zero = (state << 1) & mask
            one = zero | 1

            following[:, :, zero] = np.logaddexp(
                following[:, :, zero],
                forward[:, :, state],
            )

            following[:, 1:, one] = np.logaddexp(
                following[:, 1:, one],
                forward[:, :-1, state]
                + local[:, state, None],
            )

        forward = following

        if marginals:
            cache.append(forward)

    log_z = logsumexp(forward[:, k, :], axis=1)

    require(
        np.all(np.isfinite(log_z)),
        "Particiona funkcija nije konačna.",
    )

    if not marginals:
        return log_z, None, None

    backward = np.full_like(forward, -np.inf)
    backward[:, k, :] = 0.0

    inclusion = np.zeros_like(fields)
    pair_marginal = np.zeros_like(couplings)

    for number in range(m - 1, -1, -1):
        previous = cache[number]

        local = (
            fields[:, number, None]
            + (couplings[:, number, :] * valid[number]) @ bits.T
        )

        preceding = np.full_like(backward, -np.inf)

        for state in range(states):
            zero = (state << 1) & mask
            one = zero | 1

            selected_suffix = np.full((n, k + 1), -np.inf)

            selected_suffix[:, :-1] = (
                local[:, state, None]
                + backward[:, 1:, one]
            )

            preceding[:, :, state] = np.logaddexp(
                backward[:, :, zero],
                selected_suffix,
            )

            log_edge_probability = (
                previous[:, :-1, state]
                + local[:, state, None]
                + backward[:, 1:, one]
                - log_z[:, None]
            )

            selected_probability = np.exp(
                log_edge_probability
            ).sum(axis=1)

            inclusion[:, number] += selected_probability

            pair_marginal[:, number, :] += (
                selected_probability[:, None]
                * bits[state][None, :]
                * valid[number][None, :]
            )

        backward = preceding

    require(
        np.max(np.abs(backward[:, 0, 0] - log_z)) < 1e-8,
        "Prednja i zadnja particiona funkcija nisu saglasne.",
    )

    require(
        np.all(np.isfinite(inclusion))
        and np.all(np.isfinite(pair_marginal))
        and np.min(inclusion) >= -1e-9
        and np.max(inclusion) <= 1.0 + 1e-9,
        "Neispravne kanonske marginalne verovatnoće.",
    )

    require(
        np.max(np.abs(inclusion.sum(axis=1) - k)) < 1e-7,
        "Zbir uključnih verovatnoća nije broj izabranih brojeva.",
    )

    return log_z, inclusion, pair_marginal


def observed_score(fields, couplings, occupancy, pairs):
    return (
        np.sum(fields * occupancy, axis=1)
        + np.sum(couplings * pairs, axis=(1, 2))
    )


def objective(parameters, data, interaction_range, ridge, k, gradient=True):
    m = data["y"].shape[1]

    fields, couplings = potentials(
        parameters,
        data["x"],
        data["hp"],
        interaction_range,
    )

    log_z, inclusion, pair_marginal = infer(
        fields, couplings, k, marginals=gradient
    )

    scores = observed_score(
        fields,
        couplings,
        data["y"],
        data["yp"],
    )

    nll = log_z - scores

    require(
        np.all(np.isfinite(nll))
        and np.min(nll) >= -1e-7,
        "Neispravna verovatnoća posmatranog kola.",
    )

    penalty_weights = np.ones_like(parameters)
    penalty_weights[:m] = 0.1

    loss = float(
        nll.mean()
        + ridge * np.sum(
            penalty_weights * parameters * parameters
        )
    )

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    residual_fields = inclusion - data["y"]
    residual_pairs = pair_marginal - data["yp"]

    gradient_fields = np.empty((HISTORY_LAGS + 1, m))
    gradient_fields[0] = residual_fields.mean(axis=0)

    gradient_fields[1:] = np.einsum(
        "nm,nlm->lm",
        residual_fields,
        data["x"],
        optimize=True,
    ) / len(nll)

    mask = edge_mask(m, interaction_range)

    gradient_static = residual_pairs.mean(axis=0) * mask

    gradient_history = np.mean(
        residual_pairs * data["hp"],
        axis=0,
    ) * mask

    result = np.concatenate(
        (
            gradient_fields.ravel(),
            gradient_static.ravel(),
            gradient_history.ravel(),
        )
    )

    result += 2.0 * ridge * penalty_weights * parameters

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


def minimum_energy(fields, couplings, k):
    m = len(fields)
    interaction_range = couplings.shape[1]

    states = 1 << interaction_range
    mask = states - 1
    valid = edge_mask(m, interaction_range)

    values = np.full((k + 1, states), -np.inf)
    paths = [[None] * states for _ in range(k + 1)]

    values[0, 0] = 0.0
    paths[0][0] = ()

    for number in range(m):
        following = np.full_like(values, -np.inf)
        next_paths = [[None] * states for _ in range(k + 1)]

        for count in range(k + 1):
            for state in range(states):
                path = paths[count][state]

                if path is None:
                    continue

                local = fields[number] + sum(
                    couplings[number, distance]
                    * valid[number, distance]
                    * ((state >> distance) & 1)
                    for distance in range(interaction_range)
                )

                for selected in (0, 1):
                    new_count = count + selected

                    if new_count > k:
                        continue

                    destination = (
                        (state << 1) | selected
                    ) & mask

                    score = values[count, state] + selected * local

                    candidate = (
                        path + (number,)
                        if selected
                        else path
                    )

                    old_path = next_paths[new_count][destination]
                    old_score = following[new_count, destination]

                    if (
                        old_path is None
                        or score > old_score
                        or (score == old_score and candidate < old_path)
                    ):
                        following[new_count, destination] = score
                        next_paths[new_count][destination] = candidate

        values = following
        paths = next_paths

    candidates = [
        (float(values[k, state]), paths[k][state])
        for state in range(states)
        if paths[k][state] is not None
    ]

    require(
        bool(candidates),
        "Nije pronađeno stanje najmanje energije.",
    )

    best_score = max(value for value, _ in candidates)

    best = min(
        path
        for value, path in candidates
        if value == best_score
    )

    require(
        len(best) == k and len(set(best)) == k,
        "Stanje najmanje energije nema ispravnu veličinu.",
    )

    return best, best_score


def suffix_table(fields, couplings, k):
    m = len(fields)
    interaction_range = couplings.shape[1]

    states = 1 << interaction_range
    mask = states - 1
    bits = state_bits(interaction_range)
    valid = edge_mask(m, interaction_range)

    backward = np.full((m + 1, k + 1, states), -np.inf)
    backward[m, k, :] = 0.0

    for number in range(m - 1, -1, -1):
        local = (
            fields[number]
            + bits @ (couplings[number] * valid[number])
        )

        for state in range(states):
            zero = (state << 1) & mask
            one = zero | 1

            for count in range(k + 1):
                log_zero = backward[number + 1, count, zero]

                log_one = (
                    local[state]
                    + backward[number + 1, count + 1, one]
                    if count < k
                    else -np.inf
                )

                backward[number, count, state] = np.logaddexp(
                    log_zero,
                    log_one,
                )

    require(
        np.isfinite(backward[0, 0, 0]),
        "Kvantna priprema nema konačnu normalizaciju.",
    )

    return backward


def conditional_angles(fields, couplings, k, backward, number):
    interaction_range = couplings.shape[1]
    available_memory = min(number, interaction_range)

    count_bits = max(1, int(k).bit_length())
    count_dimension = 1 << count_bits
    memory_dimension = 1 << available_memory

    angles = np.zeros(count_dimension * memory_dimension)
    mask = (1 << interaction_range) - 1
    valid = edge_mask(len(fields), interaction_range)

    for memory in range(memory_dimension):
        local = fields[number] + sum(
            couplings[number, distance]
            * valid[number, distance]
            * ((memory >> distance) & 1)
            for distance in range(interaction_range)
        )

        zero = (memory << 1) & mask
        one = zero | 1

        for count in range(k + 1):
            if not np.isfinite(backward[number, count, memory]):
                continue

            log_zero = backward[number + 1, count, zero]

            log_one = (
                local + backward[number + 1, count + 1, one]
                if count < k
                else -np.inf
            )

            largest = max(log_zero, log_one)

            if not np.isfinite(largest):
                continue

            amplitude_zero = (
                math.exp(0.5 * (log_zero - largest))
                if np.isfinite(log_zero)
                else 0.0
            )

            amplitude_one = (
                math.exp(0.5 * (log_one - largest))
                if np.isfinite(log_one)
                else 0.0
            )

            control_index = count | (memory << count_bits)

            angles[control_index] = 2.0 * math.atan2(
                amplitude_one,
                amplitude_zero,
            )

    require(
        np.all(np.isfinite(angles))
        and np.min(angles) >= 0
        and np.max(angles) <= math.pi,
        "Neispravni uglovi kvantne Gibbsove pripreme.",
    )

    return angles, count_bits, available_memory


def multiplexed_rotation(angles):
    circuit = QuantumCircuit(
        1 + (len(angles) - 1).bit_length(),
        name="ConditionalGibbsRY",
    )

    # UCRY target is the first wire, followed by control wires.
    circuit.append(
        UCRYGate([float(value) for value in angles]),
        list(range(circuit.num_qubits)),
    )

    return circuit


def gibbs_preparation_circuit(fields, couplings, k):
    m = len(fields)
    interaction_range = couplings.shape[1]
    count_bits = max(1, int(k).bit_length())

    backward = suffix_table(fields, couplings, k)

    circuit = QuantumCircuit(m + count_bits, name="CoherentGibbs")
    count_register = list(range(m, m + count_bits))

    for number in range(m):
        angles, _, available = conditional_angles(
            fields, couplings, k, backward, number
        )

        memory_register = [
            number - 1 - distance
            for distance in range(available)
        ]

        wires = [number] + count_register + memory_register

        circuit.append(
            UCRYGate([float(value) for value in angles]),
            wires,
        )

        # Reversible increment of the occupation counter,
        # controlled by the newly selected occupation qubit.
        for bit in range(count_bits - 1, 0, -1):
            controls = [number] + count_register[:bit]

            if len(controls) == 2:
                circuit.ccx(
                    controls[0],
                    controls[1],
                    count_register[bit],
                )
            else:
                circuit.mcx(
                    controls,
                    count_register[bit],
                )

        circuit.cx(number, count_register[0])

    return circuit, backward


def quantum_path_log_probability(fields, couplings, k, selected):
    """
    Exact branch-amplitude contraction using local Qiskit circuits.
    No sampling and no full 42-qubit statevector.
    """
    backward = suffix_table(fields, couplings, k)

    interaction_range = couplings.shape[1]
    mask = (1 << interaction_range) - 1

    selected = set(selected)
    count = 0
    memory = 0
    log_amplitude = 0.0

    largest_error = 0.0

    for number in range(len(fields)):
        angles, count_bits, available = conditional_angles(
            fields, couplings, k, backward, number
        )

        reduced_memory = memory & ((1 << available) - 1)
        control_index = count | (reduced_memory << count_bits)

        circuit = multiplexed_rotation(angles)
        dimension = 1 << circuit.num_qubits

        initial_index = control_index << 1

        state = Statevector.from_int(
            initial_index,
            dimension,
        ).evolve(circuit)

        zero_amplitude = complex(state.data[initial_index])
        one_amplitude = complex(state.data[initial_index | 1])

        angle = float(angles[control_index])

        largest_error = max(
            largest_error,
            abs(zero_amplitude - math.cos(angle / 2.0)),
            abs(one_amplitude - math.sin(angle / 2.0)),
        )

        require(
            abs(float(np.vdot(
                state.data, state.data
            ).real) - 1.0) < 2e-9,
            "Lokalno Qiskit stanje nije normalizovano.",
        )

        selected_bit = int(number in selected)

        amplitude = (
            one_amplitude
            if selected_bit
            else zero_amplitude
        )

        magnitude = abs(amplitude)

        require(
            magnitude > 0 and math.isfinite(magnitude),
            "Izabrana Gibbsova putanja ima nultu numeričku amplitudu.",
        )

        log_amplitude += math.log(magnitude)

        count += selected_bit
        memory = ((memory << 1) | selected_bit) & mask

    require(
        count == k,
        "Kvantna putanja nema tačno sedam zauzetih stanja.",
    )

    require(
        largest_error < 2e-9,
        "Qiskit uslovne rotacije nisu prošle proveru.",
    )

    return 2.0 * log_amplitude, largest_error


def quantum_hamiltonian(fields, couplings):
    m = len(fields)
    interaction_range = couplings.shape[1]
    terms = {}

    def add(indices, coefficient):
        label = ["I"] * m

        for index in indices:
            label[m - 1 - index] = "Z"

        label = "".join(label)
        terms[label] = terms.get(label, 0.0) + float(coefficient)

    for number in range(m):
        value = float(fields[number])

        add((), -value / 2.0)
        add((number,), value / 2.0)

        for distance in range(interaction_range):
            other = number - distance - 1

            if other < 0:
                continue

            value = float(couplings[number, distance])

            add((), -value / 4.0)
            add((number,), value / 4.0)
            add((other,), value / 4.0)
            add((number, other), -value / 4.0)

    return SparsePauliOp.from_list(
        [
            (label, coefficient)
            for label, coefficient in terms.items()
            if coefficient != 0.0
        ]
        or [("I" * m, 0.0)]
    )


def load_draws(path):
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = []
    blank = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                blank += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = tuple(int(item.strip()) for item in row)
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == DRAW_SIZE
                and min(numbers) >= 1
                and max(numbers) <= NUMBER_COUNT,
                f"{path.name}, red {line_number}: neispravna sedmorka.",
            )

            rows.append(
                tuple(value - 1 for value in sorted(numbers))
            )

    require(len(rows) >= 100, "Nedovoljno istorijskih kola.")

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros((len(draws), NUMBER_COUNT))
    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def checkpoint_path(data, interaction_range, ridge, k):
    digest = hashlib.sha256()

    digest.update(repr(
        (
            CHECKPOINT_VERSION,
            interaction_range,
            float(ridge),
            k,
            OPTIMIZER_STEPS,
            qiskit.__version__,
            np.__version__,
        )
    ).encode("utf-8"))

    for key in sorted(data):
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    return CHECKPOINT_DIR / (digest.hexdigest() + ".npz")


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                parameters=np.asarray(parameters, dtype=float),
                iterations=np.array(int(iterations)),
                done=np.array(int(done)),
                message=np.array(str(message)),
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, shape):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            parameters = saved["parameters"].astype(float)
            iterations = int(saved["iterations"].item())
            done = bool(saved["done"].item())
            message = str(saved["message"].item())

        require(
            parameters.shape == shape
            and np.all(np.isfinite(parameters))
            and iterations >= 0,
            "Neispravna kontrolna tačka.",
        )

        return parameters, iterations, done, message

    except (
        OSError, ValueError, KeyError, RuntimeError,
        EOFError, BadZipFile,
    ) as error:
        say(
            f"Kontrolna tačka nije učitana: {path.name}",
            str(error),
        )
        return None


@dataclass
class Fitted:
    parameters: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    message: str


def fit(data, interaction_range, ridge, k, label):
    m = data["y"].shape[1]

    initial = np.zeros(parameter_count(m, interaction_range))
    path = checkpoint_path(data, interaction_range, ridge, k)
    saved = load_checkpoint(path, initial.shape)

    parameters = initial.copy()
    used_iterations = 0
    done = False
    message = "Novi proračun."

    if saved is not None:
        parameters, used_iterations, done, message = saved

    started = time.perf_counter()

    say(
        "",
        label,
        f"Domet={interaction_range} | λ={ridge:g}"
        f" | prelaza={len(data['y'])}",
    )

    try:
        best_loss, best_gradient = objective(
            parameters, data, interaction_range, ridge, k
        )

    except (
        RuntimeError, FloatingPointError, np.linalg.LinAlgError,
    ):
        require(
            saved is not None,
            "Početni Hamiltonijan nije numerički validan.",
        )

        parameters = initial.copy()
        used_iterations = 0
        done = False

        best_loss, best_gradient = objective(
            parameters, data, interaction_range, ridge, k
        )

    best_parameters = parameters.copy()
    best_gradient = best_gradient.copy()

    added_iterations = 0
    optimizer_success = False

    if done:
        say("Učitan završen proračun iz kontrolne tačke.")

    else:
        save_checkpoint(
            path, best_parameters, used_iterations, False,
            "Početna ili nastavljena validna tačka.",
        )

        def function(current):
            nonlocal best_loss, best_parameters, best_gradient

            loss, gradient = objective(
                current, data, interaction_range, ridge, k
            )

            if loss < best_loss:
                best_loss = float(loss)
                best_parameters = current.copy()
                best_gradient = gradient.copy()

            return loss, gradient

        def callback(current):
            nonlocal added_iterations
            added_iterations += 1

            if added_iterations % 25 == 0:
                total = used_iterations + added_iterations

                save_checkpoint(
                    path, best_parameters, total, False,
                    "Sačuvana najbolja validna tačka.",
                )

                say(
                    f"Iteracija {total}"
                    f" | proteklo {time.perf_counter() - started:.1f} s",
                    f"Najbolji cilj: {best_loss:.9f}",
                    f"Max|grad|: {np.max(np.abs(best_gradient)):.3e}",
                )

        remaining = max(0, OPTIMIZER_STEPS - used_iterations)

        if remaining:
            try:
                result = minimize(
                    function,
                    parameters,
                    method="L-BFGS-B",
                    jac=True,
                    callback=callback,
                    options={
                        "maxiter": remaining,
                        "maxls": 60,
                        "maxcor": 20,
                        "ftol": 1e-13,
                        "gtol": 1e-6,
                    },
                )

                optimizer_success = bool(result.success)
                message = str(result.message)
                function(result.x)

            except (
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                message = (
                    "Pretraga prekinuta; zadržana validna tačka: "
                    + str(error)
                )
        else:
            message = "Dostignut sačuvani budžet iteracija."

        used_iterations += added_iterations

    loss, gradient = objective(
        best_parameters, data, interaction_range, ridge, k
    )

    maximum_gradient = float(np.max(np.abs(gradient)))
    converged = maximum_gradient <= GRADIENT_LIMIT

    if not done:
        message = (
            "OPTIMIZER_SUCCESS: "
            if optimizer_success
            else "OPTIMIZER_NOT_CONFIRMED: "
        ) + message

        save_checkpoint(
            path, best_parameters, used_iterations, True, message
        )

    say(
        f"Zadržana validna tačka: {loss:.9f}",
        f"Max|grad|: {maximum_gradient:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if converged else "NIJE dostignut"),
        message,
    )

    return Fitted(
        best_parameters.copy(),
        float(loss),
        maximum_gradient,
        converged,
        message,
    )


def evaluate(fitted, data, interaction_range, k, hits=False):
    total = 0.0
    histogram = np.zeros(k + 1, dtype=np.int64)

    for start in range(0, len(data["y"]), 512):
        batch = subset(data, start, start + 512)

        fields, couplings = potentials(
            fitted.parameters,
            batch["x"],
            batch["hp"],
            interaction_range,
        )

        log_z = infer(fields, couplings, k)[0]

        score = observed_score(
            fields, couplings, batch["y"], batch["yp"]
        )

        nll = log_z - score

        require(
            np.all(np.isfinite(nll))
            and np.min(nll) >= -1e-7,
            "Neispravna test verovatnoća.",
        )

        total += float(nll.sum())

        if hits:
            for row in range(len(fields)):
                selected, _ = minimum_energy(
                    fields[row], couplings[row], k
                )

                actual = np.flatnonzero(batch["y"][row])

                histogram[
                    len(set(selected).intersection(actual.tolist()))
                ] += 1

    return total / len(data["y"]), histogram


def self_checks():
    say("", "Pokrećem ugrađene Qiskit i matematičke provere.")

    # Verify UCRY target/control ordering before any fitting.
    probe_angles = [0.17, 0.39, 0.71, 1.03]
    probe = multiplexed_rotation(probe_angles)

    for control, angle in enumerate(probe_angles):
        index = control << 1
        state = Statevector.from_int(
            index,
            1 << probe.num_qubits,
        ).evolve(probe)

        require(
            abs(state.data[index] - math.cos(angle / 2.0)) < 1e-10
            and abs(state.data[index | 1] - math.sin(angle / 2.0)) < 1e-10,
            "UCRY raspored cilja i kontrola nije prošao proveru.",
        )

    occupancy = np.array(
        [
            [1, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
        ],
        dtype=float,
    )

    k = 3
    largest_gradient_error = 0.0
    largest_quantum_error = 0.0

    for interaction_range in (1, 2):
        data = make_data(occupancy, k, interaction_range)

        parameters = 0.04 * np.sin(
            np.arange(
                parameter_count(6, interaction_range)
            ) + 0.3
        )

        ridge = 0.03

        _, analytic = objective(
            parameters, data, interaction_range, ridge, k
        )

        numerical = np.empty_like(parameters)
        epsilon = 1e-6

        for parameter in range(len(parameters)):
            positive = parameters.copy()
            negative = parameters.copy()

            positive[parameter] += epsilon
            negative[parameter] -= epsilon

            numerical[parameter] = (
                objective(
                    positive, data, interaction_range, ridge, k,
                    gradient=False,
                )
                - objective(
                    negative, data, interaction_range, ridge, k,
                    gradient=False,
                )
            ) / (2.0 * epsilon)

        gradient_error = float(
            np.max(np.abs(analytic - numerical))
        )

        largest_gradient_error = max(
            largest_gradient_error,
            gradient_error,
        )

        require(
            gradient_error < 3e-6,
            f"Provera gradijenta nije prošla: {gradient_error:.3e}",
        )

        fields, couplings = potentials(
            parameters,
            data["x"],
            data["hp"],
            interaction_range,
        )

        log_z, marginal, pair_marginal = infer(
            fields, couplings, k, marginals=True
        )

        subsets = list(combinations(range(6), k))
        all_y = np.zeros((len(subsets), 6))

        for row, selected in enumerate(subsets):
            all_y[row, list(selected)] = 1.0

        all_pairs = pair_features(all_y, interaction_range)

        for row in range(len(fields)):
            scores = (
                all_y @ fields[row]
                + np.sum(
                    all_pairs * couplings[row][None, :, :],
                    axis=(1, 2),
                )
            )

            direct_log_z = float(logsumexp(scores))
            probabilities = np.exp(scores - direct_log_z)

            require(
                abs(direct_log_z - log_z[row]) < 1e-10,
                "Potpuna provera particione funkcije nije prošla.",
            )

            require(
                np.max(np.abs(
                    probabilities @ all_y - marginal[row]
                )) < 1e-9,
                "Potpuna provera margina nije prošla.",
            )

            direct_pairs = np.einsum(
                "n,nmr->mr",
                probabilities,
                all_pairs,
            )

            require(
                np.max(np.abs(
                    direct_pairs - pair_marginal[row]
                )) < 1e-9,
                "Potpuna provera parnih margina nije prošla.",
            )

            best, score = minimum_energy(
                fields[row], couplings[row], k
            )

            require(
                best == subsets[int(np.argmax(scores))]
                and abs(score - float(np.max(scores))) < 1e-10,
                "Provera tačnog minimuma energije nije prošla.",
            )

        circuit, backward = gibbs_preparation_circuit(
            fields[0], couplings[0], k
        )

        state = Statevector.from_instruction(circuit)

        actual = state.probabilities(
            qargs=list(range(6))
        )

        expected = np.zeros(1 << 6)

        for selected in subsets:
            y = np.zeros((1, 6))
            y[0, list(selected)] = 1.0

            score = observed_score(
                fields[:1],
                couplings[:1],
                y,
                pair_features(y, interaction_range),
            )[0]

            index = sum(1 << number for number in selected)
            expected[index] = math.exp(score - log_z[0])

        quantum_error = float(
            np.max(np.abs(actual - expected))
        )

        largest_quantum_error = max(
            largest_quantum_error,
            quantum_error,
        )

        require(
            quantum_error < 2e-9,
            "Puna mala Qiskit Gibbsova priprema nije prošla.",
        )

        best, score = minimum_energy(
            fields[0], couplings[0], k
        )

        quantum_logp, local_error = quantum_path_log_probability(
            fields[0], couplings[0], k, best
        )

        require(
            abs(quantum_logp - (score - log_z[0])) < 2e-8,
            "Kompaktna Qiskit putanja nije prošla proveru.",
        )

        hamiltonian = quantum_hamiltonian(
            fields[0], couplings[0]
        )

        diagonal = hamiltonian.to_matrix().diagonal().real

        for index in range(1 << 6):
            y = np.array(
                [[(index >> number) & 1 for number in range(6)]],
                dtype=float,
            )

            score = observed_score(
                fields[:1],
                couplings[:1],
                y,
                pair_features(y, interaction_range),
            )[0]

            require(
                abs(diagonal[index] + score) < 1e-10,
                "Qiskit Pauli Hamiltonijan nije saglasan sa energijom.",
            )

        mean_score = float(
            np.sum(fields[0] * marginal[0])
            + np.sum(couplings[0] * pair_marginal[0])
        )

        quantum_mean_energy = float(actual @ diagonal)

        require(
            abs(quantum_mean_energy + mean_score) < 2e-9,
            "Srednja energija kvantnog očitavanja nije prošla proveru.",
        )

        for boundary_k in (1, 6):
            zero_fields = np.zeros((1, 6))
            zero_pairs = np.zeros((1, 6, interaction_range))

            boundary_z, boundary_marginal, _ = infer(
                zero_fields,
                zero_pairs,
                boundary_k,
                marginals=True,
            )

            require(
                abs(
                    boundary_z[0]
                    - math.log(math.comb(6, boundary_k))
                ) < 1e-10
                and np.max(np.abs(
                    boundary_marginal[0] - boundary_k / 6
                )) < 1e-10,
                "Granična kanonska provera nije prošla.",
            )

    first = history_context(occupancy, [1], k)

    require(
        np.array_equal(first[0, 0], occupancy[0] - k / 6)
        and np.all(first[0, 1] == 0),
        "Prvi istorijski prelaz nije ispravan.",
    )

    say(
        "Ugrađene provere su prošle.",
        f"Najveća greška gradijenta: {largest_gradient_error:.3e}",
        f"Najveća greška pune Qiskit raspodele: {largest_quantum_error:.3e}",
        "Provereni su Hamiltonijan, amplitude, margine i minimum energije.",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def print_result(result):
    say(
        "",
        result["name"],
        "NEXT: "
        + " ".join(f"{number:02d}" for number in result["next"]),
        f"Validacioni NLL: {result['validation']:.9f}",
        f"Odvojeni test NLL: {result['test']:.9f}",
        "Pogodaka 0–7: "
        + " ".join(str(int(value)) for value in result["histogram"]),
        f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
        f"Najmanja energija NEXT: {result['ground_energy']:.10f}",
        f"Srednja energija U: {result['U']:.10f}",
        f"Slobodna energija F: {result['F']:.10f}",
        f"Termodinamička entropija S: {result['S']:.10f}",
        f"Greška Qiskit log-verovatnoće: {result['quantum_error']:.3e}",
        f"Greška lokalnih Qiskit amplituda: {result['local_error']:.3e}",
        f"Osetljivost margina: {result['sensitivity']:.9e}",
        f"Max|grad|: {result['gradient']:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if result["converged"] else "NIJE dostignut"),
    )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process_file(path, draws, occupancy):
    n = len(draws) - 1

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno istorije za hronološku podelu.",
    )

    datasets = {
        interaction_range: make_data(
            occupancy, DRAW_SIZE, interaction_range
        )
        for interaction_range in (1, 2)
    }

    scores = []

    for interaction_range, ridge in CANDIDATES:
        data = datasets[interaction_range]
        total_nll = 0.0
        total_count = 0
        incomplete_folds = 0

        for fold, (train_end, validation_end) in enumerate(
            ((split60, split70), (split70, split80)),
            1,
        ):
            fitted = fit(
                subset(data, 0, train_end),
                interaction_range,
                ridge,
                DRAW_SIZE,
                f"{path.name} / validacija {fold}",
            )

            incomplete_folds += int(not fitted.converged)

            score = evaluate(
                fitted,
                subset(data, train_end, validation_end),
                interaction_range,
                DRAW_SIZE,
            )[0]

            count = validation_end - train_end
            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count

        scores.append(
            (score, interaction_range, ridge, incomplete_folds)
        )

        say(
            f"KANDIDAT: domet={interaction_range} | λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
        )

    validation_nll, interaction_range, ridge, incomplete_folds = min(
        scores, key=lambda item: item[0]
    )

    data = datasets[interaction_range]

    say(
        "",
        f"IZABRANO: domet={interaction_range} | λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit(
        subset(data, 0, split80),
        interaction_range,
        ridge,
        DRAW_SIZE,
        f"{path.name} / odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_model,
        subset(data, split80, n),
        interaction_range,
        DRAW_SIZE,
        hits=True,
    )

    final_model = fit(
        data,
        interaction_range,
        ridge,
        DRAW_SIZE,
        f"{path.name} / završno učenje svih {n} prelaza",
    )

    next_context = history_context(
        occupancy, [len(occupancy)], DRAW_SIZE
    )

    next_pairs = pair_features(
        occupancy[-1:], interaction_range
    )

    fields, couplings = potentials(
        final_model.parameters,
        next_context,
        next_pairs,
        interaction_range,
    )

    log_z, inclusion, pair_marginal = infer(
        fields, couplings, DRAW_SIZE, marginals=True
    )

    best, best_score = minimum_energy(
        fields[0], couplings[0], DRAW_SIZE
    )

    logp = best_score - float(log_z[0])

    require(
        -math.log(math.comb(NUMBER_COUNT, DRAW_SIZE)) - 1e-7
        <= logp <= 1e-7,
        "Neispravna verovatnoća maksimuma.",
    )

    # Construct the full logical circuit, but do not allocate its statevector.
    circuit, backward = gibbs_preparation_circuit(
        fields[0], couplings[0], DRAW_SIZE
    )

    hamiltonian = quantum_hamiltonian(
        fields[0], couplings[0]
    )

    quantum_logp, local_error = quantum_path_log_probability(
        fields[0], couplings[0], DRAW_SIZE, best
    )

    quantum_error = abs(quantum_logp - logp)

    require(
        quantum_error < 2e-8,
        "Qiskit Born verovatnoća NEXT nije saglasna sa Gibbsovom raspodelom.",
    )

    require(
        abs(float(backward[0, 0, 0]) - float(log_z[0])) < 1e-8,
        "Kvantna i kanonska normalizacija nisu saglasne.",
    )

    mean_score = float(
        np.sum(fields * inclusion)
        + np.sum(couplings * pair_marginal)
    )

    U = -mean_score
    F = -float(log_z[0])
    entropy = float(log_z[0]) - mean_score

    require(
        -1e-7 <= entropy
        <= math.log(math.comb(NUMBER_COUNT, DRAW_SIZE)) + 1e-7,
        "Termodinamička entropija nije u dozvoljenim granicama.",
    )

    previous_fields, previous_couplings = potentials(
        final_model.parameters,
        data["x"][-8:],
        data["hp"][-8:],
        interaction_range,
    )

    _, previous_marginal, _ = infer(
        previous_fields,
        previous_couplings,
        DRAW_SIZE,
        marginals=True,
    )

    sensitivity = float(np.max(np.abs(
        previous_marginal - inclusion[0]
    )))

    warnings = []

    if incomplete_folds:
        warnings.append(
            "Izabrani kandidat ima validacione treninge"
            " bez potvrđenog gradijentnog uslova."
        )

    if not test_model.converged:
        warnings.append(
            "Model odvojenog testa nema potvrđen gradijentni uslov."
        )

    if not final_model.converged:
        warnings.append(
            "Završni model nema potvrđenu konvergenciju;"
            " NEXT je maksimum njegove validne raspodele."
        )

    run = longest_run(best)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if best[-1] - best[0] <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva je u rasponu od deset brojeva."
        )

    repeats = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if repeats:
        warnings.append(
            f"NEXT se pojavljuje {repeats} puta u istoriji."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Margine slabo reaguju na proverene istorijske kontekste."
        )

    say(
        "",
        f"Qiskit Hamiltonijan: {hamiltonian.num_qubits} kubita",
        f"Broj Pauli članova: {len(hamiltonian)}",
        f"Logičko kolo pripreme: {circuit.num_qubits} kubita",
        "Gust vektor tog kola nije alociran.",
        f"Operacije kola: {dict(circuit.count_ops())}",
        f"Odvojeni test: {n - split80} prelaza",
    )

    result = {
        "name": path.name,
        "next": tuple(value + 1 for value in best),
        "validation": validation_nll,
        "test": test_nll,
        "histogram": histogram,
        "probability": math.exp(quantum_logp),
        "ground_energy": -best_score,
        "U": U,
        "F": F,
        "S": entropy,
        "quantum_error": quantum_error,
        "local_error": local_error,
        "sensitivity": sensitivity,
        "gradient": final_model.gradient_norm,
        "converged": final_model.converged,
        "warnings": warnings,
    }

    print_result(result)

    return result


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    say(
        "QUANTUM TERMODINAMIKA V1 — Qiskit Gibbs — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Uči se dijagonalni Hamiltonijan sa lokalnim poljima i parovima.",
        "Beta=1 određuje bezdimenzionalne energetske jedinice.",
        "Qiskit kolo priprema amplitude proporcionalne exp(-E/2).",
        "Bornovo očitavanje daje Gibbsovu raspodelu.",
        "Koherentno stanje pripreme nije mešano termalno stanje.",
        "U, F i S odnose se na dijagonalno Gibbsovo stanje.",
        "Hamiltonijan je komutirajući; ne tvrdi se kvantna prednost.",
        "Trening koristi tačnu kanonsku funkciju gubitka i njene izvode.",
        "Qiskit priprema i očitavanje proveravaju se zasebno.",
        "Završni NEXT proverava se lokalnim Qiskit amplitudnim kontrakcijama.",
        "Svaki uzorak koristi samo prethodna kola.",
        "Završno učenje koristi sve istorijske prelaze.",
        "Nema slučajnog generisanja, šuma ili mešanja redova.",
        "Limit iteracija ne odbacuje validnu sačuvanu raspodelu.",
        f"Kontrolne tačke: {CHECKPOINT_DIR}",
    )

    require(
        len(set(path.resolve() for path in FILES)) == len(FILES),
        "Ulazne putanje nisu različite.",
    )

    datasets = []
    hashes = []

    for path in FILES:
        require(path.is_file(), f"Fajl nije pronađen: {path}")

        draws, occupancy, fingerprint = load_draws(path)
        datasets.append((path, draws, occupancy))
        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = [
        process_file(path, draws, occupancy)
        for path, draws, occupancy in datasets
    ]

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if results[first]["next"] == results[second]["next"]:
                results[first]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[second]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

                results[second]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[first]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

    say("", "ZAVRŠNA TRI NEXT REZULTATA")

    for result in results:
        print_result(result)

    say(
        "",
        "Obrada je završena.",
        "NEXT je stanje najmanje energije naučenog Hamiltonijana.",
        "Njegova Born verovatnoća proverena je Qiskit kontrakcijom.",
        "Nepotvrđena konvergencija ostaje označena.",
        "Prediktivna vrednost procenjuje se odvojenim testom.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM TERMODINAMIKA V1 — Qiskit Gibbs — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Uči se dijagonalni Hamiltonijan sa lokalnim poljima i parovima.
Beta=1 određuje bezdimenzionalne energetske jedinice.
Qiskit kolo priprema amplitude proporcionalne exp(-E/2).
Bornovo očitavanje daje Gibbsovu raspodelu.
Koherentno stanje pripreme nije mešano termalno stanje.
U, F i S odnose se na dijagonalno Gibbsovo stanje.
Hamiltonijan je komutirajući; ne tvrdi se kvantna prednost.
Trening koristi tačnu kanonsku funkciju gubitka i njene izvode.
Qiskit priprema i očitavanje proveravaju se zasebno.
Završni NEXT proverava se lokalnim Qiskit amplitudnim kontrakcijama.
Svaki uzorak koristi samo prethodna kola.
Završno učenje koristi sve istorijske prelaze.
Nema slučajnog generisanja, šuma ili mešanja redova.
Limit iteracija ne odbacuje validnu sačuvanu raspodelu.
Kontrolne tačke: /Users/4c/Desktop/GHQ/Astra/Quantum_Termodinamika/Quantum_Termodinamika_v1_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem ugrađene Qiskit i matematičke provere.
Ugrađene provere su prošle.
Najveća greška gradijenta: 4.583e-10
Najveća greška pune Qiskit raspodele: 1.318e-16
Provereni su Hamiltonijan, amplitude, margine i minimum energije.

loto7_4710_k86.csv / validacija 1
Domet=1 | λ=0.03 | prelaza=2825
Zadržana validna tačka: 16.536976511
Max|grad|: 4.547e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.554325935

loto7_4710_k86.csv / validacija 2
Domet=1 | λ=0.03 | prelaza=3296
Zadržana validna tačka: 16.538618230
Max|grad|: 4.856e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.553734775
KANDIDAT: domet=1 | λ=0.03
Zbirni validacioni NLL: 16.554030355

loto7_4710_k86.csv / validacija 1
Domet=1 | λ=0.3 | prelaza=2825
Zadržana validna tačka: 16.543393944
Max|grad|: 8.594e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.551035327

loto7_4710_k86.csv / validacija 2
Domet=1 | λ=0.3 | prelaza=3296
Zadržana validna tačka: 16.544115581
Max|grad|: 4.443e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549118824
KANDIDAT: domet=1 | λ=0.3
Zbirni validacioni NLL: 16.550077075

loto7_4710_k86.csv / validacija 1
Domet=1 | λ=3 | prelaza=2825
Zadržana validna tačka: 16.547290359
Max|grad|: 2.252e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548669574

loto7_4710_k86.csv / validacija 2
Domet=1 | λ=3 | prelaza=3296
Zadržana validna tačka: 16.547475495
Max|grad|: 1.517e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548130762
KANDIDAT: domet=1 | λ=3
Zbirni validacioni NLL: 16.548400168

loto7_4710_k86.csv / validacija 1
Domet=2 | λ=0.03 | prelaza=2825
Zadržana validna tačka: 16.535739701
Max|grad|: 4.092e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.555494799

loto7_4710_k86.csv / validacija 2
Domet=2 | λ=0.03 | prelaza=3296
Zadržana validna tačka: 16.537566027
Max|grad|: 5.390e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.553550237
KANDIDAT: domet=2 | λ=0.03
Zbirni validacioni NLL: 16.554522518

loto7_4710_k86.csv / validacija 1
Domet=2 | λ=0.3 | prelaza=2825
Zadržana validna tačka: 16.543215535
Max|grad|: 8.502e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.551101951

loto7_4710_k86.csv / validacija 2
Domet=2 | λ=0.3 | prelaza=3296
Zadržana validna tačka: 16.543957146
Max|grad|: 2.430e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549108346
KANDIDAT: domet=2 | λ=0.3
Zbirni validacioni NLL: 16.550105149

loto7_4710_k86.csv / validacija 1
Domet=2 | λ=3 | prelaza=2825
Zadržana validna tačka: 16.547265023
Max|grad|: 1.763e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548657725

loto7_4710_k86.csv / validacija 2
Domet=2 | λ=3 | prelaza=3296
Zadržana validna tačka: 16.547451232
Max|grad|: 1.652e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548133349
KANDIDAT: domet=2 | λ=3
Zbirni validacioni NLL: 16.548395537

IZABRANO: domet=2 | λ=3
Validacioni NLL: 16.548395537

loto7_4710_k86.csv / odvojeni test
Domet=2 | λ=3 | prelaza=3767
Zadržana validna tačka: 16.547503732
Max|grad|: 9.129e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86.csv / završno učenje svih 4709 prelaza
Domet=2 | λ=3 | prelaza=4709
Zadržana validna tačka: 16.547416003
Max|grad|: 2.900e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Qiskit Hamiltonijan: 39 kubita
Broj Pauli članova: 115
Logičko kolo pripreme: 42 kubita
Gust vektor tog kola nije alociran.
Operacije kola: {'ucry': 39, 'mcx': 39, 'ccx': 39, 'cx': 39}
Odvojeni test: 942 prelaza

loto7_4710_k86.csv
NEXT: 08 11 22 23 26 34 37
Validacioni NLL: 16.548395537
Odvojeni test NLL: 16.547291316
Pogodaka 0–7: 207 376 272 74 13 0 0 0
Modelna verovatnoća NEXT: 7.08148946422e-08
Najmanja energija NEXT: -0.0857942296
Srednja energija U: -0.0005903001
Slobodna energija F: -16.5489907115
Termodinamička entropija S: 16.5484004114
Greška Qiskit log-verovatnoće: 1.631e-10
Greška lokalnih Qiskit amplituda: 9.276e-11
Osetljivost margina: 1.534182216e-04
Max|grad|: 2.900e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=1 | λ=0.03 | prelaza=1785
Zadržana validna tačka: 16.530803097
Max|grad|: 8.959e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.553265729

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=1 | λ=0.03 | prelaza=2083
Zadržana validna tačka: 16.532418637
Max|grad|: 6.394e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.560832230
KANDIDAT: domet=1 | λ=0.03
Zbirni validacioni NLL: 16.557042621

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=1 | λ=0.3 | prelaza=1785
Zadržana validna tačka: 16.539770965
Max|grad|: 2.616e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548681504

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=1 | λ=0.3 | prelaza=2083
Zadržana validna tačka: 16.540388898
Max|grad|: 2.157e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.555629330
KANDIDAT: domet=1 | λ=0.3
Zbirni validacioni NLL: 16.552149579

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=1 | λ=3 | prelaza=1785
Zadržana validna tačka: 16.546303058
Max|grad|: 3.362e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.547464181

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=1 | λ=3 | prelaza=2083
Zadržana validna tačka: 16.546454706
Max|grad|: 3.468e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549575080
KANDIDAT: domet=1 | λ=3
Zbirni validacioni NLL: 16.548517857

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=2 | λ=0.03 | prelaza=1785
Zadržana validna tačka: 16.528600042
Max|grad|: 4.679e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.556339594

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=2 | λ=0.03 | prelaza=2083
Zadržana validna tačka: 16.530772474
Max|grad|: 4.262e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.561743536
KANDIDAT: domet=2 | λ=0.03
Zbirni validacioni NLL: 16.559037024

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=2 | λ=0.3 | prelaza=1785
Zadržana validna tačka: 16.539450461
Max|grad|: 8.770e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549046922

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=2 | λ=0.3 | prelaza=2083
Zadržana validna tačka: 16.540151528
Max|grad|: 6.903e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.555757462
KANDIDAT: domet=2 | λ=0.3
Zbirni validacioni NLL: 16.552396553

loto7_4710_k86_loto_2977.csv / validacija 1
Domet=2 | λ=3 | prelaza=1785
Zadržana validna tačka: 16.546256774
Max|grad|: 3.263e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.547497084

loto7_4710_k86_loto_2977.csv / validacija 2
Domet=2 | λ=3 | prelaza=2083
Zadržana validna tačka: 16.546419754
Max|grad|: 1.738e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549592934
KANDIDAT: domet=2 | λ=3
Zbirni validacioni NLL: 16.548543248

IZABRANO: domet=1 | λ=3
Validacioni NLL: 16.548517857

loto7_4710_k86_loto_2977.csv / odvojeni test
Domet=1 | λ=3 | prelaza=2380
Zadržana validna tačka: 16.546819648
Max|grad|: 1.244e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza
Domet=1 | λ=3 | prelaza=2976
Zadržana validna tačka: 16.546922083
Max|grad|: 3.421e-09
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Qiskit Hamiltonijan: 39 kubita
Broj Pauli članova: 78
Logičko kolo pripreme: 42 kubita
Gust vektor tog kola nije alociran.
Operacije kola: {'ucry': 39, 'mcx': 39, 'ccx': 39, 'cx': 39}
Odvojeni test: 596 prelaza

loto7_4710_k86_loto_2977.csv
NEXT: 08 16 21 22 23 34 35
Validacioni NLL: 16.548517857
Odvojeni test NLL: 16.547774437
Pogodaka 0–7: 129 242 163 52 10 0 0 0
Modelna verovatnoća NEXT: 7.19158969537e-08
Najmanja energija NEXT: -0.1013913701
Srednja energija U: -0.0008585716
Slobodna energija F: -16.5491598687
Termodinamička entropija S: 16.5483012971
Greška Qiskit log-verovatnoće: 3.553e-15
Greška lokalnih Qiskit amplituda: 3.331e-16
Osetljivost margina: 2.064194605e-04
Max|grad|: 3.421e-09
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=1 | λ=0.03 | prelaza=1039
Zadržana validna tačka: 16.511859375
Max|grad|: 2.737e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.563458460

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=1 | λ=0.03 | prelaza=1212
Zadržana validna tačka: 16.517258333
Max|grad|: 8.573e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.562521986
KANDIDAT: domet=1 | λ=0.03
Zbirni validacioni NLL: 16.562990223

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=1 | λ=0.3 | prelaza=1039
Zadržana validna tačka: 16.530670934
Max|grad|: 7.377e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.554981498

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=1 | λ=0.3 | prelaza=1212
Zadržana validna tačka: 16.533469514
Max|grad|: 3.895e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.559117707
KANDIDAT: domet=1 | λ=0.3
Zbirni validacioni NLL: 16.557049603

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=1 | λ=3 | prelaza=1039
Zadržana validna tačka: 16.543929662
Max|grad|: 4.544e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548153698

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=1 | λ=3 | prelaza=1212
Zadržana validna tačka: 16.544671145
Max|grad|: 5.167e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549877910
KANDIDAT: domet=1 | λ=3
Zbirni validacioni NLL: 16.549015804

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=2 | λ=0.03 | prelaza=1039
Zadržana validna tačka: 16.508624830
Max|grad|: 2.794e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.560089318

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=2 | λ=0.03 | prelaza=1212
Zadržana validna tačka: 16.513981814
Max|grad|: 2.594e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.565760674
KANDIDAT: domet=2 | λ=0.03
Zbirni validacioni NLL: 16.562924996

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=2 | λ=0.3 | prelaza=1039
Zadržana validna tačka: 16.530220568
Max|grad|: 3.020e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.554356148

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=2 | λ=0.3 | prelaza=1212
Zadržana validna tačka: 16.533002217
Max|grad|: 6.482e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.559548924
KANDIDAT: domet=2 | λ=0.3
Zbirni validacioni NLL: 16.556952536

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Domet=2 | λ=3 | prelaza=1039
Zadržana validna tačka: 16.543863222
Max|grad|: 6.365e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.548053915

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Domet=2 | λ=3 | prelaza=1212
Zadržana validna tačka: 16.544602109
Max|grad|: 4.136e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.549936127
KANDIDAT: domet=2 | λ=3
Zbirni validacioni NLL: 16.548995021

IZABRANO: domet=2 | λ=3
Validacioni NLL: 16.548995021

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Domet=2 | λ=3 | prelaza=1385
Zadržana validna tačka: 16.545158215
Max|grad|: 4.027e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza
Domet=2 | λ=3 | prelaza=1732
Zadržana validna tačka: 16.545797127
Max|grad|: 6.238e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Qiskit Hamiltonijan: 39 kubita
Broj Pauli članova: 115
Logičko kolo pripreme: 42 kubita
Gust vektor tog kola nije alociran.
Operacije kola: {'ucry': 39, 'mcx': 39, 'ccx': 39, 'cx': 39}
Odvojeni test: 347 prelaza

loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 07 08 11 23 26 34
Validacioni NLL: 16.548995021
Odvojeni test NLL: 16.548482025
Pogodaka 0–7: 80 141 95 25 4 2 0 0
Modelna verovatnoća NEXT: 7.44758653484e-08
Najmanja energija NEXT: -0.1358273386
Srednja energija U: -0.0005184966
Slobodna energija F: -16.5486180579
Termodinamička entropija S: 16.5480995613
Greška Qiskit log-verovatnoće: 2.119e-10
Greška lokalnih Qiskit amplituda: 6.831e-11
Osetljivost margina: 3.904579743e-04
Max|grad|: 6.238e-07
Gradijentni uslov: dostignut



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 08 11 22 23 26 34 37
Validacioni NLL: 16.548395537
Odvojeni test NLL: 16.547291316
Pogodaka 0–7: 207 376 272 74 13 0 0 0
Modelna verovatnoća NEXT: 7.08148946422e-08
Najmanja energija NEXT: -0.0857942296
Srednja energija U: -0.0005903001
Slobodna energija F: -16.5489907115
Termodinamička entropija S: 16.5484004114
Greška Qiskit log-verovatnoće: 1.631e-10
Greška lokalnih Qiskit amplituda: 9.276e-11
Osetljivost margina: 1.534182216e-04
Max|grad|: 2.900e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv
NEXT: 08 16 21 22 23 34 35
Validacioni NLL: 16.548517857
Odvojeni test NLL: 16.547774437
Pogodaka 0–7: 129 242 163 52 10 0 0 0
Modelna verovatnoća NEXT: 7.19158969537e-08
Najmanja energija NEXT: -0.1013913701
Srednja energija U: -0.0008585716
Slobodna energija F: -16.5491598687
Termodinamička entropija S: 16.5483012971
Greška Qiskit log-verovatnoće: 3.553e-15
Greška lokalnih Qiskit amplituda: 3.331e-16
Osetljivost margina: 2.064194605e-04
Max|grad|: 3.421e-09
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 07 08 11 23 26 34
Validacioni NLL: 16.548995021
Odvojeni test NLL: 16.548482025
Pogodaka 0–7: 80 141 95 25 4 2 0 0
Modelna verovatnoća NEXT: 7.44758653484e-08
Najmanja energija NEXT: -0.1358273386
Srednja energija U: -0.0005184966
Slobodna energija F: -16.5486180579
Termodinamička entropija S: 16.5480995613
Greška Qiskit log-verovatnoće: 2.119e-10
Greška lokalnih Qiskit amplituda: 6.831e-11
Osetljivost margina: 3.904579743e-04
Max|grad|: 6.238e-07
Gradijentni uslov: dostignut

Obrada je završena.
NEXT je stanje najmanje energije naučenog Hamiltonijana.
Njegova Born verovatnoća proverena je Qiskit kontrakcijom.
Nepotvrđena konvergencija ostaje označena.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
Gibbsova raspodela sa lokalnim poljima i parnim interakcijama dometa 1 ili 2. 
Hamiltonijan predstavljen u Qiskitu, a kolo priprema amplitude proporcionalne exp(-E/2), uz tačno sedam zauzetih stanja. 
Hamiltonijan je dijagonalan, raspodela; kvantni deo je priprema i Bornovo očitavanje stanja. 
Ovo ne predstavlja nekomutativni kvantni model niti dokaz kvantne prednosti.

Gibbsova energija, parne interakcije dometa 1 ili 2 i termodinamičke veličine. 
Qiskit kolo priprema koherentno kodiranje Gibbsove raspodele, sa amplitudama proporcionalnim exp(-E/2).
Termodinamička entropija pripada dijagonalnom Gibbsovom stanju, a ne čistom stanju koje priprema za očitavanje.

Matematički sam proverio uslovnu pripremu na 70 malih putanja.
Nezavisna matematička provera dala je najveće odstupanje log-verovatnoće približno \(9·10^{-16}\)
"""
