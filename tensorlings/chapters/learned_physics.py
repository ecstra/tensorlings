"""108_learned_physics, a capstone of Your own modules: a simulator that learns how particles on springs move, built
one design step at a time and measured against simple baselines over long rollouts."""

from __future__ import annotations

import copy
import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from .design_method import DIAGNOSE_SOLUTION, GAP_PART, HEALTH_SOLUTION, ROTATE_PART, pinned_threads
from .linear_layers import expect_module

LEARNED_PHYSICS = Chapter(
    folder="108_learned_physics",
    title="Project: learned physics",
    readme=source_file(
        """
        # Project: learned physics

        This project is a capstone for the Your own modules part of the course.
        First you write a classic simulator for particles joined by springs.
        Then you teach a network to simulate the same particles, one design
        step at a time, and measure each step. The finished simulator handles
        systems four times bigger than any it trained on. It keeps their energy
        nearly constant and stays close to the true motion over long rollouts.

        ## Further reading

        - [Battaglia and colleagues, Interaction Networks](https://arxiv.org/abs/1612.00222)
        - [Sanchez-Gonzalez and colleagues, learning to simulate complex physics with graph networks](https://arxiv.org/abs/2002.09405)
        - [Satorras, Hoogeboom and Welling, E(n) equivariant graph neural networks](https://arxiv.org/abs/2102.09844)
        - [Wikipedia, the semi-implicit Euler method](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "copy": copy,
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    exec(code, namespace)

    return namespace


# ---------------------------------------------------------------------------
# The world, shared by every exercise
# ---------------------------------------------------------------------------

SETTINGS_PART = '''
REST = 1.0
DT = 0.05
'''

SYSTEMS_PART = '''
# Particles of mass 1 in a box from -1.5 to 1.5, joined by springs of rest length REST. springs[s, i, j] is the
# stiffness of the spring between particles i and j in system s. It is the same in both directions, and 0 where there
# is no spring. A random path through every particle keeps each system connected. The velocities are shifted so that
# no system drifts away as a whole.
def random_systems(
    count: int,
    particles: int,
    generator: torch.Generator,
    low: float = 0.5,
    high: float = 1.5,
) -> tuple[Tensor, Tensor, Tensor]:
    x = (torch.rand(count, particles, 2, generator=generator) * 2 - 1) * 1.5
    v = torch.randn(count, particles, 2, generator=generator) * 0.3
    stiffness = torch.rand(count, particles, particles, generator=generator) * (high - low) + low
    joined = torch.rand(count, particles, particles, generator=generator) < 0.6

    for system in range(count):
        order = torch.randperm(particles, generator=generator)
        joined[system, order[:-1], order[1:]] = True

    joined = (joined | joined.transpose(1, 2)) & ~torch.eye(particles, dtype=torch.bool)
    stiffness = torch.minimum(stiffness, stiffness.transpose(1, 2))

    return x, v - v.mean(dim=1, keepdim=True), torch.where(joined, stiffness, torch.zeros(()))
'''

PHYSICS_SOLUTION = '''
def spring_accelerations(
    x: Tensor,
    springs: Tensor,
) -> Tensor:
    offsets = x.unsqueeze(1) - x.unsqueeze(2)
    lengths = offsets.norm(dim=3, keepdim=True).clamp_min(1e-9)
    pulls = springs.unsqueeze(3) * (lengths - REST) * offsets / lengths

    return pulls.sum(dim=2)


def euler_step(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    springs: Tensor,
) -> tuple[Tensor, Tensor]:
    v = v + accelerate(x, springs) * DT

    return x + v * DT, v


def rollout(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    springs: Tensor,
    steps: int,
) -> Tensor:
    path = [x]

    with torch.no_grad():
        for _ in range(steps):
            x, v = euler_step(accelerate, x, v, springs)
            path.append(x)

    return torch.stack(path, dim=1)


def energy(
    x: Tensor,
    v: Tensor,
    springs: Tensor,
) -> Tensor:
    kinetic = 0.5 * (v ** 2).sum(dim=(1, 2))
    lengths = (x.unsqueeze(1) - x.unsqueeze(2)).norm(dim=3)
    potential = 0.25 * (springs * (lengths - REST) ** 2).sum(dim=(1, 2))

    return kinetic + potential
'''

WORLD_PART = SETTINGS_PART + SYSTEMS_PART + PHYSICS_SOLUTION
WORLD_REFERENCE = namespace_of(WORLD_PART)

# ---------------------------------------------------------------------------
# physics2's pieces
# ---------------------------------------------------------------------------

FLAT_PART = '''
# An MLP that reads every coordinate and every stiffness of a system of 5 particles at once, and outputs every
# acceleration.
class FlatMLP(nn.Module):
    def __init__(
        self,
        particles: int = 5,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(particles * 2 + particles * particles, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, particles * 2),
        )

    def forward(
        self,
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        return self.net(torch.cat([x.flatten(1), springs.flatten(1)], dim=1)).view(x.shape)
'''

POOL_PART = '''
# Training states. The true simulator runs random systems, and every frame becomes a state, with its stiffnesses and
# true accelerations.
def state_pool(
    systems: int,
    particles: int,
    frames: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    x, v, springs = random_systems(systems, particles, generator)
    xs = rollout(spring_accelerations, x, v, springs, frames).flatten(0, 1)
    ss = springs.repeat_interleave(frames + 1, dim=0)

    return xs, ss, spring_accelerations(xs, ss)
'''

# design2's diagnosis, without its recall baseline.
DIAGNOSE_PART = DIAGNOSE_SOLUTION[DIAGNOSE_SOLUTION.index("def diagnose") :]

BASELINE_SOLUTION = '''
def coasting(
    x: Tensor,
    springs: Tensor,
) -> Tensor:
    return torch.zeros_like(x)


def train_accelerations(
    model: nn.Module,
    pool: tuple[Tensor, Tensor, Tensor],
    steps: int,
    generator: torch.Generator,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    xs, ss, targets = pool

    for _ in range(steps):
        rows = torch.randint(0, len(xs), (64,), generator=generator)
        loss = F.mse_loss(model(xs[rows], ss[rows]), targets[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def share_close(
    model: nn.Module,
    pool: tuple[Tensor, Tensor, Tensor],
    tolerance: float,
) -> float:
    xs, ss, targets = pool

    with torch.no_grad():
        misses = (model(xs, ss) - targets).norm(dim=2)

    return (misses <= tolerance).float().mean().item()


def rollout_error(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    springs: Tensor,
    truth: Tensor,
) -> Tensor:
    path = rollout(accelerate, x, v, springs, truth.shape[1] - 1)

    return (path - truth).norm(dim=3).mean(dim=(0, 2))
'''

PHYSICS2_NAMES = '''
pool = state_pool(200, 5, 100, torch.Generator().manual_seed(1))
held = state_pool(50, 5, 100, torch.Generator().manual_seed(2))
torch.manual_seed(0)
flat = train_accelerations(FlatMLP(), pool, 1000, torch.Generator().manual_seed(0))
fit = share_close(flat, pool, 0.1)
held_fit = share_close(flat, held, 0.1)
verdict = diagnose(fit, held_fit, 0.9, 0.1)
'''

BASELINE_REFERENCE = namespace_of(WORLD_PART + FLAT_PART + POOL_PART + DIAGNOSE_PART + BASELINE_SOLUTION)

# ---------------------------------------------------------------------------
# physics3's pieces
# ---------------------------------------------------------------------------

RELATIVE_INIT = '''
class RelativeForces(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
'''

RELATIVE_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        offsets = x.unsqueeze(1) - x.unsqueeze(2)
        lengths = offsets.norm(dim=3, keepdim=True)
        pairs = torch.cat([offsets, lengths, springs.unsqueeze(3)], dim=3)
        joined = (springs > 0).unsqueeze(3)

        return (self.message(pairs) * joined).sum(dim=2)
'''

CENTRAL_INIT = '''
class CentralForces(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.law = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
'''

CENTRAL_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        offsets = x.unsqueeze(1) - x.unsqueeze(2)
        lengths = offsets.norm(dim=3, keepdim=True).clamp_min(1e-9)
        pulls = springs.unsqueeze(3) * self.law(lengths) * offsets / lengths

        return pulls.sum(dim=2)
'''

MODELS_SOLUTION = RELATIVE_INIT + RELATIVE_FORWARD + "\n" + CENTRAL_INIT + CENTRAL_FORWARD

GAPS_SOLUTION = '''
def moved_gap(
    model: nn.Module,
    x: Tensor,
    springs: Tensor,
    shift: Tensor,
) -> float:
    with torch.no_grad():
        return (model(x + shift, springs) - model(x, springs)).abs().max().item()


def turned_gap(
    model: nn.Module,
    x: Tensor,
    springs: Tensor,
    angle: float,
) -> float:
    with torch.no_grad():
        return equivariance_gap(lambda points: model(points, springs), lambda points: rotate(points, angle), x)


def shuffled_gap(
    model: nn.Module,
    x: Tensor,
    springs: Tensor,
    order: Tensor,
) -> float:
    with torch.no_grad():
        reordered = model(x[:, order], springs[:, order][:, :, order])

        return (reordered - model(x, springs)[:, order]).abs().max().item()
'''

# design4's check that every weight learns, without its identity gap.
SILENT_PART = HEALTH_SOLUTION[HEALTH_SOLUTION.index("def silent_after_one_step") :]

PHYSICS3_NAMES = '''
x, v, springs = random_systems(8, 5, torch.Generator().manual_seed(3))
shift = torch.tensor([4.0, -3.0])
order = torch.tensor([3, 0, 4, 1, 2])
gaps = {}

for name, make in (("flat", FlatMLP), ("relative", RelativeForces), ("central", CentralForces)):
    torch.manual_seed(0)
    model = make()
    gaps[name] = (
        moved_gap(model, x, springs, shift),
        turned_gap(model, x, springs, 1.0),
        shuffled_gap(model, x, springs, order),
    )

torch.manual_seed(0)
silent = silent_after_one_step(
    CentralForces(),
    lambda model: F.mse_loss(model(x, springs), spring_accelerations(x, springs)),
    0.01,
)
'''

STRUCTURE_REFERENCE = namespace_of(
    WORLD_PART + FLAT_PART + POOL_PART + DIAGNOSE_PART + BASELINE_SOLUTION + GAP_PART + ROTATE_PART + MODELS_SOLUTION + GAPS_SOLUTION + SILENT_PART,
)

# ---------------------------------------------------------------------------
# physics4's pieces
# ---------------------------------------------------------------------------

DRIFT_SOLUTION = '''
def energy_drift(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    springs: Tensor,
    steps: int,
) -> float:
    start = energy(x, v, springs)

    with torch.no_grad():
        for _ in range(steps):
            x, v = euler_step(accelerate, x, v, springs)

    return ((energy(x, v, springs) - start) / start).abs().median().item()


def case_errors(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    cases: dict[str, tuple[Tensor, Tensor, Tensor]],
    steps: int,
) -> dict[str, float]:
    errors = {}

    for name, (x, v, springs) in cases.items():
        truth = rollout(spring_accelerations, x, v, springs, steps)
        errors[name] = rollout_error(accelerate, x, v, springs, truth)[-1].item()

    return errors
'''

PHYSICS4_NAMES = '''
pool = state_pool(200, 5, 100, torch.Generator().manual_seed(1))
cases = {
    "5 particles": random_systems(50, 5, torch.Generator().manual_seed(3)),
    "20 particles": random_systems(50, 20, torch.Generator().manual_seed(5)),
    "stiffer springs": random_systems(50, 5, torch.Generator().manual_seed(6), 1.5, 3.0),
}
torch.manual_seed(0)
relative = train_accelerations(RelativeForces(), pool, 1500, torch.Generator().manual_seed(0))
torch.manual_seed(0)
central = train_accelerations(CentralForces(), pool, 1500, torch.Generator().manual_seed(0))
table = {
    "coasting": case_errors(coasting, cases, 100),
    "relative": case_errors(relative, cases, 100),
    "central": case_errors(central, cases, 100),
}
drifts = {name: energy_drift(model, *cases["5 particles"], 1000) for name, model in (("relative", relative), ("central", central))}
'''

ROLLOUT_REFERENCE = namespace_of(
    WORLD_PART + FLAT_PART + POOL_PART + DIAGNOSE_PART + BASELINE_SOLUTION + GAP_PART + ROTATE_PART + MODELS_SOLUTION + GAPS_SOLUTION + SILENT_PART + DRIFT_SOLUTION,
)


# ---------------------------------------------------------------------------
# Shared checking pieces
# ---------------------------------------------------------------------------

# Three particles, the first two joined by a spring of stiffness 1 stretched to length 2, the last two by one of
# stiffness 2 squeezed, and the first and last not joined.
TRIANGLE_X = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 0.5]]])
TRIANGLE_V = torch.tensor([[[0.1, 0.0], [0.0, -0.2], [-0.1, 0.2]]])
TRIANGLE_SPRINGS = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]])


def systems(
    count: int,
    particles: int,
    seed: int,
    low: float = 0.5,
    high: float = 1.5,
) -> tuple[Described, Described, Described]:
    x, v, springs = WORLD_REFERENCE["random_systems"](count, particles, torch.Generator().manual_seed(seed), low, high)
    text = f"{count} random systems of {particles}"

    return (
        Described(x, f"the positions of {text}"),
        Described(v, f"the velocities of {text}"),
        Described(springs, f"the springs of {text}"),
    )


def constant_pull(
    x: Tensor,
    springs: Tensor,
) -> Tensor:
    return torch.full_like(x, 0.5) - 0.1 * x


CONSTANT_PULL = Described(constant_pull, "a made up pull of 0.5 - 0.1 x on every coordinate")
TRUE_PULL = Described(WORLD_REFERENCE["spring_accelerations"], "the true spring_accelerations")


def grad_recorder(seen: list[bool]) -> Callable[[Tensor, Tensor], Tensor]:
    """A pull that notes whether autograd was recording each time it ran."""

    def pull(
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        seen.append(torch.is_grad_enabled())

        return WORLD_REFERENCE["spring_accelerations"](x, springs)

    return pull


def expect_scaled(
    call: str,
    actual: object,
    expected: float,
) -> None:
    """A plain number close to the expected one, within a millionth of its size."""
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-6 * max(1.0, abs(expected)),
    )


def expect_number_dict(
    call: str,
    actual: object,
    expected: dict[str, float],
    tolerance: float = 1e-5,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, dict):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a dict from each case's name to a number.")

    if sorted(actual) != sorted(expected):
        raise Mismatch(f"{call} has the keys {sorted(actual)}, but it should have {sorted(expected)}.")

    for key, wanted in expected.items():
        expect_close(
            call=f"{call}[{key!r}]",
            actual=actual[key],
            expected=wanted,
            tolerance=tolerance * max(1.0, abs(wanted)),
        )


def seeded_model(
    make: type,
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)

    return make()


# ---------------------------------------------------------------------------
# physics1
# ---------------------------------------------------------------------------

PHYSICS1_LESSON = prose(
    """
    This project simulates particles of mass 1 in a flat box, with some pairs joined by springs.
    physics1 builds the true simulator in the classic way, from the physics equations. physics2 to
    physics4 then teach a network to simulate the same world. They follow the method of design1 to
    design5, one step at a time.

    A spring of stiffness k and rest length 1 pulls its ends together when it is stretched, and
    pushes them apart when it is squeezed. Hooke's law says the force is k times the stretch, which
    is the length minus 1. The force points along the line between the ends.

    ```text
    pull on i from j = k_ij * (|x_j - x_i| - 1) * (x_j - x_i) / |x_j - x_i|
    ```

    With mass 1, a particle's acceleration is the sum of the pulls on it. Broadcasting computes
    every offset `x_j - x_i` at once, as equivariance2's squared_distances did. `springs` holds 0
    where no spring joins a pair, so a missing spring adds no pull.

    A simulator moves time forward in steps of `DT`, here 0.05 seconds. The semi-implicit Euler
    step updates the velocity first, then moves each particle with the new velocity.

    ```text
    v = v + a(x) * DT
    x = x + v * DT
    ```

    Plain Euler moves each particle with the old velocity instead. That small change has a large
    effect. With no friction, a system's energy should stay constant. The table shows how much the
    energy changed after 1,000 steps, as the median over 100 systems of 5 particles.

    | Step | Energy change after 1,000 steps |
    |---|---|
    | semi-implicit Euler | 1.7% |
    | plain Euler | about 1,281,000%, flying apart |

    Wikipedia's article on the method says the semi-implicit step is symplectic, which means it
    keeps the energy nearly constant. Plain Euler adds a little energy at every step. DeepMind's
    learned simulator, GNS, has its network predict accelerations. It then moves the particles with
    this same semi-implicit Euler step. physics3's networks do the same, so how well they simulate
    depends only on how accurate their accelerations are.
    """,
)

PHYSICS1_SOURCE = source_file(
    '''
    # physics1.py

    from collections.abc import Callable

    import torch
    from torch import Tensor

    points = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    gaps = points.unsqueeze(0) - points.unsqueeze(1)   # gaps[i, j] is points[j] - points[i], shape (2, 2, 2)
    length = gaps[0, 1].norm()                          # tensor(5.)
    along = gaps[0, 1] / length                         # tensor([0.6000, 0.8000]), length 1, pointing the same way
    both = torch.stack([points, points], dim=1).shape   # torch.Size([2, 2, 2]), a new dim 1

''' + indented(SETTINGS_PART) + '''


''' + indented(SYSTEMS_PART) + '''


    # TODO: Return every particle's acceleration, of shape (systems, particles,
    #   2). x holds positions of shape (systems, particles, 2), and springs
    #   holds the stiffnesses, of shape (systems, particles, particles). The
    #   spring between i and j pulls i with the force
    #   springs[s, i, j] * (length - REST) * offset / length, where offset is
    #   x[s, j] - x[s, i] and length is its norm. Mass is 1, so i's
    #   acceleration is the sum of the pulls on it. Keep each length above
    #   1e-9 with .clamp_min, since a particle's offset to itself has length 0.
    #   Example: for x = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]) and
    #   springs = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), it returns
    #   tensor([[[1., 0.], [-1., 0.]]]).
    def spring_accelerations(
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the positions and velocities after one semi-implicit Euler
    #   step of DT. First add accelerate(x, springs) * DT to the velocity. Then
    #   add the new velocity times DT to the positions. Return (x, v).
    def euler_step(
        accelerate: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        v: Tensor,
        springs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Run `steps` euler_steps with accelerate, starting from x and v.
    #   Return the positions at every step, starting with x, stacked on a new
    #   dim 1, so the shape is (systems, steps + 1, particles, 2). Work inside
    #   torch.no_grad().
    def rollout(
        accelerate: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        v: Tensor,
        springs: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return each system's total energy, of shape (systems,). The kinetic
    #   energy is 0.5 times the sum of every velocity component squared. A
    #   spring stores 0.5 * stiffness * ((length - REST) ** 2) of potential
    #   energy. springs holds each spring twice, at [i, j] and at [j, i], so
    #   the potential energy is 0.25 times that sum over all pairs i, j.
    #   Example: for the x and springs of spring_accelerations' example, with
    #   v = torch.zeros(1, 2, 2), it returns tensor([0.5000]).
    def energy(
        x: Tensor,
        v: Tensor,
        springs: Tensor,
    ) -> Tensor:
        ...
    ''',
)

PHYSICS1_SOLUTION = source_file(
    '''
    # physics1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    from torch import Tensor

    points = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    gaps = points.unsqueeze(0) - points.unsqueeze(1)
    length = gaps[0, 1].norm()
    along = gaps[0, 1] / length
    both = torch.stack([points, points], dim=1).shape

''' + indented(SETTINGS_PART) + '''


''' + indented(SYSTEMS_PART) + '''


''' + indented(PHYSICS_SOLUTION) + '''
    ''',
)


def check_rollout_records_nothing(found: object) -> None:
    seen: list[bool] = []
    x, v, springs = WORLD_REFERENCE["random_systems"](2, 4, torch.Generator().manual_seed(1083))
    found(grad_recorder(seen), x, v, springs, 3)

    if any(seen):
        raise Mismatch("rollout ran accelerate while recording for autograd. Wrap its steps in torch.no_grad().")

    if not seen:
        raise Mismatch("rollout never called accelerate. Each step should go through euler_step with it.")


def check_physics1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="spring_accelerations",
        description="sums each particle's spring pulls",
        inputs=(
            (TRIANGLE_X, TRIANGLE_SPRINGS),
            systems(4, 5, 1081)[0::2],
            systems(2, 7, 1082, 1.5, 3.0)[0::2],
        ),
        reference=WORLD_REFERENCE["spring_accelerations"],
        expect=expect_tensor,
    )
    triangle = (TRIANGLE_X, TRIANGLE_V, TRIANGLE_SPRINGS)
    check_on_inputs(
        checker=checker,
        function="euler_step",
        description="updates the velocity first, then moves the positions with the new velocity",
        inputs=(
            (CONSTANT_PULL, *triangle),
            (TRUE_PULL, *systems(3, 5, 1084)),
        ),
        reference=WORLD_REFERENCE["euler_step"],
        expect=expect_parts("the positions", "the velocities"),
    )
    check_on_inputs(
        checker=checker,
        function="rollout",
        description="stacks the positions of every step, the start first",
        inputs=(
            (CONSTANT_PULL, *triangle, 3),
            (TRUE_PULL, *systems(2, 6, 1085), 5),
        ),
        reference=WORLD_REFERENCE["rollout"],
        expect=expect_tensor,
    )
    checker.test(
        function="rollout",
        description="records nothing for autograd",
        run=check_rollout_records_nothing,
    )
    check_on_inputs(
        checker=checker,
        function="energy",
        description="adds the kinetic energy and each spring's potential energy, counted once",
        inputs=(
            triangle,
            systems(4, 5, 1086),
        ),
        reference=WORLD_REFERENCE["energy"],
        expect=expect_tensor,
    )


PHYSICS1 = Exercise(
    name="physics1",
    chapter=LEARNED_PHYSICS,
    title="Project: springs and a true simulator",
    lesson=PHYSICS1_LESSON,
    source=PHYSICS1_SOURCE,
    solution=PHYSICS1_SOLUTION,
    hints={
        "spring_accelerations": "Make the offsets by broadcasting, as the gaps example does, with the system dimension "
        "in front. Then offsets[s, i, j] is x[s, j] - x[s, i]. Take their lengths with .norm over the last dim with "
        "keepdim=True, and clamp them. Multiply the stretch, the unit offsets and springs with a dim added at the end. "
        "Then sum over j, which is dim 2.",
        "euler_step": "This takes two lines. The new v comes from the old v and accelerate of the old x. The new x comes "
        "from the old x and the new v. Return both.",
        "rollout": "Keep a list that starts as [x]. Inside torch.no_grad(), call euler_step steps times and append each "
        "new x. Then torch.stack the list on dim 1.",
        "energy": "The kinetic energy is 0.5 times v squared, summed over dims 1 and 2. For the springs, compute every "
        "pairwise length as spring_accelerations did. Then take 0.25 times springs times the squared stretch, summed "
        "over dims 1 and 2.",
    },
    introduces=(),
    check=check_physics1,
)


# ---------------------------------------------------------------------------
# physics2
# ---------------------------------------------------------------------------

PHYSICS2_LESSON = prose(
    """
    Before building a clever model, set up baselines and a way to diagnose their failures, as
    design2 did.

    - Coasting gives accelerations of zero, so every particle keeps its starting velocity.
    - `FlatMLP` reads all 10 coordinates and 25 stiffnesses of a system of 5 particles at once, and outputs all 10 accelerations. It has 22,410 weights.

    Training samples from a pool of states. `state_pool` runs random systems with the true
    simulator, and keeps every frame together with its true accelerations. The MLP in the table
    trained on 500 systems of 101 frames each, for 3,000 steps with batches of 64 states.

    Two scores measure the result. A typical acceleration here has a length of about 1.5.
    `share_close` gives the fraction of particles whose predicted acceleration is within 0.1 of the
    true one. `rollout_error` runs a whole simulation with the model. It measures the average
    distance between each particle and where the true simulator puts it. For scale, the box is 3
    wide.

    | Model | Close, training | Close, held out | Rollout error after 50 steps | After 100 steps |
    |---|---|---|---|---|
    | coasting | 2.0% | 1.9% | 1.65 | 2.13 |
    | flat MLP | 6.9% to 7.7% | 4.4% to 5.2% | 0.57 to 0.60 | 0.75 to 0.77 |

    The ranges are over 3 seeds, and the rollouts run 100 new systems for up to 5 seconds.

    design2's diagnosis says the flat MLP failed to fit, because it cannot match the accelerations
    even on its own training states. More data will not fix that. With only 4 training systems, it
    does learn its training states, 96% to 98% close. But then it gets under 1% of new states close,
    so it fails to generalize.

    design2 named two cures for a failure to fit. One is a layer that can express the skill, and the
    other is a fix to the training. Longer training and a wider MLP try the second cure, over 3
    seeds.

    | Flat MLP | Close, training | Close, held out |
    |---|---|---|
    | 3,000 steps, as above | 6.9% to 7.7% | 4.4% to 5.2% |
    | 20,000 steps | 17% to 18% | 6.4% to 7.0% |
    | 20,000 steps, 512 wide, 286,218 weights | 34% to 36% | 8.7% to 10.5% |

    Even the wide MLP fits only a third of its own training states, far from 90%. So the structure
    has to change. physics3 builds a structure from what the physics already tells us. The lines at
    the end of the file run a shorter version, 200 systems for 1,000 steps, and reach the same
    diagnosis.
    """,
)

PHYSICS2_GIVEN = SETTINGS_PART + SYSTEMS_PART + PHYSICS_SOLUTION

PHYSICS2_SOURCE = source_file(
    '''
    # physics2.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    misses = torch.tensor([[[0.03, 0.04], [0.3, 0.4]]]).norm(dim=2)    # tensor([[0.0500, 0.5000]])
    close = (misses <= 0.1).float().mean()                            # tensor(0.5000)


    # physics1's world and true simulator.
''' + indented(PHYSICS2_GIVEN) + '''


''' + indented(FLAT_PART) + '''


''' + indented(POOL_PART) + '''


    # design2's diagnosis.
''' + indented(DIAGNOSE_PART) + '''


    # TODO: Return accelerations of zero, of x's shape, so a rollout keeps every
    #   particle at its starting velocity.
    def coasting(
        x: Tensor,
        springs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train model with Adam at learning rate 0.003 for `steps` steps.
    #   pool holds (positions, springs, true accelerations). Each step draws 64
    #   random row indices with torch.randint and generator. It minimizes the
    #   mean squared error between model's accelerations for those rows and
    #   the true ones. Return model.
    def train_accelerations(
        model: nn.Module,
        pool: tuple[Tensor, Tensor, Tensor],
        steps: int,
        generator: torch.Generator,
    ) -> nn.Module:
        ...


    # TODO: Return the fraction of particles in pool whose predicted
    #   acceleration is within tolerance of the true one, as a Python float.
    #   The miss for a particle is the norm of the difference over the last
    #   dim. Work inside torch.no_grad().
    def share_close(
        model: nn.Module,
        pool: tuple[Tensor, Tensor, Tensor],
        tolerance: float,
    ) -> float:
        ...


    # TODO: Roll accelerate out from x and v for truth.shape[1] - 1 steps, so
    #   the path has as many frames as truth. At every frame, compute the
    #   distance between each particle and its position in truth. Return the
    #   mean over the systems and the particles, of shape (truth.shape[1],).
    def rollout_error(
        accelerate: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        v: Tensor,
        springs: Tensor,
        truth: Tensor,
    ) -> Tensor:
        ...


    # The flat MLP, trained and diagnosed. These lines run as the file loads.
''' + indented(PHYSICS2_NAMES) + '''
    ''',
)

PHYSICS2_SOLUTION = source_file(
    '''
    # physics2.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    misses = torch.tensor([[[0.03, 0.04], [0.3, 0.4]]]).norm(dim=2)
    close = (misses <= 0.1).float().mean()


''' + indented(PHYSICS2_GIVEN) + '''


''' + indented(FLAT_PART) + '''


''' + indented(POOL_PART) + '''


''' + indented(DIAGNOSE_PART) + '''


''' + indented(BASELINE_SOLUTION) + '''


''' + indented(PHYSICS2_NAMES) + '''
    ''',
)

_PHYSICS2_EXPECTED: dict[str, object] = {}


def physics2_expected() -> dict[str, object]:
    """The names physics2's last lines set, run with the reference functions, once per process."""
    if not _PHYSICS2_EXPECTED:
        namespace = dict(BASELINE_REFERENCE)
        exec(PHYSICS2_NAMES, namespace)
        _PHYSICS2_EXPECTED.update({name: namespace[name] for name in ("fit", "held_fit", "verdict")})

    return _PHYSICS2_EXPECTED


def small_pool() -> tuple[Tensor, Tensor, Tensor]:
    return BASELINE_REFERENCE["state_pool"](6, 5, 20, torch.Generator().manual_seed(1090))


def check_train_accelerations(found: object) -> None:
    results = []

    for train in (found, BASELINE_REFERENCE["train_accelerations"]):
        model = seeded_model(BASELINE_REFERENCE["FlatMLP"])
        returned = train(model, small_pool(), 5, torch.Generator().manual_seed(1091))

        if returned is not model:
            raise Mismatch("train_accelerations should return the model it trained.")

        results.append(model.net[0].weight.detach().clone())

    expect_tensor(
        call="the flat MLP's first weights after train_accelerations for 5 steps",
        actual=results[0],
        expected=results[1],
    )


def check_share_close_records_nothing(found: object) -> None:
    seen: list[bool] = []
    model = seeded_model(BASELINE_REFERENCE["FlatMLP"])
    model.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))
    found(model, small_pool(), 0.1)

    if any(seen):
        raise Mismatch("share_close ran the model while recording for autograd. Wrap it in torch.no_grad().")


def check_physics2_name(
    name: str,
) -> Callable[[object], None]:
    def run(value: object) -> None:
        wanted = physics2_expected()[name]

        if isinstance(wanted, str):
            expect_value(
                call=name,
                actual=value,
                expected=wanted,
            )
            return

        expect_close(
            call=name,
            actual=value,
            expected=wanted,
            tolerance=1e-6,
        )

    return run


def check_physics2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="coasting",
        description="gives accelerations of zero",
        inputs=(
            (TRIANGLE_X, TRIANGLE_SPRINGS),
            systems(3, 6, 1092)[0::2],
        ),
        reference=BASELINE_REFERENCE["coasting"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_accelerations",
        description="trains with Adam on the mean squared error of the accelerations",
        run=check_train_accelerations,
    )
    pool = small_pool()
    check_on_inputs(
        checker=checker,
        function="share_close",
        description="gives the fraction of accelerations within tolerance",
        inputs=(
            (Described(seeded_model(BASELINE_REFERENCE["FlatMLP"]), "an untrained FlatMLP"), Described(pool, "states of 6 systems"), 0.5),
            (Described(seeded_model(BASELINE_REFERENCE["FlatMLP"], 1), "another untrained FlatMLP"), Described(pool, "states of 6 systems"), 1.5),
        ),
        reference=BASELINE_REFERENCE["share_close"],
        expect=expect_close,
    )
    checker.test(
        function="share_close",
        description="records nothing for autograd",
        run=check_share_close_records_nothing,
    )
    x, v, springs = WORLD_REFERENCE["random_systems"](3, 5, torch.Generator().manual_seed(1093))
    truth = WORLD_REFERENCE["rollout"](WORLD_REFERENCE["spring_accelerations"], x, v, springs, 6)
    check_on_inputs(
        checker=checker,
        function="rollout_error",
        description="averages each particle's distance from the truth at every step",
        inputs=(
            (Described(BASELINE_REFERENCE["coasting"], "coasting"), x, v, springs, Described(truth, "the true rollout of 6 steps")),
            (CONSTANT_PULL, x, v, springs, Described(truth[:, :4], "the true rollout of 3 steps")),
        ),
        reference=BASELINE_REFERENCE["rollout_error"],
        expect=expect_tensor,
    )

    for name, description in (
        ("fit", "is the flat MLP's fraction close on its own training states"),
        ("held_fit", "is its fraction close on held out states"),
        ("verdict", "is design2's diagnosis of the flat MLP"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_physics2_name(name),
        )


PHYSICS2 = Exercise(
    name="physics2",
    chapter=LEARNED_PHYSICS,
    title="Project: baselines and a diagnosis",
    lesson=PHYSICS2_LESSON,
    source=PHYSICS2_SOURCE,
    solution=PHYSICS2_SOLUTION,
    hints={
        "coasting": "Return a tensor of zeros with x's shape, which torch.zeros_like makes.",
        "train_accelerations": "Unpack the pool into xs, ss and targets. Each step, draw 64 row numbers below "
        "len(xs) with torch.randint and the generator. Then take F.mse_loss of model on those rows against their "
        "targets, and do zero_grad, backward and step.",
        "share_close": "Inside torch.no_grad(), take model of the positions and springs minus the targets, and its "
        ".norm over dim 2. Compare with tolerance, convert to float, take the mean and call .item().",
        "rollout_error": "Run physics1's rollout for truth.shape[1] - 1 steps and subtract truth. Take the .norm over "
        "the last dim, then the mean over dims 0 and 2.",
        "fit": "This name comes from the lines at the end of the file. Finish share_close first.",
        "held_fit": "This name comes from the lines at the end of the file. Finish share_close first.",
        "verdict": "This name comes from the lines at the end of the file. Finish the functions above it first.",
    },
    introduces=(),
    check=check_physics2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# physics3
# ---------------------------------------------------------------------------

PHYSICS3_LESSON = prose(
    """
    The physics tells us three things about the forces between particles. physics3 builds each one
    into the module and measures how much it helps.

    - Every pair follows the same law. So one shared network reads each joined pair, and each particle sums the messages it gets, as in message1.
    - Only offsets matter. So the pair network reads x[j] - x[i], as GNS does. Its authors found that offsets work better than absolute positions.
    - A spring pulls along the line between its ends. So the network outputs only the pull's size, as equivariance2's EGNN moved points along differences.

    A shared pair network works for any number of particles, in any order. `RelativeForces` uses the
    first two facts. Its network turns each pair's offset, length and stiffness into a message, a
    vector of 2. `CentralForces` uses all three facts, plus the form of Hooke's law. A small network
    learns the size of the pull as a function of the length, and the stiffness multiplies it. The
    tables also include a pair network that reads both particles' absolute positions instead of
    their offset.

    Three functions test each property directly, as design5 did. `moved_gap` shifts every particle,
    `turned_gap` rotates the system, and `shuffled_gap` reorders the particles. The table shows the
    gaps of the untrained modules. A gap near 1e-7 is only float rounding, so the property holds.

    | Module | Weights | Moved | Turned | Shuffled |
    |---|---|---|---|---|
    | flat MLP | 22,410 | 0.43 | 0.26 | 0.27 |
    | pairs on absolute positions | 4,674 | 0.60 | 0.49 | 1e-8 |
    | `RelativeForces` | 4,610 | 2e-7 | 1.5 | 1e-7 |
    | `CentralForces` | 4,353 | 2e-7 | 2e-7 | 6e-8 |

    Each module trains as physics2's flat MLP did, but its learning rate falls along a cosine from
    3e-3 to 0, as schedule1 showed. At a constant learning rate the weights kept jumping around, and
    one run happened to stop after a bad step, at 66%. GNS decays its learning rate too. The table
    shows the fraction of held out accelerations within 0.1, over 5 seeds. It also scores the same
    states moved by (4, -3) and rotated by 1 radian.

    | Module | Held out | Moved | Rotated |
    |---|---|---|---|
    | flat MLP | 4.2% to 5.3% | 1.3% to 1.9% | 4.3% to 5.1% |
    | pairs on absolute positions | 87% to 91% | 0.7% to 10% | 87% to 90% |
    | `RelativeForces` | 89% to 96% | 89% to 96% | 89% to 95% |
    | `CentralForces` | 100% | 100% | 100% |

    On states like the training ones, the two pair modules score about the same, because the
    training states cover every place in the box. When the states move out of the box, absolute
    positions fall to 10% or less, while offsets lose nothing. Rotation costs no module anything
    here, because the training states already come in every orientation. To see what pulling along
    the line adds, train each module only on systems that lie along one line. Then test it on new
    systems on that line, and on the same states rotated 90 degrees.

    | Module | On the line | Rotated 90 degrees |
    |---|---|---|
    | flat MLP | 21% to 24% | 0.8% to 1.3% |
    | pairs on absolute positions | 96% to 97% | under 0.3% |
    | `RelativeForces` | 97% to 98% | 0% to 2.2% |
    | `CentralForces` | 100% | 100% |

    Each fact from the physics pays off. Pair networks fit where the flat MLP failed, and offsets
    keep working after a move. Pulls along the line keep working after a rotation that no training
    state showed. That is because the network gives only each pull's size, and the offset gives its
    direction. `silent` runs design4's check, and every weight of `CentralForces` gets a gradient
    from the first step.
    """,
)

PHYSICS3_GIVEN = PHYSICS2_GIVEN + FLAT_PART

PHYSICS3_SOURCE = source_file(
    '''
    # physics3.py

    import copy
    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    offset = torch.tensor([3.0, 4.0])
    pull = 2.0 * offset / offset.norm()           # tensor([1.2000, 1.6000]), length 2, pointing along offset
    flags = torch.tensor([0.0, 1.5]) > 0           # tensor([False,  True])
    kept = torch.tensor([7.0, 7.0]) * flags        # tensor([0., 7.]), 0 where the flag is False


    # physics1's world and true simulator, and physics2's flat MLP.
''' + indented(PHYSICS3_GIVEN) + '''


''' + indented(GAP_PART) + '''


''' + indented(ROTATE_PART) + '''


    # design4's check that every weight gets a gradient.
''' + indented(SILENT_PART) + '''


    # TODO: Finish forward. Compute the offsets x[s, j] - x[s, i] as physics1
    #   did, and their lengths, keeping the last dim. Each pair's input is its
    #   offset, its length and springs[s, i, j], joined along a last dim of
    #   size 4. self.message turns each input into a message, a vector of 2.
    #   Keep only the messages of joined pairs, where springs is above 0, and
    #   sum them over j. The result has shape (systems, particles, 2).
''' + indented(RELATIVE_INIT) + '''

        def forward(
            self,
            x: Tensor,
            springs: Tensor,
        ) -> Tensor:
            ...


    # TODO: Finish forward. Each spring pulls along its unit offset,
    #   offset / length, as in physics1. The size of the pull is
    #   springs[s, i, j] times self.law(length). Sum the pulls over j. Clamp
    #   the lengths to at least 1e-9.
''' + indented(CENTRAL_INIT) + '''

        def forward(
            self,
            x: Tensor,
            springs: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the largest absolute change in any entry of model's
    #   accelerations when every particle moves by shift, as a Python float.
    #   Work inside torch.no_grad().
    def moved_gap(
        model: nn.Module,
        x: Tensor,
        springs: Tensor,
        shift: Tensor,
    ) -> float:
        ...


    # TODO: Return the equivariance_gap of model under a rotation by angle, as
    #   a Python float. Its f runs model on positions with springs held fixed,
    #   and its g rotates points by angle with rotate. Work inside
    #   torch.no_grad().
    def turned_gap(
        model: nn.Module,
        x: Tensor,
        springs: Tensor,
        angle: float,
    ) -> float:
        ...


    # TODO: Test whether reordering the particles reorders the accelerations
    #   the same way. Run model on the particles reordered by order. Compare
    #   that with model's original accelerations, reordered by order. Return
    #   the largest absolute difference as a Python float. springs must be
    #   reordered along both of its particle dims. Work inside
    #   torch.no_grad().
    def shuffled_gap(
        model: nn.Module,
        x: Tensor,
        springs: Tensor,
        order: Tensor,
    ) -> float:
        ...


    # The gaps of each untrained structure, and the weights that get no gradient. These lines run as the file loads.
''' + indented(PHYSICS3_NAMES) + '''
    ''',
)

PHYSICS3_SOLUTION = source_file(
    '''
    # physics3.py, a solution for comparison

    import copy
    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    offset = torch.tensor([3.0, 4.0])
    pull = 2.0 * offset / offset.norm()
    flags = torch.tensor([0.0, 1.5]) > 0
    kept = torch.tensor([7.0, 7.0]) * flags


''' + indented(PHYSICS3_GIVEN) + '''


''' + indented(GAP_PART) + '''


''' + indented(ROTATE_PART) + '''


''' + indented(SILENT_PART) + '''


''' + indented(MODELS_SOLUTION) + '''


''' + indented(GAPS_SOLUTION) + '''


''' + indented(PHYSICS3_NAMES) + '''
    ''',
)

_PHYSICS3_EXPECTED: dict[str, object] = {}


def physics3_expected() -> dict[str, object]:
    if not _PHYSICS3_EXPECTED:
        namespace = dict(STRUCTURE_REFERENCE)
        exec(PHYSICS3_NAMES, namespace)
        _PHYSICS3_EXPECTED.update({name: namespace[name] for name in ("gaps", "silent")})

    return _PHYSICS3_EXPECTED


def check_forces(
    name: str,
) -> Callable[[type], None]:
    def run(found: type) -> None:
        for count, particles, seed in ((3, 5, 1094), (2, 8, 1095)):
            x, _, springs = WORLD_REFERENCE["random_systems"](count, particles, torch.Generator().manual_seed(seed))
            model = expect_module(f"{name}()", seeded_model(found))

            with torch.no_grad():
                actual = model(x, springs)
                expected = seeded_model(STRUCTURE_REFERENCE[name])(x, springs)

            expect_tensor(
                call=f"{name}() on {count} random systems of {particles}",
                actual=actual,
                expected=expected,
            )

    return run


def check_gaps_name(value: object) -> None:
    expected = physics3_expected()["gaps"]
    reject_missing_return(
        call="gaps",
        actual=value,
    )

    if not isinstance(value, dict) or sorted(value) != sorted(expected):
        raise Mismatch(f"gaps is {value_text(value)}, but it should be a dict with the keys {sorted(expected)}.")

    for name, wanted in expected.items():
        expect_parts("the moved gap", "the turned gap", "the shuffled gap")(
            call=f"gaps[{name!r}]",
            actual=value[name],
            expected=wanted,
        )


def check_silent_name(value: object) -> None:
    expect_value(
        call="silent",
        actual=value,
        expected=physics3_expected()["silent"],
    )


def gap_records_nothing(
    function: str,
    extra: object,
) -> Callable[[object], None]:
    def run(found: object) -> None:
        seen: list[bool] = []
        model = seeded_model(STRUCTURE_REFERENCE["RelativeForces"])
        model.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))
        x, _, springs = WORLD_REFERENCE["random_systems"](2, 5, torch.Generator().manual_seed(1102))
        found(model, x, springs, extra)

        if any(seen):
            raise Mismatch(f"{function} ran the model while recording for autograd. Wrap its work in torch.no_grad().")

    return run


def gap_inputs() -> tuple[tuple[object, ...], ...]:
    relative = Described(seeded_model(STRUCTURE_REFERENCE["RelativeForces"], 2), "an untrained RelativeForces")
    flat = Described(seeded_model(BASELINE_REFERENCE["FlatMLP"], 3), "an untrained FlatMLP")
    x, _, springs = systems(3, 5, 1096)

    return (relative, x, springs), (flat, x, springs)


def check_physics3(checker: Checker) -> None:
    checker.test_method(
        class_name="RelativeForces",
        method="forward",
        description="sums one shared message per joined pair, read from their offset",
        run=check_forces("RelativeForces"),
    )
    checker.test_method(
        class_name="CentralForces",
        method="forward",
        description="sums pulls along each offset, sized by the stiffness and a learned law",
        run=check_forces("CentralForces"),
    )

    for (model, x, springs), (shift, angle, order) in zip(
        gap_inputs(),
        ((torch.tensor([2.0, -1.0]), 0.7, torch.tensor([4, 2, 0, 1, 3])), (torch.tensor([-3.0, 5.0]), 2.0, torch.tensor([1, 0, 3, 4, 2]))),
    ):
        check_on_inputs(
            checker=checker,
            function="moved_gap",
            description="gives the largest change when every particle moves",
            inputs=((model, x, springs, shift),),
            reference=STRUCTURE_REFERENCE["moved_gap"],
            expect=expect_close,
        )
        check_on_inputs(
            checker=checker,
            function="turned_gap",
            description="gives equivariance_gap under a rotation, with the springs held fixed",
            inputs=((model, x, springs, angle),),
            reference=STRUCTURE_REFERENCE["turned_gap"],
            expect=expect_close,
        )
        check_on_inputs(
            checker=checker,
            function="shuffled_gap",
            description="reorders the particles and both dims of springs",
            inputs=((model, x, springs, order),),
            reference=STRUCTURE_REFERENCE["shuffled_gap"],
            expect=expect_close,
        )

    for function, extra in (
        ("moved_gap", torch.tensor([1.0, 2.0])),
        ("turned_gap", 0.5),
        ("shuffled_gap", torch.tensor([1, 2, 3, 4, 0])),
    ):
        checker.test(
            function=function,
            description="records nothing for autograd",
            run=gap_records_nothing(function, extra),
        )

    checker.check_name(
        name="gaps",
        description="holds each structure's three gaps, untrained",
        run=check_gaps_name,
    )
    checker.check_name(
        name="silent",
        description="lists CentralForces' weights that get no gradient after a step",
        run=check_silent_name,
    )


PHYSICS3 = Exercise(
    name="physics3",
    chapter=LEARNED_PHYSICS,
    title="Project: a simulator built from the physics",
    lesson=PHYSICS3_LESSON,
    source=PHYSICS3_SOURCE,
    solution=PHYSICS3_SOLUTION,
    hints={
        "RelativeForces.forward": "Compute offsets and lengths as in physics1, without the clamp. Join the offsets, "
        "lengths and springs.unsqueeze(3) with torch.cat on dim 3, and run self.message on the result. Multiply by "
        "(springs > 0).unsqueeze(3), and sum over dim 2.",
        "CentralForces.forward": "Copy physics1's spring_accelerations, with self.law(lengths) in place of the "
        "stretch lengths - REST.",
        "moved_gap": "Inside torch.no_grad(), take model of x + shift minus model of x, then .abs().max().item().",
        "turned_gap": "Inside torch.no_grad(), call equivariance_gap with f a lambda of points giving "
        "model(points, springs), and g a lambda of points giving rotate(points, angle).",
        "shuffled_gap": "x[:, order] reorders the particles. springs[:, order][:, :, order] reorders both dims. "
        "Compare model of those with model(x, springs)[:, order].",
        "gaps": "This name comes from the lines at the end of the file. Finish the classes and the gaps first.",
        "silent": "This name comes from the lines at the end of the file. Finish CentralForces first.",
    },
    introduces=(),
    check=check_physics3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# physics4
# ---------------------------------------------------------------------------

PHYSICS4_LESSON = prose(
    """
    A simulator feeds its own output back in as input for hundreds of steps, so small errors can
    grow. physics4 rolls out every model and checks two more properties that the physics guarantees.

    - Energy stays constant, apart from the integrator's own change of 1.7% after 1,000 steps, from physics1.
    - A system's total momentum stays at 0, since every pull has an equal and opposite partner.

    Each model trained on systems of 5 particles, as in physics3. The new test cases are systems of
    10 and 20 particles, and springs with stiffness from 1.5 to 3, while training never went above
    1.5. The table shows the rollout error after 100 steps, which is 5 seconds. It also shows the
    energy change on 5 particles after 1,000 steps. The ranges are over 3 seeds.

    | Model | 5 particles | 10 | 20 | Stiffer | Energy change |
    |---|---|---|---|---|---|
    | coasting | 2.13 | 2.19 | 2.20 | 2.06 | 87,800% |
    | flat MLP | 0.75 to 0.77 | can't run | can't run | 0.98 to 1.05 | 470% to 980% |
    | `RelativeForces` | 0.18 to 1.00 | 0.29 to 2.10 | 0.50 to 4.30 | 2.9 to 6.6 | 6.5% to 1,090% |
    | central, stiffness read as an input | 0.027 to 0.042 | 0.042 to 0.082 | 0.12 to 0.21 | 0.55 to 0.80 | 1.4% to 1.5% |
    | `CentralForces` | 0.012 to 0.018 | 0.023 to 0.034 | 0.087 to 0.11 | 0.033 to 0.043 | 1.3% to 1.8% |

    `RelativeForces` fit most states, but its rollouts drift away from the truth. Its pull on i from
    j is never exactly minus its pull on j from i. So each step adds momentum that should not exist.
    After 1,000 steps, a system's total momentum, which starts at 0, has a median size of 1.1 to 19.
    A system that drifts as a whole also gains energy.

    `CentralForces` makes the two pulls of every pair equal and opposite by construction. Both ends
    see the same length and stiffness, and they pull along opposite offsets. That is Newton's third
    law. And with a force that depends only on the distance, energy is conserved as well. Its
    momentum stays near 1e-6. Its energy changes about as much as the true simulator's, 2.0% to 2.6%
    on 20 particles against 2.3%.

    The Stiffer column shows why Hooke's form helps. The row "central, stiffness read as an input"
    feeds the stiffness into the network instead of multiplying by it. A stiffness above any seen
    in training then leaves the network guessing. Its error is 15 to 24 times larger than when the
    stiffness multiplies the pull.

    Interaction Networks simulated "dozens of objects over thousands of time steps". GNS ran an
    order of magnitude more particles than it trained on. Both use this design, messages between
    pairs, at a larger scale. `CentralForces`, trained on 5 particles, simulates 20 particles with an
    error of at most 0.11.
    """,
)

PHYSICS4_GIVEN = PHYSICS3_GIVEN + POOL_PART + BASELINE_SOLUTION

PHYSICS4_SOURCE = source_file(
    '''
    # physics4.py

    import copy
    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    changes = torch.tensor([0.01, -0.03, 0.5]).abs().median()    # tensor(0.0300), the middle value of the three
    named = {"a": 1.5}["a"]                                       # 1.5


    # physics1's world, physics2's baselines and training, and physics3's simulators.
''' + indented(PHYSICS4_GIVEN) + '''


''' + indented(MODELS_SOLUTION) + '''


    # TODO: Run `steps` euler_steps with accelerate. For each system, compute
    #   the relative change in energy, (end energy - start energy) / start
    #   energy. Return the median over the systems of its absolute value, as a
    #   Python float. Work inside torch.no_grad().
    def energy_drift(
        accelerate: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        v: Tensor,
        springs: Tensor,
        steps: int,
    ) -> float:
        ...


    # TODO: Return a dict that maps each case's name to accelerate's
    #   rollout_error at the last of `steps` steps, as a Python float. cases
    #   maps each name to (x, v, springs). Each case's truth is the rollout of
    #   the true spring_accelerations for `steps` steps.
    def case_errors(
        accelerate: Callable[[Tensor, Tensor], Tensor],
        cases: dict[str, tuple[Tensor, Tensor, Tensor]],
        steps: int,
    ) -> dict[str, float]:
        ...


    # Coasting and both simulators, trained on 5 particles and rolled out on every case. These lines run as the file
    # loads.
''' + indented(PHYSICS4_NAMES) + '''
    ''',
)

PHYSICS4_SOLUTION = source_file(
    '''
    # physics4.py, a solution for comparison

    import copy
    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    changes = torch.tensor([0.01, -0.03, 0.5]).abs().median()
    named = {"a": 1.5}["a"]


''' + indented(PHYSICS4_GIVEN) + '''


''' + indented(MODELS_SOLUTION) + '''


''' + indented(DRIFT_SOLUTION) + '''


''' + indented(PHYSICS4_NAMES) + '''
    ''',
)

_PHYSICS4_EXPECTED: dict[str, object] = {}


def physics4_expected() -> dict[str, object]:
    if not _PHYSICS4_EXPECTED:
        namespace = dict(ROLLOUT_REFERENCE)
        exec(PHYSICS4_NAMES, namespace)
        _PHYSICS4_EXPECTED.update({name: namespace[name] for name in ("table", "drifts")})

    return _PHYSICS4_EXPECTED


def check_energy_drift_records_nothing(found: object) -> None:
    seen: list[bool] = []
    x, v, springs = WORLD_REFERENCE["random_systems"](2, 4, torch.Generator().manual_seed(1097))
    found(grad_recorder(seen), x, v, springs, 3)

    if any(seen):
        raise Mismatch("energy_drift ran accelerate while recording for autograd. Wrap its steps in torch.no_grad().")


def check_table_name(value: object) -> None:
    expected = physics4_expected()["table"]
    reject_missing_return(
        call="table",
        actual=value,
    )

    if not isinstance(value, dict) or sorted(value) != sorted(expected):
        raise Mismatch(f"table is {value_text(value)}, but it should be a dict with the keys {sorted(expected)}.")

    for name, row in expected.items():
        expect_number_dict(
            call=f"table[{name!r}]",
            actual=value[name],
            expected=row,
        )

    for case in expected["central"]:
        if not value["central"][case] < min(value["coasting"][case], value["relative"][case]):
            raise Mismatch(f"CentralForces should beat coasting and RelativeForces on {case}, but it doesn't.")


def check_drifts_name(value: object) -> None:
    expect_number_dict(
        call="drifts",
        actual=value,
        expected=physics4_expected()["drifts"],
    )


def check_physics4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="energy_drift",
        description="gives the median size of each system's relative energy change",
        inputs=(
            (TRUE_PULL, *systems(20, 5, 1098), 200),
            (CONSTANT_PULL, *systems(7, 4, 1099), 30),
        ),
        reference=ROLLOUT_REFERENCE["energy_drift"],
        expect=expect_scaled,
    )
    checker.test(
        function="energy_drift",
        description="records nothing for autograd",
        run=check_energy_drift_records_nothing,
    )
    small_cases = {
        "four": tuple(WORLD_REFERENCE["random_systems"](3, 4, torch.Generator().manual_seed(1100))),
        "seven": tuple(WORLD_REFERENCE["random_systems"](2, 7, torch.Generator().manual_seed(1101))),
    }
    check_on_inputs(
        checker=checker,
        function="case_errors",
        description="gives each case's rollout error at the last step",
        inputs=(
            (Described(BASELINE_REFERENCE["coasting"], "coasting"), Described(small_cases, "2 cases, of 4 and 7 particles"), 12),
            (CONSTANT_PULL, Described(small_cases, "2 cases, of 4 and 7 particles"), 5),
        ),
        reference=ROLLOUT_REFERENCE["case_errors"],
        expect=expect_number_dict,
    )
    checker.check_name(
        name="table",
        description="holds each simulator's rollout error on every case, the central one best",
        run=check_table_name,
    )
    checker.check_name(
        name="drifts",
        description="holds both simulators' energy drift over 1,000 steps",
        run=check_drifts_name,
    )


PHYSICS4 = Exercise(
    name="physics4",
    chapter=LEARNED_PHYSICS,
    title="Project: long rollouts and new systems",
    lesson=PHYSICS4_LESSON,
    source=PHYSICS4_SOURCE,
    solution=PHYSICS4_SOLUTION,
    hints={
        "energy_drift": "Save energy(x, v, springs) as the start. Inside torch.no_grad(), call euler_step steps "
        "times. Then compute (end - start) / start and take its .abs(), its .median() and .item().",
        "case_errors": "Loop over cases.items(). For each case, truth is the rollout of spring_accelerations for "
        "steps steps. Take rollout_error of accelerate against it, and keep its last entry with .item().",
        "table": "This name comes from the lines at the end of the file. Finish case_errors first.",
        "drifts": "This name comes from the lines at the end of the file. Finish energy_drift first.",
    },
    introduces=(),
    check=check_physics4,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

class AbsolutePairs(nn.Module):
    """A pair network that reads both particles' own coordinates, where RelativeForces reads only their offset."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: Tensor, springs: Tensor) -> Tensor:
        count, particles, _ = x.shape
        pairs = torch.cat(
            [
                x.unsqueeze(2).expand(count, particles, particles, 2),
                x.unsqueeze(1).expand(count, particles, particles, 2),
                springs.unsqueeze(3),
            ],
            dim=3,
        )

        return (self.message(pairs) * (springs > 0).unsqueeze(3)).sum(dim=2)


class StiffnessRead(nn.Module):
    """Central forces whose law reads the stiffness as an input, where CentralForces multiplies by it."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.law = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: Tensor, springs: Tensor) -> Tensor:
        offsets = x.unsqueeze(1) - x.unsqueeze(2)
        lengths = offsets.norm(dim=3, keepdim=True).clamp_min(1e-9)
        sizes = self.law(torch.cat([lengths, springs.unsqueeze(3)], dim=3))

        return (sizes * offsets / lengths * (springs > 0).unsqueeze(3)).sum(dim=2)


def explicit_drift(
    x: Tensor,
    v: Tensor,
    springs: Tensor,
    steps: int,
) -> float:
    """Plain Euler, which moves with the old velocity, for physics1's comparison."""
    accelerate = WORLD_REFERENCE["spring_accelerations"]
    start = WORLD_REFERENCE["energy"](x, v, springs)

    for _ in range(steps):
        x, v = x + v * 0.05, v + accelerate(x, springs) * 0.05

    return ((WORLD_REFERENCE["energy"](x, v, springs) - start) / start).abs().median().item()


def measure_integrators() -> dict[str, float]:
    """physics1's table, the median change of energy over 100 systems of 5 after 1,000 steps, and physics2's typical
    acceleration, the root mean square length over the held out states."""
    x, v, springs = WORLD_REFERENCE["random_systems"](100, 5, torch.Generator().manual_seed(3))

    with pinned_threads():
        _, held = lesson_pools()

        return {
            "semi-implicit": ROLLOUT_REFERENCE["energy_drift"](WORLD_REFERENCE["spring_accelerations"], x, v, springs, 1000),
            "explicit": explicit_drift(x, v, springs, 1000),
            "typical acceleration": held[2].norm(dim=2).pow(2).mean().sqrt().item(),
        }


MEASURE_STEPS = 3000
SEEDS = range(3)


def lesson_pools() -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    return (
        BASELINE_REFERENCE["state_pool"](500, 5, 100, torch.Generator().manual_seed(1)),
        BASELINE_REFERENCE["state_pool"](50, 5, 100, torch.Generator().manual_seed(2)),
    )


def trained(
    make: Callable[[], nn.Module],
    pool: tuple[Tensor, Tensor, Tensor],
    seed: int,
    steps: int = MEASURE_STEPS,
) -> nn.Module:
    torch.manual_seed(seed)

    return BASELINE_REFERENCE["train_accelerations"](make(), pool, steps, torch.Generator().manual_seed(seed))


# physics3's tables run 5 seeds, since one seed in 3 at a constant rate once stopped on a bad step.
PHYSICS3_SEEDS = range(5)


def trained_with_decay(
    make: Callable[[], nn.Module],
    pool: tuple[Tensor, Tensor, Tensor],
    seed: int,
    steps: int = MEASURE_STEPS,
) -> nn.Module:
    """train_accelerations's run with its rate falling along a cosine from 3e-3 to 0, for physics3's tables. At a
    constant 3e-3 the held out share swings by up to 0.49 between checkpoints 1,000 steps apart, so where a run stops
    decides its score. GNS decays its rate too."""
    torch.manual_seed(seed)
    model = make()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    xs, ss, targets = pool

    for _ in range(steps):
        rows = torch.randint(0, len(xs), (64,), generator=generator)
        loss = F.mse_loss(model(xs[rows], ss[rows]), targets[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


def lesson_cases() -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    systems = WORLD_REFERENCE["random_systems"]

    return {
        "5 particles": systems(100, 5, torch.Generator().manual_seed(3)),
        "10 particles": systems(100, 10, torch.Generator().manual_seed(4)),
        "20 particles": systems(100, 20, torch.Generator().manual_seed(5)),
        "stiffer springs": systems(100, 5, torch.Generator().manual_seed(6), 1.5, 3.0),
    }


def measure_baselines() -> dict[str, object]:
    """physics2's table, the flat MLP over 3 seeds and coasting, on the lesson's pool and held out states."""
    share = BASELINE_REFERENCE["share_close"]
    diagnose = BASELINE_REFERENCE["diagnose"]
    rollout_error = BASELINE_REFERENCE["rollout_error"]
    results: dict[str, object] = {}

    with pinned_threads():
        pool, held = lesson_pools()
        x, v, springs = lesson_cases()["5 particles"]
        truth = WORLD_REFERENCE["rollout"](WORLD_REFERENCE["spring_accelerations"], x, v, springs, 100)
        coast = rollout_error(BASELINE_REFERENCE["coasting"], x, v, springs, truth)
        results["coasting"] = (share(BASELINE_REFERENCE["coasting"], pool, 0.1), share(BASELINE_REFERENCE["coasting"], held, 0.1), coast[50].item(), coast[100].item())

        for seed in SEEDS:
            model = trained(BASELINE_REFERENCE["FlatMLP"], pool, seed)
            errors = rollout_error(model, x, v, springs, truth)
            fit, held_fit = share(model, pool, 0.1), share(model, held, 0.1)
            results[f"flat {seed}"] = (fit, held_fit, errors[50].item(), errors[100].item(), diagnose(fit, held_fit, 0.9, 0.1))

        small = BASELINE_REFERENCE["state_pool"](4, 5, 100, torch.Generator().manual_seed(7))

        for seed in SEEDS:
            model = trained(BASELINE_REFERENCE["FlatMLP"], small, seed)
            fit, held_fit = share(model, small, 0.1), share(model, held, 0.1)
            results[f"flat on 4 systems {seed}"] = (fit, held_fit, diagnose(fit, held_fit, 0.9, 0.1))

        results["weights"] = sum(parameter.numel() for parameter in BASELINE_REFERENCE["FlatMLP"]().parameters())

    return results


def measure_longer_training() -> dict[str, object]:
    """physics2's second table, the flat MLP trained 20,000 steps at its own width and at a width of 512, over 3
    seeds, to see whether its failure to fit is its training's."""
    share = BASELINE_REFERENCE["share_close"]
    results: dict[str, object] = {}

    with pinned_threads():
        pool, held = lesson_pools()

        for name, make in (
            ("20000 steps", BASELINE_REFERENCE["FlatMLP"]),
            ("20000 steps, width 512", lambda: BASELINE_REFERENCE["FlatMLP"](5, 512)),
        ):
            for seed in SEEDS:
                model = trained(make, pool, seed, 20000)
                results[f"{name} {seed}"] = (share(model, pool, 0.1), share(model, held, 0.1))

    return results


def measure_structures() -> dict[str, object]:
    """physics3's tables, each structure's property gaps untrained and its share of held out accelerations within
    0.1 after training with a decaying rate, plain, moved by (4, -3) and turned by 1 radian, over 5 seeds."""
    share = BASELINE_REFERENCE["share_close"]
    rotate = STRUCTURE_REFERENCE["rotate"]
    makers = {
        "flat": BASELINE_REFERENCE["FlatMLP"],
        "absolute pairs": AbsolutePairs,
        "relative": STRUCTURE_REFERENCE["RelativeForces"],
        "central": STRUCTURE_REFERENCE["CentralForces"],
    }
    results: dict[str, object] = {}

    with pinned_threads():
        x, v, springs = WORLD_REFERENCE["random_systems"](8, 5, torch.Generator().manual_seed(3))
        shift = torch.tensor([4.0, -3.0])
        order = torch.tensor([3, 0, 4, 1, 2])

        for name, make in makers.items():
            torch.manual_seed(0)
            model = make()
            results[f"{name} gaps"] = (
                STRUCTURE_REFERENCE["moved_gap"](model, x, springs, shift),
                STRUCTURE_REFERENCE["turned_gap"](model, x, springs, 1.0),
                STRUCTURE_REFERENCE["shuffled_gap"](model, x, springs, order),
            )
            results[f"{name} weights"] = sum(parameter.numel() for parameter in model.parameters())

        pool, held = lesson_pools()
        xs, ss, targets = held
        moved = (xs + shift, ss, targets)
        turned = (rotate(xs, 1.0), ss, rotate(targets, 1.0))

        for name, make in makers.items():
            for seed in PHYSICS3_SEEDS:
                model = trained_with_decay(make, pool, seed)
                results[f"{name} {seed}"] = (share(model, held, 0.1), share(model, moved, 0.1), share(model, turned, 0.1))

    return results


def line_pool(
    systems: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """state_pool's states for systems laid out along the x axis and moving along it, so they stay on it."""
    x, v, springs = WORLD_REFERENCE["random_systems"](systems, 5, torch.Generator().manual_seed(seed))
    on_line = torch.tensor([1.0, 0.0])
    xs = WORLD_REFERENCE["rollout"](WORLD_REFERENCE["spring_accelerations"], x * on_line, v * on_line, springs, 100).flatten(0, 1)
    ss = springs.repeat_interleave(101, dim=0)

    return xs, ss, WORLD_REFERENCE["spring_accelerations"](xs, ss)


def measure_line() -> dict[str, object]:
    """physics3's second table, each structure trained with a decaying rate only on systems along one line, then
    scored on new systems along it and on the same states turned 90 degrees, over 5 seeds."""
    share = BASELINE_REFERENCE["share_close"]
    rotate = STRUCTURE_REFERENCE["rotate"]
    makers = {
        "flat": BASELINE_REFERENCE["FlatMLP"],
        "absolute pairs": AbsolutePairs,
        "relative": STRUCTURE_REFERENCE["RelativeForces"],
        "central": STRUCTURE_REFERENCE["CentralForces"],
    }
    results: dict[str, object] = {}

    with pinned_threads():
        pool, held = line_pool(500, 1), line_pool(50, 2)
        xs, ss, targets = held
        turned = (rotate(xs, math.pi / 2), ss, rotate(targets, math.pi / 2))

        for name, make in makers.items():
            for seed in PHYSICS3_SEEDS:
                model = trained_with_decay(make, pool, seed)
                results[f"{name} {seed}"] = (share(model, held, 0.1), share(model, turned, 0.1))

    return results


def momentum_drift(
    accelerate: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    springs: Tensor,
    steps: int,
) -> float:
    """The median size of a system's total momentum after steps, which starts at 0 and stays 0 under the true forces."""
    with torch.no_grad():
        for _ in range(steps):
            x, v = WORLD_REFERENCE["euler_step"](accelerate, x, v, springs)

    return v.sum(dim=1).norm(dim=1).median().item()


def measure_rollouts() -> dict[str, object]:
    """physics4's table, each simulator's rollout error after 50 and 100 steps on every case, and its energy and
    momentum drift on 5 and 20 particles after 1,000 steps, over 3 seeds."""
    rollout_error = BASELINE_REFERENCE["rollout_error"]
    drift = ROLLOUT_REFERENCE["energy_drift"]
    makers = {
        "flat": BASELINE_REFERENCE["FlatMLP"],
        "relative": STRUCTURE_REFERENCE["RelativeForces"],
        "stiffness read": StiffnessRead,
        "central": STRUCTURE_REFERENCE["CentralForces"],
    }
    results: dict[str, object] = {}

    with pinned_threads():
        pool, _ = lesson_pools()
        cases = lesson_cases()
        truths = {name: WORLD_REFERENCE["rollout"](WORLD_REFERENCE["spring_accelerations"], *case, 100) for name, case in cases.items()}

        def score(accelerate: Callable[[Tensor, Tensor], Tensor], name: str) -> dict[str, object]:
            row: dict[str, object] = {}

            for case, (x, v, springs) in cases.items():
                if name == "flat" and x.shape[1] != 5:
                    continue

                errors = rollout_error(accelerate, x, v, springs, truths[case])
                row[case] = (errors[50].item(), errors[100].item())

            row["energy 5"] = drift(accelerate, *cases["5 particles"], 1000)
            row["momentum 5"] = momentum_drift(accelerate, *cases["5 particles"], 1000)

            if name != "flat":
                row["energy 20"] = drift(accelerate, *cases["20 particles"], 1000)

            return row

        results["true"] = score(WORLD_REFERENCE["spring_accelerations"], "true")
        results["coasting"] = score(BASELINE_REFERENCE["coasting"], "coasting")

        for name, make in makers.items():
            for seed in SEEDS:
                results[f"{name} {seed}"] = score(trained(make, pool, seed), name)

    return results
