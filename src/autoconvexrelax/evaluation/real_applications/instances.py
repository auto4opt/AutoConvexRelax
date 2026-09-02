from dataclasses import dataclass
from typing import Callable

from autoconvexrelax.core.problem import QCQPProblem

from autoconvexrelax.evaluation.real_applications.beamforming import build_multicast_beamforming_problem
from autoconvexrelax.evaluation.real_applications.snl import build_snl_least_squares_problem


@dataclass(frozen=True)
class RealApplicationInstanceSpec:
    app_key: str
    instance_key: str
    instance_index: int
    builder: Callable[..., QCQPProblem]
    kwargs: dict


REAL_APPLICATION_INSTANCE_SPECS = [
    RealApplicationInstanceSpec(
        app_key="beamforming",
        instance_key="beamforming_1",
        instance_index=1,
        builder=build_multicast_beamforming_problem,
        kwargs={},
    ),
    RealApplicationInstanceSpec(
        app_key="beamforming",
        instance_key="beamforming_2",
        instance_index=2,
        builder=build_multicast_beamforming_problem,
        kwargs={"name": "real_multicast_beamforming_seed13", "seed": 13},
    ),
    RealApplicationInstanceSpec(
        app_key="beamforming",
        instance_key="beamforming_3",
        instance_index=3,
        builder=build_multicast_beamforming_problem,
        kwargs={"name": "real_multicast_beamforming_seed23", "seed": 23},
    ),
    RealApplicationInstanceSpec(
        app_key="beamforming",
        instance_key="beamforming_4",
        instance_index=4,
        builder=build_multicast_beamforming_problem,
        kwargs={"name": "real_multicast_beamforming_seed37", "seed": 37},
    ),
    RealApplicationInstanceSpec(
        app_key="snl",
        instance_key="snl_1",
        instance_index=1,
        builder=build_snl_least_squares_problem,
        kwargs={},
    ),
    RealApplicationInstanceSpec(
        app_key="snl",
        instance_key="snl_2",
        instance_index=2,
        builder=build_snl_least_squares_problem,
        kwargs={"name": "real_snl_bounded_noise_seed19", "seed": 19},
    ),
    RealApplicationInstanceSpec(
        app_key="snl",
        instance_key="snl_3",
        instance_index=3,
        builder=build_snl_least_squares_problem,
        kwargs={"name": "real_snl_bounded_noise_seed29", "seed": 29},
    ),
    RealApplicationInstanceSpec(
        app_key="snl",
        instance_key="snl_4",
        instance_index=4,
        builder=build_snl_least_squares_problem,
        kwargs={"name": "real_snl_bounded_noise_seed41", "seed": 41},
    ),
]


def iter_real_application_specs(app_filter: str = "all"):
    for spec in REAL_APPLICATION_INSTANCE_SPECS:
        if app_filter == "all" or spec.app_key == app_filter:
            yield spec


def build_real_application_instance(spec: RealApplicationInstanceSpec) -> QCQPProblem:
    prob = spec.builder(**spec.kwargs)
    data = getattr(prob, "real_application_data", None)
    if isinstance(data, dict):
        data["instance_key"] = spec.instance_key
        data["instance_index"] = int(spec.instance_index)
    return prob


def build_real_application_instances(app_filter: str = "all"):
    return [build_real_application_instance(spec) for spec in iter_real_application_specs(app_filter)]
