import os
import argparse
import json
import sqlite3
from pathlib import Path

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):

    scenario_types = None

    scenario_tokens  # List of scenario tokens to include
    log_names = log_names  # Filter scenarios by log names
    map_names = None  # Filter scenarios by map names

    num_scenarios_per_type  # Number of scenarios per type
    limit_total_scenarios  # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None  # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None  # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True  # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = (
        False  # Whether to remove scenarios where the mission goal is invalid
    )
    shuffle  # Whether to shuffle the scenarios

    ego_start_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from below
    )
    ego_stop_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from above
    )
    speed_noise_tolerance = None  # Value at or below which a speed change between two timepoints should be ignored as noise.

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def filter_valid_log_names(data_path, log_names):
    valid_log_names = []
    invalid_log_names = []

    total = len(log_names)
    existing_db_names = {
        os.path.splitext(name)[0]
        for name in os.listdir(data_path)
        if name.endswith(".db")
    }
    candidate_log_names = [name for name in log_names if name in existing_db_names]
    missing_count = total - len(candidate_log_names)

    print(
        f"Checking database readability for {len(candidate_log_names)} existing logs (missing={missing_count})..."
    )

    for idx, log_name in enumerate(candidate_log_names, start=1):
        db_path = os.path.join(data_path, f"{log_name}.db")
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Probe frequently used tables; much faster than full quick_check.
            cur.execute("SELECT token FROM lidar_pc LIMIT 1;")
            cur.fetchone()
            valid_log_names.append(log_name)
        except sqlite3.DatabaseError:
            invalid_log_names.append(log_name)
        finally:
            if conn is not None:
                conn.close()

        if idx % 500 == 0 or idx == len(candidate_log_names):
            print(
                f"Checked {idx}/{len(candidate_log_names)}, valid={len(valid_log_names)}, invalid={len(invalid_log_names)}"
            )

    if invalid_log_names:
        print(f"Skipping {len(invalid_log_names)} invalid/missing DB logs")
        print("Examples:", invalid_log_names[:5])

    return valid_log_names


def get_valid_log_names(data_path, log_names, cache_path=None, refresh_cache=False):
    if cache_path and (not refresh_cache) and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            cached_logs = cache_data.get("valid_log_names", [])
            existing_db_names = {
                os.path.splitext(name)[0]
                for name in os.listdir(data_path)
                if name.endswith(".db")
            }
            valid_from_cache = [name for name in cached_logs if name in existing_db_names]
            print(
                f"Loaded {len(valid_from_cache)} valid logs from cache: {cache_path}"
            )
            if valid_from_cache:
                return valid_from_cache
        except Exception as e:
            print(f"DB cache load failed, rebuilding cache: {e}")

    valid_logs = filter_valid_log_names(data_path, log_names)

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"valid_log_names": valid_logs}, f)
        print(f"Saved DB validation cache to: {cache_path}")

    return valid_logs


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument(
        "--data_path",
        default="/data/wyf/lgq/nuplan/dataset",
        type=str,
        help="path to raw data",
    )
    parser.add_argument(
        "--map_path",
        default="/data/wyf/lgq/nuplan/dataset/maps",
        type=str,
        help="path to map data",
    )

    parser.add_argument(
        "--save_path", default="./cache", type=str, help="path to save processed data"
    )
    parser.add_argument(
        "--scenarios_per_type",
        type=int,
        default=None,
        help="number of scenarios per type",
    )
    parser.add_argument(
        "--total_scenarios",
        type=int,
        default=10,
        help="limit total number of scenarios",
    )
    parser.add_argument(
        "--shuffle_scenarios", type=str2bool, default=True, help="shuffle scenarios"
    )

    parser.add_argument("--agent_num", type=int, help="number of agents", default=32)
    parser.add_argument(
        "--static_objects_num", type=int, help="number of static objects", default=5
    )

    parser.add_argument("--lane_len", type=int, help="number of lane point", default=20)
    parser.add_argument("--lane_num", type=int, help="number of lanes", default=70)

    parser.add_argument(
        "--route_len", type=int, help="number of route lane point", default=20
    )
    parser.add_argument(
        "--route_num", type=int, help="number of route lanes", default=25
    )
    parser.add_argument(
        "--db_validation_cache",
        type=str,
        default=None,
        help="optional cache file path for valid DB log names",
    )
    parser.add_argument(
        "--refresh_db_validation_cache",
        type=str2bool,
        default=False,
        help="rebuild DB validation cache instead of reusing existing cache",
    )
    parser.add_argument(
        "--train_set_list_output",
        type=str,
        default=None,
        help="optional output path for training set json list",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")
    if not os.path.exists(args.map_path):
        raise FileNotFoundError(f"map_path not found: {args.map_path}")

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    # Only preprocess the training data
    with open(script_dir / "nuplan_train.json", "r", encoding="utf-8") as file:
        log_names = json.load(file)

    cache_path = args.db_validation_cache
    if cache_path is None:
        cache_path = str(script_dir / "valid_train_logs_cache.json")

    log_names = get_valid_log_names(
        args.data_path,
        log_names,
        cache_path=cache_path,
        refresh_cache=args.refresh_db_validation_cache,
    )
    if not log_names:
        raise RuntimeError("No valid logs found after DB validation")

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(
        args.data_path, args.map_path, sensor_root, db_files, map_version
    )
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            args.total_scenarios,
            args.shuffle_scenarios,
            log_names=log_names,
        )
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(scenarios)

    npz_files = sorted(f for f in os.listdir(args.save_path) if f.endswith(".npz"))

    train_list_output = args.train_set_list_output
    if train_list_output is None:
        train_list_output = os.path.join(args.save_path, "train_list.json")

    train_list_output = os.path.abspath(train_list_output)
    os.makedirs(os.path.dirname(train_list_output), exist_ok=True)
    with open(train_list_output, "w", encoding="utf-8") as json_file:
        json.dump(npz_files, json_file, indent=4)

    # Keep legacy output for compatibility with existing workflows.
    with open(script_dir / "diffusion_planner_training.json", "w") as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved training list to {train_list_output}")
    print(f"Saved {len(npz_files)} .npz file names")
