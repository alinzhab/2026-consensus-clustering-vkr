import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from scipy.io import loadmat
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = DATASETS_DIR / "uploaded"
PYTHON_PORT_DIR = BASE_DIR / "python_port"
sys.path.insert(0, str(PYTHON_PORT_DIR))

from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired_dataset
from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust_dataset
from sdgca import run_sdgca
from sdgca_modified import resolve_params, run_sdgca_modified


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

ALGORITHM_LABELS = {
    "hierarchical_baseline": "Иерархическая базовая версия",
    "hierarchical_weighted": "Иерархическая взвешенная версия",
    "sdgca": "SDGCA",
    "sdgca_modified": "SDGCA, модифицированная версия",
}

METHOD_LABELS = {
    "average": "average",
    "complete": "complete",
    "single": "single",
    "ward": "ward",
}


def ensure_dirs():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def allowed_dataset(filename):
    return Path(filename).suffix.lower() in {".npz", ".mat"}


def summarize_dataset(path: Path):
    summary = {
        "name": path.name,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
    try:
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            summary["keys"] = list(data.files)
            if "X" in data:
                summary["x_shape"] = tuple(data["X"].shape)
            if "gt" in data:
                summary["gt_shape"] = tuple(data["gt"].shape)
                summary["n_classes"] = int(np.unique(data["gt"]).size)
            if "members" in data:
                summary["members_shape"] = tuple(data["members"].shape)
            if "meta" in data:
                meta_raw = data["meta"]
                if np.isscalar(meta_raw):
                    meta_raw = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8", errors="ignore")
                if isinstance(meta_raw, str):
                    summary["meta"] = json.loads(meta_raw)
        elif path.suffix.lower() == ".mat":
            data = loadmat(path)
            keys = [key for key in data.keys() if not key.startswith("__")]
            summary["keys"] = keys
            if "gt" in data:
                gt = np.asarray(data["gt"]).reshape(-1)
                summary["gt_shape"] = tuple(gt.shape)
                summary["n_classes"] = int(np.unique(gt).size)
            if "members" in data:
                summary["members_shape"] = tuple(np.asarray(data["members"]).shape)
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def list_datasets():
    files = (
        sorted(DATASETS_DIR.glob("*.npz"))
        + sorted(DATASETS_DIR.glob("*.mat"))
        + sorted(UPLOADS_DIR.glob("*.npz"))
        + sorted(UPLOADS_DIR.glob("*.mat"))
    )
    seen = set()
    items = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        items.append(summarize_dataset(path))
    return items


def list_results():
    result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    items = []
    for path in result_files[:50]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["algorithm_label"] = ALGORITHM_LABELS.get(payload.get("algorithm"), payload.get("algorithm"))
            payload["method_label"] = METHOD_LABELS.get(payload.get("method"), payload.get("method"))
            items.append(payload)
        except Exception:
            continue
    return items


def build_base_context(active_page):
    datasets = list_datasets()
    results = list_results()
    return {
        "active_page": active_page,
        "datasets": datasets,
        "results": results,
        "dataset_count": len(datasets),
        "result_count": len(results),
    }


def find_dataset_path(dataset_name):
    candidates = [
        DATASETS_DIR / f"{dataset_name}.npz",
        DATASETS_DIR / f"{dataset_name}.mat",
        UPLOADS_DIR / f"{dataset_name}.npz",
        UPLOADS_DIR / f"{dataset_name}.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset not found: {dataset_name}")


def save_result_record(payload):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = payload["dataset"].replace(" ", "_")
    output_path = RESULTS_DIR / f"{timestamp}_{payload['algorithm']}_{safe_name}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _optional_float(value):
    value = (value or "").strip()
    return None if value == "" else float(value)


@app.route("/", methods=["GET"])
def index():
    ensure_dirs()
    context = build_base_context("home")
    context["recent_results"] = context["results"][:5]
    return render_template("home.html", **context)


@app.route("/datasets", methods=["GET", "POST"])
def datasets_page():
    ensure_dirs()
    upload_message = None
    if request.method == "POST":
        file = request.files.get("dataset_file")
        upload_message = "Файл не выбран."
        if file and file.filename:
            if not allowed_dataset(file.filename):
                upload_message = "Можно загружать только файлы форматов .npz и .mat."
            else:
                filename = secure_filename(file.filename)
                target = UPLOADS_DIR / filename
                file.save(target)
                upload_message = f"Датасет {filename} успешно загружен."
    context = build_base_context("datasets")
    context["upload_message"] = upload_message
    return render_template("datasets.html", **context)


@app.route("/generate", methods=["GET", "POST"])
def generate_page():
    ensure_dirs()
    generation_result = None
    if request.method == "POST":
        generator_type = request.form.get("generator_type", "densired")
        dataset_name = request.form.get("dataset_name", "").strip() or f"generated_{generator_type}"
        try:
            if generator_type == "densired":
                params = {
                    "name": dataset_name,
                    "dim": int(request.form.get("densired_dim", 2)),
                    "clunum": int(request.form.get("densired_clunum", 6)),
                    "core_num": int(request.form.get("densired_core_num", 80)),
                    "data_num": int(request.form.get("densired_data_num", 2000)),
                    "seed": int(request.form.get("densired_seed", 19)),
                    "domain_size": float(request.form.get("densired_domain_size", 20)),
                    "radius": float(request.form.get("densired_radius", 0.04)),
                    "step": float(request.form.get("densired_step", 0.06)),
                    "noise_ratio": float(request.form.get("densired_noise_ratio", 0.1)),
                    "density_factors": [float(x.strip()) for x in request.form.get("densired_density_factors", "1,1,0.8,1.2,0.6,1.4").split(",") if x.strip()],
                    "momentum": float(request.form.get("densired_momentum", 0.4)),
                    "branch": float(request.form.get("densired_branch", 0.05)),
                    "star": float(request.form.get("densired_star", 0.1)),
                    "distribution": request.form.get("densired_distribution", "uniform"),
                }
                if len(params["density_factors"]) != params["clunum"]:
                    params["density_factors"] = [1.0] * params["clunum"]
                x, gt, members, meta = generate_densired_style_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_densired_dataset(output_path, x, gt, members, meta)
            else:
                distributions = [x.strip() for x in request.form.get("repliclust_distributions", "normal,student_t,lognormal").split(",") if x.strip()]
                proportions = [float(x.strip()) for x in request.form.get("repliclust_distribution_proportions", "0.5,0.3,0.2").split(",") if x.strip()]
                if len(distributions) != len(proportions):
                    raise ValueError("Число распределений и число пропорций должно совпадать.")
                params = {
                    "name": dataset_name,
                    "n_clusters": int(request.form.get("repliclust_n_clusters", 6)),
                    "dim": int(request.form.get("repliclust_dim", 2)),
                    "n_samples": int(request.form.get("repliclust_n_samples", 2500)),
                    "aspect_ref": float(request.form.get("repliclust_aspect_ref", 4.0)),
                    "aspect_maxmin": float(request.form.get("repliclust_aspect_maxmin", 6.0)),
                    "radius_ref": float(request.form.get("repliclust_radius_ref", 1.0)),
                    "radius_maxmin": float(request.form.get("repliclust_radius_maxmin", 2.0)),
                    "min_overlap": float(request.form.get("repliclust_min_overlap", 0.08)),
                    "max_overlap": float(request.form.get("repliclust_max_overlap", 0.18)),
                    "imbalance_ratio": float(request.form.get("repliclust_imbalance_ratio", 2.5)),
                    "distributions": distributions,
                    "distribution_proportions": proportions,
                    "seed": int(request.form.get("repliclust_seed", 19)),
                }
                x, gt, members, meta = generate_archetype_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_repliclust_dataset(output_path, x, gt, members, meta)
            generation_result = {
                "message": f"Датасет {output_path.name} успешно сгенерирован и сохранён.",
                "file": output_path.name,
                "x_shape": tuple(x.shape),
                "gt_shape": tuple(gt.shape),
                "members_shape": tuple(members.shape),
                "meta_pretty": json.dumps(meta, ensure_ascii=False, indent=2),
            }
        except Exception as exc:
            generation_result = {"error": str(exc)}
    context = build_base_context("generate")
    context["generation_result"] = generation_result
    return render_template("generate.html", **context)


@app.route("/test", methods=["GET", "POST"])
def test_page():
    ensure_dirs()
    run_result = None
    if request.method == "POST":
        dataset_name = request.form.get("selected_dataset")
        algorithm = request.form.get("algorithm")
        try:
            dataset_path = find_dataset_path(dataset_name)
            seed = int(request.form.get("seed", 19))
            m = int(request.form.get("m", 20))
            runs = int(request.form.get("runs", 5))
            method = request.form.get("method", "average")
            if algorithm == "hierarchical_baseline":
                result = run_hierarchical_consensus(dataset_path, dataset_name, seed=seed, m=m, cnt_times=runs, method=method)
            elif algorithm == "hierarchical_weighted":
                sharpen = float(request.form.get("sharpen", 1.5))
                result = run_weighted_hierarchical_consensus(dataset_path, dataset_name, seed=seed, m=m, cnt_times=runs, method=method, sharpen=sharpen)
            elif algorithm == "sdgca":
                lambda_ = float(request.form.get("lambda_", 0.09))
                eta = float(request.form.get("eta", 0.75))
                theta = float(request.form.get("theta", 0.65))
                result = run_sdgca(dataset_path, dataset_name, seed=seed, m=m, cnt_times=runs, nwca_para=lambda_, eta=eta, theta=theta, method=method)
            elif algorithm == "sdgca_modified":
                params = resolve_params(
                    dataset_name,
                    _optional_float(request.form.get("lambda_mod")),
                    _optional_float(request.form.get("eta_mod")),
                    _optional_float(request.form.get("theta_mod")),
                )
                result = run_sdgca_modified(
                    dataset_path,
                    dataset_name,
                    seed=seed,
                    m=m,
                    cnt_times=runs,
                    nwca_para=params["lambda_"],
                    eta=params["eta"],
                    theta=params["theta"],
                    diffusion_time=params["diffusion_time"],
                    method=method,
                )
            else:
                raise ValueError("Неизвестный алгоритм.")

            run_result = {
                "dataset": dataset_name,
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": seed,
                "m": m,
                "runs": runs,
                "nmi_mean": round(result["nmi_mean"], 6),
                "nmi_std": round(result["nmi_std"], 6),
                "ari_mean": round(result["ari_mean"], 6),
                "ari_std": round(result["ari_std"], 6),
                "f_mean": round(result["f_mean"], 6),
                "f_std": round(result["f_std"], 6),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            if algorithm == "hierarchical_weighted":
                run_result["sharpen"] = float(request.form.get("sharpen", 1.5))
            if algorithm == "sdgca":
                run_result["lambda_"] = float(request.form.get("lambda_", 0.09))
                run_result["eta"] = float(request.form.get("eta", 0.75))
                run_result["theta"] = float(request.form.get("theta", 0.65))
            if algorithm == "sdgca_modified":
                run_result["params"] = params
            save_result_record(run_result)
        except Exception as exc:
            run_result = {"error": str(exc)}
    context = build_base_context("test")
    context["run_result"] = run_result
    return render_template("test.html", **context)


@app.route("/results", methods=["GET"])
def results_page():
    ensure_dirs()
    context = build_base_context("results")
    return render_template("results.html", **context)


if __name__ == "__main__":
    ensure_dirs()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
