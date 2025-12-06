import os
import json
import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv
from blackboard.codebase.config.logging_config import setup_root_logger
from blackboard.codebase.components.attribute_mapper import AttributeMapper
from copy import deepcopy
import copy
from blackboard.evaluation_methods.evalutions import run_evaluations
from blackboard.codebase.components.reasoning_agent import ReasoningAgent
from blackboard.codebase.components.discussion_engine import DiscussionEngine
from datacorpus.tools.top_k_eval import evaluate_top_k
setup_root_logger(log_level="INFO")
logger = logging.getLogger(__name__)
gptmodel = "gpt-5"


def load_environment(vcslam_path_env: str):

    CURRENT = Path(__file__).resolve()
    METHODS_DIR = CURRENT.parent
    BLACKBOARD_DIR = METHODS_DIR.parent.parent
    ROOT = BLACKBOARD_DIR.parent
    ENV_DIR = ROOT / "env"

    load_dotenv(os.path.join(ENV_DIR, ".env"))
    openai_key = os.getenv("OPENAIKEY")

    base_dir = Path(vcslam_path_env)
    ontology_path = base_dir / "ontology" / "ontology.ttl"

    return {
        "openai_key": openai_key,
        "base_dir": base_dir,
        "ontology_path": ontology_path
    }



# ---------------------------------------------------------
# SAMPLE LOADER
# ---------------------------------------------------------
def load_sample(base_dir: Path, sid: str):
    folder = base_dir / sid
    json_path = folder / f"{sid}_samples.json"
    unmapped_path = folder / f"{sid}_unmapped.json"
    doc_path = folder / f"{sid}.txt"
    ref_path = folder / f"{sid}_mapped.json"

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    documentation = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    unmapped = json.loads(unmapped_path.read_text(encoding="utf-8")) if unmapped_path.exists() else {}
    ref_mapping = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    return json_data, documentation, unmapped, ref_mapping



def compute_reasoning_effect(before_map, after_map):
    effects = []

    for attr, before in before_map.items():
        before_cand = before.get("final_mapping")["candidate"] if before.get("final_mapping") else None
        after_cand = after_map[attr]["state"].get("final_mapping")["candidate"] if after_map[attr]["state"].get("final_mapping") else None

        if before_cand == after_cand:
            continue

        effects.append({
            "attribute": attr,
            "before": before_cand,
            "after": after_cand,
        })

    return effects

def extract_leaf_paths(data, parent_key=""):
    leaf_paths = set()

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                leaf_paths.update(extract_leaf_paths(value, full_key))
            else:
                leaf_paths.add(full_key)

    elif isinstance(data, list):
        for item in data:
            leaf_paths.update(extract_leaf_paths(item, parent_key))

    else:
        leaf_paths.add(parent_key)

    return leaf_paths

def extract_prefix_block(ontology: str) -> str:
    return "\n".join(
        line.strip()
        for line in ontology.splitlines()
        if line.strip().startswith("@prefix")
    )

# ---------------------------------------------------------
# Debug Methods
# ---------------------------------------------------------
def _reset_mapper_to_after_historical(mapper: AttributeMapper, documentation: str):

    state = mapper.state


    validated = state.get("validated_candidates") or []
    for cand in validated:
        cand.pop("documentation_vote", None)
        cand.pop("selection_vote", None)



    matrix = state.get("matrix") or []
    for row in matrix:
        agents = row.get("agents", {})
        agents.pop("documentation", None)
        agents.pop("selection", None)
    state["matrix"] = matrix


    state["final_mapping"] = None


    mapper.logs["documentation_reasoning"] = {
        "has_documentation": bool(documentation),
        "validated_candidates_count": len(validated),
        "raw_llm_response": None,
    }
    mapper.logs["select_final_mappings"] = {
        "validated_candidates_count": len(validated),
        "matrix_snapshot": None,
        "raw_llm_response": None,
        "warnings": [],
    }

def reset_to_post_validation(mapper: AttributeMapper):

    state = mapper.state

    validated = state.get("validated_candidates") or []
    for cand in validated:
        cand.pop("documentation_vote", None)
        cand.pop("historical_vote", None)
        cand.pop("selection_vote", None)


    matrix = state.get("matrix") or []
    for row in matrix:
        agents = row.get("agents", {})
        agents.pop("documentation", None)
        agents.pop("historical", None)
        agents.pop("selection", None)
    state["matrix"] = matrix

    state["final_mapping"] = None

    mapper.logs["documentation_reasoning"] = {
    }
    mapper.logs["select_final_mappings"] = {
    }
def restore_original_final_mapping_from_logs(mapper):


    logs = copy.deepcopy(mapper.logs.get("select_final_mappings"))
    if not logs:
        mapper.state["final_mapping"] = None
        return

    matrix_snapshot = logs.get("matrix_snapshot")
    raw_llm_response = logs.get("raw_llm_response")

    if not matrix_snapshot or not raw_llm_response:
        mapper.state["final_mapping"] = None
        return

    selected_idx = None
    for i, vote in enumerate(raw_llm_response):
        if vote.get("accepted", False):
            selected_idx = i
            break

    if selected_idx is None:
        mapper.state["final_mapping"] = None
        return

    selected = matrix_snapshot[selected_idx]


    mapper.state["final_mapping"] = {
        "candidate": selected["candidate"],
        "score": 1,
        "meta": selected,
        "selection_reason": selected["selection_vote"]["reason"]
        if "selection_vote" in selected
        else None,
        "index": selected_idx
    }
def reset_agent_checkpoints(mapper: AttributeMapper, reset_map : list):
    state = mapper.state


    validated = state.get("validated_candidates") or []
    raw_candidates = copy.deepcopy(state.get("candidates", []))
    reset_reason = False
    for cand in validated:
        reset_reason = False
        if "documentation" in reset_map:
            cand.pop("documentation_vote", None)
        if "historical" in reset_map:
            cand.pop("historical_vote", None)
        if "selection" in reset_map:
            cand.pop("selection_vote", None)
            reset_reason = True
        if "reasoning" in reset_map:
            reset_reason = True

        if reset_reason:
            for raw_cand in raw_candidates:
                if cand["candidate"] == raw_cand["candidate"]:
                    cand["reason"] = raw_cand["reason"]
    if "selection" in reset_map:
        state["final_mapping"] = {}

    elif "reasoning" in reset_map:
        restore_original_final_mapping_from_logs(mapper)


    matrix = state.get("matrix") or []
    for row in matrix:
        agents = row.get("agents", {})

        if "documentation" in reset_map:
            agents.pop("documentation", None)
        if "historical" in reset_map:
            agents.pop("historical", None)
        if "selection" in reset_map:
            agents.pop("selection", None)
    state["matrix"] = matrix


    if "selection" in reset_map:
        state["final_mapping"] = None


    if "documentation" in reset_map:
        mapper.logs["documentation_reasoning"] = {
    }
    if "historical" in reset_map:
        mapper.logs["select_final_mappings"] = {
    }

def rerun_mapping(mapper: AttributeMapper , rerun_map : list):
    if "documentation" in rerun_map:
        mapper.documentation_reasoning()
    if "historical" in rerun_map:
        mapper.historical_references_reasoning()
    if "example_value" in rerun_map:
        mapper.example_value_reasoning()
    if "proximity" in rerun_map:
        mapper.attribute_label_proximity_reasoning()
    if "selection" in rerun_map:
        mapper.select_final_mappings()



def clone_mapper_set(mappers_base, api_key, json_data, doc, hist, ontology_str, gpt_model):
    cloned = {}
    for attr, mapper in mappers_base.items():
        new_mapper = AttributeMapper(
            attribute=attr,
            api_key=api_key,
            input_data={
                "json_data": json_data,
                "documentation": doc,
                "historical_references": hist,
                "ontology": ontology_str,
            },
            gpt_model=gpt_model
        )
        new_mapper.state = json.loads(json.dumps(mapper.state))
        new_mapper.logs = json.loads(json.dumps(mapper.logs))
        cloned[attr] = new_mapper
    return cloned


def merge_label_and_example(label_mapper: AttributeMapper,
                            example_mapper: AttributeMapper,
                            api_key, json_data, documentation, hist, ontology, gpt_model):

    import copy

    merged = AttributeMapper(
        attribute=label_mapper.name,
        api_key=api_key,
        input_data={
            "json_data": json_data,
            "documentation": documentation,
            "historical_references": hist,
            "ontology": ontology,
        },
        gpt_model=gpt_model
    )


    merged.state = copy.deepcopy(label_mapper.state)


    val_label   = label_mapper.state.get("validated_candidates") or []
    val_example = example_mapper.state.get("validated_candidates") or []

    merged_val = []
    for c_label in val_label:
        c = copy.deepcopy(c_label)

        ex_match = next((x for x in val_example if x["candidate"] == c["candidate"]), None)

        if ex_match:

            if "example_value_vote" in ex_match:
                c["example_value_vote"] = copy.deepcopy(ex_match["example_value_vote"])
            if "attribute_label_proximity_vote" in ex_match:
                c["attribute_label_proximity_vote"] = copy.deepcopy(ex_match["attribute_label_proximity_vote"])
        merged_val.append(c)

    merged.state["validated_candidates"] = merged_val

    # ---- MATRIX MERGE ----
    matrix_label = label_mapper.state.get("matrix") or []
    matrix_example = example_mapper.state.get("matrix") or []


    merged_matrix = copy.deepcopy(matrix_label)


    for i, row in enumerate(merged_matrix):
        ag = row.setdefault("agents", {})

        if i < len(matrix_example):
            ex_ag = matrix_example[i].get("agents", {})
            if "example_value" in ex_ag:
                ag["example_value"] = copy.deepcopy(ex_ag["example_value"])

    merged.state["matrix"] = merged_matrix


    merged.logs = {}
    merged.logs.update(copy.deepcopy(label_mapper.logs))
    merged.logs.update(copy.deepcopy(example_mapper.logs))


    merged.state["final_mapping"] = None

    return merged

def run_pipeline(
    vcslam_path: str,
    sample_ids: list[str],
    historical_ids: list[str],
    export_root: str,
    run_evaluation: bool,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = Path(export_root) / timestamp
    base_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Export will be saved in: {base_output_dir}")


    env = load_environment(vcslam_path)
    api_key = env["openai_key"]
    base_dir = env["base_dir"]

    ontology_str = env["ontology_path"].read_text(encoding="utf-8")


    historical = []
    for hid in historical_ids:
        try:
            j, d, u, m = load_sample(base_dir, hid)
            historical.append({"json_data": j, "documentation": d, "unmapped": u, "mapping": m, "sid": hid})
        except FileNotFoundError:
            logger.warning(f"Historical Sample {hid} not found – skip")


    for sid in sample_ids:

        logger.info(f"\nProcessing {sid}")

        json_data, documentation, unmapped, ref = load_sample(base_dir, sid)
        filtered_hist = [h for h in historical if h["sid"] != sid]

        attributes = extract_leaf_paths(json_data)

        if isinstance(unmapped, dict):
            attributes = {a for a in attributes if a not in set(unmapped.keys())}

        sample_output_dir = base_output_dir / sid
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        results = {"attributes": {}, "evaluation": {}, "discussions": {}}

        mappers = {}
        attributes_list = list(attributes)
        total = len(attributes_list)
        iteration = 1
        for attr in attributes_list:
            logger.info(f"Mapping {attr} ({iteration}/{total})")
            iteration += 1
            mapper = AttributeMapper(
                attribute=attr,
                api_key=api_key,
                input_data={
                    "json_data": json_data,
                    "documentation": documentation,
                    "historical_references": filtered_hist,
                    "ontology": ontology_str
                },
                gpt_model=gptmodel
            )

            mapper.generate_mappings()
            mapper.validate_mappings()
            mapper.documentation_reasoning()
            mapper.historical_references_reasoning()
            mapper.example_value_reasoning()
            mapper.attribute_label_proximity_reasoning()
            mapper.select_final_mappings()

            mappers[attr] = mapper
            results["attributes"][attr] = {
                "state": mapper.state,
                "logs": mapper.logs
            }

        before_reasoning_snapshot = {
            attr: {
                "final_mapping": copy.deepcopy(mappers[attr].state["final_mapping"])
            }
            for attr in mappers
        }

        eval_json_before = {
            "prefix": extract_prefix_block(ontology_str),
            "mappings_candidates": {}
        }

        for attr, mapper in mappers.items():
            fm = mapper.state.get("final_mapping")
            if not fm:
                continue
            eval_json_before["mappings_candidates"][attr] = [{
                "candidate": fm["candidate"],
                "score": 1
            }]

        results["evaluation"]["before_reasoning"] = evaluate_top_k(
            k=1,
            reference_model=ref,
            to_evaluate=eval_json_before
        )

        reasoning = ReasoningAgent(api_key=api_key, gpt_model=gptmodel)

        attribute_map_for_reasoning = {
            attr: {
                "final": mappers[attr].state.get("final_mapping"),
                "matrix": mappers[attr].state.get("matrix")
            }
            for attr in mappers
        }

        discussions = reasoning.determine_discussions(attribute_map=attribute_map_for_reasoning, original_json_data=json_data, documentation=documentation, historical_references=filtered_hist)

        engine = DiscussionEngine(api_key=api_key, gpt_model=gptmodel)

        for disc_id, disc in discussions.items():
            discussions[disc_id] = engine.run_discussion(disc, mappers)

        results["discussions"] = discussions

        for attr, mapper in mappers.items():
            results["attributes"][attr]["state"]["final_mapping"] = mapper.state["final_mapping"]


        after_map_snapshot = {
            attr: results["attributes"][attr]
            for attr in results["attributes"]
        }

        results["reasoning_effect"] = compute_reasoning_effect(
            before_reasoning_snapshot,
            results["attributes"]
        )

        eval_json_after = {
            "prefix": extract_prefix_block(ontology_str),
            "mappings_candidates": {}
        }

        for attr, mapper in mappers.items():
            fm = mapper.state.get("final_mapping")
            if not fm:
                continue
            eval_json_after["mappings_candidates"][attr] = [{
                "candidate": fm["candidate"],
                "score": 1
            }]

        results["evaluation"]["after_reasoning"] = evaluate_top_k(
            k=1,
            reference_model=ref,
            to_evaluate=eval_json_after
        )

        output_file = sample_output_dir / f"{sid}_mapping_results.json"
        output_file.write_text(json.dumps(results, indent=4), encoding="utf-8")

        logger.info(f"Sample {sid} saved in:\n{output_file}")
    logger.info("\nPipeline finished.")

    if run_evaluation:
        try:
            run_evaluations(str(base_output_dir))
        except Exception as e:
            logger.error(f"Error while running evaluations: {e}")
def main(
    vcslam_path: str,
    sample_ids: list[str],
    historical_ids: list[str],
    export_path="export",
    evaluation_run = False
):

    run_pipeline(vcslam_path, sample_ids, historical_ids, export_path , evaluation_run)



