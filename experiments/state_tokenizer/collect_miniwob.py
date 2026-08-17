"""Collect deterministic MiniWoB state records for the tokenizer pilot."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import gymnasium
import numpy as np
from PIL import Image

from miniwob.action import ActionTypes

from .common import (
    PILOT_TASKS,
    compact_dom,
    derive_probe_labels,
    normalize_scalar,
    split_for_episode,
    write_json,
)


SIDECAR_SCRIPT = r"""
return Array.from(document.querySelectorAll('*')).map((element) => {
  const tag = element.tagName.toLowerCase();
  const inputType = element instanceof HTMLInputElement ? element.type : '';
  const interactive = ['button', 'select', 'textarea', 'option'].includes(tag) ||
      tag === 'a' || tag === 'input' || element.hasAttribute('role');
  return {
    ref: Number(element.dataset.wob_ref || 0),
    tag: inputType ? `${tag}_${inputType}` : tag,
    role: element.getAttribute('role') || '',
    checked: ('checked' in element) ? Boolean(element.checked) : null,
    selected: ('selected' in element) ? Boolean(element.selected) : null,
    disabled: ('disabled' in element) ? Boolean(element.disabled) : null,
    interactive: Boolean(interactive),
  };
}).filter((item) => item.ref !== 0 || item.role || item.interactive);
"""


def _safe_sidecar(env) -> list[dict]:
    try:
        return list(env.unwrapped.instance.driver.execute_script(SIDECAR_SCRIPT))
    except Exception:
        logging.exception("Could not collect DOM sidecar; continuing with observation labels")
        return []


INTERACTIVE_TAGS = {
    "input_checkbox", "input_radio", "input_text", "input_password", "textarea",
    "button", "a", "label", "input_button", "input_submit", "option", "select",
}
TYPING_TAGS = {"input_text", "input_password", "textarea"}


def _random_action(env, observation: dict, rng: np.random.Generator):
    """Uniformly sample one legal action over the page's interactive elements.

    The scripted policy is a deterministic priority cascade, so ``u_t`` is very
    nearly a function of ``y_t`` -- which makes "does the action carry
    information beyond the state?" unanswerable no matter how much data is
    collected. Sampling uniformly over legal actions breaks that dependence.
    """
    elements = [
        element for element in observation.get("dom_elements", ())
        if element.get("tag") in INTERACTIVE_TAGS
    ]
    if not elements:
        return None, None
    element = elements[int(rng.integers(len(elements)))]
    ref = int(normalize_scalar(element["ref"]))
    if element.get("tag") in TYPING_TAGS:
        text = f"state-{int(rng.integers(0, 100000)):05d}"
        return (
            env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=ref, text=text
            ),
            {"type": "FOCUS_ELEMENT_AND_TYPE_TEXT", "ref": ref, "text": text,
             "tag": element.get("tag"), "policy": "random"},
        )
    return (
        env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref),
        {"type": "CLICK_ELEMENT", "ref": ref, "text": None,
         "tag": element.get("tag"), "policy": "random"},
    )


def _make_action(env, observation: dict, rng: np.random.Generator, step: int):
    elements = list(observation.get("dom_elements", ()))
    if not elements:
        return None, None

    def shuffled(candidates):
        candidates = list(candidates)
        rng.shuffle(candidates)
        return candidates

    unchecked = [
        element for element in elements
        if element.get("tag") in {"input_checkbox", "input_radio"}
        and not bool(normalize_scalar(element.get("value")))
    ]
    if unchecked:
        ref = int(normalize_scalar(unchecked[0]["ref"]))
        return (
            env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref),
            {"type": "CLICK_ELEMENT", "ref": ref, "text": None,
             "tag": unchecked[0].get("tag"), "policy": "scripted"},
        )

    textboxes = shuffled(
        element for element in elements
        if element.get("tag") in {"input_text", "textarea", "input_password"}
    )
    if textboxes and step == 0:
        ref = int(normalize_scalar(textboxes[0]["ref"]))
        text = f"state-{int(rng.integers(0, 100000)):05d}"
        return (
            env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=ref, text=text
            ),
            {"type": "FOCUS_ELEMENT_AND_TYPE_TEXT", "ref": ref, "text": text,
             "tag": textboxes[0].get("tag"), "policy": "scripted"},
        )

    options = shuffled(element for element in elements if element.get("tag") == "option")
    if options:
        ref = int(normalize_scalar(options[0]["ref"]))
        return (
            env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref),
            {"type": "CLICK_ELEMENT", "ref": ref, "text": None,
             "tag": options[0].get("tag"), "policy": "scripted"},
        )

    clickable_tags = {"button", "a", "label", "input_button", "input_submit", "div"}
    clickable = shuffled(
        element for element in elements
        if element.get("tag") in clickable_tags
        and (element.get("text") or element.get("tag") != "div")
    )
    if clickable:
        ref = int(normalize_scalar(clickable[0]["ref"]))
        return (
            env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref),
            {"type": "CLICK_ELEMENT", "ref": ref, "text": None,
             "tag": clickable[0].get("tag"), "policy": "scripted"},
        )
    return None, None


def _choose_action(env, observation, rng, step, random_prob: float):
    """Scripted cascade with probability ``1 - random_prob``, else a random action.

    Pure exploration would maximise action entropy but terminate multi-step tasks
    early (a wrong click ends form-sequence), starving exactly the deep states the
    history question needs. Mixing keeps trajectory depth while still making half
    the transitions interventional.
    """
    if random_prob > 0.0 and float(rng.random()) < random_prob:
        action, described = _random_action(env, observation, rng)
        if action is not None:
            return action, described
    return _make_action(env, observation, rng, step)


def _save_state(
    output: Path,
    task: str,
    task_index: int,
    episode_index: int,
    step: int,
    observation: dict,
    sidecar: list[dict],
) -> dict | None:
    dom_elements = observation.get("dom_elements", ())
    screenshot = np.asarray(observation.get("screenshot"))
    if not dom_elements or screenshot.ndim != 3 or screenshot.size == 0:
        return None
    state_id = f"task{task_index:02d}-ep{episode_index:06d}-t{step:02d}"
    image_rel = Path("images") / f"{state_id}.png"
    image_path = output / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(screenshot.astype(np.uint8), mode="RGB").save(image_path, optimize=True)
    fields = [
        [str(key), str(value)] for key, value in observation.get("fields", ())
    ]
    return {
        "state_id": state_id,
        "task": task,
        "episode_id": f"task{task_index:02d}-ep{episode_index:06d}",
        "episode_index": episode_index,
        "step": step,
        "split": split_for_episode(episode_index),
        "instruction": str(observation.get("utterance", "")),
        "fields": fields,
        "dom": compact_dom(dom_elements),
        "screenshot": image_rel.as_posix(),
        "probe": derive_probe_labels(observation, sidecar),
    }


def collect(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.tasks or PILOT_TASKS)
    assigned = [
        (index, task) for index, task in enumerate(tasks)
        if index % args.num_workers == args.worker_id
    ]
    per_task = int(np.ceil(args.target_states / len(tasks)))
    records_path = output / f"records-worker{args.worker_id:02d}.jsonl"
    counts = {"train": 0, "validation": 0, "test": 0}
    task_counts: dict[str, int] = {}
    total = 0
    with records_path.open("w", encoding="utf-8") as handle:
        for task_index, task in assigned:
            env = gymnasium.make(
                task,
                action_space_config="all_supported",
                wait_ms=args.wait_ms,
                refresh_freq=args.refresh_freq,
            )
            task_total = 0
            episode_index = args.worker_id
            try:
                while task_total < per_task:
                    seed = args.seed + task_index * 1_000_000 + episode_index
                    episode: list[dict] = []
                    try:
                        observation, _ = env.reset(
                            seed=seed, options={"record_screenshots": True, "data_mode": "train"}
                        )
                        rng = np.random.default_rng(seed)
                        # The action is only known after its state has been built,
                        # so the episode is buffered and flushed once complete:
                        # record[t]["action"] is what led to record[t+1].
                        for step in range(args.max_steps + 1):
                            record = _save_state(
                                output, task, task_index, episode_index, step,
                                observation, _safe_sidecar(env),
                            )
                            if record is not None:
                                record["action"] = None
                                episode.append(record)
                                split = record["split"]
                                counts[split] += 1
                                task_total += 1
                                total += 1
                                if total % args.log_every == 0:
                                    logging.info(
                                        "worker=%d records=%d current_task=%s task_records=%d/%d",
                                        args.worker_id, total, task, task_total, per_task,
                                    )
                                if task_total >= per_task:
                                    break
                            action, described = _choose_action(
                                env, observation, rng, step, args.random_action_prob
                            )
                            if action is None:
                                break
                            if record is not None:
                                record["action"] = described
                            observation, _, terminated, truncated, _ = env.step(action)
                            if terminated or truncated:
                                break
                    except Exception:
                        logging.exception("Episode failed: task=%s episode=%d", task, episode_index)
                    finally:
                        for item in episode:
                            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()
                    episode_index += args.num_workers
            finally:
                env.close()
            task_counts[task] = task_total
    summary = {
        "worker_id": args.worker_id,
        "random_action_prob": args.random_action_prob,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "records": total,
        "split_counts": counts,
        "task_counts": task_counts,
        "records_path": str(records_path),
    }
    write_json(output / f"collect-worker{args.worker_id:02d}.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-states", type=int, default=10_000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="probability of replacing the scripted action with a "
                             "uniform draw over the page's legal actions. 0 "
                             "reproduces the v4/v5 collection exactly; the scripted "
                             "policy is near-deterministic given the observation, so "
                             "a positive value is what makes the action carry "
                             "information beyond the state")
    parser.add_argument("--wait-ms", type=float, default=150.0)
    parser.add_argument("--refresh-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    if not 0.0 <= args.random_action_prob <= 1.0:
        parser.error("random-action-prob must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(collect(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
