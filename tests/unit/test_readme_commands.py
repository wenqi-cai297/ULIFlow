from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path
import re
import shlex
from string import Template
import sys

import pytest

from hoi_latent.training.stage1.phase1_warmup import load_phase1_training_config
from hoi_latent.training.stage2.rectified_flow import load_stage2_rectified_flow_training_config


ROOT = Path(__file__).resolve().parents[2]


def _readme_commands():
    variables = {}
    commands = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for block in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if not words:
                continue
            if len(words) == 1 and "=" in words[0]:
                key, value = words[0].split("=", 1)
                variables[key] = value
                continue
            words = [Template(word).substitute(variables) for word in words]
            for index, word in enumerate(words):
                if word.startswith("scripts/") and word.endswith(".py"):
                    commands.append((word, words[index + 1 :]))
                    break
    return variables, commands


def _parse_command(script, arguments, monkeypatch):
    path = ROOT / script
    assert path.is_file(), script
    spec = importlib.util.spec_from_file_location("readme_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [script, *arguments])
    spec.loader.exec_module(module)
    parse_args = argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def capture(parser, *args, **kwargs):
        raise ParsedArguments(parse_args(parser, *args, **kwargs))

    with monkeypatch.context() as context:
        context.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(ParsedArguments) as result:
            module.main()
    return result.value.args[0]


@pytest.mark.parametrize("script,arguments", _readme_commands()[1])
def test_readme_commands_parse_and_reference_existing_configs(script, arguments, monkeypatch):
    monkeypatch.chdir(ROOT)
    args = _parse_command(script, arguments, monkeypatch)
    for name in ("data_config", "train_config"):
        config = getattr(args, name, None)
        if config is not None:
            assert Path(config).is_file(), config


def test_readme_training_outputs_match_downstream_inputs(monkeypatch):
    monkeypatch.chdir(ROOT)
    variables, commands = _readme_commands()
    assert len(commands) == 6
    ae, whitening, post, flow, sample, evaluation = [
        _parse_command(script, arguments, monkeypatch) for script, arguments in commands
    ]
    _, _, ae_config = load_phase1_training_config(ae.data_config, ae.train_config)
    _, _, post_config = load_phase1_training_config(post.data_config, post.train_config)
    _, _, flow_config = load_stage2_rectified_flow_training_config(flow.data_config, flow.train_config)
    for config, stage, variable in (
        (ae_config, "stage1", "AE_RUN"),
        (post_config, "stage1", "POST_RUN"),
        (flow_config, "stage2", "STAGE2_RUN"),
    ):
        training = config["training"][stage]
        assert Path(training["output_root"]) / training["run_name"] == Path(variables[variable])
    assert Path(whitening.run_dir) == Path(post.source_run) == Path(variables["AE_RUN"])
    assert Path(flow.source_run) == sample.source_run == evaluation.source_run == Path(variables["POST_RUN"])
    assert sample.stage2_run == evaluation.stage2_run == Path(variables["STAGE2_RUN"])
    assert Path(whitening.out) == Path(post.whitening) == Path(flow.whitening) == sample.whitening == evaluation.whitening
