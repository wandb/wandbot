import sys
from unittest.mock import patch

import pytest

from wandbot.evaluation.eval_config import EvalConfig, get_eval_config


def test_eval_config_defaults():
    """Test default values of EvalConfig."""
    config = EvalConfig()
    assert config.lang == "en"
    assert config.eval_judge_model == "gpt-4-1106-preview"
    assert config.eval_judge_temperature == 0.1
    assert config.experiment_name == "wandbot-eval"
    assert config.evaluation_name == "wandbot-eval"
    assert config.n_trials == 3
    assert config.n_weave_parallelism == 20
    assert config.wandbot_url == "http://0.0.0.0:8000"
    assert config.wandb_entity == "wandbot"
    assert config.wandb_project == "wandbot-eval"
    assert config.debug is False
    assert config.n_debug_samples == 3

def test_eval_dataset_property():
    """Test eval_dataset property returns correct dataset based on language."""
    config = EvalConfig()
    
    # Test English dataset
    config.lang = "en"
    assert "wandbot-eval/object/wandbot_eval_data:" in config.eval_dataset
    
    # Test Japanese dataset
    config.lang = "ja"
    assert "wandbot-eval-jp/object/wandbot_eval_data_jp:" in config.eval_dataset

def test_get_eval_config_with_args():
    """Test get_eval_config with command line arguments."""
    test_args = [
        "--lang", "ja",
        "--eval_judge_model", "gpt-4",
        "--eval_judge_temperature", "0.2",
        "--experiment_name", "test-exp",
        "--debug", "true"
    ]
    
    with patch.object(sys, 'argv', ['script.py'] + test_args):
        config = get_eval_config()
        assert config.lang == "ja"
        assert config.eval_judge_model == "gpt-4"
        assert config.eval_judge_temperature == 0.2
        assert config.experiment_name == "test-exp"
        assert config.debug is True

def test_get_eval_config_invalid_args():
    """Test get_eval_config with invalid arguments."""
    test_args = [
        "--lang", "invalid",  # Invalid language
        "--eval_judge_temperature", "invalid"  # Invalid float
    ]
    
    with patch.object(sys, 'argv', ['script.py'] + test_args):
        with pytest.raises(SystemExit):
            get_eval_config()

def test_get_eval_config_type_validation():
    """Test type validation in get_eval_config."""
    test_cases = [
        (["--n_trials", "abc"], "n_trials should be an integer"),
        (["--debug", "not_bool"], "debug should be a boolean"),
        (["--eval_judge_temperature", "abc"], "eval_judge_temperature should be a float"),
    ]
    
    for args, error_msg in test_cases:
        with patch.object(sys, 'argv', ['script.py'] + args):
            with pytest.raises(SystemExit) as exc_info:
                get_eval_config()
            assert exc_info.value.code == 2  # Standard argparse error exit code 