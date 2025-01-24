import os
import pytest
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

from wandbot.evaluation.eval.experimental.pure_evaluator import PureCorrectnessEvaluator

# Load environment variables from .env file
dot_env_path = os.path.join(os.path.dirname(__file__), '../../../../../.env')
load_dotenv(dotenv_path=dot_env_path, override=True)

# Test data
TEST_CASES = [
    # Perfect matches
    {
        "query": "How do I create a new wandb project?",
        "response": "To create a new W&B project, you can initialize it in your code using wandb.init(project='your-project-name'). This will create the project if it doesn't exist.",
        "reference": "You can create a new W&B project by using wandb.init(project='your-project-name') in your code. If the project doesn't exist, it will be created automatically.",
        "reference_notes": "This is the standard way to create projects programmatically",
        "contexts": ["To create a new project, use wandb.init() with the project parameter"],
        "expected_passing": True,
    },
    {
        "query": "How do I log metrics in wandb?",
        "response": "To log metrics in W&B, use wandb.log({'metric_name': value}). For example: wandb.log({'loss': 0.2, 'accuracy': 0.95}). This will send the metrics to your W&B dashboard.",
        "reference": "You can log metrics using wandb.log() with a dictionary of metric names and values. Example: wandb.log({'loss': 0.2, 'accuracy': 0.95})",
        "reference_notes": "This shows both the method and practical example",
        "contexts": ["Log metrics using wandb.log() with a dictionary"],
        "expected_passing": True,
    },
    # Partially correct answers
    {
        "query": "How do I delete a run?",
        "response": "You can delete a run by clicking the delete button.",
        "reference": "To delete a run, you can either use the web interface by clicking the delete button on the run page, or programmatically using the wandb.Api().delete_run(path) method.",
        "reference_notes": "Both UI and API methods are valid ways to delete runs",
        "contexts": ["Runs can be deleted via UI or API"],
        "expected_passing": False,
    },
    {
        "query": "How do I set up wandb in a Jupyter notebook?",
        "response": "Just import wandb and call wandb.init()",
        "reference": "To use W&B in a Jupyter notebook: 1) Import wandb 2) Call wandb.init(project='project-name') 3) Use magic command %wandb to display the W&B panel in your notebook",
        "reference_notes": "Complete setup includes magic command for notebook integration",
        "contexts": ["Jupyter setup requires import, init, and magic command"],
        "expected_passing": False,
    },
    # Incorrect answers
    {
        "query": "How do I save a model checkpoint?",
        "response": "Use model.save() to save your checkpoint",
        "reference": "To save a model checkpoint with W&B, use wandb.save('path/to/model.h5') or log it as an artifact using wandb.log_artifact('path/to/model.h5', type='model')",
        "reference_notes": "Shows both simple save and artifact methods",
        "contexts": ["Save models using wandb.save() or wandb.log_artifact()"],
        "expected_passing": False,
    },
    # Technical details correct
    {
        "query": "What's the difference between wandb.init() and wandb.login()?",
        "response": "wandb.login() authenticates your account with an API key, while wandb.init() starts a new run and creates a working directory for your experiment.",
        "reference": "wandb.login() is used for authentication with your W&B account using an API key. wandb.init() initializes a new run and sets up the working directory for your experiment.",
        "reference_notes": "Clear distinction between authentication and run initialization",
        "contexts": ["Login handles auth, init handles run creation"],
        "expected_passing": True,
    },
    # Missing context but correct
    {
        "query": "How do I visualize my model architecture?",
        "response": "You can log your model architecture using wandb.watch(model), which will track gradients and parameters during training.",
        "reference": "Use wandb.watch(model) to automatically log model architecture and track gradients and parameters during training.",
        "reference_notes": "Covers automatic model tracking",
        "contexts": ["wandb.watch() enables model architecture tracking"],
        "expected_passing": True,
    },
    # Extra helpful information
    {
        "query": "How do I compare runs?",
        "response": "You can compare runs in the W&B UI by selecting multiple runs and clicking 'Compare'. You can also use parallel coordinates plots and scatter plots to visualize relationships between hyperparameters and metrics.",
        "reference": "Select multiple runs in the W&B UI and click 'Compare' to analyze them together.",
        "reference_notes": "Basic comparison functionality",
        "contexts": ["Compare runs using the UI"],
        "expected_passing": True,
    },
    # Missing critical information
    {
        "query": "How do I handle sensitive data in wandb?",
        "response": "Just don't log the sensitive data.",
        "reference": "To handle sensitive data: 1) Use wandb.init(anonymous='allow') for anonymous runs 2) Set WANDB_API_KEY as an environment variable 3) Use wandb.config to exclude sensitive parameters 4) Consider self-hosting W&B server for complete data control",
        "reference_notes": "Comprehensive security measures",
        "contexts": ["Security features include anonymous runs and self-hosting"],
        "expected_passing": False,
    },
    # Technically wrong
    {
        "query": "How do I resume a crashed run?",
        "response": "Create a new run with the same name to continue.",
        "reference": "To resume a crashed run: 1) Initialize with the same run_id: wandb.init(id='previous_run_id', resume='allow') 2) Load the latest checkpoint 3) Continue training",
        "reference_notes": "Proper resume procedure using run_id",
        "contexts": ["Resume runs using run_id and resume flag"],
        "expected_passing": False,
    },
    # Outdated information
    {
        "query": "How do I use sweeps for hyperparameter optimization?",
        "response": "Use wandb.agent() to start a sweep.",
        "reference": "To use sweeps: 1) Define sweep configuration in YAML or dict 2) Initialize with wandb.sweep(sweep_config) 3) Run wandb.agent(sweep_id) to start optimization 4) Monitor results in the W&B dashboard",
        "reference_notes": "Complete sweep setup process",
        "contexts": ["Sweeps require config, initialization, and agent"],
        "expected_passing": False,
    },
    # Correct but could be more detailed
    {
        "query": "How do I log images in wandb?",
        "response": "Use wandb.log({'image_name': wandb.Image(image_data)}) to log images.",
        "reference": "Log images using wandb.log({'image_name': wandb.Image(image_data)}). You can log numpy arrays, PIL Images, or file paths. Optional caption and masks can be included.",
        "reference_notes": "Basic image logging with additional options",
        "contexts": ["Image logging supports multiple formats"],
        "expected_passing": True,
    },
    # Complex technical answer
    {
        "query": "How do I implement custom visualizations?",
        "response": "Use wandb.plot.line() for custom line plots, wandb.plot.scatter() for scatter plots, and wandb.plot.bar() for bar charts. You can also create custom plots using plotly and log them with wandb.log().",
        "reference": "Create custom visualizations using wandb.plot.* methods or by logging plotly/matplotlib figures directly with wandb.log()",
        "reference_notes": "Multiple visualization options",
        "contexts": ["Custom plots via built-in methods or external libraries"],
        "expected_passing": True,
    },
    # Misleading information
    {
        "query": "What's the wandb sync timeout?",
        "response": "wandb sync never times out.",
        "reference": "The default wandb sync timeout is 2 minutes. You can modify it using WANDB_TIMEOUT environment variable or timeout parameter in wandb.init()",
        "reference_notes": "Timeout behavior and configuration",
        "contexts": ["Sync timeout defaults to 2 minutes"],
        "expected_passing": False,
    },
    # Correct with examples
    {
        "query": "How do I group runs together?",
        "response": "Use the 'group' parameter in wandb.init(group='experiment_name') to group related runs. You can also use tags=['tag1', 'tag2'] for additional organization.",
        "reference": "Group runs using wandb.init(group='experiment_name'). Additionally, use tags for flexible organization.",
        "reference_notes": "Grouping and tagging functionality",
        "contexts": ["Group runs with group parameter and tags"],
        "expected_passing": True,
    },
    # Missing steps
    {
        "query": "How do I set up wandb in PyTorch Lightning?",
        "response": "Add WandbLogger to your trainer.",
        "reference": "To use W&B with PyTorch Lightning: 1) Import WandbLogger 2) Create logger: logger = WandbLogger(project='project_name') 3) Add to trainer: trainer = Trainer(logger=logger) 4) Optional: use log_hyperparams() for config",
        "reference_notes": "Complete Lightning integration steps",
        "contexts": ["Lightning integration requires WandbLogger setup"],
        "expected_passing": False,
    },
    # Correct but brief
    {
        "query": "How do I disable wandb logging temporarily?",
        "response": "Set WANDB_MODE=disabled or use wandb.init(mode='disabled')",
        "reference": "Disable W&B logging by setting WANDB_MODE=disabled environment variable or using wandb.init(mode='disabled')",
        "reference_notes": "Two methods to disable logging",
        "contexts": ["Disable via environment variable or init parameter"],
        "expected_passing": True,
    },
    # Wrong API usage
    {
        "query": "How do I download artifacts?",
        "response": "Use artifact.get() to download artifacts",
        "reference": "Download artifacts using: 1) api = wandb.Api() 2) artifact = api.artifact('entity/project/artifact:version') 3) artifact_dir = artifact.download()",
        "reference_notes": "Correct artifact download procedure",
        "contexts": ["Artifact download requires API and download() call"],
        "expected_passing": False,
    },
    # Correct with best practices
    {
        "query": "How do I organize my wandb projects?",
        "response": "Organize projects by: 1) Using descriptive project names 2) Grouping related runs 3) Using tags for filtering 4) Adding descriptions to runs 5) Using notes for important details",
        "reference": "Best practices for project organization: Use descriptive names, group related runs, apply tags, add descriptions, and maintain notes",
        "reference_notes": "Project organization best practices",
        "contexts": ["Project organization guidelines"],
        "expected_passing": True,
    },
    # Security issue
    {
        "query": "How do I share my API key with my team?",
        "response": "Add your API key to the shared code repository.",
        "reference": "Never share API keys in code. Instead: 1) Each team member should get their own key from their W&B account 2) Use environment variables 3) For CI/CD, use secure secrets management",
        "reference_notes": "API key security best practices",
        "contexts": ["API key security guidelines"],
        "expected_passing": False,
    }
]

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a technical support chatbot.

You are given the following information:
- a user query,
- the documentation used to generate the answer
- a reference answer
- the reason why the reference answer is correct, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
- Consider whether the answer addresses all aspects of the question.
- The generated answer must provide only correct information according to the documentation.
- Compare the generated answer to the reference answer for completeness and correctness.
- Output a score and a decision that represents a holistic evaluation of the generated answer.

IMPORTANT: You must return ONLY a JSON object in the following format. Do not include ANY text before or after the JSON object:
{
    "reason": "Your explanation here",
    "score": A number between 1 and 3,
    "decision": Either "correct" or "incorrect"
}

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not correct in comparison to the reference, you should give a score of 1.
- If the generated answer is correct in comparison to the reference but contains mistakes, you should give a score of 2.
- If the generated answer is correct in comparison to the reference and completely answer's the user's query, you should give a score of 3.

Example Response 1:
{
    "reason": "The generated answer has the exact details as the reference answer and completely answer's the user's query.",
    "score": 3,
    "decision": "correct"
}

Example Response 2:
{
    "reason": "The generated answer doesn't match the reference answer, and deviates from the documentation provided",
    "score": 1,
    "decision": "incorrect"
}

Example Response 3:
{
    "reason": "The generated answer follows the same steps as the reference answer. However, it includes assumptions about methods that are not mentioned in the documentation.",
    "score": 2,
    "decision": "incorrect"
}"""

@pytest.fixture
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return AsyncOpenAI(api_key=api_key)

@pytest.fixture
def pure_evaluator(openai_client):
    return PureCorrectnessEvaluator(
        openai_client=openai_client,
        model="gpt-4-1106-preview",
        temperature=0.1
    )

@pytest.mark.asyncio
async def test_pure_evaluator_basic(pure_evaluator):
    """Test that the pure evaluator works as expected."""
    
    for test_case in TEST_CASES:
        result = await pure_evaluator.evaluate(
            query=test_case["query"],
            response=test_case["response"],
            reference=test_case["reference"],
            contexts=test_case["contexts"],
            reference_notes=test_case["reference_notes"]
        )
        
        # Verify basic properties
        assert result.query == test_case["query"]
        assert result.response == test_case["response"]
        assert result.passing is not None
        assert isinstance(result.score, (int, float))
        assert result.feedback is not None
        
        # Check score range
        assert 1 <= result.score <= 3, f"Score {result.score} outside valid range 1-3"
        
        # Check against expected passing
        assert result.passing == test_case["expected_passing"], \
            f"Result doesn't match expected for query: {test_case['query']}"
            
        print(f"\nResults for query: {test_case['query']}")
        print(f"Score: {result.score}")
        print(f"Passing: {result.passing}")
        print(f"Feedback: {result.feedback}")

@pytest.mark.asyncio
async def test_pure_evaluator_error_handling(pure_evaluator):
    """Test error handling in the pure evaluator."""
    
    # Test missing required parameters
    with pytest.raises(ValueError):
        await pure_evaluator.evaluate(
            query="test",
            response=None,
            reference="test"
        )
    
    with pytest.raises(ValueError):
        await pure_evaluator.evaluate(
            query=None,
            response="test",
            reference="test"
        )
    
    with pytest.raises(ValueError):
        await pure_evaluator.evaluate(
            query="test",
            response="test",
            reference=None
        )

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 