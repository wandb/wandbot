import json
import pathlib
from datetime import datetime

import wandb
import wandb.apis.reports as wr


def log_raw_counts(
    metadata: dict,
):
    data = {}
    for source, info in metadata.items():
        data[source] = info["num_documents"]
    wandb.run.log(data)
    return list(data.keys())


def log_processed_counts(
    metadata: dict,
):
    data = {}
    for source, info in metadata.items():
        data[source] = info["num_transformed_documents"]
    wandb.run.log(data)
    return list(data.keys())


def get_metadata_from_artifacts(raw_artifact, preprocessed_artifact, vectorstore_artifact):
    raw_artifact = wandb.run.use_artifact(raw_artifact, type="dataset")
    raw_artifact_dir = raw_artifact.download()
    preprocessed_artifact = wandb.run.use_artifact(preprocessed_artifact, type="dataset")
    preprocessed_artifact_dir = preprocessed_artifact.download()
    vectorstore_artifact = wandb.run.use_artifact(vectorstore_artifact, type="vectorstore")
    vectorstore_artifact_dir = vectorstore_artifact.download()

    raw_metadata_files = list(pathlib.Path(raw_artifact_dir).rglob("metadata.json"))
    preprocessed_metadata_files = list(pathlib.Path(preprocessed_artifact_dir).rglob("metadata.json"))
    vectorstore_metadata_files = list(pathlib.Path(vectorstore_artifact_dir).rglob("metadata.json"))

    raw_metadata = {}
    for metadata_file in raw_metadata_files:
        with metadata_file.open("r") as f:
            raw_metadata[metadata_file.parent.name] = json.load(f)
    preprocessed_metadata = {}
    for metadata_file in preprocessed_metadata_files:
        with metadata_file.open("r") as f:
            preprocessed_metadata[metadata_file.parent.name] = json.load(f)
    vectorstore_metadata = {}
    for metadata_file in vectorstore_metadata_files:
        with metadata_file.open("r") as f:
            vectorstore_metadata[metadata_file.parent.name] = json.load(f)
    return raw_metadata, preprocessed_metadata, vectorstore_metadata


def create_ingestion_report(
    project: str,
    entity: str,
    raw_artifact: str,
    preprocessed_artifact: str,
    vectorstore_artifact: str,
):
    report = wr.Report(
        project=project,
        entity=entity,
        title=f"Wandbot Data Ingestion Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        description=f"This report contains details of the data ingestion process "
        f"for the Wandbot run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    if wandb.run is None:
        run = wandb.init(project=project, entity=entity)
    else:
        run = wandb.run

    (
        raw_metadata,
        preprocessed_metadata,
        vectorstore_metadata,
    ) = get_metadata_from_artifacts(raw_artifact, preprocessed_artifact, vectorstore_artifact)

    raw_sources = log_raw_counts(raw_metadata)
    preprocessed_sources = log_processed_counts(preprocessed_metadata)
    pg_raw = wr.PanelGrid(
        runsets=[
            wr.Runset(run.entity, run.project, query=run.name),
        ],
        panels=[wr.BarPlot(title="Data Sources", metrics=raw_sources)],
    )
    pg_processed = wr.PanelGrid(
        runsets=[
            wr.Runset(run.entity, run.project, query=run.name),
        ],
        panels=[wr.BarPlot(title="Data Sources", metrics=raw_sources)],
    )

    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Raw Data Sources"),
        wr.UnorderedList(list(raw_metadata.keys())),
        pg_raw,
        wr.H1("Preprocessed Data Sources"),
        wr.UnorderedList(list(preprocessed_metadata.keys())),
        pg_processed,
        wr.H1("Raw Datasources Metadata"),
        wr.CodeBlock([json.dumps(dict(raw_metadata), indent=2)], language="json"),
        wr.H1("Preprocessed Datasources Metadata"),
        wr.CodeBlock([json.dumps(dict(preprocessed_metadata), indent=2)], language="json"),
        wr.H1("VectorsStore Artifact Summary"),
        wr.WeaveBlockArtifact(
            wandb.run.entity,
            wandb.run.project,
            vectorstore_artifact.split("/")[-1].split(":")[0],
            "overview",
        ),
    ]
    report.save()
    print(f"Report saved to {report.url}")


def main():
    project = "wandbot-dev"
    entity = "wandbot"
    raw_artifact = "wandbot/wandbot-dev/raw_dataset:latest"
    preprocessed_artifact = "wandbot/wandbot-dev/transformed_dataset:latest"
    vectorstore_artifact = "wandbot/wandbot-dev/vectorstores:latest"
    create_ingestion_report(project, entity, raw_artifact, preprocessed_artifact, vectorstore_artifact)


if __name__ == "__main__":
    main()
