import json
from datetime import datetime

import wandb
import wandb.apis.reports as wr
from wandbot.ingestion.datastore import VectorIndex


def log_datasource_counts(metadata: dict, wandb_run: wandb.sdk.wandb_run.Run):
    data = {}
    for source, info in metadata.items():
        data[source] = info["num_docs"]
    wandb_run.log(data)
    return list(data.keys())


def create_ingestion_report(
    vectorindex: VectorIndex, raw_dataset_artifact: wandb.Artifact
):
    config = vectorindex.config
    report = wr.Report(
        project=config.wandb_project,
        entity=config.wandb_entity,
        title=f"Wandbot Data Ingestion Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        description=f"This report contains details of the data ingestion process "
        f"for the Wandbot run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    docstore_metadata = vectorindex.datastore.ref_doc_info
    saved_artifact = vectorindex.saved_artifact.wait()
    wandb_run = vectorindex.wandb_run
    metrics = log_datasource_counts(docstore_metadata["metadata"], wandb_run)

    pg = wr.PanelGrid(
        runsets=[
            wr.Runset(wandb_run.entity, wandb_run.project, query=wandb_run.name),
        ],
        panels=[wr.BarPlot(title="Data Sources", metrics=metrics)],
    )

    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Data Sources"),
        wr.UnorderedList(list(docstore_metadata["metadata"].keys())),
        pg,
        wr.H1("Datastore Metadata"),
        wr.CodeBlock([json.dumps(dict(docstore_metadata), indent=2)], language="json"),
        wr.H1("Artifact Summary"),
        wr.WeaveBlockArtifact(
            vectorindex.wandb_run.entity,
            vectorindex.wandb_run.project,
            saved_artifact.name.split(":")[0],
            "overview",
        ),
        wr.H1("Raw Dataset Summary"),
        wr.WeaveBlockArtifact(
            entity=vectorindex.wandb_run.entity,
            project=vectorindex.wandb_run.project,
            artifact=raw_dataset_artifact.name.split(":")[0],
            tab="overview",
        ),
    ]
    report.save()
    print(f"Report saved to {report.url}")
