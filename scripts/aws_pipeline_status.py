#!/usr/bin/env python3
"""Show or stop the pipeline run in AWS (used by the pipeline workflow's status and stop actions).

    python scripts/aws_pipeline_status.py status --stack sepsis-dev --region ap-southeast-2
    python scripts/aws_pipeline_status.py stop   --stack sepsis-dev --region ap-southeast-2

Why: a real-data run takes hours and GitHub's AWS login lasts one hour, so the workflow starts the run and ends instead of
following it (run 5 died of an expired token while the run itself carried on in AWS). This answers the questions a person has
while it goes on: which step is it in, how long has it run, what is running on Fargate (that is what bills), what failed, and
what to do next. ``stop`` ends the run (Step Functions also stops the Fargate task it started) when it is not wanted.
"""

import argparse
import sys
from datetime import UTC, datetime

NEXT = {
    "WaitForApproval": "It is waiting for you: run **promote** with the model version from the email, then **approve**.",
}


def outputs(cloudformation, stack: str) -> dict:
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}


def latest(stepfunctions, machine: str) -> dict | None:
    items = stepfunctions.list_executions(stateMachineArn=machine, maxResults=1).get("executions", [])
    return items[0] if items else None


def current_step(stepfunctions, arn: str) -> str:
    events = stepfunctions.get_execution_history(executionArn=arn, reverseOrder=True, maxResults=50)["events"]
    for event in events:
        name = (event.get("stateEnteredEventDetails") or {}).get("name")
        if name:
            return name
    return "starting"


def failures(stepfunctions, arn: str) -> list[str]:
    found = []
    for event in stepfunctions.get_execution_history(executionArn=arn, maxResults=200)["events"]:
        details = event.get("taskFailedEventDetails") or event.get("executionFailedEventDetails")
        if details:
            found.append(f"{details.get('error', 'error')}: {str(details.get('cause', ''))[:300]}")
    return found[:5]


def hours(start: datetime, end: datetime | None = None, now: datetime | None = None) -> str:
    seconds = int(((end or now or datetime.now(UTC)) - start).total_seconds())
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def running_tasks(ecs, cluster: str, now: datetime | None = None) -> list[str]:
    arns = ecs.list_tasks(cluster=cluster, desiredStatus="RUNNING").get("taskArns", [])
    if not arns:
        return []
    lines = []
    for task in ecs.describe_tasks(cluster=cluster, tasks=arns)["tasks"]:
        family = task["taskDefinitionArn"].rsplit("/", 1)[-1].split(":")[0]
        started = task.get("startedAt")
        lines.append(f"{family}: {int(task['cpu']) / 1024:g} vCPU, {int(task['memory']) / 1024:g} GB"
                     + (f", running for {hours(started, now=now)}" if started else ""))
    return lines


def status(stepfunctions, ecs, machine: str, cluster: str, out=print, now: datetime | None = None) -> int:
    execution = latest(stepfunctions, machine)
    if not execution:
        out("No pipeline run yet. Start one with the pipeline workflow (action start).")
    else:
        state = execution["status"]
        step = current_step(stepfunctions, execution["executionArn"]) if state == "RUNNING" else ""
        out(f"Run {execution['name']}: {state}" + (f", in step {step}" if step else "")
            + f", started {execution['startDate']:%Y-%m-%d %H:%M} UTC, "
            + (f"ran {hours(execution['startDate'], execution.get('stopDate'))}" if execution.get("stopDate")
               else f"running for {hours(execution['startDate'], now=now)}"))
        if state == "RUNNING" and step in NEXT:
            out(NEXT[step])
        if state in ("FAILED", "TIMED_OUT", "ABORTED"):
            for line in failures(stepfunctions, execution["executionArn"]) or ["no failure details were recorded"]:
                out(f"  {line}")
    tasks = running_tasks(ecs, cluster, now)
    out("Running on Fargate now (this is what bills):" if tasks else "Nothing is running on the training cluster.")
    for line in tasks:
        out(f"  {line}")
    return 0


def stop(stepfunctions, machine: str, out=print) -> int:
    execution = latest(stepfunctions, machine)
    if not execution or execution["status"] != "RUNNING":
        out("No pipeline run is running; nothing to stop.")
        return 0
    stepfunctions.stop_execution(executionArn=execution["executionArn"], error="StoppedByUser",
                                 cause="stopped from the pipeline workflow")
    out(f"Stopped run {execution['name']}. Step Functions stops the Fargate task it started; check with action status.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["status", "stop"])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    found = outputs(session.client("cloudformation"), args.stack)
    if not found.get("PipelineArn"):
        print("FAIL: the stack has no pipeline", file=sys.stderr)
        return 1
    stepfunctions = session.client("stepfunctions")
    if args.action == "stop":
        return stop(stepfunctions, found["PipelineArn"])
    return status(stepfunctions, session.client("ecs"), found["PipelineArn"], found.get("EcsClusterArn", f"{args.stack}-training-cluster"))


if __name__ == "__main__":
    sys.exit(main())
