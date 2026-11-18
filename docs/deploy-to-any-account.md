# Deploy to any AWS account

Three steps, run once per account. They work the same for dev, staging and prod,
in any region. Nothing in the repository is specific to one account.

## Before you start

- An AWS account and a shell with admin credentials for it (for example
  CloudShell, or a profile in `~/.aws`). This is used only for the one-time setup.
- AWS CLI v2 and Docker (for the image build).
- The GitHub CLI (`gh`), logged in, if you want the variables set automatically.
  Otherwise the script prints the commands to run.
- The region you want. The SageMaker MLflow App must be available in it; check
  the list in `docs/aws-serverless.md`. Both Singapore (`ap-southeast-1`) and Sydney (`ap-southeast-2`) are supported and checked; the examples use Sydney, so replace it with `ap-southeast-1` for Singapore.

## 1. Connect GitHub to the account (once per account and region)

```bash
export AWS_REGION=ap-southeast-2
export GITHUB_REPO=owner/SUTD-PML-team2       # the repository that will run the deploy workflow
bash scripts/bootstrap_github_oidc.sh
```

This creates the GitHub OIDC identity provider (only if the account has none),
and one deploy role each for `dev`, `staging` and `prod`. It sets the GitHub
variables `AWS_ACCOUNT_ID`, `AWS_REGION` (the default region) and `AWS_DEPLOY_ROLE_ARN` for each
environment. No access keys are created. The roles are named `github-deploy-<environment>-<region>`.

**To deploy to both Singapore and Sydney, run the script once per region** (for example with
`AWS_REGION=ap-southeast-1`, then with `AWS_REGION=ap-southeast-2`). The second run reuses the OIDC
provider and adds the roles for the second region. The deploy workflow then finds the right role for
the region you choose.

**Different account per environment:** run the script with credentials for each account in turn.
The variables are set per GitHub environment, so `dev` can deploy to one account and `prod` to another.
To target an account by hand, set `AWS_ACCOUNT_ID` for that GitHub environment.

Then, in GitHub, under Settings → Environments, add required reviewers to
`staging` and `prod`.

The deploy roles start with `AdministratorAccess`, because the first deployment
creates IAM roles. After the first successful deploy, replace it with a narrower
policy: pass `DEPLOY_POLICY_ARN=<arn>` to the script, or edit the role in IAM.

## 2. Create the stack and the first model (from the repository)

Option A, from GitHub (recommended):

1. Actions → **deploy** → Run workflow. Choose the **environment** (`dev`, `staging` or `prod`) and the
   **region** (`ap-southeast-1`, `ap-southeast-2`, or "environment default"). `staging` and `prod` pause
   for the reviewers you set on those GitHub environments. The run summary names the region and role used.
   The first run fails at `publish` if no model has been promoted in this
   account yet. That is expected: step 3 below creates the model first.

Option B, from your shell (for a first look, or without GitHub):

```bash
export AWS_REGION=ap-southeast-2 ENVIRONMENT=dev BUDGET_EMAIL=you@example.com
bash scripts/aws_deploy.sh stack          # S3, KMS, ECR, MLflow App, roles, pipeline, alarms
bash scripts/aws_deploy.sh images         # builds and pushes the three images
```

## 3. Train, review and promote a model

The training pipeline is the production route: it trains, checks the promotion
gates, waits for a named approver, and publishes. Its first use in an account:

```bash
bash scripts/aws_deploy.sh data           # after data-use approval; synthetic is fine for a dev rehearsal
bash scripts/aws_deploy.sh pipeline       # starts training and waits for approval
bash scripts/aws_deploy.sh approve <token> # the token arrives by email; the approver is not the trainer
```

For a dev rehearsal on synthetic data, set `ALLOW_SYNTHETIC=1`. Then run the
deploy workflow again (or `bash scripts/aws_deploy.sh publish && serve`) to put
the model behind the scoring API.

## Checking it worked

```bash
bash scripts/aws_deploy.sh call GET /health     # the bundle hash should match the published model
bash scripts/aws_deploy.sh outputs              # every stack output, including the API URL
```

## Deploying without GitHub (when the organization blocks OIDC)

Some accounts (for example a university or company account in an AWS Organization) have a service
control policy (SCP) that denies `iam:ListOpenIDConnectProviders` or `iam:CreateOpenIDConnectProvider`.
The error names `an explicit deny in a service control policy`. Nothing in the account can override an
SCP; only the organization's administrator can. Two ways forward:

1. **Ask the administrator** to create the provider (URL `https://token.actions.githubusercontent.com`,
   audience `sts.amazonaws.com`) and to allow IAM role creation, or to lift the deny for your role. Then
   run the bootstrap with `CREATE_OIDC_PROVIDER=false` (it skips the lookup the SCP blocks).
2. **Deploy from a shell that already has access**, such as CloudShell or an EC2 instance. This needs no
   OIDC and no stored keys; the shell's own role is used. Use the same scripts the workflow runs:

```bash
export AWS_REGION=ap-southeast-2 ENVIRONMENT=dev BUDGET_EMAIL=you@example.com
bash scripts/aws_deploy.sh stack && bash scripts/aws_deploy.sh images
bash scripts/aws_deploy.sh destroy      # plan of the teardown; CONFIRM_DESTROY=<stack> to delete
```

The stack itself also creates IAM roles, so if the SCP blocks those as well, the deployment needs the
administrator's help in either route. Do not work around an SCP with long-lived access keys.

## Tearing it down

From GitHub: Actions → **teardown** → Run workflow. Choose the environment and region, then a scope:

| Scope | What it removes |
|---|---|
| plan only (default) | Nothing. It lists what would be removed. |
| stack only | The CloudFormation stack. Models, data, keys, images and logs stay. |
| everything | The stack plus the retained resources: both S3 buckets with every object version (all published models and training outputs), the ECR repositories, the DynamoDB table, the log groups, and the template bucket. The KMS key is scheduled for deletion in 7 days, the shortest AWS allows. This cannot be undone. |

For anything but "plan only", type the stack name (`sepsis-dev`, `sepsis-staging`, `sepsis-prod`) in the
confirm box. `staging` and `prod` also wait for their required reviewers. From a shell:

```bash
export AWS_REGION=ap-southeast-2 ENVIRONMENT=dev
bash scripts/aws_deploy.sh destroy                       # plan only
CONFIRM_DESTROY=sepsis-dev bash scripts/aws_deploy.sh destroy   # delete everything (prod also needs CONFIRM_PROD=yes)
bash scripts/aws_deploy.sh down                          # stack only
```

The GitHub connection itself (the OIDC provider and deploy roles) is separate: delete the
`sepsis-github-oidc` stack in each region to remove it. Run the teardown for a region before deleting
that region's bootstrap stack, because the workflow uses the deploy role.

## Moving to another account

Repeat step 1 in the new account, with its region. Then run the deploy workflow
with `ENVIRONMENT` set to that account's environment. The code and the templates
do not change between accounts.

## What the setup creates

| Resource | Why |
|---|---|
| GitHub OIDC provider (IAM) | Lets GitHub Actions obtain short-lived credentials without stored keys |
| Three deploy roles | One per GitHub environment; the trust policy accepts only jobs in that environment |
| Stack `sepsis-github-oidc` | Holds the above; delete it to remove the connection (after tearing down the deployment) |

## Troubleshooting

- **"has neither AWS_ACCOUNT_ID nor AWS_DEPLOY_ROLE_ARN"** in the workflow: run step 1
  for that account, or set the variables for that GitHub environment.
- **"AWS_DEPLOY_ROLE_ARN belongs to another region"**: you chose a region the bootstrap has not been run
  for. Run step 1 with that `AWS_REGION`, or set `AWS_ACCOUNT_ID` for the environment.
- **Role not found for a region**: step 1 has not been run in that region of that account.
- **"set AWS_REGION"** from a script: export `AWS_REGION` in the shell first.
- **"this environment sets a monthly budget; set BUDGET_EMAIL"**: set `BUDGET_EMAIL`,
  or set `MonthlyBudgetUsd` to 0 in `infra/params/<env>.json`.
- **Role trust errors (`Not authorized to perform sts:AssumeRoleWithWebIdentity`)**:
  the repository name in `GITHUB_REPO` must match the repository running the workflow,
  and the job must run in the matching GitHub environment.
- **MLflow App not found in the region**: choose a region from the list in
  `docs/aws-serverless.md`.
