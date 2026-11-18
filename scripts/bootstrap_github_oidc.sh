#!/usr/bin/env bash
# Prepare any AWS account for GitHub Actions deployments (run once per account and region).
#
#   AWS_REGION=ap-southeast-2 GITHUB_REPO=owner/name bash scripts/bootstrap_github_oidc.sh
#
# Creates the OIDC provider (only if the account lacks one) and one deploy role per
# GitHub environment (dev, staging, prod) through infra/bootstrap/github-oidc.yaml.
# Then, if the GitHub CLI is installed and logged in, sets the repository variables
# AWS_ACCOUNT_ID, AWS_REGION (the default region) and AWS_DEPLOY_ROLE_ARN for each environment.
# Otherwise it prints the commands. Run it once per region: the deploy roles are named
# github-deploy-<environment>-<region>, and the deploy workflow picks the role for the region
# chosen when the workflow is started.
#
# Needs AWS credentials for the target account in your shell (for example an admin
# profile, or CloudShell). It creates no access keys and stores no secrets.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${AWS_REGION:?set AWS_REGION, ap-southeast-1 (Singapore) or ap-southeast-2 (Sydney)}"
: "${GITHUB_REPO:?set GITHUB_REPO, for example owner/name}"
STACK="${BOOTSTRAP_STACK:-sepsis-github-oidc}"
fail() { echo "FAIL: $*" >&2; exit 1; }

command -v aws >/dev/null || fail "AWS CLI v2 is not installed"
account="$(aws sts get-caller-identity --query Account --output text)" || fail "no AWS credentials in this shell"
echo "Account $account, region $AWS_REGION, repository $GITHUB_REPO"

# The OIDC provider is one per account. CREATE_OIDC_PROVIDER=false skips the lookup and the creation:
# use it when an administrator already created the provider, or when a service control policy
# (SCP) of your organization denies iam:ListOpenIDConnectProviders to you.
create_provider="${CREATE_OIDC_PROVIDER:-auto}"
if [ "$create_provider" = auto ]; then
  if ! provider_exists=$(aws iam list-open-id-connect-providers --query \
    "OpenIDConnectProviderList[?contains(Arn, 'token.actions.githubusercontent.com')] | length(@)" --output text); then
    cat >&2 <<MSG

FAIL: this account does not let you look up OIDC providers. If the message above names a service control
policy (SCP), your organization blocks it and only its administrator can change that. Ask them to either:
  1. create the GitHub OIDC provider (https://token.actions.githubusercontent.com, audience sts.amazonaws.com)
     and allow you to create IAM roles, then run this script again with CREATE_OIDC_PROVIDER=false; or
  2. allow iam:CreateOpenIDConnectProvider and iam:ListOpenIDConnectProviders for your role.
Without OIDC you can still deploy from this shell: see "Deploying without GitHub" in docs/deploy-to-any-account.md.
MSG
    exit 1
  fi
  create_provider=true
  [ "$provider_exists" != "0" ] && create_provider=false && echo "GitHub OIDC provider already exists; reusing it"
fi

aws cloudformation deploy --region "$AWS_REGION" --stack-name "$STACK" \
  --template-file "$ROOT/infra/bootstrap/github-oidc.yaml" \
  --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "GitHubRepository=$GITHUB_REPO" "CreateOidcProvider=$create_provider" \
  ${DEPLOY_POLICY_ARN:+DeployPolicyArn="$DEPLOY_POLICY_ARN"}

role_arn() {  # role_arn OutputKey
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text
}
declare -A roles=([dev]="$(role_arn DevRoleArn)" [staging]="$(role_arn StagingRoleArn)" [prod]="$(role_arn ProdRoleArn)")

if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  for environment in dev staging prod; do
    gh api --method PUT "repos/$GITHUB_REPO/environments/$environment" >/dev/null
    gh variable set AWS_ACCOUNT_ID --repo "$GITHUB_REPO" --env "$environment" --body "$account"
    gh variable set AWS_REGION --repo "$GITHUB_REPO" --env "$environment" --body "$AWS_REGION"
    gh variable set AWS_DEPLOY_ROLE_ARN --repo "$GITHUB_REPO" --env "$environment" --body "${roles[$environment]}"
    echo "Set variables for GitHub environment $environment"
  done
  echo "Done. Add required reviewers to the staging and prod environments in GitHub settings."
else
  echo "GitHub CLI not available or not logged in. Set these, then add reviewers to staging and prod:"
  for environment in dev staging prod; do
    echo "  gh variable set AWS_ACCOUNT_ID --repo $GITHUB_REPO --env $environment --body $account"
    echo "  gh variable set AWS_REGION --repo $GITHUB_REPO --env $environment --body $AWS_REGION"
    echo "  gh variable set AWS_DEPLOY_ROLE_ARN --repo $GITHUB_REPO --env $environment --body ${roles[$environment]}"
  done
fi
