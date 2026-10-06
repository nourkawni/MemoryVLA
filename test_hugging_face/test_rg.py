"""
test_rg.py

Standalone check of Hugging Face access through the F26S-NourKawni resource
group of the FatimaInstitute org. It prints which account is logged in, tries
one Inference Providers call billed to the resource group, and tries to start
one tiny CPU Job in the FatimaInstitute namespace.

Not part of the FF_Project training pipeline; it only exists to produce
screenshot-ready output for verifying HF billing/permissions.
Run `hf auth login` once before running this script.
"""

from huggingface_hub import InferenceClient, run_job, whoami
from huggingface_hub.errors import LocalTokenNotFoundError

RESOURCE_GROUP_ID = "6abbeed151f0022e9a402d73"  # str — F26S-NourKawni resource group
ORG_NAMESPACE = "FatimaInstitute"  # str
INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # str


def check_account():
    """
    What it does:
        Asks the Hub which account the saved token belongs to and prints the
        username and the orgs it belongs to.

    Returns:
        str — the logged-in username.

    Example input:
        check_account()

    Example output:
        "NourKawni"   (prints "Logged in as: NourKawni | orgs: ['FatimaInstitute']")
    """
    try:
        me = whoami()  # dict
    except LocalTokenNotFoundError:
        raise SystemExit("Not logged in. Run `hf auth login` first, then re-run this script.")
    org_names = [org["name"] for org in me.get("orgs", [])]  # list[str]
    print("Logged in as:", me["name"], "| orgs:", org_names)
    return me["name"]


def test_inference():
    """
    What it does:
        Sends one short chat completion to Inference Providers, billed to the
        resource group via bill_to, and prints the reply or the error.

    Returns:
        bool — True if the call succeeded, False otherwise.

    Example input:
        test_inference()

    Example output:
        False   (prints "Inference FAILED: 403 ... exceeded your monthly spending limit ...")
    """
    try:
        client = InferenceClient(bill_to=RESOURCE_GROUP_ID)  # InferenceClient
        completion = client.chat.completions.create(
            model=INFERENCE_MODEL,
            messages=[{"role": "user", "content": "How many 'G's in 'huggingface'?"}],
            max_tokens=50,
        )  # ChatCompletionOutput
        print("Inference OK:", completion.choices[0].message.content)
        return True
    except Exception as e:
        print("Inference FAILED:", e)
        return False


def test_job():
    """
    What it does:
        Starts one tiny CPU Job in the org namespace that just prints a line.
        run_job() has no resource-group argument (huggingface_hub 1.14), so
        the job is billed to the namespace.

    Returns:
        bool — True if the job was created, False otherwise.

    Example input:
        test_job()

    Example output:
        False   (prints "Job FAILED: 403 Forbidden ... missing permissions: job.write")
    """
    try:
        job = run_job(
            image="python:3.12",
            command=["python", "-c", "print('RESOURCE_GROUP_JOB_OK')"],
            flavor="cpu-basic",
            namespace=ORG_NAMESPACE,
        )  # JobInfo
        print("Job started:", job.id, job.url)
        return True
    except Exception as e:
        print("Job FAILED:", e)
        return False


if __name__ == "__main__":
    print("=== 1. Account ===")
    check_account()
    print(f"\n=== 2. Inference Providers (bill_to={RESOURCE_GROUP_ID}) ===")
    test_inference()
    print(f"\n=== 3. Jobs (namespace={ORG_NAMESPACE}) ===")
    test_job()
