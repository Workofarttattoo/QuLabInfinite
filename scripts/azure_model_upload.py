"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Azure Model Wisdom Upload Bridge
Registers ECH0 wisdom dataset and system identity in Azure AI Studio / ML Workspace.
"""

import os
import argparse
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Data
    from azure.identity import DefaultAzureCredential
except ImportError:
    print("⚠️  Azure AI ML SDK not found. Please install with: pip install azure-ai-ml azure-identity")
    MLClient = None

def upload_wisdom(subscription_id, resource_group, workspace_name):
    if MLClient is None:
        print("❌ Cannot proceed without Azure SDK.")
        return

    print(f"🚀 Connecting to Azure Workspace: {workspace_name}...")

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(subscription_id, resource_group, workspace_name, credential)

        # 1. Register Wisdom Dataset
        print("📦 Registering Wisdom Dataset (data/echo_wisdom.jsonl)...")
        wisdom_data = Data(
            path="data/echo_wisdom.jsonl",
            type="uri_file",
            description="Scientific wisdom ingested from 70+ labs and Cancer PhD training.",
            name="echo-wisdom-dataset",
            version="1.0.0"
        )
        ml_client.data.create_or_update(wisdom_data)
        print("✅ Wisdom Dataset registered.")

        # 2. Upload System Identity
        print("🆔 Uploading System Identity (qulab/ech0/system_identity.md)...")
        identity_data = Data(
            path="qulab/ech0/system_identity.md",
            type="uri_file",
            description="ECH0 persona, expertise, and tool-calling protocol.",
            name="echo-system-identity",
            version="1.0.0"
        )
        ml_client.data.create_or_update(identity_data)
        print("✅ System Identity registered.")

        print("\n🎉 ECH0 Wisdom Transfer Complete!")
        print("Next Steps:")
        print("1. Go to Azure AI Studio -> Fine-tuning.")
        print("2. Select Meta-Llama-3-70B-Instruct (or your 17B model).")
        print("3. Use 'echo-wisdom-dataset' for training.")
        print("4. Paste the content of 'qulab/ech0/system_identity.md' into the System Message field.")

    except Exception as e:
        print(f"❌ Error during Azure upload: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ECH0 wisdom to Azure.")
    parser.add_argument("--sub", help="Azure Subscription ID")
    parser.add_argument("--rg", help="Resource Group Name")
    parser.add_argument("--ws", help="Workspace Name")

    args = parser.parse_args()

    if args.sub and args.rg and args.ws:
        upload_wisdom(args.sub, args.rg, args.ws)
    else:
        print("ℹ️  Usage: python scripts/azure_model_upload.py --sub <SUB_ID> --rg <RG> --ws <WS>")
