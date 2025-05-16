# Advanced Integration of Azure ML, OpenAI, and LLMs for Healthcare Claims Retrieval and Q&A

# 1: Install required packages
!pip install openai langchain azure-ai-ml azure-identity azure-search-documents pandas tqdm

# Step 2: Import necessary libraries
import os
import pandas as pd
from tqdm import tqdm
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import OpenAIEmbeddings

# 3: Authenticate and configure Azure ML workspace
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<subscription-id XXX>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace-name>"
)

# 4: Configure Azure OpenAI access
openai.api_type = "azure"
openai.api_base = "https://<openai-endpoint>.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

# 5: LangChain LLM and embedding setup
llm = AzureChatOpenAI(
    openai_api_base=openai.api_base,
    deployment_name="gpt-4-healthcare",
    openai_api_key=openai.api_key,
    openai_api_version=openai.api_version
)

embedding_function = OpenAIEmbeddings(deployment="text-embedding-ada-002")

# 6: Load and process healthcare claims data
claims_df = pd.read_csv("claims_data.csv")  

# Combine relevant fields (e.g., diagnosis, procedure, notes)
def combine_fields(row):
    return f"Patient ID: {row['patient_id']}, Diagnosis: {row['diagnosis']}, Procedure: {row['procedure']}, Notes: {row['clinical_notes']}"

claims_df["combined_text"] = claims_df.apply(combine_fields, axis=1)

# 7: Vectorize and ingest into Azure Cognitive Search
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import Vector
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint="https://<search-endpoint>.search.windows.net",
    index_name="claims-index",
    credential=AzureKeyCredential("<search-key>")
)

upload_batch = []
for i, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
    doc_text = row["combined_text"]
    embedding = embedding_function.embed_query(doc_text)
    upload_batch.append({
        "id": str(i),
        "content": doc_text,
        "embedding": embedding
    })

search_client.upload_documents(documents=upload_batch)

# 8: Connect LangChain to Azure Cognitive Search
vectorstore = AzureSearch(
    endpoint="https://<search-endpoint>.search.windows.net",
    key="<search-admin-key XXX>",
    index_name="claims-index",
    embedding_function=embedding_function
)

# 9: RAG pipeline for claim-based question answering
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 10: Example query
query = "What is the typical treatment path for patients with diabetes and hypertension based on historical claims?"
result = qa_chain.run(query)
print("Answer:", result)

# 11: Deploy as Azure ML endpoint for API access
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(name="claims-qa-endpoint", description="Claims-based Q&A via LLM")
deployment = ManagedOnlineDeployment(
    name="claims-qa-deployment",
    endpoint_name=endpoint.name,
    model="<registered-llm-id>",
    instance_type="Standard_DS3_v2",
    instance_count=1
)

ml_client.online_endpoints.begin_create_or_update(endpoint)
ml_client.online_deployments.begin_create_or_update(deployment)

# Simple Sample (input): Patient ID: 1001, Diagnosis: Type 2 Diabetes, Hypertension, Procedure: Metformin, Blood Pressure Monitoring, 
# Notes: Patient exhibits elevated A1C and systolic blood pressure. Advised dietary adjustments and started on metformin and lisinopril. Follow-up in 3 months.

# query = "What is the typical treatment path for patients with diabetes and hypertension based on historical claims?"
# result = qa_chain.run(query)

# Typical treatment for patients with both diabetes and hypertension includes lifestyle modification (e.g., diet and exercise), 
# glucose control medications such as metformin, and antihypertensive agents like ACE inhibitors (e.g., lisinopril). 
# Claims data often show follow-ups every 3–6 months and ongoing monitoring of A1C and blood pressure levels.

