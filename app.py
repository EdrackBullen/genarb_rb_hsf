#%% Importing variables

from flask import Flask, request, render_template, redirect, url_for
import fitz  # PyMuPDF
import pandas as pd
import io
import sys
import os

import dotenv
import openai
import azure.identity

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader  # PyMuPDF
from langchain_community.vectorstores.azuresearch import AzureSearch

#%% Load environment variables
dotenv.load_dotenv()
azure_credential = azure.identity.AzureDeveloperCliCredential(tenant_id=os.getenv("AZURE_TENANT_ID"))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_MODEL_NAME = os.getenv("AZURE_MODEL_NAME")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = "index_srch_docs_rbgenai_003"
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") 


#%%  Create contextual variables
token_provider = azure.identity.get_bearer_token_provider(azure_credential,
    "https://cognitiveservices.azure.com/.default")
openai_client = openai.AzureOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider)

llm = AzureChatOpenAI(deployment_name=AZURE_DEPLOYMENT_NAME, model_name=AZURE_MODEL_NAME, temperature=0)
embedding = AzureOpenAIEmbeddings(model=AZURE_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
def get_embedding(text):
    return embedding.embed_query(text)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=azure_credential)


def close_search_client(client):
    if client is not None:
        client.close()
    else:
        print("search_client is None")

#%% Making the index
# index_client.create_index(index) - already created 

#%% Schema of index creation

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)


#%% Flask app
app = Flask(__name__)

#Defining gen_arb_df functions 

def prepare_document(pdf_file, search_client): #Converting pdf to vector store
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Extract text from each page of the PDF
    texts = [page.get_text() for page in pdf_document]
    doc_embeddings = embedding.embed_documents(texts)


    # Generate unique IDs and add content for each document
    documents_to_upload = [{"id": str(i + 1), "embedding": embedding, "content": text} for i, (embedding, text) in enumerate(zip(doc_embeddings, texts))]

    search_client.upload_documents(documents=documents_to_upload)

    return documents_to_upload

def genai_query(search_query, question, documents_to_upload, search_client): # GenAI queries 
    # Retrieve top sources for context
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name= AZURE_SEARCH_INDEX,
        embedding_function=embedding.embed_query,
    )

    search_vector = get_embedding(search_query)
    results = search_client.search(search_text=search_query, vector_queries=[
        VectorizedQuery(vector=search_vector, k_nearest_neighbors=10, fields="embedding")
    ])

    excerpt_number = 1
    genai_rag_context = ""  # Initialize the variable
    max_excerpts = 7  # Limit the number of excerpts to 7
    for doc in results:
        if excerpt_number > max_excerpts:
            break
        content = next((d["content"] for d in documents_to_upload if d["id"] == doc["id"]), "")
        genai_rag_context += f"Excerpt {excerpt_number}: {content} "
        excerpt_number += 1

    # System message
    SYSTEM_MESSAGE = '''Assistant answers questions about arbitration awards ONLY using the context provided. 
    Answer only in the format specified. If you don't know the answer, just say "N/A". 
    Do not generate answers that don't use the context provided.'''
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    # Pass query to LLM for the answer
    USER_MESSAGE_ANSWER = question + "\n Context: " + genai_rag_context
    user_message_answer = {"role": "user", "content": USER_MESSAGE_ANSWER}
    response_answer = llm.invoke(input=[system_message, user_message_answer])
    answer = response_answer.content

    USER_MESSAGE_SOURCE = f'''If the answer is "N/A", explain the reasoning step by step. Else, Identify the page number and paragraph number and extract the most relevant sentence from the context below where the answer to the question - {question} is mentioned. The answer should be - {answer}. The context is 
    {genai_rag_context}'''
    user_message_source = {"role": "user", "content": USER_MESSAGE_SOURCE}
    response_source = llm.invoke(input=[system_message, user_message_source])
    source = response_source.content

 
    return answer, source

def genai_process(documents_to_upload, search_client): #Dataframe for questions and search queries
   
    data = [
            {"Field": "Country of Dispute", "Question": "In which country or countries is the dispute located?"},
            {"Field": "Acting for", "Question": "Who is Herbert Smith Freehills acting for in this matter? Choose one option among\nClaimant(s)\nRespondent(s)\nAmicus curiae\nHerbert Smith Freehills is not acting in this matter"},
            {"Field": "Total Number of Claimants", "Question": "What is the total number of claimants in the dispute? The Claimants are the parties who have brought this dispute."},
            {"Field": "Total Number of Respondents", "Question": "What is the total number of respondents in the dispute? The Respondents are the parties against whom the dispute has been brought. In investment arbitration, it is usually the country against which the dispute has been brought."},
            {"Field": "Name of client", "Question": "What is the name of Herbert Smith Freehill's client, if it is acting in the matter?"},
            {"Field": "Client Sector", "Question": "What sector does the Herbert Smith Freehill's client belong to? If the client is a government, choose Government and Public Sector. Choose one option among\nAgribusiness\nAirports\nAsset and Wealth Management\nAutomotive\nAviation\nBanks\nBanks and other Financial Institutions\nConnected and Autonomous Vehicles\nConsumer\nDefence\nDiagnostics and medical Devices\nElectrification\nEnergy\nEnergy Disputes\nEnergy mergers and acquisitions\nFinancial Buyers\nFintech\nGovernment and Public Sector\nHealthcare\nInfrastructure\nInsurance\nLeisure and Sport\nManufacturing and Industrials\nMining\nMobility as a service\nNuclear\nOil and Gas\nPharmaceuticals\nPharmaceuticals and Healthcare\nPharmaceuticals and Healthcare regulatory\nPorts\nPower\nPPP\nProfessional Support and Business Services\nRail\nReal Estate\nReal Estate Disputes\nRenewables\nRoads\nSocial Infrastructure\nSports Disputes\nTechnology, Media and Telecommunications\nWater and Waste"},
            {"Field": "Name (not Herbert Smith Freehills' client)", "Question": "Apart from hsf'S Client, who are the parties to the case?"},
            {"Field": "Sector(s)", "Question": "For each individual or individuals or entity or entities involved other than Herbert Smith Freehills' client, what sectors are involved in the matter? Choose one among:\nAgribusiness\nAirports\nAsset and Wealth Management\nAutomotive\nAviation\nBanks\nBanks and other Financial Institutions\nConnected and Autonomous Vehicles\nConsumer\nDefence\nDiagnostics and medical Devices\nElectrification\nEnergy\nEnergy Disputes\nEnergy mergers and acquisitions\nFinancial Buyers\nFintech\nGovernment and Public Sector\nHealthcare\nInfrastructure\nInsurance\nLeisure and Sport\nManufacturing and Industrials\nMining\nMobility as a service\nNuclear\nOil and Gas\nPharmaceuticals\nPharmaceuticals and Healthcare\nPharmaceuticals and Healthcare regulatory\nPorts\nPower\nPPP\nProfessional Support and Business Services\nRail\nReal Estate\nReal Estate Disputes\nRenewables\nRoads\nSocial Infrastructure\nSports Disputes\nTechnology, Media and Telecommunications\nWater and Waste"},
            {"Field": "Role (not Herbert Smith Freehills' client)", "Question": "For each individual/individuals or entity/entities involved as a party to the dispute other than Herbert Smith Freehills' client, what is the role of the individual or entity in the matter? Do not include experts, law firms and legal counsel. Choose one among:\nCounterparty (select this option ONLY if the individual or entity is named in the case name (e.g. in X v. Y and Z, Y and Z are counterparties))\nNon-client (but same side as client)\nAmicus curiae\nOther"},
            {"Field": "Name of firm", "Question": "For each law firm or independent lawyer representing a party in the dispute, what is the name of the firm or independent lawyer involved? Exclude Herbert Smith Freehills and do not include individual experts or non-lawyers, or firms of non-lawyers or experts."},
            {"Field": "Acting for", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, who is the firm or independent lawyer acting for? Do not include firms of non-lawyers or experts. Choose one among:\nClaimant(s)\nRespondent(s)\nAmicus curiae"},
            {"Field": "Location of firm", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, where is the firm or independent lawyer located? Do not include individual experts or non-lawyers, or firms of non-lawyers or experts"},
            {"Field": "Role of firm", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, what is the role of the firm or independent lawyer in the matter in relation to Herbert Smith Freehills? Do not include individual experts or non-lawyers, or firms of non-lawyers or experts. Choose one among:\nOpposing counsel\nCo-counsel\nLocal counsel\nOpposing local counsel\nHerbert Smith Freehills is not acting in this dispute"},
            {"Field": "Arbitration Seat (Country)", "Question": "What is the country of the city of the arbitration seat? Answer only with the name of the country"},
            {"Field": "Arbitration Seat (City)", "Question": "What is the city of the arbitration seat? Answer only with the name of the city"},
            {"Field": "Case Number / Reference", "Question": "What is the case number or reference? It is available on the first page"},
            {"Field": "Arbitrator Name", "Question": "For each arbitrator, what is the name of the arbitrator? Create a list of all the arbitrators separated by \"/\"."},
            {"Field": "Appointment Date", "Question": "For each arbitrator, what is the appointment date of the arbitrator?"},
            {"Field": "Was Herbert Smith Freehills involved in the arbitrator appointment?", "Question": "Was Herbert Smith Freehills or the {Claimant} involved in the appointment of {Christopher Feit} as arbitrator? This includes but is not limited to situations where ranking for an arbitrator was submitted.\nDeclaring a conflict of interest or mentioning previous association between an arbitrator and Herbert Smith Freehills is not considered involvement in the appointment of an arbitrator."},
            {"Field": "Expert Name", "Question": "In international arbitration, an expert refers to a person or firm with specialized knowledge and expertise relevant to the dispute at hand. Create a list of all the experts separated by \"/\"."},
            {"Field": "Evidence given at hearing?", "Question": "[Specific prompt]"},
            {"Field": "Date of Award or Order", "Question": "What is the date of the award or order?"},
            {"Field": "Dissenting Arbitrators", "Question": "Were there any dissenting arbitrators? If yes, mention their names"},
            {"Field": "Sums Awarded (US Dollars)", "Question": "What sums were awarded in US Dollars?"},
            {"Field": "Sums Awarded (Local Currency)", "Question": "What sums were awarded in a currency other than US dollars?"},
            {"Field": "Award or Order issued in favour of", "Question": "In whose favor was the award or order issued? Herbert Smith Freehills 's Client/Counterparty"}]
     
    data_all = [
            {"Field": "Type of arbitration or PIL work", "Question": "What types of arbitration or PIL work does the Award deal with? Choose all options that apply among the following. Choose Arbitrator appointment only when Herbert Smith Freehills has appointed an arbitrator.\nArbitrator appointment\nInternational Commercial Arbitration\nPre-arbitration advice (International Commercial Arbitration)\nInternational Commercial Arbitration Advice (transactional support)\nInvestment Treaty Advice (transactional support)\nInvestment Treaty Arbitration\nState-to-State Arbitration\nPre-arbitration advice (Investment Treaty Arbitration)\nPre-arbitration advice (state to state arbitration)\nPublic International Law advice\nEnforcement/Recognition of Arbitral Award"},
            {"Field": "Type of Public International Law (If relevant)", "Question": "What types of Public International Law are relevant to the matter? Choose all options that apply among the following:\nBusiness and Human Rights\nClimate Change / Environment\nCrown Immunity\nInternational Organisational Immunity\nLand Boundaries\nLaw of the Sea\nMaritime Boundaries\nSanctions\nState Immunity\nState Personality / Recognition\nTreaty Compliance\nTreaty Interpretation\nTreaty Negotiation"},
            {"Field": "If insurance arbitration, Policy Type involved", "Question": "What types of insurance policies are involved in the arbitration? Choose all options that apply among:\nCAR\nCredit Risk\nCrime / BBB\nCyber\nD&O\nDefective Title\nEmployers Liability\nEnergy\nGeneral Liability\nLatent Defects\nManagement Liability\nPolitical Risk\nPOSI\nProduct Liability\nProfessional\nIndemnity\nProperty Damage / Business Interruption\nPublic Liability\nReinsurance\nTerrorism W&I"},
            {"Field": "Matter Industry", "Question": "Only mention the industries where the parties to the dispute have operations or activities. Do not include industries just because they are mentioned in the award. Choose all that apply among:\nBanking, Capital Markets, Commodity trading, Financial Buyers\nConstruction and Real Estate\nConsumer Products\nCorporate and M&A\nDefence\nEnergy: Nuclear\nEnergy: Oil and Gas\nEnergy: Power\nEnergy: Renewables\nInfrastructure: Air\nInfrastructure: Shipping\nInfrastructure: Rail\nInfrastructure: Roads\nInfrastructure: Social Infrastructure\nInfrastructure: Water and Waste\nInsurance\nLeisure, Travel, Hotels and Sport\nManufacturing: Automotive\nManufacturing: Metals\nManufacturing: other\nMining\nPharmaceuticals and Healthcare\nProfessional Support and Business Services\nTechnology, Media and Telecommunications\nShipping"},
            {"Field": "Significant Matter Aspects", "Question": "What are the significant aspects of the matter? Choose all that apply among:\nAward Winning\nComplexity\nEmerging Markets / Industry Sector\nFirst Matter for New Client\nFirst Matter in Country\nFirst Matter of Kind\nGovernment / State Party Involved\nHigh / Public Profile\nHigh Value\nInnovative\nNature of Parties\nSecond Matter of Kind\nSignificant Client\nUnique / Uncommon"},
            {"Field": "Covid-19 Related?", "Question": "Is the matter related to Covid-19? Yes/No"},
            {"Field": "Primary Language of Dispute", "Question": "What is the primary language of the dispute?"},
            {"Field": "Other Language(s)", "Question": "What other languages are involved in the dispute?"},
            {"Field": "Country of Dispute", "Question": "In which country or countries is the dispute located?"},
            {"Field": "Funded Party", "Question": "Is the party funded by a third party? Yes/No"},
            {"Field": "Name of Third Party Funder", "Question": "A third-party funder in international arbitration is an entity that provides financial support to a party involved in the arbitration in exchange for a share of the proceeds if the case is successful. What is the name of the third-party funder, if there is one?"},
            {"Field": "Acting for", "Question": "Who is Herbert Smith Freehills acting for in this matter? Choose one option among\nClaimant(s)\nRespondent(s)\nAmicus curiae\nHerbert Smith Freehills is not acting in this matter"},
            {"Field": "Total Number of Claimants", "Question": "What is the total number of claimants in the dispute? The Claimants are the parties who have brought this dispute."},
            {"Field": "Total Number of Respondents", "Question": "What is the total number of respondents in the dispute? The Respondents are the parties against whom the dispute has been brought. In investment arbitration, it is usually the country against which the dispute has been brought."},
            {"Field": "Name of client", "Question": "What is the name of Herbert Smith Freehill's client, if it is acting in the matter?"},
            {"Field": "Client Sector", "Question": "What sector does the Herbert Smith Freehill's client belong to? If the client is a government, choose Government and Public Sector. Choose one option among\nAgribusiness\nAirports\nAsset and Wealth Management\nAutomotive\nAviation\nBanks\nBanks and other Financial Institutions\nConnected and Autonomous Vehicles\nConsumer\nDefence\nDiagnostics and medical Devices\nElectrification\nEnergy\nEnergy Disputes\nEnergy mergers and acquisitions\nFinancial Buyers\nFintech\nGovernment and Public Sector\nHealthcare\nInfrastructure\nInsurance\nLeisure and Sport\nManufacturing and Industrials\nMining\nMobility as a service\nNuclear\nOil and Gas\nPharmaceuticals\nPharmaceuticals and Healthcare\nPharmaceuticals and Healthcare regulatory\nPorts\nPower\nPPP\nProfessional Support and Business Services\nRail\nReal Estate\nReal Estate Disputes\nRenewables\nRoads\nSocial Infrastructure\nSports Disputes\nTechnology, Media and Telecommunications\nWater and Waste"},
            {"Field": "Name (not Herbert Smith Freehills' client)", "Question": "Apart from hsf'S Client, who are the parties to the case?"},
            {"Field": "Sector(s)", "Question": "For each individual or individuals or entity or entities involved other than Herbert Smith Freehills' client, what sectors are involved in the matter? Choose one among:\nAgribusiness\nAirports\nAsset and Wealth Management\nAutomotive\nAviation\nBanks\nBanks and other Financial Institutions\nConnected and Autonomous Vehicles\nConsumer\nDefence\nDiagnostics and medical Devices\nElectrification\nEnergy\nEnergy Disputes\nEnergy mergers and acquisitions\nFinancial Buyers\nFintech\nGovernment and Public Sector\nHealthcare\nInfrastructure\nInsurance\nLeisure and Sport\nManufacturing and Industrials\nMining\nMobility as a service\nNuclear\nOil and Gas\nPharmaceuticals\nPharmaceuticals and Healthcare\nPharmaceuticals and Healthcare regulatory\nPorts\nPower\nPPP\nProfessional Support and Business Services\nRail\nReal Estate\nReal Estate Disputes\nRenewables\nRoads\nSocial Infrastructure\nSports Disputes\nTechnology, Media and Telecommunications\nWater and Waste"},
            {"Field": "Role (not Herbert Smith Freehills' client)", "Question": "For each individual/individuals or entity/entities involved as a party to the dispute other than Herbert Smith Freehills' client, what is the role of the individual or entity in the matter? Do not include experts, law firms and legal counsel. Choose one among:\nCounterparty (select this option ONLY if the individual or entity is named in the case name (e.g. in X v. Y and Z, Y and Z are counterparties))\nNon-client (but same side as client)\nAmicus curiae\nOther"},
            {"Field": "Name of firm", "Question": "For each law firm or independent lawyer representing a party in the dispute, what is the name of the firm or independent lawyer involved? Exclude Herbert Smith Freehills and do not include individual experts or non-lawyers, or firms of non-lawyers or experts."},
            {"Field": "Acting for", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, who is the firm or independent lawyer acting for? Do not include firms of non-lawyers or experts. Choose one among:\nClaimant(s)\nRespondent(s)\nAmicus curiae"},
            {"Field": "Location of firm", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, where is the firm or independent lawyer located? Do not include individual experts or non-lawyers, or firms of non-lawyers or experts"},
            {"Field": "Role of firm", "Question": "For each law firm or independent lawyer other than Herbert Smith Freehills, what is the role of the firm or independent lawyer in the matter in relation to Herbert Smith Freehills? Do not include individual experts or non-lawyers, or firms of non-lawyers or experts. Choose one among:\nOpposing counsel\nCo-counsel\nLocal counsel\nOpposing local counsel\nHerbert Smith Freehills is not acting in this dispute"},
            {"Field": "Governing law of Arbitration Agreement", "Question": "What is the governing law of the Arbitration Agreement? Answer only with the name of the country whose law or the institution whose rules are applicable to matters like procedure and the arbitration agreement's validity."},
            {"Field": "Governing Law of Main Contract", "Question": "What is the governing law of the Main Contract? Answer only with the name of the country whose law is applicable. If it is not a contractual claim, say \"Non-Contractual Claim\" instead."},
            {"Field": "Arbitration Seat (Country)", "Question": "What is the country of the city of the arbitration seat? Answer only with the name of the country"},
            {"Field": "Arbitration Seat (City)", "Question": "What is the city of the arbitration seat? Answer only with the name of the city"},
            {"Field": "Type", "Question": "What is the type of the matter? Choose between Ad hoc and Institutional. Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules. if UNCITRAL Rules are applicable, it is ad hoc."},
            {"Field": "Institutional Arbitration Rules", "Question": "Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules. If it is an ad hoc arbitration, state N/A\nIf it is an institutional arbitration, what are the institutional arbitration rules applicable? Choose one option among:\nAAA Institutional Rules\nACICA Institutional Rules\nADCCAC Institutional Rules\nAFA Institutional Rules\nAIAC Institutional Rules\nARIAS Institutional Rules\nCAM Institutional Rules\nCAM Santiago Institutional Rules\nCAS Institutional Rules\nCIArb Institutional Rules\nCIETAC Institutional Rules\nCRCICA Institutional Rules\nDIAC Institutional Rules\nDIFC-LCIA Institutional Rules\nDIS Institutional Rules\nHKIAC Institutional Rules\nIAMA Institutional Rules\nICC Institutional Rules\nICDR Institutional Rules\nICSID Convention\nICSID Additional Facility\nIDRC Institutional Rules\nJCAA Institutional Rules\nKCAB\nInstitutional Rules\nLCIA Institutional Rules\nPCA Institutional Rules\nSCC Institutional Rules\nSIAC\nInstitutional Rules\nTAI Institutional Rules\nTOMAC Institutional Rules\nVIAC Institutional Rules\nUNCITRAL Rules\nAd hoc"},
            {"Field": "Arbitral Institution", "Question": "Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules. Administering authority is not the same as appointing authority. If it is an ad hoc arbitration, state \"N/A\". If it is an institutional arbitration, what is the arbitral institution overseeing the arbitration? Choose one among:\nAbu Dhabi Commercial Conciliation & Arbitration Centre (ADCCAC)\nAmerican Arbitration Association (AAA)\nArbitration and Mediation Center of the Santiago Chamber of Commerce (CAM Santiago)\nArbitration Institute of the Stockholm Chamber of Commerce (SCC)\nAsian International Arbitration Centre (AIAC)\nAssociation Française d'Arbitrage (AFA)\nAustralian Centre for International Commercial Arbitration (ACICA)\nCairo Regional Center for International Commercial Arbitration (CRCICA)\nChartered Institute of Arbitrators (CIArb)\nChina International\nEconomic and Trade Arbitration Commission (CIETAC)\nCourt of Arbitration for Sport (CAS)\nDubai International Arbitration Centre (DIAC)\nGerman Arbitration Institute (DIS)\nHong Kong Dubai International Finance Centre-London Court of International Arbitration (DIFC-LCIA)\nInternational Arbitration Centre (HKIAC)\nInstitute of Arbitrators and Mediators Australia (IAMA)\nInsurance and Reinsurance Arbitration Society (ARIAS)\nInternational Centre for Dispute Resolution (ICDR)\nInternational Centre for Settlement of Investment Disputes (ICSID)\nInternational Chamber of Commerce (ICC)\nInternational Court of Arbitration\nInternational Dispute Resolution Centre (IDRC)\nJapan Commercial Arbitration Association (JCAA)\nKorean Commercial Arbitration Board (KCAB)\nLondon Court of International Arbitration (LCIA)\nMadrid Court of Arbitration (CAM)\nPermanent Court of Arbitration (PCA)\nShanghai Arbitration Commission\nSingapore International Arbitration Centre (SIAC)\nSpanish Court of Arbitration\nThai Arbitration Institute (TAI)\nTokyo Maritime Arbitration Commission (TOMAC)\nVienna International Arbitral Centre (VIAC)\nAd Hoc"},
            {"Field": "Case Number / Reference", "Question": "What is the case number or reference? It is available on the first page"},
            {"Field": "Ad-hoc Rules", "Question": "Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules.\nIf it is an ad hoc arbitration, what ad-hoc rules are applicable to the arbitration?"},
            {"Field": "Appointing Authority (If UNCITRAL Rules)", "Question": "Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules.\nIf it is an ad hoc arbitration and UNCITRAL rules apply, who is the appointing authority? Choose among:\nAbu Dhabi Commercial Conciliation & Arbitration Centre (ADCCAC)\nAmerican Arbitration Association (AAA)\nArbitration and Mediation Center of the Santiago Chamber of Commerce (CAM Santiago)\nArbitration Institute of the Stockholm Chamber of Commerce (SCC)\nAsian International Arbitration Centre (AIAC)\nAssociation Française d'Arbitrage (AFA)\nAustralian Centre for International Commercial Arbitration (ACICA)\nCairo Regional Center for International Commercial Arbitration (CRCICA)\nChartered Institute of Arbitrators (CIArb)\nChina International Economic and Trade Arbitration Commission (CIETAC)\nCourt of Arbitration for Sport (CAS)\nDubai International Arbitration Centre (DIAC)\nGerman Arbitration Institute (DIS)\nHong Kong Dubai International Finance Centre-London Court of International Arbitration (DIFC-LCIA)\nInternational Arbitration Centre (HKIAC)\nInstitute of Arbitrators and Mediators Australia (IAMA)\nInsurance and Reinsurance Arbitration Society (ARIAS)\nInternational Centre for Dispute Resolution (ICDR)\nInternational Centre for Settlement of Investment Disputes (ICSID)\nInternational Chamber of Commerce (ICC)\nInternational Court of Arbitration\nInternational Dispute Resolution Centre (IDRC)\nJapan Commercial Arbitration Association (JCAA)\nKorean Commercial Arbitration Board (KCAB)\nLondon Court of International Arbitration (LCIA)\nMadrid Court of Arbitration (CAM)\nPermanent Court of Arbitration (PCA)\nShanghai Arbitration Commission\nSingapore International Arbitration Centre (SIAC)\nSpanish Court of Arbitration\nThai Arbitration Institute (TAI)\nTokyo Maritime Arbitration Commission (TOMAC)\nVienna International Arbitral Centre (VIAC)"},
            {"Field": "Administering Institution (If Administered UNCITRAL)", "Question": "Ad hoc arbitration is a process where parties appoint arbitrators and manage proceedings themselves, while institutional arbitration involves a permanent institution that administers the arbitration process according to its established rules.\nIf it is an institutional arbitration, what is the arbitral institution overseeing the arbitration? Choose one among:\nAbu Dhabi Commercial Conciliation & Arbitration Centre (ADCCAC)\nAmerican Arbitration Association (AAA)\nArbitration and Mediation Center of the Santiago Chamber of Commerce (CAM Santiago)\nArbitration Institute of the Stockholm Chamber of Commerce (SCC)\nAsian International Arbitration Centre (AIAC)\nAssociation Française d'Arbitrage (AFA)\nAustralian Centre for International Commercial Arbitration (ACICA)\nCairo Regional Center for International Commercial Arbitration (CRCICA)\nChartered Institute of Arbitrators (CIArb)\nChina International Economic and Trade Arbitration Commission (CIETAC)\nCourt of Arbitration for Sport (CAS)\nDubai International Arbitration Centre (DIAC)\nGerman Arbitration Institute (DIS)\nHong Kong Dubai International Finance Centre-London Court of International Arbitration (DIFC-LCIA)\nInternational Arbitration Centre (HKIAC)\nInstitute of Arbitrators and Mediators Australia (IAMA)\nInsurance and Reinsurance Arbitration Society (ARIAS)\nInternational Centre for Dispute Resolution (ICDR)\nInternational Centre for Settlement of Investment Disputes (ICSID)\nInternational Chamber of Commerce (ICC)\nInternational Court of Arbitration\nInternational Dispute Resolution Centre (IDRC)\nJapan Commercial Arbitration Association (JCAA)\nKorean Commercial Arbitration Board (KCAB)\nLondon Court of International Arbitration (LCIA)\nMadrid Court of Arbitration (CAM)\nPermanent Court of Arbitration (PCA)\nShanghai Arbitration Commission\nSingapore International Arbitration Centre (SIAC)\nSpanish Court of Arbitration\nThai Arbitration Institute (TAI)\nTokyo Maritime Arbitration Commission (TOMAC)\nVienna International Arbitral Centre (VIAC)\nAd Hoc"},
            {"Field": "Value of Claim(s) in Local Currency (If applicable)", "Question": "What is the value of the claim(s) in local currency?"},
            {"Field": "Local Currency", "Question": "What is the name of the local currency? It is the currency in which the claim or counterclaim is denominated other than US dollars."},
            {"Field": "Is declaratory relief sought?", "Question": "Is declaratory relief being sought in the arbitration? YES/NO If yes, list all the declaratory reliefs that are sought, which usually starts with \"A DECLARATION that...\""},
            {"Field": "Value of Claim(s) in US Dollars", "Question": "What is the value of the amount that has been asked for by the Claimant or Claimants in US Dollars. If the amount is in a different currency, convert it into US Dollars. If the amount is a range, provide the lowest and highest amounts. If the amount is not specified, provide the answer as \"N/A\"."},
            {"Field": "Value of Counterclaim(s) in US Dollars", "Question": "A counterclaim is a claim made by a respondent (the party against whom the initial claim is made) against the claimant (the party who initiated the arbitration) within the same proceeding. What is the value of the counterclaim(s) in US Dollars? Do not include amounts paid as costs in arbitration. Costs generally refer to the expenses associated with the arbitration process."},
            {"Field": "Value of Counterclaim(s) in Local Currency", "Question": "What is the value of the counterclaim(s) in local currency? If no claim in local currency is mentioned, state \"Not applicable\""},
            {"Field": "Additional Information on Claim and Counterclaim Values", "Question": "Can you provide additional information on the values of the claim and counterclaim? If either party seeks declaratory relief, mention it here. Separate the different heads of relief under sub-headings."},
            {"Field": "Arbitrator Name", "Question": "For each arbitrator, what is the name of the arbitrator? Create a list of all the arbitrators separated by \"/\"."},
            {"Field": "Role", "Question": "For each arbitrator, what is the role of the arbitrator in the arbitration? Herbert Smith Freehills' client is our client. Choose among:\nSole arbitrator – appointed by agreement\nSole arbitrator – appointed by institution\nParty-nominated arbitrator - our client\nParty-nominated arbitrator - not our client\nPresiding arbitrator – appointed by agreement of parties\nPresiding arbitrator – appointed by agreement of tribunal\nPresiding arbitrator - appointed by institution\nCo-arbitrator – appointed by institution\nEmergency Arbitrator – appointed by institution\nEmergency Arbitrator – appointed by agreement of parties\nPresiding arbitrator - appointment unknown\nRole unknown (not appointment by Herbert Smith Freehills)"},
            {"Field": "Appointment Date of Arbitrator", "Question": "For each arbitrator, what is the appointment date of the arbitrator?"},
            {"Field": "Was Herbert Smith Freehills involved in the arbitrator appointment?", "Question": "Was Herbert Smith Freehills or the {Claimant} involved in the appointment of {Christopher Feit} as arbitrator? This includes but is not limited to situations where ranking for an arbitrator was submitted.\nDeclaring a conflict of interest or mentioning previous association between an arbitrator and Herbert Smith Freehills is not considered involvement in the appointment of an arbitrator."},
            {"Field": "Nominating Party", "Question": "For each arbitrator, which party nominated the arbitrator?"},
            {"Field": "Arbitrator Challenged or Removed", "Question": "For each arbitrator was the arbitrator challenged or removed?"},
            {"Field": "Challenge Successful?", "Question": "For each arbitrator, was the challenge to the arbitrator successful?"},
            {"Field": "Date of Removal from Tribunal", "Question": "For each arbitrator, what is the date of removal of the arbitrator from the tribunal?"},
            {"Field": "Reason for Removal", "Question": "For each arbitrator, what was the reason for the removal of the arbitrator, if they were removed? Choose one among:\nVoluntary Resignation\nSuccessful Challenge\nDeath\nIllness"},
            {"Field": "Has a Request for Arbitration been filed?", "Question": "Has a request for arbitration been filed? Yes/No"},
            {"Field": "Date the Request for Arbitration was filed", "Question": "What is the date the request for arbitration was filed?"},
            {"Field": "Pleadings Style", "Question": "What is the style of the pleadings? Choose one option between Memorial OR Statement of Case"},
            {"Field": "Important procedural events", "Question": "What are the important procedural events in the arbitration? Choose all that apply among:\nApplication for bifurcation (jurisdiction/merits)\nApplication for bifurcation (merits/quantum)\nApplication to national court for interim relief/provisional measures\nConsolidation\nEarly determination\nExpedited procedure\nJoinder\nJurisdictional challenge"},
            {"Field": "Additional Information on procedure", "Question": "Can you provide additional information on the arbitration procedure? Include information in relation to requests for redaction, submissions from non-disputing parties etc."},
            {"Field": "Measures Sought", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. What interim or provisional measures are being sought in the arbitration? Choose all that apply among:\nSecurity for Costs\nSecurity for Claim\nSecurity for Enforcement\nSecurity (Other)\nFreezing Order\nEvidence Preservation\nAsset Preservation\nAdvance Payment\nAnti-suit Injunction (Court)\nAnti-suit Injunction (arbitration)\nWitness attendance"},
            {"Field": "Party Seeking Measures", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. Which party is seeking interim measures? Choose among:\nHerbert Smith Freehills 's Client\nCounterParty"},
            {"Field": "Associated Hearing", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. Is there an associated hearing with the interim measures sought?"},
            {"Field": "Relief Granted? (yes / no)", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. Was the interim relief sought granted? State N/A if no interim relief was sought."},
            {"Field": "Forum", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. If interim relief was sought, what is the forum for deciding the interim relief application? Choose one among:\nTribunal\nEmergency Arbitrator\nNational Court. State N/A if no interim relief was sought."},
            {"Field": "National Court Location(s)", "Question": "Interim relief in arbitration refers to temporary measures granted by the arbitration tribunal to preserve the status quo, prevent harm, or protect assets until the final resolution of the dispute. If interim relief was sought, what are the national court locations involved for the interim relief."},
            {"Field": "No. Factual witnesses that contributed a witness statement for the Client(s)", "Question": "How many factual witnesses contributed a witness statement for Herbert Smith Freehills 's client(s)? Include their names. Do not count or include expert witnesses. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "No. Factual witnesses delivered witness statement for Counterparty(ies)", "Question": "How many factual witnesses contributed a witness statement for the Respondent? Include their names. Do not count or include expert witnesses. A factual witness provides testimony based on personal observation or knowledge of events. An expert witness offers specialized opinions and interpretations based on their expertise in a specific field. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "Number of rounds of witness statements", "Question": "How many rounds of witness statements were there? Answer in reference the highest number that precedes a reference to a witness statement. For example, if there is a mention of a n-numbered witness statement, the answer is number n. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "Number of factual witnesses who gave evidence for client at merits hearing", "Question": "How many factual witnesses gave evidence for the Herbert Smith Freehills 's client at the merits hearing? Include their names. Do not include expert witnesses. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "Number of factual witnesses who gave evidence for counterparty at merits hearing", "Question": "How many factual witnesses gave evidence for Herbert Smith Freehills 's counterparty at the merits hearing? Include their names. Do not include expert witnesses. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "Total Number of Joint Expert Reports", "Question": "A joint expert report in arbitration is a document prepared by experts appointed by both parties involved in the arbitration. What is the total number of joint expert reports?"},
            {"Field": "Expert Name", "Question": "In international arbitration, an expert refers to a person or firm with specialized knowledge and expertise relevant to the dispute at hand. Create a list of all the experts separated by \"/\"."},
            {"Field": "Herbert Smith Freehills involved in appointment?", "Question": "In international arbitration, an expert refers to a person or firm with specialized knowledge and expertise relevant to the dispute at hand. If Herbert Smith Freehills or its client was involved in the appointment of an expert, state \"yes\" and list their names. If no expert was appointed by Herbert Smith Freehills or its client, state \"No\"."},
            {"Field": "Role of expert", "Question": "How many factual witnesses contributed a witness statement for the Respondent? Include their names. Do not count or include expert witnesses. A factual witness provides testimony based on personal observation or knowledge of events. An expert witness offers specialized opinions and interpretations based on their expertise in a specific field. If the number is not clear from the award, state \"Number not clear\""},
            {"Field": "Date of Appointment", "Question": "In international arbitration, an expert refers to a person or firm with specialized knowledge and expertise relevant to the dispute at hand. For each expert, what is the date of appointment of the expert?"},
            {"Field": "Number of expert reports produced by the expert", "Question": "[Specific prompt]"},
            {"Field": "Appointment Method", "Question": "In international arbitration, an expert refers to a person or firm with specialized knowledge and expertise relevant to the dispute at hand. For each expert, what was the method of appointment of the expert? Choose one among:\nExpert (appointed by Herbert Smith Freehills 's client)\nExpert (appointed by counterparty)\nExpert (appointed by tribunal)\nExpert (other appointment)"},
            {"Field": "Evidence given at hearing?", "Question": "[Specific prompt]"},
            {"Field": "Type of Hearing", "Question": "For each hearing, what was the type of hearing? Choose all that apply among:\nProcedural\nJurisdictional\nMerits\nMerits (Liability)\nMerits (Quantum) (including where the Tribunal decided that no damages are to be awarded)\nCosts\nEmergency Arbitrator\nEarly Determination"},
            {"Field": "Hearing Start Date", "Question": "For each hearing in the cour proceedings, what was the start date of the hearing?"}
            {"Field": "Hearing End Date", "Question": "For each hearing, what was the end date of the hearing?"},
            {"Field": "Exclude weekends from hearing dates (If relevant)", "Question": "Should weekends be excluded from the hearing dates? Yes/No"},
            {"Field": "Herbert Smith Freehills Internal Advocate(s)", "Question": "Who are the Herbert Smith Freehills internal advocates involved? Only mention those persons who were involved in advocacy"},
            {"Field": "External Advocate(s) involved", "Question": "Who are the advocates external to Herbert Smith Freehills who are involved? Only mention those persons who were involved in advocacy and split them under two headings based on whether they are senior or junior."},
            {"Field": "Hearing Format", "Question": "What was the format of the hearing? Choose one among:\nIn person\nVirtually (Video Conferencing)\nHybrid Virtual/In person\nTelephone"},
            {"Field": "Select all experts who gave evidence at the hearing", "Question": "For each hearing, which experts gave evidence at the hearing?"},
            {"Field": "Type of Award or Order", "Question": "What is the type of award or order issued? Choose one among:\nInterim Award\nPartial Award\nFinal Award\nFinal Partial Award\nEmergency Arbitrator Award\nCosts Award\nOrder\nConsent Award"},
            {"Field": "Subject Matter", "Question": "What is the subject matter of the award or order? Choose all that apply among the options below. If the award or order discusses the compensation payable to the claimant, also select Merits (Quantum) from the options below.\nInterim Relief\nPreliminary Issue\nJurisdiction\nMerits (Liability)\nMerits (Quantum)\nCosts\nEarly Determination"},
            {"Field": "Date of Award or Order", "Question": "What is the date of the award or order?"},
            {"Field": "Dissenting Arbitrators", "Question": "Were there any dissenting arbitrators? If yes, mention their names"},
            {"Field": "Sums Awarded (US Dollars)", "Question": "What sums were awarded in US Dollars?"},
            {"Field": "Sums Awarded (Local Currency)", "Question": "What sums were awarded in a currency other than US dollars?"},
            {"Field": "Local Currency Name", "Question": "What is the name of the currency other than US dollars in which sums were awarded?"},
            {"Field": "Award or Order issued in favour of", "Question": "In whose favor was the award or order issued? Herbert Smith Freehills 's Client/Counterparty"}]

    df = pd.DataFrame(data)
    
    df["Question"] = df["Question"].astype(str)
    df["Field"] = df["Field"].astype(str)

    # Loop through dataframe to get answers and sources
    for i in range(len(df)):
        question = df.at[i, "Question"]
        search_query = df.at[i, "Field"]
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.at[i, "Answer"] = answer
        df.at[i, "Source"] = source
        df.reset_index(drop=True, inplace=True)
        print(f"Field: {df.at[i, 'Field']}: Success")
    
    # Double checking ad hoc conditionality

    df.index = df["Field"]
    if "Type" in df.index and (df.loc["Type", "Answer"] == "Ad hoc"):
        df.loc["Institutional Arbitration Rules", "Answer"] = "N/A"
        df.loc["Institutional Arbitration Rules", "Source"] = "N/A"
        df.loc["Arbitral Institution", "Answer"] = "N/A"
        df.loc["Arbitral Institution", "Source"] = "N/A"
    elif "Type" in df.index:
        df.loc["Appointing Authority (If UNCITRAL Rules)", "Answer"] = "N/A"
        df.loc["Appointing Authority (If UNCITRAL Rules)", "Source"] = "N/A"
        df.loc["Administering Institution (If Administered UNCITRAL)", "Answer"] = "N/A"
        df.loc["Administering Institution (If Administered UNCITRAL)", "Source"] = "N/A"

    # Assigning hsf_client 
    if "Acting for" in df.index:
        hsf_client = df.loc["Acting for", "Answer"]
        not_hsf_client = f"Counterparty other than {hsf_client}"

    # Name of counterparty
    if "Name (not Herbert Smith Freehills' client)" in df.index:
        df.loc["Name (not Herbert Smith Freehills' client)", "Answer"] = ""
        df.loc["Name (not Herbert Smith Freehills' client)", "Source"] = ""

        question = f"What is the name of {not_hsf_client}"
        search_query = f"Name of {not_hsf_client}"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["Name (not Herbert Smith Freehills' client)", "Answer"] = answer
        df.loc["Name (not Herbert Smith Freehills' client)", "Source"] = source

    # Role of other parties
    if "Role (not Herbert Smith Freehills' client)" in df.index:
        df.loc["Role (not Herbert Smith Freehills' client)", "Answer"] = ""
        df.loc["Role (not Herbert Smith Freehills' client)", "Source"] = ""

        question = f'''For each individual/individuals or entity/entities involved other than {hsf_client}, what is the role of the individual or entity in the matter? Do not include experts, law firms and legal counsel. Choose one among:
                    Counterparty (select this option ONLY if the individual or entity is named in the case name (e.g. in X v. Y and Z, Y and Z are counterparties))
                    Non-client (but same side as client)
                    Amicus curiae
                    Other'''
        search_query = f"Role of individual/individuals or entity/entities involved other than {hsf_client}"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["Role (not Herbert Smith Freehills' client)", "Answer"] = answer
        df.loc["Role (not Herbert Smith Freehills' client)", "Source"] = source

    # No. Factual witnesses that contributed a witness statement for the Client(s)
    if "No. Factual witnesses that contributed a witness statement for the Client(s)" in df.index:
        df.loc["No. Factual witnesses that contributed a witness statement for the Client(s)", "Answer"] = ""
        df.loc["No. Factual witnesses that contributed a witness statement for the Client(s)", "Source"] = ""

        question = f'''How many factual witnesses contributed a witness statement for {hsf_client}? Include their names. Do not count or include expert witnesses.  If the number is not clear from the award, state "Number not clear"'''
        search_query = f"Factual witnesses that contributed a witness statement for {hsf_client}"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["No. Factual witnesses that contributed a witness statement for the Client(s)", "Answer"] = answer
        df.loc["No. Factual witnesses that contributed a witness statement for the Client(s)", "Source"] = source

    # No. Factual witnesses that contributed a witness statement for the Counterparty(ies)
    if "No. Factual witnesses that contributed a witness statement for the Counterparty(ies)" in df.index:
        df.loc["No. Factual witnesses that contributed a witness statement for the Counterparty(ies)", "Answer"] = ""
        df.loc["No. Factual witnesses that contributed a witness statement for the Counterparty(ies)", "Source"] = ""

        question = f'''How many factual witnesses contributed a witness statement for {not_hsf_client}? Include their names. Do not count or include expert witnesses.  If the number is not clear from the award, state "Number not clear"'''
        search_query = f"Factual witnesses that contributed a witness statement for {not_hsf_client}"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["No. Factual witnesses that contributed a witness statement for the Counterparty(ies)", "Answer"] = answer
        df.loc["No. Factual witnesses that contributed a witness statement for the Counterparty(ies)", "Source"] = source

    # Number of factual witnesses who gave evidence for client at merits hearing
    if "Number of factual witnesses who gave evidence for client at merits hearing" in df.index:
        df.loc["Number of factual witnesses who gave evidence for client at merits hearing", "Answer"] = ""
        df.loc["Number of factual witnesses who gave evidence for client at merits hearing", "Source"] = ""

        question = f'''How many factual witnesses gave evidence for the {hsf_client} at the merits hearing?  Include their names. DO not include expert witnesses.  If the number is not clear from the award, state "Number not clear"'''
        search_query = f"Number of factual witnesses who gave evidence for {hsf_client} at merits hearing"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["Number of factual witnesses who gave evidence for client at merits hearing", "Answer"] = answer
        df.loc["Number of factual witnesses who gave evidence for client at merits hearing", "Source"] = source

    # Number of factual witnesses who gave evidence for counterparty at merits hearing
    if "Number of factual witnesses who gave evidence for counterparty at merits hearing" in df.index:
        df.loc["Number of factual witnesses who gave evidence for counterparty at merits hearing", "Answer"] = ""
        df.loc["Number of factual witnesses who gave evidence for counterparty at merits hearing", "Source"] = ""

        question = f'''How many factual witnesses gave evidence for {not_hsf_client} at the merits hearing?  
                        Include their names. DO not include expert witnesses.
                        If the number is not clear from the award, state "Number not clear"'''
        search_query = f"Number of factual witnesses who gave evidence for {not_hsf_client} at merits hearing"
        answer, source = genai_query(search_query, question, documents_to_upload, search_client)
        df.loc["Number of factual witnesses who gave evidence for counterparty at merits hearing", "Answer"] = answer
        df.loc["Number of factual witnesses who gave evidence for counterparty at merits hearing", "Source"] = source

    # Appointment of arbitrator
    if "Was Herbert Smith Freehills involved in the arbitrator appointment?" in df.index:
        df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Answer"] = ""
        df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Source"] = ""

        names = df.loc["Arbitrator Name", "Answer"].split(" / ")
        for name in names:
            question = f'''Was Herbert Smith Freehills or the {hsf_client} involved in the appointment of {name} as arbitrator? This includes but is not limited to situations where ranking for an arbitrator was submitted.
    Declaring a conflict of interest or mentioning previous association between an arbitrator and Herbert Smith Freehills is not considered involvement in the appointment of an arbitrator.'''
            search_query= f"Appointment of {name} as arbitrator"
            answer, source = genai_query(search_query, question, documents_to_upload, search_client)
            df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Answer"] = df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Answer"] + f" {name}: " + answer + "<br>"
            df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Source"] = df.loc["Was Herbert Smith Freehills involved in the arbitrator appointment?", "Source"] + f"{name}: " + source + "<br>"

    # Sending questions for each expert
    if "Number of expert reports produced by the expert" in df.index:
        df.loc["Number of expert reports produced by the expert", "Answer"] = ""
        df.loc["Number of expert reports produced by the expert", "Source"] = ""

        names = df.loc["Expert Name", "Answer"].split(" / ")
        for name in names:
            question = f"state how many expert reports were produced by {name}"
            search_query= f"expert reports produced by {name}"
            answer, source = genai_query(search_query, question, documents_to_upload, search_client)
            df.loc["Number of expert reports produced by the expert", "Answer"] = df.loc["Number of expert reports produced by the expert", "Answer"] + f" {name}: " + answer + "<br>"
            df.loc["Number of expert reports produced by the expert", "Source"] = df.loc["Number of expert reports produced by the expert", "Source"] + f"{name}: " + source + "<br>"

    # Evidence given at hearing
    if "Evidence given at hearing?" in df.index:
        df.loc["Evidence given at hearing?", "Answer"] = ""
        df.loc["Evidence given at hearing?", "Source"] = ""

        for name in names:
            question = f"state whether evidence was given at the hearing by {name}"
            search_query = f"evidence given at the hearing by {name}"
            answer, source = genai_query(search_query, question, documents_to_upload, search_client)
            df.loc["Evidence given at hearing?", "Answer"] = df.loc["Evidence given at hearing?", "Answer"] + f" {name}: " + answer + "<br>"
            df.loc["Evidence given at hearing?", "Source"] = df.loc["Evidence given at hearing?", "Source"] + f"{name}: " + source + "<br>"

    df = df[["Field", "Answer", "Source"]]
    return df
df_html = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
#%% Flask elements
    global df_html
    if request.method == 'POST':
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )

        if 'pdf' not in request.files:
            return redirect(url_for('index'))
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return redirect(url_for('index'))
        if pdf_file:
            # Capture console output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            documents_to_upload = prepare_document(pdf_file, search_client)
            df = genai_process(documents_to_upload, search_client)

         #%%Check number of documents in index
            results = search_client.search(search_text="*", select=["id"])
            document_ids = [doc["id"] for doc in results]
            print("Total pages reviewed: " + str(len(document_ids)))
            if len(document_ids)>0:
                search_client.delete_documents(documents=[{"id": doc_id} for doc_id in document_ids])
                print("Documents deleted:" + str(len(document_ids)))


#%% gen_arb_df import

            # Replace \n with <br> in the dataframe
            df = df.replace('\n', '<br>', regex=True)

            # Convert dataframe to HTML with custom styles, removing the index column
            df_html = df.to_html(classes='dataframe', index=False, border=0, escape=False)
            
            # Get console output
            sys.stdout = old_stdout
            console_output = buffer.getvalue()

            # Ensure that the search client is closed properly
            if search_client is not None:
                close_search_client(search_client)
            else:
                print("search_client is None before closing")

            return redirect(url_for('progress', console_output=console_output))
            
    elif request.method == 'GET' and df_html is not None:
        return render_template('display.html', table=df_html)
    return redirect(url_for('index'))

@app.route('/progress')
def progress():
    console_output = request.args.get('console_output', '')
    return render_template('progress.html', console_output=console_output)

if __name__ == '__main__':
    app.run(debug=True)
