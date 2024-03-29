from django.shortcuts import render
from django.http import HttpResponse
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os
import torch



from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=500
)

local_llm = HuggingFacePipeline(pipeline=pipe)



restaurant_template = """
I want you to become an indian lawyer and enlist me the acts and rights with a one liner description, for every issue i'll raise, you've to only give me relevant acts and rights coving from that issue for example:
The Indian Contract Act, 1872: This act governs the basic principles of contract law, which includes rent agreements. It sets the legal framework for agreements and their enforceability.

The Rent Control Act (varies by state): Different Indian states have their own Rent Control Acts that regulate rent agreements, rent control, and tenant-landlord relationships. You should consult the specific Rent Control Act relevant to your location for details.

Right to Property: As per the Indian Constitution, the right to property is a fundamental right, but it's not absolute. You have the right to enjoy your property and enter into agreements, including rent agreements, as long as it is in compliance with the law.

Tenant's Rights: Tenants in India also have rights, such as the right to peaceful enjoyment of the rented premises, right to a fair and reasonable rent, and right to notice for eviction. These rights may vary by state. 
etc.
you've to answer me like this for every issue i'll raise,

here's the example issue:
{issue}
"""


def main(request):
    return render(request, 'index.html')


def getResponse(request):
    userMessage = request.GET.get('userMessage')
    
    if userMessage:
        prompt = PromptTemplate(
                input_variables=["issue"],
                template=restaurant_template,
            )
        
        # Create an LLMChain
        chain = LLMChain(llm=local_llm, prompt=prompt)


        # Generate a response
        response = chain.run(userMessage)

    return HttpResponse(response)