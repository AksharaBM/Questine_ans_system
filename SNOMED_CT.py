import spacy  # Importing spaCy library for natural language processing
from spacy.matcher import PhraseMatcher  # Importing PhraseMatcher from spaCy's matcher module
import re
import numpy as np
def SNOMED_CT(questions):
    def is_medical_term(term):
        snomedct=np.load("SNOMED_CT.npy")
        snomedct=list(snomedct)
        medical_terms = snomedct
       
        
        return term in medical_terms
    
        
        # return term.lower() in medical_terms  # Checking if the term is in the set of medical terms
    
    # Function to extract matched terms from a list of questions
    def extract_matched_terms(questions):
        relevant_terms = set()  # Initializing an empty set to store relevant terms
        for text in questions:  # Looping through each question in the list of questions
            # Tokenize the text
            tokens = text.split()  # Splitting the text into tokens (words)
            # Identify relevant terms
            for token in tokens:  # Looping through each token in the tokens
                if is_medical_term(token):  # Checking if the token is a medical term
                    relevant_terms.add(token)  # Adding the medical term to the set of relevant terms
    
        # Create patterns for matching
        patterns = list(relevant_terms)  # Converting the set of relevant terms to a list
    
        # Initialize PhraseMatcher with the patterns
        nlp = spacy.load("en_core_web_sm")  # Loading the English language model for spaCy
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Initializing the PhraseMatcher
    
        for pattern in patterns:  # Looping through each pattern in the patterns
            matcher.add("SNOMED_CT", None, nlp(pattern))  # Adding the pattern to the PhraseMatcher
    
        # Process each question and find matches of SNOMED CT terms
        matched_terms_all = []  # Initializing an empty list to store all matched terms
        for question in questions:  # Looping through each question in the list of questions
            doc = nlp(question)  # Processing the question with the spaCy NLP pipeline
            matches = matcher(doc)  # Finding matches of SNOMED CT terms in the question
            matched_terms = [doc[start:end].text for match_id, start, end in matches]  # Extracting matched terms
            matched_terms_all.extend(matched_terms)  # Adding the matched terms to the list of all matched terms
    
        return list(set(matched_terms_all))  # Returning unique matched terms
    
    


    # Extract matched SNOMED CT concepts from the list of questions
    matched_terms = extract_matched_terms(questions)  # Calling the function to extract matched terms
       
    return matched_terms



