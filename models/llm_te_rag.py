import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import gensim
from gensim import corpora
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, Dict, List

import os
import ray
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup

from openai import OpenAI

from tqdm import tqdm


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---


class ChunkExtractor:
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class LLMPreCheckTopicModelingRAG:
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, num_topics=10):
        
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def check_llm_knowledge(self, query: str, query_time: str) -> tuple[bool, str]:
        
        system_prompt = """You are a knowledgeable AI assistant. For the given question:
1. First, assess if you can answer it confidently using only your general knowledge
2. Begin your response with either "CONFIDENT:" or "UNCERTAIN:"
3. If confident, provide a direct answer
4. If uncertain, just respond with "UNCERTAIN:" and nothing else"""

        user_message = f"Current Time: {query_time}\nQuestion: {query}"

        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            response_text = response.choices[0].message.content
        else:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            response = self.llm.generate(
                [formatted_prompt],
                vllm.SamplingParams(
                    temperature=0.1,
                    max_tokens=100,
                ),
                use_tqdm=False
            )
            response_text = response[0].outputs[0].text

        can_answer = response_text.strip().startswith("CONFIDENT:")
        answer = response_text.replace("CONFIDENT:", "").strip() if can_answer else ""
        
        return can_answer, answer

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        
        answers = []
        rag_queries = []
        rag_query_times = []
        rag_search_results = []
        rag_interaction_ids = []
        
        for idx, query in enumerate(queries):
            can_answer, direct_answer = self.check_llm_knowledge(query, query_times[idx])
            
            if can_answer:
                answers.append(direct_answer)
            else:
                answers.append(None)  
                rag_queries.append(query)
                rag_query_times.append(query_times[idx])
                rag_search_results.append(batch_search_results[idx])
                rag_interaction_ids.append(batch_interaction_ids[idx])

        # If there are queries that need RAG, process them
        if rag_queries:
            rag_batch = {
                "interaction_id": rag_interaction_ids,
                "query": rag_queries,
                "search_results": rag_search_results,
                "query_time": rag_query_times
            }
            
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                rag_interaction_ids, rag_search_results
            )
            
            self.train_lda_model(chunks)
            chunk_topic_dists = self.calculate_chunk_topic_distribution(chunks)
            
            # Process each RAG query
            batch_retrieval_results = []
            for idx, query in enumerate(rag_queries):
                aligned_chunks_idxs = self.align_query_with_chunks(query, chunk_topic_dists)
                aligned_chunks = chunks[aligned_chunks_idxs]
                batch_retrieval_results.append(aligned_chunks)
            
            # Generate answers for RAG queries
            formatted_prompts = self.format_prompts(rag_queries, rag_query_times, batch_retrieval_results)
            
            if self.is_server:
                rag_responses = []
                for prompt in formatted_prompts:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_name,
                        messages=prompt,
                        n=1,
                        top_p=0.9,
                        temperature=0.1,
                        max_tokens=50,
                    )
                    rag_responses.append(response.choices[0].message.content)
            else:
                responses = self.llm.generate(
                    formatted_prompts,
                    vllm.SamplingParams(
                        n=1,
                        top_p=0.9,
                        temperature=0.1,
                        skip_special_tokens=True,
                        max_tokens=50,
                    ),
                    use_tqdm=False
                )
                rag_responses = [response.outputs[0].text for response in responses]
            
            # Merge RAG answers back into main answers list
            rag_idx = 0
            for idx in range(len(answers)):
                if answers[idx] is None:
                    answers[idx] = rag_responses[rag_idx]
                    rag_idx += 1
        
        return answers

    def train_lda_model(self, chunks: List[str]):
        """
        Train an LDA model on the provided chunks of text.
        
        Parameters:
            chunks (List[str]): List of text chunks (sentences) to train the LDA model.
        """
        # Preprocess the text using TF-IDF vectorizer
        #tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        #tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)

        # Convert to a format suitable for Gensim LDA
        texts = [text.split() for text in chunks]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Train the LDA model
        self.lda_model = gensim.models.LdaMulticore(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=10,
            workers=4,
            minimum_probability=0 # always output all topics
        )
        print("Finished training an LDA model.")
        
        
    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def get_topic_distribution(self, text: str):
        """
        Get the topic distribution for a single piece of text (query or chunk).
        
        Parameters:
            text (str): The text whose topic distribution needs to be calculated.
        
        Returns:
            List[tuple]: List of topic probabilities for each topic.
        """
        # Preprocess the text and convert it to the bow format
        bow = self.dictionary.doc2bow(text.split())
        
        # get the topic distribution of the given text
        topic_dist = self.lda_model[bow]
        dense_topic_vector = np.zeros(self.num_topics)
        for topic_id, proportion in topic_dist:
            dense_topic_vector[topic_id] = proportion
        
        return dense_topic_vector

    def calculate_query_topic_distribution(self, query: str):
        """
        Calculate the topic distribution for the input query.
        
        Parameters:
            query (str): The input query.
        
        Returns:
            List[tuple]: List of topic probabilities for the query.
        """
        return self.get_topic_distribution(query)

    def calculate_chunk_topic_distribution(self, chunks: List[str]):
        """
        Calculate the topic distribution for all chunks of text.
        
        Parameters:
            chunks (str): Chunks of text (external text).
        
        Returns:
            List[ndarray]: List of topic probabilities for the chunk.
        """
        chunk_topics = []
        for chunk in chunks:
            chunk_topics.append(self.get_topic_distribution(chunk))
        return chunk_topics

    def align_query_with_chunks(self, query: str, chunk_topics: np.ndarray) -> List[int]:
        """
        Align the input query with the external chunks based on topic distributions.
        
        Parameters:
            query (str): The input query.
            chunk_topics (np.ndarray): Array of topic distribution for each chunk.
        
        Returns:
            List[int]: List of chunk indices that best align with the query’s topic distribution.
        """
        # Calculate the topic distribution for the query
        query_topic_dist = self.calculate_query_topic_distribution(query)
        
        # Store similarity scores for chunks
        chunk_similarities = []
        
        # align with the query
        for idx in range(len(chunk_topics)):
            # Calculate cosine similarity between query and chunk topic distributions
            similarity_score = self.calculate_cosine_similarity(query_topic_dist, chunk_topics[idx])
            chunk_similarities.append((idx, similarity_score))
        
        # Sort chunks by similarity score (highest first)
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N most relevant chunks based on similarity
        top_chunks_idx = [chunk_idx for chunk_idx, _ in chunk_similarities[:NUM_CONTEXT_SENTENCES]]
        
        return top_chunks_idx

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Format the input query and aligned chunks into a prompt for the LLM.
        
        Parameters:
            query (str): The input query.
            aligned_chunks (List[str]): Chunks that are aligned with the query.
        
        Returns:
            str: The formatted prompt to send to the LLM.
        """
        system_prompt = "You are provided with a question and various references. Answer questions directly and precisely without reasoning, using the existing knowledge and references provided. Only say 'I don't know' if no relevant information exists. "
        formatted_prompts = []
        
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using the references listed above and what you already know, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        
        #references = "\n".join(f"- {chunk.strip()}" for chunk in aligned_chunks)
        #prompt = f"### Question: {query}\n\n### Context:\n{references}\n\nAnswer the question based on the above context."
        return formatted_prompts

    
    def calculate_cosine_similarity(self, query_topic_dist: np.ndarray, chunk_topic_dist: np.ndarray) -> float:
        """
        Calculate cosine similarity between the topic distributions of the query and a chunk.
        
        Parameters:
            query_topic_dist (np.ndarray): Topic distribution vector of the query.
            chunk_topic_dist (np.ndarray): Topic distribution vector of the chunk.
        
        Returns:
            float: Cosine similarity between the two topic distributions.
        """
        similarity = np.dot(query_topic_dist, chunk_topic_dist) / (
            np.linalg.norm(query_topic_dist) * np.linalg.norm(chunk_topic_dist)
        )
        
        return similarity

    def calculate_kl_divergence(self, query_topic_dist: np.ndarray, chunk_topic_dist: np.ndarray) -> float:
        return np.sum(query_topic_dist * np.log(query_topic_dist / chunk_topic_dist))
        