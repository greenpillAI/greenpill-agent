from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import numpy as np
from transformers import AutoTokenizer
import spacy
import os
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class ChunkMetadata:
    document_name: str
    section: str
    chunk_index: int
    total_chunks: int
    content_hash: str
    document_type: str
    semantic_density: float
    key_terms: List[str]
    position: str
    word_count: int
    embedding_model: str
    chunking_strategy: str
    created_at: str
    context_overlap: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class ProcessingStats:
    total_chunks: int = 0
    unique_terms: int = 0
    avg_chunk_size: float = 0.0
    semantic_scores: List[float] = field(default_factory=list)
    processing_time: float = 0.0

class EnhancedDocumentProcessor:
    def __init__(self, db_path: str = "db"):
        load_dotenv()
        self.setup_logging()
        self.initialize_components(db_path)
        self.load_nlp_models()
        
    def setup_logging(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self, db_path: str):
        """Initialize database and embedding components"""
        self.client = PersistentClient(path=db_path)
        
        # Support multiple embedding models
        self.embedding_functions = {
            "openai": embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            ),
            "sentence-transformers": embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        }
        
        self.default_embedding = "openai"
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(
                name="knowledge_base",
                embedding_function=self.embedding_functions[self.default_embedding]
            )
        except:
            self.collection = self.client.create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_functions[self.default_embedding]
            )
            
    def load_nlp_models(self):
        """Load NLP models and components"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            raise

    def calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density score for text chunk"""
        doc = self.nlp(text)
        
        # Count named entities, technical terms, and important POS tags
        entities = len(doc.ents)
        technical_terms = len([token for token in doc 
                             if token.pos_ in ['NOUN', 'PROPN'] 
                             and not token.is_stop])
        
        # Calculate density metrics
        word_count = len(text.split())
        term_density = (entities + technical_terms) / word_count if word_count > 0 else 0
        
        # Add readability score
        sentences = len(list(doc.sents))
        avg_sentence_length = word_count / sentences if sentences > 0 else 0
        readability = 1.0 - (avg_sentence_length / 100)  # Normalize
        
        # Combine metrics
        semantic_density = (term_density + readability) / 2
        return min(1.0, semantic_density)

    def extract_key_terms(self, text: str) -> List[str]:
        """Extract important technical terms and phrases"""
        doc = self.nlp(text)
        
        # Get named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Find technical terms using POS patterns
        technical_patterns = [
            [{'POS': 'NOUN'}, {'POS': 'NOUN'}],
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}]
        ]
        
        technical_terms = []
        for pattern in technical_patterns:
            matcher = spacy.matcher.Matcher(doc.vocab)
            matcher.add("tech_terms", [pattern])
            matches = matcher(doc)
            for _, start, end in matches:
                technical_terms.append(doc[start:end].text)
        
        # Combine and deduplicate
        all_terms = list(set(entities + noun_phrases + technical_terms))
        
        # Sort by importance using TF-IDF scores
        if all_terms:
            tfidf_matrix = self.tfidf.fit_transform([text])
            term_scores = dict(zip(self.tfidf.get_feature_names_out(), 
                                 tfidf_matrix.toarray()[0]))
            all_terms.sort(key=lambda x: term_scores.get(x, 0), reverse=True)
        
        return all_terms[:10]  # Return top 10 terms

    def adaptive_chunk_size(self, text: str) -> int:
        """Determine optimal chunk size based on content complexity"""
        doc = self.nlp(text)
        
        # Analyze content characteristics
        avg_sentence_length = sum(len(sent) for sent in doc.sents) / len(list(doc.sents))
        entity_density = len(doc.ents) / len(doc)
        
        # Calculate base chunk size
        if entity_density > 0.1:  # High density of named entities
            base_size = 800  # Smaller chunks for dense content
        elif avg_sentence_length > 30:  # Long sentences
            base_size = 1200  # Larger chunks for flowing content
        else:
            base_size = 1000  # Default size
            
        return int(base_size)

    def create_semantic_chunks(self, text: str) -> List[str]:
        """Create chunks based on semantic boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > 1000:
                # Check if adding this sentence would make a better semantic unit
                current_semantic_score = self.calculate_semantic_density(" ".join(current_chunk))
                potential_chunk = current_chunk + [sentence]
                potential_semantic_score = self.calculate_semantic_density(" ".join(potential_chunk))
                
                if potential_semantic_score > current_semantic_score and current_length < 1200:
                    current_chunk.append(sentence)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def calculate_chunk_overlap(self, chunks: List[str]) -> List[Dict[str, float]]:
        """Calculate semantic overlap between adjacent chunks"""
        overlaps = []
        
        for i in range(len(chunks) - 1):
            current_embedding = self.embedding_functions[self.default_embedding].embed_documents([chunks[i]])[0]
            next_embedding = self.embedding_functions[self.default_embedding].embed_documents([chunks[i + 1]])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, next_embedding) / \
                        (np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding))
            
            overlaps.append({
                "next_chunk": similarity,
                "semantic_continuity": similarity > 0.7
            })
            
        # Add empty overlap for last chunk
        overlaps.append({"next_chunk": 0.0, "semantic_continuity": False})
        
        return overlaps

    def process_document(self, text: str, strategy: ChunkingStrategy = ChunkingStrategy.HYBRID) -> Tuple[List[str], List[ChunkMetadata]]:
        """Process document with specified chunking strategy"""
        if strategy == ChunkingStrategy.SEMANTIC:
            chunks = self.create_semantic_chunks(text)
        elif strategy == ChunkingStrategy.ADAPTIVE:
            chunk_size = self.adaptive_chunk_size(text)
            chunks = self.chunk_text(text, chunk_size)
        elif strategy == ChunkingStrategy.HYBRID:
            # Combine fixed-size and semantic approaches
            initial_chunks = self.chunk_text(text, 1000)
            chunks = []
            for chunk in initial_chunks:
                semantic_subchunks = self.create_semantic_chunks(chunk)
                chunks.extend(semantic_subchunks)
        else:  # FIXED_SIZE
            chunks = self.chunk_text(text, 1000)
            
        # Calculate overlaps
        chunk_overlaps = self.calculate_chunk_overlap(chunks)
        
        # Create metadata for each chunk
        metadatas = []
        for i, (chunk, overlap) in enumerate(zip(chunks, chunk_overlaps)):
            metadata = ChunkMetadata(
                document_name="",  # To be filled by caller
                section="",  # To be filled by caller
                chunk_index=i,
                total_chunks=len(chunks),
                content_hash="",  # To be filled by caller
                document_type="",  # To be filled by caller
                semantic_density=self.calculate_semantic_density(chunk),
                key_terms=self.extract_key_terms(chunk),
                position="start" if i == 0 else "end" if i == len(chunks) - 1 else "middle",
                word_count=len(chunk.split()),
                embedding_model=self.default_embedding,
                chunking_strategy=strategy.value,
                created_at=datetime.now().isoformat(),
                context_overlap=overlap
            )
            metadatas.append(metadata)
            
        return chunks, metadatas

    def create_embeddings(self, input_dir: str, strategy: ChunkingStrategy = ChunkingStrategy.HYBRID) -> Tuple[ProcessingStats, List[Dict]]:
        """Create embeddings with enhanced processing"""
        stats = ProcessingStats()
        documents_processed = []
        start_time = datetime.now()
        
        # Process all txt files
        txt_files = sorted(Path(input_dir).glob('*.txt'))
        
        for file_path in txt_files:
            try:
                self.logger.info(f"Processing file: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Get base metadata
                base_metadata = self.get_document_metadata(file_path)
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Process document
                chunks, chunk_metadatas = self.process_document(content, strategy)
                
                # Update metadata with document-specific info
                for metadata in chunk_metadatas:
                    metadata.document_name = base_metadata["document_name"]
                    metadata.section = base_metadata["section"]
                    metadata.document_type = base_metadata["document_type"]
                    metadata.content_hash = content_hash
                
                # Create IDs
                ids = [f"{base_metadata['document_name']}_{metadata.section}_{i}" 
                      for i, metadata in enumerate(chunk_metadatas)]
                
                # Add to collection
                self.collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=[vars(m) for m in chunk_metadatas]
                )
                
                # Update stats
                stats.total_chunks += len(chunks)
                stats.unique_terms += len(set(sum([m.key_terms for m in chunk_metadatas], [])))
                stats.semantic_scores.extend([m.semantic_density for m in chunk_metadatas])
                
                # Record processed document
                documents_processed.append({
                    "filename": str(file_path.name),
                    "chunks": len(chunks),
                    "metadata": base_metadata,
                    "avg_semantic_density": sum(stats.semantic_scores) / len(stats.semantic_scores)
                })
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Finalize stats
        stats.processing_time = (datetime.now() - start_time).total_seconds()
        stats.avg_chunk_size = sum(len(chunk) for chunk in self.collection.get()["documents"]) / stats.total_chunks
        
        return stats, documents_processed

    def get_document_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract and enhance document metadata"""
        filename = file_path.name
        parts = filename.replace('.txt', '').split('_')
        
        metadata = {
            "source_file": str(filename),
            "document_name": parts[0],
            "document_type": "unknown",
            "section": "main",
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "file_size": file_path.stat().st_size
        }
        
        # Enhanced metadata extraction
        # Continued from document metadata function
        if len(parts) > 1:
            if 'chapter' in parts[1].lower():
                metadata["document_type"] = "book"
                metadata["section"] = f"Chapter {parts[2]}" if len(parts) > 2 else "Unknown Chapter"
            elif parts[1].lower() in ['article', 'paper', 'report', 'guide', 'doc']:
                metadata["document_type"] = parts[1].lower()
            else:
                metadata["document_type"] = parts[1]
                
            # Extract additional section information
            if len(parts) > 2 and not metadata["section"].startswith("Chapter"):
                metadata["section"] = parts[2]
                
        # Add file content analysis
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc = self.nlp(content[:5000])  # Analyze first 5000 chars
                
                metadata.update({
                    "language": doc.lang_,
                    "estimated_reading_time": len(content.split()) / 200,  # Words per minute
                    "complexity_score": self.calculate_complexity_score(doc),
                    "main_topics": self.extract_main_topics(content)
                })
        except Exception as e:
            self.logger.warning(f"Error analyzing file content for metadata: {e}")
            
        return metadata

    def calculate_complexity_score(self, doc) -> float:
        """Calculate text complexity score based on multiple factors"""
        try:
            # Calculate average sentence length
            sentences = list(doc.sents)
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
            
            # Calculate lexical diversity
            words = [token.text.lower() for token in doc if not token.is_punct]
            lexical_diversity = len(set(words)) / len(words) if words else 0
            
            # Calculate technical term ratio
            technical_terms = len([token for token in doc 
                                 if token.pos_ in ['NOUN', 'PROPN'] 
                                 and not token.is_stop])
            technical_ratio = technical_terms / len(doc)
            
            # Combine metrics
            complexity = (
                0.3 * (avg_sentence_length / 40) +  # Normalize to ~0-1
                0.4 * lexical_diversity +
                0.3 * technical_ratio
            )
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.warning(f"Error calculating complexity score: {e}")
            return 0.5

    def extract_main_topics(self, content: str, num_topics: int = 5) -> List[str]:
        """Extract main topics using TF-IDF and noun phrase analysis"""
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform([content])
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get top terms by TF-IDF score
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Extract noun phrases from content
            doc = self.nlp(content[:10000])  # Limit for performance
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Combine and score topics
            topics = {}
            for term, score in sorted_scores[:20]:  # Look at top 20 terms
                # Find related noun phrases
                related_phrases = [np for np in noun_phrases if term in np.lower()]
                if related_phrases:
                    best_phrase = max(related_phrases, key=len)
                    topics[best_phrase] = score
                else:
                    topics[term] = score
            
            # Return top topics
            return list(dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:num_topics]).keys())
            
        except Exception as e:
            self.logger.warning(f"Error extracting main topics: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics and analysis"""
        try:
            collection_data = self.collection.get()
            documents = {}
            
            # Advanced statistics
            stats = {
                "total_documents": 0,
                "total_chunks": self.collection.count(),
                "document_types": {},
                "avg_semantic_density": 0.0,
                "avg_chunk_size": 0.0,
                "unique_terms": set(),
                "embedding_models_used": set(),
                "chunking_strategies_used": set(),
                "creation_date_range": {"earliest": None, "latest": None}
            }
            
            total_density = 0
            total_size = 0
            
            # Process each document
            for metadata in collection_data["metadatas"]:
                doc_name = metadata["document_name"]
                
                # Update document tracking
                if doc_name not in documents:
                    documents[doc_name] = {
                        "document_type": metadata["document_type"],
                        "total_chunks": metadata["total_chunks"],
                        "sections": set(),
                        "avg_semantic_density": 0.0,
                        "key_terms": set(),
                        "creation_date": metadata["created_at"]
                    }
                    stats["total_documents"] += 1
                
                # Update document type stats
                doc_type = metadata["document_type"]
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
                
                # Track sections
                documents[doc_name]["sections"].add(metadata["section"])
                
                # Update semantic density
                total_density += metadata["semantic_density"]
                documents[doc_name]["avg_semantic_density"] += metadata["semantic_density"]
                
                # Track chunk size
                total_size += metadata["word_count"]
                
                # Collect unique terms
                stats["unique_terms"].update(metadata["key_terms"])
                
                # Track embedding models and chunking strategies
                stats["embedding_models_used"].add(metadata["embedding_model"])
                stats["chunking_strategies_used"].add(metadata["chunking_strategy"])
                
                # Track creation dates
                created_at = datetime.fromisoformat(metadata["created_at"])
                if not stats["creation_date_range"]["earliest"] or created_at < datetime.fromisoformat(stats["creation_date_range"]["earliest"]):
                    stats["creation_date_range"]["earliest"] = metadata["created_at"]
                if not stats["creation_date_range"]["latest"] or created_at > datetime.fromisoformat(stats["creation_date_range"]["latest"]):
                    stats["creation_date_range"]["latest"] = metadata["created_at"]
            
            # Calculate averages
            stats["avg_semantic_density"] = total_density / stats["total_chunks"]
            stats["avg_chunk_size"] = total_size / stats["total_chunks"]
            
            # Finalize document averages
            for doc_name in documents:
                documents[doc_name]["avg_semantic_density"] /= documents[doc_name]["total_chunks"]
            
            return {
                "statistics": stats,
                "documents": documents
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return None

    def cleanup_collection(self, older_than_days: Optional[int] = None) -> int:
        """Clean up old or invalid entries from the collection"""
        try:
            collection_data = self.collection.get()
            ids_to_remove = []
            
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                
            for i, metadata in enumerate(collection_data["metadatas"]):
                should_remove = False
                
                # Check document age
                if older_than_days:
                    created_at = datetime.fromisoformat(metadata["created_at"])
                    if created_at < cutoff_date:
                        should_remove = True
                
                # Check for invalid or incomplete metadata
                if not all(key in metadata for key in ["document_name", "section", "content_hash"]):
                    should_remove = True
                
                # Check for empty or invalid content
                if not collection_data["documents"][i] or len(collection_data["documents"][i].strip()) < 10:
                    should_remove = True
                
                if should_remove:
                    ids_to_remove.append(collection_data["ids"][i])
            
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)
                self.logger.info(f"Removed {len(ids_to_remove)} invalid or old entries from collection")
            
            return len(ids_to_remove)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up collection: {e}")
            return 0

def main():
    processor = EnhancedDocumentProcessor()
    input_directory = "../data"
    
    # Process documents with different strategies
    strategies = [
        ChunkingStrategy.HYBRID,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.ADAPTIVE
    ]
    
    results = {}
    for strategy in strategies:
        print(f"\nProcessing with {strategy.value} strategy...")
        stats, processed_docs = processor.create_embeddings(input_directory, strategy)
        results[strategy.value] = {
            "stats": vars(stats),
            "documents": processed_docs
        }
    
    # Get collection information
    collection_info = processor.get_collection_stats()
    
    # Display results
    print("\nProcessing Results:")
    for strategy, result in results.items():
        print(f"\n{strategy} Strategy:")
        print(f"Total chunks: {result['stats']['total_chunks']}")
        print(f"Average chunk size: {result['stats']['avg_chunk_size']:.2f}")
        print(f"Processing time: {result['stats']['processing_time']:.2f} seconds")
    
    print("\nCollection Summary:")
    if collection_info:
        print(f"Total documents: {collection_info['statistics']['total_documents']}")
        print(f"Total chunks: {collection_info['statistics']['total_chunks']}")
        print(f"Average semantic density: {collection_info['statistics']['avg_semantic_density']:.3f}")
        print(f"Unique terms: {len(collection_info['statistics']['unique_terms'])}")

if __name__ == "__main__":
    main()