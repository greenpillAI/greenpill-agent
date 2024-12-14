from openai import OpenAI
import tweepy
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import random
import os
from dotenv import load_dotenv
import time
import schedule
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from content_generation import ContentGenerator

class GreenPillBot:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self._setup_openai()
        self._setup_twitter()
        self._setup_db()
        self._setup_topics()
        self._setup_state()
        self.content_generator = ContentGenerator()

    def _setup_state(self):
        """Initialize bot state variables"""
        self.recent_topics = []
        self.max_history = 5
        self.tweets_since_cta = 0
        self.tweets_since_question = 0
        self.last_tweet_time = None
        self.total_tweets = 0
        self.tweet_history: List[Dict] = []

    def _setup_openai(self):
        """Initialize OpenAI API client with error handling"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.openai = OpenAI(api_key=api_key)
            self.logger.info("OpenAI setup successful")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def _setup_twitter(self):
        """Initialize Twitter API client with comprehensive error handling"""
        required_env_vars = [
            "TWITTER_BEARER_TOKEN",
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET"
        ]
        
        # Check for missing environment variables
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        try:
            self.twitter = tweepy.Client(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                consumer_key=os.getenv("TWITTER_API_KEY"),
                consumer_secret=os.getenv("TWITTER_API_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
                wait_on_rate_limit=True
            )
            # Verify credentials
            self.twitter.get_me()
            self.logger.info("Twitter authentication successful")
        except tweepy.errors.Unauthorized:
            self.logger.error("Twitter authentication failed - check credentials")
            raise
        except tweepy.errors.TooManyRequests:
            self.logger.error("Twitter rate limit exceeded")
            raise
        except Exception as e:
            self.logger.error(f"Twitter setup error: {e}")
            raise

    def _setup_db(self):
        """Initialize ChromaDB with error handling and validation"""
        try:
            db_path = os.getenv("CHROMA_DB_PATH", "db")
            self.client = PersistentClient(path=db_path)
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for embedding function")
                
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"
            )
            
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function
            )
            
            # Validate collection exists and is accessible
            if len(self.collection.get()['ids']) == 0:
                self.logger.warning("Knowledge base collection is empty")
                
            self.logger.info("Database setup successful")
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
            raise

    def _setup_topics(self):
        """Initialize topic list with validation"""
        self.topics = [
            "blockchain environmental solutions",
            "crypto social impact",
            "web3 community building",
            "decentralized governance",
            "sustainable blockchain",
            "crypto environmental benefits",
            "green blockchain technology",
            "web3 social good"
            "token engineering for good",
            "circular economy blockchain",
            "impact investing web3",
            "quadratic funding implementation",
            "retroactive public goods",
            "regenerative cryptoeconomics",
            "solarpunk technology",
            
            # Economic Models
            "token weighted voting",
            "conviction voting systems",
            "bonding curves design",
            "curation markets",
            "impact certificates",
            "proof of impact metrics",
            
            # Environmental Focus
            "regenerative finance",
            "carbon credit tokens",
            "climate DAOs",
            "environmental impact tracking",
            "sustainable blockchain scaling",
            "green consensus mechanisms",
            
            # Governance Innovation
            "decentralized governance research",
            "optimistic governance",
            "futarchy implementations",
            "quadratic voting systems",
            "governance minimization",
            "progressive decentralization",
            
            # Social Impact
            "community coordination tools",
            "public goods funding",
            "impact DAOs",
            "social token design",
            "community ownership",
            "stakeholder capitalism",
            
            # Technical Implementation
            "mechanism design patterns",
            "governance frameworks",
            "smart contract coordination",
            "token engineering tools",
            "on-chain governance",
            "blockchain coordination games",
            "mycelial networks web3",
            "biomimetic economics",
            "nature based solutions",
            "ecological economics",
            "fungal coordination patterns",
            "network infrastructure design",
            "fractal governance systems",
            "emergent coordination systems",
            "dynamic flow systems",
            "mutual reciprocity mechanisms",
            "polycentric governance",
            "natural system design",
            "ecological system patterns",
            "biosphere regeneration",
            "mycelial economics",
            
            # Regenerative Systems
            "bioregional economies",
            "regenerative tokenomics",
            "community currencies",
            "collaborative finance",
            "regenerative memecoins",
            "regenerative rewards",
            "refi systems",
            "prosperity mechanisms",
            "universal basic income systems",
            "crisis aid coordination",
            "democratic funding systems",
            "regenerative yield systems",
            "impact verification systems",
            "conservation incentives",
            "rural preservation mechanisms",
            
            # Practical Implementation
            "zero-fee donation systems",
            "direct funding mechanisms",
            "transparent governance",
            "democratic resource allocation",
            "community-driven platforms",
            "verified impact systems",
            "regenerative staking",
            "impact metrics tracking",
            "conservation validation",
            "regenerative agriculture dao",
            "reforestation incentives",
            "civic innovation systems",
            "community stewardship",
            "systemic solutions design",
            
            # Economic Patterns
            "collaborative resource networks",
            "nutrient flow systems",
            "distributed value networks",
            "pluralistic value systems",
            "reciprocal exchange systems",
            "commons-based economics",
            "obligation networks",
            "mutual interdependence systems",
            "resource metabolism tracking",
            "regenerative distribution mechanisms",
            
            # Advanced Concepts
            "sovereignty systems",
            "diplomatic coordination",
            "trust choreography",
            "commoning mechanisms",
            "confluencer systems",
            "integrities verification",
            "biomimetic token design",
            "ecological economics modeling",
            "nutrient token systems",
            "physically distributed networks",
            "biological diversity systems",
            "regenerative cryptosystems",
            "ecological adaptation mechanisms",
            "evolutionary stability patterns",
            "planetary healing systems"
        ]
        
        if not self.topics:
            raise ValueError("Topics list cannot be empty")
        self.logger.info(f"Initialized with {len(self.topics)} topics")

    def should_add_cta(self) -> bool:
        """Determine if next tweet should include a call-to-action"""
        if self.tweets_since_cta >= 3:
            chance = 0.9
        elif self.tweets_since_cta >= 2:
            chance = 0.6
        else:
            chance = 0.2
        return random.random() < chance

    def should_add_question(self) -> bool:
        """Determine if next tweet should include a question"""
        if self.tweets_since_question >= 4:
            chance = 0.9
        elif self.tweets_since_question >= 3:
            chance = 0.6
        else:
            chance = 0.1
        return random.random() < chance

    def get_diverse_content(self) -> str:
        """Get diverse content from the knowledge base with error handling"""
        try:
            if not self.topics:
                raise ValueError("No topics available")
                
            available_topics = [t for t in self.topics if t not in self.recent_topics]
            if not available_topics:
                self.recent_topics = self.recent_topics[-1:]
                available_topics = [t for t in self.topics if t not in self.recent_topics]
            
            topic = random.choice(available_topics)
            self.recent_topics.append(topic)
            if len(self.recent_topics) > self.max_history:
                self.recent_topics.pop(0)
            
            results = self.collection.query(
                query_texts=[topic],
                n_results=3
            )
            
            if not results['documents'][0]:
                raise ValueError(f"No documents found for topic: {topic}")
                
            content = " ".join(random.sample(results['documents'][0], 2))
            self.logger.debug(f"Generated content for topic: {topic}")
            return content
            
        except Exception as e:
            self.logger.error(f"Error getting diverse content: {e}")
            raise

    def generate_tweet(self) -> str:
        """Generate a tweet using OpenAI API"""
        try:
            content = self.get_diverse_content()
            include_question = self.should_add_question()
            include_cta = self.should_add_cta()
            
            prompt = self._create_tweet_prompt(
                content=content,
                include_question=include_question,
                include_cta=include_cta
            )
            
            response = self.openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.9
            )
            
            tweet = response.choices[0].message.content.strip()
            
            # Update counters
            self.tweets_since_question = 0 if include_question else self.tweets_since_question + 1
            self.tweets_since_cta = 0 if include_cta else self.tweets_since_cta + 1
            
            # Truncate if necessary
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
                
            return tweet
            
        except Exception as e:
            self.logger.error(f"Error generating tweet: {e}")
            raise
    
    def generate_reply(self, mention_text: str) -> str:
      """Generate a reply to a mention using OpenAI API"""
      try:
          # Get context from our knowledge base
          results = self.collection.query(
              query_texts=[mention_text],
              n_results=2
          )
          context = " ".join(results['documents'][0]) if results['documents'][0] else ""
          
          # Get reply style from content generator
          style = self.content_generator.get_reply_style()
          
          # Create prompt using the style
          prompt = self.content_generator.format_reply_prompt(
              mention_text=mention_text,
              context=context,
              style=style
          )
          
          # Generate reply using OpenAI
          response = self.openai.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "user", "content": prompt}],
              max_tokens=100,
              temperature=0.85
          )
          
          reply = response.choices[0].message.content.strip()
          
          # Ensure reply length
          if len(reply) > 280:
              reply = reply[:277] + "..."
              
          self.logger.info(f"Generated reply: {reply}")
          return reply
          
      except Exception as e:
          self.logger.error(f"Error generating reply: {e}")
          raise

    def _create_tweet_prompt(self, content: str, include_question: bool, include_cta: bool) -> str:
        """Create prompt for tweet generation"""
        prompt = f"""Generate a tweet based on this content: "{content}"

        Style Guidelines:
        - Be technical but witty
        - Mix deep insights with humor
        - Use "I" statements naturally
        - Can be edgy but not offensive
        - Reference technical concepts playfully
        - Sound like a real builder having fun
        - Add emojis sparingly but effectively
        """
        
        if include_question:
            prompt += "\n- Include a specific question about implementation details or metrics."
        
        if include_cta:
            prompt += "\n- Include a clear call-to-action"
            
        prompt += "\nKeep tweet strictly under 200 characters. Be creative and diverse."
        return prompt

    def post_tweet(self) -> Optional[Dict]:
        """Post tweet with retry logic and state management"""
        max_retries = 3
        retry_delay = 60
        
        for attempt in range(max_retries):
            try:
                tweet = self.generate_tweet()
                response = self.twitter.create_tweet(text=tweet)
                
                # Update state
                self.last_tweet_time = datetime.now()
                self.total_tweets += 1
                self.tweet_history.append({
                    'tweet': tweet,
                    'timestamp': self.last_tweet_time,
                    'tweet_id': response.data['id']
                })
                
                self.logger.info(f"Posted tweet: {tweet}")
                return response
                
            except tweepy.errors.Forbidden as e:
                self.logger.error(f"403 Forbidden error: Check your authentication tokens")
                raise
            except tweepy.errors.TooManyRequests:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Rate limit hit. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error posting tweet (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to post tweet after {max_retries} attempts")
                    raise

    def schedule_tweets(self, interval_hours: int = 4):
        """Schedule regular tweets with error handling and monitoring"""
        if interval_hours < 1:
            raise ValueError("Tweet interval must be at least 1 hour")
            
        schedule.every(interval_hours).hours.do(self.post_tweet)
        self.logger.info(f"Scheduled tweets every {interval_hours} hours")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Schedule error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def get_bot_stats(self) -> Dict:
        """Get current bot statistics"""
        return {
            'total_tweets': self.total_tweets,
            'last_tweet_time': self.last_tweet_time,
            'tweets_since_cta': self.tweets_since_cta,
            'tweets_since_question': self.tweets_since_question,
            'recent_topics': self.recent_topics,
            'active_topics': len(self.topics)
        }

    def reset_state(self):
        """Reset bot state - useful for testing"""
        self._setup_state()
        self.logger.info("Bot state reset")

if __name__ == "__main__":
    try:
        bot = GreenPillBot()
        bot.schedule_tweets()
    except Exception as e:
        logging.error(f"Bot initialization failed: {e}")
        raise